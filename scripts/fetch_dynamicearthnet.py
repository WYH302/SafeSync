"""Small, source-anchored DynamicEarthNet access and acquisition utility.

Only anonymous public HTTP reads are allowed. The default metadata pass never
downloads large archives or selects samples using pixel/method outcomes.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import ftplib
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/dynamicearthnet_pilot"
SOURCES={
    "author_readme":"https://raw.githubusercontent.com/aysim/dynnet/master/README.md",
    "official_record":"https://mediatum.ub.tum.de/1650201",
    "official_splits":"https://vision.in.tum.de/webshare/u/toker/dynnet_training_splits/",
    "official_checkpoints":"https://vision.in.tum.de/webshare/u/toker/dynnet_ckpt/",
}
FTP_HOST="dataserv.ub.tum.de"
# These are repository-wide public download credentials printed on the official
# mediaTUM record. They are not a personal/user account or imported login state.
PUBLIC_REPOSITORY="m1650201"
READ_CAP=256_000_000
TRANSPORT="http"


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def metadata():
    folder=OUT/"source";folder.mkdir(parents=True,exist_ok=True)
    records={}
    for name,url in SOURCES.items():
        try:
            response=requests.get(url,timeout=(15,25))
            blob=response.content
            if len(blob)>4_000_000:raise ValueError("metadata response unexpectedly exceeds 4MB")
            path=folder/(name+".txt")
            path.write_bytes(blob)
            links=[urljoin(response.url,x) for x in re.findall(r'href=["\']([^"\']+)',response.text)]
            records[name]=dict(url=url,final_url=response.url,status=response.status_code,bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),path=path.relative_to(ROOT).as_posix(),links=links)
            print(name,response.status_code,len(blob),response.url,flush=True)
            print("\n".join(links[:40]),flush=True)
        except (requests.RequestException,ValueError) as error:
            records[name]=dict(url=url,error=str(error))
            print(name,type(error).__name__,str(error)[:300],flush=True)
    write_json(OUT/"access_report.json",dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),sources=records))


class FTPRange(io.RawIOBase):
    """Bounded, cached FTP REST reads; ZipFile never fetches the whole archive."""
    consumed=0

    def __init__(self,name):
        self.name=name;self.position=0
        ftp=self.connect()
        try:self.length=ftp.size(name)
        finally:ftp.close()
        self.cache=OUT/"ranges"/name
        self.cache.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def connect():
        for attempt in range(4):
            ftp=None
            try:
                ftp=ftplib.FTP(FTP_HOST,timeout=30)
                ftp.login(PUBLIC_REPOSITORY,PUBLIC_REPOSITORY)
                ftp.voidcmd("TYPE I")
                return ftp
            except (OSError,EOFError,ftplib.Error):
                if ftp is not None:ftp.close()
                if attempt==3:raise
                time.sleep(2+attempt)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if not 0<=target<=self.length:raise ValueError("seek outside archive")
        self.position=target
        return target

    def read(self,n=-1):
        if n<0 or n>64_000_000:raise ValueError("refusing unbounded/large archive read")
        n=min(n,self.length-self.position)
        if not n:return b""
        start=self.position
        path=self.cache/f"{start}_{n}.bin"
        if path.exists():
            result=path.read_bytes()
            if len(result)!=n:raise ValueError("truncated cached range")
        else:
            if FTPRange.consumed+n>READ_CAP:raise ValueError("explicit pilot read cap reached")
            result=None
            for attempt in range(4):
                ftp=None;sock=None
                try:
                    ftp=self.connect();sock=ftp.transfercmd("RETR "+self.name,rest=start)
                    data=bytearray()
                    while len(data)<n:
                        blob=sock.recv(min(256*1024,n-len(data)))
                        if not blob:raise EOFError("short FTP range")
                        data.extend(blob);FTPRange.consumed+=len(blob)
                    result=bytes(data)
                    break
                except (OSError,EOFError,ftplib.Error):
                    if attempt==3:raise
                    time.sleep(1+attempt)
                finally:
                    if sock is not None:sock.close()
                    if ftp is not None:ftp.close()
            temporary=path.with_suffix(".part");temporary.write_bytes(result);temporary.replace(path)
        self.position+=n
        return result


class HTTPRange(FTPRange):
    def __init__(self,name):
        self.name=name;self.position=0
        self.url="https://dataserv.ub.tum.de/public.php/dav/files/"+PUBLIC_REPOSITORY+"/"+name
        known=OUT/"source"/(name+".inventory.json")
        if known.exists():
            self.length=json.loads(known.read_text(encoding="utf-8"))["archive_bytes"]
            self.cache=OUT/"ranges"/name;self.cache.mkdir(parents=True,exist_ok=True)
            return
        response=requests.get(self.url,params={"range_start":0,"range_length":16},headers={"Range":"bytes=0-15","Accept-Encoding":"identity"},stream=True,timeout=(15,30))
        try:
            if response.status_code!=206 or not re.fullmatch(r'bytes 0-15/[0-9]+',response.headers.get("Content-Range","")):
                raise ValueError(f"archive endpoint probe: HTTP {response.status_code}, Content-Range={response.headers.get('Content-Range')}")
            self.length=int(response.headers["Content-Range"].split('/')[-1])
            if len(response.raw.read(16))!=16:raise EOFError("short archive size probe")
            FTPRange.consumed+=16
        finally:response.close()
        self.cache=OUT/"ranges"/name;self.cache.mkdir(parents=True,exist_ok=True)

    def read(self,n=-1):
        if n<0 or n>64_000_000:raise ValueError("refusing unbounded/large archive read")
        n=min(n,self.length-self.position)
        if not n:return b""
        start=self.position;path=self.cache/f"{start}_{n}.bin"
        if n>196608:
            spans=[(start+i,min(196608,n-i)) for i in range(0,n,196608)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                result=b''.join(pool.map(lambda span:self.fetch_range(*span),spans))
            self.position+=n
            return result
        result=self.fetch_range(start,n)
        self.position+=n
        return result

    def fetch_range(self,start,n):
        path=self.cache/f"{start}_{n}.bin"
        if path.exists():
            result=path.read_bytes()
            if len(result)!=n:raise ValueError("truncated cached range")
        else:
            if FTPRange.consumed+n>READ_CAP:raise ValueError("explicit pilot read cap reached")
            for attempt in range(4):
                response=None
                try:
                    response=requests.get(self.url,params={"range_start":start,"range_length":n},headers={"Range":f"bytes={start}-{start+n-1}","Accept-Encoding":"identity"},stream=True,timeout=(15,45))
                    if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{start+n-1}/{self.length}":
                        raise ValueError("server did not honor the exact requested range")
                    result=response.raw.read(n)
                    FTPRange.consumed+=len(result)
                    if len(result)!=n:raise EOFError("short HTTP range")
                    break
                except (requests.RequestException,EOFError,ValueError):
                    if attempt==3:raise
                    time.sleep(1+attempt)
                finally:
                    if response is not None:response.close()
            temporary=path.with_suffix(".part");temporary.write_bytes(result);temporary.replace(path)
        return result


def inventory(name):
    source=OUT/"source"/(name+".inventory.json")
    if source.exists():return json.loads(source.read_text(encoding="utf-8"))
    remote=HTTPRange(name)
    with zipfile.ZipFile(remote) as archive:
        members=[dict(name=x.filename,compressed_bytes=x.compress_size,bytes=x.file_size,
                      crc32=f"{x.CRC:08x}",compression=x.compress_type,header_offset=x.header_offset)
                 for x in archive.infolist()]
    result=dict(archive=name,archive_bytes=remote.length,members=members,
                source_record=SOURCES["official_record"],endpoint="anonymous official HTTPS DAV file GET with exact HTTP Range validation",
                entire_archive_sha512_verified=False,member_verification="ZIP CRC32 on selected complete members")
    write_json(source,result)
    return result


def member_date(name):
    match=re.search(r'(201[89])[-_]([0-9]{1,2})[-_]([0-9]{1,2})\.tif$',name)
    if match:
        year,month,day=map(int,match.groups())
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def prepare_plan():
    labels=inventory("labels.zip")
    groups={}
    for x in labels["members"]:
        if '/Raster/' in x["name"] and x["name"].endswith('.tif'):
            groups.setdefault(x["name"].split('/')[1],[]).append(x)
    if len(groups)!=55:raise ValueError("V1 publicly labelled cohort no longer matches documented 55 training AOIs")
    aids=sorted(groups)[:4]
    if any(not re.fullmatch(r'[0-9]+_[0-9]+_13[_-][0-9]{2}[NS]',aid) for aid in aids):raise ValueError("unsafe/unrecognized AOI identifier")
    selected=dict(selected_before_pixel_access=True,rule="lexicographically first four AOIs in the 55-AOI V1 public label archive; fixed 2018-01-01 through 2018-01-16 inclusive",
                  official_split_basis="V1 archive has exactly 55 labelled AOIs; official paper says train=55, val=10,test=10, and V2 specifically releases val/test labels. The original split text URL is now404; this is a source-derived train cohort, not a verified copy of train.txt.",
                  aoi_ids=aids,date_start="2018-01-01",date_end="2018-01-16",product="Planet Fusion fused daily surface-reflectance product, not independent daily raw captures")
    target=OUT/"selection.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8"))!=selected:raise ValueError("frozen selection changed")
    write_json(target,selected)
    archives={};jobs=[]
    for aid in aids:
        zone=re.search(r'([0-9]{2}[NS])$',aid).group(1)
        archive_name=f"planet.{zone}.zip"
        if archive_name not in archives:archives[archive_name]=inventory(archive_name)
        # Folder uses the AOI identity and a UTM suffix; compare metadata only.
        short=aid.split('_13')[0]
        image_members=[x for x in archives[archive_name]["members"] if short in x["name"] and '/PF-SR/' in x["name"] and x["name"].endswith('.tif') and member_date(x["name"])]
        images=sorted((dict(x,date=member_date(x["name"]),archive=archive_name) for x in image_members
                       if selected["date_start"]<=member_date(x["name"])<=selected["date_end"]),key=lambda x:x["date"])
        january=sorted((dict(x,date=member_date(x["name"]),archive="labels.zip") for x in groups[aid] if member_date(x["name"])=="2018-01-01"),key=lambda x:x["name"])
        qa=sorted((dict(x,date=member_date(x["name"]),archive=archive_name) for x in archives[archive_name]["members"] if short in x["name"] and '/PF-QA/' in x["name"] and member_date(x["name"]) and selected["date_start"]<=member_date(x["name"])<=selected["date_end"]),key=lambda x:x["date"])
        jobs.append(dict(aoi_id=aid,images=images,quality=qa,monthly_labels=january))
    plan=dict(selection=selected,aois=jobs,planned_image_count=sum(len(x["images"]) for x in jobs),
              planned_original_uncompressed_bytes=sum(y["bytes"] for x in jobs for y in x["images"]+x["quality"]+x["monthly_labels"]),
              planned_zip_member_compressed_bytes=sum(y["compressed_bytes"] for x in jobs for y in x["images"]+x["quality"]+x["monthly_labels"]),
              metadata_bytes_read_this_process=FTPRange.consumed,pixels_downloaded=0)
    write_json(OUT/"plan.json",plan)
    print(f"fixed plan: {len(jobs)} AOIs, {plan['planned_image_count']} SR dates, compressed members={plan['planned_zip_member_compressed_bytes']} bytes",flush=True)
    return plan


def stac_alternative():
    """Metadata only: real Sentinel-2 acquisitions, without a cloud filter."""
    source=OUT/"source"
    window="2023-01-01T00:00:00Z/2023-01-16T23:59:59Z"
    points=[("L15-0331E-1257N_1327_3160_13",-121.66260,37.97885),
            ("L15-1200E-0847N_4802_4803_13",31.04736,-29.66896),
            ("L15-1709E-1112N_6838_3742_13",120.52002,15.34776),
            ("L15-1049E-1370N_4196_2710_13",4.41650,51.87649)]
    plan=dict(source="https://earth-search.aws.element84.com/v1",collection="sentinel-2-l2a",datetime=window,
              selected_before_pixel_access=True,selection="same four preselected geographic centers; fixed consecutive 16-day interval; no cloud/performance filter",
              points=[dict(aoi_id=x,longitude=lon,latitude=lat) for x,lon,lat in points],image_pixels_downloaded=0)
    write_json(OUT/"sentinel2_alternative_selection.json",plan)
    collection=requests.get(plan["source"]+"/collections/sentinel-2-l2a",timeout=(15,40))
    collection.raise_for_status();(source/"sentinel2_collection.json").write_bytes(collection.content)
    records=[]
    for aid,lon,lat in points:
        params={"collections":"sentinel-2-l2a","datetime":window,"intersects":json.dumps({"type":"Point","coordinates":[lon,lat]}),"limit":100}
        response=requests.get(plan["source"]+"/search",params=params,timeout=(15,45));response.raise_for_status()
        raw=response.json();write_json(source/("sentinel2_"+aid+".json"),raw)
        items=raw.get("features",[])
        # A single deterministic MGRS grid prevents counting neighboring-tile
        # products from one acquisition as repeated observations.
        grids=sorted(set(x["properties"].get("grid:code",x["properties"].get("s2:mgrs_tile","unknown")) for x in items))
        grid=grids[0] if grids else None
        items=[x for x in items if x["properties"].get("grid:code",x["properties"].get("s2:mgrs_tile","unknown"))==grid]
        unique={}
        for item in sorted(items,key=lambda x:x["id"]):
            unique.setdefault(item["properties"]["datetime"],item)
        for timestamp,item in sorted(unique.items()):
            assets={}
            for band in ("red","green","blue","scl"):
                if band in item["assets"]:assets[band]=item["assets"][band]
            records.append(dict(aoi_id=aid,item_id=item["id"],datetime=timestamp,grid=grid,cloud_cover=item["properties"].get("eo:cloud_cover"),
                properties=item["properties"],assets=assets,source_item=item.get("links",[])))
        print("STAC",aid,"grid",grid,"real acquisitions",len(unique),flush=True)
    write_json(OUT/"sentinel2_alternative_manifest.json",dict(**plan,items=records,
        collection_license=collection.json().get("license"),collection_links=collection.json().get("links"),
        limitation="5-day-class revisit, not daily; atmospheric correction product; original acquisition timestamp and per-asset scale/offset preserved; RGB8 requires an explicit lossy reflectance display transform"))


def extract_member(entry,path):
    origin=TRANSPORT
    sidecar=path.with_suffix(".source.json")
    if path.exists():
        blob=path.read_bytes()
        import zlib
        if len(blob)!=entry["bytes"] or f"{zlib.crc32(blob)&0xffffffff:08x}"!=entry["crc32"]:
            raise ValueError("existing original member fails ZIP integrity")
        if sidecar.exists():origin=json.loads(sidecar.read_text(encoding="utf-8"))["transport"]
        elif (OUT/"format_sample.json").exists():
            previous=json.loads((OUT/"format_sample.json").read_text(encoding="utf-8"))
            origin=next((x.get("transport",TRANSPORT) for x in previous["source_files"].values() if x["path"]==path.relative_to(ROOT).as_posix()),TRANSPORT)
    else:
        if TRANSPORT=="mirror":
            import zlib
            source=OUT/"source"
            if (source/"mirror_checksums.sha512").read_bytes()!=(source/"checksums.sha512").read_bytes() or (source/"mirror_LICENSE").read_bytes()!=(source/"LICENSE").read_bytes():
                raise ValueError("mirror checksum manifest/license do not exactly match official repository")
            url="https://files.ivry3.cc/datasets/DynamicEarthNet/"+entry["name"]
            partial=path.with_suffix(".download.part");partial.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(4):
                response=None
                start=partial.stat().st_size if partial.exists() else 0
                if start==entry["bytes"]:break
                try:
                    response=requests.get(url,headers={"Range":f"bytes={start}-{entry['bytes']-1}"} if start else {},stream=True,timeout=(15,35));response.raise_for_status()
                    if start and (response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{entry['bytes']-1}/{entry['bytes']}"):
                        if response.status_code==200:start=0
                        else:raise ValueError("mirror resume range mismatch")
                    with partial.open("ab" if start else "wb") as stream:
                        received=start
                        for chunk in response.iter_content(196608):
                            received+=len(chunk);FTPRange.consumed+=len(chunk)
                            if received>entry["bytes"] or FTPRange.consumed>READ_CAP:raise ValueError("mirror member exceeds expected bytes/read cap")
                            stream.write(chunk)
                    if received!=entry["bytes"]:raise EOFError("mirror member truncated")
                    break
                except (requests.RequestException,EOFError):
                    if attempt==3:raise
                    time.sleep(2+attempt)
                finally:
                    if response is not None:response.close()
            blob=partial.read_bytes()
            if len(blob)!=entry["bytes"] or f"{zlib.crc32(blob)&0xffffffff:08x}"!=entry["crc32"]:raise ValueError("mirror member does not match official ZIP length/CRC32")
        else:
            remote=(FTPRange if TRANSPORT=="ftp" else HTTPRange)(entry["archive"])
            with zipfile.ZipFile(remote) as archive:
                info=archive.getinfo(entry["name"])
                if (info.file_size,info.compress_size,f"{info.CRC:08x}")!=(entry["bytes"],entry["compressed_bytes"],entry["crc32"]):
                    raise ValueError("archive member metadata changed since selection")
                blob=archive.read(info) # ZipFile verifies member CRC automatically.
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(".part");temporary.write_bytes(blob);temporary.replace(path)
        if TRANSPORT=="mirror" and partial.exists():partial.unlink()
    record=dict(path=path.relative_to(ROOT).as_posix(),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                archive=entry["archive"],archive_member=entry["name"],zip_crc32=entry["crc32"],transport=origin,
                transport_note="third-party unpacked mirror; checksum manifest and LICENSE exactly match official; complete selected member length/CRC32 matches official directory" if origin=="mirror" else "official repository")
    write_json(sidecar,record)
    return record


def format_sample():
    plan=prepare_plan()
    aoi=plan["aois"][0]
    sr=next(x for x in aoi["images"] if x["date"]=="2018-01-01")
    qa=next(x for x in aoi["quality"] if x["date"]=="2018-01-01")
    label=aoi["monthly_labels"][0]
    folder=OUT/"format_sample"/aoi["aoi_id"]/"2018-01-01";folder.mkdir(parents=True,exist_ok=True)
    records={name:extract_member(item,folder/(name+".tif")) for name,item in (("surface_reflectance",sr),("quality",qa),("label",label))}
    # Reuse the already isolated, read-only Python3.12 Rasterio runtime. No
    # global environment or another worker's dependencies are modified.
    sys.path.insert(0,str(ROOT/"data/spacenet7_pilot/.deps"))
    import rasterio
    import numpy as np
    from PIL import Image
    arrays={}
    for name,record in records.items():
        with rasterio.open(ROOT/record["path"]) as dataset:
            a=dataset.read();arrays[name]=a
            record.update(dimensions=[dataset.width,dataset.height],bands=dataset.count,dtypes=list(dataset.dtypes),nodata=dataset.nodata,
                crs=str(dataset.crs),transform=list(dataset.transform)[:6],scales=list(dataset.scales),offsets=list(dataset.offsets),descriptions=list(dataset.descriptions),
                dataset_tags=dataset.tags(),band_tags={str(i):dataset.tags(i) for i in range(1,dataset.count+1)},
                band_min=[int(x.min()) for x in a],band_max=[int(x.max()) for x in a])
            if name=="surface_reflectance":valid=np.all(dataset.read_masks()[:3]>0,axis=0)
    if arrays["surface_reflectance"].shape[0]!=4 or arrays["quality"].shape[0]!=9 or arrays["label"].shape[0]!=7:
        raise ValueError("source bands differ from documented SR4/QA9/label7")
    if len({(r["dimensions"][0],r["dimensions"][1],str(r["crs"]),str(r["transform"])) for r in records.values()})!=1:
        raise ValueError("image/quality/labels do not share the native grid")
    original=arrays["surface_reflectance"]
    rgb=original[[2,1,0]].transpose(1,2,0).copy()
    np.save(folder/"rgb_native.npy",rgb,allow_pickle=False)
    np.save(folder/"multispectral_native.npy",original.transpose(1,2,0),allow_pickle=False)
    Image.fromarray(valid.astype(np.uint8)).save(folder/"valid_raster.png")
    annotation=arrays["label"]
    # Preserve seven source classes. The original six-class loader ignores
    # source class6 (snow/ice); never conflate that with valid impervious class0.
    hits=annotation==255
    overlap=hits.sum(axis=0)
    semantic=np.full(overlap.shape,255,np.uint8)
    for k in range(7):semantic[hits[k]]=k
    semantic[overlap!=1]=255
    Image.fromarray(semantic).save(folder/"semantic7.png")
    qa_native=arrays["quality"]
    observed=valid&(qa_native[0]==1)&(qa_native[1]==0)&(qa_native[2]==1)&(qa_native[3]>0)
    other_sensor=valid&(qa_native[0]==1)&(np.abs(qa_native[1].astype(np.int32))==900)
    future=valid&(qa_native[1]>0)&(qa_native[1]<900)
    for name,mask in (("same_day_planetscope_support",observed),("same_day_other_sensor_support",other_sensor),("future_gapfill_support",future)):
        Image.fromarray(mask.astype(np.uint8)).save(folder/(name+".png"))
    report=dict(aoi_id=aoi["aoi_id"],product_date="2018-01-01",source_files=records,
        source_bands="B,G,R,NIR from author loader; RGB-native is exact reordered source bands3,2,1",
        rgb_native_path=(folder/"rgb_native.npy").relative_to(ROOT).as_posix(),
        rgb8_path=None,rgb8_status="not generated: source values/bit depth preserved; a fixed product transform must be chosen and disclosed before use in an RGB8 codec",
        valid_raster_path=(folder/"valid_raster.png").relative_to(ROOT).as_posix(),valid_raster_fraction=float(valid.mean()),
        valid_raster_scope="native TIFF masks only; not proof of same-day observed or cloud-free pixels",
        semantic7_path=(folder/"semantic7.png").relative_to(ROOT).as_posix(),
        semantic_classes={0:"impervious surfaces",1:"agriculture",2:"forest and other vegetation",3:"wetlands",4:"soil",5:"water",6:"snow and ice",255:"unlabelled or invalid one-hot assignment"},
        semantic_overlap_pixels=int((overlap>1).sum()),semantic_unlabelled_pixels=int((overlap==0).sum()),
        physical_reflectance_scale=0.0001,physical_reflectance_offset=0,
        physical_scale_source="Planet-authored March2021 technical specification p7 table3; TIFF's own scales=1 are not the physical scale",
        same_day_planetscope_support_path=(folder/"same_day_planetscope_support.png").relative_to(ROOT).as_posix(),same_day_planetscope_support_fraction=float(observed.mean()),
        same_day_other_sensor_support_path=(folder/"same_day_other_sensor_support.png").relative_to(ROOT).as_posix(),same_day_other_sensor_support_fraction=float(other_sensor.mean()),
        future_gapfill_support_path=(folder/"future_gapfill_support.png").relative_to(ROOT).as_posix(),future_gapfill_support_fraction=float(future.mean()),
        quality_mask_scope="QA-derived source-support masks, not raw radiance or proof of online/causal availability; unqueried dates are not screened by them",
        qa_unique_values={str(i+1):np.unique(a).tolist() for i,a in enumerate(arrays["quality"])},
        limitations=["Planet Fusion model-harmonized and gap-filled daily product; not independent daily raw exposure",
                     "monthly labels apply to first day, not every day without qualification",
                     "original six-class loader ignores snow/ice; native7 semantics retained here",
                     "whole ZIP SHA512 not verified because only selected members were read; ZIP CRC and local SHA256 verified"],
        bytes_read_this_process=FTPRange.consumed)
    write_json(OUT/"format_sample.json",report)
    derived=[x for x in folder.iterdir() if x.suffix in (".npy",".png")]
    report["derived_sha256"]={x.relative_to(ROOT).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in derived}
    write_json(OUT/"format_sample.json",report)
    manifest=dict(schema="dynamicearthnet-format-pilot-v1",status="format_sample_only",selection=plan["selection"],
        full_plan_image_count=plan["planned_image_count"],actual_sr_image_count=1,actual_qa_count=1,actual_monthly_label_count=1,
        full_16day_plan_downloaded=False,source_record=SOURCES["official_record"],license="CC-BY-SA-4.0",
        product_scope="Planet Fusion harmonized/gap-filled product synchronization boundary only; not raw daily satellite acquisition",
        observations=[report],rgb8_product_generated=False)
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps(dict(aoi_id=report["aoi_id"],source_files={k:{kk:vv for kk,vv in v.items() if kk in ['bands','dtypes','nodata','dimensions','scales','offsets','band_min','band_max']} for k,v in records.items()},qa_unique_values=report["qa_unique_values"],bytes_read_this_process=FTPRange.consumed),indent=2),flush=True)
    return report


def validate():
    import zlib
    sys.path.insert(0,str(ROOT/"data/spacenet7_pilot/.deps"))
    import rasterio
    import numpy as np
    from PIL import Image
    report=json.loads((OUT/"format_sample.json").read_text(encoding="utf-8"))
    for r in report["source_files"].values():
        blob=(ROOT/r["path"]).read_bytes()
        if len(blob)!=r["bytes"] or hashlib.sha256(blob).hexdigest()!=r["sha256"] or f"{zlib.crc32(blob)&0xffffffff:08x}"!=r["zip_crc32"]:
            raise ValueError("raw source changed")
    with rasterio.open(ROOT/report["source_files"]["surface_reflectance"]["path"]) as ds:a=ds.read()
    rgb=np.load(ROOT/report["rgb_native_path"],allow_pickle=False)
    if not np.array_equal(rgb,a[[2,1,0]].transpose(1,2,0)):raise ValueError("RGB reorder loses source values")
    for path,digest in report["derived_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError("derived output changed")
    stac=json.loads((OUT/"sentinel2_alternative_manifest.json").read_text())
    if len(stac["items"])!=12 or len(set((x["aoi_id"],x["datetime"]) for x in stac["items"]))!=12:raise ValueError("STAC acquisition identity mismatch")
    if stac["image_pixels_downloaded"]!=0 or any(not all(k in x["assets"] for k in ("red","green","blue","scl")) for x in stac["items"]):raise ValueError("STAC source contract differs")
    result=dict(status="pass",actual_sr_images=1,source_files=3,source_crc_and_sha256=True,derived_sha256=True,
        native_RGB_exact=True,native_dtype=str(rgb.dtype),new_image_dates_materialized=1,sentinel2_acquisition_metadata=12,sentinel2_pixel_downloads=0,
        scope="one native format sample and fixed retrieval metadata; not temporal method evaluation")
    write_json(OUT/"validation.json",result);print(json.dumps(result,indent=2),flush=True)


def prepare_day(aoi,date,sr_entry,qa_entry,label_record):
    sys.path.insert(0,str(ROOT/"data/spacenet7_pilot/.deps"))
    import rasterio
    import numpy as np
    from PIL import Image
    folder=OUT/"observations"/aoi/date;folder.mkdir(parents=True,exist_ok=True)
    # Reuse the already verified format originals without another download.
    sample=OUT/"format_sample"/aoi/date
    def original(name,entry):
        path=(sample if (sample/(name+".tif")).exists() else folder)/(name+".tif")
        return extract_member(entry,path)
    sr=original("surface_reflectance",sr_entry);qa=original("quality",qa_entry)
    with rasterio.open(ROOT/sr["path"]) as ds:
        raw=ds.read();native_valid=np.all(ds.read_masks()[:3]>0,axis=0)
        meta=dict(dimensions=[ds.width,ds.height],crs=str(ds.crs),transform=list(ds.transform)[:6],dtype=str(raw.dtype),
                  tiff_scales=list(ds.scales),tiff_offsets=list(ds.offsets),nodata=ds.nodata)
        sr_tags=ds.tags()
    with rasterio.open(ROOT/qa["path"]) as ds:
        q=ds.read();qgrid=(ds.width,ds.height,str(ds.crs),list(ds.transform)[:6]);qa_tags=ds.tags();qa_band_tags={str(i):ds.tags(i) for i in range(1,10)}
    if raw.shape!=(4,1024,1024) or q.shape!=(9,1024,1024) or qgrid!=(meta["dimensions"][0],meta["dimensions"][1],meta["crs"],meta["transform"]):
        raise ValueError("unexpected native SR/QA grid/bands")
    rgb_native=raw[[2,1,0]].transpose(1,2,0).copy()
    np.save(folder/"rgb_native.npy",rgb_native,allow_pickle=False)
    reflectance=rgb_native.astype(np.float64)*0.0001
    clipped=np.clip(reflectance,0,0.5)
    rgb8=np.rint(clipped/0.5*255).astype(np.uint8)
    Image.fromarray(rgb8).save(folder/"rgb8.png")
    support=native_valid&(q[0]==1)&(q[1]==0)&(q[2]==1)&(q[3]>0)
    future=native_valid&(q[1]>0)&(q[1]<900)
    for name,mask in (("native_valid",native_valid),("causal_observed_valid",support),("future_gapfill_support",future)):
        Image.fromarray(mask.astype(np.uint8)).save(folder/(name+".png"))
    nvalid=int(support.sum());samples=nvalid*3;dequant=rgb8.astype(np.float64)/255*0.5
    metrics=dict(valid_pixels=nvalid,valid_rgb_samples=samples,
        negative_rgb_samples=int((reflectance[support]<0).sum()),above_range_rgb_samples=int((reflectance[support]>0.5).sum()),
        physical_mse_on_valid=float(np.mean((dequant[support]-reflectance[support])**2)) if samples else None,
        quantization_only_mse_on_valid=float(np.mean((dequant[support]-clipped[support])**2)) if samples else None,
        physical_mse_on_product_valid=float(np.mean((dequant[native_valid]-reflectance[native_valid])**2)) if native_valid.any() else None)
    record=dict(date=date,aoi_id=aoi,status="ready" if nvalid else "no_same_day_observed_pixels",**meta,
        files=dict(surface_reflectance=sr,quality=qa),rgb_native_path=(folder/"rgb_native.npy").relative_to(ROOT).as_posix(),
        rgb_path=(folder/"rgb8.png").relative_to(ROOT).as_posix(),valid_mask_path=(folder/"causal_observed_valid.png").relative_to(ROOT).as_posix(),
        native_valid_path=(folder/"native_valid.png").relative_to(ROOT).as_posix(),future_gapfill_support_path=(folder/"future_gapfill_support.png").relative_to(ROOT).as_posix(),
        valid_fraction=float(support.mean()),future_gapfill_fraction=float(future.mean()),rgb8_metrics=metrics,
        source_tags=sr_tags,quality_tags=qa_tags,quality_band_tags=qa_band_tags,
        semantic_label_path=label_record["semantic7_path"] if date=="2018-01-01" else None,
        monthly_label_reference=label_record["semantic7_path"],daily_ground_truth_available=date=="2018-01-01")
    record["derived_sha256"]={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in (".png",".npy")}
    write_json(folder/"record.json",record)
    return record


def pilot(workers=3):
    plan=prepare_plan();aoi=plan["aois"][0]
    sr={x["date"]:x for x in aoi["images"]};qa={x["date"]:x for x in aoi["quality"]}
    dates=sorted(set(sr)&set(qa))
    planned=sum(sr[d]["compressed_bytes"]+qa[d]["compressed_bytes"] for d in dates)+sum(x["compressed_bytes"] for x in aoi["monthly_labels"])
    if planned>READ_CAP:dates=[d for d in dates if d<="2018-01-08"]
    planned=sum(sr[d]["compressed_bytes"]+qa[d]["compressed_bytes"] for d in dates)+sum(x["compressed_bytes"] for x in aoi["monthly_labels"])
    if planned>READ_CAP:raise ValueError("even fixed first8 days exceed256MB cap")
    selection=dict(aoi_id=aoi["aoi_id"],date_start="2018-01-01",date_end="2018-01-16" if len(dates)>8 else "2018-01-08",dates=dates,
        selection_rule="fixed first AOI and entire fixed date prefix; no cloud, quality, label or method-performance filtering",planned_compressed_member_bytes=planned,
        original_zip_member_bytes=sum(sr[d]["bytes"]+qa[d]["bytes"] for d in dates)+sum(x["bytes"] for x in aoi["monthly_labels"]),byte_cap=READ_CAP)
    path=OUT/"pilot_selection.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=selection:raise ValueError("pilot selection changed")
    write_json(path,selection)
    label_record=json.loads((OUT/"format_sample.json").read_text(encoding="utf-8"))
    completed={}
    base=dict(schema="dynamicearthnet-product-pilot-v1",selection=selection,source_record=SOURCES["official_record"],license="CC-BY-SA-4.0",
        product_scope="Planet Fusion harmonized/gap-filled product; same-day source-support mask does not restore raw radiance or prove an online processing chain",
        valid_definition="native RGB raster masks AND QA1==1 AND QA2==0 AND QA3==1(clear) AND QA4>0(known PlanetScope scene)",
        rgb8_definition="RGB reorder3,2,1; reflectance=DN*0.0001; round(255*clip(reflectance,0,0.5)/0.5), identical for all dates; lossy visible product only",
        physical_reflectance_scale=0.0001,monthly_label_date="2018-01-01",monthly_label=label_record["source_files"]["label"],
        semantic7_path=label_record["semantic7_path"],semantic_classes=label_record["semantic_classes"],
        temporal_ground_truth="Only Jan1 has verified monthly semantic GT; Jan2-Jan16 reference the month label but it is not daily GT",
        sensor_information_scope="Original4-band int16 TIFF preserved including NIR; RGB8 coding does not preserve full16-bit multispectral radiometry",
        source_integrity="official ZIP member sizes/CRC32 and local SHA256; mirror checksum list/LICENSE exactly equal official; entireZIP SHA512 not verified")
    def save(done=False):
        value=dict(base,status="complete" if done else "partial",observations=[completed[d] for d in sorted(completed)],completed_dates=len(completed),expected_dates=len(dates))
        write_json(OUT/"manifest.partial.json",value)
        if done:write_json(OUT/"manifest.json",value)
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prepare_day,aoi["aoi_id"],d,sr[d],qa[d],label_record):d for d in dates}
        for future in concurrent.futures.as_completed(futures):
            record=future.result();completed[record["date"]]=record;save()
            print(f"ready {record['date']} same-day-support={record['valid_fraction']:.4f}",flush=True)
    save(True)
    print(f"complete {len(completed)} genuine-dated Fusion products; received member bytes this invocation={FTPRange.consumed}",flush=True)


def validate_pilot():
    import zlib
    sys.path.insert(0,str(ROOT/"data/spacenet7_pilot/.deps"))
    import rasterio
    import numpy as np
    from PIL import Image
    m=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    if m.get("status")!="complete":raise ValueError("pilot acquisition is incomplete")
    dates=[x["date"] for x in m["observations"]]
    if dates!=m["selection"]["dates"] or len(set(dates))!=len(dates):raise ValueError("dates changed or were dropped")
    originals=[m["monthly_label"]]
    for row in m["observations"]:originals.extend(row["files"].values())
    for r in originals:
        blob=(ROOT/r["path"]).read_bytes()
        if len(blob)!=r["bytes"] or hashlib.sha256(blob).hexdigest()!=r["sha256"] or f"{zlib.crc32(blob)&0xffffffff:08x}"!=r["zip_crc32"]:raise ValueError("original source failed integrity")
    for row in m["observations"]:
        with rasterio.open(ROOT/row["files"]["surface_reflectance"]["path"]) as ds:
            raw=ds.read();valid=np.all(ds.read_masks()[:3]>0,axis=0)
        with rasterio.open(ROOT/row["files"]["quality"]["path"]) as ds:q=ds.read()
        native=np.load(ROOT/row["rgb_native_path"],allow_pickle=False)
        if not np.array_equal(native,raw[[2,1,0]].transpose(1,2,0)):raise ValueError("native RGB mismatch")
        expected=valid&(q[0]==1)&(q[1]==0)&(q[2]==1)&(q[3]>0)
        mask=np.array(Image.open(ROOT/row["valid_mask_path"]))>0
        if not np.array_equal(expected,mask):raise ValueError("source support mask mismatch")
        rgb=np.array(Image.open(ROOT/row["rgb_path"]))
        fixed=np.rint(np.clip(native.astype(np.float64)*.0001,0,.5)/.5*255).astype(np.uint8)
        if not np.array_equal(rgb,fixed):raise ValueError("RGB8 conversion is not the fixed product transform")
        for path,digest in row["derived_sha256"].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError("derived artifact checksum failed")
        if row["daily_ground_truth_available"]!=(row["date"]=="2018-01-01"):raise ValueError("month label misrepresented as daily GT")
    result=dict(status="pass",dates=len(dates),source_files=len(originals),all_source_crc_sha256=True,all_native_RGB_exact=True,
        fixed_RGB8_transform_verified=True,causal_support_masks_verified=True,derived_sha256=True,
        empty_support_dates=[x["date"] for x in m["observations"] if not x["rgb8_metrics"]["valid_pixels"]],
        monthly_GT_not_replicated_as_daily=True,product_limit="Planet Fusion data, not raw daily exposures")
    write_json(OUT/"validation.json",result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata",action="store_true")
    parser.add_argument("--inventory",metavar="ARCHIVE")
    parser.add_argument("--stac-alternative",action="store_true")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--format-sample",action="store_true")
    parser.add_argument("--transport",choices=["http","ftp","mirror"],default="http")
    parser.add_argument("--validate",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--validate-pilot",action="store_true")
    args=parser.parse_args()
    global TRANSPORT
    TRANSPORT=args.transport
    if args.validate_pilot:validate_pilot()
    elif args.pilot:pilot(args.workers)
    elif args.validate:validate()
    elif args.format_sample:format_sample()
    elif args.plan:prepare_plan()
    elif args.stac_alternative:stac_alternative()
    elif args.inventory:
        result=inventory(args.inventory)
        print(json.dumps({"archive":result["archive"],"archive_bytes":result["archive_bytes"],"member_count":len(result["members"]),
                          "first_members":result["members"][:8],"bytes_read_this_process":FTPRange.consumed},indent=2),flush=True)
    else:metadata()


if __name__=="__main__":main()
