"""Traceable layered-state development/confirmation execution.

Confirmation requires a launch specification whose source/manifest hashes were
frozen before reading its pixels. It does not fit models or select parameters.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,time,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.layered_cache import default_config
from safesync.certificates import joint_rows
from safesync.layered_wire import LayeredSender,LayeredReceiver
from safesync.pixel_wire import PixelState,encode_update,decode_update,UPDATED
from safesync.full_image_codecs import fit as fit_full,receive as receive_full
from safesync.reference import blocks,exact_mse
from scripts.evaluate_pixel_receipts import block_mask_quality

ROOT=Path(__file__).resolve().parents[1]
SOURCES=["scripts/run_layered_sequences.py","safesync/layered_cache.py","safesync/layered_wire.py",
 "safesync/certificates.py","safesync/reference.py","safesync/pixel_wire.py","safesync/full_image_codecs.py",
 "scripts/run_pixel_sequences.py","scripts/evaluate_pixel_receipts.py"]
DEFAULT_METHODS=["gaussian_20261029","gaussian_1729","gaussian_2718","dct_bound","full_cache_mse",
                 "random_1729","random_2718","random_31415","full_jpeg420","full_jpeg2000","full_webp"]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pad(a):
    h,w=a.shape[:2]
    return np.pad(a,[(0,(-h)%64),(0,(-w)%64)]+([(0,0)] if a.ndim==3 else []))


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8");temporary.replace(path)


def targets(aoi,months,out):
    # Offline GT only: this function is called after a complete transmission
    # sequence. No masks/IDs enter sender.propose or reference cache construction.
    deps=ROOT/"data/spacenet7_pilot/.deps"
    if str(deps) not in sys.path:sys.path.insert(0,str(deps))
    from rasterio.features import rasterize
    from affine import Affine
    first=json.loads((ROOT/months[0]["label_geojson_path"]).read_text())
    ids={str(f["properties"]["Id"]) for f in first["features"]}
    result={}
    for m in months:
        path=out/"gt"/aoi["aoi_id"]/m["month"]/"new_track.png";path.parent.mkdir(parents=True,exist_ok=True)
        data=json.loads((ROOT/m["label_geojson_path"]).read_text())
        shapes=[(f["geometry"],1) for f in data["features"] if f.get("geometry") and str(f["properties"]["Id"]) not in ids]
        w,h=m["dimensions"]
        mask=rasterize(shapes,out_shape=(h,w),transform=Affine(*m["native_transform"]),fill=0,dtype="uint8",all_touched=False) if shapes else np.zeros((h,w),np.uint8)
        Image.fromarray(mask).save(path);result[m["month"]]=path
    return result


def config_for(method):
    if method.startswith("gaussian_"):seed=int(method.rsplit("_",1)[1]);summary="gaussian"
    elif method.startswith("random_"):seed=20261029;summary="gaussian"
    else:seed=20261029;summary="dct_bound"
    config=default_config(summary,seed);config["gate_psnr"]=25
    return config


def launch_spec(args,manifest):
    return dict(schema="layered-experiment-v1",scope="confirmation" if args.confirmation else "development",
        manifest_path=args.manifest,manifest_sha256=sha(ROOT/args.manifest),aoi_ids=[x["aoi_id"] for x in manifest["aois"]],
        methods=args.methods,budgets=args.budgets,area_indices=args.area_indices,limit_months=args.limit_months,
        source_sha256={p:sha(ROOT/p) for p in SOURCES},method_configs={m:config_for(m) for m in args.methods if not m.startswith("full_") or m=="full_cache_mse"},
        thresholds=dict(gate_psnr=25,evidence_psnr=[25,30,35]),protocol="LB01 bootstrap / LA01 layered updates; IF06 whole-image controls; actual counted ACK",
        statistical_unit="geographic AOI; random/measurement seeds are nested within AOI",reference_uplink_bytes=0,
        quality_scope="block RGB MSE; no semantic or arbitrary future-task guarantee",teacher_frozen=True,
        teacher_weight_sha256="8135967879f5e42635508251739e2e9e30e4a58e3abc5df9875340ebc0be7081",
        teacher_metadata_sha256=sha(ROOT/"resources/pretrained_ground/metadata.json"),teacher_building_threshold=.5,teacher_event_threshold=.961)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",default="data/spacenet7_pilot/development_manifest_v1.json")
    ap.add_argument("--methods",nargs="+",default=DEFAULT_METHODS);ap.add_argument("--budgets",nargs="+",type=int,default=[32768,65536,131072])
    ap.add_argument("--area-indices",nargs="+",type=int,default=[0,1,2,3]);ap.add_argument("--limit-months",type=int,default=0)
    ap.add_argument("--out",default="results/layered/development_v1");ap.add_argument("--confirmation",action="store_true")
    ap.add_argument("--freeze-only",action="store_true");args=ap.parse_args()
    manifest=json.loads((ROOT/args.manifest).read_text(encoding="utf-8"))
    if args.confirmation and args.limit_months:raise ValueError("confirmation cannot shorten sequences")
    if "confirmation" in args.manifest and not args.confirmation:raise ValueError("new AOIs require explicit confirmation mode")
    if any(i<0 or i>=len(manifest["aois"]) for i in args.area_indices):raise ValueError("bad area index")
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True);spec=launch_spec(args,manifest)
    existing=out/"run_spec.json"
    if existing.exists():
        if json.loads(existing.read_text())!=spec:raise ValueError("frozen experiment differs: use a new output directory")
    else:
        if args.confirmation and not args.freeze_only:raise ValueError("freeze confirmation specification before reading pixels")
        write_json(existing,spec)
        snap=out/"source_snapshot";snap.mkdir(exist_ok=True)
        for p in SOURCES:shutil.copyfile(ROOT/p,snap/p.replace("/","__"))
    if args.freeze_only:print("frozen metadata/config/source hashes; no pixels read",flush=True);return
    records=list(csv.DictReader((out/"records.csv").open(encoding="utf-8"))) if (out/"records.csv").exists() else []
    expected=sum((min(args.limit_months,len(manifest["aois"][i]["months"])) if args.limit_months else len(manifest["aois"][i]["months"]))-1 for i in args.area_indices)*len(args.methods)*len(args.budgets)
    write_json(out/"progress.json",dict(status="in_progress",expected=expected,records=len(records),spec_sha256=sha(existing)))
    for ai in args.area_indices:
        aoi=manifest["aois"][ai];months=aoi["months"][:args.limit_months or None]
        first=pad(np.array(Image.open(ROOT/months[0]["rgb_path"])));h,w=first.shape[:2];n=h*w//4096
        region=hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16];folder=out/aoi["aoi_id"];folder.mkdir(exist_ok=True)
        for method in args.methods:
            for budget in args.budgets:
                old=[r for r in records if r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget]
                if len(old)==len(months)-1:continue
                records=[r for r in records if not (r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget)]
                seq=folder/f"{method}_{budget}";seq.mkdir(exist_ok=True)
                full=method.startswith("full_") and method!="full_cache_mse"
                if full:
                    blank=PixelState(np.zeros_like(first),region)
                    boot,_=encode_update(shape=(h,w),block=64,region=region,base_version=0,base_digest=blank.digest(),observed_at=0,current_rgb=first,indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec="png")
                    state,boot_ack,_=decode_update(blank,boot);sender=receiver=None
                else:
                    cfg=config_for(method);sender=LayeredSender(keep_full_archive=method=="full_cache_mse");receiver=LayeredReceiver()
                    boot,_=sender.bootstrap(first,region,cfg);boot_ack=receiver.receive(boot);sender.acknowledge(boot_ack)
                (seq/"initial.bin").write_bytes(boot);(seq/"initial.ack").write_bytes(boot_ack)
                cumulative_down=len(boot);cumulative_up=len(boot_ack)
                # Complete sender/decoder execution first, before GT construction.
                images=[];sequence=[]
                for t,month in enumerate(months[1:],1):
                    current=pad(np.array(Image.open(ROOT/month["rgb_path"])));start=time.perf_counter()
                    if full:
                        codec=method.removeprefix("full_")
                        packet,meta=fit_full(current,codec=codec,budget=budget,region=region,base_version=state.version,base_digest=state.digest(),observation=t)
                        state,ack=receive_full(state,packet);image=state.rgb;statuses=state.statuses
                        cache_bytes=metadata_bytes=0;archive_bytes=0
                        matrix_bytes=0
                    else:
                        ranking="exact" if method=="full_cache_mse" else "random" if method.startswith("random_") else "bound"
                        rseed=int(method.rsplit("_",1)[1])+t*100003+ai*1009 if ranking=="random" else 1729
                        packet,meta=sender.propose(current,t,budget,ranking=ranking,random_seed=rseed)
                        receiver.expect(t);ack=receiver.receive(packet);sender.acknowledge(ack)
                        if sender.history!=receiver.state.history or sender.cache.wire!=receiver.state.cache.wire:raise AssertionError("local cache or history ACK mismatch")
                        image=receiver.state.render();statuses=receiver.state.statuses
                        cache_bytes=len(sender.cache.wire);metadata_bytes=40+len(json.dumps(cfg,sort_keys=True,separators=(",",":")).encode())
                        archive_bytes=0 if sender.full_archive is None else sender.full_archive.nbytes
                        matrix_bytes=joint_rows(64,cfg["side"]**2//2*3,cfg["seed"]).nbytes if cfg["summary"]=="gaussian" else 0
                        meta={k:v for k,v in meta.items() if k not in ("bound","point")}
                    if len(packet)>budget or len(ack)!=60:raise AssertionError("actual frame/ACK budget")
                    (seq/f"{t:03d}.bin").write_bytes(packet);(seq/f"{t:03d}.ack").write_bytes(ack)
                    cumulative_down+=len(packet);cumulative_up+=len(ack)
                    record=dict(aoi=aoi["aoi_id"],area_index=ai,scope=spec["scope"],method=method,budget=budget,step=t,month=month["month"],
                        packet_path=(seq/f"{t:03d}.bin").relative_to(ROOT).as_posix(),bootstrap_path=(seq/"initial.bin").relative_to(ROOT).as_posix(),
                        packet_sha256=sha(seq/f"{t:03d}.bin"),ack_sha256=sha(seq/f"{t:03d}.ack"),down_bytes=len(packet),up_bytes=len(ack),reference_up_bytes=0,
                        bootstrap_down_bytes=len(boot),bootstrap_up_bytes=len(boot_ack),cumulative_down_bytes=cumulative_down,cumulative_up_bytes=cumulative_up,
                        receiver_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),sender_summary_bytes=cache_bytes,sender_metadata_bytes=metadata_bytes,sender_full_archive_bytes=archive_bytes,
                        shared_projection_matrix_bytes=matrix_bytes,
                        compute_ms=(time.perf_counter()-start)*1000,encoder_meta=json.dumps(meta,sort_keys=True))
                    images.append((image.copy(),current,statuses.copy(),month));sequence.append(record)
                target_paths=targets(aoi,months,out)
                first_valid=pad(np.array(Image.open(ROOT/months[0]["valid_mask_path"]))>0)
                for record,(image,current,statuses,month) in zip(sequence,images):
                    valid=pad(np.array(Image.open(ROOT/month["valid_mask_path"]))>0)
                    building=pad(np.array(Image.open(ROOT/month["building_mask_path"]))>0)
                    event=pad(np.array(Image.open(target_paths[month["month"]]))>0)&valid&first_valid
                    keep=blocks(np.repeat(valid[:,:,None],3,2)).mean((1,2,3))>=.99
                    err=exact_mse(current,image);e=np.mean(((current.astype(np.float64)-image)/255)**2,axis=-1)
                    record.update(valid_mse=float(e[valid].mean()),valid_blocks=int(keep.sum()),stable_blocks=int(((statuses==1)&keep).sum()),
                        updated_blocks=int(((statuses==2)&keep).sum()),deferred_blocks=int(((statuses==3)&keep).sum()),
                        false_stable_blocks=int(((statuses==1)&(err>10**(-2.5))&keep).sum()))
                    for name,mask in (("all",valid),("building",building&valid),("new_track",event)):
                        for threshold in (25,30,35):
                            q=block_mask_quality(current,image,mask,psnr=threshold)
                            record.update({f"{name}_psnr{threshold}_{k}":v for k,v in q.items()})
                    records.append(record)
                tmp=out/"records.tmp.csv"
                with tmp.open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                tmp.replace(out/"records.csv")
                write_json(out/"progress.json",dict(status="in_progress",expected=expected,records=len(records),spec_sha256=sha(existing)))
                print(f"{aoi['aoi_id']} {method} {budget}: {len(sequence)} decoded frames, cache={cache_bytes}, reference_up=0",flush=True)
    if len(records)!=expected:raise AssertionError("completed record count differs from frozen plan")
    write_json(out/"progress.json",dict(status="complete",expected=expected,records=len(records),spec_sha256=sha(existing),records_sha256=sha(out/"records.csv")))

if __name__=="__main__":main()
