"""Offline image/evidence quality on independently replayed LIC receipts.

This module never imports the codec or ground task network. Sender decisions
are already committed before any label or validity mask is read.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.evaluate_pixel_receipts import block_mask_quality

MANIFEST="data/spacenet7_pilot/confirmation_manifest_v1.json"
EXPECTED_MANIFEST="8060f59d35f21c83548cd7ba3b19a8d8307131549f77d77262a39ec6966985b8"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def csv_save(path,rows):
    if not rows:return
    temporary=path.with_suffix(".tmp.csv")
    with temporary.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def pad(a):
    return np.pad(a,[(0,(-a.shape[0])%64),(0,(-a.shape[1])%64)]+([(0,0)] if a.ndim==3 else []))


def evaluate_area(ai,aoi,receipt,run):
    source=list(csv.DictReader((ROOT/f"results/layered/confirmation_v1/area_{ai}/records.csv").open(encoding="utf-8")))
    denominators={int(r["step"]):r for r in source if r["method"]=="gaussian_20261029" and int(r["budget"])==32768}
    first_valid=pad(np.array(Image.open(ROOT/aoi["months"][0]["valid_mask_path"]))>0)
    output=[]
    for row in receipt["records"]:
        step=int(row["step"]);month=aoi["months"][step]
        if month["month"]!=row["month"]:raise ValueError("LIC month differs from fixed sequence")
        image_path=ROOT/row["fresh_padded_png_path"]
        if sha(image_path)!=row["fresh_padded_png_sha256"]:raise ValueError("fresh decoded PNG bytes changed")
        rgb=np.array(Image.open(image_path).convert("RGB"))
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=row["fresh_padded_rgb_sha256"] or row["receiver_rgb_sha256"]!=row["fresh_padded_rgb_sha256"]:
            raise ValueError("fresh image is not the actual decoded receiver")
        current_path=ROOT/month["rgb_path"]
        if sha(current_path)!=month["derived_sha256"]["rgb_path"]:raise ValueError("offline current source changed")
        current=pad(np.array(Image.open(current_path).convert("RGB")))
        valid=pad(np.array(Image.open(ROOT/month["valid_mask_path"]))>0)
        building=pad(np.array(Image.open(ROOT/month["building_mask_path"]))>0)&valid
        gt_path=ROOT/f"results/layered/confirmation_v1/area_{ai}/gt"/aoi["aoi_id"]/month["month"]/"new_track.png"
        event=pad(np.array(Image.open(gt_path))>0)&valid&first_valid
        base=dict(aoi=aoi["aoi_id"],area_index=ai,method="hyperprior_q1_8",budget=int(row["cap"]),
            step=step,month=month["month"],packet_sha256=row["packet_sha256"],
            receiver_rgb_sha256=row["receiver_rgb_sha256"],down_bytes=int(row["down_bytes"]),
            ack_bytes=int(row["up_bytes"]),initial_down_bytes=int(row["initial_down_bytes"]),
            initial_ack_bytes=int(row["initial_ack_bytes"]),new_track_gt_sha256=sha(gt_path))
        for name,mask in (("all",valid),("building",building),("new_track",event)):
            for threshold in (25,30,35):
                values=block_mask_quality(current,rgb,mask,psnr=threshold)
                base.update({f"{name}_psnr{threshold}_{k}":v for k,v in values.items()})
        if base["all_psnr30_pixels"]!=int(denominators[step]["all_psnr30_pixels"]) or base["new_track_psnr30_pixels"]!=int(denominators[step]["new_track_psnr30_pixels"]):
            raise ValueError("LIC and main-method metric support differ")
        output.append(base)
    return output


def aggregate(records):
    per=[]
    for aid in sorted({r["aoi"] for r in records}):
        for budget in (32768,65536,131072):
            rows=[r for r in records if r["aoi"]==aid and r["budget"]==budget]
            if not rows:continue
            q=dict(aoi=aid,method="hyperprior_q1_8",budget=budget,updates=len(rows))
            for name in ("all","building","new_track"):
                for threshold in (25,30,35):
                    p=f"{name}_psnr{threshold}_"
                    total=sum(r[p+"pixels"] for r in rows)
                    usable=sum(r[p+"usable_pixels"] for r in rows)
                    error=sum(r[p+"error_sum"] for r in rows)
                    q.update({p+"pixels":total,p+"usable_pixels":usable,p+"coverage":usable/total if total else None,
                              p+"mse":error/total if total else None})
            mse=q["all_psnr30_mse"]
            q["valid_psnr_db"]=float(-10*np.log10(mse)) if mse and mse>0 else None
            q["update_down_mean"]=sum(r["down_bytes"] for r in rows)/len(rows)
            q["update_ack_mean"]=sum(r["ack_bytes"] for r in rows)/len(rows)
            q["update_total_bytes_mean"]=q["update_down_mean"]+q["update_ack_mean"]
            q["bootstrap_total_bytes"]=rows[0]["initial_down_bytes"]+rows[0]["initial_ack_bytes"]
            q["sequence_bytes_per_update"]=q["update_total_bytes_mean"]+q["bootstrap_total_bytes"]/len(rows)
            per.append(q)
    macros=[]
    keys=["valid_psnr_db","update_down_mean","update_ack_mean","update_total_bytes_mean","sequence_bytes_per_update"]+[
        f"{name}_psnr{threshold}_coverage" for name in ("all","building","new_track") for threshold in (25,30,35)]
    for budget in (32768,65536,131072):
        subset=[r for r in per if r["budget"]==budget]
        if not subset:continue
        result=dict(method="hyperprior_q1_8",budget=budget,AOIs=len(subset))
        for key in keys:
            values=[r[key] for r in subset if r[key] is not None]
            result[key]=float(np.mean(values)) if values else None
            result[key+"_AOIs"]=len(values)
        macros.append(result)
    return per,macros


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",default="results/learned_codec/confirmation_q1_8_v1")
    parser.add_argument("--out",default="results/learned_quality/confirmation_v1")
    parser.add_argument("--follow",action="store_true")
    args=parser.parse_args()
    run,out=ROOT/args.run,ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    if sha(ROOT/MANIFEST)!=EXPECTED_MANIFEST:raise ValueError("confirmation manifest changed")
    manifest=load(ROOT/MANIFEST);script_hash=sha(Path(__file__))
    all_expected=sum(len(a["months"])-1 for a in manifest["aois"])*3
    while True:
        records=[];receipts={}
        for ai,aoi in enumerate(manifest["aois"]):
            marker=run/"fresh_replay/complete_aois"/(aoi["aoi_id"]+".json")
            if not marker.exists():continue
            committed=load(marker);receipt_path=ROOT/committed["receipt_path"]
            if committed["status"]!="complete" or sha(receipt_path)!=committed["receipt_sha256"] or committed["spec_sha256"]!=sha(run/"run_spec.json"):
                raise ValueError("uncommitted/corrupt LIC fresh replay receipt")
            key=dict(receipt_sha256=committed["receipt_sha256"],script_sha256=script_hash,
                     manifest_sha256=EXPECTED_MANIFEST)
            cached=out/"areas"/(aoi["aoi_id"]+".json")
            if cached.exists():
                saved=load(cached)
                if saved["identity"]!=key:raise ValueError("offline quality cache identity changed")
                rows=saved["records"]
            else:
                rows=evaluate_area(ai,aoi,load(receipt_path),run)
                save(cached,dict(identity=key,records=rows))
                print(aoi["aoi_id"],len(rows),"offline quality outputs",flush=True)
            expected=(len(aoi["months"])-1)*3
            if len(rows)!=expected:raise ValueError("incomplete committed quality sequence")
            records.extend(rows);receipts[aoi["aoi_id"]]=key
        per,macro=aggregate(records)
        csv_save(out/"records.csv",records);csv_save(out/"per_aoi.csv",per);csv_save(out/"macro.csv",macro)
        status="complete" if len(records)==all_expected else "in_progress"
        save(out/"status.json",dict(status=status,records=len(records),expected=all_expected,
            AOIs=len(receipts),expected_AOIs=len(manifest["aois"]),source_sha256=script_hash,
            input_receipts=receipts,codec_or_ground_model_imported=False,online_sender_labels_opened=False))
        if status=="complete" or not args.follow:break
        time.sleep(30)


if __name__=="__main__":main()
