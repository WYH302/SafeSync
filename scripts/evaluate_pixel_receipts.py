"""Replay actual image packets and measure real current/region quality.

Source current RGB and human masks enter only offline metrics. Ground model
prediction, if requested, accepts only packet-reconstructed RGB.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,math,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.pixel_wire import PixelState,TransactionReceiver

ROOT=Path(__file__).resolve().parents[1]


def block_mask_quality(current,received,mask,block=64,psnr=30):
    h,w=mask.shape;error=np.mean(((current.astype(np.float64)-received)/255)**2,axis=-1)
    counts=mask.reshape(h//block,block,w//block,block).sum((1,3)).reshape(-1)
    totals=(error*mask).reshape(h//block,block,w//block,block).sum((1,3)).reshape(-1)
    good=totals<=counts*10**(-psnr/10)
    return dict(pixels=int(counts.sum()),usable_pixels=int(counts[good].sum()),error_sum=float(totals.sum()))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--static-task",action="store_true");args=ap.parse_args()
    out=ROOT/args.run;spec=json.loads((out/"run_spec.json").read_text())
    manifest_path=ROOT/"data/spacenet7_pilot/development_manifest_v1.json"
    if not manifest_path.exists():manifest_path=ROOT/"data/spacenet7_pilot/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    amap={a["aoi_id"]:a for a in manifest["aois"]}
    rows=list(csv.DictReader((out/"records.csv").open(encoding="utf-8")))
    snapshot=out/"receiver_evaluation_source_snapshot";snapshot.mkdir(exist_ok=True)
    source_hashes={}
    for name in ("scripts/evaluate_pixel_receipts.py","safesync/pixel_wire.py"):
        source=ROOT/name;shutil.copyfile(source,snapshot/name.replace("/","__"))
        source_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    marker=dict(status="in_progress",expected_frames=len(rows),actual_frames=0,source_snapshot_sha256=source_hashes)
    (out/"replay_verification.json").write_text(json.dumps(marker,indent=2),encoding="utf-8")
    groups={}
    for row in rows:groups.setdefault((row["aoi"],row["method"],int(row["budget"])),[]).append(row)
    evaluator=None
    if args.static_task:
        from safesync.ground import GroundEvaluator
        evaluator=GroundEvaluator(ROOT)
    results=[]
    for (aid,method,budget),seq in groups.items():
        aoi=amap[aid];months={m["month"]:m for m in aoi["months"]}
        first=aoi["months"][0];width,height=first["dimensions"]
        h,w=math.ceil(height/64)*64,math.ceil(width/64)*64
        region=hashlib.sha256(aid.encode()).digest()[:16]
        receiver=TransactionReceiver(PixelState(np.zeros((h,w,3),np.uint8),region))
        init=(out/aid/"initial.bin").read_bytes();receiver.receive(init)
        first_build=np.array(Image.open(ROOT/first["building_mask_path"]))>0
        first_valid=np.array(Image.open(ROOT/first["valid_mask_path"]))>0
        for row in sorted(seq,key=lambda r:int(r["step"])):
            packet=(ROOT/row["packet_path"]).read_bytes()
            if len(packet)!=int(row["down_bytes"]):raise AssertionError("saved packet length changed")
            receiver.expect(int(row["step"]));ack,_=receiver.receive(packet)
            image=receiver.state.rgb
            if hashlib.sha256(image.tobytes()).hexdigest()!=row["receiver_sha256"]:raise AssertionError("receiver replay differs")
            month=months[row["month"]]
            current=np.zeros_like(image);current[:height,:width]=np.array(Image.open(ROOT/month["rgb_path"]))
            valid=np.zeros((h,w),bool);valid[:height,:width]=np.array(Image.open(ROOT/month["valid_mask_path"]))>0
            building=np.zeros((h,w),bool);building[:height,:width]=np.array(Image.open(ROOT/month["building_mask_path"]))>0
            footprint=np.zeros((h,w),bool);footprint[:height,:width]=(building[:height,:width]&~first_build)
            track=np.zeros((h,w),bool)
            track_path=ROOT/"resources/ground_change_eval/targets"/aid/row["month"]/"new_building.png"
            if not track_path.exists():raise ValueError("official-ID event raster is missing")
            track[:height,:width]=(np.array(Image.open(track_path))>0)&first_valid
            r=dict(aoi=aid,method=method,budget=budget,month=row["month"],step=int(row["step"]),down_bytes=len(packet),ack_bytes=len(ack))
            for name,mask in (("all",valid),("building",building&valid),("new_footprint",footprint&valid),("new_track",track&valid)):
                q=block_mask_quality(current,image,mask)
                r.update({f"{name}_{key}":value for key,value in q.items()})
            if evaluator is not None:
                probabilities=evaluator.predict(Image.fromarray(image[:height,:width]))
                pred=probabilities>=evaluator.threshold;truth=building[:height,:width];v=valid[:height,:width]
                tp=int((pred&truth&v).sum());fp=int((pred&~truth&v).sum());fn=int((~pred&truth&v).sum())
                r.update(task_tp=tp,task_fp=fp,task_fn=fn,task_f1=2*tp/max(2*tp+fp+fn,1),task_iou=tp/max(tp+fp+fn,1),task_scope="weak frozen static building head; auxiliary only")
            results.append(r)
        print(f"replayed {aid} {method} {budget}: {len(seq)} actual packets",flush=True)
        temporary=out/"receiver_evaluation.tmp.csv"
        with temporary.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        temporary.replace(out/"receiver_evaluation.csv")
        marker["actual_frames"]=len(results)
        (out/"replay_verification.json").write_text(json.dumps(marker,indent=2),encoding="utf-8")
    (out/"replay_verification.json").write_text(json.dumps(dict(status="complete",expected_frames=len(rows),actual_frames=len(results),all_receiver_hashes_match=True,quality_threshold_db=30,
        evaluation_csv_sha256=hashlib.sha256((out/"receiver_evaluation.csv").read_bytes()).hexdigest(),source_snapshot_sha256=source_hashes,
        new_footprint_scope="current building pixels outside first-month footprint, not a tracking-ID event label",
        new_track_scope="current official building IDs absent in first-month ID set; current AND first valid",task_requested=args.static_task),indent=2),encoding="utf-8")

if __name__=="__main__":main()
