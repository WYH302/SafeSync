"""Fixed development experiment for sparse reference queries plus JPEG fusion."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.pixel_wire import PixelState,encode_update,decode_update,UPDATED,checksum
from safesync.fusion import FusionSender,FusionReceiver,descriptor,ground_reply,HISTORY_GOOD,CURRENT_GOOD,CURRENT_POOR
from safesync.reference import exact_mse,blocks
from scripts.evaluate_pixel_receipts import block_mask_quality

ROOT=Path(__file__).resolve().parents[1]


def pad(a):
    h,w=a.shape[:2]
    return np.pad(a,[(0,(-h)%64),(0,(-w)%64)]+([(0,0)] if a.ndim==3 else []))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--methods",nargs="+",default=["jpeg","gaussian","dct_bound","codec_jpeg","full_reference"])
    ap.add_argument("--budgets",nargs="+",type=int,default=[16384,32768,65536]);ap.add_argument("--aoi-indices",nargs="+",type=int,default=[0,1,2,3])
    ap.add_argument("--side",type=int,default=4);ap.add_argument("--psnr",type=float,default=25);ap.add_argument("--alpha",type=float,default=.05)
    ap.add_argument("--base-codec",choices=["jpeg","jpeg2000"],default="jpeg");ap.add_argument("--risk-scope",choices=["decision","frame"],default="decision")
    ap.add_argument("--limit-months",type=int,default=0);ap.add_argument("--out",default="results/fusion/development");args=ap.parse_args()
    # An archival evidence item is distinct from the latest lossy state. Fixed
    # first-frame evidence is available to all methods and fully charged at boot.
    reference_source="initial_lossless_archive"
    if any(i>=4 for i in args.aoi_indices):raise ValueError("confirmation requires the frozen confirmation driver")
    path=ROOT/"data/spacenet7_pilot/development_manifest_v1.json"
    if not path.exists():path=ROOT/"data/spacenet7_pilot/manifest.json"
    m=json.loads(path.read_text(encoding="utf-8"));out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    source_files=["scripts/run_fusion_sequences.py","safesync/fusion.py","safesync/reference.py","safesync/certificates.py","safesync/codec_reference.py","safesync/pixel_wire.py"]
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    spec=dict(vars(args),source_sha256=hashes,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              protocol="SF05 + SQ05 query/bound-reference reply + counted immediate ACK",reference_source=reference_source,scope="development, no erasure, explicit one extra round trip when querying",
              known_quality="whole 64x64 RGB block MSE, not a guarantee on individual unknown objects",query_max=64,
              alpha_scope="per fixed queried pair" if args.risk_scope=="decision" else "Bonferroni per queried block; per-observation confidence .95")
    if (out/"run_spec.json").exists() and json.loads((out/"run_spec.json").read_text())!=spec:raise ValueError("existing run differs; use a new output directory")
    (out/"run_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    snap=out/"source_snapshot";snap.mkdir(exist_ok=True)
    for p in source_files:shutil.copyfile(ROOT/p,snap/p.replace("/","__"))
    records=list(csv.DictReader((out/"records.csv").open(encoding="utf-8"))) if (out/"records.csv").exists() else []
    for ai in args.aoi_indices:
        aoi=m["aois"][ai];months=aoi["months"][:args.limit_months or None]
        def load(month,key):return np.array(Image.open(ROOT/month[key]))
        first=pad(load(months[0],"rgb_path"));h,w=first.shape[:2];n=h*w//4096
        region=hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16];blank=PixelState(np.zeros_like(first),region)
        init,cost=encode_update(shape=(h,w),block=64,region=region,base_version=0,base_digest=blank.digest(),observed_at=0,current_rgb=first,indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec="png")
        initialized,initial_ack,_=decode_update(blank,init)
        folder=out/aoi["aoi_id"];folder.mkdir(exist_ok=True);(folder/"initial.bin").write_bytes(init)
        for method in args.methods:
            for budget in args.budgets:
                existing=[r for r in records if r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget]
                if existing and max(int(r["step"]) for r in existing)==len(months)-1:continue
                records=[r for r in records if not (r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget)]
                start_state=PixelState(initialized.rgb.copy(),region,64,initialized.version,initialized.observed_at,initialized.statuses.copy())
                archive=initialized.rgb.copy();receiver=FusionReceiver(start_state,archive)
                cache=PixelState(initialized.rgb.copy(),region,64,initialized.version,initialized.observed_at,initialized.statuses.copy()) if method=="full_reference" else None
                sender=FusionSender(descriptor(start_state,checksum(archive.tobytes())),cache,archive.copy() if cache is not None else None)
                seq=folder/f"{method}_{budget}";seq.mkdir(exist_ok=True)
                down_total=len(init);up_total=len(initial_ack)
                for t,month in enumerate(months[1:],1):
                    current=pad(load(month,"rgb_path"));valid=pad(load(month,"valid_mask_path")>0);building=pad(load(month,"building_mask_path")>0)
                    seed=int.from_bytes(hashlib.sha256((aoi["aoi_id"]+month["month"]+":query-fresh-Gaussian-v1").encode()).digest()[:4],"little")
                    callback=lambda query,design,side,seed:ground_reply(query,receiver.state,design,side,seed,archive)
                    started=time.perf_counter()
                    packet,query,reply,timing=sender.propose(current,t,budget,callback,method=method,side=args.side,psnr=args.psnr,alpha=args.alpha,seed=seed,base_codec=args.base_codec,risk_scope=args.risk_scope)
                    receiver.expect(t);ack=receiver.receive(packet);sender.acknowledge(ack)
                    if sender.desc.digest!=receiver.state.digest():raise AssertionError("ACK mismatch")
                    for suffix,data in (("bin",packet),("query",query),("reply",reply),("ack",ack)):
                        (seq/f"{t:03d}.{suffix}").write_bytes(data)
                    down=len(packet)+len(query);up=len(reply)+len(ack);down_total+=down;up_total+=up
                    if down>budget:raise AssertionError("actual full downlink exceeds budget")
                    keep=blocks(np.repeat(valid[:,:,None],3,2)).mean((1,2,3))>=.99
                    error=exact_mse(current,receiver.state.rgb);s=receiver.state.statuses;tau=10**(-args.psnr/10)
                    pixel_error=np.mean(((current.astype(np.float64)-receiver.state.rgb)/255)**2,axis=-1)
                    r=dict(aoi=aoi["aoi_id"],aoi_index=ai,split="calibration" if ai<2 else "development",month=month["month"],step=t,method=method,budget=budget,
                        packet_path=(seq/f"{t:03d}.bin").relative_to(ROOT).as_posix(),initial_down_bytes=len(init),initial_ack_bytes=len(initial_ack),
                        actual_down_bytes=down,actual_up_bytes=up,cumulative_down_bytes=down_total,cumulative_up_bytes=up_total,ack_bytes=len(ack),
                        block_count=int(keep.sum()),bad_blocks=int(((error>tau)&keep).sum()),reused_blocks_valid=int(((s==HISTORY_GOOD)&keep).sum()),
                        false_reuse=int(((s==HISTORY_GOOD)&(error>tau)&keep).sum()),valid_mse=float(pixel_error[valid].mean()),
                        receiver_sha256=hashlib.sha256(receiver.state.rgb.tobytes()).hexdigest(),end_to_end_compute_ms=(time.perf_counter()-started)*1000,**timing)
                    for name,mask in (("all",valid),("building",building&valid)):
                        q=block_mask_quality(current,receiver.state.rgb,mask)
                        r.update({f"{name}_{key}":v for key,v in q.items()})
                    records.append(r)
                with (out/"records.csv").open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                print(f"{aoi['aoi_id']} {method} {budget}: down={down_total} up={up_total}",flush=True)
    print(f"saved {len(records)} actual fusion frames",flush=True)

if __name__=="__main__":main()
