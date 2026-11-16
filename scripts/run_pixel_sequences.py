"""Actual codec/ACK closed loop on the four frozen development AOIs.

All quality decisions use sender-available reference messages. Full-reference
MSE is a larger-cache baseline, not an oracle using semantic labels. Unknown
labels never enter routing. Saved packets can be replayed without new images.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.reference import encode_reference,detail_thumbnail_score,restore_coarse,exact_mse,blocks,bind_reference,unbind_reference
from safesync.certificates import encode_certificate_reference,certificate_score
from safesync.codec_reference import encode_codec_reference,codec_reference_score
from safesync.pixel_wire import PixelState,TransactionSender,TransactionReceiver,encode_update,STABLE,UPDATED,DEFERRED

ROOT=Path(__file__).resolve().parents[1]


def pad(a):
    h,w=a.shape[:2]
    return np.pad(a,[(0,(-h)%64),(0,(-w)%64)]+([(0,0)] if a.ndim==3 else []))


def choose_packet(sender,a,region,obs,score,tau,budget,coarse_side=4,codec="jpeg",quality=60):
    count=len(score)
    candidate=np.flatnonzero(score>tau)
    candidate=candidate[np.argsort(-score[candidate],kind="stable")]
    common=dict(shape=a.shape[:2],block=64,region=region,base_version=sender.version,
                base_digest=sender.digest,observed_at=obs,current_rgb=a,codec=codec,quality=quality,coarse_side=coarse_side)
    trials=0
    def make(k):
        nonlocal trials
        ids=candidate[:k]
        status=np.where(score<=tau,STABLE,DEFERRED).astype(np.uint8);status[ids]=UPDATED
        trials+=1
        return encode_update(indices=ids,statuses=status,**common)
    best,cost=make(0);selected=0
    if len(best)>budget:raise ValueError("budget cannot hold mandatory coarse update and manifest")
    low,high=1,len(candidate)
    while low<=high:
        mid=(low+high)//2;wire,counts=make(mid)
        if len(wire)<=budget:best,cost,selected=wire,counts,mid;low=mid+1
        else:high=mid-1
    # Compression size need not be perfectly monotone. Probe nearby prefixes;
    # result is the largest tested fitting prefix, not a knapsack optimum.
    for k in range(selected+1,min(len(candidate),selected+3)+1):
        wire,counts=make(k)
        if len(wire)<=budget:best,cost,selected=wire,counts,k
    sender.pending=best
    return best,cost,selected,trials


def full_packet(sender,a,region,obs,budget,codec):
    n=a.shape[0]*a.shape[1]//4096
    common=dict(shape=a.shape[:2],block=64,region=region,base_version=sender.version,
                base_digest=sender.digest,observed_at=obs,current_rgb=a,indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec=codec)
    trials=0;best=None;cost=None;full_fits=False
    lo,hi=1,(95 if codec=="jpeg" else 512)
    while lo<=hi:
        q=(lo+hi)//2;trials+=1
        wire,counts=encode_update(quality=q,**common)
        fit=len(wire)<=budget
        if fit:best,cost=wire,counts;full_fits=True
        if codec=="jpeg":
            if fit:lo=q+1
            else:hi=q-1
        else:
            if fit:hi=q-1
            else:lo=q+1
    if best is None:
        # No free thumbnail, resize, or original pixels when codec cannot fit.
        common.update(indices=[],statuses=np.full(n,DEFERRED,np.uint8))
        best,cost=encode_update(quality=1,**common)
    sender.pending=best
    return best,cost,n if full_fits else 0,trials


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--methods",nargs="+",default=["thumbnail_raw","thumbnail_calibrated","gaussian_detail","dct_bound","full_reference_mse","full_jpeg","full_jpeg2000"])
    ap.add_argument("--budgets",nargs="+",type=int,default=[32768,65536,131072]);ap.add_argument("--side",type=int,default=8);ap.add_argument("--psnr",type=float,default=25)
    ap.add_argument("--quality",type=int,default=60);ap.add_argument("--limit-months",type=int,default=0);ap.add_argument("--aoi-indices",nargs="+",type=int,default=[0,1,2,3]);ap.add_argument("--out",default="results/pixel_sequences/development");args=ap.parse_args()
    if any(i>=4 for i in args.aoi_indices):
        raise ValueError("new AOIs require a separately frozen confirmation driver")
    mpath=ROOT/"data/spacenet7_pilot/development_manifest_v1.json"
    if not mpath.exists():mpath=ROOT/"data/spacenet7_pilot/manifest.json"
    manifest=json.loads(mpath.read_text(encoding="utf-8"))
    calibration=json.loads((ROOT/"results/quality_certificates/spacenet7_joint_detail/calibration.json").read_text())
    factor=calibration["factor_by_side"][str(args.side)]
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    source_files=["scripts/run_pixel_sequences.py","safesync/reference.py","safesync/certificates.py","safesync/codec_reference.py","safesync/pixel_wire.py"]
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    spec=dict(vars(args),manifest_sha256=hashlib.sha256(mpath.read_bytes()).hexdigest(),calibration_factor=factor,
              quality_scope="whole 64x64 RGB block PSNR; STABLE means reusable detail after explicit coarse update",
              protocol="SP03 image atlas, immediate counted ACK, no transaction erasure",reference="fresh server-generated summary each window, all uplink bytes counted",
              statistics_unit="AOI, not pixels or months",test_data_access=False,source_sha256=source_hashes,
              calibration_sha256=hashlib.sha256((ROOT/"results/quality_certificates/spacenet7_joint_detail/calibration.json").read_bytes()).hexdigest())
    if (out/"run_spec.json").exists():
        old_spec=json.loads((out/"run_spec.json").read_text())
        if old_spec!=spec:raise ValueError("existing experiment differs; choose a new output directory")
    (out/"run_spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    saved_source=out/"source_snapshot";saved_source.mkdir(exist_ok=True)
    for p in source_files:shutil.copyfile(ROOT/p,saved_source/p.replace("/","__"))
    records=list(csv.DictReader((out/"records.csv").open(encoding="utf-8"))) if (out/"records.csv").exists() else []
    for ai in args.aoi_indices:
        aoi=manifest["aois"][ai];months=aoi["months"]
        if args.limit_months:months=months[:args.limit_months]
        def load(month,key):return np.array(Image.open(ROOT/month[key]))
        first=pad(load(months[0],"rgb_path"));h,w=first.shape[:2];n=h*w//4096
        region=hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
        initial=PixelState(np.zeros_like(first),region)
        init,init_cost=encode_update(shape=(h,w),block=64,region=region,base_version=0,base_digest=initial.digest(),observed_at=0,current_rgb=first,indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec="png")
        folder=out/aoi["aoi_id"];folder.mkdir(exist_ok=True)
        (folder/"initial.bin").write_bytes(init)
        for method in args.methods:
            for budget in args.budgets:
                completed=[r for r in records if r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget]
                if completed and max(int(r["step"]) for r in completed)==len(months)-1:
                    print(f"resume: already complete {aoi['aoi_id']} {method} {budget}",flush=True);continue
                records=[r for r in records if not (r["aoi"]==aoi["aoi_id"] and r["method"]==method and int(r["budget"])==budget)]
                seq=folder/f"{method}_{budget}";seq.mkdir(exist_ok=True)
                receiver=TransactionReceiver(PixelState(np.zeros_like(first),region))
                init_ack,_=receiver.receive(init)
                sender=TransactionSender(receiver.state.version,receiver.state.digest())
                cumulative_down=len(init);cumulative_up=len(init_ack)
                for t,month in enumerate(months[1:],1):
                    start=time.perf_counter()
                    current=pad(load(month,"rgb_path"));valid=pad(load(month,"valid_mask_path")>0)
                    keep=blocks(np.repeat(valid[:,:,None],3,2)).mean((1,2,3))>=.99
                    base=receiver.state.rgb
                    corrected=restore_coarse(base,current)
                    truth=exact_mse(current,corrected)
                    seed=int.from_bytes(hashlib.sha256((aoi["aoi_id"]+month["month"]+":frozen-fresh-Gaussian-2").encode()).digest()[:4],"little")
                    ref=b"";score=None;reference_ms=0;server_ms=0;sender_ms=0
                    ref_start=time.perf_counter()
                    if method.startswith("thumbnail"):
                        ref=encode_reference(base,"thumbnail",args.side,seed=seed,version=receiver.state.version)
                        ref=bind_reference(ref,sender.version,sender.digest,t)
                        server_ms=(time.perf_counter()-ref_start)*1000;sender_start=time.perf_counter()
                        score=detail_thumbnail_score(current,unbind_reference(ref,sender.version,sender.digest,t))
                        if method=="thumbnail_calibrated":score=score*factor
                        sender_ms=(time.perf_counter()-sender_start)*1000
                    elif method in ("gaussian_detail","dct_bound"):
                        design="gaussian_joint" if method=="gaussian_detail" else "dct_bound"
                        ref=encode_certificate_reference(base,design,args.side,seed=seed,detail_side=4)
                        ref=bind_reference(ref,sender.version,sender.digest,t)
                        server_ms=(time.perf_counter()-ref_start)*1000;sender_start=time.perf_counter()
                        _,score,_=certificate_score(current,unbind_reference(ref,sender.version,sender.digest,t))
                        sender_ms=(time.perf_counter()-sender_start)*1000
                    elif method.startswith("codec_norm_"):
                        codec=method.removeprefix("codec_norm_")
                        from safesync.certificates import HEAD
                        cap=n*args.side*args.side*3+HEAD.size+16
                        ref=encode_codec_reference(base,cap,codec)
                        ref=bind_reference(ref,sender.version,sender.digest,t)
                        server_ms=(time.perf_counter()-ref_start)*1000;sender_start=time.perf_counter()
                        _,score=codec_reference_score(current,unbind_reference(ref,sender.version,sender.digest,t))
                        sender_ms=(time.perf_counter()-sender_start)*1000
                    elif method=="full_reference_mse":score=truth.copy()
                    elif method not in ("full_jpeg","full_jpeg2000"):raise ValueError(method)
                    reference_ms=(time.perf_counter()-ref_start)*1000
                    coding_start=time.perf_counter()
                    if method.startswith("full_jpeg"):
                        codec="jpeg2000" if method=="full_jpeg2000" else "jpeg"
                        packet,counts,_,trials=full_packet(sender,current,region,t,budget,codec)
                    else:
                        packet,counts,_,trials=choose_packet(sender,current,region,t,score,10**(-args.psnr/10),budget,quality=args.quality)
                    coding_ms=(time.perf_counter()-coding_start)*1000
                    receiver.expect(t)
                    decode_start=time.perf_counter();ack,_=receiver.receive(packet);sender.acknowledge(ack)
                    decode_ms=(time.perf_counter()-decode_start)*1000
                    if sender.digest!=receiver.state.digest():raise AssertionError("ACK state mismatch")
                    if len(packet)!=sum(counts.values()) or len(packet)>budget:raise AssertionError("actual byte budget/accounting failure")
                    (seq/f"{t:03d}.bin").write_bytes(packet)
                    if ref:(seq/f"{t:03d}.ref").write_bytes(ref)
                    prediction=receiver.state.rgb
                    error=exact_mse(current,prediction)
                    statuses=receiver.state.statuses
                    tau=10**(-args.psnr/10);stable=statuses==STABLE
                    down=len(packet);up=len(ack)+len(ref)
                    cumulative_down+=down;cumulative_up+=up
                    mse=float(np.mean(((current.astype(np.float32)-prediction)/255)**2,axis=-1)[valid].mean())
                    record=dict(aoi=aoi["aoi_id"],aoi_index=ai,split="calibration" if ai<2 else "development",month=month["month"],step=t,method=method,budget=budget,
                        packet_path=(seq/f"{t:03d}.bin").relative_to(ROOT).as_posix(),reference_bytes=len(ref),ack_bytes=len(ack),down_bytes=down,up_bytes=up,
                        cumulative_down_bytes=cumulative_down,cumulative_up_bytes=cumulative_up,initial_down_bytes=len(init),initial_ack_bytes=len(init_ack),
                        block_count=int(keep.sum()),reuse=int((stable&keep).sum()),updated=int(((statuses==UPDATED)&keep).sum()),deferred=int(((statuses==DEFERRED)&keep).sum()),
                        bad=int(((error>tau)&keep).sum()),false_reuse=int((stable&(error>tau)&keep).sum()),valid_mse=mse,psnr=-10*np.log10(max(mse,1e-12)),
                        reference_ms=reference_ms,reference_server_ms=server_ms,reference_sender_ms=sender_ms,search_encoding_ms=coding_ms,receiver_ms=decode_ms,total_ms=(time.perf_counter()-start)*1000,encoding_trials=trials,
                        receiver_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),persistent_sender_reference_bytes=len(ref) if ref else (base.nbytes if method=="full_reference_mse" else 0),**{f"bytes_{k}":v for k,v in counts.items()})
                    records.append(record)
                with (out/"records.csv").open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                print(f"{aoi['aoi_id']} {method} B={budget}: down={cumulative_down} up={cumulative_up}",flush=True)
    print(f"saved {len(records)} real decoded frames to {out}",flush=True)

if __name__=="__main__":main()
