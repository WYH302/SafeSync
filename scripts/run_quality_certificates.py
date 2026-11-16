"""Real-data test of reference-relative image-quality decisions.

No learned detector enters decisions. Ground truth is used only for evaluation;
thumbnail calibration is fitted on preselected calibration image pairs/AOIs.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.reference import encode_reference,reference_score,exact_mse,blocks,detail_thumbnail_score,restore_coarse
from safesync.certificates import encode_certificate_reference,certificate_score
from safesync.data import read_pair,block_fraction

ROOT=Path(__file__).resolve().parents[1]


def pad(a,block=64):
    h,w=a.shape[:2]
    pads=[(0,(-h)%block),(0,(-w)%block)]+([(0,0)] if a.ndim==3 else [])
    return np.pad(a,pads)


def pairs(dataset):
    if dataset=="second":
        m=json.loads((ROOT/"data/second_scaled/manifest.json").read_text(encoding="utf-8"))
        if not m["complete"]:raise RuntimeError("wait for fixed SECOND acquisition to finish")
        for item in m["pairs"]:
            a,b,old,labels,change=read_pair(ROOT,item)
            yield item["id"],item["group_id"],item["split"],np.array(a),np.array(b),np.ones(labels.shape,bool),(labels==5)&change
    else:
        path=ROOT/"data/spacenet7_pilot/development_manifest_v1.json"
        if not path.exists():path=ROOT/"data/spacenet7_pilot/manifest.json"
        m=json.loads(path.read_text(encoding="utf-8"))
        for i,aoi in enumerate(m["aois"][:4]):
            months=aoi["months"]
            for previous,current in zip(months,months[1:]):
                def img(x,key):return np.array(Image.open(ROOT/x[key]))
                a,b=img(previous,"rgb_path"),img(current,"rgb_path")
                valid=(img(previous,"valid_mask_path")>0)&(img(current,"valid_mask_path")>0)
                event=(img(current,"building_mask_path")>0)&(img(previous,"building_mask_path")==0)&valid
                yield aoi["aoi_id"]+"_"+current["month"],aoi["aoi_id"],"calibration" if i<2 else "validation",a,b,valid,event


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",choices=["second","spacenet7"],required=True);ap.add_argument("--joint",action="store_true");ap.add_argument("--detail",action="store_true");args=ap.parse_args()
    out=ROOT/"results/quality_certificates"/(args.dataset+("_joint" if args.joint else "")+("_detail" if args.detail else ""));out.mkdir(parents=True,exist_ok=True)
    gaussian="gaussian_joint" if args.joint else "gaussian"
    hybrid="hybrid_joint" if args.joint else "gaussian_hybrid"
    records=[]
    for i,(name,group,split,a,b,valid,event) in enumerate(pairs(args.dataset)):
        a,b=pad(a),pad(b);valid,event=pad(valid),pad(event)
        keep=blocks(np.repeat(valid[:,:,None],3,2)).mean((1,2,3))>=0.99
        unknown=blocks(np.repeat(event[:,:,None],3,2)).mean((1,2,3))
        truth=exact_mse(b,restore_coarse(a,b) if args.detail else a)
        seed=int.from_bytes(hashlib.sha256((args.dataset+name+":fresh-reference-1").encode()).digest()[:4],"little")
        frame={"name":name,"group":group,"split":split,"truth":truth,"keep":keep,"event":unknown,"methods":{}}
        for side in (4,8,16):
            methods={}
            start=time.perf_counter()
            wire=encode_reference(a,"thumbnail",side,seed=seed)
            point=detail_thumbnail_score(b,wire) if args.detail else reference_score(b,wire)
            methods["thumbnail"]=dict(point=point,upper=point,reference_bytes=len(wire),ms=(time.perf_counter()-start)*1000)
            for method in (gaussian,hybrid,"dct_bound"):
                start=time.perf_counter()
                wire=encode_certificate_reference(a,method,side,seed=seed,detail_side=4 if args.detail else 0)
                point,upper,meta=certificate_score(b,wire)
                methods[method]=dict(point=point,upper=upper,reference_bytes=len(wire),ms=(time.perf_counter()-start)*1000)
            frame["methods"][side]=methods
        records.append(frame)
        if (i+1)%8==0:print(f"{args.dataset}: quality certificates {i+1}",flush=True)
    # One fixed baseline calibration: upper quantile of true/thumbnail residual
    # ratio, with q=0.95 and finite quantile interpolation='higher'. Pixel blocks
    # are correlated; this empirical calibration has no IID coverage claim.
    calibration={}
    for side in (4,8,16):
        ratios=np.concatenate([r["truth"][r["keep"]]/np.maximum(r["methods"][side]["thumbnail"]["point"][r["keep"]],1e-8) for r in records if r["split"]=="calibration"])
        calibration[side]=float(np.quantile(ratios,.95,method="higher"))
    rows=[]
    for r in records:
        keep=r["keep"]; truth=r["truth"][keep]; event=r["event"][keep]
        for side in (4,8,16):
            methods=dict(r["methods"][side])
            raw=methods["thumbnail"]
            methods["thumbnail_calibrated"]=dict(raw,upper=raw["point"]*calibration[side])
            methods["gaussian_point"]=dict(methods[gaussian],upper=methods[gaussian]["point"])
            methods["hybrid_point"]=dict(methods[hybrid],upper=methods[hybrid]["point"])
            for method,values in methods.items():
                upper=values["upper"][keep]
                for psnr in (20,25,30):
                    tau=10**(-psnr/10)
                    stable=upper<=tau; bad=truth>tau
                    rows.append(dict(dataset=args.dataset,pair=r["name"],group=r["group"],split=r["split"],side=side,method=method,psnr=psnr,
                        reference_bytes=values["reference_bytes"],score_ms=values["ms"],blocks=len(truth),stable=int(stable.sum()),bad=int(bad.sum()),
                        false_stable=int((stable&bad).sum()),needless_send=int(((~stable)&(~bad)).sum()),
                        event_weight=float(event.sum()),false_stable_event_weight=float(event[stable&bad].sum()),
                        undercovered=int((truth>upper+1e-6).sum())))
    with (out/"records.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"calibration.json").write_text(json.dumps({"factor_by_side":calibration,"alpha":.05,"selection":"calibration only, patchwise 95th quantile, spatially correlated","scope":"Reference-relative RGB MSE, not semantic stability"},indent=2),encoding="utf-8")
    summaries=[]
    for side in (4,8,16):
        for method in ("thumbnail","thumbnail_calibrated","gaussian_point","hybrid_point",gaussian,hybrid,"dct_bound"):
            rr=[r for r in rows if r["split"]=="validation" and r["side"]==side and r["method"]==method and r["psnr"]==25]
            s={k:sum(r[k] for r in rr) for k in ("blocks","stable","bad","false_stable","needless_send","undercovered")}
            summaries.append(dict(side=side,method=method,**s,
                stable_fraction=s["stable"]/s["blocks"] if s["blocks"] else None,
                false_accept_of_bad=s["false_stable"]/s["bad"] if s["bad"] else None,
                error_among_stable=s["false_stable"]/s["stable"] if s["stable"] else None))
    (out/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=="__main__":main()
