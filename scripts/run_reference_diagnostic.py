"""Development-only measurement diagnostic; does not claim downlink savings."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys, json, csv, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from safesync.data import load_manifest, read_pair, block_fraction
from safesync.reference import encode_reference, reference_score, exact_mse


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/"results/reference_diagnostic"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    pairs=load_manifest(root)["pairs"]
    for i,item in enumerate(pairs):
        before,after,old,label,changed=read_pair(root,item)
        truth=exact_mse(after,before)
        weights={"all":np.ones(64),"change":block_fraction(changed,8),
                 "building":block_fraction((label==5)&changed,8),
                 "tree":block_fraction((label==3)&changed,8)}
        for side in (4,8,16):
            for method in ("thumbnail","projection","hybrid","dct_low","dct_spread","full_reference"):
                start=time.perf_counter()
                if method=="full_reference":
                    score=truth; nbytes=np.asarray(before).nbytes
                else:
                    wire=encode_reference(before,method,side)
                    score=reference_score(after,wire); nbytes=len(wire)
                ms=(time.perf_counter()-start)*1000
                order=np.argsort(-score,kind="stable")
                for k in (4,8,16,32):
                    selected=order[:k]
                    for target,weight in weights.items():
                        value=truth*weight
                        rows.append(dict(pair=item["id"],split=item["split"],side=side,method=method,
                            reference_bytes=nbytes,selected=k,target=target,
                            recoverable_error=float(value[selected].sum()),total_error=float(value.sum()),
                            covered_weight=float(weight[selected].sum()),total_weight=float(weight.sum()),score_ms=ms))
        if (i+1)%8==0: print(f"reference diagnostic {i+1}/{len(pairs)}",flush=True)
    with (out/"records.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for side in (4,8,16):
        for method in ("thumbnail","projection","hybrid","dct_low","dct_spread","full_reference"):
            r=[x for x in rows if x["side"]==side and x["method"]==method and x["split"]!="calibration" and x["selected"]==16 and x["target"]=="building"]
            summary.append(dict(side=side,method=method,reference_bytes=r[0]["reference_bytes"],
                recovered_error=sum(x["recoverable_error"] for x in r)/max(sum(x["total_error"] for x in r),1e-12),
                covered_unknown=sum(x["covered_weight"] for x in r)/max(sum(x["total_weight"] for x in r),1e-12)))
    (out/"summary.json").write_text(json.dumps({"scope":"Previously viewed development pairs only; ranking diagnostic, not a communication result", "summary":summary},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
