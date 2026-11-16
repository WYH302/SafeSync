"""Strong image-codec/norm baseline under the same reference byte caps."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,csv,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.run_quality_certificates import pairs,pad
from safesync.reference import blocks,restore_coarse,exact_mse
from safesync.codec_reference import encode_codec_reference,codec_reference_score
from safesync.certificates import HEAD


def main():
    root=Path(__file__).resolve().parents[1];out=root/"results/quality_certificates/codec_reference";out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(name,group,split,a,b,valid,event) in enumerate(pairs("spacenet7")):
        a,b=pad(a),pad(b);valid=pad(valid);truth=exact_mse(b,restore_coarse(a,b))
        keep=blocks(np.repeat(valid[:,:,None],3,2)).mean((1,2,3))>=.99
        for side in (4,8):
            cap=len(truth)*side*side*3+HEAD.size+16
            for codec in ("jpeg","jpeg2000"):
                start=time.perf_counter();wire=encode_codec_reference(a,cap,codec);encode_ms=(time.perf_counter()-start)*1000
                start=time.perf_counter();point,upper=codec_reference_score(b,wire);score_ms=(time.perf_counter()-start)*1000
                for psnr in (20,25,30):
                    bad=truth>10**(-psnr/10);stable=upper<=10**(-psnr/10)
                    rows.append(dict(pair=name,group=group,split=split,side=side,method="codec_norm_"+codec,psnr=psnr,reference_bytes=len(wire),reference_cap=cap,
                                     blocks=int(keep.sum()),stable=int((stable&keep).sum()),bad=int((bad&keep).sum()),false_stable=int((stable&bad&keep).sum()),undercovered=int(((truth>upper+1e-6)&keep).sum()),
                                     server_encoding_ms=encode_ms,sender_score_ms=score_ms))
        if (i+1)%4==0:print(f"codec reference baseline {i+1}/92",flush=True)
        with (out/"records.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps([dict(side=s,method=m,psnr=p,**{k:sum(r[k] for r in rows if r["split"]=="validation" and r["side"]==s and r["method"]==m and r["psnr"]==p) for k in ("blocks","stable","false_stable","undercovered")}) for s in (4,8) for m in ("codec_norm_jpeg","codec_norm_jpeg2000") for p in (20,25,30)],indent=2))

if __name__=="__main__":main()
