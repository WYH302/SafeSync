"""One preselected novelty estimator; held-out labels used only offline."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from safesync.open_set import KnownClassDistance
from safesync.data import read_pair,block_fraction,masks
from safesync.scoring import change_score
from safesync.metrics import conditional_unknown_diagnostic


def main():
    root=Path(__file__).resolve().parents[1];m=json.loads((root/"data/second_scaled/manifest.json").read_text(encoding="utf-8"))
    with np.load(root/"cache/second_scaled/features.npz",allow_pickle=False) as f:features=f["features"].astype(np.float32);ids=f["ids"]
    if ids.tolist()!=[p["id"] for p in m["pairs"]]:raise ValueError("cache IDs differ")
    out=root/"results/unknown_scaled";out.mkdir(parents=True,exist_ok=True);reports=[]
    data=[]
    for item in m["pairs"]:
        _,_,old,label,changed=read_pair(root,item);data.append((old,label,changed))
    for heldout in (5,3,4):
        fit_x=[];fit_y=[]
        for item,f,(old,label,changed) in zip(m["pairs"],features,data):
            if item["split"]!="calibration":continue
            unknown=block_fraction((old==heldout)|(label==heldout),16)>0
            for time,semantic in enumerate((old,label)):
                for k in range(1,7):
                    if k==heldout:continue
                    valid=(block_fraction(semantic==k,16)>=.75)&~unknown
                    fit_x.append(f[time][valid]);fit_y.extend([k]*int(valid.sum()))
        x=np.concatenate(fit_x);model=KnownClassDistance.fit(x,np.array(fit_y));model.save(out/f"known_{heldout}.npz")
        raw=[];c_all=[];u_all=[];p_all=[];n_all=[]
        for item,f,(old,label,changed) in zip(m["pairs"],features,data):
            change=change_score(f[1],f[0]);unknown=model.score(f[1])
            mask=masks(old,label,changed,heldout,16,.75)
            np.savez_compressed(out/f"scores_{heldout}_{item['id']}.npz",change=change,unknown=unknown)
            if item["split"]=="validation":
                c_all.extend(change);u_all.extend(unknown);p_all.extend(mask["diagnostic_unknown"]);n_all.extend(mask["diagnostic_known"])
        report=conditional_unknown_diagnostic(np.array(c_all),np.array(u_all),np.array(p_all),np.array(n_all),10)
        reports.append(dict(heldout=heldout,known_class_counts=model.counts,fit_cells=len(x),validation_pairs=64,diagnostic=report,
            interpretation="Association within coarse change bins; not proof of conditional independence, open-set probability calibration, or unseen foundation-model pretraining classes"))
    (out/"diagnostics.json").write_text(json.dumps(reports,indent=2),encoding="utf-8");print(json.dumps(reports,indent=2),flush=True)

if __name__=="__main__":main()
