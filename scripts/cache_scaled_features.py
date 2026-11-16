"""One frozen feature pass; no labels are used by the encoder."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from safesync.data import read_pair
from safesync.features import FrozenEncoder


def main():
    root=Path(__file__).resolve().parents[1];mp=root/"data/second_scaled/manifest.json"
    m=json.loads(mp.read_text(encoding="utf-8"))
    if not m["complete"]:raise ValueError("fixed acquisition is incomplete")
    config={"seed":20261028,"backbone_input":448,"pool_patch_grid":2,"grid":16,"feature_dim":384}
    encoder=FrozenEncoder(root,config);features=[]
    for i,item in enumerate(m["pairs"]):
        before,after,*_=read_pair(root,item)
        f0,_=encoder.cached(before);f1,_=encoder.cached(after)
        features.append(np.stack([f0,f1]).astype(np.float16))
        if (i+1)%16==0:print(f"frozen SECOND cache {i+1}/{len(m['pairs'])}",flush=True)
    folder=root/"cache/second_scaled";folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/"features.npz",features=np.stack(features),ids=np.array([x["id"] for x in m["pairs"]]))
    (folder/"metadata.json").write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),encoder=encoder.metadata,
        shape=[len(features),2,256,384],label_inputs_to_encoder=False),indent=2),encoding="utf-8")

if __name__=="__main__":main()
