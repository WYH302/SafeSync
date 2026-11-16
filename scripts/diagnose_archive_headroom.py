"""Zero-additional-rate diagnostic, not an achievable communication result."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import sys,json,csv,hashlib,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from safesync.pixel_wire import PixelState,TransactionReceiver
from safesync.reference import restore_coarse,exact_mse


def main():
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"resources/ground_eval/dataset_manifest.json").read_text(encoding="utf-8"))
    run=root/"results/pixel_sequences/development_side4"
    records=list(csv.DictReader((run/"records.csv").open(encoding="utf-8")))
    rows=[]
    for aoi in m["aois"]:
        monthmap={x["month"]:x for x in aoi["months"]};width,height=aoi["months"][0]["dimensions"]
        h,w=math.ceil(height/64)*64,math.ceil(width/64)*64
        first=np.zeros((h,w,3),np.uint8);first[:height,:width]=np.array(Image.open(root/aoi["months"][0]["rgb_path"]))
        for budget in (32768,65536):
            region=hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
            receiver=TransactionReceiver(PixelState(np.zeros_like(first),region))
            receiver.receive((run/aoi["aoi_id"]/"initial.bin").read_bytes())
            for r in [r for r in records if r["aoi"]==aoi["aoi_id"] and r["method"]=="full_jpeg" and int(r["budget"])==budget]:
                receiver.receive((root/r["packet_path"]).read_bytes());jpeg=receiver.state.rgb
                current=np.zeros_like(first);current[:height,:width]=np.array(Image.open(root/monthmap[r["month"]]["rgb_path"]))
                archive=restore_coarse(first,current)
                err_j,err_h=exact_mse(current,jpeg),exact_mse(current,archive)
                use=err_h<err_j
                pixelmask=use.reshape(h//64,w//64).repeat(64,0).repeat(64,1)
                mixed=np.where(pixelmask[:,:,None],archive,jpeg)
                valid=np.zeros((h,w),bool);valid[:height,:width]=np.array(Image.open(root/monthmap[r["month"]]["valid_mask_path"]))>0
                def mse(a):return float(np.mean(((current.astype(np.float64)-a)/255)**2,axis=-1)[valid].mean())
                rows.append(dict(aoi=aoi["aoi_id"],month=r["month"],budget=budget,jpeg_mse=mse(jpeg),zero_cost_archive_fusion_mse=mse(mixed),beneficial_blocks=int(use.sum())))
    out=root/"results/archive_headroom_diagnostic.json"
    out.write_text(json.dumps(dict(scope="Full initial archive, exact current pixel error, unlimited zero-cost selection/coarse metadata: diagnostic only, not a global bound for other methods",records=rows),indent=2),encoding="utf-8")
    for budget in (32768,65536):
        q=[r for r in rows if r["budget"]==budget]
        print(budget,"relative MSE reduction",1-sum(r["zero_cost_archive_fusion_mse"] for r in q)/sum(r["jpeg_mse"] for r in q),"mean beneficial blocks",sum(r["beneficial_blocks"] for r in q)/len(q))

if __name__=="__main__":main()
