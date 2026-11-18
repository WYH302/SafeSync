"""Deterministic, GT-supported crops from actual decoded confirmation packets.

Select the first evaluable new-track month in confirmation AOIs 0 and 1;
center on its largest connected target component, without consulting method
scores. The same native 256x256 crop is used for every displayed method.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from safesync.layered_wire import LayeredReceiver
from safesync.pixel_wire import PixelState, decode_update
from safesync.full_image_codecs import receive


def replay(rows, height, width):
    rows=sorted(rows,key=lambda r:int(r["step"]))
    initial=(ROOT/rows[0]["bootstrap_path"]).read_bytes()
    if initial[:4]==b"LB01":
        decoder=LayeredReceiver()
        decoder.receive(initial)
        first=decoder.state.render().copy()
        state=None
    else:
        # Native shape is fixed; padded layout is part of the bootstrap.
        hh,ww=(height+63)//64*64,(width+63)//64*64
        region=hashlib.sha256(rows[0]["aoi"].encode()).digest()[:16]
        state,_,_=decode_update(PixelState(np.zeros((hh,ww,3),np.uint8),region),initial)
        first=state.rgb.copy()
        decoder=None
    for row in rows:
        packet=(ROOT/row["packet_path"]).read_bytes()
        if decoder:
            decoder.expect(int(row["step"]))
            ack=decoder.receive(packet)
            rgb=decoder.state.render()
        else:
            state,ack=receive(state,packet)
            rgb=state.rgb
        actual=hashlib.sha256(rgb.tobytes()).hexdigest()
        if actual!=row["receiver_rgb_sha256"]:
            raise ValueError("qualitative output is not the recorded receiver")
        if hashlib.sha256(ack).hexdigest()!=row["ack_sha256"]:
            raise ValueError("qualitative ACK differs")
    return first[:height,:width],rgb[:height,:width]


def main():
    manifest=json.loads((ROOT/"data/spacenet7_pilot/confirmation_manifest_v1.json").read_text())
    methods=["gaussian_20261029","full_webp"]
    labels=["Current RGB (GT)","Bootstrap (t = 0)","SafeSync-ZFSS (Ours)","Stateless WebP"]
    out=ROOT/"paper/figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8.0,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,4,figsize=(7.16,3.85))
    selections=[]
    for ai in (0,1):
        aoi=manifest["aois"][ai];run=ROOT/f"results/layered/confirmation_v1/area_{ai}"
        progress=json.loads((run/"progress.json").read_text())
        if progress["status"]!="complete":
            raise ValueError("complete frozen AOI required")
        rows=list(csv.DictReader((run/"records.csv").open(encoding="utf-8")))
        first_valid=np.array(Image.open(ROOT/aoi["months"][0]["valid_mask_path"]))>0
        chosen=None
        for t,m in enumerate(aoi["months"][1:],1):
            gt=np.array(Image.open(run/"gt"/aoi["aoi_id"]/m["month"]/"new_track.png"))>0
            valid=np.array(Image.open(ROOT/m["valid_mask_path"]))>0
            target=gt&valid&first_valid
            if target.sum()>=256:
                chosen=(t,m,target);break
        if chosen is None:
            raise ValueError("predeclared crop support absent; report rather than substitute a favorable AOI")
        step,month,target=chosen
        components,n=label(target)
        counts=np.bincount(components.ravel());counts[0]=0
        selected=int(np.argmax(counts))
        yy,xx=np.nonzero(components==selected)
        cy,cx=int(np.rint(yy.mean())),int(np.rint(xx.mean()))
        h,w=target.shape
        y=max(0,min(h-256,cy-128));x=max(0,min(w-256,cx-128))
        crop=(slice(y,y+256),slice(x,x+256))
        current=np.array(Image.open(ROOT/month["rgb_path"]))
        decoded={};packets={}
        for method in methods:
            sequence=[r for r in rows if r["method"]==method and int(r["budget"])==32768 and int(r["step"])<=step]
            first,rgb=replay(sequence,h,w)
            decoded[method]=rgb
            row=max(sequence,key=lambda r:int(r["step"]))
            packets[method]=dict(down_bytes=int(row["down_bytes"]),ack_bytes=int(row["up_bytes"]),
                                packet_sha256=row["packet_sha256"],receiver_rgb_sha256=row["receiver_rgb_sha256"])
        displayed=[current,first,decoded[methods[0]],decoded[methods[1]]]
        scores=[]
        for col,(rgb,title) in enumerate(zip(displayed,labels)):
            ax=axes[ai,col]
            ax.imshow(rgb[crop])
            if target[crop].any():ax.contour(target[crop],levels=[.5],colors=["#00E5FF"],linewidths=.75)
            ax.set_xticks([]);ax.set_yticks([])
            values=((current[crop].astype(float)-rgb[crop])/255)**2
            mask=target[crop]
            mse=float(values[mask].mean()) if mask.any() else None
            score=None if mse is None or mse==0 else float(-10*np.log10(mse))
            scores.append(dict(method=title,crop_target_pixel_mse=mse,crop_target_psnr=score,
                               exact_match=mse==0 if mse is not None else None))
            if score is not None and col>=1:
                ax.text(0.97,0.04,f"Target PSNR: {score:.1f} dB",transform=ax.transAxes,
                        ha="right",va="bottom",fontsize=6.8,fontweight="bold",color="white",
                        bbox=dict(boxstyle="round,pad=0.22",facecolor="#0F172A",edgecolor="none",alpha=0.78))
            tcol="#0F3D8A" if "SafeSync" in title else "#0F172A"
            ax.set_title(title if ai==0 else "",fontsize=8.3,fontweight="bold",color=tcol,pad=5)
            bcol="#1D4ED8" if "SafeSync" in title else "#94A3B8"
            blw=1.1 if "SafeSync" in title else 0.55
            for spine in ax.spines.values():spine.set_color(bcol);spine.set_linewidth(blw)
        axes[ai,0].set_ylabel(f"Area {ai+1}\n({month['month']})",fontsize=8.0,fontweight="bold")
        selections.append(dict(area_index=ai,aoi=aoi["aoi_id"],month=month["month"],step=step,
            crop_xywh=[x,y,256,256],target_pixels=int(target[crop].sum()),component_pixels=int(counts[selected]),
            selection="first chronological month with >=256 evaluable target pixels, center largest component; no method scores used",
            nominal_down_cap=32768,actual_packets=packets,scores=scores,
            run_spec_sha256=hashlib.sha256((run/"run_spec.json").read_bytes()).hexdigest()))
    fig.subplots_adjust(left=.055,right=.995,top=.93,bottom=.04,wspace=.04,hspace=.08)
    for extension in ("pdf","svg","png"):
        fig.savefig(out/f"receiver_crops.{extension}",dpi=220,facecolor="white")
    plt.close(fig)
    (out/"receiver_crops_selection.json").write_text(json.dumps(selections,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(selections,indent=2),flush=True)


if __name__=="__main__":main()
