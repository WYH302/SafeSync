"""Plot actual receiver evidence and image quality with both link directions."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/"results/pixel_sequences/development_side4"
    marker=json.loads((run/"replay_verification.json").read_text())
    data=run/"receiver_evaluation.csv"
    if marker.get("status")!="complete" or hashlib.sha256(data.read_bytes()).hexdigest()!=marker["evaluation_csv_sha256"]:raise ValueError("evaluation is incomplete or changed")
    records=list(csv.DictReader((run/"records.csv").open(encoding="utf-8")))
    evaluation=list(csv.DictReader(data.open(encoding="utf-8")))
    lookup={(r["aoi"],r["method"],r["budget"],r["month"]):r for r in evaluation}
    methods={"full_jpeg":("Whole JPEG","#0072B2","o"),"full_jpeg2000":("Whole JPEG2000","#56B4E9","s"),
             "full_reference_mse":("Full-cache MSE","#009E73","D"),"gaussian_detail":("Gaussian reference","#D55E00","^"),"dct_bound":("DCT + norm reference","#CC79A7","v")}
    aois=sorted({r["aoi"] for r in records if r["split"]=="development"})
    summary=[]
    for method in methods:
        for budget in (32768,65536,131072):
            parts=[]
            for aid in aois:
                rr=[r for r in records if r["aoi"]==aid and r["method"]==method and int(r["budget"])==budget]
                ev=[lookup[(r["aoi"],r["method"],r["budget"],r["month"])] for r in rr]
                n=sum(int(x["new_track_pixels"]) for x in ev);good=sum(int(x["new_track_usable_pixels"]) for x in ev)
                parts.append(dict(aoi=aid,update_bytes=np.mean([int(r["down_bytes"])+int(r["up_bytes"]) for r in rr]),
                    boot_bytes=int(rr[0]["initial_down_bytes"])+int(rr[0]["initial_ack_bytes"]),frames=len(rr),
                    mean_mse=np.mean([float(r["valid_mse"]) for r in rr]),event_pixels=n,usable_event_pixels=good,event_coverage=good/n if n else None))
            valid=[p["event_coverage"] for p in parts if p["event_coverage"] is not None]
            summary.append(dict(method=method,budget=budget,update_bytes=float(np.mean([p["update_bytes"] for p in parts])),
                amortized_total_bytes=float(np.mean([p["update_bytes"]+p["boot_bytes"]/p["frames"] for p in parts])),
                mean_rgb_psnr=float(-10*np.log10(np.mean([p["mean_mse"] for p in parts]))),
                macro_aoi_event_coverage=float(np.mean(valid)),pooled_event_coverage=sum(p["usable_event_pixels"] for p in parts)/sum(p["event_pixels"] for p in parts),
                aoi_parts=parts))
    out=ROOT/"results/figures/development_v3";out.mkdir(parents=True,exist_ok=True)
    (out/"tradeoff_source.json").write_text(json.dumps(dict(scope="Historical development only; two AOIs; old router source not reconstructable; no confidence/generalization claim",quality_unit="new-track mask pixels grouped within 64x64 blocks; masked mean PSNR >=30 dB",summary=summary),indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.6))
    for method,(label,color,marker_shape) in methods.items():
        rr=[r for r in summary if r["method"]==method]
        x=[r["update_bytes"]/1024 for r in rr]
        axes[0].plot(x,[r["mean_rgb_psnr"] for r in rr],marker=marker_shape,color=color,label=label,linewidth=1.4,markersize=4)
        axes[1].plot(x,[r["macro_aoi_event_coverage"]*100 for r in rr],marker=marker_shape,color=color,label=label,linewidth=1.4,markersize=4)
    for ax in axes:
        ax.set_xlabel("Actual downlink + uplink per update (KiB)")
        ax.grid(axis="y",color="#dddddd",linewidth=.5)
    axes[0].set_ylabel("RGB PSNR from mean scene MSE (dB)")
    axes[1].set_ylabel("New-track evidence coverage (%)")
    axes[0].set_title("(a) Whole-image reconstruction")
    axes[1].set_title("(b) Region quality, macro by AOI")
    axes[1].set_ylim(0,100)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.005),fontsize=8)
    fig.suptitle("Development diagnostic: two geographic regions, 45 monthly updates",fontsize=10,y=.98)
    fig.tight_layout(rect=(0,.15,1,.94),w_pad=1.5)
    for suffix in ("pdf","png"):fig.savefig(out/f"quality_vs_evidence.{suffix}",dpi=220,bbox_inches="tight")
    print(out/"quality_vs_evidence.png")

if __name__=="__main__":main()
