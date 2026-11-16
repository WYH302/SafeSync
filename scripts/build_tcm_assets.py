"""Bind complete modern-codec comparisons to paper tables, plots and reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(residual_current="Residual + distance", residual_change="Residual + change", absolute_current="Absolute + distance",
    hyperprior="Hyperprior", full_webp="WebP", tcm="TCM", gaussian_20261029="Gaussian", dct_bound="DCT",
    full_cache_mse="Full history (25 dB)", full_cache_gate30="Full history (30 dB)", learned_hyperprior_q1_8="Hyperprior")
COLORS = dict(residual_current="#0072B2",residual_change="#009E73",hyperprior="#CC79A7",full_webp="#555555",tcm="#D55E00",
    gaussian_20261029="#0072B2",dct_bound="#E69F00",full_cache_gate30="#009E73",learned_hyperprior_q1_8="#CC79A7")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8",newline="")))


def table(path, caption, label, columns, body):
    end = chr(92)*2
    lines = [r"\begin{table*}[t]\centering\footnotesize",r"\caption{"+caption+"}",r"\label{"+label+"}",
        r"\begin{tabular}{l"+"r"*(len(columns)-1)+r"}\toprule"," & ".join(columns)+end+r"\midrule"]
    lines += [" & ".join(map(str,row))+end for row in body]
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines)+"\n","utf-8")


def build(study):
    second = study == "second"
    base = ROOT / ("results/task_evidence/tcm_analysis_v1" if second else "results/tcm_sn7/analysis_v1")
    analysis = json.loads((base / "analysis.json").read_text("utf-8"))
    if analysis["status"] != "complete":
        raise ValueError("Modern comparator is incomplete")
    for relative, expected in analysis["inputs"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Changed modern-comparator analysis input")
    summary, comparisons = rows(base / "summary.csv"), rows(base / "comparisons.csv")
    outputs = []
    tables = ROOT / ("paper/supplement/generated" if second else "paper/generated")
    report = ["# TCM追加比较结果", "", analysis["scope"], ""]
    if second:
        primary = [r for r in summary if int(r["heldout"]) == 5 and int(r["budget"]) == 8192]
        path = tables / "tcm_second_primary.tex"
        table(path,"Additional TCM control on the third 256-pair cohort, building-held-out at 8 KiB. Every method receives the exact same current task array. Actual transaction sizes differ within the shared cap.",
            "tab:tcm-second",["Method",r"$U_{30}$","Later class F1","Mean bytes"],
            [[LABELS[r["method"]],f"{float(r['unknown_coverage_30']):.4f}",f"{float(r['later_class_f1']):.4f}",f"{float(r['mean_total_bytes']):.1f}"] for r in primary])
        outputs.append(path)
        report += ["| 方法 | U30 | 后续F1 | 实际平均字节 |","|---|---|---|---|"]
        report += [f"| {r['method']} | {float(r['unknown_coverage_30']):.5f} | {float(r['later_class_f1']):.5f} | {float(r['mean_total_bytes']):.1f} |" for r in primary]
        report += ["", "| 相对TCM的方法 | U30差值，百分点 | 95%区间 | 后续F1差值，百分点 | 95%区间 |","|---|---|---|---|---|"]
        contrasts = [r for r in comparisons if int(r["heldout"]) == 5 and int(r["budget"]) == 8192]
        for r in contrasts:
            report.append(f"| {r['method']} | {100*float(r['coverage_difference']):+.3f} | [{100*float(r['coverage_low']):+.3f}, {100*float(r['coverage_high']):+.3f}] | {100*float(r['later_f1_difference']):+.3f} | [{100*float(r['later_low']):+.3f}, {100*float(r['later_high']):+.3f}] |")
        report += ["", "该参照属于已获取第三批上的追加比较，保留原联合确认的失败判定。现代TCM编码及分离用途评价共3,072项，完整三类别四预算汇总保留。首要区间为描述性95%图像对重采样区间，其他类别预算采用原定家族区间。地理独立性缺少元数据支持。"]
        methods = ("residual_current","residual_change","hyperprior","full_webp","tcm")
        fig,axes = plt.subplots(2,3,figsize=(7.1,4),constrained_layout=True)
        for column,(h,title) in enumerate(((5,"Building"),(3,"Trees"),(4,"Water"))):
            axes[0,column].set_title(title)
            for method in methods:
                group = sorted([r for r in summary if int(r["heldout"]) == h and r["method"] == method],key=lambda r:int(r["budget"]))
                for line,field in enumerate(("unknown_coverage_30","later_class_f1")):
                    axes[line,column].plot([float(r["mean_total_bytes"])/1024 for r in group],[float(r[field]) for r in group],
                        marker="o",markersize=3,linewidth=1.3,label=LABELS[method],color=COLORS[method])
            for ax in axes[:,column]:
                ax.set_xscale("log",base=2)
                ax.set_xticks([4,8,16,32],labels=["4","8","16","32"])
            axes[1,column].set_xlabel("Actual transaction size (KiB)")
        axes[0,0].set_ylabel("Qualified evidence coverage")
        axes[1,0].set_ylabel("Later held-out-class F1")
        report_path = ROOT / "docs/TCM_SECOND_RESULTS.md"
        pdf = ROOT / "paper/supplement/figures/tcm_second_curves.pdf"
    else:
        methods = ("gaussian_20261029","dct_bound","full_cache_mse","full_cache_gate30","full_webp","learned_hyperprior_q1_8","tcm")
        lookup = {(r["method"],int(r["budget"])):r for r in summary}
        body = []
        for method in methods:
            cells = [LABELS[method]]
            for cap in (32768,65536,131072):
                r = lookup[method,cap]
                cells += [f"{float(r['building_f1']):.3f}",f"{float(r['event_f1']):.3f}",f"{float(r['mean_bytes'])/1024:.1f}"]
            body.append(cells)
        path = tables / "tcm_oof.tex"
        table(path,"Additional modern-codec comparison using the same author-fold ground assay on eight areas. Within each 32/64/128-KiB downlink-cap group, columns give building F1, event F1, and actual mean bidirectional update KiB. AOIs have equal weight. The TCM control was appended after inspecting the main cohort.",
            "tab:tcm-oof",["Method","B32","E32","KiB32","B64","E64","KiB64","B128","E128","KiB128"],body)
        outputs.append(path)
        report += ["| 方法 | 上限KiB | 建筑F1 | 事件F1 | 实际平均更新KiB |","|---|---|---|---|---|"]
        report += [f"| {r['method']} | {int(r['budget'])//1024} | {float(r['building_f1']):.5f} | {float(r['event_f1']):.5f} | {float(r['mean_bytes'])/1024:.2f} |" for r in summary]
        report += ["", "| 相对TCM的方法 | 上限KiB | 任务 | F1差值，百分点 | 98.333%区间 |","|---|---|---|---|---|"]
        report += [f"| {r['method']} | {int(r['budget'])//1024} | {r['task']} | {100*float(r['mean_difference']):+.3f} | [{100*float(r['low']):+.3f}, {100*float(r['high']):+.3f}] |" for r in comparisons]
        report += ["", "固定35候选，原八AOI与178次更新全部保留，完成534个实际码流及对应OOF任务观测。先在AOI内累计混淆计数，再等权平均八个AOI。10,000次成对AOI重采样给出三预算家族区间。两方法同上限，实际交易字节不同。原作者使用验证折选择最佳epoch的限制仍适用。"]
        fig,axes = plt.subplots(1,2,figsize=(7.1,2.8),constrained_layout=True)
        for method in ("gaussian_20261029","dct_bound","full_cache_gate30","full_webp","learned_hyperprior_q1_8","tcm"):
            group = [lookup[method,cap] for cap in (32768,65536,131072)]
            for ax,field in zip(axes,("building_f1","event_f1")):
                ax.plot([float(r["mean_bytes"])/1024 for r in group],[float(r[field]) for r in group],marker="o",markersize=3,linewidth=1.3,label=LABELS[method],color=COLORS[method])
        for ax,title in zip(axes,("Building F1","New-track event F1")):
            ax.set_ylabel(title)
            ax.set_xlabel("Actual mean bidirectional update (KiB)")
        report_path = ROOT / "docs/TCM_SN7_RESULTS.md"
        pdf = ROOT / "paper/figures/tcm_oof_curves.pdf"
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncols=3,frameon=False)
    fig.savefig(pdf)
    png = base / "curves.png"
    fig.savefig(png,dpi=180)
    plt.close(fig)
    report_path.write_text("\n".join(report)+"\n","utf-8")
    outputs += [pdf,png,report_path]
    manifest = tables / ("tcm_second_manifest.json" if second else "tcm_sn7_manifest.json")
    manifest.write_text(json.dumps(dict(status="complete",generator_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (base / "analysis.json",base / "summary.csv",base / "comparisons.csv")},
        output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs}),indent=2)+"\n","utf-8")
    print(json.dumps(dict(study=study,outputs=len(outputs))),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",choices=("second","sn7"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"ps.fonttype":42})
    build(args.study)
