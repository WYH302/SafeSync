"""Tables, curves and a factual report for the complete fresh joint confirmation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from safesync import task_evidence_wire as wire

ANALYSIS = ROOT / "results/task_evidence/fresh_analysis_v1"
TABLES = ROOT / "paper/supplement/generated"
FIGURES = ROOT / "paper/supplement/figures"
NAMES = {5:"Building", 3:"Tree", 4:"Water"}
METHODS = (("adaptive_mix", "Temporal novelty"), ("adaptive_change", "Change"),
    ("adaptive_current_distance", "Current distance"), ("adaptive_random", "Random"),
    ("full_webp", "WebP"), ("hyperprior", "Hyperprior"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def interval(row, low="coverage_low", high="coverage_high", scale=100):
    return f"[{scale*float(row[low]):+.2f}, {scale*float(row[high]):+.2f}]"


def main():
    analysis_path = ANALYSIS / "analysis.json"
    analysis = json.loads(analysis_path.read_text("utf-8"))
    if analysis["status"] != "complete":
        raise ValueError("Incomplete joint analysis")
    for relative, expected in analysis["input_sha256"].items():
        if sha(ROOT/relative) != expected:
            raise ValueError("Changed joint-analysis input")
    summary_path, comparison_path = ANALYSIS/"summary.csv", ANALYSIS/"comparisons.csv"
    if sha(summary_path) != analysis["summary_sha256"] or sha(comparison_path) != analysis["comparisons_sha256"]:
        raise ValueError("Changed summary or contrasts")
    summary = {(int(r["heldout"]), int(r["budget"]), r["method"]):r for r in read_csv(summary_path)}
    comparisons = {(int(r["heldout"]), int(r["budget"]), r["comparator"]):r for r in read_csv(comparison_path)}
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    end = chr(92)*2
    outputs = []
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Fresh, internally locked joint confirmation. Building is held out of sender calibration, with an 8-KiB transaction cap. Differences and two-sided paired 95\% intervals use percentage points. All compared methods recover exactly the same current known-task array.}",
        r"\label{tab:context-primary}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{7.0pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Comparator} & \textbf{$\Delta U_{30}$} & \textbf{95\% interval} & \textbf{$\Delta$ later F1} & \textbf{95\% interval} & \textbf{Bytes, mix/base}"+end+r"\midrule"]
    for name, row in analysis["primary"].items():
        label = dict(adaptive_change="Adaptive change", full_webp="Full WebP", hyperprior="Hyperprior")[name]
        lines.append(f"{label} & {100*row['coverage_difference']:+.2f} & {interval(row)} & {100*row['later_class_f1_difference']:+.2f} & {interval(row,'later_low','later_high')} & {row['mean_method_bytes']:.0f}/{row['mean_baseline_bytes']:.0f} "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    primary_path = TABLES/"context_primary.tex"
    primary_path.write_text("\n".join(lines)+"\n", "utf-8")
    outputs.append(primary_path)
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Every fresh held-out class and cap. Each cell gives quality-qualified evidence coverage $U_{30}$ followed by later held-out-class F1. Random results average three seeds within each source pair. Curves use actual mean transaction bytes.}",
        r"\label{tab:context-all}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{5.2pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Class} & \textbf{KiB} & "+" & ".join(rf"\textbf{{{label}}}" for _,label in METHODS)+" "+end+r"\midrule"]
    for h in (5,3,4):
        for cap in (4096,8192,16384,32768):
            cells = []
            for method, _ in METHODS:
                row = summary[h,cap,method]
                cells.append(f"{float(row['unknown_coverage_30']):.3f}/{float(row['later_class_f1']):.3f}")
            lines.append(f"{NAMES[h]} & {cap//1024} & "+" & ".join(cells)+" "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    all_path = TABLES/"context_all.tex"
    all_path.write_text("\n".join(lines)+"\n", "utf-8")
    outputs.append(all_path)
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Fresh component contrasts at the primary building/8-KiB setting. The temporal-novelty method is compared with each listed alternative. These additional contrasts use descriptive 99.5833\% intervals.}",
        r"\label{tab:context-ablations}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{8.0pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Comparator} & \textbf{$\Delta U_{30}$, pp} & \textbf{Interval} & \textbf{$\Delta$ later F1, pp} & \textbf{Interval}"+end+r"\midrule"]
    for name,label in (("adaptive_random","Random ranking"),("adaptive_current_distance","Current known distance"),
        ("roi_mix","Context-free ROI"),("context_mix","Context without full-frame selection")):
        row = comparisons[5,8192,name]
        lines.append(f"{label} & {100*float(row['coverage_difference']):+.2f} & {interval(row)} & {100*float(row['later_class_f1_difference']):+.2f} & {interval(row,'later_low','later_high')} "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    ablation_path = TABLES/"context_ablations.tex"
    ablation_path.write_text("\n".join(lines)+"\n", "utf-8")
    outputs.append(ablation_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":8.2, "axes.labelsize":8.2, "axes.titlesize":8.6,
        "xtick.labelsize":7.6, "ytick.labelsize":7.6, "text.color":"#000000", "axes.labelcolor":"#000000",
        "axes.edgecolor":"#1F2937", "xtick.color":"#000000", "ytick.color":"#000000",
        "axes.linewidth":0.55, "xtick.major.width":0.55, "ytick.major.width":0.55,
        "pdf.fonttype":42, "ps.fonttype":42, "axes.spines.top":False, "axes.spines.right":False, "savefig.dpi":300})
    colors = ["#1B4965", "#5C88AA", "#4D7C6F", "#8C6D46", "#64748B", "#334155"]
    linestyles = ["-", "--", "-", "-.", ":", "-"]
    markers = ["o","s","^","v","D","x"]
    fig, axes = plt.subplots(2,3,figsize=(7.16,4.35),sharex=True)
    tags = ("(a)", "(b)", "(c)")
    for col,h in enumerate((5,3,4)):
        for idx, ((method,label), color, ls, marker) in enumerate(zip(METHODS,colors,linestyles,markers)):
            group = [summary[h,b,method] for b in (4096,8192,16384,32768)]
            x = [float(r["mean_total_bytes"])/1024 for r in group]
            lw = 1.15 if idx == 0 else 0.88
            axes[0,col].plot(x,[float(r["unknown_coverage_30"]) for r in group],label=label,color=color,linestyle=ls,marker=marker,markersize=2.8,lw=lw)
            axes[1,col].plot(x,[float(r["later_class_f1"]) for r in group],color=color,linestyle=ls,marker=marker,markersize=2.8,lw=lw)
        axes[0,col].set_title(f"{tags[col]} {NAMES[h]}", fontsize=8.6, fontweight="bold", loc="center")
        for axis in axes[:,col]:
            axis.set_xscale("log", base=2)
            axis.set_xticks([4,8,16,32],labels=["4","8","16","32"])
            axis.set_ylim(0,1.02)
            axis.grid(alpha=.22, lw=.38, color="#CBD5E1")
        axes[1,col].set_xlabel("Actual mean bytes (KiB)")
    axes[0,0].set_ylabel("Evidence coverage $U_{30}$")
    axes[1,0].set_ylabel("Later held-out-class F1")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="lower center",frameon=False,bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.105,1,1),pad=.7)
    curve_path = FIGURES/"task_context_curves.pdf"
    png_path = ANALYSIS/"task_context_curves.png"
    fig.savefig(curve_path)
    fig.savefig(png_path)
    plt.close(fig)
    outputs += [curve_path,png_path]

    # Decompose real primary transactions, including feedback and nested context.
    source = read_csv(ROOT/"results/task_evidence/fresh_context_ground_v1/records.csv")
    byte_rows = []
    for method in ("adaptive_mix","adaptive_change","full_webp","hyperprior"):
        group = [r for r in source if int(r["heldout"]) == 5 and int(r["budget"]) == 8192 and r["method"] == method]
        components = []
        for row in group:
            path = ROOT/row["packet_path"]
            if sha(path) != row["packet_sha256"]:
                raise ValueError("Changed actual primary packet")
            packet = path.read_bytes()
            head = wire.HEADER.unpack_from(packet)
            cost = dict(header=wire.HEADER.size, manifest=head[14], indices=head[11]*2,
                task_labels=head[15], image=head[16], checksum=16, ack=int(row["ack_bytes"]))
            if sum(cost.values()) != int(row["total_bytes"]):
                raise AssertionError("Primary transaction components do not sum")
            components.append(cost)
        byte_rows.append(dict(method=method, **{k:float(np.mean([r[k] for r in components])) for k in components[0]}))
    byte_path = ANALYSIS/"byte_components.csv"
    with byte_path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(byte_rows[0]))
        writer.writeheader()
        writer.writerows(byte_rows)
    outputs.append(byte_path)
    status = "通过" if analysis["primary_joint_pass"] else "未通过"
    note = ["# 新256对独立确认结果", "", f"首要联合条件{status}。共256个互斥图像对ID，54,528条接收端评价记录，包含48,384条主实验/学习型控制和6,144条额外当前距离消融。", "",
        "首要设置为建筑留出、8 KiB事务上限。以下区间为图像对重采样的双侧95%区间，差值单位为百分点。", "",
        "| 对照 | U30差值 | 区间 | 后续F1差值 | 区间 | 实际平均字节，新/对照 |", "|---|---|---|---|---|---|"]
    for name,r in analysis["primary"].items():
        note.append(f"| {name} | {100*r['coverage_difference']:+.3f} | {interval(r)} | {100*r['later_class_f1_difference']:+.3f} | {interval(r,'later_low','later_high')} | {r['mean_method_bytes']:.1f}/{r['mean_baseline_bytes']:.1f} |")
    note += ["", "## 冻结条件逐项结果", ""]
    note += [f"- {name}，{'通过' if passed else '未通过'}。" for name,passed in analysis["primary_conditions"].items()]
    note += ["", "## 解释范围", "",
        "已知任务数组在所有当前服务之间逐样本完全一致。它由冻结骨干和known校准原型产生，质量上限受该头限制。未接收更新的旧状态对照单独保留。", "",
        "U30按接收端RGB在明确留出类像素上的局部质量门累计。后续F1来自固定全类别地面头，两者分别报告。随机种子先在同一原图对内平均，未被当作独立样本。", "",
        "本次新确认在下载像素前锁定。先前256对结果已用于诊断修正，不能合并成同一个未调参确认集。当前距离为首要锁之后新增的独立锁定消融，结果保留为额外比较。", "",
        "SECOND只有图像对ID互斥可核查，地理独立性未建立。留出类在发送端校准中排除，公开骨干预训练可能见过这些概念。后续头只代表一个具体语义用途。", "",
        "本次协议为双时相、无丢包确认。先前真实月度差分和故障组件保持独立范围。论文总验收仍依赖原始科学目标与已获得证据的实际对应。", "",
        "## 可复算输出", "", "- `results/task_evidence/fresh_analysis_v1/analysis.json`保存首要条件、区间和输入哈希。",
        "- 同目录`summary.csv`、`comparisons.csv`保存全部类别、预算、控制和消融。",
        "- 同目录`byte_components.csv`保存实际事务开销分解。",
        "- `paper/supplement/generated/context_*.tex`和`paper/supplement/figures/task_context_curves.pdf`可直接纳入稿件。"]
    report_path = ROOT/"docs/TASK_CONTEXT_CONFIRMATION_RESULTS.md"
    report_path.write_text("\n".join(note)+"\n","utf-8")
    outputs.append(report_path)
    manifest_path = TABLES/"task_context_manifest.json"
    manifest_path.write_text(json.dumps(dict(status="complete", generator_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (analysis_path,summary_path,comparison_path)},
        output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs}),indent=2)+"\n","utf-8")
    print(json.dumps(dict(primary_joint_pass=analysis["primary_joint_pass"], generated=len(outputs)),indent=2))


if __name__ == "__main__":
    main()
