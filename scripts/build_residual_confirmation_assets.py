"""Generate traceable third-cohort results and the two bounded diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.analyze_task_context_confirmation import later_counts

ANALYSIS = ROOT / "results/task_evidence/residual_confirmation_analysis_v1"
TABLES = ROOT / "paper/supplement/generated"
GATE = ROOT / "results/task_evidence/feature_gate_development_v1"
REFERENCE = ROOT / "results/task_evidence/sender_reference_v1/verification.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row, low, high):
    return f"[{100*row[low]:+.2f}, {100*row[high]:+.2f}]"


def gate_summary():
    receipt = json.loads((GATE / "complete.json").read_text("utf-8"))
    if receipt["records_sha256"] != sha(GATE / "records.csv") or not receipt["feature_condition_exact"]:
        raise ValueError("Changed exploratory gate observations")
    rows = list(csv.DictReader((GATE / "records.csv").open(encoding="utf-8")))
    source = ROOT / "results/task_evidence/residual_confirmation_ground_v1/records.csv"
    if receipt["source_ground_sha256"] != sha(source):
        raise ValueError("Changed original receiver observations")
    ids = {r["pair_id"] for r in rows}
    controls = [r for r in csv.DictReader(source.open(encoding="utf-8")) if r["pair_id"] in ids
        and r["method"] in ("full_webp", "hyperprior", "residual_change")]
    result = []
    for method in ("feature_gate", "quality_only", "residual_change", "hyperprior", "full_webp"):
        group = [r for r in rows + controls if r["method"] == method and int(r["heldout"]) == 5 and int(r["budget"]) == 8192]
        if len(group) != 64:
            raise ValueError("Incomplete primary gate diagnostic")
        retained = sum(float(r["unknown_retained_pixels_30"]) for r in group)
        support = sum(float(r["unknown_pixels"]) for r in group)
        future = np.sum([np.array(json.loads(r["future_confusion"]), float) for r in group], axis=0)
        numerator, denominator = later_counts(future, 5)
        result.append(dict(method=method, pairs=64, coverage=retained/support, later_f1=float(numerator/denominator),
            mean_bytes=float(np.mean([float(r["total_bytes"]) for r in group]))))
    return result


def main():
    analysis = json.loads((ANALYSIS / "analysis.json").read_text("utf-8"))
    for name, expected in analysis["inputs"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed third-cohort analysis input")
    if analysis["primary_joint_pass"]:
        raise ValueError("This report expects the retained failed joint confirmation")
    reference = json.loads(REFERENCE.read_text("utf-8"))
    for name, expected in reference["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed compact-reference verification input")
    gate = gate_summary()
    TABLES.mkdir(parents=True, exist_ok=True)
    end = chr(92)*2
    names = dict(residual_change="Residual/change", absolute_current="Absolute/current distance", hyperprior="Full hyperprior", full_webp="Full WebP")
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Third pre-access-locked confirmation, 256 further pairs, building-held-out at 8 KiB. Residual/current-distance allocation is compared with each control. Differences and paired 95\% intervals are percentage points. All current task arrays match exactly. The seven-condition joint test fails.}",
        r"\label{tab:residual-confirmation}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{7.5pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Comparator} & \textbf{$\Delta U_{30}$} & \textbf{Interval} & \textbf{$\Delta$ later F1} & \textbf{Interval} & \textbf{Bytes, method/control}"+end+r"\midrule"]
    report = ["# 第三组冻结确认与后续诊断", "", "## 冻结确认结果", "",
        "第三组256对SECOND图像在像素访问前固定方案，完成29,952个实际接收端观测。七项联合条件通过五项，联合验收未通过。各方法接收逐样本完全一致的当前任务数组，首要条件为建筑留出与8 KiB总交易上限。这个比较固定预算上限，实际平均字节仍有差别。", "",
        "| 对照 | U30差值，百分点 | 95%区间 | 后续F1差值，百分点 | 95%区间 | 实际字节，新/对照 |", "|---|---|---|---|---|---|"]
    for key, r in analysis["primary"].items():
        values = f"{100*r['coverage_difference']:+.3f} | {interval(r,'coverage_low','coverage_high')} | {100*r['later_class_f1_difference']:+.3f} | {interval(r,'later_low','later_high')} | {r['mean_method_bytes']:.1f}/{r['mean_baseline_bytes']:.1f}"
        report.append(f"| {key} | {values} |")
        lines.append(f"{names[key]} & {100*r['coverage_difference']:+.2f} & {interval(r,'coverage_low','coverage_high')} & {100*r['later_class_f1_difference']:+.2f} & {interval(r,'later_low','later_high')} & {r['mean_method_bytes']:.0f}/{r['mean_baseline_bytes']:.0f} "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    outputs = []
    path = TABLES / "residual_confirmation.tex"
    path.write_text("\n".join(lines)+"\n", "utf-8")
    outputs.append(path)
    report += ["", "| 预设条件 | 结果 |", "|---|---|"]
    report += [f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in analysis["primary_conditions"].items()]
    report += ["", "Unknown当前距离组合相对同残差编码Change排序只增加0.081个百分点，区间跨零。对WebP的后续F1差值下界低于−0.02，未通过非劣标准。残差编码相对绝对编码及旧Hyperprior的质量合格覆盖有正向证据，原Unknown独立贡献仍未成立。相对Hyperprior与WebP的漏失比例分别降低25.22%和41.39%，这些数值对应指定的质量覆盖代理。", "",
        "U30先在64×64格内计算真实未知支持上的RGB均方误差，再累计达到30 dB条件的未知像素。它是质量代理。后续F1另外使用固定全类别地面头与实际接收RGB，在明确语义变化支持上评价。三个类别、四个上限与随机重复全部保留。10,000次成对图像重采样用于首要区间，次要比较保留原多重比较区间。图像对ID与前两批互斥，地理独立性没有可核查元数据。", "",
        "## 发送端紧凑参考", "",
        f"在32对已查看过的开发图像上，3,072个实际交易全部通过紧凑参考编码字节一致与ACK核验。序列化参考为424 bytes，包含尺寸、任务数组、状态、版本和缓存digest，另保留所有方法共用的196,608-byte历史FP16特征。旧RGB PNG平均{reference['mean_actual_png_bytes']:.1f} bytes。合并同一特征缓存后，序列化历史大小平均减少{100*reference['mean_combined_history_reduction_vs_png']:.2f}%。", "",
        "该组件适用于携带完整当前画面的FULL/LEARNED/LAYERED模式，验证的是替代编码器与实际既存交易的等价性。原确认编码过程仍使用完整参考。模型、共享gallery、当前帧、工作区和在途包均不属于上述序列化参考计数。", "",
        "## 零容差特征门的开发负结果", "",
        "在第三批已查看过的64对中，限定候选为完整WebP、完整Hyperprior与Residual/Change。特征门只允许当前特征重建误差不超过WebP的候选，再最大化同一Change加权质量。发送端决策先提交，随后复用哈希完全相同的已有接收端测量。完成1,536项选包观测，属于探索性重复分析。", "",
        "| 方法 | U30 | 后续建筑F1 | 实际平均字节 |", "|---|---|---|---|"]
    report += [f"| {r['method']} | {r['coverage']:.4f} | {r['later_f1']:.4f} | {r['mean_bytes']:.1f} |" for r in gate]
    report += ["", "零容差特征门使质量覆盖接近WebP，原覆盖增量消失。保留该失败，暂停门阈值和Unknown权重的继续搜索。TCM作者公开现代codec参照的执行与结果另行记录。", "",
        "## 文件与结论范围", "", "原始结果位于results/task_evidence/residual_confirmation_v1、residual_confirmation_ground_v1及residual_confirmation_analysis_v1。协议锁位于RESIDUAL_CONFIRMATION_PROTOCOL.md。原始实施计划、验收标准与准备工作保持原文。原科研总验收仍未通过，历史FINAL-02包仅代表其当时收缩范围。"]
    path = ROOT / "docs/RESIDUAL_CONFIRMATION_RESULTS.md"
    path.write_text("\n".join(report)+"\n", "utf-8")
    outputs.append(path)
    summaries = list(csv.DictReader((ANALYSIS / "summary.csv").open(encoding="utf-8")))
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":8.2, "axes.labelsize":8.2, "axes.titlesize":8.6,
        "xtick.labelsize":7.6, "ytick.labelsize":7.6, "text.color":"#000000", "axes.labelcolor":"#000000",
        "axes.edgecolor":"#1F2937", "xtick.color":"#000000", "ytick.color":"#000000",
        "axes.linewidth":0.55, "xtick.major.width":0.55, "ytick.major.width":0.55, "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.85), constrained_layout=True)
    methods = (("residual_current","Residual + distance","#1B4965","-","o"),
               ("residual_change","Residual + change","#5C88AA","--","s"),
               ("absolute_current","Absolute + distance","#8C6D46","-.","^"),
               ("hyperprior","Hyperprior","#4D7C6F","-","D"),
               ("full_webp","WebP","#334155",":","v"))
    tags = ("(a)", "(b)", "(c)")
    for column, (heldout, title) in enumerate(((5,"Building"),(3,"Trees"),(4,"Water"))):
        for method, label, color, ls, mk in methods:
            group = sorted([r for r in summaries if int(r["heldout"]) == heldout and r["method"] == method and int(r["budget"])], key=lambda r:int(r["budget"]))
            x = [float(r["mean_total_bytes"])/1024 for r in group]
            lw = 1.15 if method == "residual_current" else 0.88
            for row, field in enumerate(("unknown_coverage_30","later_class_f1")):
                axes[row,column].plot(x,[float(r[field]) for r in group],label=label,color=color,linestyle=ls,marker=mk,markersize=2.8,linewidth=lw)
        axes[0,column].set_title(f"{tags[column]} {title}", fontsize=8.6, fontweight="bold", loc="center")
        for ax in axes[:,column]:
            ax.set_xscale("log",base=2)
            ax.set_xticks([4,8,16,32],labels=["4","8","16","32"])
            ax.grid(alpha=.22, lw=.38, color="#CBD5E1")
            ax.spines[["top","right"]].set_visible(False)
        axes[1,column].set_xlabel("Actual transaction size (KiB)")
    axes[0,0].set_ylabel("Qualified evidence coverage")
    axes[1,0].set_ylabel("Later held-out-class F1")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncols=3,frameon=False)
    figure = ROOT / "paper/supplement/figures/residual_confirmation_curves.pdf"
    fig.savefig(figure)
    png = ANALYSIS / "residual_confirmation_curves.png"
    fig.savefig(png,dpi=180)
    plt.close(fig)
    outputs += [figure,png]
    sources = (ANALYSIS / "analysis.json", ANALYSIS / "summary.csv", REFERENCE, GATE / "complete.json", GATE / "records.csv")
    (TABLES / "residual_confirmation_manifest.json").write_text(json.dumps(dict(status="complete",generator_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},feature_gate_primary=gate),indent=2)+"\n","utf-8")
    print(json.dumps(dict(generated=len(outputs),joint_pass=analysis["primary_joint_pass"])),flush=True)


if __name__ == "__main__":
    main()
