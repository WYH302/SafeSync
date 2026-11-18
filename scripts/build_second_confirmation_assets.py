"""Paper tables, plots and report from complete confirmation statistics."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "results/second_confirmation_analysis/v1"
FUTURE = ROOT / "results/future_semantic_probe/v1/confirmation"
OUT = ROOT / "paper/supplement/generated"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main():
    analysis = json.loads((sys_path / "analysis.json").read_text("utf-8"))
    future = json.loads((FUTURE / "analysis.json").read_text("utf-8"))
    for relative, expected in analysis["inputs"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Changed confirmation analysis input")
    summary = {(int(r["heldout"]), int(r["budget"]), r["method"]): r for r in rows(sys_path / "summary.csv")}
    future_summary = {(int(r["heldout"]), int(r["budget"]), r["method"]): r for r in rows(FUTURE / "summary.csv")}
    future_diff = {(int(r["heldout"]), int(r["budget"]), r["method"], r["comparator"]): r for r in rows(FUTURE / "comparisons.csv")}
    end = chr(92) * 2
    lines = [r"\begin{table*}[t]", r"\centering",
        r"\caption{Internally locked primary contrasts on all 256 reserved SECOND pairs. Coverage differences and intervals are percentage points. Known-F1 intervals use pooled receiver confusion counts.}",
        r"\label{tab:second-primary}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{6.5pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrrr}\toprule",
        r"\rowcolor{tableheadbg} Protocol & KiB & $\Delta U_{30}$ & 95\% interval & $\Delta$ known F1 & 95\% interval & Mean bytes, new/base" + end + r"\midrule"]
    for key, label in (("original_image", "Image routing"), ("hybrid_extension", "PQ extension")):
        r = analysis["primary"][key]
        lines.append(f"{label} & {r['budget']//1024} & {100*r['delta_unknown_coverage_30']:+.2f} & [{100*r['unknown_low']:+.2f},{100*r['unknown_high']:+.2f}] & {r['delta_known_macro_f1']:+.4f} & [{r['known_low']:+.4f},{r['known_high']:+.4f}] & {r['mean_new_bytes']:.0f}/{r['mean_old_bytes']:.0f} " + end)
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table*}"]
    primary_table = OUT / "second_primary.tex"
    primary_table.write_text("\n".join(lines) + "\n", "utf-8")
    lines = [r"\begin{table*}[t]", r"\centering",
        r"\caption{All reserved-set classes and caps. Evidence coverage is $U_{30}$. The PQ emergence mixture is compared with full-image WebP and the public hyperprior under actual packet-plus-feedback caps.}",
        r"\label{tab:second-budgets}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{5.2pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrrrrr}\toprule",
        r"\rowcolor{tableheadbg} Held out & KiB & PQ change $U$ & PQ mix $U$ & WebP $U$ & Hyperprior $U$ & PQ known F1 & WebP known F1 & Hyperprior known F1" + end + r"\midrule"]
    labels = {5: "Building", 3: "Tree", 4: "Water"}
    for h in (5, 3, 4):
        for b in (4096, 8192, 16384, 32768):
            c, m, w, l = [summary[(h, b, method)] for method in ("hybrid_change", "hybrid_emergent_mix", "image_full_webp", "image_hyperprior")]
            lines.append(f"{labels[h]} & {b//1024} & {float(c['unknown_coverage_30']):.3f} & {float(m['unknown_coverage_30']):.3f} & {float(w['unknown_coverage_30']):.3f} & {float(l['unknown_coverage_30']):.3f} & {float(m['known_macro_f1']):.3f} & {float(w['known_macro_f1']):.3f} & {float(l['known_macro_f1']):.3f} " + end)
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table*}"]
    budget_table = OUT / "second_budgets.tex"
    budget_table.write_text("\n".join(lines) + "\n", "utf-8")
    lines = [r"\begin{table*}[t]", r"\centering",
        r"\caption{Additional later semantic use on explicit changed support. This all-class ground head is fitted only to calibration labels. It includes the class held out of sender fitting. RGB columns use 8-KiB-cap outputs; the PQ-feature column uses the smaller feature-only stream.}",
        r"\label{tab:second-future}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{5.8pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrrr}\toprule",
        r"\rowcolor{tableheadbg} Target & Full RGB F1 & PQ-feature F1 & PQ-mix RGB F1 & WebP RGB F1 & Hyperprior RGB F1 & $\Delta$ mix/change, interval" + end + r"\midrule"]
    for h in (5, 3, 4):
        f = [future_summary[(h, b, method)] for b, method in ((0, "image_full_current"), (0, "hybrid_PQ_feature_only"),
            (8192, "hybrid_emergent_mix"), (8192, "image_full_webp"), (8192, "learned_hyperprior"))]
        d = future_diff[(h, 8192, "hybrid_emergent_mix", "hybrid_change")]
        values = " & ".join(f"{float(r['later_class_f1']):.3f}" for r in f)
        lines.append(f"{labels[h]} & {values} & {float(d['delta_later_class_f1']):+.3f} [{float(d['interval_low']):+.3f},{float(d['interval_high']):+.3f}] " + end)
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table*}"]
    future_table = OUT / "second_future.tex"
    future_table.write_text("\n".join(lines) + "\n", "utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    styles = (("image_feature_change", "RGB change", "#777777", "o"),
              ("image_emergent_mix", "RGB novelty", "#9467bd", "s"),
              ("hybrid_emergent_mix", "PQ + evidence", "#277da8", "^"),
              ("image_full_webp", "WebP", "#d18528", "d"),
              ("image_hyperprior", "Hyperprior", "#278158", "v"))
    for column, h in enumerate((5, 3, 4)):
        for method, label, color, marker in styles:
            values = [summary[(h, b, method)] for b in (4096, 8192, 16384, 32768)]
            x = [float(r["mean_total_bytes"]) / 1024 for r in values]
            axes[0, column].plot(x, [float(r["unknown_coverage_30"]) for r in values], color=color, marker=marker, lw=1, ms=3, label=label)
            axes[1, column].plot(x, [float(r["known_macro_f1"]) for r in values], color=color, marker=marker, lw=1, ms=3)
        axes[0, column].set(title=labels[h], ylim=(-.02, 1.02))
        axes[1, column].set(xlabel="Actual mean KiB / update")
        for row in range(2):
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xticks([4, 8, 16, 32], labels=["4", "8", "16", "32"])
            axes[row, column].grid(alpha=.18, lw=.5)
    axes[0, 0].set_ylabel("Usable evidence coverage")
    axes[1, 0].set_ylabel("Known macro-F1")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    figure = ROOT / "paper/supplement/figures/second_confirmation.pdf"
    fig.savefig(figure)
    fig.savefig(sys_path / "confirmation_curves.png", dpi=200)
    plt.close(fig)
    report = ["# SECOND完整独立验证", "",
        "本页汇总全部256对保留样本。原图像协议29,184条观测、PQ扩展19,200条、Hyperprior 1,024个实际包及3,072条任务观测均已完成。另一个全类别地面评价器完成46,080条后续用途观测。它们共享相同图像对，独立样本数为256。", "",
        "## 两个锁定首要比较", "",
        "原图像路由在建筑16 KiB下提高U30约0.942个百分点，95%区间为[0.363,1.512]个百分点。known macro-F1差为+0.00231，区间为[−0.00416,+0.01219]，满足−.02容差。", "",
        "PQ扩展在建筑8 KiB下提高U30约2.280个百分点，95%区间为[1.194,3.332]个百分点。known任务的逐对混淆计数完全相同。两协议敏感性97.5%区间仍为[1.034,3.495]个百分点。实际平均字节为7765.05对7778.50。", "",
        "原图像方法在test像素下载前锁定。PQ扩展在像素已下载、原实验已运行但仅查看进度计数时锁定，选择依据来自64对开发集。完整访问说明保存在各自method_lock.json。", "",
        "## 强比较和后续用途", "",
        "建筑8 KiB下，PQ扩展的known F1/U30为0.3962/0.4757，WebP为0.4224/0.5668，Hyperprior为0.4079/0.6722。Unknown评分相对强change路由有增量，现有PQ＋JPEG组合仍缺少对强图像codec的综合优势。", "",
        "全类别后续地面头只用校准标签拟合，包括发送端拟合时留出的类别。建筑8 KiB下，PQ混合路由实际RGB的后续F1为0.6412，change为0.6352，差值99.583%区间跨零。单独收到的PQ特征为0.7131，WebP RGB为0.8197，Hyperprior RGB为0.8107，完整RGB为0.8322。视觉质量覆盖的改善需要与具体后续任务结果分别解释。", "",
        "## 下一项受控修正", "",
        "只在原64对开发集进行任务标签／PQ表征与JPEG／WebP区域编码的交叉消融。全图codec也获得相同任务服务。新结果需要使用另行预留的256对未读取样本确认，旧保留集继续保留其原结果。原科研总验收仍由全部独立结论与强对照效果判定。", ""]
    report_path = ROOT / "docs/SECOND_CONFIRMATION_RESULTS.md"
    report_path.write_text("\n".join(report), "utf-8")
    inputs = [sys_path / "analysis.json", sys_path / "summary.csv", FUTURE / "analysis.json", FUTURE / "summary.csv", FUTURE / "comparisons.csv"]
    outputs = [primary_table, budget_table, future_table, figure, report_path]
    (OUT / "second_confirmation_manifest.json").write_text(json.dumps(dict(status="complete", generator_sha256=sha(Path(__file__)),
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}), indent=2) + "\n", "utf-8")
    print(json.dumps(dict(status="complete", tables=3, plots=1)), flush=True)


if __name__ == "__main__":
    main()
