"""Keep exploratory codec-selection and residual-atlas findings in the paper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/"paper/supplement/generated"
SCORE = ROOT/"results/task_evidence/hyperprior_risk_analysis_v1/analysis.json"
RESIDUAL = ROOT/"results/task_evidence/residual_analysis_v1/analysis.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(row,lo,hi):
    return f"[{100*row[lo]:+.2f},{100*row[hi]:+.2f}]"


def main():
    analyses = [json.loads(path.read_text("utf-8")) for path in (SCORE,RESIDUAL)]
    for analysis in analyses:
        if analysis["status"] != "complete":
            raise ValueError("Incomplete exploratory study")
        for relative,expected in analysis["inputs"].items():
            if sha(ROOT/relative) != expected:
                raise ValueError("Changed exploratory analysis input")
    TABLES.mkdir(parents=True,exist_ok=True)
    end = chr(92)*2
    outputs = []
    score,residual = analyses
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Exploratory score-only intervention on 64 previously inspected pairs, building-held-out at 8 KiB. The same hyperprior quality/scale candidates and exact task states are used throughout. Differences and descriptive 95\% paired intervals are percentage points.}",
        r"\label{tab:risk-objective}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{7.5pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{lXrrrr}\toprule",
        r"\rowcolor{tableheadbg} Score & Comparator & $\Delta U_{30}$ & Interval & $\Delta$ later F1 & Interval"+end+r"\midrule"]
    for r in score["primary_diagnostic"]:
        if r["comparator"] == "full_webp":
            continue
        method = "Current distance" if r["method"] == "current_distance_quality" else "Temporal increment"
        comparator = {"mse":"RGB MSE","uniform_quality":"Uniform quality","change_quality":"Change quality"}[r["comparator"]]
        lines.append(f"{method} & {comparator} & {100*r['coverage_difference']:+.2f} & {interval(r,'coverage_low','coverage_high')} & {100*r['later_class_f1_difference']:+.2f} & {interval(r,'later_low','later_high')} "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    path = TABLES/"risk_objective.tex"
    path.write_text("\n".join(lines)+"\n","utf-8")
    outputs.append(path)
    lines = [r"\begin{table*}[t]\centering",
        r"\caption{Exploratory learned-base absolute/residual factorial on 32 previously inspected pairs, building-held-out at 8 KiB. Residual/current-distance ranking is compared with each control. Differences and descriptive 95\% intervals are percentage points.}",
        r"\label{tab:residual-factorial}",
        r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{7.5pt}\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrr}\toprule",
        r"\rowcolor{tableheadbg} Comparator & $\Delta U_{30}$ & Interval & $\Delta$ later F1 & Interval & Bytes, residual/base"+end+r"\midrule"]
    labels = dict(absolute_current="Absolute atlas",residual_change="Residual/change",residual_random="Residual/random",hyperprior="Full hyperprior",full_webp="Full WebP")
    for r in residual["primary_diagnostic"]:
        lines.append(f"{labels[r['comparator']]} & {100*r['coverage_difference']:+.2f} & {interval(r,'coverage_low','coverage_high')} & {100*r['later_class_f1_difference']:+.2f} & {interval(r,'later_low','later_high')} & {r['mean_method_bytes']:.0f}/{r['mean_baseline_bytes']:.0f} "+end)
    lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    path = TABLES/"residual_factorial.tex"
    path.write_text("\n".join(lines)+"\n","utf-8")
    outputs.append(path)
    report = ["# 强codec后的两个受控开发检查","",
        "这两项使用已经查看过的新确认样本，分别固定64对和其中32对。原256对冻结联合结果仍为未通过。所有类别和预算保留在对应summary.csv与comparisons.csv中。这里的区间只作开发诊断。","",
        "## 只改变强codec候选选择目标","",
        "固定8质量×5分辨率实际Hyperprior候选、任务数组和预算，比较全局MSE、均匀有界质量、Change质量，以及当前距离/时序增量加权。完成3,840个实际码流和分离地面评价。","",
        "| 排序 | 对照 | U30差值，百分点 | 95%区间 | 后续F1差值，百分点 | 95%区间 |","|---|---|---|---|---|---|"]
    for r in score["primary_diagnostic"]:
        report.append(f"| {r['method']} | {r['comparator']} | {100*r['coverage_difference']:+.3f} | {interval(r,'coverage_low','coverage_high')} | {100*r['later_class_f1_difference']:+.3f} | {interval(r,'later_low','later_high')} |")
    report += ["","## 学习型底图与区域绝对/残差编码","",
        "固定半预算Hyperprior底图，交叉比较WebP70绝对/残差区域与Change/当前known距离排序，另给三种随机保护。各候选按同一实际解码质量代理与完整Hyperprior比较，取完整帧处理平局。RE01载荷记录底图、区域索引、模式和长度，接收端通过实际注册codec解码。完成3,072个实际码流与分离地面评价。","",
        "| 对照 | U30差值，百分点 | 95%区间 | 后续F1差值，百分点 | 95%区间 | 实际平均字节，新/对照 |","|---|---|---|---|---|---|"]
    for r in residual["primary_diagnostic"]:
        report.append(f"| {r['comparator']} | {100*r['coverage_difference']:+.3f} | {interval(r,'coverage_low','coverage_high')} | {100*r['later_class_f1_difference']:+.3f} | {interval(r,'later_low','later_high')} | {r['mean_method_bytes']:.1f}/{r['mean_baseline_bytes']:.1f} |")
    report += ["","## 复算范围","",
        "known任务状态在所有对照间完全一致。随机种子先在图像对内平均。后续F1来自相同冻结全类别头与明确变化标签支持。开发样本的地理独立性仍无可用信息。","",
        "原始汇总分别位于results/task_evidence/hyperprior_risk_analysis_v1与residual_analysis_v1。包完整性、离线复算成功和期刊科研验收分别记录。"]
    path = ROOT/"docs/TASK_EVIDENCE_FOLLOWUPS.md"
    path.write_text("\n".join(report)+"\n","utf-8")
    outputs.append(path)
    (TABLES/"task_followup_manifest.json").write_text(json.dumps(dict(status="complete",generator_sha256=sha(Path(__file__)),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (SCORE,RESIDUAL)},
        output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs}),indent=2)+"\n","utf-8")
    print(json.dumps(dict(generated=len(outputs)),indent=2))


if __name__ == "__main__":
    main()
