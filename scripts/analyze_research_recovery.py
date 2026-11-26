"""Bind completed recovery runs to descriptive statistics, plots and report."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_recovery_v1"
RATE = ROOT / "results/feature_rate/equal_decode_v3"
LINK = ROOT / "results/link_recovery/real_features_v2"
UNKNOWN = ROOT / "results/unknown_recovery/codec_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def validate(folder, expected):
    complete = json.loads((folder / "complete.json").read_text("utf-8"))
    if complete["status"] != "complete":
        raise ValueError("Incomplete run")
    records = rows(folder / "records.csv")
    if len(records) != expected or sha(folder / "records.csv") != complete["records_sha256"]:
        raise ValueError("Incomplete or changed records")
    if sha(folder / "run_spec.json") != complete["spec_sha256"]:
        raise ValueError("Changed run specification")
    if sha(folder / "source_snapshot.py") != complete["source_sha256"]:
        raise ValueError("Source snapshot differs from completed run")
    return records


def interval(values, probability=.95):
    values = np.asarray(values, np.float64)
    sample = np.random.default_rng(20261104).integers(0, len(values), (10000, len(values)))
    limits = np.quantile(values[sample].mean(1), [(1 - probability) / 2, (1 + probability) / 2])
    return dict(mean=float(values.mean()), low=float(limits[0]), high=float(limits[1]))


def main():
    OUT.mkdir(exist_ok=True)
    validate(RATE, 270)
    validate(LINK, 19440)
    validate(UNKNOWN, 7680)
    rate = [r for r in rows(RATE / "per_aoi.csv") if r["split"] == "confirmation"]
    rate_stats = {key: interval([float(r[key]) for r in rate]) for key in (
        "relative_update_saving", "relative_complete_sequence_saving")}
    group = defaultdict(list)
    for row in rows(LINK / "per_aoi_seed.csv"):
        if row["split"] == "confirmation":
            group[(row["profile"], row["policy"], row["aoi"])].append(row)
    metrics = ("mean_update_bytes", "mean_world_mse", "mean_reference_mse", "mean_age", "current_fraction")
    per_area = {key: {metric: float(np.mean([float(r[metric]) for r in value])) for metric in metrics}
                for key, value in group.items()}
    link_stats = []
    for profile in ("lossless", "iid10", "iid30", "burst"):
        ids = sorted(key[2] for key in per_area if key[:2] == (profile, "risk_anchor"))
        for comparator in ("delta_arq", "periodic_4", "periodic_8", "periodic_16"):
            new = [per_area[(profile, "risk_anchor", a)] for a in ids]
            old = [per_area[(profile, comparator, a)] for a in ids]
            result = dict(profile=profile, method="risk_anchor", comparator=comparator, paired_AOIs=len(ids))
            for metric in metrics:
                result["delta_" + metric] = interval([a[metric] - b[metric] for a, b in zip(new, old)], .9875)
                result["new_" + metric] = float(np.mean([a[metric] for a in new]))
                result["old_" + metric] = float(np.mean([b[metric] for b in old]))
            result["relative_world_mse_reduction"] = 1 - result["new_mean_world_mse"] / result["old_mean_world_mse"]
            result["relative_update_byte_change"] = result["new_mean_update_bytes"] / result["old_mean_update_bytes"] - 1
            link_stats.append(result)
    unknown = rows(UNKNOWN / "comparisons.csv")
    real = [r for r in unknown if r["diagnostic_oracle"] == "False" and r["comparator"] == "feature_change"]
    positive = [r for r in real if r["ci_low"] and float(r["ci_low"]) > 0]
    joint_mse = [r for r in positive if float(r["delta_known_mse"]) <= 0]
    result = dict(status="complete", source_sha256=sha(Path(__file__)),
                  inputs={str(p.relative_to(ROOT)): sha(p) for p in (
                      RATE / "complete.json", RATE / "per_aoi.csv", LINK / "complete.json",
                      LINK / "per_aoi_seed.csv", UNKNOWN / "complete.json", UNKNOWN / "comparisons.csv")},
                  rate=rate_stats, link=link_stats,
                  unknown=dict(comparisons_against_feature_change=len(real), coverage_intervals_above_zero=len(positive),
                               coverage_positive_and_nonincreased_known_MSE=len(joint_mse), positive_rows=positive,
                               scope="development scorer/router comparisons; MSE is a fidelity proxy, not known-task F1"),
                  interval_scope="paired AOIs with seeds nested; rate95%; link98.75% descriptive intervals per contrast/metric across four profiles; no correction across all policy/metric/scorer families")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    arq = [r for r in link_stats if r["comparator"] == "delta_arq"]
    report = ["# 2026-11-04 补充实验结果", "",
              "本轮完成282个月的冻结特征缓存、270次同解码码率比较、19,440次链路传输机会和7,680条Unknown修正观测。所有正式表格从完整运行汇总。", "",
              "## 历史状态的同效用成本", "",
              f"八地区Delta相对全量特征的平均更新字节节省为{100*rate_stats['relative_update_saving']['mean']:.2f}%，95%地区重采样区间为[{100*rate_stats['relative_update_saving']['low']:.2f}%, {100*rate_stats['relative_update_saving']['high']:.2f}%]。含初始化的平均节省为{100*rate_stats['relative_complete_sequence_saving']['mean']:.2f}%。两路解码特征逐元素相同。", "",
              "该比较证明同一量化表征的历史条件编码收益。新增SafeSync路由相对正确Delta的增量需要分别评价。", "",
              "## Anchor与正确重传", "",
              "| 仿真链路 | 风险Anchor相对ARQ的世界特征MSE降低 | 更新字节变化 | 最新观测可用率，风险 / ARQ |", "|---|---:|---:|---:|"]
    for row in arq:
        report.append(f"| {row['profile']} | {100*row['relative_world_mse_reduction']:.2f}% | {100*row['relative_update_byte_change']:+.2f}% | {row['new_current_fraction']:.3f} / {row['old_current_fraction']:.3f} |")
    report.extend(["", "完整4/8/16步周期对照与配对区间保存在analysis.json。反馈延迟以观测机会为单位，结果来自真实特征上的受控链路仿真。它们量化恢复成本和新鲜度之间的取舍。", "",
                   "## Unknown评分与路由", "",
                   f"非oracle方法相对强feature-change共有{len(real)}个类别/预算/方法比较，覆盖差值区间全正的有{len(positive)}个，其中known MSE点值未增加的有{len(joint_mse)}个。", "",
                   "完整summary.csv、comparisons.csv保留全部新方法与GT诊断。新近邻评分的图库只从known校准标签建立。开发数据上的阳性点需要独立验证，known MSE结果需要与接收端任务效用分别解释。", "",
                   "## 原目标状态", "",
                   "历史复用得到新增同解码证据。风险Anchor补齐了相对正确重传与三个周期的实际链路仿真比较。Unknown、跨场景稳健性及强基线联合优势继续按各自结果验收。当前技术进展不能直接改写成原始科研总验收通过。", ""])
    (ROOT / "docs/RESEARCH_RECOVERY_RESULTS.md").write_text("\n".join(report), "utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
                         "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "text.color": "#000000",
                         "axes.labelcolor": "#000000", "axes.edgecolor": "#1F2937",
                         "xtick.color": "#000000", "ytick.color": "#000000",
                         "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.50), constrained_layout=True)
    axes[0].bar(np.arange(1, 9), [100*float(r["relative_update_saving"]) for r in rate],
                color="#1B4965", edgecolor="#1F2937", linewidth=0.45, alpha=0.92)
    axes[0].axhline(20, color="#64748B", ls="--", lw=.65)
    axes[0].set(xlabel="Reserved area", ylabel="Update-byte saving (%)")
    axes[0].set_title("(a) Identical Received Features", fontsize=8.6, fontweight="bold", loc="center")
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.38, color="#CBD5E1")
    for label, color, ls, marker in (("delta_arq", "#64748B", "--", "o"), ("periodic_4", "#8C6D46", "-.", "s"), ("risk_anchor", "#1B4965", "-", "^")):
        values = []
        for profile in ("iid10", "iid30", "burst"):
            selected = [v for k, v in per_area.items() if k[:2] == (profile, label)]
            values.append(float(np.mean([v["mean_world_mse"] for v in selected])) * 1e4)
        axes[1].plot(range(3), values, marker=marker, markersize=3.5, lw=1.10 if label == "risk_anchor" else 0.88,
                     linestyle=ls, color=color, label=label.replace("_", " "))
    axes[1].set(xticks=range(3), xticklabels=["IID 10%", "IID 30%", "Burst"], ylabel="World feature MSE ($\\times 10^{-4}$)")
    axes[1].set_title("(b) Actual Encoded Link Simulation", fontsize=8.6, fontweight="bold", loc="center")
    axes[1].legend(frameon=False, fontsize=7.2)
    axes[1].grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.16, lw=0.5)
        ax.tick_params(labelsize=7.8)
    figure = OUT / "recovery_summary"
    fig.savefig(figure.with_suffix(".png"), dpi=200)
    fig.savefig(figure.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps(dict(status="complete", rate=result["rate"], unknown=result["unknown"])), flush=True)


if __name__ == "__main__":
    main()
