"""Build exp0200 paper tables/figures from immutable actual-receiver results.

CPU-only artifact production. No models, source images or frozen evaluators are
imported. All methods, all three budgets and the appended controls are retained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
MAIN = ROOT / "results/layered_tasks/confirmation_v1"
FULL = ROOT / "results/full_cache_tasks/gate30"
LIC = ROOT / "results/learned_tasks/confirmation_q1_8_v1"
QUALITY = ROOT / "results/layered_quality/confirmation_v1"
LQUALITY = ROOT / "results/learned_quality/confirmation_v1"
CAPS = [32768, 65536, 131072]
ORDER = ["gaussian_20261029", "dct_bound", "full_cache_gate25", "random_1729", "full_jpeg420",
         "full_jpeg2000", "full_webp", "full_cache_gate30", "learned_hyperprior_q1_8"]
NAMES = {"gaussian_20261029": "Gaussian", "dct_bound": "DCT", "full_cache_gate25": "Full history (25 dB)",
    "random_1729": "Random", "full_jpeg420": "JPEG 4:2:0", "full_jpeg2000": "JPEG2000", "full_webp": "WebP",
    "full_cache_gate30": "Full history (30 dB)", "learned_hyperprior_q1_8": "Hyperprior"}
ADDED = {"full_cache_gate30", "learned_hyperprior_q1_8"}
PLOT_METHODS = ["gaussian_20261029", "dct_bound", "random_1729", "full_cache_gate30", "full_webp", "learned_hyperprior_q1_8"]
STYLE = {
    "gaussian_20261029": ("#0072B2", "o", "-"),
    "dct_bound": ("#009E73", "s", (0, (4, 2))),
    "random_1729": ("#E69F00", "^", (0, (1, 2))),
    "full_cache_gate30": ("#CC79A7", "D", "-."),
    "full_webp": ("#56B4E9", "v", (0, (6, 2, 1, 2))),
    "learned_hyperprior_q1_8": ("#D55E00", "P", "-")}
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    INPUTS[rel(path)] = sha(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    INPUTS[rel(path)] = sha(path)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, name):
    if a is None or b is None:
        if a is not b:
            raise ValueError("missingness differs: " + name)
    elif not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-11):
        raise ValueError(f"different streams/aggregation for {name}: {a} != {b}")


def original_label(method):
    return "full_cache_gate25" if method == "full_cache_mse" else method


def build_data():
    manifest = read_json(ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json")
    ids = [a["aoi_id"] for a in manifest["aois"]]
    if len(ids) != 8:
        raise ValueError("fixed eight-AOI cohort changed")
    main, full, lic = [read_json(folder / "summary.json") for folder in [MAIN, FULL, LIC]]
    for folder, expected in [(MAIN, 3738), (FULL, 534), (LIC, 534)]:
        status = read_json(folder / "progress.json")
        if status["status"] != "complete" or status.get("actual_outputs", status.get("additional_outputs")) != expected:
            raise ValueError("task source incomplete: " + rel(folder))
        if status["summary_sha256"] != sha(folder / "summary.json"):
            raise ValueError("task completion summary hash differs")
    if not (main["model_hash"] == full["model_hash"] == lic["model_hash"] and
            main["threshold_hash"] == full["threshold_hash"] == lic["threshold_hash"]):
        raise ValueError("task teacher/threshold identities differ")
    qstatus = read_json(QUALITY / "status.json")
    lstatus = read_json(LQUALITY / "status.json")
    if qstatus["status"] != "complete" or qstatus["complete_AOIs"] != 8:
        raise ValueError("primary image-quality results are not complete")
    if lstatus["status"] != "complete" or lstatus["records"] != 534 or lstatus["AOIs"] != 8:
        raise ValueError("LIC masked image-quality results are not complete; sender MSE is not an alternative")
    primary_task = read_csv(MAIN / "task_table.csv")
    full_task = read_csv(FULL / "comparison.csv")
    learned_task = read_csv(LIC / "comparison.csv")
    qmacro = read_csv(QUALITY / "macro_single_seed.csv")
    qarea = read_csv(QUALITY / "per_seed_aoi.csv")
    lmacro, larea = read_csv(LQUALITY / "macro.csv"), read_csv(LQUALITY / "per_aoi.csv")
    # Keep the exact preselected methods; never replace them by 3-seed means.
    quality = {(original_label(r["method"]), int(r["budget"]), r["aoi"]): r for r in qarea
        if r["method"] in ["gaussian_20261029", "dct_bound", "full_cache_mse", "random_1729", "full_jpeg420", "full_jpeg2000", "full_webp"]}
    task = {}
    for g in main["per_aoi_seed"]:
        task[(original_label(g["method"]), g["budget"], g["aoi"])] = g
    for g in full["per_aoi"]:
        if g["label"] == "full_cache_gate30":
            task[(g["label"], g["budget"], g["aoi"])] = g
    for g in lic["per_aoi"]:
        if g["label"] == "learned_hyperprior_q1_8":
            task[(g["label"], g["budget"], g["aoi"])] = g
    for r in larea:
        quality[("learned_hyperprior_q1_8", int(r["budget"]), r["aoi"])] = r
    # Gate30 has no independent global-PSNR macro yet. Derive it using the exact
    # counted stream errors, pooling valid pixels within AOI before taking dB.
    for index, aid in enumerate(ids):
        base = ROOT / f"results/full_cache_controls/gate30/area_{index}"
        status, spec = read_json(base / "progress.json"), read_json(base / "run_spec.json")
        if status["status"] != "complete" or spec["control_gate_psnr"] != 30 or status["records_sha256"] != sha(base / "records.csv"):
            raise ValueError("additional gate30 image stream is incomplete/changed")
        raw = read_csv(base / "records.csv")
        for cap in CAPS:
            rows = [r for r in raw if r["aoi"] == aid and int(r["budget"]) == cap]
            pixels = sum(int(r["all_psnr30_pixels"]) for r in rows)
            error = sum(float(r["all_psnr30_error_sum"]) for r in rows)
            event = sum(int(r["new_track_psnr30_pixels"]) for r in rows)
            usable = sum(int(r["new_track_psnr30_usable_pixels"]) for r in rows)
            down = sum(int(r["down_bytes"]) for r in rows)
            ack = sum(int(r["up_bytes"]) for r in rows)
            mse = error / pixels if pixels else None
            quality[("full_cache_gate30", cap, aid)] = {"new_track_psnr30_pixels": event,
                "new_track_psnr30_usable_pixels": usable, "new_track_psnr30_coverage": usable/event if event else None,
                "valid_psnr_db": -10*math.log10(mse) if mse and mse > 0 else None,
                "update_total_bytes_mean": (down+ack)/len(rows), "bootstrap_total_bytes":
                    int(rows[0]["bootstrap_down_bytes"]) + int(rows[0]["bootstrap_up_bytes"])}
    per_aoi = []
    for method in ORDER:
        for cap in CAPS:
            for aid in ids:
                key = method, cap, aid
                g, q = task[key], quality[key]
                if g["frames"] != len(next(a for a in manifest["aois"] if a["aoi_id"] == aid)["months"]) - 1:
                    raise ValueError("task omitted real observations")
                close(g["mean_update_bidirectional_bytes"], q["update_total_bytes_mean"], str(key)+" actual bytes")
                close(g["event"]["positive_pixels"], q["new_track_psnr30_pixels"], str(key)+" event GT")
                close(g["bootstrap_bidirectional_bytes"], q["bootstrap_total_bytes"], str(key)+" bootstrap")
                # LIC quality and ground must reference the exact same committed
                # fresh receipt, not merely the same method name or nominal cap.
                if method == "learned_hyperprior_q1_8":
                    index = ids.index(aid)
                    ground_spec = read_json(LIC / f"area_{index}/run_spec.json")
                    receipt_path = f"results/learned_codec/confirmation_q1_8_v1/fresh_replay/aoi_receipts/{aid}.json"
                    if ground_spec["source_hashes"][receipt_path] != lstatus["input_receipts"][aid]["receipt_sha256"]:
                        raise ValueError("LIC task and quality use different decoded streams")
                row = {"method": method, "cap_bytes": cap, "aoi": aid, "updates": g["frames"],
                    "additional_control": method in ADDED, "update_down_bytes": g["down_bytes"], "update_ack_bytes": g["ack_bytes"],
                    "bootstrap_down_bytes": g["bootstrap_down_bytes"], "bootstrap_ack_bytes": g["bootstrap_ack_bytes"],
                    "actual_update_bytes_mean": g["mean_update_bidirectional_bytes"],
                    "amortized_bytes_per_update": g["total_bytes_including_bootstrap"]/g["frames"],
                    "building_f1": g["building"]["f1"], "event_f1": g["event"]["f1"],
                    "task_valid_pixels": g["event"]["valid_pixels"], "event_GT_positive_pixels": g["event"]["positive_pixels"],
                    "event_U30_usable_pixels": int(q["new_track_psnr30_usable_pixels"]),
                    "event_U30": float(q["new_track_psnr30_coverage"]), "global_PSNR_dB": float(q["valid_psnr_db"])}
                per_aoi.append(row)
    table = []
    for cap in CAPS:
        for method in ORDER:
            rows = [r for r in per_aoi if r["method"] == method and r["cap_bytes"] == cap]
            if len(rows) != 8 or any(r["task_valid_pixels"] <= 0 or r["event_GT_positive_pixels"] <= 0 for r in rows):
                raise ValueError("shared eight-AOI GT support changed")
            row = {"cap_bytes": cap, "cap_KiB": cap//1024, "method": method, "display_name": NAMES[method],
                "additional_control": method in ADDED, "AOIs": 8}
            for field in ["actual_update_bytes_mean", "amortized_bytes_per_update", "building_f1", "event_f1", "event_U30", "global_PSNR_dB"]:
                row[field] = float(np.mean([r[field] for r in rows]))
            row["actual_update_KiB_mean"] = row["actual_update_bytes_mean"]/1024
            row["amortized_KiB_per_update"] = row["amortized_bytes_per_update"]/1024
            row["event_U30_percent"] = 100*row["event_U30"]
            table.append(row)
    # Join checks against each existing public result table, not recomputed task
    # predictions. This catches seed mixing and cost/GT mismatches cheaply.
    lookup = {(r["method"], r["cap_bytes"]): r for r in table}
    families = {"gaussian": "gaussian_20261029", "random": "random_1729", "full_cache_mse": "full_cache_gate25"}
    for r in primary_task:
        label = families.get(r["family"], r["family"])
        target = lookup[(label, int(r["packet_cap_bytes"]))]
        for a, b in [("building_f1", "building_f1"), ("event_f1", "event_f1"), ("actual_mean_update_bidirectional_bytes", "actual_update_bytes_mean")]:
            close(r[a], target[b], "primary task macro "+a)
    for r in full_task:
        if r["method"] == "full_cache_gate30":
            target = lookup[(r["method"], int(r["budget"]))]
            for a, b in [("building_f1", "building_f1"), ("event_f1", "event_f1"), ("new_track_U30", "event_U30")]:
                close(r[a], target[b], "gate30 macro "+a)
    for r in learned_task:
        if r["method"] == "learned_hyperprior_q1_8":
            target = lookup[(r["method"], int(r["cap"]))]
            close(r["building_f1"], target["building_f1"], "LIC task macro")
            close(r["event_f1"], target["event_f1"], "LIC event macro")
    for rows, mapper in [(qmacro, original_label), (lmacro, lambda _: "learned_hyperprior_q1_8")]:
        for r in rows:
            label = mapper(r["method"])
            if (label, int(r["budget"])) in lookup:
                target = lookup[(label, int(r["budget"]))]
                close(r["valid_psnr_db"], target["global_PSNR_dB"], "quality macro PSNR")
                close(r["new_track_psnr30_coverage"], target["event_U30"], "quality macro U30")
    intervals = []
    for r in main["paired_contrasts"]:
        if r["comparator"] == "full_webp":
            ci = r["simultaneous_three_budget_CI98_333_percent"]
            intervals.append({"comparator": "full_webp", "comparison_scope": "original primary", "cap_bytes": r["budget"],
                "task": r["task"], "difference": r["macro_difference"], "lower": ci["lower"], "upper": ci["upper"],
                "paired_AOIs": r["paired_AOIs"], "paired_AOI_ids": r["paired_AOI_ids"]})
    for r in lic["paired_comparisons"]:
        if r["comparator"] == "gaussian_20261029":
            intervals.append({"comparator": "learned_hyperprior_q1_8", "comparison_scope": "post-confirmation secondary",
                "cap_bytes": r["cap"], "task": r["task"], "difference": -r["macro_F1_difference"],
                "lower": -r["CI98_333_upper"], "upper": -r["CI98_333_lower"],
                "paired_AOIs": r["paired_AOIs"], "paired_AOI_ids": r["paired_AOI_ids"]})
    for r in intervals:
        g, b = lookup[("gaussian_20261029", r["cap_bytes"])], lookup[(r["comparator"], r["cap_bytes"])]
        close(r["difference"], g[r["task"]+"_f1"]-b[r["task"]+"_f1"], "paired forest orientation")
        if r["paired_AOIs"] != 8 or r["paired_AOI_ids"] != ids:
            raise ValueError("paired support is not the same eight AOIs")
    return table, per_aoi, intervals, main


def tex_table(table):
    paradigms = {
        "gaussian_20261029": "Asymmetric State (Ours)",
        "dct_bound": "Deterministic State",
        "random_1729": "Random Priority Ctrl.",
        "full_cache_gate25": "Full-Archive Ref.",
        "full_cache_gate30": "Full-Archive Ref.",
        "full_jpeg420": "Stateless Transform",
        "full_jpeg2000": "Stateless Wavelet",
        "full_webp": "Stateless Transform",
        "learned_hyperprior_q1_8": "Learned Neural Codec",
    }
    disp_override = {
        "gaussian_20261029": r"\textbf{SafeSync-ZFSS (Gaussian)}",
    }
    lines = [
        r"% Generated by scripts/build_task_paper_assets.py; do not hand-edit numbers.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-temporal building segmentation, change-event detection, and localized evidence coverage across eight confirmation AOIs (178 updates) under the fixed EfficientNet-B3 U-Net (\texttt{exp0200}) evaluator. Best and second-best non-oracle methods per downlink cap are \textbf{bolded} and \underline{underlined}.}",
        r"\label{tab:task-exp0200}",
        r"\begingroup",
        r"\fontsize{8}{10.2}\selectfont",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cllrrrrr@{}}",
        r"\toprule",
        r"Cap (KiB) & Method & Paradigm & Update (KiB) $\downarrow$ & Building F1 $\uparrow$ & Event F1 $\uparrow$ & $U_{30}$ (\%) $\uparrow$ & PSNR (dB) $\uparrow$ \\",
        r"\midrule",
    ]
    for cap in CAPS:
        if cap != CAPS[0]:
            lines.append(r"\cmidrule(lr){1-8}")
        cap_rows = [r for r in table if r["cap_bytes"] == cap]
        # Rank practical methods (excluding full-archive upper-reference controls) for bold/underline
        practical = [r for r in cap_rows if not r["method"].startswith("full_cache")]
        def fmt_metric(r, field, fmt_str):
            val = r[field]
            s = format(val, fmt_str)
            if r["method"].startswith("full_cache"):
                return s
            vals = sorted({x[field] for x in practical}, reverse=True)
            if len(vals) >= 1 and math.isclose(val, vals[0], abs_tol=1e-12):
                return rf"\textbf{{{s}}}"
            if len(vals) >= 2 and math.isclose(val, vals[1], abs_tol=1e-12):
                return rf"\underline{{{s}}}"
            return s

        for i, r in enumerate(cap_rows):
            base_name = disp_override.get(r["method"], r["display_name"])
            name = base_name + (r"$^{\dagger}$" if r["additional_control"] else "")
            cap_text = rf"\textbf{{{cap//1024}}}" if i == 0 else ""
            row_prefix = r"\rowcolor{safesyncbg} " if r["method"] == "gaussian_20261029" else ""
            b_str = fmt_metric(r, "building_f1", ".4f")
            e_str = fmt_metric(r, "event_f1", ".4f")
            u_str = fmt_metric(r, "event_U30_percent", ".2f")
            p_str = fmt_metric(r, "global_PSNR_dB", ".2f")
            lines.append(
                f"{row_prefix}{cap_text} & {name} & {paradigms[r['method']]} & "
                f"{r['actual_update_KiB_mean']:.2f} & {b_str} & {e_str} & {u_str} & {p_str} " + r"\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\vspace{3pt}",
        r"\begin{minipage}{\textwidth}",
        r"\fontsize{7.5}{9.2}\selectfont",
        r"\emph{Notes.} Update reports mean actual bidirectional payload per pass (downlink packet plus 60-byte ACK); bootstrap initialization is accounted separately. $^{\dagger}$ denotes strong reference controls appended with identical frozen configurations. Gaussian and random routing use fixed seeds 20261029 and 1729. Task F1 and new-track coverage $U_{30}$ are evaluated over the intersection of current and bootstrap valid masks, whereas global PSNR pools valid pixel squared errors per AOI prior to equal-weight macro averaging across the eight geographic areas.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ]
    (GEN / "task_table.tex").write_text("\n".join(lines), encoding="utf-8")


def setup_style():
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        family = "Times New Roman"
    except ValueError:
        family = "DejaVu Serif"
    plt.rcParams.update({"font.family": "serif", "font.serif": [family], "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "safesync-task-assets-v1", "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white"})
    return family


def export(fig, stem):
    fig.savefig(FIG / (stem+".pdf"), metadata={"Creator": "build_task_paper_assets.py", "CreationDate": None, "ModDate": None})
    fig.savefig(FIG / (stem+".svg"), metadata={"Creator": "build_task_paper_assets.py", "Date": None})
    fig.savefig(FIG / (stem+".png"), dpi=300)
    plt.close(fig)


def plot_tradeoff(table):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    fig.subplots_adjust(left=.075, right=.99, bottom=.18, top=.76, wspace=.26)
    for axis, metric, title in [(axes[0], "building_f1", "(a) Building segmentation"), (axes[1], "event_f1", "(b) New-track events")]:
        for method in PLOT_METHODS:
            rows = sorted([r for r in table if r["method"] == method], key=lambda r: r["cap_bytes"])
            color, marker, line = STYLE[method]
            name = NAMES[method] + ("†" if method in ADDED else "")
            axis.plot([r["actual_update_KiB_mean"] for r in rows], [r[metric] for r in rows],
                label=name, color=color, linestyle=line, marker=marker, markersize=4,
                markerfacecolor="white" if method not in ["gaussian_20261029", "learned_hyperprior_q1_8"] else color,
                markeredgewidth=.85, linewidth=1.7 if method == "gaussian_20261029" else 1.25, zorder=3)
        axis.set_title(title, loc="left", pad=5)
        axis.set_xlabel("Actual update downlink + ACK (KiB)")
        axis.set_ylabel("F1")
        axis.set_xlim(20, 130)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylim(.32, .63)
    axes[1].set_ylim(.12, .415)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=3,
               frameon=False, handlelength=2.7, columnspacing=1.55, labelspacing=.55)
    fig.text(.53, .025, "All three fixed downlink caps (32, 64, 128 KiB); segments connect measured points only.",
             ha="center", fontsize=7.7)
    export(fig, "task_tradeoff")


def plot_uncertainty(intervals):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.17, top=.87, wspace=.23, hspace=.42)
    for col, comparator in enumerate(["full_webp", "learned_hyperprior_q1_8"]):
        color = STYLE[comparator][0]
        for row, task in enumerate(["building", "event"]):
            axis = axes[row, col]
            entries = sorted([r for r in intervals if (r["comparator"], r["task"]) == (comparator, task)], key=lambda r: r["cap_bytes"])
            for y, value in zip([2, 1, 0], entries):
                axis.errorbar(value["difference"], y,
                    xerr=[[value["difference"]-value["lower"]], [value["upper"]-value["difference"]]],
                    color=color, marker="o" if col == 0 else "s", markersize=4,
                    elinewidth=1.2, capsize=3, capthick=.9, linestyle="none", zorder=3)
            axis.axvline(0, color="#333333", linewidth=.8, linestyle="--", zorder=1)
            axis.set_yticks([2, 1, 0], ["32", "64", "128"])
            axis.set_ylim(-.55, 2.55)
            axis.set_xlim(-.145, .25)
            axis.set_xticks([-.1, 0, .1, .2])
            axis.grid(axis="y", visible=False)
            if col == 0:
                axis.set_ylabel(("Building F1" if row == 0 else "New-track F1") + "\ncap (KiB)")
            if row == 0:
                axis.set_title("Primary: vs WebP" if col == 0 else "Secondary: vs Hyperprior†", loc="left", pad=8)
            if row == 1:
                axis.set_xlabel("Gaussian − comparator (F1)")
    fig.text(.53, .035, "98.333% paired-AOI bootstrap CIs: three-cap family, separately per contrast and outcome.",
             ha="center", fontsize=7.7)
    export(fig, "task_uncertainty")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing bound input/output hashes only")
    args = parser.parse_args()
    if args.check:
        receipt = json.loads((GEN / "task_assets_manifest.json").read_text(encoding="utf-8"))
        if sha(Path(__file__)) != receipt["builder_sha256"]:
            raise ValueError("the task asset builder source changed")
        for name, expected in {**receipt["input_sha256"], **receipt["output_sha256"]}.items():
            if sha(ROOT / name) != expected:
                raise ValueError("asset binding changed: " + name)
        print("all task figure/table input and output hashes match")
        return
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table, per_aoi, intervals, original = build_data()
    save_csv(GEN / "task_table.csv", table)
    save_csv(GEN / "task_per_aoi.csv", per_aoi)
    tex_table(table)
    source_hash = sha(Path(__file__))
    common = {"builder_sha256": source_hash, "input_sha256": INPUTS, "task_model_hash": original["model_hash"],
        "task_threshold_hash": original["threshold_hash"], "AOI_count": 8, "OOF_results_included": False,
        "task_scope": "exp0200 fixed communication-distortion evaluator; all8 expected training-fold overlap",
        "seed_scope": "Gaussian20261029/random1729 only; no three-seed quality mean substituted for task metrics"}
    save_json(GEN / "task_tradeoff_data.json", {**common, "points": table, "plotted_methods": PLOT_METHODS,
        "x_unit": "AOI-mean actual downlink plus ACK KiB per update; bootstrap excluded and separately reported",
        "line_scope": "measured points connected for readability; no fitted or universal Pareto frontier"})
    save_json(GEN / "task_uncertainty_data.json", {**common, "intervals": intervals,
        "difference_orientation": "Gaussian minus comparator", "nominal_interval_percent": 98.33333333333333,
        "bootstrap_draws": 10000, "bootstrap_seed": 20261029,
        "multiplicity_scope": "three caps per specified contrast/outcome, not every contrast/outcome jointly"})
    font = setup_style()
    plot_tradeoff(table)
    plot_uncertainty(intervals)
    claims = {**common, "main_table_rows": 27, "per_aoi_rows": 216, "all_caps_retained_bytes": CAPS,
        "additional_controls": {method: "post-initial-confirmation, not original blind preregistration" for method in sorted(ADDED)},
        "aggregation": "pool task counts within AOI; quality pixel-MSE pooled within AOI before dB; equal AOI macro",
        "PSNR_mask": "current-valid pixels over the full image", "task_and_U30_mask": "current and initial valid intersection; official new-track GT for U30",
        "rates_not_exactly_matched": True, "mean_metrics": table, "paired_task_differences": intervals,
        "training_overlap": original["training_overlap"], "known_task_internal_noninferiority_margin": -.02,
        "publication_acceptance_claim": False, "new_inference_or_training_performed": False,
        "plot_font": font, "table_font_points": 8, "table_note_font_points": 7.5,
        "figure_body_font_points_minimum": 7.7, "render_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}}
    save_json(GEN / "numeric_claims.json", claims)
    captions = {
        "task_tradeoff": "Actual update cost and decoded receiver utility under the fixed exp0200 evaluator. Each method retains the three preselected caps; segments only connect measured points. All methods are tabulated in Table~\\ref{tab:task-exp0200}. Daggers mark controls appended after initial confirmation. Costs include update downlink and ACK; bootstrap amortization is separate. Task-model training-region overlap is disclosed in the table note.",
        "task_uncertainty": "Paired AOI F1 differences, Gaussian minus the original primary WebP comparator and the appended secondary Hyperprior comparator. Bars are nominal 98.333\\% percentile intervals from 10,000 equal-weight AOI resamples, adjusting three caps separately for each contrast/outcome; they do not provide simultaneous coverage across all contrasts and outcomes. All eight AOIs share the same GT support."}
    save_json(GEN / "task_figure_captions.json", captions)
    include = []
    for stem in ["task_tradeoff", "task_uncertainty"]:
        include.extend([r"\begin{figure*}[t]", r"\centering", f"\\includegraphics[width=\\textwidth]{{figures/{stem}.pdf}}",
            "\\caption{"+captions[stem]+"}", "\\label{fig:"+stem.replace("_", "-")+"}", r"\end{figure*}", ""])
    (GEN / "task_figures.tex").write_text("\n".join(include), encoding="utf-8")
    outputs = [GEN / name for name in ["task_table.csv", "task_per_aoi.csv", "task_table.tex", "numeric_claims.json",
        "task_tradeoff_data.json", "task_uncertainty_data.json", "task_figure_captions.json", "task_figures.tex"]]
    outputs += [FIG / (stem+"."+extension) for stem in ["task_tradeoff", "task_uncertainty"] for extension in ["pdf", "svg", "png"]]
    visual_path = GEN / "task_visual_check.json"
    visually_checked = False
    if visual_path.exists():
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
        visually_checked = visual.get("status") == "passed" and len(visual.get("visually_checked_output_sha256", {})) == 3 and all(sha(ROOT / name) == expected
            for name, expected in visual.get("visually_checked_output_sha256", {}).items())
    save_json(GEN / "task_assets_manifest.json", {"status": "complete" if visually_checked else "generated_pending_visual_check", "builder_sha256": source_hash,
        "input_sha256": INPUTS, "output_sha256": {rel(path): sha(path) for path in outputs},
        "data_alignment_checks": "all selected task/quality actual bytes, bootstrap, event GT support, macro values and forest signs match",
        "main_table_rows": len(table), "per_aoi_rows": len(per_aoi), "font": font,
        "visual_check_receipt": rel(visual_path) if visually_checked else None})
    print("built27 measured table rows,216 AOI rows, both vector figures and hash-bound claims; visual inspection required")


if __name__ == "__main__":
    main()
