"""Bind final author-OOF and information controls to manuscript tables.

Only complete saved results are read. No model, codec, prediction, or tuning.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/generated"
SUP = ROOT / "paper/supplement/generated"
OOF = "resources/pretrained_ground_crossfit/receipts_v1"
CTRL = "resources/pretrained_ground_crossfit/lossless_controls_v1"
ORDER = ["gaussian_20261029", "dct_bound", "full_cache_gate25", "random_1729",
         "full_jpeg420", "full_jpeg2000", "full_webp", "full_cache_gate30",
         "learned_hyperprior_q1_8"]
LABELS = dict(zip(ORDER, ["Gaussian", "DCT", "Full history (25 dB)", "Random",
                       "JPEG 4:2:0", "JPEG2000", "WebP", "Full history (30 dB)",
                       "Hyperprior"]))
CAPS = [32768, 65536, 131072]
INPUTS, OUTPUTS = {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(relative):
    path = ROOT / relative
    INPUTS[relative] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(relative):
    path = ROOT / relative
    INPUTS[relative] = sha(path)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize(method):
    return "full_cache_gate25" if method == "full_cache_mse" else method


def close(a, b, reason):
    if not math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-10):
        raise ValueError(reason)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    OUTPUTS[path.relative_to(ROOT).as_posix()] = sha(path)


def table(path, caption, label, headers, rows, note, wide=True):
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [r"\begin{" + env + "}[t]", r"\centering\footnotesize",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\setlength{\tabcolsep}{0pt}",
             r"\renewcommand{\arraystretch}{1.00}",
             r"\begin{tabular*}{" + width + r"}{@{\extracolsep{\fill}}l" + "r"*(len(headers)-1) + r"@{}}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\par\smallskip",
              r"\begin{minipage}{" + width + r"}\footnotesize " + note + r"\end{minipage}",
              r"\end{" + env + "}", ""]
    save(path, "\n".join(lines))


def main():
    done = load(OOF + "/complete.json")
    summary = load(OOF + "/summary.json")
    completed_areas = summary["completed_areas"]
    if (summary["status"] != "complete" or summary["outputs"] != 4806
            or len(completed_areas) != 8 or set(completed_areas) != set(range(8))):
        raise ValueError("OOF cohort must be complete")
    if done.get("summary_sha256") is not None and done["summary_sha256"] != sha(ROOT / OOF / "summary.json"):
        raise ValueError("OOF completion marker does not bind summary")
    source = csv_rows("paper/generated/task_per_aoi.csv")
    original = {(r["method"], int(r["cap_bytes"]), r["aoi"]): r for r in source}
    areas = summary["per_aoi_seed"]
    if len(areas) != 216:
        raise ValueError("Expected 9 methods x 3 caps x 8 AOIs")
    for row in areas:
        old = original[(normalize(row["method"]), row["budget"], row["aoi"])]
        for field, old_field in [
            ("mean_update_bidirectional_bytes", "actual_update_bytes_mean"),
            ("down_bytes", "update_down_bytes"), ("ack_bytes", "update_ack_bytes"),
            ("bootstrap_down_bytes", "bootstrap_down_bytes"), ("bootstrap_ack_bytes", "bootstrap_ack_bytes"),
        ]:
            close(row[field], old[old_field], "OOF input transaction costs differ")
        close(row["event"]["valid_pixels"], old["task_valid_pixels"], "OOF evaluation support differs")
        close(row["event"]["positive_pixels"], old["event_GT_positive_pixels"], "OOF GT positives differ")
    macro = {(normalize(r["method"]), r["budget"]): r for r in summary["macro_aoi"]}
    if set(macro) != {(m, c) for m in ORDER for c in CAPS}:
        raise ValueError("OOF method or cap cohort changed")
    # Wide full table keeps every original and appended alternative.
    rows = []
    for cap in CAPS:
        for method in ORDER:
            r = macro[(method, cap)]
            if r["AOIs"] != 8:
                raise ValueError("Missing geographic support")
            rows.append([LABELS[method], cap//1024,
                         f"{r['mean_update_bidirectional_bytes']/1024:.2f}",
                         f"{r['building']['f1']:.4f}", f"{r['event']['f1']:.4f}"])
    table(SUP/"oof_full.tex", "Complete author-OOF task sensitivity analysis on unchanged decoded receipts.",
          "tab:oof-all", ["Method", "Cap KiB", "Actual KiB/update", "Building F1", "Event F1"],
          rows, "Equal means of eight AOIs after pooling monthly confusion counts. "
          "The per-AOI checkpoint follows the author's published validation fold; these are validation-best "
          "weights. Building threshold 0.5 and event threshold 0.941 are fixed independently on the original "
          "calibration areas. This is an appended sensitivity analysis, not a new blind confirmation. "
          "Communication and RGB quality are unchanged from the original receipt experiment.")
    selected = ["gaussian_20261029", "dct_bound", "random_1729", "full_cache_gate30",
                "full_webp", "learned_hyperprior_q1_8"]
    # Six policies, with both tasks and all three caps, in a full-width one-column table.
    lines = [
        r"\begin{table}[t]", r"\centering\footnotesize",
        r"\caption{Cross-validated 4-fold Author-OOF utility across 32, 64, and 128-KiB caps on unchanged decoded images.}",
        r"\label{tab:oof}", r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}", r"\toprule",
        r"& \multicolumn{3}{c}{Building F1 $\uparrow$} & \multicolumn{3}{c}{Event F1 $\uparrow$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & 32 & 64 & 128 & 32 & 64 & 128 \\", r"\midrule",
    ]
    practical_oof = [m for m in selected if not m.startswith("full_cache")]
    col_ranks = {}
    for col_idx, (task, cap) in enumerate((t, c) for t in ("building", "event") for c in CAPS):
        col_ranks[col_idx] = sorted({round(macro[(m, cap)][task]["f1"], 3) for m in practical_oof}, reverse=True)

    for method in selected:
        if method == "gaussian_20261029":
            name = r"\textbf{SafeSync-ZFSS}"
            prefix = r"\rowcolor{safesyncbg} "
        elif method == "full_cache_gate30":
            name = "Full history (30)"
            prefix = ""
        else:
            name = LABELS[method]
            prefix = ""
        vals = [macro[(method, cap)][task]["f1"] for task in ("building", "event") for cap in CAPS]
        cells_fmt = []
        for col_idx, v in enumerate(vals):
            rv = round(v, 3)
            s = f"{v:.3f}"
            if not method.startswith("full_cache"):
                ranks = col_ranks[col_idx]
                if len(ranks) >= 1 and math.isclose(rv, ranks[0], abs_tol=1e-9):
                    s = rf"\textbf{{{s}}}"
                elif len(ranks) >= 2 and math.isclose(rv, ranks[1], abs_tol=1e-9):
                    s = rf"\underline{{{s}}}"
            cells_fmt.append(s)
        lines.append(prefix + name + " & " + " & ".join(cells_fmt) + r" \\")
    lines += [
        r"\bottomrule\end{tabular*}", r"\par\smallskip",
        r"\begin{minipage}{\columnwidth}\footnotesize Caps denote per-pass downlink limits (KiB); realized bidirectional byte counts match Table~\ref{tab:task-exp0200}. Each AOI is evaluated with its corresponding out-of-fold checkpoint from the published four-fold split. Complete nine-policy results and paired intervals appear in the Supplement.\end{minipage}",
        r"\end{table}", "",
    ]
    save(MAIN/"oof_table.tex", "\n".join(lines))
    pairs = csv_rows(OOF + "/paired_comparisons.csv")
    if len(pairs) != 12:
        raise ValueError("Expected two contrasts x two tasks x three budgets")
    ci_rows = []
    for p in pairs:
        close(p["paired_AOIs"], 8, "Paired CI lacks eight AOIs")
        ci_rows.append([f"{LABELS[normalize(p['comparator'])]} / {p['task']}",
                        int(p["budget"])//1024, f"{float(p['difference_f1']):+.4f}",
                        f"[{float(p['ci_low']):+.4f}, {float(p['ci_high']):+.4f}]"])
    table(SUP/"oof_intervals.tex", "OOF paired F1 differences: Gaussian minus comparator.",
          "tab:oof-ci", ["Comparator / task", "Cap KiB", "Difference", "98.333\\% interval"],
          ci_rows, "10,000 paired geographic resamples, seed 20261029. Three-cap adjustment is per "
          "specified task and comparator. All eight areas and the same GT support enter each contrast.")
    with (SUP/"oof_full.tex").open("a", encoding="utf-8") as f:
        f.write("\n" + r"\input{generated/oof_intervals}" + "\n")
    OUTPUTS[(SUP/"oof_full.tex").relative_to(ROOT).as_posix()] = sha(SUP/"oof_full.tex")
    exp = load("results/lossless_control/summary.json")
    exp_done = load("results/lossless_control/progress.json")
    control = load(CTRL + "/summary.json")
    load(CTRL + "/complete.json")
    if exp_done["status"] != "complete" or exp_done["summary_sha256"] != sha(ROOT/"results/lossless_control/summary.json"):
        raise ValueError("Exp0200 information controls are not complete")
    if exp["outputs"] != 356 or control["outputs"] != 356 or control["status"] != "complete":
        raise ValueError("Both control evaluators require 178 observations per method")
    quality = {r["method"]: r["new_track_U30"] for r in exp["macro_aoi"]}
    controls = []
    for name, data in [("exp0200", exp), ("Author OOF", control)]:
        for row in data["macro_aoi"]:
            method = row["method"]
            if name == "exp0200":
                building, event = row["building_f1"], row["event_f1"]
                amortized = row["amortized_bytes_per_update"]
            else:
                building, event = row["building"]["f1"], row["event"]["f1"]
                a = [x for x in data["per_aoi_seed"] if x["method"] == method]
                amortized = statistics.mean(x["total_bytes_including_bootstrap"]/x["frames"] for x in a)
            controls.append({
                "evaluator": name, "method": method,
                "mean_update_bytes": row["mean_update_bidirectional_bytes"],
                "amortized_bytes": amortized, "building_f1": building,
                "event_f1": event, "U30": quality[method],
            })
    labels = {"full_lossless_png": "Full PNG", "old_only": "Bootstrap only"}
    control_rows = [[r["evaluator"]+" / "+labels[r["method"]],
                    f"{r['mean_update_bytes']/1024:.2f}", f"{r['amortized_bytes']/1024:.2f}",
                    f"{r['building_f1']:.4f}", f"{r['event_f1']:.4f}", f"{100*r['U30']:.2f}"]
                    for r in controls]
    table(SUP/"full_controls.tex", "Actual full-information and bootstrap-only controls.",
          "tab:full-controls", ["Evaluator / input", "Update KiB", "With bootstrap", "Building F1", "Event F1", r"$U_{30}$ \%"],
          control_rows, "No per-update cap is imposed on full PNG. Bootstrap is charged to both alternatives; "
          "bootstrap-only sends no current packet or stable confirmation. Byte and quality values are identical "
          "across evaluators because the received images are unchanged. Task values use the common current/initial "
          "validity intersection. These are diagnostic reference points, not constrained global task upper bounds.")
    save(MAIN/"oof_numeric_claims.json", json.dumps({
        "scope": summary["scope"], "macro_aoi": summary["macro_aoi"],
        "paired_comparisons": pairs, "information_controls": controls,
    }, indent=2, allow_nan=False))
    for relative, digest in INPUTS.items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError("Evidence changed during generation")
    save(MAIN/"oof_asset_manifest.json", json.dumps({
        "status": "complete", "source_sha256": INPUTS, "output_sha256": OUTPUTS.copy(),
        "generator_sha256": sha(__file__), "matched_original_AOI_records": len(areas),
    }, indent=2))
    print(json.dumps({"status": "complete", "OOF_outputs": 4806, "control_outputs": 712,
                      "matched_AOI_records": len(areas), "assets": len(OUTPUTS)}))


if __name__ == "__main__":
    main()
