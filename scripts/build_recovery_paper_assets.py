"""Generate supplementary tables from completed original-hypothesis tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/supplement/generated"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis_path = ROOT / "results/research_recovery_v1/analysis.json"
    analysis = json.loads(analysis_path.read_text("utf-8"))
    probe = ROOT / "results/unknown_recovery/semantic_probe_v1"
    task_complete = json.loads((probe / "complete.json").read_text("utf-8"))
    if task_complete["status"] != "complete" or task_complete["records"] != 8832:
        raise ValueError("Incomplete receiver task probe")
    with (probe / "comparisons.csv").open(encoding="utf-8") as stream:
        tasks = {(int(r["heldout"]), int(r["budget"]), r["method"]): r for r in csv.DictReader(stream)}
    with (ROOT / "results/unknown_recovery/codec_v1/comparisons.csv").open(encoding="utf-8") as stream:
        evidence = {(int(r["heldout"]), int(r["budget"]), r["method"]): r for r in csv.DictReader(stream) if r["comparator"] == "feature_change"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Risk-triggered absolute refresh relative to acknowledged Delta retransmission on the eight retained SN7 areas.}",
             r"\label{tab:recovery-link}",
             r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{4.0pt}\renewcommand{\arraystretch}{1.06}",
             r"\begin{tabularx}{\columnwidth}{Xrrr}\toprule",
             r"\rowcolor{tableheadbg} Profile & MSE red. (\%) & Byte $\Delta$ (\%) & Curr. frac.\\\midrule"]
    for row in analysis["link"]:
        if row["comparator"] != "delta_arq":
            continue
        lines.append(f"{row['profile'].replace('lossless', 'Lossless').replace('iid', 'IID ')} & {100*row['relative_world_mse_reduction']:.2f} & {100*row['relative_update_byte_change']:+.2f} & {row['new_current_fraction']:.3f} \\\\")
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table}"]
    link_path = OUT / "recovery_link.tex"
    link_path.write_text("\n".join(lines) + "\n", "utf-8")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Temporal kNN novelty mixed with change, minus feature-change routing, on the 64 SECOND development pairs. Coverage differences are percentage points.}",
             r"\label{tab:recovery-unknown}",
             r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{3.8pt}\renewcommand{\arraystretch}{1.06}",
             r"\begin{tabularx}{\columnwidth}{Xrrrr}\toprule",
             r"\rowcolor{tableheadbg} Held out & KiB & $\Delta U_{30}$ & $\Delta$ known F1 & F1 low\\\midrule"]
    for heldout, label in ((5, "Building"), (3, "Tree"), (4, "Water")):
        for budget in (4096, 8192, 16384, 32768):
            key = heldout, budget, "emergent_mix"
            coverage, task = evidence[key], tasks[key]
            lines.append(f"{label} & {budget//1024} & {100*float(coverage['delta_coverage_30']):+.2f} & {float(task['delta_known_macro_f1']):+.4f} & {float(task['ci_low']):+.4f} \\\\")
    lines += [r"\bottomrule\end{tabularx}\endgroup",
              r"\par\vspace{2.5pt}\begin{minipage}{\columnwidth}\fontsize{7.5}{9.2}\selectfont\emph{Notes.} Limits are descriptive 98.75\% pair-bootstrap limits within each four-budget class/method comparison. The internal known-F1 margin is $-0.02$. The task uses fixed known-class calibration prototypes on frozen DINO features.\end{minipage}", r"\end{table}"]
    unknown_path = OUT / "recovery_unknown.tex"
    unknown_path.write_text("\n".join(lines) + "\n", "utf-8")
    pq_path = ROOT / "results/product_quantizer/known_only_v1/complete.json"
    pq = json.loads(pq_path.read_text("utf-8"))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Known-only standard product quantization on 64 development pairs. Complete current indices and XOR indices use actual PQF1 frames. F1 uses the same fixed head on decoded features.}",
             r"\label{tab:recovery-pq}",
             r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{3.8pt}\renewcommand{\arraystretch}{1.06}",
             r"\begin{tabularx}{\columnwidth}{Xrrrrr}\toprule",
             r"\rowcolor{tableheadbg} Held out & $M$ & Curr. B & XOR B & Full F1 & PQ F1\\\midrule"]
    labels = {5: "Building", 3: "Tree", 4: "Water"}
    for row in pq["rows"]:
        lines.append(f"{labels[row['heldout']]} & {row['parts']} & {row['mean_absolute_feature_frame_bytes']:.0f} & {row['mean_xor_feature_frame_bytes']:.0f} & {row['known_macro_f1']['full_feature']:.4f} & {row['known_macro_f1']['PQ']:.4f} " + chr(92) * 2)
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table}"]
    pq_table = OUT / "recovery_pq.tex"
    pq_table.write_text("\n".join(lines) + "\n", "utf-8")
    hybrid = ROOT / "results/hybrid_pq/development_v2"
    comparisons = list(csv.DictReader((hybrid / "comparisons.csv").open(encoding="utf-8")))
    summary = {(int(r["heldout"]), int(r["budget"]), r["method"]): r for r in csv.DictReader((hybrid / "summary.csv").open(encoding="utf-8"))}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Complete PQ feature service plus optional JPEG evidence on the 64 development pairs. Every router gives identical known-task confusion counts. Coverage changes and intervals are percentage points relative to change-only.}",
             r"\label{tab:recovery-hybrid}",
             r"\begingroup\fontsize{8}{10.0}\selectfont\setlength{\tabcolsep}{3.8pt}\renewcommand{\arraystretch}{1.06}",
             r"\begin{tabularx}{\columnwidth}{Xrrrr}\toprule",
             r"\rowcolor{tableheadbg} Held out & KiB & $\Delta U_{30}$ & 98.75\% CI & Known F1\\\midrule"]
    for row in comparisons:
        if row["method"] != "emergent_mix" or row["comparator"] != "change":
            continue
        h, b = int(row["heldout"]), int(row["budget"])
        task = summary[(h, b, "emergent_mix")]
        lines.append(f"{labels[h]} & {b//1024} & {100*float(row['delta_unknown_coverage_30']):+.2f} & [{100*float(row['interval_low']):+.2f},{100*float(row['interval_high']):+.2f}] & {float(task['known_macro_f1']):.4f} " + chr(92) * 2)
    lines += [r"\bottomrule\end{tabularx}\endgroup", r"\end{table}"]
    hybrid_table = OUT / "recovery_hybrid.tex"
    hybrid_table.write_text("\n".join(lines) + "\n", "utf-8")
    figure = ROOT / "paper/supplement/figures/recovery_summary.pdf"
    shutil.copyfile(ROOT / "results/research_recovery_v1/recovery_summary.pdf", figure)
    inputs = [analysis_path, probe / "complete.json", probe / "comparisons.csv",
              ROOT / "results/unknown_recovery/codec_v1/comparisons.csv", pq_path,
              hybrid / "analysis.json", hybrid / "summary.csv", hybrid / "comparisons.csv",
              ROOT / "results/link_recovery/fixed_reference_v1/analysis.json"]
    manifest = dict(source_sha256=sha(Path(__file__)),
                    input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                    output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (link_path, unknown_path, pq_table, hybrid_table, OUT / "recovery_fixed.tex", figure)})
    (OUT / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
