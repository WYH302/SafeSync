"""Generate the three core decision figures, the comprehensive cost table, and diagnostic tables for the manuscript and supplement."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_figure1_sn7_comm_vs_utility(lowres_analysis: dict, sn7_repaired: dict) -> list[Path]:
    """Figure 1: Actual communication bytes vs current update utility on SpaceNet 7 (Building F1 & New-track Event F1)."""
    summaries = lowres_analysis["summaries"]
    lookup = {(r["method"], int(r["budget"])): r for r in summaries}
    for r in sn7_repaired["repair_1b_decoupled_gating_and_audit"]["summary_by_method_and_budget"]:
        if r["method"] == "gaussian_decoupled":
            lookup["gaussian_decoupled", int(r["budget"])] = {
                "mean_bytes": r["mean_bytes"],
                "building_f1": r["building_f1_mean"],
                "event_f1": r["event_f1_mean"],
            }

    methods_cfg = [
        ("gaussian_decoupled", "SafeSync-DGAD (12.3 KB, 0 B mat.)", "#004488", "o", "-", 1.7),
        ("gaussian_20261029", "SafeSync-ZFSS (Coupled, 12.3 KB)", "#0072B2", "o", "--", 1.5),
        ("dct_bound", "DCT bound (13.9 KB, 0 shared)", "#E69F00", "s", "-", 1.4),
        ("lowres_4x_webp_12k", "Low-res 4x WebP ref (<=12.3 KB)", "#56B4E9", "^", "--", 1.4),
        ("lowres_4x_webp_12k_gate30", "Low-res 4x WebP ref (30 dB gate)", "#009E73", "v", ":", 1.3),
        ("lowres_16x16_coarse_only", "Coarse-only 16x16 (0 detail)", "#999999", "d", "--", 1.2),
        ("full_cache_gate30", "Full history (30 dB, 1.41 MB PNG)", "#2B9F78", "P", "-.", 1.4),
        ("full_webp", "WebP (0 hist. state)", "#555555", "X", "-", 1.3),
        ("learned_hyperprior_q1_8", "Hyperprior (61 MB weights)", "#CC79A7", "D", "-", 1.4),
        ("tcm", "LIC-TCM (4.18 GB weights)", "#D55E00", "*", "-", 1.7),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    for m, label, color, marker, ls, lw in methods_cfg:
        pts = [lookup[m, cap] for cap in (32768, 65536, 131072)]
        xs = [float(r["mean_bytes"]) / 1024.0 for r in pts]
        b_f1 = [float(r["building_f1"]) for r in pts]
        e_f1 = [float(r["event_f1"]) for r in pts]
        axes[0].plot(xs, b_f1, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=4.5)
        axes[1].plot(xs, e_f1, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=4.5)

    # Annotate Bootstrap-only (0 update bytes: Building F1 = 0.5475, Event F1 = 0.0000)
    axes[0].axhline(0.5475, color="#333333", linestyle=":", linewidth=1.0, alpha=0.75)
    axes[0].text(34.0, 0.531, "Bootstrap-only (0 KiB/upd, Event F1 = 0.000)", fontsize=6.4, color="#333333")
    axes[0].set_ylim(0.265, 0.568)

    axes[0].set_title("(a) Static Building F1 (Author OOF)", fontsize=8.5)
    axes[1].set_title("(b) New-Track Change Event F1 (Author OOF)", fontsize=8.5)
    axes[0].set_ylabel("Building F1")
    axes[1].set_ylabel("New-track Event F1")
    for ax in axes:
        ax.set_xlabel("Actual mean bidirectional update size (KiB)")
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=6.8)

    pdf_path = ROOT / "paper/figures/tcm_oof_curves.pdf"
    png_path = ROOT / "results/lowres_reference_sn7/v1/sn7_decision_curves.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return [pdf_path, png_path]


def build_figure2_storage_vs_utility(
    lowres_analysis: dict, png_profile: dict, sender_ref: dict, diag: dict, sec_repaired: dict
) -> list[Path]:
    """Figure 2: 2x2 Comprehensive State Scaling, Safety/Drift Immunity, State-Utility, and SECOND Open-World Discovery."""
    summaries = lowres_analysis["summaries"]
    lookup32 = {r["method"]: r for r in summaries if int(r["budget"]) == 32768}

    cap32_png_8aoi = next(r for r in png_profile["per_cap"] if int(r["budget"]) == 32768)["sum_AOI_mean_PNG_bytes"]
    mean_png_per_aoi = cap32_png_8aoi / 8.0

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.35), constrained_layout=True)

    # Panel (a): Total Sender Historical State vs Monitored Areas N (1..16)
    ns = np.arange(1, 17)
    raw_rgb_mb = (3145728.0 * ns) / (1024**2)
    png_hist_mb = (mean_png_per_aoi * ns) / (1024**2)
    gauss_cached_mb = (2359296.0 + 12328.0 * ns) / (1024**2)
    gauss_stream_mb = (12328.0 * ns) / (1024**2)
    dct_mb = (13864.0 * ns) / (1024**2)
    lowres_mb = (lookup32["lowres_4x_webp_12k"]["mean_sender_summary_bytes"] * ns) / (1024**2)

    ax = axes[0, 0]
    ax.plot(ns, raw_rgb_mb, label="Full history (Raw RGB)", color="#888888", linestyle="--", linewidth=1.2)
    ax.plot(ns, png_hist_mb, label="Full history (Lossless PNG)", color="#2B9F78", marker="P", markersize=3.4, linewidth=1.4)
    ax.plot(ns, gauss_cached_mb, label="SafeSync (cached 2.36M mat.)", color="#56B4E9", marker="o", markersize=2.8, linewidth=1.2, linestyle="--")
    ax.plot(ns, gauss_stream_mb, label="SafeSync-ZFSS (0 B matrix)", color="#004488", marker="o", markersize=3.8, linewidth=1.7)
    ax.plot(ns, dct_mb, label="DCT bound (0 shared mat.)", color="#E69F00", marker="s", markersize=3.2, linewidth=1.25)
    ax.plot(ns, lowres_mb, label="Low-res 4x WebP ref", color="#009E73", marker="^", markersize=3.2, linewidth=1.2, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(0.007, 480.0)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Monitored geographic areas $N$")
    ax.set_ylabel("Sender historical state (MiB)")
    ax.set_title("(a) Multi-AOI Onboard State Scaling", fontsize=8.2, fontweight="bold")
    ax.grid(alpha=0.22, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none", fontsize=5.3, ncol=2, loc="upper left")

    # Panel (b): Reuse Certificate Violations & Pass-to-Pass Orbital-Drift False Triggers (32 KiB)
    ax = axes[0, 1]
    methods_bar = ["SafeSync\n(ZFSS / DGAD)", "Coarse-Only\n16x16 Ref", "Direct 4x\nWebP Ref"]
    viol_pct = [0.00, 17.35, 21.79]     # % of violating blocks falsely certified STABLE
    drift_pct = [19.37, 23.15, 79.80]   # % background false triggers under orbital drift
    xb = np.arange(len(methods_bar))
    wb = 0.34
    b1 = ax.bar(xb - wb / 2, viol_pct, width=wb, color="#DC2626", alpha=0.88, label="False-Stable Violation Rate (%)")
    b2 = ax.bar(xb + wb / 2, drift_pct, width=wb, color="#2563EB", alpha=0.88, label="Orbital-Drift False Triggers (%)")
    for rect, v in zip(b1, viol_pct):
        lbl = "0% (0)" if v == 0 else f"{v:.1f}%"
        x_pos = rect.get_x() + rect.get_width() / 2 - (0.04 if v == 0 else 0.0)
        ax.text(x_pos, v + 1.8, lbl, ha="center", fontsize=6.0, color="#991B1B", fontweight="bold")
    for rect, v in zip(b2, drift_pct):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", fontsize=6.0, color="#1E40AF", fontweight="bold")
    ax.set_xticks(xb, methods_bar, fontsize=6.8)
    ax.set_ylim(0, 96.0)
    ax.set_ylabel("Error / False-trigger rate (%)")
    ax.set_title("(b) Reuse Safety & Orbital-Drift Immunity", fontsize=8.2, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel (c): N=8 Total Sender Historical State (MiB) vs 32 KiB Event F1 (zero overlap!)
    ax = axes[1, 0]
    pts_b = [
        ("SafeSync-ZFSS", (8 * 12328) / (1024**2), lookup32["gaussian_20261029"]["event_f1"], "#004488", "o", (0.045, 0.015)),
        ("SafeSync (cached)", (2359296 + 8 * 12328) / (1024**2), lookup32["gaussian_20261029"]["event_f1"], "#0072B2", "o", (0.28, -0.020)),
        ("DCT", (8 * 13864) / (1024**2), lookup32["dct_bound"]["event_f1"], "#E69F00", "s", (1.35, -0.004)),
        ("4x WebP (25dB)", (8 * lookup32["lowres_4x_webp_12k"]["mean_sender_summary_bytes"]) / (1024**2), lookup32["lowres_4x_webp_12k"]["event_f1"], "#56B4E9", "^", (1.38, -0.008)),
        ("4x WebP (30dB)", (8 * lookup32["lowres_4x_webp_12k_gate30"]["mean_sender_summary_bytes"]) / (1024**2), lookup32["lowres_4x_webp_12k_gate30"]["event_f1"], "#009E73", "v", (1.38, 0.005)),
        ("Coarse 16x16", (8 * 12328) / (1024**2), lookup32["lowres_16x16_coarse_only"]["event_f1"], "#777777", "d", (1.35, -0.003)),
        ("Full PNG Hist.", cap32_png_8aoi / (1024**2), lookup32["full_cache_gate30"]["event_f1"], "#2B9F78", "P", (0.25, 0.011)),
        ("WebP", 0.002, lookup32["full_webp"]["event_f1"], "#555555", "X", (1.45, -0.003)),
        ("Hyperprior*", 0.002, lookup32["learned_hyperprior_q1_8"]["event_f1"], "#CC79A7", "D", (1.45, -0.003)),
        ("LIC-TCM*", 0.002, lookup32["tcm"]["event_f1"], "#D55E00", "*", (1.45, -0.004)),
    ]
    for name, st_mib, ef1, col, mk, (dx, dy) in pts_b:
        ax.scatter(st_mib, ef1, color=col, marker=mk, s=38, zorder=3)
        ap = dict(arrowstyle="-", color=col, lw=0.7, alpha=0.75) if "SafeSync-ZFSS" in name else None
        ax.annotate(name, (st_mib, ef1), xytext=(st_mib * dx, ef1 + dy), fontsize=5.8, color=col,
                    fontweight="bold" if "SafeSync-ZFSS" in name else "normal", arrowprops=ap)
    ax.set_xscale("log")
    ax.set_xlim(0.001, 38.0)
    ax.set_ylim(0.088, 0.242)
    ax.set_xlabel("8-AOI sender state (MiB; *excl. 61M/4.18G weights)")
    ax.set_ylabel("32-KiB New-Track Event F1")
    ax.set_title("(c) SpaceNet 7 State Footprint vs. Event F1", fontsize=8.2, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel (d): SECOND Cohort 3 Open-World Discovery (U30 Coverage vs. Held-Out Later Building F1)
    ax = axes[1, 1]
    cells = diag["factorial_2x2"]["h5_b8192"]["cells"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    rep = _index_sec_repaired(sec_repaired)
    sec_pts = [
        ("WebP", cells["full_webp"]["u30"], cells["full_webp"]["later_f1"], "#555555", "X", (0.008, -0.004)),
        ("Hyperprior", cells["hyperprior"]["u30"], cells["hyperprior"]["later_f1"], "#CC79A7", "D", (-0.048, 0.005)),
        ("Abs.+Change", cells["absolute_change"]["u30"], cells["absolute_change"]["later_f1"], "#E69F00", "s", (-0.072, 0.002)),
        ("Locked Res.", cells["residual_current"]["u30"], cells["residual_current"]["later_f1"], "#56B4E9", "^", (-0.078, 0.003)),
        ("TC-PCR (4A+4B)", rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel"]["later_f1"], "#004488", "o", (-0.105, -0.008)),
        ("TC-PCR Refined", rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel_calprobe"]["later_f1"], "#15803D", "P", (-0.115, -0.004)),
        ("LIC-TCM (4.18G)", cells["tcm"]["u30"], cells["tcm"]["later_f1"], "#D55E00", "*", (0.006, 0.003)),
        ("GT Oracle", ora["oracle_hybrid_with_full_hp"]["u30"], ora["oracle_hybrid_with_full_hp"]["later_f1"], "#666666", "p", (-0.036, -0.009)),
    ]
    for lbl, u30, lf1, col, mk, (ox, oy) in sec_pts:
        ax.scatter(u30, lf1, color=col, marker=mk, s=42, zorder=4)
        ax.annotate(lbl, (u30, lf1), xytext=(u30 + ox, lf1 + oy), fontsize=5.8, color=col,
                    fontweight="bold" if "TC-PCR" in lbl else "normal")
    ax.annotate("", xy=(rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel_calprobe"]["later_f1"]),
                xytext=(rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel"]["later_f1"]),
                arrowprops=dict(arrowstyle="->", color="#15803D", lw=1.1, ls="--"))
    ax.set_xlim(0.515, 0.845)
    ax.set_ylim(0.796, 0.858)
    ax.set_xlabel("Held-out building 30-dB coverage $U_{30}$")
    ax.set_ylabel("Held-out building Later F1")
    ax.set_title("(d) SECOND Open-World Discovery (8 KiB)", fontsize=8.2, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    pdf_path = ROOT / "paper/figures/storage_utility_tradeoff.pdf"
    png_path = ROOT / "results/lowres_reference_sn7/v1/storage_utility_tradeoff.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return [pdf_path, png_path]


def _index_sec_repaired(sec_repaired: dict) -> dict[str, dict]:
    out = {}
    for r in sec_repaired["summary_rows"]:
        rc = dict(r)
        rc["later_fnr"] = float(r["heldout_fnr"])
        rc["mean_total_bytes"] = float(r["mean_down_bytes"]) + float(r["mean_ack_bytes"])
        out[r["method"]] = rc
    return out


def build_figure3_second_four_cell_and_oracle(diag: dict, sec_repaired: dict) -> list[Path]:
    """Figure 3: Evidence coverage (U30) vs Later-task F1 and Held-out FNR on SECOND Cohort 3 (2x2 + Repaired + Oracle)."""
    cells = diag["factorial_2x2"]["h5_b8192"]["cells"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    rep = _index_sec_repaired(sec_repaired)

    points = [
        ("Cell A: Abs.+Change", cells["absolute_change"]["u30"], cells["absolute_change"]["later_f1"], cells["absolute_change"]["later_fnr"], "#E69F00", "s"),
        ("Cell B: Abs.+Unknown", cells["absolute_current"]["u30"], cells["absolute_current"]["later_f1"], cells["absolute_current"]["later_fnr"], "#D55E00", "D"),
        ("Cell C: Res.+Change (locked)", cells["residual_change"]["u30"], cells["residual_change"]["later_f1"], cells["residual_change"]["later_fnr"], "#009E73", "o"),
        ("Cell D: Res.+Unknown (locked)", cells["residual_current"]["u30"], cells["residual_current"]["later_f1"], cells["residual_current"]["later_fnr"], "#56B4E9", "^"),
        ("TC-PCR Change (4A+4B)", rep["repaired_residual_change_sel"]["u30"], rep["repaired_residual_change_sel"]["later_f1"], rep["repaired_residual_change_sel"]["later_fnr"], "#117733", "o"),
        ("TC-PCR Unknown (4A+4B)", rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel"]["later_f1"], rep["repaired_residual_unknown_sel"]["later_fnr"], "#004488", "^"),
        ("TC-PCR Unknown (Raw, 4B)", rep["repaired_residual_unknown_raw"]["u30"], rep["repaired_residual_unknown_raw"]["later_f1"], rep["repaired_residual_unknown_raw"]["later_fnr"], "#332288", "D"),
        ("TC-PCR Refined Probe (4C)", rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel_calprobe"]["later_f1"], rep["repaired_residual_unknown_sel_calprobe"]["later_fnr"], "#002255", "P"),
        ("Oracle Hybrid (8 KiB)", ora["oracle_hybrid_with_full_hp"]["u30"], ora["oracle_hybrid_with_full_hp"]["later_f1"], ora["oracle_hybrid_with_full_hp"]["later_fnr"], "#AA4499", "*"),
        ("Full Hyperprior (8 KiB)", cells["hyperprior"]["u30"], cells["hyperprior"]["later_f1"], cells["hyperprior"]["later_fnr"], "#CC79A7", "h"),
        ("Full WebP (8 KiB)", cells["full_webp"]["u30"], cells["full_webp"]["later_f1"], cells["full_webp"]["later_fnr"], "#333333", "X"),
        ("Full TCM (8 KiB)", cells["tcm"]["u30"], cells["tcm"]["later_f1"], cells["tcm"]["later_fnr"], "#CC3311", "p"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)

    # Panel (a): U30 vs Later-Task Building F1
    ax = axes[0]
    for label, u30, lf1, fnr, col, mk in points:
        ax.scatter(u30, lf1, color=col, marker=mk, s=42, label=label, zorder=4)

    # Draw arrows for 2x2 Row effect (A->C) and TC-PCR Unknown gain
    ax.annotate(
        "",
        xy=(cells["residual_change"]["u30"], cells["residual_change"]["later_f1"]),
        xytext=(cells["absolute_change"]["u30"], cells["absolute_change"]["later_f1"]),
        arrowprops=dict(arrowstyle="->", color="#009E73", lw=1.2),
    )
    ax.text(0.555, 0.8035, "Row (Coding C-A):\n+3.21 pp U30", fontsize=5.5, color="#009E73")
    ax.annotate(
        "",
        xy=(rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel"]["later_f1"]),
        xytext=(rep["repaired_residual_change_sel"]["u30"], rep["repaired_residual_change_sel"]["later_f1"]),
        arrowprops=dict(arrowstyle="->", color="#004488", lw=1.3),
    )
    ax.text(0.680, 0.8195, "TC-PCR Unknown:\n+1.44 pp vs Change\n(U30=0.7594 ~ TCM)", fontsize=5.5, color="#004488")
    ax.annotate(
        "",
        xy=(rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel_calprobe"]["later_f1"]),
        xytext=(rep["repaired_residual_unknown_sel"]["u30"], rep["repaired_residual_unknown_sel"]["later_f1"]),
        arrowprops=dict(arrowstyle="->", color="#002255", lw=1.1, ls="--"),
    )
    ax.text(0.610, 0.8440, "Refined ROI probe (4C):\nF1 = 0.8503 > WebP/TCM", fontsize=5.5, color="#002255")

    # WebP non-inferiority line (0.8332 - 0.02 = 0.8132)
    ax.axhline(cells["full_webp"]["later_f1"], color="#333333", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(cells["full_webp"]["later_f1"] - 0.02, color="#CC0000", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(0.535, cells["full_webp"]["later_f1"] - 0.0190, "WebP -2.0 pp non-inferiority floor (0.8132)", fontsize=5.5, color="#CC0000")

    ax.set_xlabel("Qualified unknown evidence coverage $U_{30}$")
    ax.set_ylabel("Later held-out building F1")
    ax.set_title("(a) 2x2 Attribution, TC-PCR Routing & Oracle (8 KiB)", fontsize=8.0)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel (b): Held-out Building False Negative Rate (FNR) vs Later-Task Building F1
    ax = axes[1]
    for label, u30, lf1, fnr, col, mk in points:
        ax.scatter(fnr, lf1, color=col, marker=mk, s=42, label=label, zorder=4)
    ax.annotate(
        "TC-PCR Raw (4B): FNR=0.162 < WebP/HP/TCM\nRefined Region (4C): FNR=0.154, F1=0.850",
        xy=(rep["repaired_residual_unknown_raw"]["later_fnr"], rep["repaired_residual_unknown_raw"]["later_f1"]),
        xytext=(0.155, 0.835),
        arrowprops=dict(arrowstyle="->", color="#332288", lw=1.0),
        fontsize=5.5,
        color="#332288",
    )
    ax.set_xlabel("Held-out building False Negative Rate (FNR = FN / GT)")
    ax.set_ylabel("Later held-out building F1")
    ax.set_title("(b) Target Miss (FNR) vs Later Building F1", fontsize=8.0)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=5.6)

    pdf_path = ROOT / "paper/supplement/figures/second_four_cell_and_oracle.pdf"
    png_path = ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/second_four_cell_and_oracle.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return [pdf_path, png_path]


def build_tables(
    lowres_analysis: dict,
    png_profile: dict,
    diag: dict,
    sn7_repaired: dict,
    sec_repaired: dict,
) -> list[Path]:
    """Generate the comprehensive cost & utility table and diagnostic tables."""
    summaries = lowres_analysis["summaries"]
    lookup = {(r["method"], int(r["budget"])): r for r in summaries}
    for r in sn7_repaired["repair_1b_decoupled_gating_and_audit"]["summary_by_method_and_budget"]:
        lookup[r["method"], int(r["budget"])] = {
            "mean_bytes": r["mean_bytes"],
            "building_f1": r["building_f1_mean"],
            "event_f1": r["event_f1_mean"],
            "new_track_u30_pooled": r["new_track_u30_pooled"],
        }
    cap32_png_8aoi = next(r for r in png_profile["per_cap"] if int(r["budget"]) == 32768)["sum_AOI_mean_PNG_bytes"]
    mean_png_1aoi = cap32_png_8aoi / 8.0

    # Measure mean TCM search & selected encode ms on SN7 (32 KiB) and SECOND (8 KiB)
    sn7_tcm_recs = [
        r
        for r in json.loads((ROOT / "results/tcm_sn7/transport_v1/records.json").read_text(encoding="utf-8"))
        if int(r["cap"]) == 32768
    ]
    sn7_tcm_search_ms = float(np.mean([float(r["sender_search_ms_shared"]) for r in sn7_tcm_recs]))
    sn7_tcm_comp_ms = float(np.mean([float(r["codec_metadata"]["sender_compress_ms"]) for r in sn7_tcm_recs]))

    sec_tcm_recs = [
        r
        for r in json.loads((ROOT / "results/task_evidence/tcm_control_v1/records.json").read_text(encoding="utf-8"))
        if int(r["heldout"]) == 5 and int(r["budget"]) == 8192
    ]
    sec_tcm_search_ms = float(np.mean([float(r["complete_search_ms"]) for r in sec_tcm_recs]))
    sec_tcm_comp_ms = float(np.mean([float(r["codec_metadata"]["sender_compress_ms"]) for r in sec_tcm_recs]))

    end = chr(92) * 2

    # 1. Comprehensive Cost & Utility Table (uniform 8pt font, spacious columns, header & row shading)
    cost_lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Comprehensive onboard state footprint, compute latency, and downstream utility across SpaceNet 7 monthly synchronization (32-KiB cap, 8 AOIs, 4-fold Author-OOF) and SECOND Cohort 3 open-world synchronization (8-KiB cap, 256 pairs, Building held-out). Sender historical state is shown for $N=1$ and $N=8$ monitored areas; proposed SafeSync variants are shaded.}",
        r"\label{tab:full-cost}",
        r"\begingroup\fontsize{8}{10.1}\selectfont",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabularx}{\textwidth}{lllcrrrX}\toprule",
        r"\rowcolor{tableheadbg} Method / Variant & State ($N\!=\!1$ / $8$) & Weights / Matrix & Sender Time & KiB & M1 $\uparrow$ & M2 $\uparrow$ & Key Regime / Audit" + end,
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel A: SpaceNet 7 Monthly Pixel-Tier Sync (32-KiB cap, 178 updates; M1 = Author-OOF Building F1, M2 = Event F1)}}" + end,
        r"\cmidrule(lr){1-8}",
    ]

    sn7_rows = [
        (
            r"\rowcolor{safesyncbg} \textbf{SafeSync-ZFSS}",
            r"\textbf{12.3 / 98.6 KB}",
            r"\textbf{0 B (stream)}",
            "254.7 ms (CPU)",
            f"{lookup['gaussian_20261029', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['gaussian_20261029', 32768]['building_f1']:.4f}",
            f"{lookup['gaussian_20261029', 32768]['event_f1']:.4f}",
            r"\textbf{0 viol.}; $U_{30}\!=\!0.355$",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{SafeSync-DGAD}",
            r"\textbf{12.3 / 98.6 KB}",
            r"\textbf{0 B (stream)}",
            "254.7 ms (CPU)",
            rf"\textbf{{{lookup['gaussian_decoupled', 32768]['mean_bytes']/1024:.2f}}}",
            rf"\textbf{{{lookup['gaussian_decoupled', 32768]['building_f1']:.4f}}}",
            f"{lookup['gaussian_decoupled', 32768]['event_f1']:.4f}",
            r"\textbf{0 viol.}; top certified CPU",
        ),
        (
            "DCT summary bound",
            "13.9 / 110.9 KB",
            "0 B (CPU DCT)",
            "291.2 ms (CPU)",
            f"{lookup['dct_bound', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['dct_bound', 32768]['building_f1']:.4f}",
            f"{lookup['dct_bound', 32768]['event_f1']:.4f}",
            "Deterministic 48-B",
        ),
        (
            "Low-res 4x WebP ref",
            "10.7 / 85.4 KB",
            "0 B (CPU WebP)",
            "226.4 ms (CPU)",
            f"{lookup['lowres_4x_webp_12k', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['lowres_4x_webp_12k', 32768]['building_f1']:.4f}",
            f"{lookup['lowres_4x_webp_12k', 32768]['event_f1']:.4f}",
            "1,495 false-stable viol.",
        ),
        (
            "Coarse-only 16x16",
            "12.3 / 98.6 KB",
            "0 B (CPU mean)",
            "206.8 ms (CPU)",
            f"{lookup['lowres_16x16_coarse_only', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['lowres_16x16_coarse_only', 32768]['building_f1']:.4f}",
            f"{lookup['lowres_16x16_coarse_only', 32768]['event_f1']:.4f}",
            "1,190 false-stable viol.",
        ),
        (
            "Full history (30 dB)",
            f"{mean_png_1aoi/(1024**2):.2f} / {cap32_png_8aoi/(1024**2):.2f} MB",
            "0 B (PNG hist.)",
            "257.5 ms (+PNG)",
            f"{lookup['full_cache_gate30', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['full_cache_gate30', 32768]['building_f1']:.4f}",
            f"{lookup['full_cache_gate30', 32768]['event_f1']:.4f}",
            "Full-archive upper ref.",
        ),
        (
            "Stateless WebP",
            "0.24 / 1.9 KB",
            "0 B (CPU WebP)",
            "2.06 s / 56 ms",
            f"{lookup['full_webp', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['full_webp', 32768]['building_f1']:.4f}",
            f"{lookup['full_webp', 32768]['event_f1']:.4f}",
            "Stateless CPU baseline",
        ),
        (
            "Scale-Hyperprior",
            "0.24 / 1.9 KB",
            "61.1 MB (8 ckpt)",
            "11.5 s GPU / 36 s",
            f"{lookup['learned_hyperprior_q1_8', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['learned_hyperprior_q1_8', 32768]['building_f1']:.4f}",
            f"{lookup['learned_hyperprior_q1_8', 32768]['event_f1']:.4f}",
            "Lightweight neural",
        ),
        (
            "LIC-TCM (35 cand.)",
            "0.24 / 1.9 KB",
            "4.18 GB (6 ckpt)",
            f"{sn7_tcm_search_ms/1000:.1f} s / {sn7_tcm_comp_ms/1000:.2f} s GPU",
            f"{lookup['tcm', 32768]['mean_bytes']/1024:.2f}",
            rf"\underline{{{lookup['tcm', 32768]['building_f1']:.4f}}}",
            rf"\textbf{{{lookup['tcm', 32768]['event_f1']:.4f}}}",
            "Heavy GPU SOTA",
        ),
    ]
    for r in sn7_rows:
        cost_lines.append(" & ".join(r) + end)

    cells = diag["factorial_2x2"]["h5_b8192"]["cells"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    rep = _index_sec_repaired(sec_repaired)
    cost_lines.append(r"\midrule")
    cost_lines.append(
        r"\multicolumn{8}{l}{\textit{Panel B: SECOND Cohort 3 Semantic \& Open-World Sync (8-KiB cap, 256 pairs; M1 = Held-Out $U_{30}$, M2 = Later F1)}}" + end
    )
    cost_lines.append(r"\cmidrule(lr){1-8}")
    sec_rows = [
        (
            "Cell A: Abs. + Change",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.60 s (GPU+CPU)",
            f"{cells['absolute_change']['mean_total_bytes']/1024:.2f}",
            f"{cells['absolute_change']['u30']:.4f}",
            f"{cells['absolute_change']['later_f1']:.4f}",
            "Absolute ROI baseline",
        ),
        (
            "Cell C: Res. + Change",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.60 s (GPU+CPU)",
            f"{cells['residual_change']['mean_total_bytes']/1024:.2f}",
            f"{cells['residual_change']['u30']:.4f}",
            f"{cells['residual_change']['later_f1']:.4f}",
            "+9.05 pp vs. Hyperprior",
        ),
        (
            "Cell D: Res. + Unknown",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.56 s (GPU+CPU)",
            f"{cells['residual_current']['mean_total_bytes']/1024:.2f}",
            f"{cells['residual_current']['u30']:.4f}",
            f"{cells['residual_current']['later_f1']:.4f}",
            "Uncalibrated kNN",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (Change)}",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.58 s (GPU+CPU)",
            f"{rep['repaired_residual_change_sel']['mean_total_bytes']/1024:.2f}",
            f"{rep['repaired_residual_change_sel']['u30']:.4f}",
            f"{rep['repaired_residual_change_sel']['later_f1']:.4f}",
            "Active-support + 30 dB",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (Unknown)}",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.58 s (GPU+CPU)",
            f"{rep['repaired_residual_unknown_sel']['mean_total_bytes']/1024:.2f}",
            rf"\underline{{{rep['repaired_residual_unknown_sel']['u30']:.4f}}}",
            f"{rep['repaired_residual_unknown_sel']['later_f1']:.4f}",
            r"\textbf{+1.44 pp} vs. Change",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (Raw)}",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.56 s (GPU+CPU)",
            f"{rep['repaired_residual_unknown_raw']['mean_total_bytes']/1024:.2f}",
            f"{rep['repaired_residual_unknown_raw']['u30']:.4f}",
            f"{rep['repaired_residual_unknown_raw']['later_f1']:.4f}",
            r"\textbf{FNR 0.1616} $<$ WebP/TCM",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (Refined)}",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.58 s (GPU+CPU)",
            f"{rep['repaired_residual_unknown_sel']['mean_total_bytes']/1024:.2f}",
            rf"\underline{{{rep['repaired_residual_unknown_sel']['u30']:.4f}}}",
            rf"\textbf{{{rep['repaired_residual_unknown_sel_calprobe']['later_f1']:.4f}}}",
            r"\textbf{Best Refined ROI F1}",
        ),
        (
            "Oracle Hybrid (8 KiB)",
            "Post-hoc GT bound",
            "88.2M + 61.1M HP",
            "Offline GT bound",
            f"{ora['oracle_hybrid_with_full_hp']['mean_total_bytes']/1024:.2f}",
            f"{ora['oracle_hybrid_with_full_hp']['u30']:.4f}",
            f"{ora['oracle_hybrid_with_full_hp']['later_f1']:.4f}",
            "Selection ceiling",
        ),
        (
            "Full Hyperprior",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "2.80 s (GPU+CPU)",
            f"{cells['hyperprior']['mean_total_bytes']/1024:.2f}",
            f"{cells['hyperprior']['u30']:.4f}",
            f"{cells['hyperprior']['later_f1']:.4f}",
            "Learned base control",
        ),
        (
            "Full WebP (8 KiB)",
            "197 KB / 1.58 MB",
            "88.2M DINO task",
            "0.79 s (GPU+CPU)",
            f"{cells['full_webp']['mean_total_bytes']/1024:.2f}",
            f"{cells['full_webp']['u30']:.4f}",
            rf"\underline{{{cells['full_webp']['later_f1']:.4f}}}",
            "Global Later F1 ref.",
        ),
        (
            "Full LIC-TCM (8 KiB)",
            "197 KB / 1.58 MB",
            "88.2M + 4.18G TCM",
            f"{sec_tcm_search_ms/1000:.1f} s / {sec_tcm_comp_ms/1000:.2f} s GPU",
            rf"\textbf{{{cells['tcm']['mean_total_bytes']/1024:.2f}}}",
            rf"\textbf{{{cells['tcm']['u30']:.4f}}}",
            f"{cells['tcm']['later_f1']:.4f}",
            "Heavy 4.18-GB SOTA",
        ),
    ]
    for r in sec_rows:
        cost_lines.append(" & ".join(r) + end)
    cost_lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]

    p_main_cost = ROOT / "paper/generated/full_cost_table.tex"
    p_sup_cost = ROOT / "paper/supplement/generated/full_cost_table.tex"
    p_main_cost.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")
    p_sup_cost.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    # 2. Supplement table for Low-Res Reference Control & Pseudo-Change Diagnostic
    lowres_lines = [
        r"\begin{table*}[t]\centering\footnotesize",
        r"\caption{Matched-storage low-resolution historical reference controls ($\le 12,328$ B/area) and same-protocol comparators on the eight SpaceNet 7 confirmation AOIs under the Author-OOF evaluator. Within each 32/64/128-KiB cap, columns report Building F1, New-track Event F1, and actual mean bidirectional update KiB.}",
        r"\label{tab:lowres-sn7}",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.00}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}\toprule",
        r"Method & B32 & E32 & KiB32 & B64 & E64 & KiB64 & B128 & E128 & KiB128" + end,
        r"\midrule",
    ]
    lowres_order = [
        ("lowres_16x16_coarse_only", "Coarse-only 16x16 (12.3 KB, 1,190 viol.)"),
        ("lowres_4x_webp_12k", r"Low-res 4x WebP ref ($\le$12.3 KB, 1,495 viol.)"),
        ("lowres_4x_webp_12k_gate30", r"Low-res 4x WebP ref ($\le$12.3 KB, 30 dB)"),
        ("dct_bound", "DCT bound (13.9 KB, 0 shared)"),
        ("gaussian_20261029", "SafeSync-ZFSS (12.3 KB, 0B mat., coupled, 0 viol.)"),
        ("gaussian_decoupled", "SafeSync-DGAD (12.3 KB, 0B mat., decoupled, 0 viol.)"),
        ("full_cache_mse", "Full history (25 dB gate)"),
        ("full_cache_gate30", "Full history (30 dB gate)"),
        ("full_webp", "Full WebP"),
        ("learned_hyperprior_q1_8", "Hyperprior (q1..8)"),
        ("tcm", "LIC-TCM (35 candidates)"),
    ]
    for m, label in lowres_order:
        row_cells = [label]
        for cap in (32768, 65536, 131072):
            r = lookup[m, cap]
            row_cells += [
                f"{float(r['building_f1']):.3f}",
                f"{float(r['event_f1']):.3f}",
                f"{float(r['mean_bytes'])/1024:.1f}",
            ]
        lowres_lines.append(" & ".join(row_cells) + end)
    lowres_lines += [r"\bottomrule\end{tabular*}\end{table*}"]
    p_lowres_tex = ROOT / "paper/supplement/generated/lowres_sn7_table.tex"
    p_lowres_tex.write_text("\n".join(lowres_lines) + "\n", encoding="utf-8")

    # 3. Supplement table for 2x2 Four-Cell Attribution, TC-PCR Routing & Oracle Diagnostic on SECOND Cohort 3
    c_dict = diag["factorial_2x2"]["h5_b8192"]["contrasts"]
    o_dict = diag["oracle_diagnostic_h5_b8192"]["contrasts"]
    rc_dict = sec_repaired["paired_bootstrap_contrasts"]
    fact_lines = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\caption{Complete $2\times 2$ four-cell attribution, Threshold-Crossing Prototype-Calibrated Routing (TC-PCR: active-support selector + sender-visible 30-dB threshold-crossing ranking + calibrated known-class emergence + refined-evidence-region probe), and post-hoc Oracle selection diagnostic on the third 256-pair SECOND confirmation cohort (Building held-out, 8-KiB cap).}",
        r"\label{tab:factorial-oracle}",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.00}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}\toprule",
        r"Cell / Contrast & Mechanism tested & $U_{30}$ / $\Delta U_{30}$ & 95\% CI (pp) & Later F1 / $\Delta$F1 & Held-out FNR & Mean bytes" + end,
        r"\midrule",
        f"Cell A (\\texttt{{absolute\\_change}}) & Absolute ROI + Change & {cells['absolute_change']['u30']:.4f} & -- & {cells['absolute_change']['later_f1']:.4f} & {cells['absolute_change']['later_fnr']:.4f} & {cells['absolute_change']['mean_total_bytes']:.1f}" + end,
        f"Cell B (\\texttt{{absolute\\_current}}) & Absolute ROI + Unknown & {cells['absolute_current']['u30']:.4f} & -- & {cells['absolute_current']['later_f1']:.4f} & {cells['absolute_current']['later_fnr']:.4f} & {cells['absolute_current']['mean_total_bytes']:.1f}" + end,
        f"Cell C (\\texttt{{residual\\_change}}) & Residual ROI + Change (locked) & {cells['residual_change']['u30']:.4f} & -- & {cells['residual_change']['later_f1']:.4f} & {cells['residual_change']['later_fnr']:.4f} & {cells['residual_change']['mean_total_bytes']:.1f}" + end,
        f"Cell D (\\texttt{{residual\\_current}}) & Residual ROI + Unknown (locked) & {cells['residual_current']['u30']:.4f} & -- & {cells['residual_current']['later_f1']:.4f} & {cells['residual_current']['later_fnr']:.4f} & {cells['residual_current']['mean_total_bytes']:.1f}" + end,
        f"TC-PCR Change (4A+4B) & Active-support + 30dB Change & {rep['repaired_residual_change_sel']['u30']:.4f} & -- & {rep['repaired_residual_change_sel']['later_f1']:.4f} & {rep['repaired_residual_change_sel']['later_fnr']:.4f} & {rep['repaired_residual_change_sel']['mean_total_bytes']:.1f}" + end,
        f"TC-PCR Unknown (4A+4B) & Active-support + 30dB + Cal. Unknown & {rep['repaired_residual_unknown_sel']['u30']:.4f} & -- & {rep['repaired_residual_unknown_sel']['later_f1']:.4f} & {rep['repaired_residual_unknown_sel']['later_fnr']:.4f} & {rep['repaired_residual_unknown_sel']['mean_total_bytes']:.1f}" + end,
        f"TC-PCR Unknown (Raw, 4B) & Raw residual + 30dB + Cal. Unknown & {rep['repaired_residual_unknown_raw']['u30']:.4f} & -- & {rep['repaired_residual_unknown_raw']['later_f1']:.4f} & {rep['repaired_residual_unknown_raw']['later_fnr']:.4f} & {rep['repaired_residual_unknown_raw']['mean_total_bytes']:.1f}" + end,
        f"TC-PCR Unknown (Refined ROI, 4C) & Evaluated on refined evidence ROI & {rep['repaired_residual_unknown_sel']['u30']:.4f} & -- & {rep['repaired_residual_unknown_sel_calprobe']['later_f1']:.4f} & {rep['repaired_residual_unknown_sel_calprobe']['later_fnr']:.4f} & {rep['repaired_residual_unknown_sel']['mean_total_bytes']:.1f}" + end,
        f"Post-hoc Oracle Hybrid & Ideal 8-KiB residual or HP & {ora['oracle_hybrid_with_full_hp']['u30']:.4f} & -- & {ora['oracle_hybrid_with_full_hp']['later_f1']:.4f} & {ora['oracle_hybrid_with_full_hp']['later_fnr']:.4f} & {ora['oracle_hybrid_with_full_hp']['mean_total_bytes']:.1f}" + end,
        r"\midrule",
        f"Row effect ($C - A$, locked) & Residual vs Abs. (Change) & {100*c_dict['C_minus_A_residual_in_change']['u30_diff'][0]:+.2f} pp & [{100*c_dict['C_minus_A_residual_in_change']['u30_diff'][1]:+.2f}, {100*c_dict['C_minus_A_residual_in_change']['u30_diff'][2]:+.2f}] & {100*c_dict['C_minus_A_residual_in_change']['later_f1_diff'][0]:+.2f} pp & {100*c_dict['C_minus_A_residual_in_change']['later_fnr_diff'][0]:+.2f} pp & +99.6" + end,
        f"Column effect ($D - C$, locked) & Unknown vs Change (locked) & {100*c_dict['D_minus_C_unknown_in_residual']['u30_diff'][0]:+.2f} pp & [{100*c_dict['D_minus_C_unknown_in_residual']['u30_diff'][1]:+.2f}, {100*c_dict['D_minus_C_unknown_in_residual']['u30_diff'][2]:+.2f}] & {100*c_dict['D_minus_C_unknown_in_residual']['later_f1_diff'][0]:+.2f} pp & {100*c_dict['D_minus_C_unknown_in_residual']['later_fnr_diff'][0]:+.2f} pp & -20.0" + end,
        f"TC-PCR Unknown vs Change & Calibrated Col. effect (4A+4B) & {100*rc_dict['repaired_residual_unknown_sel']['vs_repaired_residual_change_sel']['u30_diff'][0]:+.2f} pp & [{100*rc_dict['repaired_residual_unknown_sel']['vs_repaired_residual_change_sel']['u30_diff'][1]:+.2f}, {100*rc_dict['repaired_residual_unknown_sel']['vs_repaired_residual_change_sel']['u30_diff'][2]:+.2f}] & {100*rc_dict['repaired_residual_unknown_sel']['vs_repaired_residual_change_sel']['later_f1_diff'][0]:+.2f} pp & {100*rc_dict['repaired_residual_unknown_sel']['vs_repaired_residual_change_sel']['later_fnr_diff'][0]:+.2f} pp & -3.2" + end,
        f"TC-PCR Unknown vs Locked $D$ & TC-PCR gain (4A+4B vs Cell D) & {100*rc_dict['repaired_residual_unknown_sel']['vs_residual_current']['u30_diff'][0]:+.2f} pp & [{100*rc_dict['repaired_residual_unknown_sel']['vs_residual_current']['u30_diff'][1]:+.2f}, {100*rc_dict['repaired_residual_unknown_sel']['vs_residual_current']['u30_diff'][2]:+.2f}] & {100*rc_dict['repaired_residual_unknown_sel']['vs_residual_current']['later_f1_diff'][0]:+.2f} pp & {100*rc_dict['repaired_residual_unknown_sel']['vs_residual_current']['later_fnr_diff'][0]:+.2f} pp & +13.6" + end,
        f"TC-PCR Unknown vs HP & $+12.16$ pp (+33.6\\% UMR red.) & {100*rc_dict['repaired_residual_unknown_sel']['vs_hyperprior']['u30_diff'][0]:+.2f} pp & [{100*rc_dict['repaired_residual_unknown_sel']['vs_hyperprior']['u30_diff'][1]:+.2f}, {100*rc_dict['repaired_residual_unknown_sel']['vs_hyperprior']['u30_diff'][2]:+.2f}] & {100*rc_dict['repaired_residual_unknown_sel']['vs_hyperprior']['later_f1_diff'][0]:+.2f} pp & {100*rc_dict['repaired_residual_unknown_sel']['vs_hyperprior']['later_fnr_diff'][0]:+.2f} pp & +345.8" + end,
        f"TC-PCR Unknown vs TCM & Parity vs 4.18 GB TCM & {100*rc_dict['repaired_residual_unknown_sel']['vs_tcm']['u30_diff'][0]:+.2f} pp & [{100*rc_dict['repaired_residual_unknown_sel']['vs_tcm']['u30_diff'][1]:+.2f}, {100*rc_dict['repaired_residual_unknown_sel']['vs_tcm']['u30_diff'][2]:+.2f}] & {100*rc_dict['repaired_residual_unknown_sel']['vs_tcm']['later_f1_diff'][0]:+.2f} pp & {100*rc_dict['repaired_residual_unknown_sel']['vs_tcm']['later_fnr_diff'][0]:+.2f} pp & -50.8" + end,
        f"Refined ROI Probe vs WebP & Region-aligned task gain (4C) & {100*rc_dict['repaired_residual_unknown_sel_calprobe']['vs_full_webp']['u30_diff'][0]:+.2f} pp & [{100*rc_dict['repaired_residual_unknown_sel_calprobe']['vs_full_webp']['u30_diff'][1]:+.2f}, {100*rc_dict['repaired_residual_unknown_sel_calprobe']['vs_full_webp']['u30_diff'][2]:+.2f}] & {100*rc_dict['repaired_residual_unknown_sel_calprobe']['vs_full_webp']['later_f1_diff'][0]:+.2f} pp & {100*rc_dict['repaired_residual_unknown_sel_calprobe']['vs_full_webp']['later_fnr_diff'][0]:+.2f} pp & -209.6" + end,
        r"\bottomrule\end{tabular*}\end{table*}",
    ]
    p_fact_tex = ROOT / "paper/supplement/generated/factorial_oracle_table.tex"
    p_fact_tex.write_text("\n".join(fact_lines) + "\n", encoding="utf-8")

    return [p_main_cost, p_sup_cost, p_lowres_tex, p_fact_tex]


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    lowres_analysis = read_json(ROOT / "results/lowres_reference_sn7/v1/analysis.json")
    png_profile = read_json(ROOT / "results/sender_profile/compressed_archive_v1/summary.json")
    sender_ref = read_json(ROOT / "results/task_evidence/sender_reference_v1/verification.json")
    diag = read_json(ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/diagnostic.json")
    sn7_repaired = read_json(ROOT / "results/sn7_repaired_design/v1/analysis.json")
    sec_repaired = read_json(ROOT / "results/task_evidence/second_repaired_design_v1/analysis.json")

    outputs = []
    outputs.extend(build_figure1_sn7_comm_vs_utility(lowres_analysis, sn7_repaired))
    outputs.extend(build_figure2_storage_vs_utility(lowres_analysis, png_profile, sender_ref, diag, sec_repaired))
    outputs.extend(build_figure3_second_four_cell_and_oracle(diag, sec_repaired))
    outputs.extend(build_tables(lowres_analysis, png_profile, diag, sn7_repaired, sec_repaired))

    sources = [
        ROOT / "results/lowres_reference_sn7/v1/analysis.json",
        ROOT / "results/sender_profile/compressed_archive_v1/summary.json",
        ROOT / "results/task_evidence/sender_reference_v1/verification.json",
        ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/diagnostic.json",
        ROOT / "results/sn7_repaired_design/v1/analysis.json",
        ROOT / "results/task_evidence/second_repaired_design_v1/analysis.json",
        ROOT / "results/tcm_sn7/transport_v1/records.json",
        ROOT / "results/task_evidence/tcm_control_v1/records.json",
    ]
    manifest_path = ROOT / "paper/generated/decision_assets_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "generator_sha256": sha(Path(__file__)),
                "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                "output_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "generated_assets": len(outputs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
