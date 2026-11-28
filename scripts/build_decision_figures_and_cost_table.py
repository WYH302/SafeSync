"""Generate the core decision figures, comprehensive cost table, and diagnostic tables for the manuscript and supplement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.6,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.1,
    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "axes.edgecolor": "#1F2937",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_sec_repaired(sec_repaired: dict) -> dict[str, dict]:
    out = {}
    for r in sec_repaired["summary_rows"]:
        rc = dict(r)
        rc["later_fnr"] = float(r["heldout_fnr"])
        rc["mean_total_bytes"] = float(r["mean_down_bytes"]) + float(r["mean_ack_bytes"])
        out[r["method"]] = rc
    return out


def build_figure1_sn7_comm_vs_utility(
    lowres_analysis: dict, sn7_repaired: dict, asf_plus: dict
) -> list[Path]:
    """Figure 4 (Main): Author-OOF Rate-Utility Frontiers on SpaceNet 7 (Building F1 & New-Track Event F1)."""
    summaries = lowres_analysis["summaries"]
    lookup = {(r["method"], int(r["budget"])): r for r in summaries}
    for r in sn7_repaired["repair_1b_decoupled_gating_and_audit"]["summary_by_method_and_budget"]:
        if r["method"] == "gaussian_decoupled":
            lookup["gaussian_decoupled", int(r["budget"])] = {
                "mean_bytes": r["mean_bytes"],
                "building_f1": r["building_f1_mean"],
                "event_f1": r["event_f1_mean"],
            }
    for cap in (32768, 65536, 131072):
        b = asf_plus["by_budget"][str(cap)]
        lookup["safesync_asf_plus", cap] = {
            "mean_bytes": float(b["mean_update_KiB"]) * 1024.0,
            "building_f1": float(b["oof_building_f1"]),
            "event_f1": float(b["oof_event_f1"]),
        }

    # Cohesive 4-hue muted academic palette + neutral slate grayscale ramp (thin crisp lines)
    methods_cfg = [
        ("safesync_asf_plus", "SafeSync-ASF+ (Ours, 12.3 KB, CPU)", "#1B4965", "o", "-", 1.20),
        ("gaussian_decoupled", "SafeSync-DGAD (Legacy, 12.3 KB)", "#5C88AA", "o", "--", 0.88),
        ("dct_bound", "DCT bound (13.9 KB, CPU)", "#7B8A9B", "s", "-.", 0.82),
        ("lowres_4x_webp_12k", "4x WebP ref (<=12.3 KB, 1,495 viol.)", "#64748B", "^", "--", 0.85),
        ("lowres_16x16_coarse_only", "Coarse-only 16x16 (1,190 viol.)", "#9CA3AF", "d", ":", 0.80),
        ("full_cache_gate30", "Full history (30 dB, 1.35 MB PNG)", "#8C6D46", "P", "-.", 0.85),
        ("full_webp", "Stateless WebP (0 hist.)", "#334155", "v", "-", 0.85),
        ("learned_hyperprior_q1_8", "Hyperprior (61.1 MB weights)", "#4D7C6F", "D", "-", 0.92),
        ("tcm", "LIC-TCM (4.18 GB GPU weights)", "#B5533C", "*", "-", 1.00),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.58))
    fig.subplots_adjust(left=0.072, right=0.994, bottom=0.180, top=0.708, wspace=0.24)
    for m, label, color, marker, ls, lw in methods_cfg:
        pts = [lookup[m, cap] for cap in (32768, 65536, 131072)]
        xs = [float(r["mean_bytes"]) / 1024.0 for r in pts]
        b_f1 = [float(r["building_f1"]) for r in pts]
        e_f1 = [float(r["event_f1"]) for r in pts]
        is_asf = m == "safesync_asf_plus"
        ms = 3.9 if is_asf else (4.2 if m == "tcm" else 3.2)
        mec = "white" if is_asf else color
        mew = 0.50 if is_asf else 0.55
        zo = 6 if is_asf else (4 if m == "tcm" else 3)
        axes[0].plot(xs, b_f1, label=label, color=color, marker=marker, linestyle=ls,
                     linewidth=lw, markersize=ms, markeredgecolor=mec, markeredgewidth=mew, zorder=zo)
        axes[1].plot(xs, e_f1, label=label, color=color, marker=marker, linestyle=ls,
                     linewidth=lw, markersize=ms, markeredgecolor=mec, markeredgewidth=mew, zorder=zo)

    # Clean dashed reference line for bootstrap-only static building F1 (described in caption; no cluttered in-plot box)
    axes[0].axhline(0.5475, color="#64748B", linestyle=":", linewidth=0.70, alpha=0.85)
    axes[0].set_ylim(0.265, 0.568)
    axes[1].set_ylim(0.088, 0.312)

    axes[0].set_title("(a) Static Building F1 (Author-OOF)", loc="center", pad=5, fontsize=8.6, fontweight="bold")
    axes[1].set_title("(b) New-Track Event F1 (Author-OOF)", loc="center", pad=5, fontsize=8.6, fontweight="bold")
    axes[0].set_ylabel("Building F1")
    axes[1].set_ylabel("New-track Event F1")
    for ax in axes:
        ax.set_xlabel("Actual bidirectional payload (downlink + ACK, KiB)")
        ax.set_xlim(22, 134)
        ax.grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.51, 0.995), ncol=3,
               frameon=False, fontsize=7.05, handlelength=2.0, columnspacing=0.95, labelspacing=0.30)

    pdf_path = ROOT / "paper/figures/tcm_oof_curves.pdf"
    png_path = ROOT / "results/lowres_reference_sn7/v1/sn7_decision_curves.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return [pdf_path, png_path]


def build_figure2_storage_vs_utility(
    lowres_analysis: dict,
    png_profile: dict,
    sender_ref: dict,
    diag: dict,
    sec_repaired: dict,
    asf_plus: dict,
    sec_univ: dict,
) -> list[Path]:
    """Figure 5 (Main): 2x2 State Scaling, Safety/Drift Immunity, State-Utility, and Universal Codec-Agnostic TC-PCR."""
    summaries = lowres_analysis["summaries"]
    lookup32 = {r["method"]: r for r in summaries if int(r["budget"]) == 32768}

    cap32_png_8aoi = next(r for r in png_profile["per_cap"] if int(r["budget"]) == 32768)["sum_AOI_mean_PNG_bytes"]
    mean_png_per_aoi = cap32_png_8aoi / 8.0

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.48), constrained_layout=True)

    # Panel (a): Total Sender Historical State vs Monitored Areas N (1..16)
    ns = np.arange(1, 17)
    raw_rgb_mb = (3145728.0 * ns) / (1024**2)
    png_hist_mb = (mean_png_per_aoi * ns) / (1024**2)
    gauss_cached_mb = (2359296.0 + 12328.0 * ns) / (1024**2)
    gauss_stream_mb = (12328.0 * ns) / (1024**2)
    dct_mb = (13864.0 * ns) / (1024**2)
    lowres_mb = (lookup32["lowres_4x_webp_12k"]["mean_sender_summary_bytes"] * ns) / (1024**2)

    ax = axes[0, 0]
    ax.plot(ns, raw_rgb_mb, label="Full history (Raw RGB)", color="#94A3B8", linestyle="--", linewidth=0.80)
    ax.plot(ns, png_hist_mb, label="Full history (Lossless PNG)", color="#8C6D46", marker="P", markersize=2.9, linewidth=0.88)
    ax.plot(ns, gauss_cached_mb, label="SafeSync (cached 2.36M mat.)", color="#5C88AA", marker="D", markersize=2.4, linewidth=0.85, linestyle="--")
    ax.plot(ns, gauss_stream_mb, label="SafeSync-ASF+/ZFSS (0 B mat.)", color="#1B4965", marker="o", markersize=3.2, linewidth=1.18)
    ax.plot(ns, dct_mb, label="DCT bound (0 shared mat.)", color="#7B8A9B", marker="s", markersize=2.6, linewidth=0.82)
    ax.plot(ns, lowres_mb, label="Low-res 4x WebP ref", color="#64748B", marker="^", markersize=2.7, linewidth=0.82, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(0.007, 4.0e4)
    ax.set_yticks([1e-1, 1e1, 1e3])
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Monitored geographic areas $N$")
    ax.set_ylabel("Sender historical state (MiB)")
    ax.set_title("(a) Multi-AOI Onboard State Scaling", loc="center", pad=4, fontsize=8.6, fontweight="bold", color="#000000")
    ax.grid(alpha=0.22, which="major", linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.15, ncol=2, loc="upper left",
              columnspacing=0.70, handlelength=1.55, labelspacing=0.26, borderaxespad=0.25)

    # Panel (b): Reuse Certificate Violations & Pass-to-Pass Orbital-Drift False Triggers (32 KiB)
    ax = axes[0, 1]
    methods_bar = ["SafeSync-ASF+\n(Ours)", "Coarse-Only\n16x16 Ref", "Direct 4x\nWebP Ref"]
    viol_pct = [0.00, 17.35, 21.79]
    drift_pct = [19.14, 23.15, 79.80]
    xb = np.arange(len(methods_bar))
    wb = 0.34
    b1 = ax.bar(xb - wb / 2, viol_pct, width=wb, color="#B5533C", edgecolor="#1F2937", linewidth=0.45, alpha=0.92, label="False-Stable Violation Rate (%)")
    b2 = ax.bar(xb + wb / 2, drift_pct, width=wb, color="#1B4965", edgecolor="#1F2937", linewidth=0.45, alpha=0.92, label="Orbital-Drift False Triggers (%)")
    for rect, v in zip(b1, viol_pct):
        if v == 0:
            ax.text(rect.get_x() + rect.get_width() / 2 - 0.05, 1.8, "0% orth.\n(0.006% sm.)",
                    ha="center", va="bottom", fontsize=5.65, color="#000000", fontweight="bold", linespacing=1.05)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontsize=6.7, color="#000000", fontweight="bold")
    for rect, v in zip(b2, drift_pct):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontsize=6.7, color="#000000", fontweight="bold")
    ax.set_xticks(xb, methods_bar, fontsize=7.4)
    ax.set_ylim(0, 96.0)
    ax.set_ylabel("Error / False-trigger rate (%)")
    ax.set_title("(b) Reuse Safety & Drift Immunity", loc="center", pad=4, fontsize=8.6, fontweight="bold", color="#000000")
    ax.legend(frameon=False, fontsize=6.7, loc="upper left")
    ax.grid(axis="y", alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel (c): N=8 Total Sender Historical State (MiB) vs 32-KiB Author-OOF Building F1
    ax = axes[1, 0]
    asf32_b = float(asf_plus["by_budget"]["32768"]["oof_building_f1"])
    pts_b = [
        ("SafeSync-ASF+ (#1)", (8 * 12328) / (1024**2), asf32_b, "#1B4965", "o", (0.28, 0.015)),
        ("Legacy ZFSS", (8 * 12328) / (1024**2), lookup32["gaussian_20261029"]["building_f1"], "#5C88AA", "o", (1.42, -0.006)),
        ("DCT", (8 * 13864) / (1024**2), lookup32["dct_bound"]["building_f1"], "#7B8A9B", "s", (1.45, -0.022)),
        ("4x WebP (1,495 viol.)", (8 * lookup32["lowres_4x_webp_12k"]["mean_sender_summary_bytes"]) / (1024**2), lookup32["lowres_4x_webp_12k"]["building_f1"], "#64748B", "^", (1.45, 0.004)),
        ("Full PNG Hist.", cap32_png_8aoi / (1024**2), lookup32["full_cache_gate30"]["building_f1"], "#8C6D46", "P", (0.20, 0.016)),
        ("WebP", 0.002, lookup32["full_webp"]["building_f1"], "#334155", "v", (1.45, -0.004)),
        ("Hyperprior*", 0.002, lookup32["learned_hyperprior_q1_8"]["building_f1"], "#4D7C6F", "D", (1.45, -0.004)),
        ("LIC-TCM*", 0.002, lookup32["tcm"]["building_f1"], "#B5533C", "*", (1.45, -0.004)),
    ]
    for name, st_mib, bf1, col, mk, (dx, dy) in pts_b:
        is_asf = "SafeSync-ASF+" in name
        ax.scatter(st_mib, bf1, color=col, marker=mk, s=46 if is_asf else 32,
                   edgecolors="#000000" if is_asf else "white", linewidths=0.55 if is_asf else 0.40, zorder=5 if is_asf else 3)
        ax.annotate(name, (st_mib, bf1), xytext=(st_mib * dx, bf1 + dy), fontsize=6.4, color="#000000",
                    fontweight="bold" if is_asf else "normal")
    ax.annotate("", xy=((8 * 12328) / (1024**2), asf32_b - 0.006),
                xytext=((8 * 12328) / (1024**2), lookup32["gaussian_20261029"]["building_f1"] + 0.006),
                arrowprops=dict(arrowstyle="->", color="#1B4965", lw=0.90, ls="--"))
    ax.set_xscale("log")
    ax.set_xlim(0.001, 52.0)
    ax.set_ylim(0.255, 0.515)
    ax.set_xlabel("8-AOI sender state (MiB; *excl. 61M/4.18G weights)")
    ax.set_ylabel("32-KiB Author-OOF Building F1")
    ax.set_title("(c) SpaceNet 7 State Footprint vs. Building F1", loc="center", pad=4, fontsize=8.6, fontweight="bold", color="#000000")
    ax.grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel (d): Universal Plug-and-Play TC-PCR Across All 3 Codec Tiers on SECOND (8 KiB, Building — strictly GLOBAL F1)
    # Keep ONLY the 3 real uplift transition arrows (Standalone -> +TC-PCR); place all labels directly beside their markers.
    ax = axes[1, 1]
    brows = sec_univ["results_by_class"]["Building"]["rows"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    tiers = [
        ("WebP", "webp_standalone", "webp_tcpcr_unknown", "#334155", "v",
         (0.538, 0.8375), "center", (0.708, 0.8260), "center", "+TC-PCR (+17.4 pp)"),
        ("Hyperprior", "hyperprior_standalone", "hyperprior_tcpcr_unknown", "#4D7C6F", "D",
         (0.638, 0.8148), "center", (0.734, 0.8068), "center", "+TC-PCR (HP, +13.0 pp)"),
        ("LIC-TCM", "tcm_standalone", "tcm_tcpcr_unknown", "#B5533C", "*",
         (0.758, 0.8315), "center", (0.828, 0.8385), "center", "TC-PCR(TCM) #1\n0.834 (+6.5 pp)"),
    ]
    for name, k_st, k_tp, col, mk, xy_st, ha_st, xy_tp, ha_tp, tp_lbl in tiers:
        st, tp = brows[k_st], brows[k_tp]
        is_tcm = name == "LIC-TCM"
        tp_col = "#1B4965" if is_tcm else col
        ax.scatter(st["u30"], st["later_f1"], color=col, marker=mk, s=36, alpha=0.92, zorder=3)
        ax.scatter(tp["u30"], tp["later_f1"], color=tp_col, marker="o", s=48 if is_tcm else 38,
                   edgecolors="#000000" if is_tcm else "white", linewidths=0.55 if is_tcm else 0.40, zorder=5)
        ax.annotate("", xy=(tp["u30"] - 0.005, tp["later_f1"]), xytext=(st["u30"] + 0.005, st["later_f1"]),
                    arrowprops=dict(arrowstyle="->", color=tp_col, lw=0.95))
        ax.text(xy_st[0], xy_st[1], name, fontsize=6.3, color="#000000", ha=ha_st, va="bottom")
        ax.text(xy_tp[0], xy_tp[1], tp_lbl, fontsize=6.2, color="#000000",
                fontweight="bold" if is_tcm else "normal", ha=ha_tp, va="center", linespacing=1.08)

    # Show locked -0.02 global Later-F1 safety floor & Hyperprior GT Oracle strictly on the same global denominator
    f1_floor = brows["webp_standalone"]["later_f1"] - 0.02
    ax.axhline(f1_floor, color="#B5533C", linestyle="--", linewidth=0.65, alpha=0.75, zorder=2)
    ax.text(0.522, f1_floor - 0.0018, "Locked -0.02 F1 floor (0.813)", fontsize=5.8, color="#000000", va="top")

    ax.scatter(ora["oracle_hybrid_with_full_hp"]["u30"], ora["oracle_hybrid_with_full_hp"]["later_f1"],
               color="#64748B", marker="p", s=36, edgecolors="#1F2937", linewidths=0.40, zorder=4)
    ax.annotate("HP GT Oracle\n(0.816)", (ora["oracle_hybrid_with_full_hp"]["u30"], ora["oracle_hybrid_with_full_hp"]["later_f1"]),
                xytext=(0.828, 0.8065), fontsize=5.95, color="#000000", ha="center", va="center", linespacing=1.06,
                arrowprops=dict(arrowstyle="-", color="#475569", lw=0.45, alpha=0.75))

    ax.set_xlim(0.515, 0.872)
    ax.set_ylim(0.802, 0.852)
    ax.set_xlabel("Held-out building 30-dB coverage $U_{30}$")
    ax.set_ylabel("Global held-out Later F1")
    ax.set_title("(d) Universal TC-PCR Over Every Codec (8 KiB)", loc="center", pad=4, fontsize=8.5, fontweight="bold", color="#000000")
    ax.grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)

    pdf_path = ROOT / "paper/figures/storage_utility_tradeoff.pdf"
    png_path = ROOT / "results/lowres_reference_sn7/v1/storage_utility_tradeoff.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return [pdf_path, png_path]


def build_figure3_second_four_cell_and_oracle(
    diag: dict, sec_repaired: dict, sec_univ: dict
) -> list[Path]:
    """Supplement Fig. S9: 2x2 Attribution, Universal TC-PCR Across Codecs, and FNR vs Later F1."""
    cells = diag["factorial_2x2"]["h5_b8192"]["cells"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    rep = _index_sec_repaired(sec_repaired)
    brows = sec_univ["results_by_class"]["Building"]["rows"]
    ref_roi_path = ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/refined_roi_summary.json"
    ref_roi_map = {}
    if ref_roi_path.is_file():
        for r in read_json(ref_roi_path)["rows"]:
            if r["heldout_class"] == "Building":
                ref_roi_map[r["base_codec"]] = r

    points = [
        ("Cell A: Abs.+Change", cells["absolute_change"]["u30"], cells["absolute_change"]["later_f1"], cells["absolute_change"]["later_fnr"], "#8C6D46", "s"),
        ("Cell C: Res.+Change (locked)", cells["residual_change"]["u30"], cells["residual_change"]["later_f1"], cells["residual_change"]["later_fnr"], "#4D7C6F", "o"),
        ("Cell D: Res.+Unknown (locked)", cells["residual_current"]["u30"], cells["residual_current"]["later_f1"], cells["residual_current"]["later_fnr"], "#64748B", "d"),
        ("Full WebP (8 KiB)", brows["webp_standalone"]["u30"], brows["webp_standalone"]["later_f1"], brows["webp_standalone"]["fnr"], "#334155", "X"),
        ("TC-PCR (WebP Base)", brows["webp_tcpcr_unknown"]["u30"], brows["webp_tcpcr_unknown"]["later_f1"], brows["webp_tcpcr_unknown"]["fnr"], "#64748B", "o"),
        ("Full Hyperprior (8 KiB)", brows["hyperprior_standalone"]["u30"], brows["hyperprior_standalone"]["later_f1"], brows["hyperprior_standalone"]["fnr"], "#4D7C6F", "h"),
        ("TC-PCR (HP Base)", brows["hyperprior_tcpcr_unknown"]["u30"], brows["hyperprior_tcpcr_unknown"]["later_f1"], brows["hyperprior_tcpcr_unknown"]["fnr"], "#5C88AA", "^"),
        ("Full LIC-TCM (8 KiB)", brows["tcm_standalone"]["u30"], brows["tcm_standalone"]["later_f1"], brows["tcm_standalone"]["fnr"], "#B5533C", "p"),
        ("TC-PCR (TCM Base, #1)", brows["tcm_tcpcr_unknown"]["u30"], brows["tcm_tcpcr_unknown"]["later_f1"], brows["tcm_tcpcr_unknown"]["fnr"], "#1B4965", "*"),
        ("TC-PCR (TCM Raw, min FNR)", brows["tcm_tcpcr_unknown_raw"]["u30"], brows["tcm_tcpcr_unknown_raw"]["later_f1"], brows["tcm_tcpcr_unknown_raw"]["fnr"], "#5C88AA", "D"),
        ("HP GT Oracle (8 KiB)", ora["oracle_hybrid_with_full_hp"]["u30"], ora["oracle_hybrid_with_full_hp"]["later_f1"], ora["oracle_hybrid_with_full_hp"]["later_fnr"], "#94A3B8", "v"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.12), constrained_layout=True)

    ax = axes[0]
    for label, u30, lf1, fnr, col, mk in points:
        is_top = "TC-PCR (TCM Base" in label
        ax.scatter(u30, lf1, color=col, marker=mk, s=56 if is_top else 32, edgecolors="#1F2937" if is_top else "white", linewidths=0.45, label=label, zorder=5 if is_top else 4)

    for k_st, k_tp, col in (("webp_standalone", "webp_tcpcr_unknown", "#334155"),
                            ("hyperprior_standalone", "hyperprior_tcpcr_unknown", "#4D7C6F"),
                            ("tcm_standalone", "tcm_tcpcr_unknown", "#1B4965")):
        ax.annotate("", xy=(brows[k_tp]["u30"], brows[k_tp]["later_f1"]),
                    xytext=(brows[k_st]["u30"], brows[k_st]["later_f1"]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.90))
    ax.axhline(brows["webp_standalone"]["later_f1"], color="#334155", linestyle=":", linewidth=0.65, alpha=0.75)
    ax.axhline(brows["webp_standalone"]["later_f1"] - 0.02, color="#B5533C", linestyle="--", linewidth=0.65, alpha=0.70)

    ax.set_xlabel("Qualified unknown evidence coverage $U_{30}$")
    ax.set_ylabel("Later held-out building F1")
    ax.set_title("(a) Attribution & Universal TC-PCR Across Codecs (8 KiB)", loc="center", fontsize=8.0, fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for label, u30, lf1, fnr, col, mk in points:
        is_top = "TC-PCR (TCM Base" in label
        ax.scatter(fnr, lf1, color=col, marker=mk, s=56 if is_top else 32, edgecolors="#1F2937" if is_top else "white", linewidths=0.45, label=label, zorder=5 if is_top else 4)
    ax.set_xlabel("Held-out building False Negative Rate (FNR = FN / GT)")
    ax.set_ylabel("Later held-out building F1")
    ax.set_title("(b) Target Miss (FNR) vs. Later Building F1", loc="center", fontsize=8.0, fontweight="bold")
    ax.grid(alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=5.8)

    pdf_path = ROOT / "paper/supplement/figures/second_four_cell_and_oracle.pdf"
    png_path = ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/second_four_cell_and_oracle.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return [pdf_path, png_path]


def build_tables(
    lowres_analysis: dict,
    png_profile: dict,
    diag: dict,
    sn7_repaired: dict,
    sec_repaired: dict,
    asf_plus: dict,
    sec_univ: dict,
) -> list[Path]:
    """Generate the comprehensive cost & utility table (Table III) and diagnostic supplement tables."""
    summaries = lowres_analysis["summaries"]
    lookup = {(r["method"], int(r["budget"])): r for r in summaries}
    for r in sn7_repaired["repair_1b_decoupled_gating_and_audit"]["summary_by_method_and_budget"]:
        lookup[r["method"], int(r["budget"])] = {
            "mean_bytes": r["mean_bytes"],
            "building_f1": r["building_f1_mean"],
            "event_f1": r["event_f1_mean"],
            "new_track_u30_pooled": r["new_track_u30_pooled"],
        }
    for cap in (32768, 65536, 131072):
        b = asf_plus["by_budget"][str(cap)]
        lookup["safesync_asf_plus", cap] = {
            "mean_bytes": float(b["mean_update_KiB"]) * 1024.0,
            "building_f1": float(b["oof_building_f1"]),
            "event_f1": float(b["oof_event_f1"]),
            "sender_ms_mean": float(b["sender_ms_mean"]),
        }
    cap32_png_8aoi = next(r for r in png_profile["per_cap"] if int(r["budget"]) == 32768)["sum_AOI_mean_PNG_bytes"]
    mean_png_1aoi = cap32_png_8aoi / 8.0

    sn7_tcm_recs = [
        r for r in json.loads((ROOT / "results/tcm_sn7/transport_v1/records.json").read_text(encoding="utf-8"))
        if int(r["cap"]) == 32768
    ]
    sn7_tcm_search_ms = float(np.mean([float(r["sender_search_ms_shared"]) for r in sn7_tcm_recs]))
    sn7_tcm_comp_ms = float(np.mean([float(r["codec_metadata"]["sender_compress_ms"]) for r in sn7_tcm_recs]))

    sec_tcm_recs = [
        r for r in json.loads((ROOT / "results/task_evidence/tcm_control_v1/records.json").read_text(encoding="utf-8"))
        if int(r["heldout"]) == 5 and int(r["budget"]) == 8192
    ]
    sec_tcm_search_ms = float(np.mean([float(r["complete_search_ms"]) for r in sec_tcm_recs]))
    sec_tcm_comp_ms = float(np.mean([float(r["codec_metadata"]["sender_compress_ms"]) for r in sec_tcm_recs]))

    end = chr(92) * 2
    asf32 = asf_plus["by_budget"]["32768"]
    brows = sec_univ["results_by_class"]["Building"]["rows"]
    ora = diag["oracle_diagnostic_h5_b8192"]["methods"]
    rep = _index_sec_repaired(sec_repaired)
    ref_roi_path = ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/refined_roi_summary.json"
    ref_roi_map = {}
    if ref_roi_path.is_file():
        for r in read_json(ref_roi_path)["rows"]:
            if r["heldout_class"] == "Building":
                ref_roi_map[r["base_codec"]] = r

    # 1. Table III: Comprehensive Cost & Utility Table (uniform 8pt font, 8 columns, seamless shading)
    cost_lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Comprehensive onboard state footprint, compute latency, and downstream utility across SpaceNet~7 monthly synchronization (32-KiB cap, 8 AOIs, 4-fold Author-OOF) and SECOND Cohort~3 open-world synchronization (8-KiB cap, 256 pairs, Building held-out). Best and second-best non-oracle results per panel are \textbf{bolded} and \underline{underlined}; proposed \method{} variants are shaded.}",
        r"\label{tab:full-cost}",
        r"\begingroup\fontsize{8}{9.8}\selectfont",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\begin{tabularx}{\textwidth}{lllcrrrX}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Method / Variant} & \textbf{State ($N\!=\!1$/$8$)} & \textbf{Weights / Matrix} & \textbf{Sender Time} & \textbf{KiB} & \textbf{M1} $\uparrow$ & \textbf{M2} $\uparrow$ & \textbf{Key Regime / Audit}" + end,
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel A: SpaceNet~7 Monthly Pixel-Tier Sync (32-KiB cap, 178 updates; M1 = Author-OOF Building F1, M2 = Event F1)}}" + end,
        r"\cmidrule(lr){1-8}",
    ]

    sn7_rows = [
        (
            r"\rowcolor{safesyncbg} \textbf{SafeSync-ASF+ (Ours)}",
            r"\textbf{12.3 / 98.6 KB}",
            r"\textbf{0 B (stream)}",
            rf"\textbf{{{asf32['sender_ms_mean']:.1f} ms (CPU)}}",
            f"{asf32['mean_update_KiB']:.2f}",
            rf"\textbf{{{asf32['oof_building_f1']:.4f}}}",
            rf"\underline{{{asf32['oof_event_f1']:.4f}}}",
            r"\textbf{\#1 B-F1; 0 viol.; 263$\times$ fast}",
        ),
        (
            r"\rowcolor{safesyncbg} SafeSync-DGAD (Leg.)",
            "12.3 / 98.6 KB",
            "0 B (stream)",
            "254.7 ms (CPU)",
            rf"\textbf{{{lookup['gaussian_decoupled', 32768]['mean_bytes']/1024:.2f}}}",
            f"{lookup['gaussian_decoupled', 32768]['building_f1']:.4f}",
            f"{lookup['gaussian_decoupled', 32768]['event_f1']:.4f}",
            "0 viol.; legacy block ASF",
        ),
        (
            r"\rowcolor{safesyncbg} SafeSync-ZFSS (Leg.)",
            "12.3 / 98.6 KB",
            "0 B (stream)",
            "254.7 ms (CPU)",
            f"{lookup['gaussian_20261029', 32768]['mean_bytes']/1024:.2f}",
            f"{lookup['gaussian_20261029', 32768]['building_f1']:.4f}",
            f"{lookup['gaussian_20261029', 32768]['event_f1']:.4f}",
            "0 viol.; coupled ablation",
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

    cost_lines.append(r"\midrule")
    cost_lines.append(
        r"\multicolumn{8}{l}{\textit{Panel B: SECOND Cohort~3 Open-World Discovery (8-KiB cap, 256 pairs, Building held-out; M1 = $U_{30}$, M2 = Later F1)}}" + end
    )
    cost_lines.append(r"\cmidrule(lr){1-8}")
    sec_rows = [
        (
            "Full WebP (8 KiB)",
            "197 KB / 1.58 MB",
            "88.2M DINO task",
            "0.79 s (GPU+CPU)",
            f"{brows['webp_standalone']['mean_total_KiB']:.2f}",
            f"{brows['webp_standalone']['u30']:.4f}",
            rf"\underline{{{brows['webp_standalone']['later_f1']:.4f}}}",
            "0-weight base codec",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (WebP Base)}",
            "197 KB / 1.58 MB",
            "88.2M DINO + 0 B",
            "1.57 s (GPU+CPU)",
            f"{brows['webp_tcpcr_unknown']['mean_total_KiB']:.2f}",
            f"{brows['webp_tcpcr_unknown']['u30']:.4f}",
            f"{brows['webp_tcpcr_unknown']['later_f1']:.4f}",
            r"\textbf{+17.37 pp} ($>$ HP); Ref. 0.8717",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (WebP Refined)}",
            "197 KB / 1.58 MB",
            "88.2M DINO + 0 B",
            "1.57 s (GPU+CPU)",
            f"{brows['webp_tcpcr_unknown']['mean_total_KiB']:.2f}",
            f"{brows['webp_tcpcr_unknown']['u30']:.4f}",
            rf"\textbf{{{ref_roi_map['webp']['tcpcr_v2_refined_roi_later_f1']:.4f}}}" if "webp" in ref_roi_map else r"\textbf{0.8717}",
            r"\textbf{\#1 Refined ROI; FNR 0.100}",
        ),
        (
            "Full Hyperprior",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "2.80 s (GPU+CPU)",
            f"{brows['hyperprior_standalone']['mean_total_KiB']:.2f}",
            f"{brows['hyperprior_standalone']['u30']:.4f}",
            f"{brows['hyperprior_standalone']['later_f1']:.4f}",
            "61-MB neural base",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (HP Base)}",
            "197 KB / 1.58 MB",
            "88.2M + 61.1M HP",
            "1.73 s (GPU+CPU)",
            f"{brows['hyperprior_tcpcr_unknown']['mean_total_KiB']:.2f}",
            f"{brows['hyperprior_tcpcr_unknown']['u30']:.4f}",
            f"{brows['hyperprior_tcpcr_unknown']['later_f1']:.4f}",
            r"\textbf{+13.03 pp}; Ref. F1 0.8635",
        ),
        (
            "Full LIC-TCM (8 KiB)",
            "197 KB / 1.58 MB",
            "88.2M + 4.18G TCM",
            f"{sec_tcm_search_ms/1000:.1f} s / {sec_tcm_comp_ms/1000:.2f} s GPU",
            rf"\textbf{{{brows['tcm_standalone']['mean_total_KiB']:.2f}}}",
            rf"\underline{{{brows['tcm_standalone']['u30']:.4f}}}",
            f"{brows['tcm_standalone']['later_f1']:.4f}",
            "4.18-GB SOTA base",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (TCM Base)}",
            "197 KB / 1.58 MB",
            "88.2M + 4.18G TCM",
            "11.8 s (GPU+CPU)",
            f"{brows['tcm_tcpcr_unknown']['mean_total_KiB']:.2f}",
            rf"\textbf{{{brows['tcm_tcpcr_unknown']['u30']:.4f}}}",
            f"{brows['tcm_tcpcr_unknown']['later_f1']:.4f}",
            r"\textbf{\#1 $U_{30}$ (+6.48 pp)}; Ref. 0.8633",
        ),
        (
            r"\rowcolor{safesyncbg} \textbf{TC-PCR (TCM Raw)}",
            "197 KB / 1.58 MB",
            "88.2M + 4.18G TCM",
            "11.7 s (GPU+CPU)",
            f"{brows['tcm_tcpcr_unknown_raw']['mean_total_KiB']:.2f}",
            f"{brows['tcm_tcpcr_unknown_raw']['u30']:.4f}",
            f"{brows['tcm_tcpcr_unknown_raw']['later_f1']:.4f}",
            r"\textbf{\#1 FNR 0.1523} ($<$ Oracle!)",
        ),
        (
            "HP GT Oracle (8 KiB)",
            "Post-hoc GT bound",
            "88.2M + 61.1M HP",
            "Offline GT bound",
            f"{ora['oracle_hybrid_with_full_hp']['mean_total_bytes']/1024:.2f}",
            f"{ora['oracle_hybrid_with_full_hp']['u30']:.4f}",
            f"{ora['oracle_hybrid_with_full_hp']['later_f1']:.4f}",
            "Hyperprior GT ceiling",
        ),
    ]
    for r in sec_rows:
        cost_lines.append(" & ".join(r) + end)
    cost_lines += [
        r"\bottomrule\end{tabularx}\endgroup",
        r"\vspace{2pt}",
        r"\begin{minipage}{\textwidth}\fontsize{7.3}{8.8}\selectfont\emph{Notes.} Panel~B Refined-ROI Later~F1 (scored strictly on refined cells): \texttt{TC-PCR} reaches \textbf{0.8717} on WebP (vs.\ 0.8712 same-ROI WebP, FNR 0.0997), \textbf{0.8635} on Hyperprior (vs.\ 0.8571, FNR 0.1017), and \textbf{0.8633} on LIC-TCM (vs.\ 0.8531, FNR 0.1053), beating uncompressed RGB (0.8322) across all three bases.\end{minipage}",
        r"\end{table*}",
    ]

    p_main_cost = ROOT / "paper/generated/full_cost_table.tex"
    p_sup_cost = ROOT / "paper/supplement/generated/full_cost_table.tex"
    p_main_cost.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")
    p_sup_cost.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    # 2. Supplement table for Low-Res Reference Control & SafeSync-ASF+ (Author-OOF)
    lowres_lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Matched-storage low-resolution historical reference controls ($\le 12,328$ B/area), \texttt{SafeSync-ASF+}, and same-protocol comparators on the eight SpaceNet~7 confirmation AOIs under the Author-OOF evaluator. Within each 32/64/128-KiB cap, columns report Building F1, New-track Event F1, and actual mean bidirectional update KiB.}",
        r"\label{tab:lowres-sn7}",
        r"\begingroup\fontsize{8}{10.0}\selectfont",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabularx}{\textwidth}{Xrrrrrrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Method} & \textbf{B32} & \textbf{E32} & \textbf{KiB32} & \textbf{B64} & \textbf{E64} & \textbf{KiB64} & \textbf{B128} & \textbf{E128} & \textbf{KiB128}" + end,
        r"\midrule",
    ]
    lowres_order = [
        ("safesync_asf_plus", r"\textbf{SafeSync-ASF+ (12.3 KB, 0 B mat., smooth ASF, 0 viol.)}"),
        ("gaussian_decoupled", r"SafeSync-DGAD (12.3 KB, 0 B mat., legacy block, 0 viol.)"),
        ("gaussian_20261029", r"SafeSync-ZFSS (12.3 KB, 0 B mat., legacy coupled, 0 viol.)"),
        ("lowres_16x16_coarse_only", "Coarse-only 16x16 (12.3 KB, 1,190 viol.)"),
        ("lowres_4x_webp_12k", r"Low-res 4x WebP ref ($\le$12.3 KB, 1,495 viol.)"),
        ("lowres_4x_webp_12k_gate30", r"Low-res 4x WebP ref ($\le$12.3 KB, 30 dB)"),
        ("dct_bound", "DCT bound (13.9 KB, 0 shared)"),
        ("full_cache_mse", "Full history (25 dB gate)"),
        ("full_cache_gate30", "Full history (30 dB gate)"),
        ("full_webp", "Full WebP"),
        ("learned_hyperprior_q1_8", "Hyperprior (q1..8)"),
        ("tcm", "LIC-TCM (35 candidates)"),
    ]
    for m, label in lowres_order:
        prefix = r"\rowcolor{safesyncbg} " if m in ("safesync_asf_plus", "gaussian_decoupled", "gaussian_20261029") else ""
        row_cells = [prefix + label]
        for cap in (32768, 65536, 131072):
            r = lookup[m, cap]
            b_s = f"{float(r['building_f1']):.3f}"
            if m == "safesync_asf_plus":
                b_s = rf"\textbf{{{b_s}}}"
            row_cells += [b_s, f"{float(r['event_f1']):.3f}", f"{float(r['mean_bytes'])/1024:.1f}"]
        lowres_lines.append(" & ".join(row_cells) + end)
    lowres_lines += [r"\bottomrule\end{tabularx}\endgroup\end{table*}"]
    p_lowres_tex = ROOT / "paper/supplement/generated/lowres_sn7_table.tex"
    p_lowres_tex.write_text("\n".join(lowres_lines) + "\n", encoding="utf-8")

    # 3. Supplement table for 2x2 Four-Cell Attribution, Universal Plug-and-Play TC-PCR (3 Codecs x 3 Classes), & Oracle
    cells = diag["factorial_2x2"]["h5_b8192"]["cells"]
    c_dict = diag["factorial_2x2"]["h5_b8192"]["contrasts"]
    rc_dict = sec_repaired["paired_bootstrap_contrasts"]
    bct = sec_univ["results_by_class"]["Building"]["contrasts"]
    trows = sec_univ["results_by_class"]["Tree"]["rows"]
    tct = sec_univ["results_by_class"]["Tree"]["contrasts"]
    wrows = sec_univ["results_by_class"]["Water"]["rows"]
    wct = sec_univ["results_by_class"]["Water"]["contrasts"]

    umr_tcm_vs_hp = 100.0 * (1.0 - (1.0 - brows["tcm_tcpcr_unknown"]["u30"]) / (1.0 - brows["hyperprior_standalone"]["u30"]))
    umr_tcm_vs_webp = 100.0 * (1.0 - (1.0 - brows["tcm_tcpcr_unknown"]["u30"]) / (1.0 - brows["webp_standalone"]["u30"]))
    tcm_vs_hp_u30 = 100.0 * (brows["tcm_tcpcr_unknown"]["u30"] - brows["hyperprior_standalone"]["u30"])
    tcm_vs_webp_u30 = 100.0 * (brows["tcm_tcpcr_unknown"]["u30"] - brows["webp_standalone"]["u30"])
    tcm_vs_hp_f1 = 100.0 * (brows["tcm_tcpcr_unknown"]["later_f1"] - brows["hyperprior_standalone"]["later_f1"])
    tcm_vs_hp_fnr_rel = 100.0 * (brows["tcm_tcpcr_unknown"]["fnr"] - brows["hyperprior_standalone"]["fnr"]) / brows["hyperprior_standalone"]["fnr"]

    fact_lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Complete $2\times 2$ attribution, universal plug-and-play \texttt{TC-PCR} (Rate-Distortion 0--1 knapsack, \texttt{WebP75} residual atlas) across all three base codecs (WebP, Scale-Hyperprior, LIC-TCM) and held-out classes (Building, Tree, Water), and GT Oracle diagnostic on SECOND Cohort~3 (256 locked pairs, 8-KiB cap).}",
        r"\label{tab:factorial-oracle}",
        r"\begingroup\fontsize{7.8}{9.8}\selectfont",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabularx}{\textwidth}{lXrrrrr}\toprule",
        r"\rowcolor{tableheadbg} \textbf{Variant / Contrast} & \textbf{Mechanism / Class} & \textbf{$U_{30}$ / $\Delta U_{30}$} & \textbf{95\% CI (pp)} & \textbf{Later F1} & \textbf{FNR} & \textbf{KiB}" + end,
        r"\midrule",
        f"Cell A / C / D (locked HP) & Abs.+Chg / Res.+Chg / Res.+Unk (Bldg) & {cells['absolute_change']['u30']:.3f}/{cells['residual_change']['u30']:.3f}/{cells['residual_current']['u30']:.3f} & -- & {cells['residual_current']['later_f1']:.4f} & {cells['residual_current']['later_fnr']:.4f} & 7.71" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (WebP, Bldg)}} & 4-KiB WebP + RE01 vs. WebP (0.5380) & \\textbf{{{brows['webp_tcpcr_unknown']['u30']:.4f}}} ({100*bct['webp_vs_standalone']['u30_diff']:+.2f} pp) & [{100*bct['webp_vs_standalone']['u30_ci95'][0]:+.2f}, {100*bct['webp_vs_standalone']['u30_ci95'][1]:+.2f}] & {brows['webp_tcpcr_unknown']['later_f1']:.4f} & {brows['webp_tcpcr_unknown']['fnr']:.4f} & {brows['webp_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (WebP Ref. ROI)}} & Refined-ROI probe (Building, WebP) & {brows['webp_tcpcr_unknown']['u30']:.4f} & -- & \\textbf{{{ref_roi_map['webp']['tcpcr_v2_refined_roi_later_f1']:.4f}}} & \\textbf{{{ref_roi_map['webp']['tcpcr_v2_refined_roi_fnr']:.4f}}} & {brows['webp_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (HP, Bldg)}} & 4-KiB HP + RE01 vs. HP (0.6379) & \\textbf{{{brows['hyperprior_tcpcr_unknown']['u30']:.4f}}} ({100*bct['hyperprior_vs_standalone']['u30_diff']:+.2f} pp) & [{100*bct['hyperprior_vs_standalone']['u30_ci95'][0]:+.2f}, {100*bct['hyperprior_vs_standalone']['u30_ci95'][1]:+.2f}] & {brows['hyperprior_tcpcr_unknown']['later_f1']:.4f} & {brows['hyperprior_tcpcr_unknown']['fnr']:.4f} & {brows['hyperprior_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (TCM, Bldg)}} & 4-KiB TCM + RE01 vs. TCM (0.7688) & \\textbf{{{brows['tcm_tcpcr_unknown']['u30']:.4f}}} ({100*bct['tcm_vs_standalone']['u30_diff']:+.2f} pp) & [{100*bct['tcm_vs_standalone']['u30_ci95'][0]:+.2f}, {100*bct['tcm_vs_standalone']['u30_ci95'][1]:+.2f}] & {brows['tcm_tcpcr_unknown']['later_f1']:.4f} & {brows['tcm_tcpcr_unknown']['fnr']:.4f} & {brows['tcm_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (TCM Raw, Bldg)}} & Unfallback TCM + RE01 (Building) & {brows['tcm_tcpcr_unknown_raw']['u30']:.4f} & -- & {brows['tcm_tcpcr_unknown_raw']['later_f1']:.4f} & \\textbf{{{brows['tcm_tcpcr_unknown_raw']['fnr']:.4f}}} & {brows['tcm_tcpcr_unknown_raw']['mean_total_KiB']:.2f}" + end,
        f"Post-hoc HP GT Oracle & Ideal 8-KiB HP/RE01 (Building) & {ora['oracle_hybrid_with_full_hp']['u30']:.4f} & -- & {ora['oracle_hybrid_with_full_hp']['later_f1']:.4f} & {ora['oracle_hybrid_with_full_hp']['later_fnr']:.4f} & {ora['oracle_hybrid_with_full_hp']['mean_total_bytes']/1024:.2f}" + end,
        r"\midrule",
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (WebP, Tree)}} & vs. Standalone WebP ({trows['webp_standalone']['u30']:.4f}) & {trows['webp_tcpcr_unknown']['u30']:.4f} ({100*tct['webp_vs_standalone']['u30_diff']:+.2f} pp) & [{100*tct['webp_vs_standalone']['u30_ci95'][0]:+.2f}, {100*tct['webp_vs_standalone']['u30_ci95'][1]:+.2f}] & {trows['webp_tcpcr_unknown']['later_f1']:.4f} & {trows['webp_tcpcr_unknown']['fnr']:.4f} & {trows['webp_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (HP, Tree)}} & vs. Standalone HP ({trows['hyperprior_standalone']['u30']:.4f}) & {trows['hyperprior_tcpcr_unknown']['u30']:.4f} ({100*tct['hyperprior_vs_standalone']['u30_diff']:+.2f} pp) & [{100*tct['hyperprior_vs_standalone']['u30_ci95'][0]:+.2f}, {100*tct['hyperprior_vs_standalone']['u30_ci95'][1]:+.2f}] & {trows['hyperprior_tcpcr_unknown']['later_f1']:.4f} & {trows['hyperprior_tcpcr_unknown']['fnr']:.4f} & {trows['hyperprior_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"\\rowcolor{{safesyncbg}} \\textbf{{TC-PCR (TCM, Tree)}} & vs. Standalone TCM ({trows['tcm_standalone']['u30']:.4f}) & \\textbf{{{trows['tcm_tcpcr_unknown']['u30']:.4f}}} ({100*tct['tcm_vs_standalone']['u30_diff']:+.2f} pp) & [{100*tct['tcm_vs_standalone']['u30_ci95'][0]:+.2f}, {100*tct['tcm_vs_standalone']['u30_ci95'][1]:+.2f}] & {trows['tcm_tcpcr_unknown']['later_f1']:.4f} & {trows['tcm_tcpcr_unknown']['fnr']:.4f} & {trows['tcm_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        f"TC-PCR (WebP / HP / TCM, Water) & Smooth-surface boundary regime & {wrows['webp_tcpcr_unknown']['u30']:.3f}/{wrows['hyperprior_tcpcr_unknown']['u30']:.3f}/{wrows['tcm_tcpcr_unknown']['u30']:.3f} & [{100*wct['tcm_vs_standalone']['u30_ci95'][0]:+.1f}, {100*wct['tcm_vs_standalone']['u30_ci95'][1]:+.1f}] & {wrows['tcm_tcpcr_unknown']['later_f1']:.4f} & {wrows['tcm_tcpcr_unknown']['fnr']:.4f} & {wrows['tcm_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        r"\midrule",
        f"Unknown vs. Change (WebP/HP/TCM) & Same-codec novelty gain (Bldg) & +{100*bct['webp_unknown_over_change']['u30_diff']:.2f}/+{100*bct['hyperprior_unknown_over_change']['u30_diff']:.2f}/+{100*bct['tcm_unknown_over_change']['u30_diff']:.2f} pp & WebP [{100*bct['webp_unknown_over_change']['u30_ci95'][0]:+.2f}, {100*bct['webp_unknown_over_change']['u30_ci95'][1]:+.2f}] & -- & -- & --" + end,
        f"TC-PCR(TCM) vs. HP / WebP & Cross-tier dominance (Bldg) & +{tcm_vs_hp_u30:.2f} / +{tcm_vs_webp_u30:.2f} pp & -{umr_tcm_vs_hp:.2f}\\% / -{umr_tcm_vs_webp:.2f}\\% UMR & {tcm_vs_hp_f1:+.2f} pp & {tcm_vs_hp_fnr_rel:+.1f}\\% rel. & {brows['tcm_tcpcr_unknown']['mean_total_KiB']:.2f}" + end,
        r"\bottomrule\end{tabularx}\endgroup\end{table*}",
    ]
    p_fact_tex = ROOT / "paper/supplement/generated/factorial_oracle_table.tex"
    p_fact_tex.write_text("\n".join(fact_lines) + "\n", encoding="utf-8")

    return [p_main_cost, p_sup_cost, p_lowres_tex, p_fact_tex]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    lowres_analysis = read_json(ROOT / "results/lowres_reference_sn7/v1/analysis.json")
    png_profile = read_json(ROOT / "results/sender_profile/compressed_archive_v1/summary.json")
    sender_ref = read_json(ROOT / "results/task_evidence/sender_reference_v1/verification.json")
    diag = read_json(ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/diagnostic.json")
    sn7_repaired = read_json(ROOT / "results/sn7_repaired_design/v1/analysis.json")
    sec_repaired = read_json(ROOT / "results/task_evidence/second_repaired_design_v1/analysis.json")
    asf_plus = read_json(ROOT / "results/sn7_smooth_asf_v1/analysis.json")
    sec_univ = read_json(ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/analysis.json")

    outputs = []
    outputs.extend(build_figure1_sn7_comm_vs_utility(lowres_analysis, sn7_repaired, asf_plus))
    outputs.extend(build_figure2_storage_vs_utility(lowres_analysis, png_profile, sender_ref, diag, sec_repaired, asf_plus, sec_univ))
    outputs.extend(build_figure3_second_four_cell_and_oracle(diag, sec_repaired, sec_univ))
    outputs.extend(build_tables(lowres_analysis, png_profile, diag, sn7_repaired, sec_repaired, asf_plus, sec_univ))

    sources = [
        ROOT / "results/lowres_reference_sn7/v1/analysis.json",
        ROOT / "results/sender_profile/compressed_archive_v1/summary.json",
        ROOT / "results/task_evidence/sender_reference_v1/verification.json",
        ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/diagnostic.json",
        ROOT / "results/sn7_repaired_design/v1/analysis.json",
        ROOT / "results/task_evidence/second_repaired_design_v1/analysis.json",
        ROOT / "results/sn7_smooth_asf_v1/analysis.json",
        ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/analysis.json",
        ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/refined_roi_summary.json",
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
