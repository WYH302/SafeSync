"""Generate Figure 1 (Teaser): Performance-Efficiency Pareto Frontier (SpaceNet 7)
and Universal Open-World Uplift (SECOND) in unified Okabe-Ito colorblind-safe IEEE style.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Restrained Cohesive Academic Palette (4 Muted Hues + Neutral Slate Ramp)
# Zero pure RGB (#FF0000/#00FF00/#0000FF), zero rainbow/neon clutter.
PALETTE = {
    "safesync": "#1B4965",      # Deep Academic Navy (Hero: SafeSync-ASF+ / TC-PCR)
    "safesync_sub": "#5C88AA",  # Muted Steel Blue (Legacy ZFSS / secondary SafeSync)
    "tcm": "#B5533C",           # Muted Terracotta (LIC-TCM 4.18 GB / Oracle line)
    "hyperprior": "#4D7C6F",    # Muted Sage (Scale-Hyperprior 61.1 MB)
    "full_hist": "#8C6D46",     # Muted Warm Ochre (Full history 30 dB)
    "webp": "#334155",          # Dark Slate (Stateless WebP)
    "dct": "#64748B",           # Mid Slate (DCT bound)
    "lowres": "#7B8A9B",        # Cool Neutral Slate (4x WebP ref)
    "bar_base": "#D9E1E8",      # Soft Neutral Slate Tint (Standalone base bars)
    "zone_bg": "#EEF4F8",       # Subtle Ice-Navy Tint (Pareto CPU Zone)
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.0,
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
        "svg.fonttype": "none",
    })

    asf_plus = json.loads((ROOT / "results/sn7_smooth_asf_v1/analysis.json").read_text("utf-8"))
    lowres = json.loads((ROOT / "results/lowres_reference_sn7/v1/analysis.json").read_text("utf-8"))
    sec_v2 = json.loads((ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1/analysis.json").read_text("utf-8"))
    diag = json.loads((ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1/diagnostic.json").read_text("utf-8"))

    l32 = {r["method"]: r for r in lowres["summaries"] if int(r["budget"]) == 32768}
    asf32_f1 = float(asf_plus["by_budget"]["32768"]["oof_building_f1"])
    tcm32_f1 = float(l32["tcm"]["building_f1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.38), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(left=0.072, right=0.994, bottom=0.188, top=0.872, wspace=0.25)

    # ------------------------------------------------------------------
    # Panel (a): Performance-Efficiency Pareto Frontier (SpaceNet 7, 32 KiB)
    # ------------------------------------------------------------------
    ax1.add_patch(Rectangle((0.068, 0.455), 0.145, 0.068, facecolor=PALETTE["zone_bg"], edgecolor=PALETTE["safesync"],
                            linewidth=0.50, linestyle="--", alpha=0.85, zorder=1))
    ax1.text(0.073, 0.517, "Pareto CPU Zone", fontsize=6.4, fontweight="bold",
             color="#000000", ha="left", va="top", zorder=2)

    # (label, x, y, color, marker, size, text_xy, ha, is_hero, use_leader)
    # Only use a subtle leader line where two markers sit immediately adjacent (Legacy ZFSS vs. DCT).
    points = [
        ("SafeSync-ASF+ (Ours)\n0 B weights | 118 ms CPU", 0.1175, asf32_f1, PALETTE["safesync"], "o", 54,
         (0.245, 0.471), "left", True, False),
        ("4x WebP Ref (1,495 viol.)", 0.2264, float(l32["lowres_4x_webp_12k"]["building_f1"]),
         PALETTE["lowres"], "^", 32, (0.280, 0.439), "left", False, False),
        ("Full Hist. (1.34 MiB)", 0.2575, float(l32["full_cache_gate30"]["building_f1"]),
         PALETTE["full_hist"], "P", 34, (0.315, 0.410), "left", False, False),
        ("Legacy ZFSS", 0.2547, float(l32["gaussian_20261029"]["building_f1"]),
         PALETTE["safesync_sub"], "o", 30, (0.195, 0.366), "right", False, True),
        ("DCT (13.9 KB)", 0.2912, float(l32["dct_bound"]["building_f1"]),
         PALETTE["dct"], "s", 28, (0.360, 0.381), "left", False, False),
        ("Stateless WebP", 2.06, float(l32["full_webp"]["building_f1"]),
         PALETTE["webp"], "v", 32, (2.65, 0.279), "left", False, False),
        ("Hyperprior\n(61.1 MB, GPU)", 11.54, float(l32["learned_hyperprior_q1_8"]["building_f1"]),
         PALETTE["hyperprior"], "D", 36, (9.0, 0.348), "right", False, False),
        ("LIC-TCM\n(4.18 GB, GPU)", 30.88, tcm32_f1,
         PALETTE["tcm"], "*", 62, (29.5, 0.409), "center", False, False),
    ]

    for lbl, xs, ys, col, mk, sz, (tx, ty), ha, is_hero, use_leader in points:
        ax1.scatter(xs, ys, color=col, marker=mk, s=sz,
                    edgecolors="#000000" if is_hero else "white",
                    linewidths=0.60 if is_hero else 0.40,
                    zorder=7 if is_hero else 5)
        arrow_kw = dict(arrowstyle="-", color="#475569", lw=0.42, alpha=0.65) if use_leader else None
        ax1.annotate(lbl, (xs, ys), xytext=(tx, ty), fontsize=6.45 if is_hero else 6.1,
                     color="#000000", fontweight="bold" if is_hero else "normal",
                     ha=ha, va="center", linespacing=1.12, arrowprops=arrow_kw, zorder=8)

    # Single narrative dashed arc from 4.18-GB GPU LIC-TCM -> SafeSync-ASF+ (cleanly clear of callout box)
    ax1.annotate("", xy=(0.145, asf32_f1 + 0.010), xytext=(27.5, tcm32_f1 + 0.011),
                 arrowprops=dict(arrowstyle="-|>", color=PALETTE["safesync"], lw=0.88,
                                 linestyle="--", connectionstyle="arc3,rad=0.14"), zorder=6)
    ax1.text(7.2, 0.514, "263x faster, +3.86 pp\n(0 B vs. 4.18 GB)", fontsize=6.2, fontweight="bold",
             color="#000000", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=PALETTE["safesync"], lw=0.48, alpha=0.96),
             zorder=9)

    ax1.set_xscale("log")
    ax1.set_xlim(0.065, 62.0)
    ax1.set_ylim(0.255, 0.542)
    ax1.set_xticks([0.1, 1.0, 10.0])
    ax1.set_xticklabels(["0.1", "1", "10"])
    ax1.set_xlabel("Onboard sender latency per frame (s, log scale)")
    ax1.set_ylabel("4-Fold OOF Building F1 (32 KiB)")
    ax1.set_title("(a) Performance-Efficiency Pareto Frontier", loc="center", pad=5, fontweight="bold", color="#000000")
    ax1.grid(alpha=0.22, which="major", linewidth=0.38, color="#CBD5E1")
    ax1.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel (b): Universal Codec-Agnostic Open-World Uplift (SECOND, 8 KiB)
    # ------------------------------------------------------------------
    brows = sec_v2["results_by_class"]["Building"]["rows"]
    ora_u30 = float(diag["oracle_diagnostic_h5_b8192"]["methods"]["oracle_hybrid_with_full_hp"]["u30"])

    base_vals = [
        float(brows["webp_standalone"]["u30"]),
        float(brows["hyperprior_standalone"]["u30"]),
        float(brows["tcm_standalone"]["u30"]),
    ]
    tcpcr_vals = [
        float(brows["webp_tcpcr_unknown"]["u30"]),
        float(brows["hyperprior_tcpcr_unknown"]["u30"]),
        float(brows["tcm_tcpcr_unknown"]["u30"]),
    ]
    gains = [(t - b) * 100.0 for b, t in zip(base_vals, tcpcr_vals)]
    x_labels = ["WebP\n(0 B)", "Hyperprior\n(61.1 MB)", "LIC-TCM\n(4.18 GB)"]
    xb = np.arange(len(x_labels))
    wb = 0.34

    bars_base = ax2.bar(xb - wb / 2, base_vals, width=wb, color=PALETTE["bar_base"], edgecolor="#1F2937",
                        linewidth=0.45, label="Standalone Base Codec", zorder=3)
    bars_tp = ax2.bar(xb + wb / 2, tcpcr_vals, width=wb, color=PALETTE["safesync"], edgecolor="#1F2937",
                      linewidth=0.45, label="+ SafeSync TC-PCR (Ours)", zorder=4)

    ax2.axhline(ora_u30, color=PALETTE["tcm"], linestyle="--", linewidth=0.85, zorder=5)
    ax2.text(-0.42, ora_u30 + 0.012, f"Hyperprior GT Oracle ({ora_u30:.3f})", fontsize=6.25,
             fontweight="bold", color="#000000", ha="left", va="bottom",
             bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.88), zorder=6)

    for r_b, v_b in zip(bars_base, base_vals):
        ax2.text(r_b.get_x() + r_b.get_width() / 2, v_b + 0.008, f"{v_b:.3f}",
                 ha="center", va="bottom", fontsize=6.4, color="#000000")
    for r_t, v_t, g in zip(bars_tp, tcpcr_vals, gains):
        cx = r_t.get_x() + r_t.get_width() / 2
        ax2.text(cx, v_t + 0.008, f"{v_t:.3f}",
                 ha="center", va="bottom", fontsize=6.55, fontweight="bold", color="#000000", zorder=7)
        ax2.text(cx, v_t - 0.022, f"+{g:.1f}\npp",
                 ha="center", va="top", fontsize=6.2, fontweight="bold", color="#FFFFFF", linespacing=1.02, zorder=7)

    ax2.set_xticks(xb, x_labels)
    ax2.set_ylim(0.42, 0.975)
    ax2.set_ylabel("Held-Out Building Coverage $U_{30}$")
    ax2.set_xlabel("Base codec tier (SECOND, 8-KiB cap)")
    ax2.set_title("(b) Universal Open-World Routing Uplift", loc="center", pad=5, fontweight="bold", color="#000000")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=6.6)
    ax2.grid(axis="y", alpha=0.22, linewidth=0.38, color="#CBD5E1")
    ax2.spines[["top", "right"]].set_visible(False)

    out_dir = ROOT / "paper/figures"
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"teaser_pareto.{ext}", facecolor="white", **kwargs)
    plt.close(fig)
    print("Generated Figure 1 Teaser Pareto -> paper/figures/teaser_pareto.{pdf,svg,png}")


if __name__ == "__main__":
    main()
