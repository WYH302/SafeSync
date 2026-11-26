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

# Unified Okabe-Ito Colorblind-Safe Palette across ALL manuscript figures
OKABE = {
    "safesync": "#004488",      # Deep Navy Blue (Hero: SafeSync-ASF+ / TC-PCR)
    "tcm": "#D55E00",           # Okabe-Ito Vermilion (LIC-TCM 4.18 GB)
    "hyperprior": "#CC79A7",    # Okabe-Ito Reddish Purple (Scale-Hyperprior 61.1 MB)
    "full_hist": "#009E73",     # Okabe-Ito Bluish Green (Full history 30 dB)
    "webp": "#4B5563",          # Muted Charcoal (Stateless WebP)
    "dct": "#E69F00",           # Okabe-Ito Orange (DCT bound)
    "lowres": "#56B4E9",        # Okabe-Ito Sky Blue (4x WebP ref / Legacy ZFSS)
    "muted": "#8C96A5",         # Cool Neutral Slate (Random / Coarse-only)
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.4,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.1,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 0.7,
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.76), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(left=0.074, right=0.992, bottom=0.172, top=0.878, wspace=0.26)

    # ------------------------------------------------------------------
    # Panel (a): Performance-Efficiency Pareto Frontier (SpaceNet 7, 32 KiB)
    # ------------------------------------------------------------------
    ax1.add_patch(Rectangle((0.068, 0.455), 0.145, 0.068, facecolor="#EBF4FF", edgecolor=OKABE["safesync"],
                            linewidth=0.65, linestyle="--", alpha=0.80, zorder=1))
    ax1.text(0.073, 0.517, "Pareto CPU Zone", fontsize=6.4, fontweight="bold",
             color="#000000", ha="left", va="top", zorder=2)

    points = [
        ("SafeSync-ASF+ (Ours)\n0 B weights | 118 ms CPU", 0.1175, asf32_f1, OKABE["safesync"], "o", 74,
         (0.28, 0.474), "left", True),
        ("4x WebP Ref (1,495 viol.)", 0.2264, float(l32["lowres_4x_webp_12k"]["building_f1"]),
         OKABE["lowres"], "^", 44, (0.42, 0.438), "left", False),
        ("Full Hist. (1.35 MB)", 0.2575, float(l32["full_cache_gate30"]["building_f1"]),
         OKABE["full_hist"], "P", 44, (0.45, 0.409), "left", False),
        ("Legacy ZFSS", 0.2547, float(l32["gaussian_20261029"]["building_f1"]),
         OKABE["lowres"], "o", 38, (0.072, 0.369), "left", False),
        ("DCT (13.9 KB)", 0.2912, float(l32["dct_bound"]["building_f1"]),
         OKABE["dct"], "s", 38, (0.48, 0.371), "left", False),
        ("Stateless WebP", 2.06, float(l32["full_webp"]["building_f1"]),
         OKABE["webp"], "v", 44, (2.75, 0.278), "left", False),
        ("Hyperprior\n(61.1 MB, GPU)", 11.54, float(l32["learned_hyperprior_q1_8"]["building_f1"]),
         OKABE["hyperprior"], "D", 48, (3.20, 0.335), "left", False),
        ("LIC-TCM\n(4.18 GB, GPU)", 30.88, tcm32_f1,
         OKABE["tcm"], "*", 84, (11.5, 0.408), "left", False),
    ]

    for lbl, xs, ys, col, mk, sz, (tx, ty), ha, is_hero in points:
        ax1.scatter(xs, ys, color=col, marker=mk, s=sz,
                    edgecolors="#000000" if is_hero else "white",
                    linewidths=0.9 if is_hero else 0.5,
                    zorder=7 if is_hero else 5)
        arrow_kw = dict(arrowstyle="-", color="#000000", lw=0.60, alpha=0.80) if is_hero else (
            dict(arrowstyle="-", color="#000000", lw=0.45, alpha=0.50) if "WebP" != lbl.split()[-1] else None
        )
        ax1.annotate(lbl, (xs, ys), xytext=(tx, ty), fontsize=6.45 if is_hero else 6.1,
                     color="#000000", fontweight="bold" if is_hero else "normal",
                     ha=ha, va="center", linespacing=1.12, arrowprops=arrow_kw, zorder=8)

    # Clean upward sweeping arrow from 4.18-GB GPU LIC-TCM -> SafeSync-ASF+ (zero overlap!)
    ax1.annotate("", xy=(0.145, asf32_f1 + 0.008), xytext=(28.0, tcm32_f1 + 0.011),
                 arrowprops=dict(arrowstyle="-|>", color=OKABE["safesync"], lw=1.25,
                                 linestyle="--", connectionstyle="arc3,rad=0.16"), zorder=6)
    ax1.text(7.2, 0.506, "263x faster, +3.87 pp\n(0 B vs. 4.18 GB)", fontsize=6.25, fontweight="bold",
             color="#000000", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=OKABE["safesync"], lw=0.65, alpha=0.96),
             zorder=9)

    ax1.set_xscale("log")
    ax1.set_xlim(0.065, 62.0)
    ax1.set_ylim(0.255, 0.534)
    ax1.set_xticks([0.1, 1.0, 10.0])
    ax1.set_xticklabels(["0.1", "1", "10"])
    ax1.set_xlabel("Onboard sender latency per frame (s, log scale)")
    ax1.set_ylabel("4-Fold OOF Building F1 (32 KiB)")
    ax1.set_title("(a) Performance-Efficiency Pareto Frontier", loc="center", pad=6, fontweight="bold", color="#000000")
    ax1.grid(alpha=0.18, which="major", linewidth=0.5)
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

    bars_base = ax2.bar(xb - wb / 2, base_vals, width=wb, color="#D1D5DB", edgecolor="#000000",
                        linewidth=0.65, label="Standalone Base Codec", zorder=3)
    bars_tp = ax2.bar(xb + wb / 2, tcpcr_vals, width=wb, color=OKABE["safesync"], edgecolor="#000000",
                      linewidth=0.65, label="+ SafeSync TC-PCR (Ours)", zorder=4)

    ax2.axhline(ora_u30, color=OKABE["tcm"], linestyle="--", linewidth=1.15, zorder=5)
    ax2.text(-0.42, ora_u30 + 0.012, f"Hyperprior GT Oracle ({ora_u30:.3f})", fontsize=6.25,
             fontweight="bold", color="#000000", ha="left", va="bottom",
             bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.85), zorder=6)

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
    ax2.set_title("(b) Universal Open-World Routing Uplift", loc="center", pad=6, fontweight="bold", color="#000000")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, fontsize=6.6)
    ax2.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    out_dir = ROOT / "paper/figures"
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"teaser_pareto.{ext}", facecolor="white", **kwargs)
    plt.close(fig)
    print("Generated Figure 1 Teaser Pareto -> paper/figures/teaser_pareto.{pdf,svg,png}")


if __name__ == "__main__":
    main()
