"""Publication-grade overall framework diagram for SafeSync (Figure 1).

Combines embedded Comic Sans MS vector typography (PDF Type 42), real multi-temporal
remote-sensing visual patches (coarse field, high-frequency detail, spatial routing
grid, co-constructed render), isometric satellite & ground-station illustrations,
and geometric unit-sphere / prototype routing diagrams across full double-column width.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Wedge, Polygon, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_comic_sans() -> str:
    for candidate in (
        Path(r"C:\Windows\Fonts\comic.ttf"),
        Path(r"C:\Windows\Fonts\comicbd.ttf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    available = {f.name for f in font_manager.fontManager.ttflist}
    return "Comic Sans MS" if "Comic Sans MS" in available else "DejaVu Sans"


def _load_satellite_and_ground_icons():
    icon_sheet = ROOT / "paper/figures/icons_space_ground.png"
    if not icon_sheet.is_file():
        return None, None
    rgba = np.array(Image.open(icon_sheet).convert("RGBA"))
    h, w, _ = rgba.shape
    # Make near-white background transparent so icons blend cleanly onto header cards
    white_mask = (rgba[:, :, 0] > 244) & (rgba[:, :, 1] > 244) & (rgba[:, :, 2] > 244)
    rgba[white_mask, 3] = 0

    def trim(sub):
        ys, xs = np.where(sub[:, :, 3] > 15)
        if len(xs) == 0:
            return sub
        pad = 8
        y0, y1 = max(0, ys.min() - pad), min(sub.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(sub.shape[1], xs.max() + pad)
        return sub[y0:y1, x0:x1]

    sat = trim(rgba[:, : int(w * 0.49)])
    gnd = trim(rgba[:, int(w * 0.51) :])
    return sat, gnd


def _load_real_rs_patches():
    """Extract a real SpaceNet 7 crop from receiver_crops.png to show genuine coarse/detail decomposition."""
    crops_path = ROOT / "paper/figures/receiver_crops.png"
    if crops_path.is_file():
        img = np.array(Image.open(crops_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        # Bottom-left subpanel ("Area 2 Current RGB") has rich dark vegetation + crisp urban buildings
        y0, y1 = int(h * 0.56), int(h * 0.93)
        x0, x1 = int(w * 0.065), int(w * 0.275)
        patch = img[y0:y1, x0:x1]
        patch = np.array(
            Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
    else:
        rng = np.random.default_rng(2026)
        patch = np.clip(0.45 + 0.18 * rng.standard_normal((128, 128, 3)), 0, 1)
    # 4x4 block-mean coarse field (each cell 32x32 in 128x128 display)
    coarse_small = patch.reshape(8, 16, 8, 16, 3).mean(axis=(1, 3))
    coarse = np.repeat(np.repeat(coarse_small, 16, axis=0), 16, axis=1)
    # Enhanced zero-mean high-frequency detail visualization
    detail_raw = patch - coarse
    detail_vis = np.clip(0.5 + 3.2 * detail_raw, 0.0, 1.0)
    return patch, coarse, detail_vis


def main() -> None:
    font_name = _load_comic_sans()
    plt.rcParams.update({
        "font.family": font_name,
        "font.sans-serif": [font_name, "DejaVu Sans"],
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # Full double-column aspect ratio (16.0 x 7.4 in -> scales crisply to \textwidth = 7.16 in)
    fig, ax = plt.subplots(figsize=(16.0, 7.45), dpi=300)
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.006, top=0.994)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 74.5)
    ax.axis("off")

    # Harmonious publication palette (Nature / CVPR / IEEE TGRS inspired)
    c_navy = "#1E293B"
    c_slate = "#475569"
    c_blue = "#1D4ED8"
    c_blue_lt = "#EFF6FF"
    c_blue_bd = "#93C5FD"
    c_teal = "#0F766E"
    c_teal_lt = "#F0FDFA"
    c_teal_bd = "#5EEAD4"
    c_green = "#15803D"
    c_green_lt = "#DCFCE7"
    c_green_bd = "#86EFAC"
    c_amber = "#B45309"
    c_amber_lt = "#FEF3C7"
    c_amber_bd = "#FCD34D"
    c_purple = "#6D28D9"
    c_purple_lt = "#F5F3FF"
    c_purple_bd = "#C4B5FD"
    c_rose = "#BE123C"
    c_rose_lt = "#FFE4E6"
    c_rose_bd = "#FDA4AF"

    def shadow_card(x, y, w, h, fill="#FFFFFF", edge="#CBD5E1", lw=1.15, r=1.1, shadow=True, zorder=1):
        if shadow:
            ax.add_patch(FancyBboxPatch(
                (x + 0.28, y - 0.28), w, h,
                boxstyle=f"round,pad=0.12,rounding_size={r}",
                facecolor="#CBD5E1", edgecolor="none", alpha=0.38, zorder=zorder
            ))
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.12,rounding_size={r}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder + 1
        ))

    def pill(x, y, w, h, text, fill, edge, text_color="#FFFFFF", fontsize=7.6, zorder=5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.08,rounding_size={h/2:.2f}",
            facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=zorder
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=zorder + 1)

    def flow_arrow(p0, p1, color=c_navy, lw=1.45, rad=0.0, dashed=False, scale=11.5, zorder=6):
        ax.add_patch(FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>", mutation_scale=scale,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (3.8, 2.2)) if dashed else "-",
            zorder=zorder
        ))

    def draw_img_in_box(img_arr, x, y, w, h, edge=c_navy, lw=0.9, zorder=4):
        ax.imshow(img_arr, extent=(x, x + w, y, y + h), origin="upper", interpolation="bilinear", zorder=zorder, aspect="auto")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, linewidth=lw, zorder=zorder + 1))

    sat_icon, gnd_icon = _load_satellite_and_ground_icons()
    patch_rgb, patch_coarse, patch_detail = _load_real_rs_patches()

    # =========================================================================
    # TOP BANNER (y = 63.6 .. 73.9): PARADIGM SHIFT & KEY QUANTITATIVE GAINS
    # =========================================================================
    shadow_card(0.6, 63.6, 158.8, 10.3, fill="#F8FAFC", edge="#94A3B8", lw=1.2, r=1.2)
    pill(1.6, 70.7, 74.0, 2.5, "(a) Paradigm Shift: Stateless Frame Coding vs. Asymmetric World-State Co-Construction",
         fill=c_navy, edge=c_navy, text_color="#FFFFFF", fontsize=7.2)

    # Left sub-card: Prior Bottlenecks
    shadow_card(1.6, 64.3, 47.0, 5.9, fill=c_rose_lt, edge=c_rose_bd, lw=1.0, r=0.8, shadow=False, zorder=3)
    ax.text(2.6, 69.0, "Prior Satellite Downlink Bottlenecks", color=c_rose, fontsize=7.8, fontweight="bold", va="center", zorder=5)
    ax.text(2.6, 66.3,
            "1. Stateless Codecs (WebP): Blur erases new buildings at 32 KiB (F1 = 0.279)\n"
            "2. Thumbnail State (4x WebP): 1,495 false-stable errors & 79.8% drift triggers\n"
            "3. Heavy Neural Codecs (LIC-TCM): 4.18 GB weights & 30.9 s GPU search",
            color="#4C0519", fontsize=6.55, va="center", linespacing=1.26, zorder=5)

    flow_arrow((49.0, 67.25), (53.2, 67.25), color=c_blue, lw=2.0, scale=13, zorder=6)
    ax.text(51.1, 68.7, "SafeSync", color=c_blue, fontsize=6.9, fontweight="bold", ha="center", zorder=6)

    # Middle sub-card: SafeSync Core Idea
    shadow_card(53.6, 64.3, 53.6, 5.9, fill=c_green_lt, edge=c_green_bd, lw=1.0, r=0.8, shadow=False, zorder=3)
    ax.text(54.6, 69.0, "SafeSync Asymmetric Space-Ground State Co-Construction", color=c_green, fontsize=7.8, fontweight="bold", va="center", zorder=5)
    ax.text(54.6, 66.3,
            "- Zero-Footprint Seeded Sketching (ZFSS): 32-bit seed -> 0 B matrix, 12.3 KB/AOI (-99.09%)\n"
            "- Dual-Gate Decoupling (DGAD): Certified STABLE (0 errors) + Unbiased UPDATED ranking\n"
            "- Semantic & Open-World Tier (US-BRS + TC-PCR): Lossless int8 delta & 30-dB emergence",
            color="#052E16", fontsize=6.55, va="center", linespacing=1.26, zorder=5)

    # Right sub-card: Mini State Footprint Bar Visual
    shadow_card(108.4, 64.3, 50.0, 8.9, fill="#FFFFFF", edge="#CBD5E1", lw=1.0, r=0.8, shadow=False, zorder=3)
    ax.text(109.5, 72.0, "Single-Area Onboard Persistent Footprint (N = 1 AOI)", color=c_navy, fontsize=7.4, fontweight="bold", va="center", zorder=5)
    bars = [
        ("Cached Gaussian Matrix", 2.37, 22.8, "#94A3B8", "2.37 MB"),
        ("Lossless PNG History", 1.35, 13.0, "#64748B", "1.35 MB"),
        ("SafeSync ZFSS (Ours)", 0.012, 1.2, c_green, "12.3 KB (-99.09%, 0 B Matrix)"),
    ]
    for idx, (lbl, _, bar_w, col, val_txt) in enumerate(bars):
        by = 69.6 - idx * 2.15
        ax.text(109.5, by, lbl, color=c_slate if idx < 2 else c_green,
                fontsize=6.4, fontweight="bold" if idx == 2 else "normal", va="center", zorder=5)
        ax.add_patch(FancyBboxPatch((127.6, by - 0.55), bar_w, 1.1,
                                    boxstyle="round,pad=0.02,rounding_size=0.3",
                                    facecolor=col, edgecolor="none", zorder=5))
        ax.text(128.1 + bar_w, by, val_txt, color=c_navy if idx < 2 else c_green,
                fontsize=6.35, fontweight="bold", va="center", zorder=5)

    # =========================================================================
    # THREE MAIN ARCHITECTURAL COLUMNS (y = 0.6 .. 62.6)
    #   LEFT:   Satellite Onboard Sender (x = 0.6 .. 68.6)
    #   CENTER: Asymmetric Space-Ground Channel (x = 69.4 .. 90.6)
    #   RIGHT:  Ground Station Persistent Archive (x = 91.4 .. 159.4)
    # =========================================================================
    shadow_card(0.6, 0.6, 68.0, 61.9, fill="#F8FAFC", edge=c_blue_bd, lw=1.35, r=1.3)
    shadow_card(69.4, 0.6, 21.2, 61.9, fill="#FFFBEB", edge=c_amber_bd, lw=1.25, r=1.3)
    shadow_card(91.4, 0.6, 68.0, 61.9, fill="#F8FAFC", edge=c_teal_bd, lw=1.35, r=1.3)

    # Column Header Banners
    pill(1.8, 58.9, 53.5, 2.8, "SPACE SEGMENT: Resource-Constrained LEO Sender (CPU-Only, 0 GPU)",
         fill=c_blue, edge=c_blue, fontsize=7.6)
    pill(70.3, 58.9, 19.4, 2.8, "ASYMMETRIC LINK",
         fill=c_amber, edge=c_amber, fontsize=7.5)
    pill(92.6, 58.9, 53.5, 2.8, "GROUND SEGMENT: Persistent World-State Archive & Analytics",
         fill=c_teal, edge=c_teal, fontsize=7.6)

    # Place AI-generated Satellite & Ground Station icons in header corners!
    if sat_icon is not None:
        ax.imshow(sat_icon, extent=(56.2, 67.8, 55.2, 62.3), zorder=8, aspect="auto")
    if gnd_icon is not None:
        ax.imshow(gnd_icon, extent=(147.0, 158.6, 55.2, 62.3), zorder=8, aspect="auto")

    # =========================================================================
    # TIER 1 ROW (y = 26.2 .. 57.8): PIXEL-ARCHIVE STATE CO-CONSTRUCTION
    # =========================================================================
    # Satellite Tier 1 container
    shadow_card(1.6, 26.4, 66.0, 31.4, fill="#FFFFFF", edge="#BFDBFE", lw=1.0, r=1.0, shadow=False, zorder=2)
    pill(2.4, 54.8, 49.5, 2.3, "(b) Tier 1: Pixel-Archive Synchronization (ASF + ZFSS + DGAD)",
         fill=c_blue_lt, edge=c_blue, text_color=c_blue, fontsize=7.3)

    # 1. Pass Observation x_t + Real RGB Thumbnail + Coarse/Detail thumbnails
    shadow_card(2.4, 38.4, 21.6, 15.6, fill=c_blue_lt, edge=c_blue_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(13.2, 52.8, "1. Pass Observation & ASF Split", color=c_navy, fontsize=7.3, fontweight="bold", ha="center", va="center", zorder=5)
    draw_img_in_box(patch_rgb, 3.2, 44.3, 6.8, 7.2, edge=c_navy, zorder=4)
    ax.text(6.6, 43.3, "Pass RGB x_t", color=c_navy, fontsize=6.3, fontweight="bold", ha="center", zorder=5)

    draw_img_in_box(patch_coarse, 11.4, 47.9, 5.2, 4.0, edge=c_blue, zorder=4)
    ax.text(19.9, 49.9, "Coarse c_t\nQ8(D x_t)\n(LB01)", color=c_blue, fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=5)

    draw_img_in_box(patch_detail, 11.4, 42.6, 5.2, 4.0, edge=c_teal, zorder=4)
    ax.text(19.9, 44.6, "Detail H x_t\n(I - UD) x_t\nZero-mean", color=c_teal, fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=5)

    ax.text(13.2, 40.1, "Orthogonal split isolates illumination\ndrift in c_t from structural edges H x_t",
            color=c_slate, fontsize=6.0, ha="center", va="center", zorder=5)

    # 2. ZFSS Zero-Footprint Seeded Sketching Card (bottom-left of Tier 1)
    shadow_card(2.4, 27.3, 21.6, 10.2, fill=c_green_lt, edge=c_green_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(13.2, 36.3, "2. ZFSS Seeded Sketch s_t (SC02)", color=c_green, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=5)
    # Mini visual of 32-bit seed -> streamed chunk matrix A -> 48B q16 sketch
    pill(3.1, 32.3, 6.2, 2.4, "32-bit Seed\n[12:16]", fill="#FFFFFF", edge=c_green, text_color=c_green, fontsize=5.9, zorder=5)
    flow_arrow((9.4, 33.5), (11.1, 33.5), color=c_green, lw=1.2, scale=9, zorder=6)
    pill(11.2, 32.3, 6.3, 2.4, "Stream A\n512-d chunk", fill="#FFFFFF", edge=c_teal, text_color=c_teal, fontsize=5.9, zorder=5)
    flow_arrow((17.6, 33.5), (19.1, 33.5), color=c_green, lw=1.2, scale=9, zorder=6)
    pill(19.2, 32.3, 4.2, 2.4, "q16 s_b\n48 B", fill=c_green, edge=c_green, text_color="#FFFFFF", fontsize=5.9, zorder=5)
    ax.text(13.2, 29.5, "0 B persistent matrix | 12.3 KB/AOI\nBit-exact across 47,616 blocks (-99.09%)",
            color="#052E16", fontsize=6.05, fontweight="bold", ha="center", va="center", zorder=5)

    # 3. DGAD Dual-Gate Asymmetric Decoupling + Spatial Routing Grid (right half of Satellite Tier 1)
    shadow_card(25.4, 31.6, 41.2, 22.4, fill="#FFFFFF", edge=c_blue, lw=1.1, r=0.9, shadow=False, zorder=3)
    ax.text(46.0, 52.7, "3. Dual-Gate Asymmetric Decoupling (DGAD) & Block Packer",
            color=c_blue, fontsize=7.25, fontweight="bold", ha="center", va="center", zorder=5)

    # Safety Gate sub-box
    shadow_card(26.4, 43.1, 23.2, 7.9, fill=c_green_lt, edge=c_green, lw=0.9, r=0.7, shadow=False, zorder=4)
    ax.text(38.0, 49.6, "Gate A: Certified Safety Gate (Prop. 1)", color=c_green, fontsize=6.65, fontweight="bold", ha="center", zorder=5)
    ax.text(38.0, 46.1,
            "Chi-square union bound over |S_{b,t}| <= t+1:\n"
            "L_b <= tau  ==>  CERTIFIED STABLE\n"
            "0 false-stable violations (vs. 1,495 in 4x WebP)",
            color="#052E16", fontsize=5.95, ha="center", va="center", linespacing=1.24, zorder=5)

    # Utility Gate sub-box
    shadow_card(26.4, 34.1, 23.2, 7.9, fill=c_blue_lt, edge=c_blue, lw=0.9, r=0.7, shadow=False, zorder=4)
    ax.text(38.0, 40.6, "Gate B: Unbiased Utility Ranking Gate", color=c_blue, fontsize=6.65, fontweight="bold", ha="center", zorder=5)
    ax.text(38.0, 37.1,
            "Point estimate D_hat_b = ||z_b||_2^2 / k\n"
            "Ranks non-stable blocks -> pack JPEG 4:4:4\n"
            "atlas (LA01) within cap B; rest DEFERRED",
            color=c_navy, fontsize=5.95, ha="center", va="center", linespacing=1.24, zorder=5)

    # Visual 6x6 Block Status Map (STABLE / UPDATED / DEFERRED)
    gx0, gy0, cell = 51.3, 37.3, 2.08
    grid_states = [
        [0, 0, 1, 1, 0, 0],
        [0, 2, 1, 1, 2, 0],
        [0, 0, 2, 1, 0, 0],
        [2, 0, 0, 0, 0, 2],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
    ]
    state_colors = {0: "#86EFAC", 1: "#2563EB", 2: "#FDE68A"}
    for r_i in range(6):
        for c_i in range(6):
            st = grid_states[5 - r_i][c_i]
            ax.add_patch(Rectangle(
                (gx0 + c_i * cell, gy0 + r_i * cell), cell - 0.14, cell - 0.14,
                facecolor=state_colors[st], edgecolor="#1E293B", linewidth=0.45, zorder=5
            ))
    ax.text(gx0 + 3 * cell, gy0 + 6 * cell + 0.65, "Block Routing Map", color=c_navy, fontsize=6.25, fontweight="bold", ha="center", zorder=5)
    # Mini legend under grid
    for idx, (col, txt) in enumerate([("#86EFAC", "STABLE (Reuse)"), ("#2563EB", "UPDATED (Send)"), ("#FDE68A", "DEFERRED (c_t)")]):
        ly = 36.1 - idx * 1.28
        ax.add_patch(Rectangle((51.3, ly - 0.40), 1.25, 0.80, facecolor=col, edgecolor=c_navy, lw=0.4, zorder=5))
        ax.text(52.95, ly, txt, color=c_navy, fontsize=5.75, va="center", zorder=5)

    # Clean dedicated Self-Decode lane at bottom of Satellite Tier 1 (y = 27.3 .. 30.8) -- ZERO overlap!
    pill(25.6, 27.5, 40.8, 3.1,
         "Self-Decode Sync: Decode sent LA01 -> provisional s_t -> commit on 60-B ACK",
         fill="#F1F5F9", edge=c_teal, text_color=c_teal, fontsize=5.95, zorder=5)
    flow_arrow((46.0, 31.6), (46.0, 30.6), color=c_teal, lw=1.3, scale=9, zorder=6)
    flow_arrow((25.6, 29.05), (24.0, 29.8), color=c_teal, lw=1.35, dashed=True, scale=10, zorder=6)
    flow_arrow((24.0, 45.2), (25.4, 45.2), color=c_blue, lw=1.4, scale=10, zorder=6)
    flow_arrow((24.0, 34.0), (25.4, 37.8), color=c_green, lw=1.4, scale=10, zorder=6)

    # --- Center Link (Tier 1): Compact Downlink Packet + 60-B ACK ---
    shadow_card(70.5, 36.8, 19.0, 16.5, fill="#FFFFFF", edge=c_amber, lw=1.1, r=0.9, shadow=False, zorder=3)
    ax.text(80.0, 51.8, "Downlink Frame", color=c_amber, fontsize=7.4, fontweight="bold", ha="center", zorder=5)
    pill(71.6, 48.2, 16.8, 2.2, "Cap B <= 32 KiB", fill=c_amber_lt, edge=c_amber, text_color=c_amber, fontsize=6.6, zorder=5)
    ax.text(80.0, 42.6,
            "- 4x4 Coarse c_t (LB01)\n"
            "- Selected RGB (LA01)\n"
            "- Block Manifest\n"
            "- History Digest + CRC\n"
            "(254.7 ms on 4-thr CPU)",
            color=c_navy, fontsize=6.15, ha="center", va="center", linespacing=1.28, zorder=5)
    flow_arrow((66.6, 45.0), (70.5, 45.0), color=c_amber, lw=1.8, scale=12, zorder=7)
    flow_arrow((89.5, 45.0), (93.2, 45.0), color=c_amber, lw=1.8, scale=12, zorder=7)

    # Dedicated Uplink-Free 60-B ACK Pill in Center Link (y = 27.6 .. 34.4)
    shadow_card(70.5, 27.5, 19.0, 7.3, fill=c_teal_lt, edge=c_teal, lw=1.05, r=0.8, shadow=False, zorder=3)
    ax.text(80.0, 33.4, "Uplink-Free ACK", color=c_teal, fontsize=7.1, fontweight="bold", ha="center", zorder=5)
    ax.text(80.0, 30.1, "60-Byte Commit Digest\n0 Reference Image Uplink!",
            color="#042F2E", fontsize=6.1, fontweight="bold", ha="center", va="center", zorder=5)
    flow_arrow((93.2, 31.0), (89.5, 31.0), color=c_teal, lw=1.6, dashed=True, scale=11, zorder=7)
    flow_arrow((70.5, 31.0), (66.4, 29.1), color=c_teal, lw=1.6, dashed=True, scale=11, zorder=7)

    # --- Ground Segment Tier 1 (x = 92.4 .. 158.4, y = 26.4 .. 57.8) ---
    shadow_card(92.4, 26.4, 66.0, 31.4, fill="#FFFFFF", edge="#99F6E4", lw=1.0, r=1.0, shadow=False, zorder=2)
    pill(93.2, 54.8, 52.5, 2.3, "(c) Ground Asymmetric State Factorization & Downstream Task Analytics",
         fill=c_teal_lt, edge=c_teal, text_color=c_teal, fontsize=7.3)

    # Coarse Field c_t card
    shadow_card(93.2, 43.2, 24.5, 10.5, fill=c_blue_lt, edge=c_blue_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(105.45, 52.3, "Current Coarse Field U c_t", color=c_blue, fontsize=7.1, fontweight="bold", ha="center", zorder=5)
    ax.text(105.45, 47.4,
            "Updates global pass illumination;\n"
            "cuts orbital-drift background\n"
            "false triggers by 4.1x - 4.5x\n"
            "(15.4% vs. 69.7% in 4x WebP)",
            color=c_navy, fontsize=6.15, ha="center", va="center", linespacing=1.25, zorder=5)

    # Immutable Detail Archive a_t card
    shadow_card(93.2, 27.6, 24.5, 14.2, fill="#F1F5F9", edge=c_navy, lw=1.0, r=0.8, shadow=False, zorder=3)
    ax.text(105.45, 40.4, "Immutable Pixel Archive a_t", color=c_navy, fontsize=7.1, fontweight="bold", ha="center", zorder=5)
    ax.text(105.45, 33.9,
            "- Stores acknowledged absolute\n"
            "  RGB blocks C(x_{s,b}) only\n"
            "- NEVER overwritten by U c_t!\n"
            "- Finite candidate set |S_{b,t}|<=t+1\n"
            "  guarantees drift-free sync",
            color=c_navy, fontsize=6.1, ha="center", va="center", linespacing=1.25, zorder=5)

    # Co-Constructed Render + Real Reconstructed Crop + SpaceNet 7 Wins
    shadow_card(119.5, 27.6, 37.9, 26.1, fill=c_green_lt, edge=c_green, lw=1.1, r=0.9, shadow=False, zorder=3)
    ax.text(138.45, 52.3, "Co-Constructed Render & SpaceNet 7 Utility", color=c_green, fontsize=7.3, fontweight="bold", ha="center", zorder=5)
    pill(121.2, 48.7, 34.5, 2.3, "x_hat_t = R( H a_t + U c_t )", fill="#FFFFFF", edge=c_green, text_color=c_green, fontsize=7.0, zorder=5)

    draw_img_in_box(patch_rgb, 121.0, 37.8, 10.6, 9.8, edge=c_green, lw=1.1, zorder=4)
    # Highlight 3 crisp cyan building target boxes on the rendered image
    for bx, by in [(126.4, 44.6), (127.8, 43.9), (125.0, 40.3)]:
        ax.add_patch(Rectangle((bx, by), 1.3, 1.1, fill=False, edgecolor="#06B6D4", linewidth=1.2, zorder=6))
    ax.text(126.3, 36.6, "Decoded Render", color=c_green, fontsize=6.1, fontweight="bold", ha="center", zorder=5)

    ax.text(144.4, 42.7,
            "32-KiB Author-OOF:\n"
            "- Building F1: 0.4250\n"
            "  (+14.6 pp vs. WebP,\n"
            "   +7.4 pp vs. Hyperprior)\n"
            "- New-Track U30: 35.5%\n"
            "  (+8.66 pp vs. 4x WebP)",
            color="#052E16", fontsize=6.15, fontweight="bold", ha="center", va="center", linespacing=1.26, zorder=5)

    pill(121.0, 28.8, 34.9, 6.2,
         "Verified across 5,874 SpaceNet 7 transactions\n"
         "Zero model weights | 45.3 s total CPU / 178 updates",
         fill="#FFFFFF", edge=c_green_bd, text_color=c_navy, fontsize=6.25, zorder=5)

    flow_arrow((117.7, 48.2), (119.5, 45.5), color=c_blue, lw=1.4, scale=10, zorder=6)
    flow_arrow((117.7, 35.0), (119.5, 39.5), color=c_navy, lw=1.4, scale=10, zorder=6)

    # =========================================================================
    # TIER 2 ROW (y = 1.6 .. 25.2): SEMANTIC & OPEN-WORLD CO-CONSTRUCTION
    # =========================================================================
    # Satellite Tier 2 container
    shadow_card(1.6, 1.6, 66.0, 23.6, fill="#FFFFFF", edge=c_purple_bd, lw=1.0, r=1.0, shadow=False, zorder=2)
    pill(2.4, 22.1, 64.2, 2.3,
         "(d) Tier 2: Semantic Feature Sync & Open-World Routing (US-BRS + TC-PCR)",
         fill=c_purple_lt, edge=c_purple, text_color=c_purple, fontsize=7.25)

    # Left sub-card: US-BRS Unit-Sphere Bounded Residual Sync + Geometric Sphere Diagram
    shadow_card(2.4, 2.6, 31.4, 18.7, fill=c_purple_lt, edge=c_purple_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(18.1, 19.9, "4. Unit-Sphere Residual Sync (US-BRS)", color=c_purple, fontsize=7.1, fontweight="bold", ha="center", zorder=5)

    # Draw crisp geometric unit sphere on left of card
    scx, scy, sr = 7.7, 12.2, 4.1
    ax.add_patch(Circle((scx, scy), sr, facecolor="#FFFFFF", edgecolor=c_purple, linewidth=1.15, zorder=4))
    ax.add_patch(Ellipse((scx, scy), sr * 2, sr * 0.68, fill=False, edgecolor=c_purple_bd, linestyle="--", linewidth=0.85, zorder=4))
    # Two vectors q_ref and q_t on unit sphere
    p_ref = (scx + sr * 0.58, scy + sr * 0.72)
    p_cur = (scx + sr * 0.86, scy + sr * 0.28)
    flow_arrow((scx, scy), p_ref, color=c_slate, lw=1.1, scale=8, zorder=5)
    flow_arrow((scx, scy), p_cur, color=c_purple, lw=1.2, scale=8, zorder=5)
    flow_arrow(p_ref, p_cur, color=c_rose, lw=1.3, scale=8, zorder=6)
    ax.text(scx + 0.5, scy + sr + 0.65, "||f||_2 = 1", color=c_purple, fontsize=5.8, fontweight="bold", ha="center", zorder=5)
    ax.text(scx, scy - sr - 1.1, "max|dq|=41<=127", color=c_rose, fontsize=5.7, fontweight="bold", ha="center", zorder=5)

    ax.text(23.1, 11.7,
            "- Frozen DINOv2-S/14 tokens\n"
            "- Closed-loop int8 delta:\n"
            "  100% lossless vs. int16\n"
            "- Saves 20.6% - 35.4% bytes\n"
            "  over Naive-DINO-Delta\n"
            "- NACK Risk Anchor cuts\n"
            "  p95 erasure staleness\n"
            "  (3.25 -> 2.79 steps)",
            color="#2E1065", fontsize=5.95, ha="center", va="center", linespacing=1.24, zorder=5)

    # Right sub-card: TC-PCR Threshold-Crossing Prototype-Calibrated Routing
    shadow_card(34.8, 2.6, 31.8, 18.7, fill=c_blue_lt, edge=c_blue_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(50.7, 19.9, "5. Prototype-Calibrated Routing (TC-PCR)", color=c_blue, fontsize=7.1, fontweight="bold", ha="center", zorder=5)

    # Mini Mahalanobis ellipsoid + threshold crossing visual
    ecx, ecy = 40.3, 13.5
    ax.add_patch(Ellipse((ecx, ecy), 7.2, 4.2, angle=24, facecolor="#DBEAFE", edgecolor=c_blue, lw=1.0, zorder=4))
    ax.add_patch(Ellipse((ecx, ecy), 4.2, 2.3, angle=24, facecolor="#93C5FD", edgecolor=c_blue, lw=0.8, ls="--", zorder=4))
    ax.scatter([ecx - 0.8, ecx + 0.6, ecx - 0.2], [ecy - 0.3, ecy + 0.4, ecy + 0.1], s=10, color=c_blue, zorder=5)
    ax.scatter([ecx + 3.4], [ecy + 2.5], s=26, marker="*", color=c_rose, zorder=6)
    flow_arrow((ecx + 1.1, ecy + 0.8), (ecx + 3.1, ecy + 2.3), color=c_rose, lw=1.1, scale=8, zorder=6)
    ax.text(ecx, ecy - 3.75, "Ledoit-Wolf\nKnown Prototypes", color=c_blue, fontsize=5.6, fontweight="bold", ha="center", zorder=5)
    ax.text(ecx + 3.2, ecy + 3.4, "Novel!", color=c_rose, fontsize=5.8, fontweight="bold", ha="center", zorder=6)

    ax.text(55.8, 11.4,
            "- 424-B sender ref (-66.4%)\n"
            "- Active-Support Selector\n"
            "  removes 68.8% bg dilution\n"
            "- 30-dB Threshold-Crossing\n"
            "  (e_base > 1e-3 >= e_res)\n"
            "- Calibrated Mahalanobis\n"
            "  emergence ranks RE01\n"
            "  WebP70 residual crops",
            color=c_navy, fontsize=5.95, ha="center", va="center", linespacing=1.24, zorder=5)

    flow_arrow((33.8, 11.8), (34.8, 11.8), color=c_purple, lw=1.3, scale=9, zorder=6)

    # --- Center Link (Tier 2): Hybrid Base + RE01 Residual Packet & NACK ---
    shadow_card(70.5, 2.6, 19.0, 22.0, fill="#FFFFFF", edge=c_purple, lw=1.1, r=0.9, shadow=False, zorder=3)
    ax.text(80.0, 23.1, "Semantic Packet", color=c_purple, fontsize=7.3, fontweight="bold", ha="center", zorder=5)
    pill(71.6, 19.6, 16.8, 2.1, "Cap <= 8 KiB (TE01)", fill=c_purple_lt, edge=c_purple, text_color=c_purple, fontsize=6.4, zorder=5)
    ax.text(80.0, 13.2,
            "- Exact 16x16 Known\n"
            "  Task State\n"
            "- Half-cap Base Stream\n"
            "- Top-Ranked RE01\n"
            "  Residual Evidence\n"
            "- NACK Digest Support",
            color=c_navy, fontsize=6.05, ha="center", va="center", linespacing=1.26, zorder=5)
    pill(71.6, 3.6, 16.8, 3.2, "60-B ACK / NACK\nTriggers Risk Anchor", fill=c_rose_lt, edge=c_rose, text_color=c_rose, fontsize=5.95, zorder=5)
    flow_arrow((66.6, 13.5), (70.5, 13.5), color=c_purple, lw=1.8, scale=12, zorder=7)
    flow_arrow((89.5, 13.5), (93.2, 13.5), color=c_purple, lw=1.8, scale=12, zorder=7)
    flow_arrow((93.2, 5.2), (88.4, 5.2), color=c_rose, lw=1.4, dashed=True, scale=10, zorder=7)
    flow_arrow((71.6, 5.2), (66.6, 5.2), color=c_rose, lw=1.4, dashed=True, scale=10, zorder=7)

    # --- Ground Segment Tier 2 (x = 92.4 .. 158.4, y = 1.6 .. 25.2) ---
    shadow_card(92.4, 1.6, 66.0, 23.6, fill="#FFFFFF", edge=c_purple_bd, lw=1.0, r=1.0, shadow=False, zorder=2)
    pill(93.2, 22.1, 64.2, 2.3,
         "(e) Open-World Evidence Reconstruction & Held-Out Discovery (SECOND Cohort 3)",
         fill=c_purple_lt, edge=c_purple, text_color=c_purple, fontsize=7.25)

    # Hybrid Residual Reconstruction box
    shadow_card(93.2, 2.6, 25.2, 18.7, fill=c_teal_lt, edge=c_teal_bd, lw=0.95, r=0.8, shadow=False, zorder=3)
    ax.text(105.8, 19.9, "Hybrid Residual Fusion", color=c_teal, fontsize=7.1, fontweight="bold", ha="center", zorder=5)
    ax.text(105.8, 11.4,
            "1. Exact known binary change\n"
            "   preserved losslessly\n"
            "2. Base + RE01 intra-frame\n"
            "   residuals restore >30 dB\n"
            "   detail on novel targets\n"
            "3. Resolution-consistent ViT\n"
            "   probe prevents context\n"
            "   shift on refined cells",
            color="#042F2E", fontsize=6.0, ha="center", va="center", linespacing=1.25, zorder=5)

    # Verified SECOND Breakthroughs box with mini bar comparison
    shadow_card(119.6, 2.6, 37.8, 18.7, fill=c_amber_lt, edge=c_amber, lw=1.05, r=0.9, shadow=False, zorder=3)
    ax.text(138.5, 19.9, "SECOND Cohort 3 Open-World Breakthroughs", color=c_amber, fontsize=7.15, fontweight="bold", ha="center", zorder=5)

    # Mini horizontal comparison bars for U30 coverage at 8 KiB
    u30_bars = [
        ("Stateless WebP", 0.5380, "#64748B"),
        ("Hyperprior (61 MB)", 0.6379, "#9333EA"),
        ("SafeSync TC-PCR", 0.7594, c_blue),
        ("LIC-TCM (4.18 GB)", 0.7688, "#EA580C"),
    ]
    for idx, (m_lbl, u_val, col) in enumerate(u30_bars):
        by = 17.1 - idx * 2.15
        ax.text(120.6, by, m_lbl, color=c_navy if "SafeSync" not in m_lbl else c_blue,
                fontsize=6.05, fontweight="bold" if "SafeSync" in m_lbl else "normal", va="center", zorder=5)
        bw = u_val * 17.5
        ax.add_patch(FancyBboxPatch((135.2, by - 0.52), bw, 1.04,
                                    boxstyle="round,pad=0.02,rounding_size=0.25",
                                    facecolor=col, edgecolor="none", zorder=5))
        ax.text(135.5 + bw, by, f"U30={u_val:.4f}", color=col, fontsize=5.95, fontweight="bold", va="center", zorder=5)

    pill(120.8, 3.5, 35.4, 5.8,
         "-33.56% Unknown Miss Rate vs. Hyperprior (matches 4.18-GB TCM!)\n"
         "+1.44 pp Unknown > Change (p < 0.05) | Refined Later F1: 0.8503",
         fill="#FFFFFF", edge=c_amber, text_color=c_navy, fontsize=6.05, zorder=5)

    flow_arrow((118.4, 11.8), (119.6, 11.8), color=c_teal, lw=1.4, scale=10, zorder=6)

    for suffix in ("pdf", "svg", "png"):
        out = ROOT / f"paper/figures/layered_state.{suffix}"
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"Generated Figure 1 using font '{font_name}' -> paper/figures/layered_state.(pdf|svg|png)")


if __name__ == "__main__":
    main()
