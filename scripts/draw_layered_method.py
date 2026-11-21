"""Publication-grade overall framework diagram for SafeSync (Figure 1).

Rendered at TRUE 1:1 IEEE double-column width (7.16 x 4.55 inches) so that every
Comic Sans MS font size (6.6--8.6 pt) is the exact physical print size on the page.
Combines real SpaceNet 7 RGB/coarse/detail patches, isometric satellite & ground
station illustrations, a 6x6 spatial routing grid, unit-sphere geometry, and
Mahalanobis prototype calibration diagrams with zero overlapping elements.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
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
    white_mask = (rgba[:, :, 0] > 244) & (rgba[:, :, 1] > 244) & (rgba[:, :, 2] > 244)
    rgba[white_mask, 3] = 0

    def trim(sub):
        ys, xs = np.where(sub[:, :, 3] > 15)
        if len(xs) == 0:
            return sub
        pad = 6
        return sub[max(0, ys.min() - pad):min(sub.shape[0], ys.max() + pad),
                   max(0, xs.min() - pad):min(sub.shape[1], xs.max() + pad)]

    return trim(rgba[:, : int(w * 0.49)]), trim(rgba[:, int(w * 0.51) :])


def _load_real_rs_patches():
    crops_path = ROOT / "paper/figures/receiver_crops.png"
    if crops_path.is_file():
        img = np.array(Image.open(crops_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        patch = img[int(h * 0.56):int(h * 0.92), int(w * 0.065):int(w * 0.275)]
        patch = np.array(
            Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8)).resize((128, 128), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
    else:
        rng = np.random.default_rng(2026)
        patch = np.clip(0.45 + 0.18 * rng.standard_normal((128, 128, 3)), 0, 1)
    coarse_small = patch.reshape(8, 16, 8, 16, 3).mean(axis=(1, 3))
    coarse = np.repeat(np.repeat(coarse_small, 16, axis=0), 16, axis=1)
    detail_vis = np.clip(0.5 + 3.2 * (patch - coarse), 0.0, 1.0)
    return patch, coarse, detail_vis


def main() -> None:
    font_name = _load_comic_sans()
    plt.rcParams.update({
        "font.family": font_name,
        "font.sans-serif": [font_name, "DejaVu Sans"],
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # Exact 1:1 IEEE TGRS \textwidth = 7.16 in -> NO LaTeX downscaling, true 7-9 pt fonts!
    fig, ax = plt.subplots(figsize=(7.16, 4.52), dpi=300)
    fig.subplots_adjust(left=0.003, right=0.997, bottom=0.005, top=0.995)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 63.2)
    ax.axis("off")

    # Refined academic color palette
    c_navy = "#0F172A"
    c_slate = "#334155"
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

    def card(x, y, w, h, fill="#FFFFFF", edge="#CBD5E1", lw=0.95, r=0.85, shadow=True, zorder=1):
        if shadow:
            ax.add_patch(FancyBboxPatch(
                (x + 0.22, y - 0.22), w, h,
                boxstyle=f"round,pad=0.08,rounding_size={r}",
                facecolor="#CBD5E1", edgecolor="none", alpha=0.40, zorder=zorder
            ))
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.08,rounding_size={r}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder + 1
        ))

    def badge(x, y, w, h, text, fill, edge, color="#FFFFFF", fs=7.0, zorder=5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.05,rounding_size={min(0.75, h/2):.2f}",
            facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=zorder
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=color, fontsize=fs, fontweight="bold", linespacing=1.15, zorder=zorder + 1)

    def arrow(p0, p1, color=c_navy, lw=1.2, rad=0.0, dashed=False, scale=9.5, zorder=7):
        ax.add_patch(FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>", mutation_scale=scale,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (3.2, 1.8)) if dashed else "-",
            zorder=zorder
        ))

    def img_box(arr, x, y, w, h, edge=c_navy, lw=0.8, zorder=4):
        ax.imshow(arr, extent=(x, x + w, y, y + h), origin="upper", interpolation="bilinear", zorder=zorder, aspect="auto")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, linewidth=lw, zorder=zorder + 1))

    sat_icon, gnd_icon = _load_satellite_and_ground_icons()
    patch_rgb, patch_coarse, patch_detail = _load_real_rs_patches()

    # =========================================================================
    # TOP STRIP (y = 54.4 .. 62.8): PARADIGM SHIFT & STATE FOOTPRINT CONTRAST
    # =========================================================================
    card(0.4, 54.4, 99.2, 8.4, fill="#F8FAFC", edge="#94A3B8", lw=1.0, r=0.9)

    card(1.1, 55.0, 29.8, 7.2, fill=c_rose_lt, edge=c_rose_bd, lw=0.85, r=0.65, shadow=False, zorder=3)
    ax.text(16.0, 60.9, "Prior Downlink Bottlenecks", color=c_rose, fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(16.0, 57.6,
            "Stateless WebP: F1 = 0.279 (blur)\n"
            "4x WebP Ref: 1,495 viol. & 79.8% drift\n"
            "LIC-TCM: 4.18 GB & 30.9 s GPU search",
            color="#4C0519", fontsize=6.4, ha="center", va="center", linespacing=1.22, zorder=5)

    arrow((31.2, 58.6), (34.4, 58.6), color=c_blue, lw=1.6, scale=10.5)
    ax.text(32.8, 60.2, "Shift", color=c_blue, fontsize=6.6, fontweight="bold", ha="center", zorder=6)

    card(34.7, 55.0, 34.2, 7.2, fill=c_green_lt, edge=c_green_bd, lw=0.85, r=0.65, shadow=False, zorder=3)
    ax.text(51.8, 60.9, "SafeSync World-State Co-Construction", color=c_green, fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(51.8, 57.6,
            "ZFSS: 32-bit seed -> 0 B matrix (-99.09%)\n"
            "ASF+: #1 OOF F1 (0.476/0.522) in 118 ms CPU\n"
            "TC-PCR: U30=0.823 (+5.5 pp vs. 4.18G TCM)",
            color="#052E16", fontsize=6.3, ha="center", va="center", linespacing=1.22, zorder=5)

    # Right mini bar chart: Single-Area Persistent State (N=1)
    card(69.6, 55.0, 29.3, 7.2, fill="#FFFFFF", edge="#CBD5E1", lw=0.85, r=0.65, shadow=False, zorder=3)
    ax.text(84.25, 61.0, "Onboard State (N = 1 Area)", color=c_navy, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=5)
    for idx, (lbl, bw, col, vtxt) in enumerate([
        ("Cached Matrix", 11.2, "#94A3B8", "2.37 MB"),
        ("PNG History", 6.4, "#64748B", "1.35 MB"),
        ("SafeSync ZFSS", 0.7, c_green, "12.3 KB (-99.1%)"),
    ]):
        by = 59.1 - idx * 1.65
        ax.text(70.3, by, lbl, color=c_green if idx == 2 else c_slate,
                fontsize=6.2, fontweight="bold" if idx == 2 else "normal", va="center", zorder=5)
        ax.add_patch(FancyBboxPatch((80.6, by - 0.42), bw, 0.84,
                                    boxstyle="round,pad=0.01,rounding_size=0.2",
                                    facecolor=col, edgecolor="none", zorder=5))
        ax.text(81.0 + bw, by, vtxt, color=c_green if idx == 2 else c_navy,
                fontsize=6.15, fontweight="bold", va="center", zorder=5)

    # =========================================================================
    # THREE COLUMNS (y = 0.4 .. 53.6): SPACE | ASYMMETRIC LINK | GROUND
    # =========================================================================
    card(0.4, 0.4, 42.8, 53.1, fill="#F8FAFC", edge=c_blue_bd, lw=1.1, r=1.0)
    card(43.8, 0.4, 13.4, 53.1, fill="#FFFBEB", edge=c_amber_bd, lw=1.0, r=0.95)
    card(57.8, 0.4, 41.8, 53.1, fill="#F8FAFC", edge=c_teal_bd, lw=1.1, r=1.0)

    badge(1.2, 50.3, 33.2, 2.5, "SPACE: LEO Sender (CPU-Only, 0 GPU)", fill=c_blue, edge=c_blue, fs=7.3)
    badge(44.4, 50.3, 12.2, 2.5, "SPACE-GROUND", fill=c_amber, edge=c_amber, fs=6.8)
    badge(58.6, 50.3, 32.8, 2.5, "GROUND: World-State Archive", fill=c_teal, edge=c_teal, fs=7.3)

    if sat_icon is not None:
        ax.imshow(sat_icon, extent=(35.0, 42.7, 47.7, 53.3), zorder=8, aspect="auto")
    if gnd_icon is not None:
        ax.imshow(gnd_icon, extent=(91.8, 99.2, 47.7, 53.3), zorder=8, aspect="auto")

    # =========================================================================
    # TIER 1 (y = 22.8 .. 49.5): PIXEL ARCHIVE SYNCHRONIZATION
    # =========================================================================
    # --- Space Tier 1 ---
    card(1.1, 22.9, 41.4, 26.7, fill="#FFFFFF", edge="#BFDBFE", lw=0.85, r=0.8, shadow=False, zorder=2)
    badge(1.7, 46.9, 32.8, 2.05, "(a) Tier 1: Pixel Sync (ASF + ZFSS + DGAD)",
          fill=c_blue_lt, edge=c_blue, color=c_blue, fs=6.9)

    # Pass RGB + Coarse/Detail thumbnails
    card(1.7, 34.4, 15.8, 11.9, fill=c_blue_lt, edge=c_blue_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(9.6, 45.2, "Pass x_t & ASF Split", color=c_navy, fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=5)
    img_box(patch_rgb, 2.2, 37.3, 5.6, 6.6, edge=c_navy)
    ax.text(5.0, 36.0, "RGB x_t", color=c_navy, fontsize=6.1, fontweight="bold", ha="center", va="center", zorder=5)
    img_box(patch_coarse, 8.6, 40.9, 4.1, 3.3, edge=c_blue)
    ax.text(15.1, 42.5, "c_t\nLB01", color=c_blue, fontsize=5.9, fontweight="bold", ha="center", va="center", zorder=5)
    img_box(patch_detail, 8.6, 36.2, 4.1, 3.3, edge=c_teal)
    ax.text(15.1, 37.8, "H x_t\nDetail", color=c_teal, fontsize=5.9, fontweight="bold", ha="center", va="center", zorder=5)

    # ZFSS Seeded Sketching card
    card(1.7, 23.6, 15.8, 10.1, fill=c_green_lt, edge=c_green_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(9.6, 32.6, "ZFSS Sketch s_t", color=c_green, fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)
    badge(2.15, 28.5, 6.7, 2.5, "32-bit Seed", fill="#FFFFFF", edge=c_green, color=c_green, fs=5.55)
    arrow((8.95, 29.75), (10.15, 29.75), color=c_green, lw=1.1, scale=8)
    badge(10.25, 28.5, 6.7, 2.5, "q16 (48 B)", fill=c_green, edge=c_green, color="#FFFFFF", fs=5.65)
    ax.text(9.6, 25.8, "Stream A: 0 B Matrix\n12.3 KB/AOI (-99.09%)",
            color="#052E16", fontsize=5.95, fontweight="bold", ha="center", va="center", linespacing=1.2, zorder=5)

    # DGAD Dual-Gate + 5x5 Block Map
    card(18.3, 27.1, 23.5, 19.2, fill="#FFFFFF", edge=c_blue, lw=0.9, r=0.7, shadow=False, zorder=3)
    ax.text(30.05, 45.1, "DGAD Dual-Gate & Packer", color=c_blue, fontsize=6.75, fontweight="bold", ha="center", va="center", zorder=5)

    card(19.0, 37.2, 13.4, 6.7, fill=c_green_lt, edge=c_green, lw=0.75, r=0.55, shadow=False, zorder=4)
    ax.text(25.7, 42.6, "Safety Gate (Prop. 1)", color=c_green, fontsize=5.95, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(25.7, 39.4, "L_b <= tau -> STABLE\n0 False-Stable Viol.",
            color="#052E16", fontsize=5.8, ha="center", va="center", linespacing=1.2, zorder=5)

    card(19.0, 29.5, 13.4, 6.8, fill=c_blue_lt, edge=c_blue, lw=0.75, r=0.55, shadow=False, zorder=4)
    ax.text(25.7, 35.0, "Utility Gate (Point Est.)", color=c_blue, fontsize=5.9, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(25.7, 31.8, "Rank ||z_b||_2^2 / k\nPack JPEG 4:4:4 (LA01)",
            color=c_navy, fontsize=5.8, ha="center", va="center", linespacing=1.2, zorder=5)

    # Crisp 5x5 Block Routing Grid
    gx0, gy0, cs = 33.2, 34.4, 1.56
    gmap = [
        [0, 1, 1, 0, 0],
        [0, 2, 1, 2, 0],
        [2, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 0, 2, 0, 0],
    ]
    scols = {0: "#86EFAC", 1: "#2563EB", 2: "#FDE68A"}
    for r_i in range(5):
        for c_i in range(5):
            ax.add_patch(Rectangle(
                (gx0 + c_i * cs, gy0 + r_i * cs), cs - 0.12, cs - 0.12,
                facecolor=scols[gmap[4 - r_i][c_i]], edgecolor=c_navy, linewidth=0.4, zorder=5
            ))
    ax.text(gx0 + 2.5 * cs, gy0 + 5 * cs + 0.65, "Block Map", color=c_navy, fontsize=6.1, fontweight="bold", ha="center", zorder=5)
    for idx, (col, txt) in enumerate([("#86EFAC", "STABLE"), ("#2563EB", "UPDATED"), ("#FDE68A", "DEFERRED")]):
        ly = 33.0 - idx * 1.35
        ax.add_patch(Rectangle((33.2, ly - 0.38), 1.05, 0.76, facecolor=col, edgecolor=c_navy, lw=0.35, zorder=5))
        ax.text(34.55, ly, txt, color=c_navy, fontsize=5.75, va="center", zorder=5)

    # Dedicated Self-Decode pill
    badge(18.3, 23.6, 23.5, 2.65, "Self-Decode LA01 -> Commit on ACK",
          fill="#F1F5F9", edge=c_teal, color=c_teal, fs=5.85)
    arrow((30.05, 27.1), (30.05, 26.25), color=c_teal, lw=1.0, scale=7.5)
    arrow((18.3, 24.9), (17.5, 26.2), color=c_teal, lw=1.1, dashed=True, scale=8)
    arrow((17.5, 40.3), (18.3, 40.3), color=c_blue, lw=1.15, scale=8.5)
    arrow((17.5, 29.8), (18.3, 33.0), color=c_green, lw=1.15, scale=8.5)

    # --- Link Tier 1 ---
    card(44.5, 33.2, 12.0, 14.8, fill="#FFFFFF", edge=c_amber, lw=0.9, r=0.7, shadow=False, zorder=3)
    ax.text(50.5, 46.5, "Downlink", color=c_amber, fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=5)
    badge(45.2, 43.3, 10.6, 2.0, "<= 32 KiB", fill=c_amber_lt, edge=c_amber, color=c_amber, fs=6.4)
    ax.text(50.5, 38.2,
            "WebP85 c_t\nJPEG 4:4:4\nDigest+CRC\n(118 ms CPU)",
            color=c_navy, fontsize=6.05, ha="center", va="center", linespacing=1.22, zorder=5)
    arrow((41.8, 40.5), (44.5, 40.5), color=c_amber, lw=1.5, scale=9.5)
    arrow((56.5, 40.5), (58.9, 40.5), color=c_amber, lw=1.5, scale=9.5)

    card(44.5, 23.6, 12.0, 8.2, fill=c_teal_lt, edge=c_teal, lw=0.9, r=0.7, shadow=False, zorder=3)
    ax.text(50.5, 30.3, "60-B ACK", color=c_teal, fontsize=6.9, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(50.5, 26.6, "Commit Digest\n0 Image Uplink",
            color="#042F2E", fontsize=6.0, fontweight="bold", ha="center", va="center", linespacing=1.2, zorder=5)
    arrow((58.9, 27.2), (56.5, 27.2), color=c_teal, lw=1.3, dashed=True, scale=9)
    arrow((44.5, 27.2), (41.8, 25.1), color=c_teal, lw=1.3, dashed=True, scale=9)

    # --- Ground Tier 1 ---
    card(58.5, 22.9, 40.4, 26.7, fill="#FFFFFF", edge="#99F6E4", lw=0.85, r=0.8, shadow=False, zorder=2)
    badge(59.1, 46.9, 32.2, 2.05, "(b) Ground ASF & Reconstructed State",
          fill=c_teal_lt, edge=c_teal, color=c_teal, fs=6.9)

    card(59.1, 36.8, 15.6, 9.4, fill=c_blue_lt, edge=c_blue_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(66.9, 44.9, "Coarse Field U c_t", color=c_blue, fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(66.9, 40.4, "Absorbs orbital drift;\n4.5x fewer false\nbackground triggers",
            color=c_navy, fontsize=6.0, ha="center", va="center", linespacing=1.2, zorder=5)

    card(59.1, 23.6, 15.6, 12.2, fill="#F1F5F9", edge=c_navy, lw=0.85, r=0.65, shadow=False, zorder=3)
    ax.text(66.9, 34.4, "Immutable Archive a_t", color=c_navy, fontsize=6.65, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(66.9, 28.6, "Stores ACK'd absolute\nblocks C(x_{s,b}) only;\nnever overwritten!\n|S_{b,t}| <= t + 1",
            color=c_navy, fontsize=5.95, ha="center", va="center", linespacing=1.2, zorder=5)

    card(75.6, 23.6, 22.6, 22.6, fill=c_green_lt, edge=c_green, lw=0.95, r=0.75, shadow=False, zorder=3)
    ax.text(86.9, 44.8, "Co-Constructed Render", color=c_green, fontsize=6.9, fontweight="bold", ha="center", va="center", zorder=5)
    badge(76.5, 41.5, 20.8, 2.1, "x_hat_t = R( H a_t + U c_t )", fill="#FFFFFF", edge=c_green, color=c_green, fs=6.4)

    img_box(patch_rgb, 76.5, 31.8, 7.6, 8.7, edge=c_green, lw=0.95)
    for bx, by in [(80.2, 37.8), (81.3, 37.1), (79.1, 34.0)]:
        ax.add_patch(Rectangle((bx, by), 1.0, 0.9, fill=False, edgecolor="#06B6D4", linewidth=1.1, zorder=6))

    ax.text(91.3, 36.1,
            "SpaceNet 7 OOF\n(ASF+ CPU #1):\n32K: 0.4760\n(+3.9 pp TCM)\n128K: 0.5216",
            color="#052E16", fontsize=5.95, fontweight="bold", ha="center", va="center", linespacing=1.22, zorder=5)
    badge(76.5, 24.5, 20.8, 5.8,
          "0 Violations / 118,317 Blocks\n263x Faster than GPU LIC-TCM",
          fill="#FFFFFF", edge=c_green_bd, color=c_navy, fs=6.05)

    arrow((74.7, 41.2), (75.6, 39.2), color=c_blue, lw=1.15, scale=8.5)
    arrow((74.7, 30.0), (75.6, 33.8), color=c_navy, lw=1.15, scale=8.5)

    # =========================================================================
    # TIER 2 (y = 1.1 .. 21.9): SEMANTIC & OPEN-WORLD CO-CONSTRUCTION
    # =========================================================================
    # --- Space Tier 2 ---
    card(1.1, 1.1, 41.4, 20.8, fill="#FFFFFF", edge=c_purple_bd, lw=0.85, r=0.8, shadow=False, zorder=2)
    badge(1.7, 19.2, 40.2, 2.05, "(c) Tier 2: Semantic & Open-World Sync (US-BRS + TC-PCR)",
          fill=c_purple_lt, edge=c_purple, color=c_purple, fs=6.3)

    # US-BRS card with Unit Sphere
    card(1.7, 1.8, 19.6, 16.7, fill=c_purple_lt, edge=c_purple_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(11.5, 17.3, "US-BRS Unit-Sphere Sync", color=c_purple, fontsize=6.65, fontweight="bold", ha="center", va="center", zorder=5)
    scx, scy, sr = 5.4, 10.2, 2.95
    ax.add_patch(Circle((scx, scy), sr, facecolor="#FFFFFF", edgecolor=c_purple, linewidth=1.0, zorder=4))
    ax.add_patch(Ellipse((scx, scy), sr * 2, sr * 0.66, fill=False, edgecolor=c_purple_bd, linestyle="--", linewidth=0.75, zorder=4))
    p_ref = (scx + sr * 0.55, scy + sr * 0.72)
    p_cur = (scx + sr * 0.86, scy + sr * 0.25)
    arrow((scx, scy), p_ref, color=c_slate, lw=0.95, scale=6.5)
    arrow((scx, scy), p_cur, color=c_purple, lw=1.0, scale=6.5)
    arrow(p_ref, p_cur, color=c_rose, lw=1.1, scale=6.5)
    ax.text(scx, scy - sr - 1.15, "|dq|<=41", color=c_rose, fontsize=5.8, fontweight="bold", ha="center", zorder=5)
    ax.text(15.0, 9.6,
            "DINOv2-S/14\nLossless int8\ndelta: -20.6%\nto -35.4% bytes\nNACK Anchor",
            color="#2E1065", fontsize=5.95, ha="center", va="center", linespacing=1.22, zorder=5)

    # TC-PCR card with Prototype Ellipsoid
    card(22.1, 1.8, 19.8, 16.7, fill=c_blue_lt, edge=c_blue_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(32.0, 17.3, "TC-PCR Prototype Router", color=c_blue, fontsize=6.65, fontweight="bold", ha="center", va="center", zorder=5)
    ecx, ecy = 26.0, 11.2
    ax.add_patch(Ellipse((ecx, ecy), 5.2, 3.1, angle=22, facecolor="#DBEAFE", edgecolor=c_blue, lw=0.9, zorder=4))
    ax.scatter([ecx - 0.5, ecx + 0.5], [ecy - 0.2, ecy + 0.3], s=8, color=c_blue, zorder=5)
    ax.scatter([ecx + 2.4], [ecy + 2.0], s=20, marker="*", color=c_rose, zorder=6)
    ax.text(ecx, ecy - 2.9, "Ledoit-Wolf", color=c_blue, fontsize=5.65, fontweight="bold", ha="center", zorder=5)
    ax.text(36.2, 9.4,
            "424-B Ref (-66%)\nActive-Support\n30-dB Crossing\nCalib. Novelty\nRE01 Residuals",
            color=c_navy, fontsize=5.95, ha="center", va="center", linespacing=1.22, zorder=5)
    arrow((21.3, 10.0), (22.1, 10.0), color=c_purple, lw=1.1, scale=8)

    # --- Link Tier 2 ---
    card(44.5, 1.8, 12.0, 19.4, fill="#FFFFFF", edge=c_purple, lw=0.9, r=0.7, shadow=False, zorder=3)
    ax.text(50.5, 19.7, "TE01 Frame", color=c_purple, fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=5)
    badge(45.2, 16.7, 10.6, 1.9, "<= 8 KiB", fill=c_purple_lt, edge=c_purple, color=c_purple, fs=6.3)
    ax.text(50.5, 11.7,
            "Exact 16x16\nTask State +\nBase + RE01\nResidual ROIs",
            color=c_navy, fontsize=5.95, ha="center", va="center", linespacing=1.22, zorder=5)
    badge(45.2, 2.6, 10.6, 3.6, "ACK / NACK\nRisk Anchor", fill=c_rose_lt, edge=c_rose, color=c_rose, fs=5.9)
    arrow((41.9, 11.5), (44.5, 11.5), color=c_purple, lw=1.5, scale=9.5)
    arrow((56.5, 11.5), (58.9, 11.5), color=c_purple, lw=1.5, scale=9.5)
    arrow((58.9, 4.4), (55.8, 4.4), color=c_rose, lw=1.2, dashed=True, scale=8.5)
    arrow((45.2, 4.4), (41.9, 4.4), color=c_rose, lw=1.2, dashed=True, scale=8.5)

    # --- Ground Tier 2 ---
    card(58.5, 1.1, 40.4, 20.8, fill="#FFFFFF", edge=c_purple_bd, lw=0.85, r=0.8, shadow=False, zorder=2)
    badge(59.1, 19.2, 39.2, 2.05, "(d) Open-World Semantic Discovery (SECOND Cohort 3)",
          fill=c_purple_lt, edge=c_purple, color=c_purple, fs=6.35)

    card(59.1, 1.8, 14.2, 16.7, fill=c_teal_lt, edge=c_teal_bd, lw=0.8, r=0.65, shadow=False, zorder=3)
    ax.text(66.2, 17.2, "Hybrid Fusion", color=c_teal, fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(66.2, 9.5,
            "Exact known\nchange +\nBase + RE01\n>30 dB novel\nROI recovery",
            color="#042F2E", fontsize=6.0, ha="center", va="center", linespacing=1.24, zorder=5)

    card(74.0, 1.8, 24.3, 16.7, fill=c_amber_lt, edge=c_amber, lw=0.9, r=0.7, shadow=False, zorder=3)
    ax.text(86.15, 17.2, "SECOND 8-KiB Held-Out U30", color=c_amber, fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=5)

    for idx, (mlbl, uval, col) in enumerate([
        ("WebP", 0.5380, "#64748B"),
        ("Hyperprior", 0.6379, "#9333EA"),
        ("LIC-TCM", 0.7688, "#D55E00"),
        ("TC-PCR(TCM)", 0.8233, c_blue),
    ]):
        by = 14.8 - idx * 2.0
        is_best = mlbl == "TC-PCR(TCM)"
        ax.text(74.5, by, mlbl, color=c_blue if is_best else c_navy,
                fontsize=5.7 if is_best else 5.95, fontweight="bold" if is_best else "normal", va="center", zorder=5)
        bw = uval * 9.8
        ax.add_patch(FancyBboxPatch((82.6, by - 0.48), bw, 0.96,
                                    boxstyle="round,pad=0.01,rounding_size=0.2",
                                    facecolor=col, edgecolor="none", zorder=5))
        ax.text(82.9 + bw, by, f"{uval:.3f}", color=col, fontsize=5.9, fontweight="bold", va="center", zorder=5)

    badge(74.7, 2.5, 22.9, 5.1,
          "#1 U30: +5.45 pp vs. 4.18G LIC-TCM\n+15.3 pp WebP | FNR 0.157 | F1 0.850",
          fill="#FFFFFF", edge=c_amber, color=c_navy, fs=5.55)
    arrow((73.3, 10.0), (74.0, 10.0), color=c_teal, lw=1.15, scale=8.5)

    for suffix in ("pdf", "svg", "png"):
        out = ROOT / f"paper/figures/layered_state.{suffix}"
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"Generated 1:1 IEEE scale Figure 1 using font '{font_name}'")


if __name__ == "__main__":
    main()
