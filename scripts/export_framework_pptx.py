"""Export a true 1:1 native-vector editable PowerPoint (.pptx) replica of Figure 1 (SafeSync Overview).

Every rounded card, header badge, text block, directional arrow, satellite/ground icon,
real SpaceNet 7 RGB/coarse/detail crop, 5x5 routing block map, unit sphere, Ledoit-Wolf
prototype diagram, and horizontal bar chart is constructed as native editable PowerPoint
shapes/images in Comic Sans MS font, preserving the exact 100 x 63.2 coordinate layout of
`paper/figures/layered_state.pdf` / `layered_state.svg`.
"""
from __future__ import annotations

import io
from pathlib import Path
import re
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FONT = "Comic Sans MS"

# Exact proportional slide size (10.00 x 6.32 in -> 1.0 unit = 0.10 in; matches 7.16 x 4.52 in 1:1 aspect ratio)
SLIDE_W = 10.00
SLIDE_H = 6.32
SCALE = SLIDE_W / 100.0
FONT_SCALE = SLIDE_W / 7.16  # Scales 7.16-in print font sizes 1:1 to 10-in PPTX canvas


def rgb(hex_str: str) -> RGBColor:
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def xywh(x: float, y: float, w: float, h: float):
    """Convert matplotlib bottom-left (x, y, w, h) in [0..100] x [0..63.2] to PPTX top-left Inches."""
    left = Inches(x * SCALE)
    top = Inches((63.2 - (y + h)) * SCALE)
    width = Inches(w * SCALE)
    height = Inches(h * SCALE)
    return left, top, width, height


def add_round_rect(slide, x, y, w, h, fill, edge, lw=0.9, adj=0.08):
    left, top, width, height = xywh(x, y, w, h)
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.adjustments[0] = adj
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if edge is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = rgb(edge)
        shp.line.width = Pt(lw * 1.1)
    shp.shadow.inherit = False
    return shp


_SUBSUP_RE = re.compile(r"(_\{[^}]+\}|\^\{[^}]+\})")


def _append_rich_runs(p, line: str, color: str, fs: float, bold: bool) -> None:
    """Render plain text plus `_{sub}` and `^{sup}` as true native PowerPoint sub/superscripts."""
    for tok in _SUBSUP_RE.split(line):
        if not tok:
            continue
        run = p.add_run()
        if tok.startswith("_{") and tok.endswith("}"):
            run.text = tok[2:-1]
            run._r.get_or_add_rPr().set("baseline", "-25000")
        elif tok.startswith("^{") and tok.endswith("}"):
            run.text = tok[2:-1]
            run._r.get_or_add_rPr().set("baseline", "30000")
        else:
            run.text = tok
        run.font.name = FONT
        run.font.size = Pt(fs * FONT_SCALE)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)


def add_text_box(slide, x, y, w, h, text, color="#0F172A", fs=7.0, bold=False, align=PP_ALIGN.CENTER):
    left, top, width, height = xywh(x, y, w, h)
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(1)
    tf.margin_right = Pt(1)
    tf.margin_top = Pt(0.5)
    tf.margin_bottom = Pt(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        _append_rich_runs(p, line, color, fs, bold)
    return tb


def add_badge(slide, x, y, w, h, text, fill, edge, color="#FFFFFF", fs=7.0):
    shp = add_round_rect(slide, x, y, w, h, fill, edge, lw=0.85, adj=0.22)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(0.5)
    tf.margin_bottom = Pt(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        _append_rich_runs(p, line, color, fs, True)
    return shp


def add_arrow_chevron(slide, x, y, w, h, color="#1D4ED8", direction="right"):
    left, top, width, height = xywh(x, y, w, h)
    shape_map = {
        "right": MSO_SHAPE.RIGHT_ARROW,
        "left": MSO_SHAPE.LEFT_ARROW,
        "down": MSO_SHAPE.DOWN_ARROW,
    }
    shp = slide.shapes.add_shape(shape_map[direction], left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(color)
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_pil_image(slide, pil_img: Image.Image, x, y, w, h, border_color=None):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    left, top, width, height = xywh(x, y, w, h)
    pic = slide.shapes.add_picture(buf, left, top, width=width, height=height)
    if border_color is not None:
        pic.line.color.rgb = rgb(border_color)
        pic.line.width = Pt(1.0)
    return pic


def load_icons_and_patches():
    sat_img = gnd_img = None
    icon_sheet = ROOT / "paper/figures/icons_space_ground.png"
    if icon_sheet.is_file():
        rgba = np.array(Image.open(icon_sheet).convert("RGBA"))
        h, w, _ = rgba.shape
        white = (rgba[:, :, 0] > 244) & (rgba[:, :, 1] > 244) & (rgba[:, :, 2] > 244)
        rgba[white, 3] = 0

        def trim(sub):
            ys, xs = np.where(sub[:, :, 3] > 15)
            if len(xs) == 0:
                return Image.fromarray(sub)
            pad = 6
            return Image.fromarray(sub[max(0, ys.min() - pad):min(sub.shape[0], ys.max() + pad),
                                       max(0, xs.min() - pad):min(sub.shape[1], xs.max() + pad)])

        sat_img = trim(rgba[:, : int(w * 0.49)])
        gnd_img = trim(rgba[:, int(w * 0.51):])

    gt_direct = ROOT / "data/spacenet7_pilot/train/L15-1848E-0793N_7394_5018_13/images_masked/global_monthly_2018_02_mosaic_L15-1848E-0793N_7394_5018_13.tif"
    crops_path = ROOT / "paper/figures/receiver_crops.png"
    patch = None
    if gt_direct.is_file():
        try:
            img = np.array(Image.open(gt_direct).convert("RGB"), dtype=np.float32) / 255.0
            patch = img[768:1024, 413:669]
        except Exception:
            patch = None
    if patch is None and crops_path.is_file():
        img = np.array(Image.open(crops_path).convert("RGB"), dtype=np.float32) / 255.0
        hh, ww, _ = img.shape
        patch = img[int(hh * 0.54):int(hh * 0.94), int(ww * 0.045):int(ww * 0.195)]
    if patch is not None:
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
    return (
        sat_img,
        gnd_img,
        Image.fromarray((patch * 255).astype(np.uint8)),
        Image.fromarray((coarse * 255).astype(np.uint8)),
        Image.fromarray((detail_vis * 255).astype(np.uint8)),
    )


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank slide

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

    sat_img, gnd_img, patch_rgb, patch_coarse, patch_detail = load_icons_and_patches()

    # TOP BANNER
    add_round_rect(slide, 0.4, 54.4, 99.2, 8.4, "#F8FAFC", "#94A3B8", lw=1.0, adj=0.08)
    add_round_rect(slide, 1.1, 55.0, 29.8, 7.2, c_rose_lt, c_rose_bd, lw=0.85, adj=0.10)
    add_text_box(slide, 1.2, 60.0, 29.6, 1.9, "Prior Downlink Bottlenecks", c_rose, fs=7.3, bold=True)
    add_text_box(slide, 1.2, 55.1, 29.6, 5.0,
                 "Stateless WebP: F1 = 0.279 (blur)\n4× WebP Ref: 1,495 viol. & 79.8% drift\nLIC-TCM: 4.18 GB & 30.9 s GPU search",
                 "#4C0519", fs=6.2)

    add_text_box(slide, 31.0, 59.6, 3.6, 1.8, "Shift", c_blue, fs=6.4, bold=True)
    add_arrow_chevron(slide, 31.3, 57.9, 3.0, 1.4, c_blue, "right")

    add_round_rect(slide, 34.7, 55.0, 34.2, 7.2, c_green_lt, c_green_bd, lw=0.85, adj=0.10)
    add_text_box(slide, 34.8, 60.0, 34.0, 1.9, "SafeSync World-State Co-Construction", c_green, fs=7.3, bold=True)
    add_text_box(slide, 34.8, 55.1, 34.0, 5.0,
                 "ZFSS: 32-bit seed → 0 B matrix (-99.09%)\nASF+: #1 OOF F1 (0.476/0.522) in 118 ms CPU\nTC-PCR: U_{30}=0.823 (+5.5 pp vs. 4.18G TCM)",
                 "#052E16", fs=6.15)

    add_round_rect(slide, 69.6, 55.0, 29.3, 7.2, "#FFFFFF", "#CBD5E1", lw=0.85, adj=0.10)
    add_text_box(slide, 69.8, 60.1, 28.9, 1.8, "Onboard State (N = 1 Area)", c_navy, fs=7.1, bold=True)
    for idx, (lbl, bw, col, vtxt) in enumerate([
        ("Cached Matrix", 10.6, "#94A3B8", "2.37 MB"),
        ("PNG History", 6.1, "#64748B", "1.35 MB"),
        ("SafeSync ZFSS", 0.8, c_green, "12.3 KB (-99.1%)"),
    ]):
        by = 58.6 - idx * 1.65
        add_text_box(slide, 70.0, by - 0.5, 10.5, 1.2, lbl, c_green if idx == 2 else c_slate,
                     fs=5.95, bold=(idx == 2), align=PP_ALIGN.LEFT)
        add_round_rect(slide, 80.7, by - 0.3, bw, 0.8, col, None, adj=0.25)
        add_text_box(slide, 80.9 + bw, by - 0.5, 17.5 - bw, 1.2, vtxt, c_green if idx == 2 else c_navy,
                     fs=5.9, bold=True, align=PP_ALIGN.LEFT)

    # THREE MAIN COLUMNS
    add_round_rect(slide, 0.4, 0.4, 42.8, 53.1, "#F8FAFC", c_blue_bd, lw=1.1, adj=0.03)
    add_round_rect(slide, 43.8, 0.4, 13.4, 53.1, "#FFFBEB", c_amber_bd, lw=1.0, adj=0.05)
    add_round_rect(slide, 57.8, 0.4, 41.8, 53.1, "#F8FAFC", c_teal_bd, lw=1.1, adj=0.03)

    add_badge(slide, 1.2, 50.3, 33.2, 2.5, "SPACE: LEO Sender (CPU-Only, 0 GPU)", c_blue, c_blue, fs=7.1)
    add_badge(slide, 44.4, 50.3, 12.2, 2.5, "SPACE-GROUND", c_amber, c_amber, fs=6.6)
    add_badge(slide, 58.6, 50.3, 32.8, 2.5, "GROUND: World-State Archive", c_teal, c_teal, fs=7.1)

    if sat_img is not None:
        add_pil_image(slide, sat_img, 35.1, 47.9, 7.5, 5.3)
    if gnd_img is not None:
        add_pil_image(slide, gnd_img, 91.9, 47.9, 7.3, 5.3)

    # TIER 1: SPACE
    add_round_rect(slide, 1.1, 22.9, 41.4, 26.7, "#FFFFFF", "#BFDBFE", lw=0.85, adj=0.04)
    add_badge(slide, 1.7, 46.9, 32.8, 2.05, "(a) Tier 1: Pixel Sync (ASF+ + ZFSS + DGAD)",
              c_blue_lt, c_blue, color=c_blue, fs=6.7)

    add_round_rect(slide, 1.7, 34.4, 15.8, 11.9, c_blue_lt, c_blue_bd, lw=0.8, adj=0.08)
    add_text_box(slide, 1.8, 44.3, 15.6, 1.8, "Pass x_{t} & ASF Split", c_navy, fs=6.6, bold=True)
    add_pil_image(slide, patch_rgb, 2.2, 37.3, 5.6, 6.6, border_color=c_navy)
    add_text_box(slide, 2.1, 35.1, 5.8, 1.8, "RGB x_{t}", c_navy, fs=5.9, bold=True)
    add_pil_image(slide, patch_coarse, 8.6, 40.9, 4.1, 3.3, border_color=c_blue)
    add_text_box(slide, 12.8, 40.8, 4.5, 3.4, "c_{t}\nWebP85", c_blue, fs=5.5, bold=True)
    add_pil_image(slide, patch_detail, 8.6, 36.2, 4.1, 3.3, border_color=c_teal)
    add_text_box(slide, 12.8, 36.1, 4.5, 3.4, "H x_{t}\nDetail", c_teal, fs=5.5, bold=True)

    add_round_rect(slide, 1.7, 23.6, 15.8, 10.1, c_green_lt, c_green_bd, lw=0.8, adj=0.08)
    add_text_box(slide, 1.8, 31.7, 15.6, 1.8, "ZFSS Sketch s_{t}", c_green, fs=6.4, bold=True)
    add_badge(slide, 2.15, 28.5, 6.6, 2.5, "32-bit Seed", "#FFFFFF", c_green, color=c_green, fs=5.3)
    add_arrow_chevron(slide, 8.95, 29.3, 1.1, 0.9, c_green, "right")
    add_badge(slide, 10.25, 28.5, 6.7, 2.5, "q16 (48 B)", c_green, c_green, color="#FFFFFF", fs=5.4)
    add_text_box(slide, 1.8, 24.0, 15.6, 4.0, "Stream A: 0 B Matrix\n12.3 KB/AOI (-99.09%)", "#052E16", fs=5.8, bold=True)

    add_round_rect(slide, 18.3, 27.1, 23.5, 19.2, "#FFFFFF", c_blue, lw=0.9, adj=0.05)
    add_text_box(slide, 18.5, 44.2, 23.1, 1.8, "DGAD Dual-Gate & Packer", c_blue, fs=6.6, bold=True)

    add_round_rect(slide, 19.0, 37.2, 13.4, 6.7, c_green_lt, c_green, lw=0.75, adj=0.10)
    add_text_box(slide, 19.1, 41.8, 13.2, 1.7, "Safety Gate (Prop. 1)", c_green, fs=5.7, bold=True)
    add_text_box(slide, 19.1, 37.5, 13.2, 4.1, "L_{b} ≤ τ → STABLE\n0 False-Stable Viol.", "#052E16", fs=5.6)

    add_round_rect(slide, 19.0, 29.5, 13.4, 6.8, c_blue_lt, c_blue, lw=0.75, adj=0.10)
    add_text_box(slide, 19.1, 34.2, 13.2, 1.7, "Utility Gate (Point Est.)", c_blue, fs=5.7, bold=True)
    add_text_box(slide, 19.1, 29.8, 13.2, 4.1, "Rank ||z_{b}||_{2}^{2} / k\nJPEG 4:4:4 (q80-90)", c_navy, fs=5.6)

    gx0, gy0, cs = 33.2, 34.4, 1.56
    gmap = [[0, 1, 1, 0, 0], [0, 2, 1, 2, 0], [2, 0, 0, 0, 0], [0, 1, 1, 0, 2], [0, 0, 2, 0, 0]]
    scols = {0: "#86EFAC", 1: "#2563EB", 2: "#FDE68A"}
    add_text_box(slide, gx0 - 0.2, gy0 + 5 * cs + 0.1, 5 * cs + 0.4, 1.5, "Block Map", c_navy, fs=5.9, bold=True)
    for ri in range(5):
        for ci in range(5):
            add_round_rect(slide, gx0 + ci * cs, gy0 + ri * cs, cs - 0.14, cs - 0.14, scols[gmap[4 - ri][ci]], c_navy, lw=0.4, adj=0.05)
    for idx, (col, txt) in enumerate([("#86EFAC", "STABLE"), ("#2563EB", "UPDATED"), ("#FDE68A", "DEFERRED")]):
        ly = 32.6 - idx * 1.4
        add_round_rect(slide, 33.2, ly, 1.05, 0.85, col, c_navy, lw=0.35, adj=0.05)
        add_text_box(slide, 34.4, ly - 0.15, 7.1, 1.15, txt, c_navy, fs=5.5, align=PP_ALIGN.LEFT)

    add_badge(slide, 18.3, 23.6, 23.5, 2.65, "Self-Decode Atlas → Commit on ACK", "#F1F5F9", c_teal, color=c_teal, fs=5.7)

    # TIER 1: LINK
    add_round_rect(slide, 44.5, 33.2, 12.0, 14.8, "#FFFFFF", c_amber, lw=0.9, adj=0.08)
    add_text_box(slide, 44.6, 45.6, 11.8, 2.0, "Downlink", c_amber, fs=6.9, bold=True)
    add_badge(slide, 45.2, 43.3, 10.6, 2.0, "≤ 32 KiB", c_amber_lt, c_amber, color=c_amber, fs=6.2)
    add_text_box(slide, 44.7, 34.0, 11.6, 8.8, "WebP85 c_{t}\nJPEG 4:4:4\nDigest+CRC\n(118 ms CPU)", c_navy, fs=5.9)
    add_arrow_chevron(slide, 42.1, 39.8, 2.1, 1.4, c_amber, "right")
    add_arrow_chevron(slide, 56.7, 39.8, 2.1, 1.4, c_amber, "right")

    add_round_rect(slide, 44.5, 23.6, 12.0, 8.2, c_teal_lt, c_teal, lw=0.9, adj=0.10)
    add_text_box(slide, 44.6, 29.4, 11.8, 2.0, "60-B ACK", c_teal, fs=6.8, bold=True)
    add_text_box(slide, 44.6, 24.4, 11.8, 4.8, "Commit Digest\n0 Image Uplink", "#042F2E", fs=5.85, bold=True)
    add_arrow_chevron(slide, 56.7, 26.6, 2.1, 1.3, c_teal, "left")
    add_arrow_chevron(slide, 42.1, 26.6, 2.1, 1.3, c_teal, "left")

    # TIER 1: GROUND
    add_round_rect(slide, 58.5, 22.9, 40.4, 26.7, "#FFFFFF", "#99F6E4", lw=0.85, adj=0.04)
    add_badge(slide, 59.1, 46.9, 32.2, 2.05, "(b) Ground Smooth ASF+ Archive", c_teal_lt, c_teal, color=c_teal, fs=6.7)

    add_round_rect(slide, 59.1, 36.8, 15.6, 9.4, c_blue_lt, c_blue_bd, lw=0.8, adj=0.08)
    add_text_box(slide, 59.2, 44.0, 15.4, 1.9, "Smooth Field Ũ c_{t}", c_blue, fs=6.5, bold=True)
    add_text_box(slide, 59.2, 37.3, 15.4, 6.4, "Bilinear drift align;\n4.5× fewer false\nbackground triggers", c_navy, fs=5.8)

    add_round_rect(slide, 59.1, 23.6, 15.6, 12.2, "#F1F5F9", c_navy, lw=0.85, adj=0.08)
    add_text_box(slide, 59.2, 33.6, 15.4, 1.9, "Immutable Archive a_{t}", c_navy, fs=6.5, bold=True)
    add_text_box(slide, 59.2, 24.2, 15.4, 9.0, "Stores ACK'd absolute\nblocks C(x_{s,b}) only;\nnever overwritten!\n|S_{b,t}| ≤ t", c_navy, fs=5.75)

    add_round_rect(slide, 75.6, 23.6, 22.6, 22.6, c_green_lt, c_green, lw=0.95, adj=0.06)
    add_text_box(slide, 75.8, 44.0, 22.2, 1.9, "Co-Constructed Render", c_green, fs=6.8, bold=True)
    add_badge(slide, 76.5, 41.5, 20.8, 2.1, "x̂_{t}^{+} = R(H a_{t} + Ũ c_{t})", "#FFFFFF", c_green, color=c_green, fs=6.1)
    add_pil_image(slide, patch_rgb, 76.5, 31.8, 7.6, 8.7, border_color=c_green)
    add_text_box(slide, 84.5, 31.6, 13.4, 9.2, "SpaceNet 7 OOF\n(ASF+ CPU #1):\n32K: 0.4760\n(+3.9 pp TCM)\n128K: 0.5216", "#052E16", fs=5.8, bold=True)
    add_badge(slide, 76.5, 24.5, 20.8, 5.8, "0 Violations / 118,317 Blocks\n263× Faster than GPU LIC-TCM", "#FFFFFF", c_green_bd, color=c_navy, fs=5.85)

    # TIER 2: SPACE
    add_round_rect(slide, 1.1, 1.1, 41.4, 20.8, "#FFFFFF", c_purple_bd, lw=0.85, adj=0.04)
    add_badge(slide, 1.7, 19.2, 40.2, 2.05, "(c) Tier 2: Semantic & Open-World Sync (US-BRS + TC-PCR)",
              c_purple_lt, c_purple, color=c_purple, fs=6.15)

    add_round_rect(slide, 1.7, 1.8, 19.6, 16.7, c_purple_lt, c_purple_bd, lw=0.8, adj=0.07)
    add_text_box(slide, 1.8, 16.4, 19.4, 1.8, "US-BRS Unit-Sphere Sync", c_purple, fs=6.45, bold=True)
    oval = slide.shapes.add_shape(MSO_SHAPE.OVAL, *xywh(2.5, 7.2, 5.8, 5.8))
    oval.fill.solid()
    oval.fill.fore_color.rgb = rgb("#FFFFFF")
    oval.line.color.rgb = rgb(c_purple)
    oval.line.width = Pt(1.1)
    add_text_box(slide, 2.0, 5.0, 6.8, 1.8, "|δq| ≤ 41", c_rose, fs=5.7, bold=True)
    add_text_box(slide, 9.0, 3.5, 11.9, 12.2, "DINOv2-S/14\nLossless int8\nδ: -20.6% to\n-35.4% bytes\nNACK Anchor", "#2E1065", fs=5.85)

    add_round_rect(slide, 22.1, 1.8, 19.8, 16.7, c_blue_lt, c_blue_bd, lw=0.8, adj=0.07)
    add_text_box(slide, 22.2, 16.4, 19.6, 1.8, "TC-PCR Prototype Router", c_blue, fs=6.45, bold=True)
    ell = slide.shapes.add_shape(MSO_SHAPE.OVAL, *xywh(23.1, 9.4, 5.6, 3.4))
    ell.fill.solid()
    ell.fill.fore_color.rgb = rgb("#DBEAFE")
    ell.line.color.rgb = rgb(c_blue)
    ell.line.width = Pt(1.0)
    add_text_box(slide, 22.6, 7.1, 6.6, 1.8, "Ledoit-Wolf", c_blue, fs=5.5, bold=True)
    add_text_box(slide, 29.4, 3.2, 12.1, 12.8, "424-B Ref (-66%)\nActive-Support\n30-dB Crossing\nCalib. Novelty\nRE01 Residuals", c_navy, fs=5.8)

    # TIER 2: LINK
    add_round_rect(slide, 44.5, 1.8, 12.0, 19.4, "#FFFFFF", c_purple, lw=0.9, adj=0.08)
    add_text_box(slide, 44.6, 18.8, 11.8, 2.0, "TE01 Frame", c_purple, fs=6.7, bold=True)
    add_badge(slide, 45.2, 16.7, 10.6, 1.9, "≤ 8 KiB", c_purple_lt, c_purple, color=c_purple, fs=6.15)
    add_text_box(slide, 44.7, 7.0, 11.6, 9.2, "Exact 16×16\nTask State +\nBase + RE01\nResidual ROIs", c_navy, fs=5.85)
    add_badge(slide, 45.2, 2.6, 10.6, 3.6, "ACK / NACK\nRisk Anchor", c_rose_lt, c_rose, color=c_rose, fs=5.75)
    add_arrow_chevron(slide, 42.1, 10.8, 2.1, 1.4, c_purple, "right")
    add_arrow_chevron(slide, 56.7, 10.8, 2.1, 1.4, c_purple, "right")
    add_arrow_chevron(slide, 56.7, 3.8, 2.1, 1.2, c_rose, "left")
    add_arrow_chevron(slide, 42.1, 3.8, 2.1, 1.2, c_rose, "left")

    # TIER 2: GROUND
    add_round_rect(slide, 58.5, 1.1, 40.4, 20.8, "#FFFFFF", c_purple_bd, lw=0.85, adj=0.04)
    add_badge(slide, 59.1, 19.2, 39.2, 2.05, "(d) Open-World Semantic Discovery (SECOND Cohort 3)",
              c_purple_lt, c_purple, color=c_purple, fs=6.2)

    add_round_rect(slide, 59.1, 1.8, 14.2, 16.7, c_teal_lt, c_teal_bd, lw=0.8, adj=0.07)
    add_text_box(slide, 59.2, 16.3, 14.0, 1.8, "Hybrid Fusion", c_teal, fs=6.5, bold=True)
    add_text_box(slide, 59.3, 3.2, 13.8, 12.6, "Exact known\nchange +\nBase + RE01\n>30 dB novel\nROI recovery", "#042F2E", fs=5.85)

    add_round_rect(slide, 74.0, 1.8, 24.3, 16.7, c_amber_lt, c_amber, lw=0.9, adj=0.06)
    add_text_box(slide, 74.2, 16.3, 23.9, 1.8, "SECOND 8-KiB Held-Out U_{30}", c_amber, fs=6.55, bold=True)
    for idx, (mlbl, uval, col) in enumerate([
        ("WebP", 0.5380, "#64748B"),
        ("Hyperprior", 0.6379, "#9333EA"),
        ("LIC-TCM", 0.7688, "#D55E00"),
        ("TC-PCR(TCM)", 0.8233, c_blue),
    ]):
        by = 14.4 - idx * 2.0
        is_best = mlbl == "TC-PCR(TCM)"
        add_text_box(slide, 74.3, by - 0.5, 8.2, 1.3, mlbl, c_blue if is_best else c_navy,
                     fs=5.45 if is_best else 5.7, bold=is_best, align=PP_ALIGN.LEFT)
        bw = uval * 9.5
        add_round_rect(slide, 82.8, by - 0.35, bw, 0.95, col, None, adj=0.22)
        add_text_box(slide, 82.9 + bw, by - 0.5, 5.2, 1.3, f"{uval:.3f}", col, fs=5.7, bold=True, align=PP_ALIGN.LEFT)

    add_badge(slide, 74.7, 2.5, 22.9, 5.1, "#1 U_{30}: +5.45 pp vs. 4.18G LIC-TCM\n+15.3 pp WebP | FNR 0.157 | F1 0.850",
              "#FFFFFF", c_amber, color=c_navy, fs=5.35)

    out_path = ROOT / "paper/figures/layered_state.pptx"
    prs.save(str(out_path))
    print(f"Exported 1:1 native-editable PowerPoint diagram -> {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
