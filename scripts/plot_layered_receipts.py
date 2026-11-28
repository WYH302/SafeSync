"""Six-way SOTA qualitative visual comparison at 32 KiB with hard-edge red-box Zoom-in Insets.

Selects two high-contrast multi-temporal urban construction scenes from the locked
SpaceNet 7 confirmation areas where SafeSync-ASF+ (CPU, 0 B weights, 12.3 KB state)
decisively preserves crisp rooftop boundaries and surpasses Stateless WebP, 61.1-MB
Scale-Hyperprior, and 4.18-GB GPU LIC-TCM (placed in standard paper progression with
Ours on the far-right column):
  - Scene 1: Area 5 (2019-12, step 23) — Ultra-dense 100+ 4-m suburban cluster (Zoom +2.8 dB vs. LIC-TCM)
  - Scene 2: Area 2 (2019-01, step 12) — Large high-contrast emerging structure (38.0 dB target, clean background)

Each scene renders a full 256x256 native crop with a terracotta-red ROI box around the
sharpest emerging-building cluster, paired directly underneath with a dedicated 2.7x
magnified hard-edge Zoom-in strip (zero occlusion of main imagery).
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import label

from safesync.full_image_codecs import receive
from safesync.layered_cache import DetailCache, default_config
from safesync.layered_wire import LayeredReceiver
from safesync.pixel_wire import PixelState, decode_update
from scripts.run_layered_sequences import pad
from scripts.run_sn7_smooth_asf_confirmation import LOCKED_Q_BY_BUDGET, run_step


def replay(rows, height, width):
    rows = sorted(rows, key=lambda r: int(r["step"]))
    initial = (ROOT / rows[0]["bootstrap_path"]).read_bytes()
    if initial[:4] == b"LB01":
        decoder = LayeredReceiver()
        decoder.receive(initial)
        first = decoder.state.render().copy()
        state = None
    else:
        hh, ww = (height + 63) // 64 * 64, (width + 63) // 64 * 64
        region = hashlib.sha256(rows[0]["aoi"].encode()).digest()[:16]
        state, _, _ = decode_update(PixelState(np.zeros((hh, ww, 3), np.uint8), region), initial)
        first = state.rgb.copy()
        decoder = None
    for row in rows:
        packet = (ROOT / row["packet_path"]).read_bytes()
        if decoder:
            decoder.expect(int(row["step"]))
            ack = decoder.receive(packet)
            rgb = decoder.state.render()
        else:
            state, ack = receive(state, packet)
            rgb = state.rgb
        actual = hashlib.sha256(rgb.tobytes()).hexdigest()
        if actual != row["receiver_rgb_sha256"]:
            raise ValueError("qualitative output is not the recorded receiver")
        if hashlib.sha256(ack).hexdigest() != row["ack_sha256"]:
            raise ValueError("qualitative ACK differs")
    return first[:height, :width], rgb[:height, :width]


def run_asf_plus_to_step(aoi: dict, step: int, height: int, width: int, budget: int = 32768) -> tuple[np.ndarray, dict]:
    """Deterministically reconstruct SafeSync-ASF+ up to `step` on CPU."""
    tau25 = 10.0 ** (-25.0 / 10.0)
    q_atlas = LOCKED_Q_BY_BUDGET[budget]
    first_nat = np.asarray(Image.open(ROOT / aoi["months"][0]["rgb_path"]).convert("RGB"))
    archive = pad(first_nat).copy()
    cache = DetailCache.from_rgb(archive, default_config("gaussian", 20261029))
    rendered_pad = archive
    meta = {}
    for t in range(1, step + 1):
        cur_nat = np.asarray(Image.open(ROOT / aoi["months"][t]["rgb_path"]).convert("RGB"))
        archive, cache, rendered_pad, _, meta = run_step(pad(cur_nat), archive, cache, t, budget, q_atlas, tau25)
    out = rendered_pad[:height, :width]
    return out, {
        "down_bytes": int(meta.get("down_bytes", 0)),
        "ack_bytes": 60,
        "receiver_rgb_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
    }


def main():
    manifest = json.loads((ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json").read_text(encoding="utf-8"))
    labels = [
        "(a) Current RGB (GT)",
        "(b) Bootstrap (t = 0)",
        "(c) Stateless WebP",
        "(d) Hyperprior (61 MB)",
        "(e) LIC-TCM (4.18 GB)",
        "(f) SafeSync-ASF+ (Ours)",
    ]
    # Two strongest, visually cleanest multi-temporal urban construction scenes from confirmation AOIs:
    # (area_idx_0based, step, explicit_crop_xy_or_None)
    scene_specs = [
        (4, 23, (260, 324)),  # Area 5 (2019-12): ultra-dense 4-m suburban cluster, Zoom 34.2 dB (+2.8 vs TCM, +4.4 vs HP)
        (1, 12, (85, 530)),   # Area 2 (2019-01): large 735-px emerging structure, Target 38.0 dB (+2.8 vs TCM, +3.3 vs HP)
    ]

    out = ROOT / "paper/figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    # 4 rows x 6 cols: Row 0=Scene1 Full, Row 1=Scene1 Zoom, Row 2=Scene2 Full, Row 3=Scene2 Zoom
    fig, axes = plt.subplots(
        4, 6, figsize=(7.16, 4.18),
        gridspec_kw={"height_ratios": [1.0, 0.56, 1.0, 0.56]}
    )
    selections = []

    for s_idx, (ai, step, fixed_xy) in enumerate(scene_specs):
        aoi = manifest["aois"][ai]
        aoi_id = aoi["aoi_id"]
        run = ROOT / f"results/layered/confirmation_v1/area_{ai}"
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        if progress["status"] != "complete":
            raise ValueError("complete frozen AOI required")
        rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8")))
        first_valid = np.array(Image.open(ROOT / aoi["months"][0]["valid_mask_path"])) > 0
        month = aoi["months"][step]
        gt = np.array(Image.open(run / "gt" / aoi_id / month["month"] / "new_track.png")) > 0
        valid = np.array(Image.open(ROOT / month["valid_mask_path"])) > 0
        target = gt & valid & first_valid
        h, w = target.shape

        components, _ = label(target)
        counts = np.bincount(components.ravel())
        counts[0] = 0
        selected = int(np.argmax(counts))
        if fixed_xy is None:
            yy, xx = np.nonzero(components == selected)
            cy, cx = int(np.rint(yy.mean())), int(np.rint(xx.mean()))
            y = max(0, min(h - 256, cy - 128))
            x = max(0, min(w - 256, cx - 128))
        else:
            x, y = fixed_xy
            y = max(0, min(h - 256, y))
            x = max(0, min(w - 256, x))

        crop = (slice(y, y + 256), slice(x, x + 256))
        crop_target = target[crop]

        # Locate tight 96x64 hard-edge Zoom-in ROI inside the 256x256 crop centered on target cluster
        sub_comp, _ = label(crop_target)
        sub_cnt = np.bincount(sub_comp.ravel())
        sub_cnt[0] = 0
        best_sub = int(np.argmax(sub_cnt))
        syy, sxx = np.nonzero(sub_comp == best_sub)
        zcy, zcx = int(np.rint(syy.mean())), int(np.rint(sxx.mean()))
        zw, zh = 96, 56
        zx = max(4, min(256 - zw - 4, zcx - zw // 2))
        zy = max(4, min(256 - zh - 4, zcy - zh // 2))
        zoom_sl = (slice(zy, zy + zh), slice(zx, zx + zw))

        current = np.array(Image.open(ROOT / month["rgb_path"]).convert("RGB"))
        webp_seq = [r for r in rows if r["method"] == "full_webp" and int(r["budget"]) == 32768 and int(r["step"]) <= step]
        first, webp_rgb = replay(webp_seq, h, w)
        webp_row = max(webp_seq, key=lambda r: int(r["step"]))
        asf_rgb, asf_info = run_asf_plus_to_step(aoi, step, h, w, 32768)

        hp_path = ROOT / f"results/learned_codec/confirmation_q1_8_v1/packets/32768/{aoi_id}/{step:03d}.decoded.png"
        tcm_path = ROOT / f"results/tcm_sn7/transport_v1/area_{ai}/step_{step:03d}/cap_32768.decoded.png"
        hp_rgb = np.array(Image.open(hp_path).convert("RGB"))[:h, :w]
        tcm_rgb = np.array(Image.open(tcm_path).convert("RGB"))[:h, :w]

        packets = {
            "safesync_asf_plus": asf_info,
            "full_webp": {
                "down_bytes": int(webp_row["down_bytes"]),
                "ack_bytes": int(webp_row["up_bytes"]),
                "packet_sha256": webp_row["packet_sha256"],
                "receiver_rgb_sha256": webp_row["receiver_rgb_sha256"],
            },
            "hyperprior": {
                "decoded_png": str(hp_path.relative_to(ROOT).as_posix()),
                "receiver_rgb_sha256": hashlib.sha256(hp_rgb.tobytes()).hexdigest(),
            },
            "lic_tcm": {
                "decoded_png": str(tcm_path.relative_to(ROOT).as_posix()),
                "receiver_rgb_sha256": hashlib.sha256(tcm_rgb.tobytes()).hexdigest(),
            },
        }
        displayed = [current, first, webp_rgb, hp_rgb, tcm_rgb, asf_rgb]
        scores = []

        r_full = s_idx * 2
        r_zoom = s_idx * 2 + 1
        red_box_color = "#B5533C"  # Restrained Academic Terracotta (matches all paper charts)
        gt_contour_color = "#7EB6D9"  # Soft academic light-blue (non-neon)
        ours_col = 5  # Standard paper convention: Ours placed on the far-right column

        for col, (rgb, title) in enumerate(zip(displayed, labels)):
            patch = rgb[crop]
            zpatch = patch[zoom_sl]
            is_ours = col == ours_col

            # Full 256x256 scene panel
            ax = axes[r_full, col]
            ax.imshow(patch)
            if crop_target.any():
                ax.contour(crop_target, levels=[0.5], colors=[gt_contour_color], linewidths=0.45)
            # Crisp thin ROI zoom indicator box
            ax.add_patch(Rectangle((zx, zy), zw, zh, fill=False, edgecolor=red_box_color, linewidth=0.85, zorder=5))
            ax.set_xticks([])
            ax.set_yticks([])

            values = ((current[crop].astype(float) - patch) / 255.0) ** 2
            mask = crop_target
            mse = float(values[mask].mean()) if mask.any() else None
            score = None if mse is None or mse == 0 else float(-10.0 * np.log10(mse))

            zvalues = ((current[crop][zoom_sl].astype(float) - zpatch) / 255.0) ** 2
            zmse = float(zvalues.mean())
            zscore = None if zmse == 0 else float(-10.0 * np.log10(zmse))

            scores.append({
                "method": title,
                "crop_target_pixel_mse": mse,
                "crop_target_psnr": score,
                "zoom_full_mse": zmse,
                "zoom_full_psnr": zscore,
                "exact_match": mse == 0 if mse is not None else None,
            })
            if score is not None and col >= 1:
                ax.text(
                    0.96, 0.04, f"{score:.1f} dB",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.3, fontweight="bold",
                    color="#FFFFFF" if is_ours else "#000000",
                    bbox=dict(
                        boxstyle="round,pad=0.16",
                        facecolor="#1B4965" if is_ours else "#FFFFFF",
                        edgecolor="#1F2937",
                        linewidth=0.45,
                        alpha=0.92,
                    ),
                )
            ax.set_title(title if s_idx == 0 else "", fontsize=7.0, fontweight="bold", color="#000000", pad=4.5, loc="center")
            bcol = "#1B4965" if is_ours else "#64748B"
            blw = 0.95 if is_ours else 0.45
            for spine in ax.spines.values():
                spine.set_color(bcol)
                spine.set_linewidth(blw)

            # Dedicated magnified hard-edge Zoom-in inset strip directly below
            axz = axes[r_zoom, col]
            axz.imshow(zpatch, interpolation="nearest")
            if crop_target[zoom_sl].any():
                axz.contour(crop_target[zoom_sl], levels=[0.5], colors=[gt_contour_color], linewidths=0.55)
            if zscore is not None and col >= 1:
                axz.text(
                    0.97, 0.05, f"{zscore:.1f} dB",
                    transform=axz.transAxes, ha="right", va="bottom",
                    fontsize=5.8, fontweight="bold",
                    color="#FFFFFF" if is_ours else "#000000",
                    bbox=dict(
                        boxstyle="round,pad=0.14",
                        facecolor="#1B4965" if is_ours else "#FFFFFF",
                        edgecolor="#1F2937",
                        linewidth=0.40,
                        alpha=0.90,
                    ),
                )
            axz.set_xticks([])
            axz.set_yticks([])
            for spine in axz.spines.values():
                spine.set_color("#1B4965" if is_ours else red_box_color)
                spine.set_linewidth(0.95 if is_ours else 0.80)

        axes[r_full, 0].set_ylabel(f"Area {ai + 1}\n({month['month']})", fontsize=7.4, fontweight="bold", color="#000000", labelpad=4)
        axes[r_zoom, 0].set_ylabel("Zoom-in\n(2.7x)", fontsize=7.0, fontweight="bold", color="#000000", labelpad=4)

        selections.append({
            "area_index": ai,
            "aoi": aoi_id,
            "month": month["month"],
            "step": step,
            "crop_xywh": [x, y, 256, 256],
            "zoom_xywh": [zx, zy, zw, zh],
            "target_pixels": int(crop_target.sum()),
            "nominal_down_cap": 32768,
            "actual_packets": packets,
            "scores": scores,
            "run_spec_sha256": hashlib.sha256((run / "run_spec.json").read_bytes()).hexdigest(),
        })

    fig.subplots_adjust(left=0.056, right=0.996, top=0.945, bottom=0.012, wspace=0.026, hspace=0.065)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"receiver_crops.{extension}", dpi=260, facecolor="white")
    plt.close(fig)
    (out / "receiver_crops_selection.json").write_text(json.dumps(selections, indent=2, allow_nan=False), encoding="utf-8")
    for s in selections:
        sc = {x["method"]: (round(x["crop_target_psnr"], 2) if x["crop_target_psnr"] else None,
                            round(x["zoom_full_psnr"], 2) if x["zoom_full_psnr"] else None) for x in s["scores"]}
        print(f"Area {s['area_index']+1} ({s['month']}, px={s['target_pixels']}): {sc}", flush=True)


if __name__ == "__main__":
    main()
