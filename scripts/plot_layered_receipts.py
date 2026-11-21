"""Deterministic, GT-supported 6-way SOTA visual crops from actual decoded 32-KiB outputs.

Select the first evaluable new-track month in confirmation AOIs 0 and 1;
center on its largest connected target component, without consulting method
scores. The same native 256x256 crop is compared side-by-side across:
  (a) Current RGB (GT)
  (b) Bootstrap (t = 0)
  (c) SafeSync-ASF+ (Ours, CPU 0 B)
  (d) Stateless WebP (CPU 0 B)
  (e) Scale-Hyperprior (61.1 MB)
  (f) LIC-TCM (4.18 GB GPU)
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
        "(c) SafeSync-ASF+ (Ours)",
        "(d) Stateless WebP",
        "(e) Hyperprior (61 MB)",
        "(f) LIC-TCM (4.18 GB)",
    ]
    out = ROOT / "paper/figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 2.74))
    selections = []
    for ai in (0, 1):
        aoi = manifest["aois"][ai]
        aoi_id = aoi["aoi_id"]
        run = ROOT / f"results/layered/confirmation_v1/area_{ai}"
        progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        if progress["status"] != "complete":
            raise ValueError("complete frozen AOI required")
        rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8")))
        first_valid = np.array(Image.open(ROOT / aoi["months"][0]["valid_mask_path"])) > 0
        chosen = None
        for t, m in enumerate(aoi["months"][1:], 1):
            gt = np.array(Image.open(run / "gt" / aoi_id / m["month"] / "new_track.png")) > 0
            valid = np.array(Image.open(ROOT / m["valid_mask_path"])) > 0
            target = gt & valid & first_valid
            if target.sum() >= 256:
                chosen = (t, m, target)
                break
        if chosen is None:
            raise ValueError("predeclared crop support absent")
        step, month, target = chosen
        components, _ = label(target)
        counts = np.bincount(components.ravel())
        counts[0] = 0
        selected = int(np.argmax(counts))
        yy, xx = np.nonzero(components == selected)
        cy, cx = int(np.rint(yy.mean())), int(np.rint(xx.mean()))
        h, w = target.shape
        y = max(0, min(h - 256, cy - 128))
        x = max(0, min(w - 256, cx - 128))
        crop = (slice(y, y + 256), slice(x, x + 256))
        current = np.array(Image.open(ROOT / month["rgb_path"]).convert("RGB"))

        # Replay stateless WebP (and bootstrap) from verified packets
        webp_seq = [r for r in rows if r["method"] == "full_webp" and int(r["budget"]) == 32768 and int(r["step"]) <= step]
        first, webp_rgb = replay(webp_seq, h, w)
        webp_row = max(webp_seq, key=lambda r: int(r["step"]))

        # Reconstruct SafeSync-ASF+ (Ours)
        asf_rgb, asf_info = run_asf_plus_to_step(aoi, step, h, w, 32768)

        # Load verified Hyperprior and LIC-TCM 32-KiB decoded PNGs
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
        displayed = [current, first, asf_rgb, webp_rgb, hp_rgb, tcm_rgb]
        scores = []
        for col, (rgb, title) in enumerate(zip(displayed, labels)):
            ax = axes[ai, col]
            ax.imshow(rgb[crop])
            if target[crop].any():
                ax.contour(target[crop], levels=[0.5], colors=["#00E5FF"], linewidths=0.65)
            ax.set_xticks([])
            ax.set_yticks([])
            values = ((current[crop].astype(float) - rgb[crop]) / 255.0) ** 2
            mask = target[crop]
            mse = float(values[mask].mean()) if mask.any() else None
            score = None if mse is None or mse == 0 else float(-10.0 * np.log10(mse))
            scores.append({
                "method": title,
                "crop_target_pixel_mse": mse,
                "crop_target_psnr": score,
                "exact_match": mse == 0 if mse is not None else None,
            })
            if score is not None and col >= 1:
                ax.text(
                    0.96, 0.04, f"{score:.1f} dB",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.4, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#0F172A", edgecolor="none", alpha=0.80),
                )
            tcol = "#0F3D8A" if "SafeSync" in title else "#0F172A"
            ax.set_title(title if ai == 0 else "", fontsize=7.0, fontweight="bold", color=tcol, pad=4.5, loc="center")
            bcol = "#1D4ED8" if "SafeSync" in title else "#94A3B8"
            blw = 1.25 if "SafeSync" in title else 0.50
            for spine in ax.spines.values():
                spine.set_color(bcol)
                spine.set_linewidth(blw)
        axes[ai, 0].set_ylabel(f"Area {ai + 1}\n({month['month']})", fontsize=7.5, fontweight="bold", labelpad=4)
        selections.append({
            "area_index": ai,
            "aoi": aoi_id,
            "month": month["month"],
            "step": step,
            "crop_xywh": [x, y, 256, 256],
            "target_pixels": int(target[crop].sum()),
            "component_pixels": int(counts[selected]),
            "selection": "first chronological month with >=256 evaluable target pixels, center largest component; no method scores used",
            "nominal_down_cap": 32768,
            "actual_packets": packets,
            "scores": scores,
            "run_spec_sha256": hashlib.sha256((run / "run_spec.json").read_bytes()).hexdigest(),
        })
    fig.subplots_adjust(left=0.042, right=0.996, top=0.915, bottom=0.015, wspace=0.028, hspace=0.045)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / f"receiver_crops.{extension}", dpi=240, facecolor="white")
    plt.close(fig)
    (out / "receiver_crops_selection.json").write_text(json.dumps(selections, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(selections, indent=2), flush=True)


if __name__ == "__main__":
    main()
