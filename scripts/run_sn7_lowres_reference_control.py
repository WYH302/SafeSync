"""Matched-storage low-resolution historical reference control and same-version module/perturbation diagnostics on SpaceNet 7.

Evaluates Earth+-inspired matched-storage (<= 12,328 B/area) low-resolution historical reference
baselines inside the identical LB01/LA01 regional JPEG 4:4:4 update pipeline and Author OOF
CrossfitGroundEvaluator across all 8 confirmation AOIs (32, 64, 128 KiB), alongside a
pseudo-change perturbation diagnostic.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))

import numpy as np
from PIL import Image, ImageFilter

from resources.pretrained_ground_crossfit.run_receipts import measure_frame
from safesync.crossfit_ground import CrossfitGroundEvaluator
from safesync.layered_cache import DetailCache, default_config, digest
from safesync.layered_wire import LayeredReceiver, history_after, parse_frame, to_layered
from safesync.pixel_wire import UPDATED, encode_update
from safesync.reference import blocks, exact_mse, restore_coarse
from scripts import evaluate_layered_receipts as common
from scripts import evaluate_learned_receipts as lic
from scripts.evaluate_pixel_receipts import block_mask_quality
from scripts.run_layered_sequences import pad, sha, targets, write_json
from scripts.run_pixel_sequences import choose_packet

OUT = ROOT / "results/lowres_reference_sn7/v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
EXISTING_OOF = ROOT / "resources/pretrained_ground_crossfit/receipts_v1"
EXISTING_MAIN = ROOT / "results/layered/confirmation_v1"
EXISTING_GATE30 = ROOT / "results/full_cache_controls/gate30"
BUDGETS = (32768, 65536, 131072)
LOWRES_METHODS = ("lowres_16x16_coarse_only", "lowres_4x_webp_12k", "lowres_4x_webp_12k_gate30")
MAX_PAYLOAD_BYTES = 12288  # + 40B header/checksum = 12,328 B/area, matching gaussian_20261029


def downsample_4x(rgb: np.ndarray) -> np.ndarray:
    """Box-downsample HxWx3 by 4x to (H//4)x(W//4)x3 uint8."""
    h, w, c = rgb.shape
    return np.rint(rgb.reshape(h // 4, 4, w // 4, 4, c).mean((1, 3))).astype(np.uint8)


def encode_webp_bounded(thumb_rgb: np.ndarray, max_bytes: int = MAX_PAYLOAD_BYTES) -> tuple[bytes, int]:
    """Encode 256x256x3 thumbnail with WebP to <= max_bytes."""
    im = Image.fromarray(thumb_rgb)
    for q in (98, 95, 92, 90, 88, 85, 82, 80, 75, 70, 65, 60, 50, 40):
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=q, method=4)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, q
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=30, method=4)
    return buf.getvalue(), 30


def decode_and_upsample_4x(webp_bytes: bytes, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode 256x256 WebP thumbnail and bicubic-upsample to (H, W, 3)."""
    with Image.open(io.BytesIO(webp_bytes)) as im:
        thumb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        up = np.asarray(
            im.convert("RGB").resize((target_shape[1], target_shape[0]), resample=Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
    return thumb, up


class LowResLayeredSender:
    """Sender using the exact LB01/LA01 wire protocol with a matched-storage low-res historical reference."""

    def __init__(self, mode: str, gate_psnr: int = 25):
        self.mode = mode
        self.gate_psnr = gate_psnr
        self.cache = None
        self.version = 0
        self.history = b"\x00" * 16
        self.pending = None
        self.region = None
        self.shape = None
        self.observation = -1
        self.provisional = None
        self.lowres_wire = None
        self.provisional_lowres_wire = None
        self.lowres_quality = None

    def bootstrap(self, image: np.ndarray, region: bytes, config: dict) -> tuple[bytes, dict]:
        a = np.asarray(image, np.uint8)
        h, w = a.shape[:2]
        n = h * w // 4096
        frame, cost = encode_update(
            shape=(h, w),
            block=64,
            region=region,
            base_version=0,
            base_digest=b"\x00" * 16,
            observed_at=0,
            current_rgb=a,
            indices=np.arange(n),
            statuses=np.full(n, UPDATED, np.uint8),
            codec="png",
        )
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        wire = b"LB01" + int(len(config_bytes)).to_bytes(4, "little") + config_bytes + to_layered(frame, b"LA00")
        decoded = parse_frame(wire[8 + len(config_bytes) :])
        archive = np.empty_like(a)
        for idx, patch in decoded["patches"]:
            y, x = divmod(idx, w // 64)
            archive[y * 64 : (y + 1) * 64, x * 64 : (x + 1) * 64] = patch
        cache = DetailCache.from_rgb(archive, config)
        self.region, self.shape = region, (h, w)
        self.pending = wire
        self.provisional = (cache, history_after(self.history, wire, cache), 1, 0)
        if self.mode == "coarse_only":
            coarse_thumb = np.rint(archive.reshape(h // 16, 16, w // 16, 16, 3).mean((1, 3))).astype(np.uint8)
            self.provisional_lowres_wire = b"LR16" + (b"\x00" * 20) + coarse_thumb.tobytes() + (b"\x00" * 16)
            self.lowres_quality = 100
        else:
            thumb4x = downsample_4x(archive)
            webp_bytes, q = encode_webp_bounded(thumb4x, MAX_PAYLOAD_BYTES)
            self.provisional_lowres_wire = b"LR4W" + (b"\x00" * 20) + webp_bytes + (b"\x00" * 16)
            self.lowres_quality = q
        return wire, dict(bootstrap_bytes=len(wire), config_bytes=8 + len(config_bytes), components=cost)

    def propose(self, current: np.ndarray, observation: int, budget: int) -> tuple[bytes, dict]:
        h, w = self.shape
        if self.mode == "coarse_only":
            raw = self.lowres_wire[24:-16]
            coarse_old = np.frombuffer(raw, dtype=np.uint8).reshape(h // 16, w // 16, 3)
            prior_est = coarse_old.repeat(16, 0).repeat(16, 1)
            restored = restore_coarse(prior_est, current)
            point = upper = exact_mse(current, restored)
            meta = dict(kind="16x16 coarse-only variance", lowres_state_bytes=len(self.lowres_wire), webp_q=100)
        else:
            webp_bytes = self.lowres_wire[24:-16]
            thumb, upsampled = decode_and_upsample_4x(webp_bytes, (h, w))
            restored = restore_coarse(upsampled, current)
            point = upper = exact_mse(current, restored)
            meta = dict(
                kind="4x downsampled WebP historical reference",
                lowres_state_bytes=len(self.lowres_wire),
                lowres_payload_bytes=len(webp_bytes),
                webp_q=self.lowres_quality,
            )
        tau = 10 ** (-self.gate_psnr / 10)
        priority = np.where(np.isfinite(upper), upper, tau + 1 + point)

        class DescriptorOnly:
            pass

        descriptor = DescriptorOnly()
        descriptor.version = self.version
        descriptor.digest = self.history
        descriptor.pending = None
        frame, cost, selected, trials = choose_packet(
            descriptor,
            np.asarray(current, np.uint8),
            self.region,
            observation,
            priority,
            tau,
            budget,
            quality=self.cache.config["roi_quality"],
        )
        wire = to_layered(frame)
        parsed = parse_frame(wire, self.cache.config["roi_quality"])
        cache = self.cache.updated(parsed["patches"])
        history = history_after(self.history, wire, cache)
        self.pending = wire
        self.provisional = (cache, history, self.version + 1, observation)

        if self.mode == "coarse_only":
            coarse_now = np.rint(current.reshape(h // 16, 16, w // 16, 16, 3).mean((1, 3))).astype(np.uint8)
            self.provisional_lowres_wire = b"LR16" + (b"\x00" * 20) + coarse_now.tobytes() + (b"\x00" * 16)
        else:
            webp_bytes = self.lowres_wire[24:-16]
            thumb, _ = decode_and_upsample_4x(webp_bytes, (h, w))
            thumb_next = thumb.copy()
            for idx, patch in parsed["patches"]:
                y, x = divmod(idx, w // 64)
                patch_4x = downsample_4x(patch)  # 16x16x3
                thumb_next[y * 16 : (y + 1) * 16, x * 16 : (x + 1) * 16] = patch_4x
            new_webp, q = encode_webp_bounded(thumb_next, MAX_PAYLOAD_BYTES)
            self.provisional_lowres_wire = b"LR4W" + (b"\x00" * 20) + new_webp + (b"\x00" * 16)
            self.lowres_quality = q

        return wire, dict(
            cost=cost,
            selected=selected,
            encoding_trials=trials,
            cache_bytes=len(self.provisional_lowres_wire),
            protocol_cache_bytes=len(cache.wire),
            reference_up_bytes=0,
            **meta,
        )

    def acknowledge(self, wire: bytes) -> None:
        cache, expected, v, t = self.provisional
        self.cache, self.history, self.version, self.observation = cache, expected, v, t
        self.lowres_wire = self.provisional_lowres_wire
        self.provisional_lowres_wire = None
        self.pending = self.provisional = None


def probability_with_shared_cache(
    evaluator: CrossfitGroundEvaluator,
    rgb: np.ndarray,
    own_folder: Path,
    shared_folder: Path,
    model_hash: str,
    threshold_hash: str,
    origin: dict,
) -> tuple[np.ndarray, dict]:
    """Reuse probability from existing OOF cache if native RGB SHA matches; otherwise run evaluator and save to own_folder."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    rgb_sha = common.sha_bytes(rgb.tobytes())
    key = common.canonical_hash(
        {
            "actual_native_RGB": rgb_sha,
            "shape": list(rgb.shape),
            "OOF_model_identity": model_hash,
            "threshold_hash": threshold_hash,
        }
    )
    shared_path = shared_folder / "probabilities" / (key + ".npz")
    own_path = own_folder / "probabilities" / (key + ".npz")
    path = shared_path if shared_path.exists() else own_path
    cached, timing = path.exists(), None
    if cached:
        with np.load(path, allow_pickle=False) as archive:
            if any(
                str(archive[k]) != value
                for k, value in {
                    "input_rgb_sha256": rgb_sha,
                    "model_hash": model_hash,
                    "threshold_hash": threshold_hash,
                }.items()
            ):
                raise ValueError("OOF probability cache identity mismatch")
            result = archive["probability"].copy()
    else:
        result = evaluator.predict(Image.fromarray(rgb))
        timing = evaluator.last_timing
        own_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = own_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            probability=result,
            input_rgb_sha256=rgb_sha,
            model_hash=model_hash,
            threshold_hash=threshold_hash,
            origin=json.dumps(origin),
        )
        tmp.replace(own_path)
        path = own_path
    return result, {
        "prediction_input_RGB_sha256": rgb_sha,
        "prediction_native_shape": list(rgb.shape),
        "probability_path": common.relative(path),
        "probability_file_sha256": common.sha(path),
        "probability_array_sha256": common.sha_bytes(result.tobytes()),
        "probability_cache_hit": cached,
        "prediction_inference_timing": timing,
    }


def run_pseudo_change_diagnostic(manifest: dict) -> dict:
    """Evaluate block score sensitivity to pseudo-changes vs real structural changes across all 8 confirmation AOIs."""
    records = []
    for ai, aoi in enumerate(manifest["aois"]):
        months = aoi["months"]
        first = pad(np.array(Image.open(ROOT / months[0]["rgb_path"])))
        last = pad(np.array(Image.open(ROOT / months[-1]["rgb_path"])))
        valid_first = pad(np.array(Image.open(ROOT / months[0]["valid_mask_path"])) > 0)
        valid_last = pad(np.array(Image.open(ROOT / months[-1]["valid_mask_path"])) > 0)
        event_path = EXISTING_MAIN / f"area_{ai}/gt" / aoi["aoi_id"] / months[-1]["month"] / "new_track.png"
        event_mask = pad(np.array(Image.open(event_path)) > 0) & valid_first & valid_last
        keep = blocks(np.repeat((valid_first & valid_last)[:, :, None], 3, 2)).mean((1, 2, 3)) >= 0.99
        event_block_frac = blocks(np.repeat(event_mask[:, :, None], 3, 2)).mean((1, 2, 3))
        pos_blocks = (event_block_frac > 0.01) & keep
        neg_blocks = (event_block_frac == 0.0) & keep

        # Synthetic pseudo-change perturbations of `first` (where zero structural change occurred)
        # 1. Uniform / block-smooth radiometric shift: 1.08 * x + 14
        pseudo_illum = np.clip(np.rint(first.astype(np.float32) * 1.08 + 14.0), 0, 255).astype(np.uint8)
        # 2. Mild optical blur (radius = 0.8)
        pseudo_blur = np.asarray(Image.fromarray(first).filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.uint8)
        # 3. 1-pixel sub- tile registration shift
        pseudo_shift = np.roll(np.roll(first, 1, axis=0), 1, axis=1)

        # Build references from `first`
        cfg_gauss = default_config("gaussian", 20261029)
        cfg_dct = default_config("dct_bound", 20261029)
        cache_gauss = DetailCache.from_rgb(first, cfg_gauss)
        cache_dct = DetailCache.from_rgb(first, cfg_dct)
        thumb4x = downsample_4x(first)
        webp_bytes, _ = encode_webp_bounded(thumb4x, MAX_PAYLOAD_BYTES)
        _, lowres_up = decode_and_upsample_4x(webp_bytes, first.shape[:2])
        coarse_16 = np.rint(first.reshape(first.shape[0] // 16, 16, first.shape[1] // 16, 16, 3).mean((1, 3))).astype(
            np.uint8
        )
        coarse_up = coarse_16.repeat(16, 0).repeat(16, 1)

        tau25 = 10 ** (-25 / 10)
        for rep_name in ("gaussian_20261029", "dct_bound", "lowres_4x_webp_12k", "lowres_16x16_coarse_only", "full_cache_mse"):
            def score_fn(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                if rep_name == "gaussian_20261029":
                    pt, up, _ = cache_gauss.scores(img, 1)
                    return pt, up
                if rep_name == "dct_bound":
                    pt, up, _ = cache_dct.scores(img, 1)
                    return pt, up
                if rep_name == "lowres_4x_webp_12k":
                    s = exact_mse(img, restore_coarse(lowres_up, img))
                    return s, s
                if rep_name == "lowres_16x16_coarse_only":
                    s = exact_mse(img, restore_coarse(coarse_up, img))
                    return s, s
                s = exact_mse(img, restore_coarse(first, img))
                return s, s

            pt_illum, up_illum = score_fn(pseudo_illum)
            pt_blur, up_blur = score_fn(pseudo_blur)
            pt_shift, up_shift = score_fn(pseudo_shift)
            pt_real, up_real = score_fn(last)

            # Top-K recall of real new-track blocks at K = 40 blocks (~32 KiB budget)
            k = 40
            valid_idx = np.flatnonzero(keep)
            top_k_idx = valid_idx[np.argsort(up_real[valid_idx])[::-1][:k]]
            top_mask = np.zeros_like(keep)
            top_mask[top_k_idx] = True
            pos_recall_top40 = (
                float((top_mask & pos_blocks).sum() / pos_blocks.sum()) if pos_blocks.sum() > 0 else float("nan")
            )

            records.append(
                {
                    "aoi": aoi["aoi_id"],
                    "method": rep_name,
                    "illum_shift_mean_point": float(pt_illum[keep].mean()),
                    "illum_shift_gate25_trig_frac": float((up_illum[keep] > tau25).mean()),
                    "blur_mean_point": float(pt_blur[keep].mean()),
                    "blur_gate25_trig_frac": float((up_blur[keep] > tau25).mean()),
                    "shift1px_mean_point": float(pt_shift[keep].mean()),
                    "shift1px_gate25_trig_frac": float((up_shift[keep] > tau25).mean()),
                    "real_pos_blocks_mean_point": float(pt_real[pos_blocks].mean()) if pos_blocks.any() else 0.0,
                    "real_neg_blocks_mean_point": float(pt_real[neg_blocks].mean()) if neg_blocks.any() else 0.0,
                    "real_pos_to_neg_ratio": (
                        float(pt_real[pos_blocks].mean() / max(pt_real[neg_blocks].mean(), 1e-9))
                        if pos_blocks.any() and neg_blocks.any()
                        else 1.0
                    ),
                    "real_top40_new_track_block_recall": pos_recall_top40,
                }
            )
    summary_by_method = {}
    for rep_name in ("gaussian_20261029", "dct_bound", "lowres_4x_webp_12k", "lowres_16x16_coarse_only", "full_cache_mse"):
        sub = [r for r in records if r["method"] == rep_name]
        summary_by_method[rep_name] = {
            k: float(np.nanmean([r[k] for r in sub]))
            for k in (
                "illum_shift_mean_point",
                "illum_shift_gate25_trig_frac",
                "blur_mean_point",
                "blur_gate25_trig_frac",
                "shift1px_mean_point",
                "shift1px_gate25_trig_frac",
                "real_pos_blocks_mean_point",
                "real_neg_blocks_mean_point",
                "real_pos_to_neg_ratio",
                "real_top40_new_track_block_recall",
            )
        }
    return {"per_aoi": records, "summary_by_method": summary_by_method}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = common.read(MANIFEST)
    calibration = common.read(ROOT / "resources/pretrained_ground_crossfit/calibration.json")
    threshold_spec = {"building": 0.5, "event": calibration["change_threshold"]}
    threshold_hash = common.canonical_hash(threshold_spec)

    transport_rows = []
    oof_rows = []

    if (OUT / "transport_records.csv").exists() and (OUT / "oof_records.csv").exists():
        with (OUT / "transport_records.csv").open(encoding="utf-8", newline="") as f:
            transport_rows = list(csv.DictReader(f))
        with (OUT / "oof_records.csv").open(encoding="utf-8", newline="") as f:
            oof_rows = list(csv.DictReader(f))

    need_run = len(transport_rows) != 1602 or len(oof_rows) != 1602
    if need_run:
        transport_rows = []
        oof_rows = []
    for index, aoi in (enumerate(manifest["aois"]) if need_run else []):
        area_out = OUT / f"area_{index}"
        area_out.mkdir(parents=True, exist_ok=True)
        shared_area = EXISTING_OOF / f"area_{index}"
        months = aoi["months"]
        first_native = np.asarray(Image.open(ROOT / months[0]["rgb_path"]).convert("RGB"))
        first_padded = pad(first_native)
        h_nat, w_nat = first_native.shape[:2]
        region = hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]

        evaluator = CrossfitGroundEvaluator(ROOT, aoi["aoi_id"])
        lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
        model_hash = common.canonical_hash(json.loads(evaluator.identity))
        reference_key = common.canonical_hash(
            dict(
                native_initial_RGB=common.sha_bytes(first_native.tobytes()),
                model=model_hash,
                threshold=threshold_hash,
            )
        )
        reference_prob, _ = probability_with_shared_cache(
            evaluator,
            first_native,
            area_out,
            shared_area,
            model_hash,
            threshold_hash,
            {"bootstrap_aoi": aoi["aoi_id"]},
        )

        first_valid = pad(np.array(Image.open(ROOT / months[0]["valid_mask_path"])) > 0)
        try:
            for method in LOWRES_METHODS:
                mode = "coarse_only" if method == "lowres_16x16_coarse_only" else "webp_4x"
                gate_psnr = 30 if method.endswith("_gate30") else 25
                cfg = default_config("dct_bound", 20261029)
                cfg["gate_psnr"] = gate_psnr
                for budget in BUDGETS:
                    sender = LowResLayeredSender(mode=mode, gate_psnr=gate_psnr)
                    receiver = LayeredReceiver()
                    boot, _ = sender.bootstrap(first_padded, region, cfg)
                    boot_ack = receiver.receive(boot)
                    sender.acknowledge(boot_ack)
                    cumulative_down = len(boot)
                    cumulative_up = len(boot_ack)
                    initial_info = dict(
                        bootstrap_path=f"area_{index}/{method}_{budget}/initial.bin",
                        bootstrap_down_bytes=len(boot),
                        bootstrap_ack_bytes=len(boot_ack),
                    )

                    for step, month in enumerate(months[1:], 1):
                        current_native = np.asarray(Image.open(ROOT / month["rgb_path"]).convert("RGB"))
                        current_padded = pad(current_native)
                        t0 = time.perf_counter()
                        packet, meta = sender.propose(current_padded, step, budget)
                        receiver.expect(step)
                        ack = receiver.receive(packet)
                        sender.acknowledge(ack)
                        compute_ms = (time.perf_counter() - t0) * 1000.0
                        if len(packet) > budget or len(ack) != 60:
                            raise AssertionError("Packet or ACK violated budget contract")
                        image_padded = receiver.state.render()
                        image_native = image_padded[:h_nat, :w_nat].copy()
                        statuses = receiver.state.statuses.copy()
                        cumulative_down += len(packet)
                        cumulative_up += len(ack)

                        # Pixel quality metrics
                        event_path = (
                            EXISTING_MAIN / f"area_{index}/gt" / aoi["aoi_id"] / month["month"] / "new_track.png"
                        )
                        valid = pad(np.array(Image.open(ROOT / month["valid_mask_path"])) > 0)
                        building = pad(np.array(Image.open(ROOT / month["building_mask_path"])) > 0)
                        event = pad(np.array(Image.open(event_path)) > 0) & valid & first_valid
                        keep = blocks(np.repeat(valid[:, :, None], 3, 2)).mean((1, 2, 3)) >= 0.99
                        err = exact_mse(current_padded, image_padded)
                        e = np.mean(((current_padded.astype(np.float64) - image_padded) / 255.0) ** 2, axis=-1)

                        t_rec = dict(
                            aoi=aoi["aoi_id"],
                            area_index=index,
                            method=method,
                            budget=budget,
                            step=step,
                            month=month["month"],
                            down_bytes=len(packet),
                            up_bytes=len(ack),
                            bootstrap_down_bytes=len(boot),
                            bootstrap_up_bytes=len(boot_ack),
                            cumulative_down_bytes=cumulative_down,
                            cumulative_up_bytes=cumulative_up,
                            sender_summary_bytes=meta["cache_bytes"],
                            sender_full_archive_bytes=0,
                            shared_projection_matrix_bytes=0,
                            compute_ms=compute_ms,
                            valid_mse=float(e[valid].mean()),
                            valid_blocks=int(keep.sum()),
                            stable_blocks=int(((statuses == 1) & keep).sum()),
                            updated_blocks=int(((statuses == 2) & keep).sum()),
                            deferred_blocks=int(((statuses == 3) & keep).sum()),
                            false_stable_blocks=int(((statuses == 1) & (err > 10 ** (-2.5)) & keep).sum()),
                        )
                        for name, mask in (("all", valid), ("building", building & valid), ("new_track", event)):
                            for thr in (25, 30, 35):
                                q = block_mask_quality(current_padded, image_padded, mask, psnr=thr)
                                t_rec.update({f"{name}_psnr{thr}_{k}": v for k, v in q.items()})
                        transport_rows.append(t_rec)

                        # OOF Ground Evaluation
                        gt = common.offline_masks(month, months[0], aoi["aoi_id"], event_path, True)
                        pred, pred_receipt = probability_with_shared_cache(
                            evaluator,
                            image_native,
                            area_out,
                            shared_area,
                            model_hash,
                            threshold_hash,
                            {"method": method, "budget": budget, "step": step},
                        )
                        event_score = np.maximum(pred - reference_prob, 0)
                        building_gt, event_gt, valid_gt, gt_hashes = gt
                        oof_rec = {
                            "aoi": aoi["aoi_id"],
                            "method": method,
                            "budget": budget,
                            "step": step,
                            "month": month["month"],
                            "down_bytes": len(packet),
                            "ack_bytes": len(ack),
                            "sender_summary_bytes": meta["cache_bytes"],
                            "compute_ms": compute_ms,
                            "probability_cache_hit": pred_receipt["probability_cache_hit"],
                            **common.flat_counts(
                                "building", common.counts(pred >= evaluator.threshold, building_gt, valid_gt)
                            ),
                            **common.flat_counts(
                                "event", common.counts(event_score >= evaluator.change_threshold, event_gt, valid_gt)
                            ),
                        }
                        oof_rows.append(oof_rec)
                    print(
                        f"[LowRes SN7] area_{index} ({aoi['aoi_id']}) {method} cap={budget} complete",
                        flush=True,
                    )
        finally:
            evaluator.close()

    if need_run:
        # Save transport and OOF records
        common.write_csv(OUT / "transport_records.csv", transport_rows)
        common.write_csv(OUT / "oof_records.csv", oof_rows)

    # Combine with existing confirmation OOF and transport records for unified 8-AOI comparison
    existing_oof_rows = []
    for i in range(8):
        with (EXISTING_OOF / f"area_{i}/records.csv").open(encoding="utf-8", newline="") as f:
            existing_oof_rows.extend(list(csv.DictReader(f)))
    with (ROOT / "results/tcm_sn7/ground_v1/records.csv").open(encoding="utf-8", newline="") as f:
        existing_oof_rows.extend(list(csv.DictReader(f)))

    existing_trans_rows = []
    for i in range(8):
        with (EXISTING_MAIN / f"area_{i}/records.csv").open(encoding="utf-8", newline="") as f:
            existing_trans_rows.extend(list(csv.DictReader(f)))
        with (EXISTING_GATE30 / f"area_{i}/records.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                r = dict(r)
                r["method"] = "full_cache_gate30"
                existing_trans_rows.append(r)

    all_methods = (
        "lowres_16x16_coarse_only",
        "lowres_4x_webp_12k",
        "lowres_4x_webp_12k_gate30",
        "gaussian_20261029",
        "dct_bound",
        "full_cache_mse",
        "full_cache_gate30",
        "full_webp",
        "learned_hyperprior_q1_8",
        "tcm",
    )

    oof_groups = defaultdict(list)
    for r in existing_oof_rows + oof_rows:
        if r["method"] in all_methods:
            oof_groups[r["method"], int(r["budget"]), r["aoi"]].append(r)

    trans_groups = defaultdict(list)
    for r in existing_trans_rows + transport_rows:
        if r["method"] in all_methods:
            trans_groups[r["method"], int(r["budget"]), r["aoi"]].append(r)

    areas = [a["aoi_id"] for a in manifest["aois"]]
    per_aoi_values = {}
    for m in all_methods:
        for cap in BUDGETS:
            for a in areas:
                grp = oof_groups[m, cap, a]
                point = {}
                for task in ("building", "event"):
                    tp, fp, fn = (sum(int(r[f"{task}_{name}"]) for r in grp) for name in ("tp", "fp", "fn"))
                    point[f"{task}_f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
                point["mean_bytes"] = float(np.mean([int(r["down_bytes"]) + int(r["ack_bytes"]) for r in grp]))
                t_grp = trans_groups.get((m, cap, a), [])
                if t_grp:
                    point["mean_sender_summary_bytes"] = float(
                        np.mean([float(r["sender_summary_bytes"]) for r in t_grp])
                    )
                    point["mean_updated_blocks"] = float(np.mean([float(r["updated_blocks"]) for r in t_grp]))
                    b_ret = sum(float(r["building_psnr30_usable_pixels"]) for r in t_grp)
                    b_tot = sum(float(r["building_psnr30_pixels"]) for r in t_grp)
                    e_ret = sum(float(r["new_track_psnr30_usable_pixels"]) for r in t_grp)
                    e_tot = sum(float(r["new_track_psnr30_pixels"]) for r in t_grp)
                    point["building_psnr30"] = b_ret / b_tot if b_tot else 0.0
                    point["new_track_psnr30"] = e_ret / e_tot if e_tot else 0.0
                else:
                    point["mean_sender_summary_bytes"] = 0.0
                    point["mean_updated_blocks"] = 0.0
                    point["building_psnr30"] = float("nan")
                    point["new_track_psnr30"] = float("nan")
                per_aoi_values[m, cap, a] = point

    summaries = []
    comparisons = []
    draws = np.random.default_rng(20261108).integers(0, 8, size=(10000, 8))
    for cap in BUDGETS:
        for m in all_methods:
            grp = [per_aoi_values[m, cap, a] for a in areas]
            summaries.append(
                dict(
                    method=m,
                    budget=cap,
                    AOIs=8,
                    **{k: float(np.nanmean([r[k] for r in grp])) for k in grp[0]},
                )
            )
        # Paired bootstrap comparisons for gaussian_20261029 vs lowres & other controls
        for ref_m in ("gaussian_20261029", "dct_bound"):
            for comp_m in (
                "lowres_16x16_coarse_only",
                "lowres_4x_webp_12k",
                "lowres_4x_webp_12k_gate30",
                "dct_bound",
                "full_cache_mse",
                "full_cache_gate30",
            ):
                if ref_m == comp_m:
                    continue
                for metric in ("building_f1", "event_f1", "building_psnr30", "new_track_psnr30"):
                    delta = np.array([per_aoi_values[ref_m, cap, a][metric] - per_aoi_values[comp_m, cap, a][metric] for a in areas])
                    lo95, hi95 = np.quantile(delta[draws].mean(1), [0.025, 0.975])
                    lo_bonf, hi_bonf = np.quantile(delta[draws].mean(1), [0.05 / 6, 1 - 0.05 / 6])
                    comparisons.append(
                        dict(
                            method=ref_m,
                            comparator=comp_m,
                            budget=cap,
                            metric=metric,
                            mean_difference=float(delta.mean()),
                            ci95_low=float(lo95),
                            ci95_high=float(hi95),
                            bonf_low=float(lo_bonf),
                            bonf_high=float(hi_bonf),
                        )
                    )

    pseudo_diag = run_pseudo_change_diagnostic(manifest)
    common.write_csv(OUT / "summary.csv", summaries)
    common.write_csv(OUT / "comparisons.csv", comparisons)
    write_json(
        OUT / "analysis.json",
        {
            "status": "complete",
            "scope": "Matched-storage low-resolution historical reference control (<=12,328 B/area) and same-version module/pseudo-change diagnostic on 8 SpaceNet 7 confirmation AOIs",
            "summaries": summaries,
            "comparisons": comparisons,
            "pseudo_change_diagnostic": pseudo_diag,
        },
    )
    print(json.dumps({"summaries_32k": [s for s in summaries if s["budget"] == 32768]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
