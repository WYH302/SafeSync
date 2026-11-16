"""Additional uncapped lossless-current and zero-update old-state controls.

SP03/SA02 packets and ACKs are executed before offline quality/task evaluation.
The exported reader reconstructs RGB from packets; it never opens source RGB.
Original confirmation scripts, streams and results are read-only dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_layered_receipts as common
from scripts.evaluate_pixel_receipts import block_mask_quality
from safesync.pixel_wire import (
    CODECS, HEADER, PixelState, TransactionReceiver, TransactionSender, UPDATED,
)
import numpy as np
from PIL import Image

OUT = ROOT / "results/lossless_control"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
MAIN = ROOT / "results/layered/confirmation_v1"
TASK_MAIN = ROOT / "results/layered_tasks/confirmation_v1"
METHODS = ["full_lossless_png", "old_only"]
THRESHOLDS = {"building": .5, "event": .9610000252723694}
SUPPORT = common.SUPPORT
MODEL_KEYS = ["model", "official_commit", "model_source_commit", "weight_sha256", "preprocessing"]


def fixed_json(path, value):
    if path.exists():
        if common.read(path) != value:
            raise ValueError("frozen control identity changed: " + str(path))
    else:
        common.save_json(path, value)


def frozen_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("committed packet/ACK differs: " + str(path))
    if not path.exists():
        path.write_bytes(data)


def native_rgb(month):
    with Image.open(ROOT / month["rgb_path"]) as image:
        rgb = np.array(image.convert("RGB"))
    if rgb.shape != (month["dimensions"][1], month["dimensions"][0], 3):
        raise ValueError("source RGB/native grid changed")
    return rgb


def pad(array):
    h, w = array.shape[:2]
    return np.pad(array, [(0, (-h) % 64), (0, (-w) % 64)] + ([(0, 0)] if array.ndim == 3 else []))


def event_path(index, aid, month):
    return MAIN / f"area_{index}" / "gt" / aid / month / "new_track.png"


def model_identity():
    metadata = common.read(ROOT / "resources/pretrained_ground/metadata.json")
    calibration = common.read(ROOT / "resources/pretrained_ground/calibration.json")
    if {"building": calibration["building_threshold"], "event": calibration["change_threshold"]} != THRESHOLDS:
        raise ValueError("fixed exp0200 task thresholds changed")
    return common.canonical_hash({key: metadata[key] for key in MODEL_KEYS}), common.canonical_hash(THRESHOLDS)


def specification():
    manifest = common.read(MANIFEST)
    if len(manifest["aois"]) != 8 or sum(len(a["months"]) - 1 for a in manifest["aois"]) != 178:
        raise ValueError("the fixed eight-AOI/178-update cohort changed")
    files = [Path(__file__), MANIFEST, ROOT / "safesync/pixel_wire.py",
        ROOT / "safesync/pretrained_ground.py", ROOT / "scripts/evaluate_layered_receipts.py",
        ROOT / "scripts/evaluate_pretrained_receipts.py", ROOT / "scripts/evaluate_pixel_receipts.py",
        ROOT / "resources/pretrained_ground/metadata.json", ROOT / "resources/pretrained_ground/calibration.json",
        ROOT / "resources/pretrained_ground/weight_receipt.json", TASK_MAIN / "summary.json",
        ROOT / "results/layered_tasks/confirmation_training_overlap.json"]
    model_hash, threshold_hash = model_identity()
    spec = {"schema": "additional-lossless-and-old-controls-v1", "methods": METHODS,
        "additional_control": True, "part_of_original_blind_confirmation": False,
        "history": "appended after original confirmation to measure actual full-information and old-state task performance",
        "expected_AOIs": 8, "expected_updates_per_method": 178, "expected_task_outputs": 356,
        "aoi_ids": [a["aoi_id"] for a in manifest["aois"]], "cap_bytes": None,
        "full_information": "all 64x64 blocks in every current RGB update; PNG level6, no lossy transform, no budget search",
        "old_only": "same lossless first-month bootstrap, no subsequent packet or ACK; receiver.expect(step) exposes UNKNOWN statuses",
        "bootstrap": "eight actual SP03 full PNG initializations plus SA02 60-byte ACK, counted separately for each alternative",
        "protocol": "SP03/SA02; all header, manifest, affine, coarse header, index, PNG, checksum and ACK bytes counted",
        "thresholds": THRESHOLDS, "model_hash": model_hash, "threshold_hash": threshold_hash,
        "weight_sha256": common.read(ROOT / "resources/pretrained_ground/weight_receipt.json")["sha256"],
        "source_hashes": {common.relative(path): common.sha(path) for path in files},
        "prediction_input": "actual decoded native receiver RGB only; source images used only by sender and separate offline equality/quality checks",
        "event_target": "frozen current official footprint track Id absent from initial Id set; same raster as original confirmation",
        "task_valid": "current and initial native valid intersection; no receiver-staleness exclusion",
        "quality": "same block-mask U30 implementation; all/building use current valid; new-track uses current AND initial valid",
        "statistical_unit": "sum month TP/FP/FN within AOI, then eight AOIs equally weighted; no replicated 32/64/128 KiB controls",
        "claim_scope": "input-information references at measured costs, not guaranteed upper/lower task-F1 bounds"}
    return manifest, spec


def freeze():
    manifest, spec = specification()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed_json(OUT / "run_spec.json", spec)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, expected in spec["source_hashes"].items():
        target = snapshot / name.replace("/", "__")
        if not target.exists():
            shutil.copyfile(ROOT / name, target)
        if common.sha(target) != expected:
            raise ValueError("control source snapshot changed: " + name)
    marker = OUT / "frozen_before_control_generation.json"
    if not marker.exists():
        common.save_json(marker, {"frozen_utc": common.now(), "run_spec_sha256": common.sha(OUT / "run_spec.json")})
    return manifest, spec


def checked_spec():
    spec = common.read(OUT / "run_spec.json")
    for name, expected in spec["source_hashes"].items():
        if common.sha(ROOT / name) != expected or common.sha(OUT / "source_snapshot" / name.replace("/", "__")) != expected:
            raise ValueError("control source/snapshot identity changed: " + name)
    return spec


def send_png(receiver, sender, rgb, step, path):
    h, w = rgb.shape[:2]
    started = time.perf_counter()
    packet, costs = sender.propose(shape=(h, w), block=64, region=receiver.state.region,
        observed_at=step, current_rgb=rgb, indices=np.arange(h * w // 4096),
        statuses=np.full(h * w // 4096, UPDATED, np.uint8), codec="png")
    receiver.expect(step)
    ack, _ = receiver.receive(packet)
    sender.acknowledge(ack)
    elapsed = (time.perf_counter() - started) * 1000
    if sum(costs.values()) != len(packet) or len(ack) != 60 or not np.array_equal(rgb, receiver.state.rgb):
        raise ValueError("PNG exact RGB/actual byte accounting failed")
    if sender.digest != receiver.state.digest() or np.any(receiver.status() != UPDATED):
        raise ValueError("full PNG sender/receiver state differs")
    frozen_bytes(path, packet)
    frozen_bytes(path.with_suffix(".ack"), ack)
    return {"packet_path": common.relative(path), "packet_sha256": common.sha_bytes(packet),
        "ack_sha256": common.sha_bytes(ack), "down_bytes": len(packet), "ack_bytes": len(ack),
        "cost_breakdown": costs, "source_encoding_ms": elapsed}


def state_receipt(receiver, height, width):
    current_status = receiver.status()
    return {"receiver_RGB_sha256_verified": common.sha_bytes(receiver.state.rgb.tobytes()),
        "receiver_native_RGB_sha256": common.sha_bytes(receiver.state.rgb[:height, :width].tobytes()),
        "receiver_native_shape": [height, width, 3], "receiver_history_digest": receiver.state.digest().hex(),
        "receiver_version": receiver.state.version, "receiver_observed_at": receiver.state.observed_at,
        "expected_observed_at": receiver.expected_at,
        "current_confirmed": receiver.state.observed_at >= receiver.expected_at,
        "current_status_counts": {name: int((current_status == code).sum()) for code, name in enumerate(["unknown", "stable", "updated", "deferred"])},
        "receiver_status_sha256": common.sha_bytes(current_status.tobytes())}


def generate(manifest):
    started = time.perf_counter()
    records, initial = [], []
    for index, aoi in enumerate(manifest["aois"]):
        aid, months = aoi["aoi_id"], aoi["months"]
        folder = OUT / f"area_{index}"
        receipt_path = folder / "transport_receipts.json"
        if receipt_path.exists():
            saved = common.read(receipt_path)
            records.extend(saved["updates"])
            initial.append(saved["initial"])
            continue
        rgb = native_rgb(months[0])
        height, width = rgb.shape[:2]
        padded = pad(rgb)
        region = hashlib.sha256(aid.encode()).digest()[:16]
        receiver = TransactionReceiver(PixelState(np.zeros_like(padded), region))
        sender = TransactionSender(receiver.state.version, receiver.state.digest())
        boot = send_png(receiver, sender, padded, 0, folder / "initial.bin")
        init = {"aoi": aid, "area_index": index, "step": 0, "month": months[0]["month"],
            "bootstrap_path": boot["packet_path"], "bootstrap_packet_sha256": boot["packet_sha256"],
            "bootstrap_ack_sha256": boot["ack_sha256"], "bootstrap_down_bytes": boot["down_bytes"],
            "bootstrap_ack_bytes": boot["ack_bytes"], "bootstrap_cost_breakdown": boot["cost_breakdown"],
            "source_RGB_path": months[0]["rgb_path"], "source_RGB_file_sha256": common.sha(ROOT / months[0]["rgb_path"]),
            "source_native_RGB_sha256": common.sha_bytes(rgb.tobytes()), "decoded_equals_source_native": True,
            **state_receipt(receiver, height, width)}
        old = TransactionReceiver(PixelState(np.zeros_like(padded), region))
        old.expect(0)
        old.receive((ROOT / init["bootstrap_path"]).read_bytes())
        down, up, updates = boot["down_bytes"], boot["ack_bytes"], []
        for step, month in enumerate(months[1:], 1):
            current = native_rgb(month)
            if current.shape != rgb.shape:
                raise ValueError("native dimensions changed within fixed sequence")
            source = {"source_RGB_path": month["rgb_path"], "source_RGB_file_sha256": common.sha(ROOT / month["rgb_path"]),
                "source_native_RGB_sha256": common.sha_bytes(current.tobytes())}
            wire = send_png(receiver, sender, pad(current), step, folder / "full_lossless_png" / f"{step:03d}.bin")
            down, up = down + wire["down_bytes"], up + wire["ack_bytes"]
            shared = {"aoi": aid, "area_index": index, "step": step, "month": month["month"], "budget": None,
                "cap_bytes": None, "bootstrap_path": init["bootstrap_path"],
                "bootstrap_down_bytes": boot["down_bytes"], "bootstrap_ack_bytes": boot["ack_bytes"], **source}
            updates.append({**shared, **wire, "method": METHODS[0], "protocol": "SP03/SA02",
                "cumulative_down_bytes": down, "cumulative_up_bytes": up, "decoded_equals_source_native": True,
                **state_receipt(receiver, height, width)})
            old.expect(step)
            old_state = state_receipt(old, height, width)
            if old_state["current_confirmed"] or old_state["current_status_counts"]["unknown"] != len(old.state.statuses):
                raise ValueError("old-only control falsely confirms a current observation")
            updates.append({**shared, "method": METHODS[1], "protocol": "SP03 initialization only; no update",
                "packet_path": None, "packet_sha256": None, "ack_sha256": None, "down_bytes": 0, "ack_bytes": 0,
                "cost_breakdown": {}, "source_encoding_ms": 0.,
                "cumulative_down_bytes": boot["down_bytes"], "cumulative_up_bytes": boot["ack_bytes"],
                "decoded_equals_source_native": bool(np.array_equal(old.state.rgb[:height, :width], current)), **old_state})
        fixed_json(receipt_path, {"initial": init, "updates": updates})
        records.extend(updates)
        initial.append(init)
        common.save_json(OUT / "transport_progress.json", {"status": "in_progress", "completed_AOIs": index + 1,
            "records": len(records), "expected": 356})
        print(f"transport area {index}: {len(months)-1} exact PNG updates and old-only observations", flush=True)
    if len(records) != 356:
        raise ValueError("incomplete control transport cohort")
    fixed_json(OUT / "transport_records.json", records)
    fixed_json(OUT / "initial_receipts.json", initial)
    paths = [OUT / "run_spec.json", OUT / "transport_records.json", OUT / "initial_receipts.json"]
    paths += [OUT / f"area_{i}" / "transport_receipts.json" for i in range(8)]
    common.save_json(OUT / "transport_progress.json", {"status": "complete", "completed_AOIs": 8,
        "records": 356, "expected": 356, "actual_PNG_updates": 178, "bootstrap_transactions": 8,
        "source_and_receipt_sha256": {common.relative(p): common.sha(p) for p in paths},
        "all_PNG_native_and_padded_RGB_equal_source": True, "generation_wall_seconds_this_invocation": time.perf_counter()-started})


def iter_receiver_frames(aoi, method):
    """Yield (native RGB, receipt): bootstrap first, then each fixed observation.

    Requires completed transport; does not read labels or source RGB. The caller
    can perform its own fixed-model inference with these actual decoded images.
    """
    if method not in METHODS:
        raise ValueError("unknown information control")
    manifest = common.read(MANIFEST)
    index = next(i for i, value in enumerate(manifest["aois"]) if value["aoi_id"] == aoi["aoi_id"])
    progress = common.read(OUT / "transport_progress.json")
    if progress["status"] != "complete":
        raise ValueError("control transport incomplete")
    path = OUT / f"area_{index}" / "transport_receipts.json"
    if common.sha(path) != progress["source_and_receipt_sha256"][common.relative(path)]:
        raise ValueError("control receipt file changed")
    saved = common.read(path)
    init, first = saved["initial"], aoi["months"][0]
    width, height = first["dimensions"]
    receiver = TransactionReceiver(PixelState(np.zeros_like(pad(np.zeros((height, width, 3), np.uint8))), hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]))
    bootstrap_path = ROOT / init["bootstrap_path"]
    bootstrap, stored_ack = bootstrap_path.read_bytes(), bootstrap_path.with_suffix(".ack").read_bytes()
    receiver.expect(0)
    ack, _ = receiver.receive(bootstrap)
    if ack != stored_ack or common.sha_bytes(bootstrap) != init["bootstrap_packet_sha256"] or common.sha_bytes(ack) != init["bootstrap_ack_sha256"]:
        raise ValueError("control bootstrap bytes/ACK changed")
    if len(bootstrap) != init["bootstrap_down_bytes"] or len(ack) != init["bootstrap_ack_bytes"]:
        raise ValueError("control bootstrap actual cost differs")
    if state_receipt(receiver, height, width) != {key: init[key] for key in state_receipt(receiver, height, width)}:
        raise ValueError("control bootstrap state changed")
    yield receiver.state.rgb[:height, :width].copy(), init
    rows = [row for row in saved["updates"] if row["method"] == method]
    if [(r["step"], r["month"]) for r in rows] != list(enumerate([m["month"] for m in aoi["months"][1:]], 1)):
        raise ValueError("control fixed observation order changed")
    down, up = len(bootstrap), len(ack)
    for row in rows:
        receiver.expect(row["step"])
        if method == METHODS[0]:
            path = ROOT / row["packet_path"]
            packet, stored_ack = path.read_bytes(), path.with_suffix(".ack").read_bytes()
            fields = HEADER.unpack(packet[:HEADER.size])
            if fields[0] != b"SP03" or fields[1] != CODECS["png"] or fields[9] != len(receiver.state.statuses):
                raise ValueError("full lossless packet lacks full PNG absolute content")
            ack, _ = receiver.receive(packet)
            if ack != stored_ack or common.sha_bytes(packet) != row["packet_sha256"] or common.sha_bytes(ack) != row["ack_sha256"]:
                raise ValueError("control packet/ACK changed")
            if len(packet) != row["down_bytes"] or len(ack) != row["ack_bytes"] or sum(row["cost_breakdown"].values()) != len(packet):
                raise ValueError("control actual packet cost differs")
            down, up = down + len(packet), up + len(ack)
        elif row["down_bytes"] or row["ack_bytes"] or row["packet_path"] is not None:
            raise ValueError("old-only must receive no update or ACK")
        actual = state_receipt(receiver, height, width)
        if any(actual[key] != row[key] for key in actual) or down != row["cumulative_down_bytes"] or up != row["cumulative_up_bytes"]:
            raise ValueError("replayed control state/cumulative bytes changed")
        yield receiver.state.rgb[:height, :width].copy(), row


def quality(manifest):
    """Offline source/mask reading after completed packet generation."""
    rows = []
    for index, aoi in enumerate(manifest["aois"]):
        first, aid = aoi["months"][0], aoi["aoi_id"]
        for method in METHODS:
            iterator = iter_receiver_frames(aoi, method)
            initial_rgb, init = next(iterator)
            if common.sha(ROOT / first["rgb_path"]) != init["source_RGB_file_sha256"] or not np.array_equal(initial_rgb, native_rgb(first)):
                raise ValueError("initial packet differs from original native RGB")
            for month, (received, receipt) in zip(aoi["months"][1:], iterator):
                current = native_rgb(month)
                if common.sha(ROOT / month["rgb_path"]) != receipt["source_RGB_file_sha256"]:
                    raise ValueError("original source RGB changed after transmission")
                equal = bool(np.array_equal(current, received))
                if equal != receipt["decoded_equals_source_native"] or (method == METHODS[0] and not equal):
                    raise ValueError("lossless current RGB equality failed")
                gt_path = event_path(index, aid, month["month"])
                building, event, valid, gt_hashes = common.offline_masks(month, first, aid, gt_path, True)
                current_valid = np.asarray(Image.open(ROOT / month["valid_mask_path"])) > 0
                qrow = {"aoi": aid, "method": method, "step": receipt["step"], "month": month["month"],
                    "gt_sha256": gt_hashes, "new_track_target_path": common.relative(gt_path),
                    "new_track_target_sha256": common.sha(gt_path), "task_valid_pixels": int(valid.sum()),
                    "task_event_positive_pixels": int((event & valid).sum()), "decoded_equals_source_native": equal}
                for label, mask in [("all", current_valid), ("building", building & current_valid), ("new_track", event & valid)]:
                    measured = block_mask_quality(pad(current), pad(received), pad(mask), psnr=30)
                    qrow.update({f"{label}_psnr30_{key}": value for key, value in measured.items()})
                if qrow["new_track_psnr30_pixels"] != qrow["task_event_positive_pixels"]:
                    raise ValueError("quality/task new-track support differs")
                rows.append(qrow)
        print(f"offline quality area {index}: exact PNG replay and old-only UNKNOWN verified", flush=True)
    fixed_json(OUT / "quality_records.json", rows)
    common.save_json(OUT / "replay_verification.json", {"status": "complete", "replayed_control_outputs": len(rows),
        "PNG_updates_equal_native_source": 178, "bootstrap_equal_native_source": 8,
        "old_only_current_confirmed": 0, "old_only_stable_statuses": 0,
        "quality_records_sha256": common.sha(OUT / "quality_records.json")})


def probability_cache_paths():
    folders = [OUT / "probabilities"]
    folders += [ROOT / base / f"area_{i}" / "probabilities" for base in [
        "results/layered_tasks/confirmation_v1", "results/full_cache_tasks/gate30",
        "results/learned_tasks/confirmation_q1_8_v1"] for i in range(8)]
    return folders + list(common.OLD_CACHES)


def probability_key(rgb, spec):
    return common.canonical_hash({"decoded_native_RGB_sha256": common.sha_bytes(rgb.tobytes()),
        "shape": list(rgb.shape), "model_hash": spec["model_hash"], "threshold_hash": spec["threshold_hash"]})


def cache_audit(manifest, spec):
    missing, found, seen = [], [], set()
    for aoi in manifest["aois"]:
        for rgb, receipt in iter_receiver_frames(aoi, METHODS[0]):
            key = probability_key(rgb, spec)
            if key in seen:
                continue
            seen.add(key)
            paths = [folder / (key + ".npz") for folder in probability_cache_paths()]
            match = next((p for p in paths if p.exists()), None)
            if match is None:
                missing.append({"aoi": aoi["aoi_id"], "step": receipt["step"], "cache_key": key})
            else:
                with np.load(match, allow_pickle=False) as saved:
                    expected = {"input_rgb_sha256": common.sha_bytes(rgb.tobytes()), "model_hash": spec["model_hash"], "threshold_hash": spec["threshold_hash"]}
                    if any(str(saved[k]) != v for k, v in expected.items()) or saved["probability"].shape != rgb.shape[:2]:
                        raise ValueError("existing exact-RGB cache identity mismatch")
                found.append({"cache_key": key, "path": common.relative(match), "sha256": common.sha(match)})
    report = {"unique_actual_RGB_inputs": len(seen), "exact_identity_cache_hits": len(found),
        "new_predictions_required": len(missing), "found": found, "missing": missing,
        "model_loaded_or_GPU_used": False}
    common.save_json(OUT / "cache_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ["found", "missing"]}), flush=True)


def evaluate(manifest, spec):
    class LazyEvaluator:
        def __init__(self):
            self.instance, self.last_timing = None, None

        def predict(self, image):
            if self.instance is None:
                self.instance = common.PretrainedGroundEvaluator(ROOT)
                if common.canonical_hash(json.loads(self.instance.identity)) != spec["model_hash"]:
                    raise ValueError("loaded model identity differs from frozen control")
            result = self.instance.predict(image)
            self.last_timing = self.instance.last_timing
            return result

    evaluator = LazyEvaluator()
    common.OLD_CACHES = probability_cache_paths()[1:]
    quality_rows = {(r["aoi"], r["method"], r["step"]): r for r in common.read(OUT / "quality_records.json")}
    rows, initial_predictions = [], []
    for index, aoi in enumerate(manifest["aois"]):
        first, aid = aoi["months"][0], aoi["aoi_id"]
        reference = None
        for method in METHODS:
            iterator = iter_receiver_frames(aoi, method)
            rgb, init = next(iterator)
            if reference is None:
                reference, reference_receipt = common.probability_from_receiver(evaluator, rgb, OUT,
                    spec["model_hash"], spec["threshold_hash"], {"bootstrap": init["bootstrap_path"]})
                initial_record = OUT / "task_records" / f"{index}_initial.json"
                if initial_record.exists():
                    initial_prediction = common.read(initial_record)
                    if initial_prediction["prediction_input_RGB_sha256"] != common.sha_bytes(rgb.tobytes()) or common.sha(ROOT / initial_prediction["probability_path"]) != initial_prediction["probability_file_sha256"]:
                        raise ValueError("resumed initial prediction changed")
                else:
                    initial_prediction = {"aoi": aid, **reference_receipt}
                    common.save_json(initial_record, initial_prediction)
                initial_predictions.append(initial_prediction)
                common.save_json(OUT / "initial_prediction_receipts.json", initial_predictions)
            for month, (received, transport) in zip(aoi["months"][1:], iterator):
                qrow = quality_rows[(aid, method, transport["step"])]
                gt_path = event_path(index, aid, month["month"])
                building, event, valid, gt_hashes = common.offline_masks(month, first, aid, gt_path, True)
                if gt_hashes != qrow["gt_sha256"]:
                    raise ValueError("offline task GT changed since quality evaluation")
                record_path = OUT / "task_records" / f"{index}_{method}_{transport['step']:03d}.json"
                if record_path.exists():
                    row = common.read(record_path)
                    expected = {**transport, **qrow, "model_hash": spec["model_hash"], "threshold_hash": spec["threshold_hash"],
                        "prediction_input_RGB_sha256": common.sha_bytes(received.tobytes())}
                    if any(row[key] != value for key, value in expected.items()) or common.sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
                        raise ValueError("resumed task-record identity changed")
                else:
                    probability, pred_receipt = common.probability_from_receiver(evaluator, received, OUT,
                        spec["model_hash"], spec["threshold_hash"], {"method": method, "aoi": aid, "step": transport["step"], "packet": transport["packet_path"]})
                    event_score = np.maximum(probability - reference, 0)
                    if method == METHODS[1] and np.any(event_score):
                        raise ValueError("old-only identical RGB must produce zero event score")
                    row = {**transport, **qrow, **pred_receipt, "model_hash": spec["model_hash"], "threshold_hash": spec["threshold_hash"],
                        **common.flat_counts("building", common.counts(probability >= THRESHOLDS["building"], building, valid)),
                        **common.flat_counts("event", common.counts(event_score >= THRESHOLDS["event"], event, valid))}
                    common.save_json(record_path, row)
                rows.append(row)
                common.save_json(OUT / "progress.json", {"status": "in_progress", "completed": len(rows), "expected": 356})
        print(f"exp0200 controls area {index}: {len(rows)}/356 task outputs", flush=True)
    fixed_json(OUT / "task_records.json", rows)
    summarize(rows, spec)


def summarize(rows, spec):
    if len(rows) != 356 or len({(r["aoi"], r["method"], r["step"]) for r in rows}) != 356:
        raise ValueError("fixed task-control cohort incomplete or duplicated")
    groups, macro = [], []
    for method in METHODS:
        for aid in spec["aoi_ids"]:
            members = [r for r in rows if r["method"] == method and r["aoi"] == aid]
            group = {"aoi": aid, "method": method, "updates": len(members),
                "down_bytes": sum(r["down_bytes"] for r in members), "ack_bytes": sum(r["ack_bytes"] for r in members),
                "bootstrap_down_bytes": members[0]["bootstrap_down_bytes"], "bootstrap_ack_bytes": members[0]["bootstrap_ack_bytes"]}
            group["actual_update_bidirectional_bytes"] = group["down_bytes"] + group["ack_bytes"]
            group["bootstrap_bidirectional_bytes"] = group["bootstrap_down_bytes"] + group["bootstrap_ack_bytes"]
            group["mean_update_bidirectional_bytes"] = group["actual_update_bidirectional_bytes"] / len(members)
            group["amortized_bytes_per_update"] = (group["actual_update_bidirectional_bytes"] + group["bootstrap_bidirectional_bytes"]) / len(members)
            for task in ["building", "event"]:
                count = {key: sum(r[task + "_" + key] for r in members) for key in SUPPORT}
                group[task] = {**count, **common.rates(count)}
            for quality_name in ["all", "building", "new_track"]:
                for name in ["pixels", "usable_pixels", "error_sum"]:
                    key = f"{quality_name}_psnr30_{name}"
                    group[key] = sum(r[key] for r in members)
                pixels = group[f"{quality_name}_psnr30_pixels"]
                group[quality_name + "_U30"] = group[f"{quality_name}_psnr30_usable_pixels"] / pixels if pixels else None
            if group["new_track_psnr30_pixels"] != group["event"]["positive_pixels"]:
                raise ValueError("aggregated task/quality event support differs")
            groups.append(group)
    support = {task: [aid for aid in spec["aoi_ids"] if all(g[task]["positive_pixels"] > 0 and g[task]["valid_pixels"] > 0 for g in groups if g["aoi"] == aid)] for task in ["building", "event"]}
    for method in METHODS:
        members = [g for g in groups if g["method"] == method]
        row = {"method": method, "AOIs": len(members), "updates": sum(g["updates"] for g in members), "cap_bytes": None}
        for name in ["mean_update_bidirectional_bytes", "amortized_bytes_per_update", "bootstrap_bidirectional_bytes"]:
            row[name] = float(np.mean([g[name] for g in members]))
        for task in ["building", "event"]:
            for metric in common.METRICS:
                row[task + "_" + metric] = common.mean_defined([g[task][metric] for g in members if g["aoi"] in support[task]])
        row["new_track_U30"] = common.mean_defined([g["new_track_U30"] for g in members if g["aoi"] in support["event"]])
        macro.append(row)
    summary = {"schema": "additional-lossless-and-old-task-summary-v1", "status": "complete", "outputs": len(rows),
        "additional_control": True, "part_of_original_blind_confirmation": False, "cap_bytes": None,
        "model_hash": spec["model_hash"], "threshold_hash": spec["threshold_hash"], "thresholds": THRESHOLDS,
        "support_AOI_ids": support, "per_aoi": groups, "macro_aoi": macro,
        "actual_update_bidirectional_bytes_by_method": {m: sum(g["actual_update_bidirectional_bytes"] for g in groups if g["method"] == m) for m in METHODS},
        "bootstrap_bytes_per_alternative": sum(g["bootstrap_bidirectional_bytes"] for g in groups if g["method"] == METHODS[0]),
        "old_only_current_confirmation": False, "old_only_stable_statuses": 0,
        "lossless_equal_native_source_updates": 178, "old_only_event_F1_expected_zero_and_measured": all(g["event"]["f1"] == 0 for g in groups if g["method"] == METHODS[1]),
        "new_update_predictions": sum(r["prediction_inference_timing"] is not None for r in rows),
        "empty_task_valid_control_outputs": sum(r["building_valid_pixels"] == 0 for r in rows),
        "source_hashes": {common.relative(OUT / name): common.sha(OUT / name) for name in ["run_spec.json", "transport_records.json", "initial_receipts.json", "quality_records.json", "task_records.json", "initial_prediction_receipts.json"]},
        "training_overlap": "all eight confirmation AOIs expected in public exp0200 gradient-training fold; fixed-model distortion measurements only",
        "information_reference_is_guaranteed_task_upper_bound": False, "independent_geographic_generalization_claim": False}
    common.save_json(OUT / "summary.json", summary)
    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    common.save_json(OUT / "progress.json", {"status": "complete", "completed": 356, "expected": 356,
        "summary_sha256": common.sha(OUT / "summary.json"), "comparison_sha256": common.sha(OUT / "comparison.csv"),
        "task_records_sha256": common.sha(OUT / "task_records.json")})
    print(json.dumps(macro, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="CPU packet/equality/quality/cache work; no model load")
    modes.add_argument("--evaluate-only", action="store_true", help="replay complete controls and evaluate fixed exp0200")
    modes.add_argument("--summarize-only", action="store_true", help="counts-only, no model inference")
    args = parser.parse_args()
    if args.summarize_only:
        spec = checked_spec()
        summarize(common.read(OUT / "task_records.json"), spec)
        return
    if args.evaluate_only:
        spec, manifest = checked_spec(), common.read(MANIFEST)
    else:
        manifest, spec = freeze()
        generate(manifest)
        quality(manifest)
        cache_audit(manifest, spec)
    if not args.prepare_only:
        evaluate(manifest, spec)


if __name__ == "__main__":
    main()
