"""Task metrics from actual SP03 receiver replay, never full sender RGB.

The old development trace lacks the original router source hash. This is a
historical distortion diagnostic, not a final confirmation experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from safesync.pixel_wire import PixelState, TransactionReceiver
from safesync.pretrained_ground import PretrainedGroundEvaluator

METHODS = ["gaussian_detail", "full_jpeg", "full_jpeg2000"]
BUDGETS = [32768, 65536]
DEFAULT_RUN = ROOT / "results/pixel_sequences/development_side4"
DEFAULT_OUT = ROOT / "results/pretrained_receipts/development_side4"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def canonical_hash(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def counts(predicted, truth, valid):
    p, y = predicted[valid], truth[valid]
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()),
            "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum()),
            "valid_pixels": int(valid.sum()), "positive_pixels": int(y.sum()),
            "predicted_positive_pixels": int(p.sum())}


def rates(value):
    tp, fp, fn = value["tp"], value["fp"], value["fn"]
    return {"f1": divide(2 * tp, 2 * tp + fp + fn), "iou": divide(tp, tp + fp + fn),
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn)}


def flat_counts(prefix, value):
    return {prefix + "_" + key: count for key, count in {**value, **rates(value)}.items()}


def selected_rows(run, manifest):
    aois = manifest["aois"][2:4]
    expected_ids = [a["aoi_id"] for a in aois]
    rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8", newline="")))
    selected = [r for r in rows if r["aoi"] in expected_ids and r["method"] in METHODS and int(r["budget"]) in BUDGETS]
    groups = {}
    for row in selected:
        groups.setdefault((row["aoi"], row["method"], int(row["budget"])), []).append(row)
    for aoi in aois:
        expected_months = [m["month"] for m in aoi["months"][1:]]
        for method in METHODS:
            for budget in BUDGETS:
                sequence = sorted(groups.get((aoi["aoi_id"], method, budget), []), key=lambda r: int(r["step"]))
                if [r["month"] for r in sequence] != expected_months:
                    raise ValueError("a sequence omits, duplicates, or reorders fixed real months")
    if len(selected) != 270 or len(groups) != 12:
        raise ValueError(f"expected fixed 270-output cohort, got {len(selected)}/{len(groups)}")
    return aois, groups


def prediction_from_receiver(evaluator, rgb, out, model_hash, threshold_hash, origin):
    """No source-file path is accepted: only the actual reconstructed array."""
    rgb = np.asarray(rgb, np.uint8)
    input_sha = sha_bytes(rgb.tobytes())
    key = canonical_hash({"decoded_native_RGB_sha256": input_sha, "shape": list(rgb.shape),
                          "model_hash": model_hash, "threshold_hash": threshold_hash})
    path = out / "probabilities" / (key + ".npz")
    cache_hit = path.exists()
    if cache_hit:
        with np.load(path, allow_pickle=False) as saved:
            probability = saved["probability"].copy()
            if str(saved["input_rgb_sha256"]) != input_sha or str(saved["model_hash"]) != model_hash:
                raise ValueError("reconstructed-image probability cache identity differs")
        timing = None
    else:
        probability = evaluator.predict(Image.fromarray(rgb))
        timing = evaluator.last_timing
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, probability=probability, input_rgb_sha256=input_sha,
                            model_hash=model_hash, threshold_hash=threshold_hash,
                            origin=json.dumps(origin, ensure_ascii=False))
        temporary.replace(path)
    if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
        raise ValueError("bad native reconstructed-image probability shape/values")
    return probability, {"prediction_input_RGB_sha256": input_sha,
                         "probability_path": path.relative_to(ROOT).as_posix(),
                         "probability_file_sha256": sha(path),
                         "probability_array_sha256": sha_bytes(probability.tobytes()),
                         "probability_cache_hit": cache_hit, "prediction_inference_timing": timing}


def offline_gt(month, first, aoi_id):
    """Only labels and valid masks are opened here; no current source RGB."""
    paths = {"building": ROOT / month["building_mask_path"],
             "current_valid": ROOT / month["valid_mask_path"],
             "first_valid": ROOT / first["valid_mask_path"],
             "event": ROOT / "resources/ground_change_eval/targets" / aoi_id / month["month"] / "new_building.png"}
    arrays = {key: np.asarray(Image.open(path)) > 0 for key, path in paths.items()}
    valid = arrays["current_valid"] & arrays["first_valid"]
    if any(value.shape != valid.shape for value in arrays.values()):
        raise ValueError("offline current/first labels do not share the native grid")
    return arrays["building"], arrays["event"], valid, {key: sha(path) for key, path in paths.items()}


def summarize(rows, expected_aois):
    by_aoi = {}
    prefixes = ["building", "event", "initial_only_building", "initial_only_event"]
    support_fields = ["tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels", "predicted_positive_pixels"]
    for row in rows:
        key = (row["method"], row["budget"], row["aoi"])
        group = by_aoi.setdefault(key, {"method": key[0], "budget": key[1], "aoi": key[2],
                    "frames": 0, "down_bytes": 0, "reference_bytes": 0, "ack_bytes": 0,
                    "initial_down_bytes": row["initial_down_bytes"], "initial_ack_bytes": row["initial_ack_bytes"],
                    **{prefix: {field: 0 for field in support_fields} for prefix in prefixes}})
        group["frames"] += 1
        for field in ["down_bytes", "reference_bytes", "ack_bytes"]:
            group[field] += row[field]
        for prefix in prefixes:
            for field in support_fields:
                group[prefix][field] += row[prefix + "_" + field]
    for group in by_aoi.values():
        group["total_down_including_initial"] = group["down_bytes"] + group["initial_down_bytes"]
        group["total_up_including_initial_ack"] = group["reference_bytes"] + group["ack_bytes"] + group["initial_ack_bytes"]
        for prefix in prefixes:
            group[prefix].update(rates(group[prefix]))
    macro = []
    for method in METHODS:
        for budget in BUDGETS:
            members = [v for key, v in by_aoi.items() if key[:2] == (method, budget)]
            if not members:
                continue
            result = {"method": method, "budget": budget, "aois_present": len(members),
                      "expected_aois": len(expected_aois), "frames": sum(m["frames"] for m in members)}
            for prefix in prefixes:
                result[prefix] = {}
                for metric in ["f1", "iou", "precision", "recall"]:
                    values = [m[prefix][metric] for m in members if m[prefix][metric] is not None]
                    result[prefix][metric] = float(np.mean(values)) if values else None
                    result[prefix][metric + "_defined_aoi_count"] = len(values)
            macro.append(result)
    return {"statistical_unit": "AOI; first aggregate TP/FP/FN within AOI, then macro average",
            "zero_denominator_rule": "null, including empty-truth-and-empty-prediction F1; never 100%",
            "per_aoi": list(by_aoi.values()), "macro_aoi": macro}


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    flat = [k for k in keys if k != "prediction_inference_timing" and not isinstance(rows[0][k], dict)]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in flat} for row in rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "resources/pretrained_ground/dataset_manifest.json"
    manifest = read_json(manifest_path)
    aois, groups = selected_rows(run, manifest)
    evaluator = PretrainedGroundEvaluator(ROOT)
    thresholds = {"building": evaluator.threshold, "event": evaluator.change_threshold}
    if thresholds != {"building": 0.5, "event": 0.9610000252723694}:
        raise ValueError("fixed pretrained task thresholds changed")
    model_hash, threshold_hash = canonical_hash(json.loads(evaluator.identity)), canonical_hash(thresholds)
    sources = [Path(__file__), ROOT / "safesync/pixel_wire.py", ROOT / "safesync/pretrained_ground.py",
               run / "records.csv", run / "run_spec.json", manifest_path,
               ROOT / "resources/pretrained_ground/calibration.json"]
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in sources}
    snapshot = out / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        shutil.copyfile(path, snapshot / str(path.relative_to(ROOT)).replace("\\", "__").replace("/", "__"))
    spec = {"run": run.relative_to(ROOT).as_posix(), "aoi_ids": [a["aoi_id"] for a in aois],
            "methods": METHODS, "budgets": BUDGETS, "expected_outputs": 270, "expected_sequences": 12,
            "model_hash": model_hash, "threshold_hash": threshold_hash, "thresholds": thresholds,
            "weight_sha256": evaluator.metadata["weight_sha256"], "source_hashes": hashes,
            "source_trace_limit": "historical development trace; original router source hash unavailable",
            "claim_scope": "fixed task-model communication distortion, not final confirmation or independent geographic generalization",
            "pretrained_training_overlap": read_json(ROOT / "resources/pretrained_ground/training_overlap.json"),
            "predict_input_contract": "actual PixelState RGB after decoding initial.bin and sequential SP03 packets only",
            "full_sender_RGB_or_features_opened_for_prediction": False,
            "new_confirmation_images_or_labels_read": False}
    spec_path = out / "run_spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("receipt evaluation specification changed; do not mix outputs")
    save_json(spec_path, spec)
    results, initial_predictions, initial_receipts = [], {}, {}
    for method in METHODS:
        for budget in BUDGETS:
            for aoi in aois:
                aid = aoi["aoi_id"]
                first = aoi["months"][0]
                month_map = {m["month"]: m for m in aoi["months"]}
                width, height = first["dimensions"]
                padded_h, padded_w = math.ceil(height/64)*64, math.ceil(width/64)*64
                region = hashlib.sha256(aid.encode()).digest()[:16]
                receiver = TransactionReceiver(PixelState(np.zeros((padded_h, padded_w, 3), np.uint8), region))
                initial_path = run / aid / "initial.bin"
                initial_wire = initial_path.read_bytes()
                initial_ack, _ = receiver.receive(initial_wire)
                initial_sha = sha_bytes(receiver.state.rgb.tobytes())
                if aid not in initial_predictions:
                    reference, receipt = prediction_from_receiver(evaluator, receiver.state.rgb[:height, :width], out,
                        model_hash, threshold_hash, {"initial_packet": initial_path.relative_to(ROOT).as_posix(),
                                                    "initial_packet_sha256": sha_bytes(initial_wire)})
                    initial_predictions[aid] = reference
                    initial_receipts[aid] = {**receipt, "receiver_full_RGB_sha256": initial_sha,
                        "initial_packet_sha256": sha_bytes(initial_wire), "initial_down_bytes": len(initial_wire),
                        "initial_ack_bytes": len(initial_ack)}
                    save_json(out / "initial_receipts.json", initial_receipts)
                elif initial_receipts[aid]["receiver_full_RGB_sha256"] != initial_sha:
                    raise ValueError("initial receiver differs across the fixed methods/budgets")
                reference = initial_predictions[aid]
                sequence = sorted(groups[(aid, method, budget)], key=lambda r: int(r["step"]))
                for source_row in sequence:
                    started = time.perf_counter()
                    packet_path = ROOT / source_row["packet_path"]
                    packet = packet_path.read_bytes()
                    packet_sha = sha_bytes(packet)
                    if len(packet) != int(source_row["down_bytes"]) or len(packet) > budget:
                        raise ValueError("packet bytes differ from trace or exceed fixed per-update budget")
                    if len(initial_wire) != int(source_row["initial_down_bytes"]) or len(initial_ack) != int(source_row["initial_ack_bytes"]):
                        raise ValueError("initial packet/ACK byte accounting differs")
                    receiver.expect(int(source_row["step"]))
                    ack, _ = receiver.receive(packet)
                    decoded_sha = sha_bytes(receiver.state.rgb.tobytes())
                    if decoded_sha != source_row["receiver_sha256"]:
                        raise ValueError("actual replayed receiver RGB SHA differs from recorded trace")
                    reference_path = packet_path.with_suffix(".ref")
                    reference_wire = reference_path.read_bytes() if reference_path.exists() else b""
                    if len(reference_wire) != int(source_row["reference_bytes"]) or len(ack) != int(source_row["ack_bytes"]):
                        raise ValueError("actual reference/ACK bytes differ")
                    if len(reference_wire) + len(ack) != int(source_row["up_bytes"]):
                        raise ValueError("uplink byte accounting differs")
                    month = month_map[source_row["month"]]
                    building, event, valid, gt_hashes = offline_gt(month, first, aid)
                    key = canonical_hash({"packet_sha256": packet_sha, "receiver_SHA": decoded_sha,
                        "model_hash": model_hash, "threshold_hash": threshold_hash,
                        "GT_sha256": gt_hashes, "evaluator_source_sha256": hashes["safesync/pretrained_ground.py"],
                        "evaluation_source_sha256": hashes["scripts/evaluate_pretrained_receipts.py"]})
                    record_path = out / "records" / (key + ".json")
                    if record_path.exists():
                        row = read_json(record_path)
                        if sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
                            raise ValueError("saved actual-receiver probability file changed")
                    else:
                        probability, receipt = prediction_from_receiver(evaluator, receiver.state.rgb[:height, :width], out,
                            model_hash, threshold_hash, {"packet_path": source_row["packet_path"], "packet_sha256": packet_sha,
                                                        "actual_receiver_RGB_sha256": decoded_sha})
                        event_score = np.maximum(probability - reference, 0)
                        row = {"aoi": aid, "method": method, "budget": budget, "month": source_row["month"],
                               "step": int(source_row["step"]), "packet_path": source_row["packet_path"],
                               "packet_sha256": packet_sha, "receiver_sha256_verified": decoded_sha,
                               "model_hash": model_hash, "threshold_hash": threshold_hash, "gt_sha256": gt_hashes,
                               "down_bytes": len(packet), "reference_bytes": len(reference_wire), "ack_bytes": len(ack),
                               "reference_sha256": sha_bytes(reference_wire) if reference_wire else None,
                               "ack_sha256": sha_bytes(ack), "initial_down_bytes": len(initial_wire),
                               "initial_ack_bytes": len(initial_ack), "initial_packet_sha256": sha_bytes(initial_wire),
                               **receipt,
                               **flat_counts("building", counts(probability >= evaluator.threshold, building, valid)),
                               **flat_counts("event", counts(event_score >= evaluator.change_threshold, event, valid)),
                               **flat_counts("initial_only_building", counts(reference >= evaluator.threshold, building, valid)),
                               **flat_counts("initial_only_event", counts(np.zeros_like(event, bool), event, valid)),
                               "evaluation_wall_ms": (time.perf_counter() - started) * 1000}
                        save_json(record_path, row)
                    results.append(row)
                    save_json(out / "progress.json", {"status": "running", "updated_utc": now(), "actual_outputs": len(results),
                               "expected_outputs": 270, "all_replayed_receiver_sha_match": True,
                               "aoi": aid, "method": method, "budget": budget, "month": row["month"]})
                    if len(results) == 1:
                        print("first actual receipt", json.dumps({k: row[k] for k in ["aoi", "method", "budget", "month", "building_f1", "event_f1", "down_bytes", "receiver_sha256_verified"]}), flush=True)
                summary = summarize(results, [a["aoi_id"] for a in aois])
                save_json(out / "summary.partial.json", summary)
                write_csv(out / "records.csv", results)
                last = next(item for item in summary["per_aoi"] if item["aoi"] == aid and item["method"] == method and item["budget"] == budget)
                print(f"receipt sequence {method} budget={budget} {aid}: {last['frames']} frames, building F1={last['building']['f1']}, event F1={last['event']['f1']}", flush=True)
            if method == "gaussian_detail":
                print("gaussian_detail budget complete", budget, flush=True)
    if len(results) != 270:
        raise ValueError("incomplete fixed receipt cohort")
    summary = summarize(results, [a["aoi_id"] for a in aois])
    summary.update(status="complete", outputs=270, sequences=12, generated_utc=now(),
                   all_replayed_receiver_sha_match=True, model_and_thresholds_unchanged=True,
                   historical_trace_router_hash_missing=True, independent_geographic_generalization_claim=False,
                   original_probability_input="actual decoded receiver RGB only")
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {"status": "complete", "updated_utc": now(), "actual_outputs": 270,
               "expected_outputs": 270, "all_replayed_receiver_sha_match": True,
               "record_csv_sha256": sha(out / "records.csv"), "summary_sha256": sha(out / "summary.json")})
    print("completed all 270 actual receiver outputs", json.dumps(summary["macro_aoi"]), flush=True)


if __name__ == "__main__":
    main()
