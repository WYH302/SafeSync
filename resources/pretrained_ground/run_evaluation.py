"""Fixed communication-distortion evaluation on the original four SN7 AOIs."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from safesync.pretrained_ground import PretrainedGroundEvaluator
from scripts.train_ground_evaluator import exact_average_precision, now, save_json, sha
from scripts.train_ground_change_evaluator import histogram_counts, histogram_rates

OUT = ROOT / "resources/pretrained_ground"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_spec(evaluator):
    original = ROOT / "resources/ground_eval/dataset_manifest.json"
    snapshot = read(original)
    if len(snapshot["aois"]) != 4:
        raise ValueError("use the original four-AOI snapshot only")
    own_snapshot = OUT / "dataset_manifest.json"
    if not own_snapshot.exists():
        own_snapshot.write_bytes(original.read_bytes())
    if sha(own_snapshot) != sha(original):
        raise ValueError("frozen communication-evaluation data changed")
    spec = {"snapshot_sha256": sha(own_snapshot), "model_identity": json.loads(evaluator.identity),
            "calibration_aoi_ids": [a["aoi_id"] for a in snapshot["aois"][:2]],
            "development_aoi_ids": [a["aoi_id"] for a in snapshot["aois"][2:]],
            "building_threshold": 0.5, "building_threshold_rule": "fixed author default; no fitting",
            "event_score": "max(P_current - P_first,0), from actual independently decoded supplied RGB",
            "event_threshold_rule": "one calibration-only macro AOI F1 selection, 0.001..0.999 step .001, highest tie",
            "statistical_unit": "AOI, not tile or month", "expanded_confirmation_AOIs_read": False,
            "scope": "fixed task distortion evaluator; pretrained training overlap possible/likely; no unseen-geography claim"}
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("fixed pretrained-evaluator experiment differs")
    save_json(spec_path, spec)
    # Reconstruct the author's fold from already-frozen directory identifiers only.
    # This never reads images or labels of the expanded confirmation AOIs.
    candidates = sorted(snapshot["selection"]["ordered_aoi_ids"])
    random.Random(777).shuffle(candidates)
    expected_fold0_validation = set(candidates[::5])
    save_json(OUT / "training_overlap.json", {
        "author_procedure": "sort 60 training AOIs, random.seed(777), shuffle, fold i uses shuffled[i::5] as validation",
        "source": "official_source/4-motokimura/code/tools/split_dataset.py; train_cnns.sh exp200 split0",
        "reconstruction_assumptions": "author used the same 60 public training IDs and published split script; exact original log not in checkpoint",
        "overlap_is_not_certified_absent": True,
        "four_AOIs": [{"aoi_id": a["aoi_id"], "expected_author_fold0_role": "validation" if a["aoi_id"] in expected_fold0_validation else "training"}
                       for a in snapshot["aois"]],
        "claim": "do not describe these scores as independent geographic generalization"})
    return snapshot, spec


def groups(snapshot):
    event_index = read(ROOT / "resources/ground_change_eval/target_index.json")
    event_rows = {(r["aoi_id"], r["month"]): r for r in event_index["observations"]}
    for position, aoi in enumerate(snapshot["aois"]):
        rows = []
        for month in aoi["months"]:
            event = event_rows[(aoi["aoi_id"], month["month"])]
            rows.append({**month, "event_mask_path": event["event_mask_path"]})
        yield aoi["aoi_id"], "calibration" if position < 2 else "development", rows


def masks(row, first):
    building = np.asarray(Image.open(ROOT / row["building_mask_path"])) > 0
    event = np.asarray(Image.open(ROOT / row["event_mask_path"])) > 0
    valid = np.asarray(Image.open(ROOT / row["valid_mask_path"])) > 0
    valid &= np.asarray(Image.open(ROOT / first["valid_mask_path"])) > 0
    return building, event, valid


def cached(evaluator, row):
    return evaluator.cached_predict(Image.open(ROOT / row["rgb_path"]).convert("RGB"))


def cache_predictions(evaluator, snapshot):
    records = []
    for aoi, role, rows in groups(snapshot):
        for row in rows:
            score = cached(evaluator, row)
            records.append({"aoi_id": aoi, "role": role, "month": row["month"], "rgb_path": row["rgb_path"],
                            "rgb_sha256": row["derived_sha256"]["rgb_path"], "shape": list(score.shape),
                            "timing": evaluator.last_timing})
            save_json(OUT / "prediction_index.partial.json", records)
            print(f"pretrained RGB {len(records)}/96 {role} {aoi} {row['month']}", flush=True)
    save_json(OUT / "prediction_index.json", records)


def calibrate(evaluator, snapshot, spec):
    path = OUT / "calibration.json"
    if path.exists():
        fixed = read(path)
        if fixed["run_spec_sha256"] != sha(OUT / "run_spec.json"):
            raise ValueError("pretrained calibration run specification differs")
        print("retaining fixed pretrained calibration", fixed["change_threshold"], flush=True)
        return fixed
    thresholds = np.arange(1, 1000, dtype=np.float32) / np.float32(1000)
    curves, records = [], []
    for aoi, role, rows in groups(snapshot):
        if role != "calibration":
            continue
        initial = cached(evaluator, rows[0])
        positive, negative = np.zeros(1000, np.int64), np.zeros(1000, np.int64)
        for row in rows[1:]:
            _, event, valid = masks(row, rows[0])
            score = np.maximum(cached(evaluator, row) - initial, 0)[valid]
            p, n = histogram_counts(event[valid], score, thresholds)
            positive += p
            negative += n
        rates = histogram_rates(positive, negative)
        curves.append(rates["f1"])
        records.append({"aoi_id": aoi, "event_positive_pixels": int(positive.sum()),
                        "valid_pixel_observations": int(positive.sum() + negative.sum())})
    objective = np.mean(curves, axis=0)
    best = np.flatnonzero(objective == objective.max())[-1]
    fixed = {"frozen_utc": now(), "weight_sha256": evaluator.metadata["weight_sha256"],
             "run_spec_sha256": sha(OUT / "run_spec.json"), "building_threshold": spec["building_threshold"],
             "change_threshold": float(thresholds[best]), "calibration_macro_event_f1": float(objective[best]),
             "per_aoi_f1": [float(c[best]) for c in curves], "candidate_count": len(thresholds),
             "calibration": records, "development_scores_used": False,
             "rule": spec["event_threshold_rule"], "weights_refitted": False}
    save_json(path, fixed)
    np.savez(OUT / "event_calibration_curve.npz", thresholds=thresholds, macro_f1=objective, per_aoi_f1=np.stack(curves))
    print("frozen pretrained event threshold", fixed["change_threshold"], "calibration F1", fixed["calibration_macro_event_f1"], flush=True)
    return fixed


def metrics(labels, scores, threshold):
    prediction = scores >= threshold
    tp, fp = int(np.sum(labels & prediction)), int(np.sum(~labels & prediction))
    fn, tn = int(np.sum(labels & ~prediction)), int(np.sum(~labels & ~prediction))
    # Constant scores have exactly prevalence AP, so no redundant giant sort.
    ap = float(labels.mean()) if np.ptp(scores) == 0 else exact_average_precision(labels, scores)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": tp / max(1, tp + fp + fn), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "ap": ap,
            "valid_pixel_observations": len(labels), "positive_pixels": int(labels.sum())}


def report(evaluator, snapshot, fixed):
    evaluator.threshold, evaluator.change_threshold = fixed["building_threshold"], fixed["change_threshold"]
    result = {"utc": now(), "frozen_calibration": fixed, "weight_sha256": evaluator.metadata["weight_sha256"],
              "scope": "fixed pretrained task distortion evaluator, original four development AOIs; training overlap disclosed",
              "model_training_overlap": read(OUT / "training_overlap.json"), "statistical_unit": "AOI",
              "metric_level": "native building/new-ID footprint pixels, not instance AP",
              "comparison": "initial month excluded; same current GT and current/first valid intersection",
              "per_aoi": [], "macro_aoi_by_role": {}}
    for aoi, role, rows in groups(snapshot):
        initial = cached(evaluator, rows[0])
        building_targets, event_targets, current_scores, old_scores, event_scores = [], [], [], [], []
        for row in rows[1:]:
            building, event, valid = masks(row, rows[0])
            current = cached(evaluator, row)
            building_targets.append(building[valid])
            event_targets.append(event[valid])
            current_scores.append(current[valid])
            old_scores.append(initial[valid])
            event_scores.append(np.maximum(current - initial, 0)[valid])
        truth, event_truth = np.concatenate(building_targets), np.concatenate(event_targets)
        current, old, change = np.concatenate(current_scores), np.concatenate(old_scores), np.concatenate(event_scores)
        unchanged = np.zeros_like(change)
        item = {"aoi_id": aoi, "role": role, "evaluated_months": [r["month"] for r in rows[1:]],
                "static": {"full_current": metrics(truth, current, evaluator.threshold),
                           "first_month_only": metrics(truth, old, evaluator.threshold)},
                "events": {"full_current": metrics(event_truth, change, evaluator.change_threshold),
                           "first_month_only": metrics(event_truth, unchanged, evaluator.change_threshold)}}
        for task in ["static", "events"]:
            item[task]["deleted_all_updates"] = dict(item[task]["first_month_only"])
        item["old_and_deleted_inputs_identical"] = True
        result["per_aoi"].append(item)
        print(f"fixed pretrained {role} {aoi}: static full/old={item['static']['full_current']['f1']:.4f}/{item['static']['first_month_only']['f1']:.4f}, event full/old={item['events']['full_current']['f1']:.4f}/{item['events']['first_month_only']['f1']:.4f}", flush=True)
    for role in ["calibration", "development"]:
        members = [r for r in result["per_aoi"] if r["role"] == role]
        result["macro_aoi_by_role"][role] = {task: {condition: {metric: float(np.mean([r[task][condition][metric] for r in members]))
                    for metric in ["f1", "iou", "precision", "recall", "ap"]}
                    for condition in ["full_current", "first_month_only", "deleted_all_updates"]}
                    for task in ["static", "events"]}
    # Real receiver RGB calls, including identity after deleting every update.
    _, _, smoke_rows = next(g for g in groups(snapshot) if g[1] == "development")
    reference_image = Image.open(ROOT / smoke_rows[0]["rgb_path"]).convert("RGB")
    current_image = Image.open(ROOT / smoke_rows[1]["rgb_path"]).convert("RGB")
    full_change = evaluator.predict_change(reference_image, current_image)
    full_timing = evaluator.last_timing
    deleted_change = evaluator.predict_change(reference_image, reference_image.copy())
    deleted_timing = evaluator.last_timing
    expected = np.maximum(cached(evaluator, smoke_rows[1]) - cached(evaluator, smoke_rows[0]), 0)
    difference = float(np.max(np.abs(full_change - expected)))
    if difference > 2e-5 or deleted_change.any():
        raise ValueError("actual RGB event API/cache replay or deletion input check failed")
    result["rgb_api_check"] = {"max_abs_cached_difference": difference, "deleted_event_scores_all_zero": True,
                                "current_sender_features_or_GT_used": False,
                                "reference_month": smoke_rows[0]["month"], "current_month": smoke_rows[1]["month"]}
    save_json(OUT / "timing.json", {"direct_actual_RGB_pair": full_timing, "direct_deleted_RGB_pair": deleted_timing,
                                    "cold_constructor_model_stage_seconds": evaluator.cold_load_seconds,
                                    "device": evaluator.device, "model": evaluator.metadata})
    save_json(OUT / "results.json", result)
    print(json.dumps(result["macro_aoi_by_role"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["cache", "calibrate", "report", "all"], default="all")
    args = parser.parse_args()
    evaluator = PretrainedGroundEvaluator(ROOT)
    snapshot, spec = freeze_spec(evaluator)
    if args.phase in ["cache", "all"]:
        cache_predictions(evaluator, snapshot)
    fixed = None
    if args.phase in ["calibrate", "all"]:
        fixed = calibrate(evaluator, snapshot, spec)
    if args.phase in ["report", "all"]:
        report(evaluator, snapshot, fixed or read(OUT / "calibration.json"))


if __name__ == "__main__":
    main()
