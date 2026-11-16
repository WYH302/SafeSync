"""Separate fixed-fold OOF task scores for actual lossless and old-state controls.

Consumes only the completed pure receiver reader. The original 4806-output OOF
experiment, model assignments and calibrated thresholds remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_lossless_control as controls
from resources.pretrained_ground_crossfit import run_receipts as oof
import numpy as np

common, lic = oof.common, oof.lic
RES = ROOT / "resources/pretrained_ground_crossfit"
OUT = RES / "lossless_controls_v1"
METHODS = ["full_lossless_png", "old_only"]


def input_plan():
    source = controls.checked_spec()
    transport = common.read(controls.OUT / "transport_progress.json")
    primary = common.read(oof.OUT / "run_spec.json")
    completed = common.read(oof.OUT / "complete.json")
    manifest = common.read(oof.MANIFEST)
    if completed["status"] != "complete" or completed["outputs"] != 4806:
        raise ValueError("complete the frozen 4806 OOF outputs before these added controls")
    if completed["source_spec_sha256"] != common.sha(oof.OUT / "run_spec.json"):
        raise ValueError("primary OOF completion identity changed")
    if transport["status"] != "complete" or transport["records"] != 356 or transport["actual_PNG_updates"] != 178:
        raise ValueError("actual lossless control transport is incomplete")
    if source["methods"] != METHODS or source["aoi_ids"] != primary["aoi_ids"]:
        raise ValueError("control methods or fixed cohort changed")
    hashes = dict(transport["source_and_receipt_sha256"])
    for name, expected in hashes.items():
        if common.sha(ROOT / name) != expected:
            raise ValueError("lossless producer receipt identity changed: " + name)
    paths = [Path(__file__), controls.OUT / "transport_progress.json", ROOT / "scripts/run_lossless_control.py",
             Path(oof.__file__), oof.OUT / "run_spec.json", oof.OUT / "complete.json", oof.MANIFEST,
             ROOT / "safesync/crossfit_ground.py", ROOT / "safesync/pretrained_ground.py",
             ROOT / "scripts/evaluate_layered_receipts.py", ROOT / "scripts/evaluate_pretrained_receipts.py",
             ROOT / "scripts/evaluate_learned_receipts.py", ROOT / "safesync/pixel_wire.py",
             RES / "fold_selection.json", RES / "calibration.json", RES / "download_receipts.json",
             RES / "cpu_tensor_metadata.json"]
    hashes.update({common.relative(path): common.sha(path) for path in paths})
    spec = {"schema": "author-fold-OOF-lossless-old-controls-v1", "methods": METHODS,
        "additional_control": True, "part_of_original_blind_confirmation": False,
        "scope": primary["scope"], "aoi_ids": primary["aoi_ids"], "fixed_fold_bindings": primary["fixed_fold_bindings"],
        "expected_AOIs": 8, "expected_updates_per_method": 178, "expected_outputs": 356, "cap_bytes": None,
        "thresholds": primary["thresholds"], "weight_sha256_by_experiment": primary["weight_sha256_by_experiment"],
        "source_hashes": hashes, "source_current_RGB_or_features_opened": False,
        "input_boundary": "pure actual packet receiver decoded native RGB; GT is offline after reconstruction",
        "cache_identity": "separate control OOF model/weight/preprocessing/calibration/threshold/native RGB bytes and shape",
        "old_exp0200_probabilities_reused": False, "original_4806_cohort_modified": False,
        "statistical_unit": "sum monthly counts within each AOI, then eight AOIs equally weighted",
        "control_scope": "one uncapped full PNG and one zero-update reference, not duplicated across the three budgets",
        "claim_scope": "input-information references at their actual byte costs, not guaranteed task-F1 upper/lower bounds"}
    return manifest, spec


def freeze(spec):
    OUT.mkdir(parents=True, exist_ok=True)
    oof.checked_freeze(OUT / "run_spec.json", spec)
    snap = OUT / "source_snapshot"
    snap.mkdir(exist_ok=True)
    for name, expected in spec["source_hashes"].items():
        path = snap / name.replace("/", "__")
        if not path.exists():
            shutil.copyfile(ROOT / name, path)
        if common.sha(path) != expected:
            raise ValueError("separate OOF control source snapshot changed")
    marker = OUT / "frozen_before_control_inference.json"
    if not marker.exists():
        common.save_json(marker, {"frozen_utc": common.now(), "spec_sha256": common.sha(OUT / "run_spec.json"),
                                  "expected_outputs": 356})


def measure(evaluator, rgb, wire, reference, initial, out, model_hash, threshold_hash, gt, spec_sha):
    building, event, valid, gt_hashes = gt
    actual_sha = common.sha_bytes(rgb.tobytes())
    if actual_sha != wire["receiver_native_RGB_sha256"]:
        raise ValueError("actual native RGB differs from the receiver receipt")
    identity = {key: wire[key] for key in ["aoi", "method", "step", "month"]}
    key = common.canonical_hash({"identity": identity, "actual_native_RGB": actual_sha, "wire": wire,
                                 "model": model_hash, "threshold": threshold_hash, "GT": gt_hashes, "spec": spec_sha})
    path = out / "records" / (key + ".json")
    if path.exists():
        row = common.read(path)
        if row["record_key"] != key or any(row[k] != value for k, value in identity.items()):
            raise ValueError("OOF control cached record identity changed")
        if common.sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
            raise ValueError("OOF control probability file changed")
        return row
    prediction, receipt = oof.probability(evaluator, rgb, out, model_hash, threshold_hash,
        {"method": wire["method"], "actual_packet": wire["packet_path"], "packet_sha256": wire["packet_sha256"]})
    event_score = np.maximum(prediction - reference, 0)
    if wire["method"] == "old_only":
        if np.any(event_score) or wire["current_confirmed"] or wire["down_bytes"] or wire["ack_bytes"]:
            raise ValueError("old-only must have identical input, zero event score, and no current confirmation/update")
    row = {**wire, **receipt, "record_key": key, "model_hash": model_hash, "threshold_hash": threshold_hash,
        "model_experiment": evaluator.experiment, "weight_sha256": evaluator.metadata["weight_sha256"],
        "author_validation_fold": evaluator.binding["author_validation_fold"], "gt_sha256": gt_hashes,
        "reference_receipt_key": initial["reference_receipt_key"], "reference_up_bytes": 0,
        "additional_control": True, "part_of_original_blind_confirmation": False,
        **common.flat_counts("building", common.counts(prediction >= evaluator.threshold, building, valid)),
        **common.flat_counts("event", common.counts(event_score >= evaluator.change_threshold, event, valid)),
        **common.flat_counts("initial_only_building", common.counts(reference >= evaluator.threshold, building, valid)),
        **common.flat_counts("initial_only_event", common.counts(np.zeros_like(event, bool), event, valid))}
    common.save_json(path, row)
    return row


def evaluate_area(index, aoi, spec):
    out = OUT / f"area_{index}"
    expected = 2 * (len(aoi["months"]) - 1)
    marker = out / "progress.json"
    if marker.exists() and common.read(marker)["status"] == "complete":
        done = common.read(marker)
        for key, name in [("records_sha256", "records.json"), ("summary_sha256", "summary.json"), ("spec_sha256", "run_spec.json")]:
            if done[key] != common.sha(out / name):
                raise ValueError("completed control AOI artifact changed")
        return common.read(out / "records.json")
    evaluator = oof.CrossfitGroundEvaluator(ROOT, aoi["aoi_id"])
    lic.require_no_codec_imports()
    lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
    model_hash = common.canonical_hash(json.loads(evaluator.identity))
    threshold_hash = common.canonical_hash(spec["thresholds"])
    spec_sha = common.sha(OUT / "run_spec.json")
    oof.checked_freeze(out / "run_spec.json", {"parent_spec_sha256": spec_sha, "area_index": index,
        "aoi_id": aoi["aoi_id"], "model_hash": model_hash, "threshold_hash": threshold_hash,
        "expected_outputs": expected, "model_identity": json.loads(evaluator.identity),
        "numerical_flags": lic.numerical_flags(evaluator.torch), "model_metadata": evaluator.metadata})
    rows, initials, gt_cache = [], {}, {}
    first = aoi["months"][0]
    months = {month["month"]: month for month in aoi["months"]}
    try:
        for method in METHODS:
            frames = controls.iter_receiver_frames(aoi, method)
            initial_rgb, initial = next(frames)
            reference, receipt = oof.probability(evaluator, initial_rgb, out, model_hash, threshold_hash,
                                                 {"bootstrap": initial["bootstrap_path"]})
            reference_key = common.canonical_hash({"native_initial_RGB": common.sha_bytes(initial_rgb.tobytes()),
                "shape": list(initial_rgb.shape), "model": model_hash, "threshold": threshold_hash})
            initial = {**initial, **receipt, "reference_receipt_key": reference_key}
            initials[method] = initial
            common.save_json(out / "initial_receipts.json", initials)
            for rgb, wire in frames:
                month = wire["month"]
                if month not in gt_cache:
                    target = oof.MAIN / f"area_{index}" / "gt" / aoi["aoi_id"] / month / "new_track.png"
                    gt_cache[month] = common.offline_masks(months[month], first, aoi["aoi_id"], target, True)
                rows.append(measure(evaluator, rgb, wire, reference, initial, out, model_hash,
                                    threshold_hash, gt_cache[month], spec_sha))
                common.save_json(marker, {"status": "in_progress", "outputs": len(rows), "expected": expected,
                                          "method": method, "updated_utc": common.now()})
        if len(rows) != expected or len({(r["method"], r["step"]) for r in rows}) != expected:
            raise ValueError("OOF control AOI has incomplete or repeated observations")
        lic.require_no_codec_imports()
        lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
        common.save_json(out / "records.json", rows)
        common.write_csv(out / "records.csv", rows)
        common.save_json(out / "summary.json", {**common.aggregate(rows), "status": "complete", "outputs": len(rows),
            "parent_spec_sha256": spec_sha, "model_hash": model_hash, "threshold_hash": threshold_hash})
        common.save_json(marker, {"status": "complete", "outputs": len(rows), "expected": expected,
            "records_sha256": common.sha(out / "records.json"), "summary_sha256": common.sha(out / "summary.json"),
            "spec_sha256": common.sha(out / "run_spec.json"), "updated_utc": common.now()})
        return rows
    finally:
        evaluator.close()


def summarize(rows, spec):
    if len(rows) != 356 or len({(r["aoi"], r["method"], r["step"]) for r in rows}) != 356:
        raise ValueError("OOF control complete cohort differs")
    result = common.aggregate(rows)
    groups = result["per_aoi_seed"]
    primary = common.read(oof.OUT / "summary.json")
    baseline = {g["aoi"]: g for g in primary["per_aoi_seed"] if g["method"] == "gaussian_20261029" and g["budget"] == 32768}
    for group in groups:
        for task in ["building", "event"]:
            for support in ["valid_pixels", "positive_pixels"]:
                if group[task][support] != baseline[group["aoi"]][task][support]:
                    raise ValueError("added OOF controls must use the same GT/valid support as the 4806 outputs")
        if group["method"] == "old_only" and group["event"]["f1"] != 0:
            raise ValueError("old-only event metric differs from the fixed zero-event control")
    macro = []
    for method in METHODS:
        members = [g for g in groups if g["method"] == method]
        row = {"method": method, "AOIs": len(members), "updates": sum(g["frames"] for g in members), "cap_bytes": None}
        for task in ["building", "event"]:
            row[task] = {metric: common.mean_defined([g[task][metric] for g in members]) for metric in common.METRICS}
        for name in ["mean_update_bidirectional_bytes", "total_bytes_including_bootstrap", "bootstrap_bidirectional_bytes"]:
            row[name] = float(np.mean([g[name] for g in members]))
        macro.append(row)
    summary = {"status": "complete", "outputs": len(rows), "expected_updates_per_method": 178, "AOIs": 8,
        "scope": spec["scope"], "control_scope": spec["control_scope"], "claim_scope": spec["claim_scope"],
        "thresholds": spec["thresholds"], "cap_bytes": None, "per_aoi_seed": groups, "macro_aoi": macro,
        "source_spec_sha256": common.sha(OUT / "run_spec.json"), "same_GT_and_valid_support_as_main_OOF": True,
        "all_actual_receiver_RGB_ACK_states_verified": True, "old_only_event_F1_is_zero_all_AOIs": True,
        "source_current_RGB_or_features_opened": False, "old_exp0200_probabilities_reused": False}
    common.save_json(OUT / "records.json", rows)
    common.write_csv(OUT / "records.csv", rows)
    common.save_json(OUT / "summary.json", summary)
    common.write_csv(OUT / "macro.csv", [{"method": r["method"], "AOIs": r["AOIs"], "updates": r["updates"],
        **{f"{task}_{metric}": r[task][metric] for task in ["building", "event"] for metric in common.METRICS},
        **{key: r[key] for key in ["mean_update_bidirectional_bytes", "total_bytes_including_bootstrap", "bootstrap_bidirectional_bytes"]}} for r in macro])
    common.save_json(OUT / "complete.json", {"status": "complete", "outputs": 356, "expected_updates_per_method": 178, "AOIs": 8,
        "source_spec_sha256": common.sha(OUT / "run_spec.json"), "summary_sha256": common.sha(OUT / "summary.json"),
        "records_sha256": common.sha(OUT / "records.json"), "completed_utc": common.now()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    lic.require_no_codec_imports()
    manifest, spec = input_plan()
    if args.check_inputs:
        print(json.dumps({"expected_outputs": 356, "thresholds": spec["thresholds"], "AOIs": spec["expected_AOIs"]}))
        return
    freeze(spec)
    started, rows = time.perf_counter(), []
    for index, aoi in enumerate(manifest["aois"]):
        rows.extend(evaluate_area(index, aoi, spec))
        print(json.dumps({"area": index, "outputs": len(rows), "expected": 356,
                          "invocation_wall_seconds": time.perf_counter() - started}), flush=True)
    summary = summarize(rows, spec)
    print(json.dumps(summary["macro_aoi"]), flush=True)


if __name__ == "__main__":
    main()
