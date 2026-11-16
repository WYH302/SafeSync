"""Focused source/weight and actual receiver-RGB API verification; no fitting."""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safesync.pretrained_ground import PretrainedGroundEvaluator
from scripts.fetch_sn7_evaluator import official_etag_matches

OUT = ROOT / "resources/pretrained_ground"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    source = read(OUT / "source_identity.json")
    model_source = read(OUT / "model_source_identity.json")
    for item in source["source_files"]:
        if sha(OUT / "official_source" / item["path"]) != item["sha256"]:
            errors.append("official source file changed: " + item["path"])
    for item in model_source["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            errors.append("author model source changed: " + item["path"])
    weight = OUT / "model_best_exp_0200.pth"
    etag_ok = official_etag_matches(weight, source["weight"]["etag"])
    if not etag_ok:
        errors.append("official multipart ETag differs")
    started = time.perf_counter()
    evaluator = PretrainedGroundEvaluator(ROOT)
    constructor_ms = (time.perf_counter() - started) * 1000
    snapshot = read(OUT / "dataset_manifest.json")
    aoi = snapshot["aois"][2]
    first, second = aoi["months"][:2]
    reference = Image.open(ROOT / first["rgb_path"]).convert("RGB")
    current = Image.open(ROOT / second["rgb_path"]).convert("RGB")
    full = evaluator.predict_change(reference, current)
    full_timing = evaluator.last_timing
    old = evaluator.predict_change(reference, reference)
    old_timing = evaluator.last_timing
    deleted = evaluator.predict_change(reference, reference.copy())
    deleted_timing = evaluator.last_timing
    cached_full = np.maximum(evaluator.cached_predict(current) - evaluator.cached_predict(reference), 0)
    maximum_difference = float(np.max(np.abs(full - cached_full)))
    if maximum_difference > 2e-5 or not np.array_equal(old, deleted) or deleted.any():
        errors.append("actual RGB/cache/omitted-update inputs disagree")
    preserved = {}
    for folder in ["ground_eval", "ground_change_eval"]:
        metadata = read(ROOT / "resources" / folder / "metadata.json")
        preserved[folder] = sha(ROOT / "resources" / folder / "head.npz") == metadata["head_sha256"]
        if not preserved[folder]:
            errors.append("frozen negative-result head changed: " + folder)
    fixed = read(OUT / "calibration.json")
    results = read(OUT / "results.json")
    if fixed["frozen_utc"] > results["utc"]:
        errors.append("event calibration followed the development report")
    if abs(fixed["calibration_macro_event_f1"] - results["macro_aoi_by_role"]["calibration"]["events"]["full_current"]["f1"]) > 1e-12:
        errors.append("calibration threshold and final metric disagree")
    report = {"utc": datetime.now(timezone.utc).isoformat(), "passed": not errors, "errors": errors,
              "weight_sha256": sha(weight), "official_multipart_etag_verified": etag_ok,
              "strict_checkpoint_tensors": evaluator.metadata["strict_loaded_state_tensors"],
              "model_frozen": evaluator.metadata["model_frozen"], "source_files_verified": len(source["source_files"]) + len(model_source["files"]),
              "actual_RGB_max_abs_cache_difference": maximum_difference,
              "old_and_deleted_actual_RGB_outputs_equal": np.array_equal(old, deleted),
              "deleted_event_scores_all_zero": not bool(deleted.any()), "native_output_shape": list(full.shape),
              "original_negative_result_heads_preserved": preserved,
              "expanded_confirmation_images_or_labels_read": False}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "timing.json").write_text(json.dumps({"constructor_wall_ms_including_imports": constructor_ms,
        "constructor_model_stage_seconds": evaluator.cold_load_seconds,
        "full_RGB_pair": full_timing, "warm_old_RGB_pair": old_timing,
        "warm_deleted_RGB_pair": deleted_timing,
        "scope": "pair timers include actual reference/current model calls; imports/constructor separate; no prediction cache used"}, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
