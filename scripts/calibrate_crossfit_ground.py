"""One bounded author-OOF calibration on the original two AOIs only."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from safesync.crossfit_ground import CrossfitGroundEvaluator
from scripts.train_ground_change_evaluator import histogram_counts, histogram_rates

OUT = ROOT / "resources/pretrained_ground_crossfit"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def counts(target, prediction):
    return dict(tp=int((target & prediction).sum()), fp=int((~target & prediction).sum()),
                fn=int((target & ~prediction).sum()), tn=int((~target & ~prediction).sum()))


def rates(value):
    tp, fp, fn = value["tp"], value["fp"], value["fn"]
    denominator = 2*tp+fp+fn
    return dict(value, f1=2*tp/denominator if denominator else None,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        iou=tp/(tp+fp+fn) if tp+fp+fn else None)


def main():
    started = time.perf_counter()
    manifest_path = ROOT / "data/spacenet7_pilot/development_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    selection = json.loads((OUT / "fold_selection.json").read_text("utf-8"))
    bindings = {row["aoi_id"]:row for row in selection["targets"] if row["role"] == "calibration"}
    areas = manifest["aois"][:2]
    assert set(bindings) == {area["aoi_id"] for area in areas} and len(areas) == 2
    receipts = json.loads((OUT / "download_receipts.json").read_text("utf-8"))
    assert receipts["complete"]
    weights = {row["experiment"]:row["sha256"] for row in receipts["weights"]}
    target_index_path = ROOT / "resources/ground_change_eval/target_index.json"
    targets = json.loads(target_index_path.read_text("utf-8"))
    events = {(row["aoi_id"], row["month"]):row for row in targets["observations"] if row["aoi_id"] in bindings}
    source_files = ["safesync/crossfit_ground.py", "safesync/pretrained_ground.py", "scripts/calibrate_crossfit_ground.py", "scripts/train_ground_change_evaluator.py"]
    spec = dict(schema="author-OOF-two-AOI-calibration-v1", dataset_manifest_sha256=sha(manifest_path),
        fold_selection_sha256=sha(OUT / "fold_selection.json"),
        weight_sha256_by_experiment=weights, calibration_aoi_ids=[area["aoi_id"] for area in areas],
        calibration_bindings=[bindings[area["aoi_id"]] for area in areas],
        static_threshold=.5, event_score="max(P_current-P_first,0)",
        event_threshold_rule="original rule: float32 0.001..0.999 step0.001, macro AOI F1, highest exact tie",
        model_training=False, confirmation_pixels_read=False, confirmation_receipts_processed=0,
        model_scope="author OOF validation-fold best-epoch models, not untouched-test checkpoints",
        target_index_sha256=sha(target_index_path),
        target_mask_sha256={f"{key[0]}:{key[1]}":sha(ROOT/row["event_mask_path"]) for key,row in events.items()},
        source_sha256={name:sha(ROOT/name) for name in source_files},
        prediction_cache="new model/weight+preprocessing+phase/threshold identity+actual supplied RGB; original0200 probabilities never reused")
    spec_path = OUT / "calibration_run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Fixed OOF calibration recipe differs")
    save(spec_path, spec)
    if (OUT / "calibration.json").exists() and (OUT / "calibration_results.json").exists():
        old = json.loads((OUT / "calibration.json").read_text("utf-8"))
        if old["run_spec_sha256"] != sha(spec_path):
            raise ValueError("Existing OOF threshold came from another recipe")
        print("OOF calibration already complete; retaining frozen threshold", old["change_threshold"], flush=True)
        return
    thresholds = np.arange(1,1000,dtype=np.float32)/np.float32(1000)
    curves, per_area, prediction_index = [], [], []
    for area in areas:
        aid, months = area["aoi_id"], area["months"]
        evaluator = CrossfitGroundEvaluator(ROOT, aid, _allow_unfrozen=True)
        initial = None
        first_valid = np.asarray(Image.open(ROOT/months[0]["valid_mask_path"])) > 0
        positive, negative = np.zeros(1000,np.int64), np.zeros(1000,np.int64)
        current_static = dict(tp=0,fp=0,fn=0,tn=0)
        old_static = dict(tp=0,fp=0,fn=0,tn=0)
        old_event = dict(tp=0,fp=0,fn=0,tn=0)
        valid_pixels, event_pixels, inference_times = 0, 0, []
        for position, month in enumerate(months):
            prediction_started = time.perf_counter()
            rgb_path = ROOT/month["rgb_path"]
            if sha(rgb_path) != month["derived_sha256"]["rgb_path"]:
                raise ValueError("Original calibration RGB changed")
            with Image.open(rgb_path) as image:
                rgb = image.convert("RGB")
                current = evaluator.cached_predict(rgb)
            timing = dict(evaluator.last_timing)
            if not timing.get("cache_hit",False):
                inference_times.append(timing["total_ms"])
            prediction_wall_ms=(time.perf_counter()-prediction_started)*1000
            prediction_index.append(dict(aoi_id=aid, month=month["month"], experiment=evaluator.experiment,
                weight_sha256=evaluator.metadata["weight_sha256"], rgb_sha256=sha(rgb_path),
                probability_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                identity_sha256=hashlib.sha256(evaluator.identity.encode()).hexdigest(),
                prediction_and_cache_wall_ms=prediction_wall_ms, timing=timing))
            save(OUT / "calibration_prediction_index.partial.json", prediction_index)
            if position == 0:
                initial = current
                continue
            valid = (np.asarray(Image.open(ROOT/month["valid_mask_path"])) > 0) & first_valid
            building = np.asarray(Image.open(ROOT/month["building_mask_path"])) > 0
            event = np.asarray(Image.open(ROOT/events[(aid,month["month"])]["event_mask_path"])) > 0
            score = np.maximum(current-initial,0)[valid]
            truth = event[valid]
            p,n = histogram_counts(truth,score,thresholds)
            positive += p; negative += n
            for key,value in counts(building[valid],current[valid]>=.5).items(): current_static[key]+=value
            for key,value in counts(building[valid],initial[valid]>=.5).items(): old_static[key]+=value
            for key,value in counts(truth,np.zeros_like(truth)).items(): old_event[key]+=value
            valid_pixels += int(valid.sum()); event_pixels += int(truth.sum())
            print(f"OOF calibration {len(prediction_index)}/49 {evaluator.experiment} {aid} {month['month']}",flush=True)
        curve = histogram_rates(positive,negative)["f1"]
        curves.append(curve)
        per_area.append(dict(aoi_id=aid, experiment=evaluator.experiment, metadata=evaluator.metadata,
            cold_load_seconds=evaluator.cold_load_seconds, evaluated_updates=len(months)-1,
            event_positive_pixels=event_pixels, valid_pixel_observations=valid_pixels,
            static_full_current=rates(current_static), static_old_only=rates(old_static),
            event_old_only=rates(old_event), event_score_full_current_hist_positive=positive.tolist(),
            event_score_full_current_hist_negative=negative.tolist(),
            model_inference_ms_mean=float(np.mean(inference_times)) if inference_times else None,
            model_inference_ms_samples=len(inference_times)))
        evaluator.close()
        del evaluator, initial
    objective = np.mean(curves,axis=0)
    best = int(np.flatnonzero(objective==objective.max())[-1])
    fixed = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        run_spec_sha256=sha(spec_path), fold_selection_sha256=spec["fold_selection_sha256"],
        weight_sha256_by_experiment=weights, building_threshold=.5, change_threshold=float(thresholds[best]),
        calibration_macro_event_f1=float(objective[best]), per_aoi_event_f1=[float(curve[best]) for curve in curves],
        candidate_count=len(thresholds), rule=spec["event_threshold_rule"],
        calibration_aoi_ids=spec["calibration_aoi_ids"], confirmation_scores_used=False,
        weights_refitted=False, author_validation_best_epoch_selection_disclosed=True)
    if (OUT / "calibration.json").exists():
        previous=json.loads((OUT / "calibration.json").read_text("utf-8"))
        if previous["run_spec_sha256"]!=fixed["run_spec_sha256"] or previous["change_threshold"]!=fixed["change_threshold"]:
            raise ValueError("Refusing to overwrite existing frozen OOF threshold")
        fixed=previous
    save(OUT / "calibration.json", fixed)
    np.savez(OUT/"event_calibration_curve.npz", thresholds=thresholds, macro_f1=objective, per_aoi_f1=np.stack(curves))
    for area in per_area:
        positive=np.array(area.pop("event_score_full_current_hist_positive"),dtype=np.int64)
        negative=np.array(area.pop("event_score_full_current_hist_negative"),dtype=np.int64)
        tp=int(positive[best+1:].sum()); fp=int(negative[best+1:].sum())
        area["event_full_current"]=rates(dict(tp=tp,fp=fp,fn=int(positive.sum())-tp,tn=int(negative.sum())-fp))
        area["deleted_all_updates"]=dict(area["event_old_only"])
        area["identical_old_inputs_imply_zero_event_score"]=True
    elapsed=time.perf_counter()-started
    times=[row["timing"]["total_ms"] for row in prediction_index if not row["timing"].get("cache_hit",False)]
    mean_ms=float(np.mean(times)) if times else None
    walls=[row["prediction_and_cache_wall_ms"] for row in prediction_index if not row["timing"].get("cache_hit",False)]
    results=dict(status="complete", scope="original two calibration AOIs only", calibration=fixed,
        per_aoi=per_area, probability_frames=len(prediction_index), expected_probability_frames=49,
        wall_seconds=elapsed, new_inference_ms_mean=mean_ms,
        projected_4806_inference_seconds=4806*mean_ms/1000 if mean_ms else None,
        prediction_and_cache_wall_ms_mean=float(np.mean(walls)) if walls else None,
        projected_4806_prediction_and_cache_seconds=4806*float(np.mean(walls))/1000 if walls else None,
        confirmation_pixels_read=False, confirmation_receipts_processed=0,
        main0200_results_or_probabilities_modified=False,
        original_pretrained_metadata_sha256=sha(ROOT/"resources/pretrained_ground/metadata.json"),
        original_pretrained_calibration_sha256=sha(ROOT/"resources/pretrained_ground/calibration.json"))
    save(OUT/"calibration_prediction_index.json",prediction_index)
    save(OUT/"calibration_results.json",results)
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()
