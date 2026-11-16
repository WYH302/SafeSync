"""Freeze one RGB-pair head for building IDs absent from the initial month."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from safesync.ground import GROUND_CONFIG, soft_building_targets
from safesync.ground_change import GroundChangeEvaluator, make_pair_head, pair_features
from scripts.train_ground_evaluator import exact_average_precision, now, save_json, sha

OUT = ROOT / "resources/ground_change_eval"
STATIC = ROOT / "resources/ground_eval"
FIT = {"seed": 20261028, "epochs": 12, "batch_size": 16384, "learning_rate": 0.002,
       "weight_decay": 0.001, "feature_std_floor": 0.001,
       "input": "concat(reference DINO token, received-current DINO token, absolute token difference)",
       "identity_negative_ratio": "one reference/reference self-pair for each real calibration pair; target zero",
       "loss": "valid-area weighted soft-target BCEWithLogits",
       "positive_weight": "min(30, max(1, sqrt(negative valid pixel-equivalent area / positive area))), calibration examples only",
       "threshold_candidates": "0.001..0.999 in increments of 0.001, float32",
       "threshold_objective": "macro AOI F1 on real calibration update pairs only, excluding identity augmentation",
       "threshold_tie_rule": "highest threshold among exact macro-F1 ties",
       "selection": "one fixed 12-epoch recipe; no development-validation feedback"}


def prepare():
    # Rasterio is isolated for the Python-3.12 data-preparation interpreter.
    sys.path.insert(0, str(ROOT / "data/spacenet7_pilot/.deps"))
    import rasterio
    from affine import Affine
    from rasterio.features import rasterize
    snapshot_path = STATIC / "dataset_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    cache = json.loads((STATIC / "cache_index.json").read_text(encoding="utf-8"))
    if len(snapshot["aois"]) != 4:
        raise ValueError("expanded confirmation AOIs must not enter this experiment")
    spec = {"dataset_manifest_sha256": sha(snapshot_path), "feature_index_sha256": sha(STATIC / "cache_index.json"),
            "backbone_identity": cache["spec"]["backbone_identity"], "encoder_config": GROUND_CONFIG,
            "calibration_aoi_ids": cache["spec"]["calibration_aoi_ids"],
            "validation_aoi_ids": cache["spec"]["validation_aoi_ids"], "fit": FIT,
            "target": "current official labels_match footprints whose Id is absent from first-month Id set",
            "observability": "current valid AND first-month valid; cloud and padding used offline only",
            "statistical_unit": "AOI; repeated months, tiles, and identity examples are correlated",
            "scope": "original four-AOI development experiment only"}
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("new event recipe differs from the frozen specification")
    save_json(spec_path, spec)
    cache_rows = {(r["aoi_id"], r["month"]): r for r in cache["observations"]}
    rows = []
    provenance = []
    for aoi in snapshot["aois"]:
        first = aoi["months"][0]
        first_labels = json.loads((ROOT / first["label_geojson_path"]).read_text(encoding="utf-8"))
        first_ids = {f["properties"]["Id"] for f in first_labels["features"]
                     if f.get("geometry") and f["geometry"].get("coordinates")}
        first_cache = cache_rows[(aoi["aoi_id"], first["month"])]
        for month in aoi["months"]:
            current_path = ROOT / month["label_geojson_path"]
            current_labels = json.loads(current_path.read_text(encoding="utf-8"))
            label_crs = rasterio.crs.CRS.from_string(current_labels["crs"]["properties"]["name"])
            if label_crs != rasterio.crs.CRS.from_string(month["crs"]):
                raise ValueError("labels_match CRS differs from the native raster")
            novel = [f for f in current_labels["features"] if f.get("geometry")
                     and f["geometry"].get("coordinates") and f["properties"]["Id"] not in first_ids]
            shape = tuple(month["dimensions"][::-1])
            mask = rasterize([(f["geometry"], 1) for f in novel], out_shape=shape,
                             transform=Affine(*month["native_transform"]), all_touched=False,
                             fill=0, dtype="uint8") if novel else np.zeros(shape, np.uint8)
            folder = OUT / "targets" / aoi["aoi_id"] / month["month"]
            folder.mkdir(parents=True, exist_ok=True)
            target_path = folder / "new_building.png"
            Image.fromarray(mask).save(target_path)
            source = cache_rows[(aoi["aoi_id"], month["month"])]
            if source["dimensions"] != first_cache["dimensions"]:
                raise ValueError("native pair grids differ; explicit registration would be required")
            tiles = []
            for current_tile, initial_tile in zip(source["tiles"], first_cache["tiles"], strict=True):
                if any(current_tile[k] != initial_tile[k] for k in ["x", "y", "width", "height"]):
                    raise ValueError("native tile coordinates differ")
                tiles.append({**current_tile, "reference_feature_path": initial_tile["feature_path"]})
            row = {**source, "tiles": tiles, "event_mask_path": target_path.relative_to(ROOT).as_posix(),
                   "initial_month": first["month"], "reference_rgb_path": first["rgb_path"],
                   "reference_valid_path": first["valid_mask_path"],
                   "is_initial": month["month"] == first["month"], "new_building_id_count": len(novel)}
            rows.append(row)
            provenance.append({"aoi_id": aoi["aoi_id"], "month": month["month"], "role": source["role"],
                               "first_month": first["month"], "first_id_count": len(first_ids),
                               "current_new_ids": [f["properties"]["Id"] for f in novel],
                               "current_label_path": month["label_geojson_path"], "current_label_sha256": sha(current_path),
                               "first_label_sha256": sha(ROOT / first["label_geojson_path"]),
                               "event_mask_sha256": sha(target_path), "native_transform": month["native_transform"],
                               "dimensions": month["dimensions"], "crs": month["crs"]})
    save_json(OUT / "target_index.json", {"spec": spec, "observations": rows})
    save_json(OUT / "target_provenance.json", {"utc": now(), "definition": spec["target"],
               "rasterio": rasterio.__version__, "GDAL": rasterio.__gdal_version__,
               "rasterization": "pixel-center, all_touched=False, native affine; holes preserved", "observations": provenance})
    print("prepared", len(rows), "genuine monthly new-ID targets; first-month IDs are fixed", flush=True)


def load_index():
    index = json.loads((OUT / "target_index.json").read_text(encoding="utf-8"))
    if index["spec"] != json.loads((OUT / "run_spec.json").read_text(encoding="utf-8")):
        raise ValueError("event targets differ from the frozen run specification")
    return index


def read_target(row):
    target = np.asarray(Image.open(ROOT / row["event_mask_path"])) > 0
    current_valid = np.asarray(Image.open(ROOT / row["valid_mask_path"])) > 0
    reference_valid = np.asarray(Image.open(ROOT / row["reference_valid_path"])) > 0
    if target.shape != current_valid.shape or target.shape != reference_valid.shape:
        raise ValueError("event target/observability shape mismatch")
    return target, current_valid & reference_valid


def read_features(path):
    with np.load(ROOT / path, allow_pickle=False) as saved:
        return saved["features"].copy()


def grouped_rows(index):
    groups = {}
    for row in index["observations"]:
        groups.setdefault(row["aoi_id"], []).append(row)
    return {aoi: sorted(rows, key=lambda r: r["month"]) for aoi, rows in groups.items()}


def fit_head(index):
    rows = [r for r in index["observations"] if r["role"] == "calibration" and not r["is_initial"]]
    if {r["aoi_id"] for r in rows} != set(index["spec"]["calibration_aoi_ids"]):
        raise ValueError("event training geography contract failed")
    refs, currents, targets, weights, ids = [], [], [], [], []
    sum_x, sum_x2, total_weight = np.zeros(1152, np.float64), np.zeros(1152, np.float64), 0.0
    for row in rows:
        actual_target, actual_valid = read_target(row)
        identity_valid = np.asarray(Image.open(ROOT / row["reference_valid_path"])) > 0
        for identity in [False, True]:
            mask = np.zeros_like(actual_target) if identity else actual_target
            valid = identity_valid if identity else actual_valid
            for tile in row["tiles"]:
                target, weight = soft_building_targets(mask, valid, tile["x"], tile["y"])
                keep = weight > 0
                reference = read_features(tile["reference_feature_path"])
                current = reference if identity else read_features(tile["feature_path"])
                pair = pair_features(reference, current)[keep].astype(np.float64)
                y, w = target[keep], weight[keep]
                sum_x += (pair * w[:, None]).sum(axis=0)
                sum_x2 += (pair ** 2 * w[:, None]).sum(axis=0)
                total_weight += w.sum()
                refs.append(reference[keep])
                currents.append(current[keep])
                targets.append(y)
                weights.append(w)
                ids.append({"aoi_id": row["aoi_id"], "reference_month": row["initial_month"],
                            "current_input_month": row["initial_month"] if identity else row["month"],
                            "identity_repeated_for_regular_month": row["month"] if identity else None,
                            "identity_negative": identity, "target_is_zero": identity,
                            "x": tile["x"], "y": tile["y"], "usable_token_count": int(keep.sum()),
                            "event_mask_sha256": sha(ROOT / row["event_mask_path"]) if not identity else None})
    mean64 = sum_x / total_weight
    mean = mean64.astype(np.float32)
    std = np.sqrt(np.maximum(sum_x2 / total_weight - mean64 ** 2, FIT["feature_std_floor"] ** 2)).astype(np.float32)
    ref, current = np.concatenate(refs), np.concatenate(currents)
    y, weight = np.concatenate(targets), np.concatenate(weights)
    del refs, currents, targets, weights
    prior = float(np.sum(y * weight) / weight.sum())
    positive_weight = float(min(30, max(1, np.sqrt((1 - prior) / max(prior, 1e-12)))))
    torch.manual_seed(FIT["seed"])
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = make_pair_head(torch).to(device)
    with torch.no_grad():
        head[2].bias.fill_(np.log(max(1e-9, positive_weight * prior) / max(1e-9, 1 - prior)))
    tr, tc = torch.from_numpy(ref).to(device), torch.from_numpy(current).to(device)
    ty, tw = torch.from_numpy(y).to(device), torch.from_numpy(weight).to(device)
    tm, ts = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    positive_tensor = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FIT["learning_rate"], weight_decay=FIT["weight_decay"])
    history, started = [], time.perf_counter()
    for epoch in range(FIT["epochs"]):
        permutation = torch.randperm(len(tr), device=device)
        numerator, denominator = 0.0, 0.0
        for start in range(0, len(tr), FIT["batch_size"]):
            batch = permutation[start:start+FIT["batch_size"]]
            first, second = tr[batch].float(), tc[batch].float()
            x = torch.cat([first, second, torch.abs(second - first)], dim=1)
            optimizer.zero_grad(set_to_none=True)
            logits = head((x - tm) / ts).squeeze(1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty[batch],
                            pos_weight=positive_tensor, reduction="none")
            loss = (losses * tw[batch]).sum() / tw[batch].sum()
            loss.backward()
            optimizer.step()
            numerator += float((losses.detach() * tw[batch]).sum().cpu())
            denominator += float(tw[batch].sum().cpu())
        history.append({"epoch": epoch + 1, "weighted_bce": numerator / denominator})
        print(f"event head epoch {epoch+1}/{FIT['epochs']} BCE={history[-1]['weighted_bce']:.6f}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    parameters = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    temporary = OUT / "head.tmp.npz"
    np.savez(temporary, **parameters, feature_mean=mean, feature_std=std)
    temporary.replace(OUT / "head.npz")
    save_json(OUT / "training_ids.json", {"aoi_ids": index["spec"]["calibration_aoi_ids"],
               "real_pair_count": len(rows), "identity_pair_count": len(rows), "identity_example_fraction": 0.5,
               "unique_identity_RGB_pairs": len(index["spec"]["calibration_aoi_ids"]),
               "usable_token_count": len(y), "tile_examples": ids,
               "warning": "identity self-pairs repeat first RGB inputs; tiles/months/examples are not independent units"})
    save_json(OUT / "training_history.json", {"fit": FIT, "epochs": history,
               "training_loop_checkpoint_seconds": time.perf_counter() - started,
               "positive_valid_pixel_equivalent_area": float(np.sum(y * weight) * 64),
               "negative_valid_pixel_equivalent_area": float(np.sum((1 - y) * weight) * 64),
               "mean_soft_target_including_identity": prior, "positive_weight": positive_weight,
               "normalization": "calibration examples only, including explicitly recorded identity negatives",
               "head_parameter_count": sum(p.numel() for p in head.parameters())})
    metadata = {"schema_version": "ground-new-building-v1", "frozen": False, "created_utc": now(),
                "encoder_config": GROUND_CONFIG, "backbone_identity": index["spec"]["backbone_identity"],
                "head_sha256": sha(OUT / "head.npz"), "run_spec_sha256": sha(OUT / "run_spec.json"),
                "training_ids_sha256": sha(OUT / "training_ids.json"), "threshold": 0.5,
                "calibration_aoi_ids": index["spec"]["calibration_aoi_ids"],
                "validation_aoi_ids": index["spec"]["validation_aoi_ids"], "fit": FIT,
                "positive_weight": positive_weight, "architecture": "concat(ref, current, abs difference) -> 64 GELU -> 1",
                "target": index["spec"]["target"],
                "output": "native HxW class-weighted sigmoid new-building scores, not calibrated posterior probabilities",
                "predict_inputs": "supplied receiver reference RGB and supplied receiver current RGB only"}
    save_json(OUT / "metadata.json", metadata)


def cached_prediction(row, evaluator, identity=False):
    output = np.empty(tuple(row["dimensions"][::-1]), np.float32)
    for tile in row["tiles"]:
        reference = read_features(tile["reference_feature_path"])
        current = reference if identity else read_features(tile["feature_path"])
        scores = evaluator.tile_probabilities(reference, current)
        x, y, width, height = (tile[k] for k in ["x", "y", "width", "height"])
        output[y:y+height, x:x+width] = scores[:height, :width]
    return output


def histogram_counts(labels, scores, thresholds):
    bins = np.searchsorted(thresholds, scores, side="right")
    positive = np.bincount(bins[labels], minlength=len(thresholds) + 1)
    negative = np.bincount(bins[~labels], minlength=len(thresholds) + 1)
    return positive, negative


def histogram_rates(positive, negative):
    tp = np.cumsum(positive[::-1], dtype=np.int64)[::-1][1:]
    fp = np.cumsum(negative[::-1], dtype=np.int64)[::-1][1:]
    fn = positive.sum() - tp
    return {"f1": 2 * tp / np.maximum(1, 2 * tp + fp + fn),
            "precision": tp / np.maximum(1, tp + fp), "recall": tp / max(1, positive.sum())}


def choose_threshold(index):
    evaluator = GroundChangeEvaluator(ROOT, _allow_unfrozen=True)
    thresholds = np.arange(1, 1000, dtype=np.float32) / np.float32(1000)
    curves, records = [], []
    for aoi, rows in grouped_rows(index).items():
        if rows[0]["role"] != "calibration":
            continue
        positive, negative = np.zeros(1000, np.int64), np.zeros(1000, np.int64)
        for row in rows[1:]:
            truth, valid = read_target(row)
            scores = cached_prediction(row, evaluator)[valid]
            p, n = histogram_counts(truth[valid], scores, thresholds)
            positive += p
            negative += n
        rates = histogram_rates(positive, negative)
        curves.append(rates["f1"])
        records.append({"aoi_id": aoi, "real_pair_count": len(rows) - 1,
                        "event_positive_pixels": int(positive.sum()),
                        "valid_pixels": int(positive.sum() + negative.sum())})
    objective = np.mean(curves, axis=0)
    best = np.flatnonzero(objective == objective.max())[-1]
    selection = {"selected_utc": now(), "threshold": float(thresholds[best]),
                 "calibration_macro_f1": float(objective[best]), "f1_per_aoi": [float(c[best]) for c in curves],
                 "aois": records, "candidate_count": len(thresholds), "objective": FIT["threshold_objective"],
                 "validation_labels_or_scores_used": False, "identity_augmented_pairs_used_for_threshold": False}
    save_json(OUT / "calibration_selection.json", selection)
    np.savez(OUT / "calibration_threshold_curve.npz", thresholds=thresholds,
             f1_per_aoi=np.stack(curves), macro_f1=objective)
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(frozen=True, frozen_utc=now(), threshold=selection["threshold"],
                    calibration_selection_sha256=sha(OUT / "calibration_selection.json"))
    save_json(OUT / "metadata.json", metadata)
    print("fixed event threshold", metadata["threshold"], "calibration macro F1", selection["calibration_macro_f1"], flush=True)


def fit():
    index = load_index()
    path = OUT / "metadata.json"
    if path.exists():
        fixed = json.loads(path.read_text(encoding="utf-8"))
        if fixed.get("frozen"):
            if fixed["run_spec_sha256"] != sha(OUT / "run_spec.json"):
                raise ValueError("frozen event run specification changed")
            print("retaining already frozen event head and threshold", flush=True)
            return
    fit_head(index)
    choose_threshold(index)


def event_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tp, fp = int(np.sum(labels & predicted)), int(np.sum(~labels & predicted))
    fn, tn = int(np.sum(labels & ~predicted)), int(np.sum(~labels & ~predicted))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "ap": exact_average_precision(labels, scores), "valid_pixel_count": len(labels),
            "positive_pixels": int(labels.sum()), "positive_fraction": float(labels.mean()),
            "predicted_positive_pixels": int(predicted.sum())}


def report():
    index = load_index()
    evaluator = GroundChangeEvaluator(ROOT)
    conditions = ["full_current", "first_month_only", "deleted_all_updates"]
    thresholds = np.arange(1, 1000, dtype=np.float32) / np.float32(1000)
    output = {"utc": now(), "metadata_sha256": sha(OUT / "metadata.json"), "threshold": evaluator.threshold,
              "statistical_unit": "AOI", "scope": "original four-AOI development evidence; expanded AOIs untouched",
              "target": index["spec"]["target"],
              "metric_level": "native pixels of current new-track-ID footprint union, not instance-detection AP",
              "comparison": "same current GT and current/first valid intersection; initial month excluded",
              "conditions": {"full_current": "actual initial RGB, actual current RGB",
                             "first_month_only": "actual initial RGB, actual initial RGB",
                             "deleted_all_updates": "actual initial RGB, initial RGB copy after omitting every current update"},
              "ap_definition": "exact noninterpolated pixel AP with score ties grouped",
              "per_aoi": [], "by_role": {}}
    pooled = {role: {"labels": [], **{c: [] for c in conditions}} for role in ["calibration", "validation"]}
    pr_arrays = {"thresholds": thresholds}
    for aoi, rows in grouped_rows(index).items():
        old = cached_prediction(rows[0], evaluator)
        deleted = cached_prediction(rows[0], evaluator, identity=True)
        if not np.array_equal(old, deleted):
            raise ValueError("with every update omitted, receiver content should be identical to the initial RGB")
        truths, scores = [], {c: [] for c in conditions}
        for row in rows[1:]:
            target, valid = read_target(row)
            current = cached_prediction(row, evaluator)
            truths.append(target[valid])
            scores["full_current"].append(current[valid])
            scores["first_month_only"].append(old[valid])
            scores["deleted_all_updates"].append(deleted[valid])
        labels = np.concatenate(truths)
        all_scores = {condition: np.concatenate(values) for condition, values in scores.items()}
        item = {"aoi_id": aoi, "role": rows[0]["role"], "initial_month": rows[0]["month"],
                "evaluated_months": [row["month"] for row in rows[1:]], "old_and_deleted_scores_identical": True}
        for condition in conditions:
            # Identical receiver inputs must receive identical metrics; do not alter GT.
            item[condition] = dict(item["first_month_only"]) if condition == "deleted_all_updates" else event_metrics(labels, all_scores[condition], evaluator.threshold)
            p, n = histogram_counts(labels, all_scores[condition], thresholds)
            rates = histogram_rates(p, n)
            for metric in ["precision", "recall"]:
                pr_arrays[f"{aoi}__{condition}__{metric}"] = rates[metric]
        output["per_aoi"].append(item)
        role = rows[0]["role"]
        pooled[role]["labels"].append(labels)
        for condition in conditions:
            pooled[role][condition].append(all_scores[condition])
        print(f"event report {role} {aoi}: full F1={item['full_current']['f1']:.4f}, old/deleted F1={item['first_month_only']['f1']:.4f}", flush=True)
    for role, arrays in pooled.items():
        labels = np.concatenate(arrays["labels"])
        role_result = {"aoi_count": len(arrays["labels"]), "pooled": {}, "macro_aoi": {}}
        for condition in conditions:
            if condition == "deleted_all_updates":
                role_result["pooled"][condition] = dict(role_result["pooled"]["first_month_only"])
            else:
                role_result["pooled"][condition] = event_metrics(labels, np.concatenate(arrays[condition]), evaluator.threshold)
            members = [r[condition] for r in output["per_aoi"] if r["role"] == role]
            role_result["macro_aoi"][condition] = {k: float(np.mean([member[k] for member in members]))
                                                   for k in ["f1", "iou", "precision", "recall", "ap"]}
        output["by_role"][role] = role_result
    np.savez(OUT / "precision_recall_curves.npz", **pr_arrays)
    # Run the public API on actual distinct RGBs and on the actual old RGB twice.
    smoke = next(r for r in index["observations"] if r["role"] == "validation" and not r["is_initial"])
    reference = Image.open(ROOT / smoke["reference_rgb_path"]).convert("RGB")
    current = Image.open(ROOT / smoke["rgb_path"]).convert("RGB")
    expected = cached_prediction(smoke, evaluator)
    direct = evaluator.predict(reference, current)
    full_timing = evaluator.last_timing
    identity = evaluator.predict(reference, reference)
    identity_timing = evaluator.last_timing
    deleted = evaluator.predict(reference, reference.copy())
    deleted_timing = evaluator.last_timing
    cached_identity = cached_prediction(smoke, evaluator, identity=True)
    difference = float(np.max(np.abs(direct - expected)))
    identity_difference = float(np.max(np.abs(identity - cached_identity)))
    if difference > 2e-5 or identity_difference > 2e-5 or not np.array_equal(identity, deleted):
        raise ValueError("two-image RGB API, cache replay, or deleted-update identity condition disagrees")
    output["rgb_api_check"] = {"aoi_id": smoke["aoi_id"], "current_month": smoke["month"],
                                "reference_month": smoke["initial_month"], "native_shape": list(direct.shape),
                                "full_max_abs_cache_difference": difference,
                                "identity_max_abs_cache_difference": identity_difference,
                                "identity_and_deleted_exactly_equal": True,
                                "sender_current_features_or_labels_entered_predict": False}
    save_json(OUT / "timing.json", {"first_full_predict": full_timing, "warm_identity_predict": identity_timing,
               "warm_deleted_predict": deleted_timing, "encoder": evaluator.encoder.metadata,
               "scope": "receiver ground RGB pair evaluator; no cached features in public API timing"})
    save_json(OUT / "environment.json", {"utc": now(), "python": sys.version, "torch": torch.__version__,
               "cuda": torch.version.cuda, "numpy": np.__version__, "device": evaluator.device,
               "gpu": torch.cuda.get_device_name(0) if evaluator.device == "cuda" else None,
               "feature_extraction_provenance": "resources/ground_eval/environment.json",
               "target_preparation_provenance": "resources/ground_change_eval/target_provenance.json",
               "argv": sys.argv})
    save_json(OUT / "results.json", output)
    print(json.dumps(output["by_role"], ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "fit", "report"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare()
    elif args.phase == "fit":
        fit()
    else:
        report()


if __name__ == "__main__":
    main()
