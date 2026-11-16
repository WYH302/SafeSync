"""Fit/freeze a small receiver RGB building head on two fixed SN7 AOIs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, __version__ as pillow_version

from safesync.features import FrozenEncoder
from safesync.ground import (GROUND_CONFIG, GroundEvaluator, cached_ground_features,
                             make_ground_head, rgb_tiles, soft_building_targets)

OUT = ROOT / "resources/ground_eval"
FIT = {"seed": 20261028, "epochs": 12, "batch_size": 16384, "learning_rate": 0.002,
       "weight_decay": 0.001, "feature_std_floor": 0.001, "loss": "valid-area weighted soft-target BCE",
       "threshold_candidates": "0.001..0.999 in increments of 0.001, float32",
       "threshold_objective": "macro F1 across the two calibration AOIs on current RGB after their initial month",
       "threshold_tie_rule": "highest threshold among exact macro-F1 ties",
       "model_selection": "one fixed MLP recipe; no validation-driven choices"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare():
    manifest_path = OUT / "dataset_manifest.json"
    if not manifest_path.exists():
        source_path = ROOT / "data/spacenet7_pilot/development_manifest_v1.json"
        if not source_path.exists():
            source_path = ROOT / "data/spacenet7_pilot/manifest.json"
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
        if len(source_data["aois"]) != 4:
            raise ValueError("original four-AOI manifest required; expanded AOIs are untouched confirmation data")
        save_json(manifest_path, source_data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(manifest["aois"]) != 4:
        raise ValueError("the fixed experiment requires the complete four-AOI pilot")
    encoder = FrozenEncoder(ROOT, GROUND_CONFIG)
    spec = {"manifest_sha256": sha(manifest_path), "encoder_config": GROUND_CONFIG,
            "backbone_identity": json.loads(encoder.identity), "fit": FIT,
            "calibration_aoi_ids": [a["aoi_id"] for a in manifest["aois"][:2]],
            "validation_aoi_ids": [a["aoi_id"] for a in manifest["aois"][2:]],
            "update_comparison": "exclude initial observation; current-label metric on current AND first-month valid pixels",
            "statistical_unit": "AOI, not tile or month", "claim_scope": "four-AOI development evidence"}
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("run specification differs from the previously frozen experiment")
    save_json(spec_path, spec)
    resource = {"utc": now(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "Pillow": pillow_version, "numpy": np.__version__, "python": sys.version,
                "logical_cpus": os.cpu_count(), "disk_free_bytes": shutil.disk_usage(ROOT).free,
                "gpu_snapshot": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip(),
                "encoder": encoder.metadata, "argv": sys.argv}
    save_json(OUT / "environment.json", resource)
    index = {"spec": spec, "observations": []}
    for aoi_index, aoi in enumerate(manifest["aois"]):
        role = "calibration" if aoi_index < 2 else "validation"
        for month in aoi["months"]:
            if month["status"] != "ready":
                continue
            image = Image.open(ROOT / month["rgb_path"]).convert("RGB")
            row = {key: month[key] for key in ["month", "rgb_path", "building_mask_path", "valid_mask_path", "dimensions", "derived_sha256"]}
            row.update(aoi_id=aoi["aoi_id"], role=role, tiles=[])
            for tile in rgb_tiles(image):
                values, timings, cache_path = cached_ground_features(ROOT, encoder, tile["image"])
                if values.shape != (4096, 384):
                    raise ValueError("unexpected ground feature shape")
                row["tiles"].append({**{key: tile[key] for key in ["x", "y", "width", "height"]},
                                     "feature_path": cache_path.relative_to(ROOT).as_posix(), "timings": timings})
            index["observations"].append(row)
            save_json(OUT / "cache_index.partial.json", index)
            print(f"features {len(index['observations'])}/96 {role} {aoi['aoi_id']} {month['month']}", flush=True)
    save_json(OUT / "cache_index.json", index)
    return encoder


def load_index():
    index = json.loads((OUT / "cache_index.json").read_text(encoding="utf-8"))
    if index["spec"] != json.loads((OUT / "run_spec.json").read_text(encoding="utf-8")):
        raise ValueError("feature index differs from the fixed run specification")
    return index


def read_masks(row):
    building = np.asarray(Image.open(ROOT / row["building_mask_path"])) > 0
    valid = np.asarray(Image.open(ROOT / row["valid_mask_path"])) > 0
    if building.shape != valid.shape or tuple(row["dimensions"][::-1]) != valid.shape:
        raise ValueError("ground image/label/mask shape mismatch")
    return building, valid


def fit_head(index):
    rows = [row for row in index["observations"] if row["role"] == "calibration"]
    allowed = set(index["spec"]["calibration_aoi_ids"])
    if {row["aoi_id"] for row in rows} != allowed:
        raise ValueError("calibration geography contract failed")
    values, targets, weights, ids = [], [], [], []
    sum_x, sum_x2, total_weight = np.zeros(384, np.float64), np.zeros(384, np.float64), 0.0
    for row in rows:
        building, valid = read_masks(row)
        for tile in row["tiles"]:
            target, weight = soft_building_targets(building, valid, tile["x"], tile["y"])
            keep = weight > 0
            with np.load(ROOT / tile["feature_path"], allow_pickle=False) as stored:
                x = stored["features"][keep].copy()
            y, w = target[keep], weight[keep]
            x64 = x.astype(np.float64)
            sum_x += (x64 * w[:, None]).sum(axis=0)
            sum_x2 += (x64 ** 2 * w[:, None]).sum(axis=0)
            total_weight += w.sum()
            values.append(x)
            targets.append(y)
            weights.append(w)
            ids.append({"aoi_id": row["aoi_id"], "month": row["month"],
                        "x": tile["x"], "y": tile["y"], "width": tile["width"], "height": tile["height"],
                        "rgb_sha256": row["derived_sha256"]["rgb_path"],
                        "building_mask_sha256": row["derived_sha256"]["building_mask_path"],
                        "valid_mask_sha256": row["derived_sha256"]["valid_mask_path"],
                        "usable_token_count": int(keep.sum())})
    mean = (sum_x / total_weight).astype(np.float32)
    std = np.sqrt(np.maximum(sum_x2 / total_weight - mean.astype(np.float64) ** 2,
                             FIT["feature_std_floor"] ** 2)).astype(np.float32)
    x = np.concatenate(values)
    y, weight = np.concatenate(targets), np.concatenate(weights)
    del values, targets, weights
    torch.manual_seed(FIT["seed"])
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = make_ground_head(torch).to(device)
    prior = float(np.sum(y * weight) / weight.sum())
    with torch.no_grad():
        head[2].bias.fill_(np.log(prior / (1 - prior)))
    tx = torch.from_numpy(x).to(device)
    ty, tw = torch.from_numpy(y).to(device), torch.from_numpy(weight).to(device)
    tm, ts = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FIT["learning_rate"], weight_decay=FIT["weight_decay"])
    history = []
    started = time.perf_counter()
    for epoch in range(FIT["epochs"]):
        permutation = torch.randperm(len(tx), device=device)
        summed_loss, denominator = 0.0, 0.0
        for start in range(0, len(tx), FIT["batch_size"]):
            batch = permutation[start:start+FIT["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = head((tx[batch].float() - tm) / ts).squeeze(1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty[batch], reduction="none")
            loss = (losses * tw[batch]).sum() / tw[batch].sum()
            loss.backward()
            optimizer.step()
            summed_loss += float((losses.detach() * tw[batch]).sum().cpu())
            denominator += float(tw[batch].sum().cpu())
        history.append({"epoch": epoch + 1, "valid_area_weighted_bce": summed_loss / denominator})
        print(f"head epoch {epoch+1}/{FIT['epochs']} BCE={history[-1]['valid_area_weighted_bce']:.6f}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    parameters = {key: value.detach().cpu().numpy() for key, value in head.state_dict().items()}
    temporary = OUT / "head.tmp.npz"
    np.savez(temporary, **parameters, feature_mean=mean, feature_std=std)
    temporary.replace(OUT / "head.npz")
    save_json(OUT / "training_ids.json", {"aoi_ids": sorted(allowed), "observation_count": len(rows),
               "tile_ids": ids, "usable_token_count": len(x), "statistical_unit": "AOI"})
    save_json(OUT / "training_history.json", {"fit": FIT, "epochs": history,
               "training_seconds": time.perf_counter() - started, "mean_soft_target": prior,
               "normalization": "calibration only, valid-area weighted mean/std of cached frozen tokens",
               "head_parameter_count": sum(p.numel() for p in head.parameters())})
    metadata = {"schema_version": "ground-building-v1", "frozen": False, "created_utc": now(),
                "encoder_config": GROUND_CONFIG, "backbone_identity": index["spec"]["backbone_identity"],
                "head_sha256": sha(OUT / "head.npz"), "run_spec_sha256": sha(OUT / "run_spec.json"),
                "training_ids_sha256": sha(OUT / "training_ids.json"), "threshold": 0.5,
                "architecture": "384 -> 64 GELU -> 1; 24,705 trainable head parameters; backbone frozen",
                "calibration_aoi_ids": index["spec"]["calibration_aoi_ids"],
                "validation_aoi_ids": index["spec"]["validation_aoi_ids"], "fit": FIT,
                "feature_precision": "FP32 extraction, explicit FP16 round trip in both cache and online prediction",
                "output": "native HxW float32 probability; 64x64 token probabilities bilinearly expanded per native 512 tile",
                "predict_inputs": "provided RGB image, frozen weights/head/normalization only; no validity or labels"}
    save_json(OUT / "metadata.json", metadata)
    return metadata


def cached_prediction(row, evaluator):
    output = np.empty(tuple(row["dimensions"][::-1]), np.float32)
    for tile in row["tiles"]:
        with np.load(ROOT / tile["feature_path"], allow_pickle=False) as stored:
            probabilities = evaluator.tile_probabilities(stored["features"])
        x, y, width, height = (tile[k] for k in ["x", "y", "width", "height"])
        output[y:y+height, x:x+width] = probabilities[:height, :width]
    return output


def grouped_rows(index):
    groups = {}
    for row in index["observations"]:
        groups.setdefault(row["aoi_id"], []).append(row)
    return {key: sorted(rows, key=lambda r: r["month"]) for key, rows in groups.items()}


def choose_threshold(index):
    evaluator = GroundEvaluator(ROOT, _allow_unfrozen=True)
    thresholds = np.arange(1, 1000, dtype=np.float32) / np.float32(1000)
    curves, records = [], []
    for aoi, rows in grouped_rows(index).items():
        if rows[0]["role"] != "calibration":
            continue
        positive, negative = np.zeros(1000, np.int64), np.zeros(1000, np.int64)
        _, first_valid = read_masks(rows[0])
        for row in rows[1:]:
            truth, valid = read_masks(row)
            valid &= first_valid
            probabilities = cached_prediction(row, evaluator)[valid]
            labels = truth[valid]
            bins = np.searchsorted(thresholds, probabilities, side="right")
            positive += np.bincount(bins[labels], minlength=1000)
            negative += np.bincount(bins[~labels], minlength=1000)
        tp = np.cumsum(positive[::-1], dtype=np.int64)[::-1][1:]
        fp = np.cumsum(negative[::-1], dtype=np.int64)[::-1][1:]
        fn = positive.sum() - tp
        f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        curves.append(f1)
        records.append({"aoi_id": aoi, "observation_count": len(rows) - 1,
                        "valid_pixel_count": int(positive.sum() + negative.sum()), "positive_pixels": int(positive.sum())})
    objective = np.mean(curves, axis=0)
    best = np.flatnonzero(objective == objective.max())[-1]
    selection = {"selected_utc": now(), "threshold": float(thresholds[best]),
                 "calibration_macro_f1": float(objective[best]), "f1_per_aoi": [float(c[best]) for c in curves],
                 "aois": records, "objective": FIT["threshold_objective"],
                 "candidate_count": len(thresholds), "tie_rule": FIT["threshold_tie_rule"],
                 "validation_labels_or_scores_used": False}
    save_json(OUT / "calibration_selection.json", selection)
    np.savez(OUT / "calibration_threshold_curve.npz", thresholds=thresholds,
             f1_per_aoi=np.stack(curves), macro_f1=objective)
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    metadata.update(threshold=selection["threshold"], frozen=True, frozen_utc=now(),
                    calibration_selection_sha256=sha(OUT / "calibration_selection.json"))
    save_json(OUT / "metadata.json", metadata)
    print("fixed calibration threshold", selection["threshold"], "macro F1", selection["calibration_macro_f1"], flush=True)
    return metadata


def fit():
    index = load_index()
    path = OUT / "metadata.json"
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("frozen"):
            if metadata["run_spec_sha256"] != sha(OUT / "run_spec.json"):
                raise ValueError("fixed evaluator run spec changed")
            print("retaining already frozen head and threshold", flush=True)
            return metadata
    fit_head(index)
    return choose_threshold(index)


def exact_average_precision(labels, scores):
    """Noninterpolated AP, with all equal-score pixels handled as one group."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float32)
    positive = int(labels.sum())
    if not positive:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    tp = np.cumsum(labels[order], dtype=np.int64)
    endpoints = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), len(scores) - 1]
    group_tp = tp[endpoints]
    increments = np.diff(np.r_[0, group_tp])
    return float(np.sum(increments.astype(np.float64) * group_tp / (endpoints + 1)) / positive)


def pixel_metrics(labels, scores, threshold):
    prediction = scores >= threshold
    tp = int(np.sum(labels & prediction))
    fp = int(np.sum(~labels & prediction))
    fn = int(np.sum(labels & ~prediction))
    tn = int(np.sum(~labels & ~prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "ap": exact_average_precision(labels, scores), "valid_pixel_count": len(labels),
            "positive_fraction": float(labels.mean())}


def report():
    index = load_index()
    evaluator = GroundEvaluator(ROOT)
    output = {"utc": now(), "metadata_sha256": sha(OUT / "metadata.json"),
              "threshold": evaluator.threshold, "statistical_unit": "AOI",
              "independent_calibration_aois": 2, "independent_validation_aois": 2,
              "scope": "four-AOI development evidence; no confidence interval from correlated tiles/months",
              "comparison": "current RGB versus only first-month RGB; initial month excluded; common current/initial valid mask",
              "ap_definition": "exact noninterpolated pixel average precision with grouped score ties",
              "per_aoi": [], "by_role": {}}
    pooled = {role: {"labels": [], "full_current": [], "first_month_only": []} for role in ["calibration", "validation"]}
    for aoi, rows in grouped_rows(index).items():
        first_prediction = cached_prediction(rows[0], evaluator)
        _, initial_valid = read_masks(rows[0])
        labels_all, current_all, old_all = [], [], []
        for row in rows[1:]:
            truth, valid = read_masks(row)
            valid &= initial_valid
            current = cached_prediction(row, evaluator)
            labels_all.append(truth[valid])
            current_all.append(current[valid])
            old_all.append(first_prediction[valid])
        labels, current, old = np.concatenate(labels_all), np.concatenate(current_all), np.concatenate(old_all)
        item = {"aoi_id": aoi, "role": rows[0]["role"], "initial_month": rows[0]["month"],
                "evaluated_months": [r["month"] for r in rows[1:]],
                "full_current": pixel_metrics(labels, current, evaluator.threshold),
                "first_month_only": pixel_metrics(labels, old, evaluator.threshold)}
        output["per_aoi"].append(item)
        role = rows[0]["role"]
        pooled[role]["labels"].append(labels)
        pooled[role]["full_current"].append(current)
        pooled[role]["first_month_only"].append(old)
        print(f"report {role} {aoi}: full F1={item['full_current']['f1']:.4f}, old F1={item['first_month_only']['f1']:.4f}", flush=True)
    for role, arrays in pooled.items():
        labels = np.concatenate(arrays["labels"])
        by_role = {"aoi_count": len(arrays["labels"]), "pooled": {}, "macro_aoi": {}}
        for condition in ["full_current", "first_month_only"]:
            scores = np.concatenate(arrays[condition])
            by_role["pooled"][condition] = pixel_metrics(labels, scores, evaluator.threshold)
            members = [r[condition] for r in output["per_aoi"] if r["role"] == role]
            by_role["macro_aoi"][condition] = {metric: float(np.mean([r[metric] for r in members])) for metric in ["f1", "iou", "ap"]}
        output["by_role"][role] = by_role
    # Direct actual-RGB execution verifies that cache replay and public API agree.
    smoke_row = next(r for r in index["observations"] if r["role"] == "validation")
    image = Image.open(ROOT / smoke_row["rgb_path"]).convert("RGB")
    cached = cached_prediction(smoke_row, evaluator)
    direct = evaluator.predict(image)
    difference = float(np.max(np.abs(cached - direct)))
    cold_timing = evaluator.last_timing
    evaluator.predict(image)
    warm_timing = evaluator.last_timing
    if direct.shape != (image.height, image.width) or difference > 2e-5:
        raise ValueError(f"RGB prediction differs from cached replay: max {difference}")
    output["rgb_api_check"] = {"aoi_id": smoke_row["aoi_id"], "month": smoke_row["month"],
                                "native_shape": list(direct.shape), "max_abs_cache_difference": difference,
                                "only_supplied_RGB_used": True}
    save_json(OUT / "timing.json", {"cold_predict": cold_timing, "warm_predict": warm_timing,
               "encoder_metadata": evaluator.encoder.metadata,
               "cached_feature_ms_per_tile": {key: float(np.mean([t["timings"][key] for r in index["observations"] for t in r["tiles"]]))
                                              for key in ["preprocessing_ms", "backbone_ms", "transfer_pool_normalize_ms", "total_ms"]},
               "timing_scope": "ground evaluator; includes preprocessing, backbone, head, transfer, tiling; not satellite router runtime"})
    save_json(OUT / "results.json", output)
    print(json.dumps(output["by_role"], ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "fit", "report", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("prepare", "all"):
        prepare()
    if args.phase in ("fit", "all"):
        fit()
    if args.phase in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
