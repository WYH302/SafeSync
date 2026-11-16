"""Fit one predetermined SECOND ground task recipe, then freeze and report once."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from safesync.data import label_array, verified_path
from safesync.second_ground import (ENCODER_CONFIG, SecondGroundEvaluator,
                                    make_head, pair_features, sha_file)

OUT = ROOT / "resources/second_ground_eval"
FIT = {"seed": 20261028, "epochs": 12, "batch_size": 4096,
       "learning_rate": 0.002, "weight_decay": 0.001, "feature_std_floor": 0.001,
       "architecture": "concat(before,current,abs(current-before)) 1152 -> 64 GELU -> 1",
       "optimizer": "AdamW; PyTorch defaults except specified lr and weight_decay",
       "loss": "sum(valid_pixel_area * soft_BCEWithLogits) / sum(valid_pixel_area)",
       "positive_weight": "min(10,sqrt(real calibration valid negative pixels / positive pixels)); excludes identity augmentation",
       "identity_negative": "one before/before target-zero example per calibration pair, same strict eligible tokens and valid weights",
       "normalization": "valid-area-weighted mean/std on calibration real+identity examples only",
       "threshold_candidates": "float32 0.001,0.002,...,0.999; prediction score >= threshold",
       "threshold_objective": "pooled pixel micro F1 on original calibration pairs; no identity examples",
       "threshold_tie_rule": "highest threshold among exactly equal micro F1 values",
       "selection": "last epoch of one fixed 12-epoch recipe; no validation selection or second recipe"}
RULES = {"binary_target": "GT_CD 0=no change, 255=change; 255 is not binary ignore",
         "semantic_zero": "semantic unlabelled background; its GT_CD=0 is a legitimate binary negative, not a known land-cover class",
         "pixel_valid": "neither time has heldout semantic class or semantic ignore=255",
         "training_token": "exclude entire 32x32 token if either time has any heldout pixel; ignore pixels have zero area weight",
         "soft_target": "valid changed pixel count / valid pixel count per eligible native 32x32 cell",
         "scoring": "native pixels satisfying pixel_valid, including unchanged unlabelled-semantic background; no GT enters predict",
         "scope": "fixed 128 calibration/64 validation original train pairs; geographic independence not established"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def dataset():
    path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cache_path = ROOT / "cache/second_scaled/features.npz"
    metadata_path = ROOT / "cache/second_scaled/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not manifest["complete"] or metadata["manifest_sha256"] != sha_file(path):
        raise ValueError("complete fixed dataset/cache manifest identity required")
    pairs = manifest["pairs"]
    if len(pairs) != 192 or len({p["group_id"] for p in pairs}) != 192:
        raise ValueError("requires 192 unique original image-pair groups")
    if any(p["source_split"] != "train" for p in pairs):
        raise ValueError("official test must not enter this ground evaluator")
    if [p["split"] for p in pairs] != ["calibration"] * 128 + ["validation"] * 64:
        raise ValueError("requires the existing fixed 128/64 pair split, before gridding")
    with np.load(cache_path, allow_pickle=False) as stored:
        features, ids = stored["features"], stored["ids"].tolist()
    if ids != [p["id"] for p in pairs] or features.shape != (192, 2, 256, 384):
        raise ValueError("frozen features are not aligned with fixed original pairs")
    identity = metadata["encoder"]["identity"]
    if (identity["input"], identity["pool"], identity["name"]) != (448, 2, "dinov2_vits14"):
        raise ValueError("unexpected existing backbone/preprocessing")
    spec = {"schema": "second-ground-run-spec-v1", "fit": FIT, "rules": RULES,
            "manifest_sha256": sha_file(path), "feature_cache_sha256": sha_file(cache_path),
            "feature_metadata_sha256": sha_file(metadata_path), "backbone_identity": identity,
            "encoder_config": ENCODER_CONFIG,
            "calibration_pair_ids": ids[:128], "validation_pair_ids": ids[128:],
            "official_test_pixels_read_by_this_script": 0,
            "software": {"python": sys.version, "torch": torch.__version__,
                         "cuda": torch.version.cuda, "numpy": np.__version__}}
    return pairs, features, spec


def read_target(row, heldout):
    first = label_array(verified_path(ROOT, row, "label_before"))
    second = label_array(verified_path(ROOT, row, "label_after"))
    with Image.open(verified_path(ROOT, row, "binary_change")) as image:
        binary = np.asarray(image)
    if first.shape != (512, 512) or second.shape != first.shape or binary.shape != first.shape:
        raise ValueError("native SECOND target dimensions changed")
    if not set(np.unique(binary)).issubset({0, 255}):
        raise ValueError("expected this mirror's GT_CD encoding 0/255")
    unknown = (first == heldout) | (second == heldout)
    valid = ~unknown & (first != 255) & (second != 255)
    return binary == 255, valid, unknown


def block_sum(values):
    return values.reshape(16, 32, 16, 32).sum(axis=(1, 3)).reshape(-1)


def training_arrays(rows, features, heldout):
    xs, ys, ws, records = [], [], [], []
    positive = negative = 0
    for row, pair in zip(rows, features, strict=True):
        truth, valid, unknown = read_target(row, heldout)
        counts = block_sum(valid).astype(np.float32)
        changed = block_sum(truth & valid).astype(np.float32)
        keep = (block_sum(unknown) == 0) & (counts > 0)
        target = changed[keep] / counts[keep]
        x = pair_features(pair[0], pair[1])[keep]
        identity = pair_features(pair[0], pair[0])[keep]
        xs.extend([x, identity])
        ys.extend([target, np.zeros_like(target)])
        ws.extend([counts[keep], counts[keep]])
        positive += int((truth & valid).sum())
        negative += int((~truth & valid).sum())
        records.append({"pair_id": row["id"], "group_id": row["group_id"],
                        "eligible_real_tokens": int(keep.sum()), "identity_tokens": int(keep.sum()),
                        "excluded_heldout_tokens": int((block_sum(unknown) > 0).sum()),
                        "valid_pixels": int(valid.sum()), "changed_valid_pixels": int((truth & valid).sum()),
                        "fit_valid_token_area": int(counts[keep].sum()),
                        "fit_changed_token_area": int(changed[keep].sum()),
                        "source_files": row["files"]})
    if positive == 0 or negative == 0:
        raise ValueError("calibration lacks positive or negative binary task support")
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), records,
            min(10.0, float(np.sqrt(negative / positive))))


def weighted_normalization(x, weights):
    denominator = float(weights.sum(dtype=np.float64))
    first, second = np.zeros(x.shape[1], np.float64), np.zeros(x.shape[1], np.float64)
    for start in range(0, len(x), 4096):
        values = x[start:start + 4096].astype(np.float64)
        area = weights[start:start + 4096, None].astype(np.float64)
        first += (values * area).sum(axis=0)
        second += (values ** 2 * area).sum(axis=0)
    mean64 = first / denominator
    std = np.sqrt(np.maximum(second / denominator - mean64 ** 2,
                             FIT["feature_std_floor"] ** 2)).astype(np.float32)
    return mean64.astype(np.float32), std


def predict_head(head, mean, std, pair):
    with torch.inference_mode():
        x = torch.from_numpy(pair_features(pair[0], pair[1])).to(mean.device)
        probability = torch.sigmoid(head((x - mean) / std)).reshape(1, 1, 16, 16)
        output = torch.nn.functional.interpolate(probability, size=(512, 512),
                                                 mode="bilinear", align_corners=False)
    return output[0, 0].cpu().numpy().astype(np.float32)


def threshold_curve(rows, features, heldout, head, mean, std):
    thresholds = np.arange(1, 1000, dtype=np.float32) / np.float32(1000)
    positive, negative = np.zeros(1000, np.int64), np.zeros(1000, np.int64)
    for row, pair in zip(rows, features, strict=True):
        truth, valid, _ = read_target(row, heldout)
        score = predict_head(head, mean, std, pair)[valid]
        bins = np.searchsorted(thresholds, score, side="right")
        positive += np.bincount(bins[truth[valid]], minlength=1000)
        negative += np.bincount(bins[~truth[valid]], minlength=1000)
    tp = np.cumsum(positive[::-1], dtype=np.int64)[::-1][1:]
    fp = np.cumsum(negative[::-1], dtype=np.int64)[::-1][1:]
    fn = positive.sum() - tp
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    selected = int(np.flatnonzero(f1 == f1.max())[-1])
    return thresholds, tp, fp, fn, f1, selected


def fit_fold(pairs, features, spec, heldout):
    folder = OUT / f"heldout_{heldout}"
    folder.mkdir(parents=True, exist_ok=True)
    fold_spec = {**spec, "heldout": heldout}
    spec_path, metadata_path = folder / "run_spec.json", folder / "metadata.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != fold_spec:
        raise ValueError("refusing to overwrite a different fixed fold recipe/data identity")
    save_json(spec_path, fold_spec)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("frozen"):
            if metadata["head_sha256"] != sha_file(folder / "head.npz"):
                raise ValueError("existing frozen SECOND head checksum changed")
            print(f"heldout={heldout}: retaining frozen head and threshold", flush=True)
            return
    started = time.perf_counter()
    x, y, weights, records, positive_weight = training_arrays(pairs[:128], features[:128], heldout)
    mean, std = weighted_normalization(x, weights)
    torch.manual_seed(FIT["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = make_head(torch).to(device)
    tx, ty, tw = (torch.from_numpy(a).to(device) for a in (x, y, weights))
    tm, ts = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    pw = torch.tensor(positive_weight, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FIT["learning_rate"],
                                 weight_decay=FIT["weight_decay"])
    history = []
    for epoch in range(FIT["epochs"]):
        permutation = torch.randperm(len(tx), device=device)
        numerator, denominator = 0.0, 0.0
        for start in range(0, len(tx), FIT["batch_size"]):
            batch = permutation[start:start + FIT["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = head((tx[batch] - tm) / ts).squeeze(1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, ty[batch], reduction="none", pos_weight=pw)
            loss = (losses * tw[batch]).sum() / tw[batch].sum()
            loss.backward()
            optimizer.step()
            numerator += float((losses.detach() * tw[batch]).sum().cpu())
            denominator += float(tw[batch].sum().cpu())
        history.append({"epoch": epoch + 1, "valid_area_weighted_bce": numerator / denominator})
        print(f"heldout={heldout} epoch {epoch + 1}/12 BCE={numerator / denominator:.6f}", flush=True)
    head.eval().requires_grad_(False)
    parameters = {key: value.cpu().numpy() for key, value in head.state_dict().items()}
    temporary = folder / "head.tmp.npz"
    np.savez(temporary, **parameters, feature_mean=mean, feature_std=std)
    temporary.replace(folder / "head.npz")
    thresholds, tp, fp, fn, f1, selected = threshold_curve(
        pairs[:128], features[:128], heldout, head, tm, ts)
    np.savez(folder / "calibration_threshold_curve.npz", thresholds=thresholds,
             tp=tp, fp=fp, fn=fn, micro_f1=f1)
    save_json(folder / "training_ids.json", {"fit_pair_ids": spec["calibration_pair_ids"],
               "validation_pair_ids_used_for_fit": [], "pairs": records,
               "real_token_count": len(x) // 2, "identity_token_count": len(x) // 2})
    save_json(folder / "training_history.json", {"fit": FIT, "epochs": history,
               "positive_weight": positive_weight, "valid_area_weighted_target_after_identity":
               float(np.sum(y.astype(np.float64) * weights) / weights.sum(dtype=np.float64)),
               "training_and_calibration_seconds": time.perf_counter() - started,
               "head_parameter_count": sum(p.numel() for p in head.parameters())})
    metadata = {"schema": "second-ground-evaluator-v1", "heldout": heldout, "frozen": True,
                "frozen_utc": now(), "threshold": float(thresholds[selected]),
                "threshold_calibration_micro_f1": float(f1[selected]),
                "threshold_tie_rule": FIT["threshold_tie_rule"], "fit": FIT, "rules": RULES,
                "encoder_config": ENCODER_CONFIG, "backbone_identity": spec["backbone_identity"],
                "head_sha256": sha_file(folder / "head.npz"), "run_spec_sha256": sha_file(spec_path),
                "training_ids_sha256": sha_file(folder / "training_ids.json"),
                "manifest_sha256": spec["manifest_sha256"],
                "feature_cache_sha256": spec["feature_cache_sha256"],
                "feature_metadata_sha256": spec["feature_metadata_sha256"],
                "fit_pair_ids": spec["calibration_pair_ids"], "fit_pair_count": 128,
                "validation_pair_ids_used_for_fit_or_threshold": [],
                "positive_weight": positive_weight, "head_parameter_count": 73857,
                "output": "float32[512,512] weighted sigmoid task score; bilinear 16x16 expansion",
                "feature_precision": "explicit FP16 round trip for cached and actual-RGB extracted tokens",
                "predict_inputs": "before PIL RGB and actually received current PIL RGB only",
                "official_test_pixels_read": 0}
    save_json(metadata_path, metadata)
    print(f"FROZEN heldout={heldout} threshold={metadata['threshold']:.3f} "
          f"calibration microF1={f1[selected]:.6f}", flush=True)


def exact_ap(labels, scores):
    positive = int(labels.sum())
    if positive == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    tp = np.cumsum(labels[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(scores) - 1]
    group_tp = tp[ends]
    increments = np.diff(np.r_[0, group_tp])
    return float(np.sum(increments.astype(np.float64) * group_tp / (ends + 1)) / positive)


def pixel_metrics(labels, scores, threshold):
    prediction = scores >= threshold
    tp, fp = int((labels & prediction).sum()), int((~labels & prediction).sum())
    fn, tn = int((labels & ~prediction).sum()), int((~labels & ~prediction).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "ap": exact_ap(labels, scores), "valid_pixels": len(labels),
            "positive_pixels": int(labels.sum()), "positive_fraction": float(labels.mean())}


def rgb_api_check(pairs, features, evaluator, folder):
    row, pair = pairs[128], features[128]
    before = Image.open(verified_path(ROOT, row, "before")).convert("RGB")
    current = Image.open(verified_path(ROOT, row, "after")).convert("RGB")
    expected = evaluator._predict_from_features(pair[0], pair[1])
    expected_old = evaluator._predict_from_features(pair[0], pair[0])
    direct = evaluator.predict(before, current)
    full_timing = evaluator.last_timing
    old = evaluator.predict(before, before)
    old_timing = evaluator.last_timing
    drop = evaluator.predict(before, before.copy())
    drop_timing = evaluator.last_timing
    difference = float(np.max(np.abs(direct - expected)))
    old_difference = float(np.max(np.abs(old - expected_old)))
    if difference > 2e-5 or old_difference > 2e-5 or not np.array_equal(old, drop):
        raise ValueError("actual receiver RGB API disagrees with cache replay or old/drop identity")
    check = {"utc": now(), "pair_id": row["id"], "heldout": evaluator.heldout,
             "native_shape": list(direct.shape), "max_abs_full_cache_difference": difference,
             "max_abs_old_cache_difference": old_difference, "old_drop_exactly_equal": True,
             "current_RGB_sha256": row["files"]["after"]["sha256"],
             "before_RGB_sha256": row["files"]["before"]["sha256"],
             "public_API_used_actual_RGB_only": True,
             "full_current_timing": full_timing, "old_timing": old_timing, "drop_timing": drop_timing}
    save_json(folder / "rgb_api_check.json", check)
    return check


def report_fold(pairs, features, heldout):
    folder = OUT / f"heldout_{heldout}"
    path = folder / "validation_results.json"
    metadata_path = folder / "metadata.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["metadata_sha256"] != sha_file(metadata_path):
            raise ValueError("refusing a second validation report with changed frozen metadata")
        print(f"heldout={heldout}: retaining existing one-time validation report", flush=True)
        return existing
    evaluator = SecondGroundEvaluator(ROOT, heldout)
    validation_started = now()
    if validation_started <= evaluator.metadata["frozen_utc"]:
        raise ValueError("validation must begin strictly after head and threshold freeze")
    save_json(folder / "validation_started.json", {"utc": validation_started,
               "frozen_utc": evaluator.metadata["frozen_utc"], "metadata_sha256": sha_file(metadata_path)})
    labels, full_scores, old_scores, per_pair = [], [], [], []
    for row, pair in zip(pairs[128:], features[128:], strict=True):
        truth, valid, _ = read_target(row, heldout)
        full = evaluator._predict_from_features(pair[0], pair[1])[valid]
        old = evaluator._predict_from_features(pair[0], pair[0])[valid]
        y = truth[valid]
        fm, om = pixel_metrics(y, full, evaluator.threshold), pixel_metrics(y, old, evaluator.threshold)
        per_pair.append({"pair_id": row["id"], "positive_pair": bool(y.any()),
                         "excluded_pixels": int((~valid).sum()), "full_current": fm,
                         "old_state": om, "dropped_all_updates": dict(om),
                         "old_and_drop_received_features": "both actually use pair[before], never current"})
        labels.append(y)
        full_scores.append(full)
        old_scores.append(old)
    y = np.concatenate(labels)
    full = pixel_metrics(y, np.concatenate(full_scores), evaluator.threshold)
    old = pixel_metrics(y, np.concatenate(old_scores), evaluator.threshold)
    api = rgb_api_check(pairs, features, evaluator, folder)
    result = {"schema": "second-ground-validation-v1", "heldout": heldout, "utc": now(),
              "validation_started_utc": validation_started, "frozen_utc": evaluator.metadata["frozen_utc"],
              "metadata_sha256": sha_file(metadata_path), "threshold": evaluator.threshold,
              "pair_count": len(per_pair), "positive_pair_count": sum(p["positive_pair"] for p in per_pair),
              "valid_pixels": len(y), "positive_pixels": int(y.sum()),
              "conditions": {"full_current": full, "old_state": old, "dropped_all_updates": dict(old)},
              "per_pair": per_pair, "rgb_api_check": api,
              "ap_definition": "exact noninterpolated pixel AP with equal-score ties grouped",
              "report_source": "frozen full-information feature replay; RGB-only public API checked on fixed first validation pair",
              "conditions_definition": {"full_current": "before/current original RGB frozen features",
                                        "old_state": "before/before original RGB frozen features",
                                        "dropped_all_updates": "same before/before features, receiver got no current update"},
              "rules": RULES, "official_test_pixels_read": 0,
              "validation_used_for_model_epoch_threshold_selection": False}
    save_json(path, result)
    print(f"VALIDATION heldout={heldout} full F1/IoU/AP={full['f1']:.6f}/{full['iou']:.6f}/{full['ap']:.6f} "
          f"old/drop={old['f1']:.6f}/{old['iou']:.6f}/{old['ap']:.6f} "
          f"positive pairs={result['positive_pair_count']}/64", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "report", "all"), default="all")
    parser.add_argument("--heldout", type=int, nargs="+", choices=(5, 3, 4), default=[5, 3, 4])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, features, spec = dataset()
    for heldout in args.heldout:
        if args.phase in ("fit", "all"):
            fit_fold(pairs, features, spec, heldout)
        if args.phase in ("report", "all"):
            report_fold(pairs, features, heldout)


if __name__ == "__main__":
    main()
