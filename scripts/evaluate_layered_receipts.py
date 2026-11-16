"""Frozen RGB task evaluation after complete LB01/LA01 or SP03/IF06 replay.

Only decoded receiver RGB enters the public SpaceNet7 model. Labels and native
valid masks are loaded offline; seeds are nested within geographic AOIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))

import numpy as np
from PIL import Image

from safesync.full_image_codecs import receive as receive_full
from safesync.layered_cache import DetailCache, digest
from safesync.layered_wire import LayeredReceiver, history_after, parse_frame
from safesync.pixel_wire import ACK, PixelState, checksum, decode_update
from safesync.pretrained_ground import PretrainedGroundEvaluator
from scripts.evaluate_pretrained_receipts import (
    canonical_hash, counts, flat_counts, offline_gt, rates, save_json, sha, sha_bytes,
)

METHODS = ["gaussian_20261029", "dct_bound", "full_cache_mse", "random_1729",
           "full_jpeg420", "full_jpeg2000", "full_webp"]
PREFIXES = ["building", "event", "initial_only_building", "initial_only_event"]
SUPPORT = ["tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels", "predicted_positive_pixels"]
METRICS = ["f1", "iou", "precision", "recall"]
OLD_CACHES = [ROOT / "results/full_codec_controls/development_v1/probabilities",
              ROOT / "results/pretrained_receipts/development_side4/probabilities",
              ROOT / "results/layered_tasks/development_v2_pre_identity_fix/probabilities"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def now():
    return datetime.now(timezone.utc).isoformat()


def seed_identity(method):
    if method.startswith("gaussian_"):
        return "gaussian", int(method.rsplit("_", 1)[1])
    if method.startswith("random_"):
        return "random", int(method.rsplit("_", 1)[1])
    return method, None


def row_identity(aoi, method, budget, step):
    """Identical packets may occur at different budgets or routing methods."""
    return {"aoi": aoi, "method": method, "budget": budget, "step": step}


def offline_masks(month, first, aoi_id, event_path, confirmation):
    if not confirmation:
        return offline_gt(month, first, aoi_id)
    paths = {"building": ROOT / month["building_mask_path"],
        "current_valid": ROOT / month["valid_mask_path"], "first_valid": ROOT / first["valid_mask_path"],
        "event": event_path}
    arrays = {key: np.asarray(Image.open(path)) > 0 for key, path in paths.items()}
    valid = arrays["current_valid"] & arrays["first_valid"]
    if any(value.shape != valid.shape for value in arrays.values()):
        raise ValueError("confirmation GT/valid masks differ from the native grid")
    # The event PNG was rasterized by the frozen run source. Preserve the
    # official current/first polygons too, so its track-Id provenance is explicit.
    hashes = {key: sha(path) for key, path in paths.items()}
    hashes.update({"current_official_track_labels": sha(ROOT / month["label_geojson_path"]),
                   "first_official_track_labels": sha(ROOT / first["label_geojson_path"])})
    return arrays["building"], arrays["event"], valid, hashes


def probability_from_receiver(evaluator, rgb, out, model_hash, threshold_hash, origin):
    """No source-image path is accepted; exact decoded input hashes key reuse."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    input_sha = sha_bytes(rgb.tobytes())
    key = canonical_hash({"decoded_native_RGB_sha256": input_sha, "shape": list(rgb.shape),
                          "model_hash": model_hash, "threshold_hash": threshold_hash})
    own = out / "probabilities" / (key + ".npz")
    candidates = [own] + [folder / (key + ".npz") for folder in OLD_CACHES]
    path = next((p for p in candidates if p.exists()), own)
    cached, timing = path.exists(), None
    if cached:
        with np.load(path, allow_pickle=False) as saved:
            expected = {"input_rgb_sha256": input_sha, "model_hash": model_hash, "threshold_hash": threshold_hash}
            if any(str(saved[k]) != v for k, v in expected.items()):
                raise ValueError("actual receiver probability cache identity changed")
            probability = saved["probability"].copy()
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
        raise ValueError("receiver probability has wrong native shape/nonfinite values")
    return probability, {"prediction_input_RGB_sha256": input_sha, "prediction_native_shape": list(rgb.shape),
        "probability_path": relative(path), "probability_file_sha256": sha(path),
        "probability_array_sha256": sha_bytes(probability.tobytes()), "probability_cache_hit": cached,
        "cache_origin": "this run" if path == own else "previous actual receiver receipt",
        "prediction_inference_timing": timing}


def checked_ack(ack, packet, magic, version, observation, state_digest):
    if len(ack) != ACK.size + 16 or checksum(ack[:-16]) != ack[-16:]:
        raise ValueError("invalid actual ACK checksum/length")
    fields = ACK.unpack(ack[:-16])
    if fields != (magic, version, observation, checksum(packet), state_digest):
        raise ValueError("actual ACK does not bind this transaction and state")


class ReplayedSequence:
    """One fresh receiver, initialized only from its own bootstrap packet."""
    def __init__(self, initial, native_shape, region, expected_config=None):
        self.height, self.width = native_shape
        h, w = math.ceil(self.height / 64) * 64, math.ceil(self.width / 64) * 64
        self.layered = initial[:4] == b"LB01"
        if self.layered:
            length = struct.unpack("<I", initial[4:8])[0]
            config = json.loads(initial[8:8 + length])
            if config != expected_config:
                raise ValueError("bootstrap detail-cache configuration differs from frozen run")
            self.receiver = LayeredReceiver()
            self.initial_ack = self.receiver.receive(initial)
            self.state = self.receiver.state
            if self.state.cache.wire != DetailCache.from_rgb(self.state.archive, config).wire:
                raise ValueError("initial receiver cache does not describe decoded ROI archive")
            checked_ack(self.initial_ack, initial, b"LK01", 1, 0, self.state.history)
        else:
            if initial[:4] != b"SP03" or expected_config is not None:
                raise ValueError("wrong whole-image bootstrap protocol")
            self.state, self.initial_ack, _ = decode_update(PixelState(np.zeros((h, w, 3), np.uint8), region), initial)
            checked_ack(self.initial_ack, initial, b"SA02", 1, 0, self.state.digest())
        if self.image().shape != (h, w, 3) or self.state.region != region:
            raise ValueError("bootstrap grid/region differs from declared native AOI")
        self.initial_sha = sha_bytes(self.image().tobytes())

    def image(self):
        return self.state.render() if self.layered else self.state.rgb

    def apply(self, packet, step):
        if self.layered:
            previous = self.state
            parsed = parse_frame(packet, previous.cache.config["roi_quality"])
            self.receiver.expect(step)
            ack = self.receiver.receive(packet)
            self.state = self.receiver.state
            if self.state.history != history_after(previous.history, packet, self.state.cache):
                raise ValueError("receiver cache/transaction history differs")
            for idx, roi in parsed["patches"]:
                y, x = divmod(idx, self.state.archive.shape[1] // 64)
                if not np.array_equal(self.state.archive[y*64:(y+1)*64, x*64:(x+1)*64], roi):
                    raise ValueError("decoded ROI was placed at incorrect archive coordinates")
            if digest(self.state.cache.wire[:-16]) != self.state.cache.wire[-16:]:
                raise ValueError("updated detail-cache checksum differs")
            checked_ack(ack, packet, b"LK01", self.state.version, step, self.state.history)
            state_info = {"receiver_cache_sha256": sha_bytes(self.state.cache.wire),
                "receiver_cache_bytes": len(self.state.cache.wire), "receiver_history_digest": self.state.history.hex(),
                "receiver_archive_RGB_sha256": sha_bytes(self.state.archive.tobytes()),
                "receiver_coarse_RGB_sha256": sha_bytes(self.state.coarse.tobytes()),
                "applied_ROI_count": len(parsed["patches"])}
        else:
            self.state, ack = receive_full(self.state, packet)
            checked_ack(ack, packet, b"IA06", self.state.version, step, self.state.digest())
            state_info = {"receiver_cache_sha256": None, "receiver_cache_bytes": 0,
                "receiver_history_digest": self.state.digest().hex(), "receiver_archive_RGB_sha256": None,
                "receiver_coarse_RGB_sha256": None, "applied_ROI_count": None}
        observation = self.state.observation if self.layered else self.state.observed_at
        if observation != step or self.state.version != step + 1:
            raise ValueError("receiver sequence version/observation differs")
        rgb = self.image()
        if rgb.shape != (math.ceil(self.height/64)*64, math.ceil(self.width/64)*64, 3):
            raise ValueError("rendered receiver dimensions changed within sequence")
        return rgb, ack, state_info


def load_run(run, manifest, budgets, allowed_indices):
    """Do not consume a mutable or partial transmission trace."""
    progress, spec = read(run / "progress.json"), read(run / "run_spec.json")
    if progress["status"] != "complete" or progress["records"] != progress["expected"]:
        raise ValueError("source sequence generation has not completed")
    if progress["records_sha256"] != sha(run / "records.csv") or progress["spec_sha256"] != sha(run / "run_spec.json"):
        raise ValueError("source completion hashes differ")
    if spec["manifest_sha256"] != canonical_manifest_sha(manifest):
        raise ValueError("source run uses another data manifest")
    if any(i not in allowed_indices for i in spec["area_indices"]):
        raise ValueError("source run includes an unauthorized AOI")
    if not set(METHODS).issubset(spec["methods"]) or not set(budgets).issubset(spec["budgets"]):
        raise ValueError("source run lacks a preselected method/budget")
    for name, expected in spec["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("frozen transmission source changed: " + name)
        if sha(run / "source_snapshot" / name.replace("/", "__")) != expected:
            raise ValueError("transmission source snapshot changed: " + name)
    rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8", newline="")))
    if len(rows) != progress["expected"]:
        raise ValueError("source trace count differs from completion marker")
    selected = [r for r in rows if r["method"] in METHODS and int(r["budget"]) in budgets]
    return spec, selected


def canonical_manifest_sha(manifest):
    return manifest["_file_sha256"]


def aggregate(rows):
    """Counts first within AOI/seed; seed rates average within AOI, then AOIs."""
    groups = {}
    for row in rows:
        family, seed = seed_identity(row["method"])
        key = row["method"], row["budget"], row["aoi"]
        group = groups.setdefault(key, {"method": key[0], "family": family, "seed": seed,
            "budget": key[1], "aoi": key[2], "frames": 0, "down_bytes": 0, "ack_bytes": 0,
            "bootstrap_down_bytes": row["bootstrap_down_bytes"], "bootstrap_ack_bytes": row["bootstrap_ack_bytes"],
            "source_encoding_ms": 0.0, "prediction_inference_ms": 0.0, "new_prediction_count": 0,
            **{prefix: {field: 0 for field in SUPPORT} for prefix in PREFIXES}})
        group["frames"] += 1
        for field in ["down_bytes", "ack_bytes", "source_encoding_ms"]:
            group[field] += row[field]
        timing = row["prediction_inference_timing"]
        if timing:
            group["new_prediction_count"] += 1
            group["prediction_inference_ms"] += timing["total_ms"]
        for prefix in PREFIXES:
            for field in SUPPORT:
                group[prefix][field] += row[prefix + "_" + field]
    for group in groups.values():
        for prefix in PREFIXES:
            group[prefix].update(rates(group[prefix]))
        group["actual_bidirectional_bytes"] = group["down_bytes"] + group["ack_bytes"]
        group["bootstrap_bidirectional_bytes"] = group["bootstrap_down_bytes"] + group["bootstrap_ack_bytes"]
        group["total_bytes_including_bootstrap"] = group["actual_bidirectional_bytes"] + group["bootstrap_bidirectional_bytes"]
        group["mean_update_bidirectional_bytes"] = group["actual_bidirectional_bytes"] / group["frames"]
    nested = {}
    for g in groups.values():
        nested.setdefault((g["family"], g["budget"], g["aoi"]), []).append(g)
    per_aoi = []
    for (family, budget, aoi), members in nested.items():
        result = {"family": family, "budget": budget, "aoi": aoi, "nested_seed_count": len(members),
                  "methods": [m["method"] for m in members], "frames_per_seed": [m["frames"] for m in members]}
        for prefix in PREFIXES:
            result[prefix] = {metric: mean_defined([m[prefix][metric] for m in members]) for metric in METRICS}
        result["mean_update_bidirectional_bytes"] = float(np.mean([m["mean_update_bidirectional_bytes"] for m in members]))
        per_aoi.append(result)
    macro = []
    for family, budget in sorted({(g["family"], g["budget"]) for g in per_aoi}):
        members = [g for g in per_aoi if (g["family"], g["budget"]) == (family, budget)]
        result = {"family": family, "budget": budget, "AOIs": len(members), "seed_unit": "nested within AOI"}
        for prefix in PREFIXES:
            result[prefix] = {}
            for metric in METRICS:
                values = [m[prefix][metric] for m in members if m[prefix][metric] is not None]
                result[prefix][metric] = mean_defined(values)
                result[prefix][metric + "_defined_AOIs"] = len(values)
        result["mean_update_bidirectional_bytes"] = float(np.mean([m["mean_update_bidirectional_bytes"] for m in members]))
        macro.append(result)
    return {"statistical_unit": "geographic AOI; sum month TP/FP/FN within AOI, seeds nested within AOI, AOI equal weights",
            "zero_denominator": "null; empty valid masks contribute zero counts and retain actual bytes",
            "per_aoi_seed": list(groups.values()), "per_aoi": per_aoi, "macro_aoi": macro}


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    if not rows:
        return
    fields = [k for k, v in rows[0].items() if not isinstance(v, (dict, list)) and k != "prediction_inference_timing"]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=[ROOT / "results/layered/development_v2/area_2", ROOT / "results/layered/development_v2/area_3"])
    parser.add_argument("--out", type=Path, default=ROOT / "results/layered_tasks/development_v2")
    parser.add_argument("--budgets", nargs="+", type=int, default=[32768, 65536, 131072])
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/spacenet7_pilot/development_manifest_v1.json")
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    runs, out, manifest_path = [p.resolve() for p in args.runs], args.out.resolve(), args.manifest.resolve()
    if args.confirmation:
        if args.budgets != [32768, 65536, 131072] or "confirmation" not in manifest_path.name:
            raise ValueError("confirmation requires all three frozen budgets and its specific manifest")
        allowed_indices = list(range(8))
    else:
        if manifest_path.name != "development_manifest_v1.json":
            raise ValueError("development only reads the original four-AOI manifest")
        allowed_indices = [2, 3]
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(manifest_path)
    manifest["_file_sha256"] = sha(manifest_path)
    run_specs, source_rows = [], []
    for run in runs:
        spec, rows = load_run(run, manifest, args.budgets, allowed_indices)
        if spec["scope"] != ("confirmation" if args.confirmation else "development"):
            raise ValueError("source and task evaluation scopes differ")
        run_specs.append(spec)
        source_rows.extend(rows)
    aois = [a for i, a in enumerate(manifest["aois"]) if any(i in s["area_indices"] for s in run_specs)]
    groups = {}
    for row in source_rows:
        groups.setdefault((row["aoi"], row["method"], int(row["budget"])), []).append(row)
    for aoi in aois:
        for method in METHODS:
            for budget in args.budgets:
                group = sorted(groups.get((aoi["aoi_id"], method, budget), []), key=lambda r: int(r["step"]))
                if [r["month"] for r in group] != [m["month"] for m in aoi["months"][1:]] or [int(r["step"]) for r in group] != list(range(1, len(aoi["months"]))):
                    raise ValueError("fixed AOI/month cohort has omitted, duplicate, or reordered updates")
    expected = sum(len(a["months"]) - 1 for a in aois) * len(METHODS) * len(args.budgets)
    if len(source_rows) != expected:
        raise ValueError("fixed receipt cohort count differs")
    evaluator = PretrainedGroundEvaluator(ROOT)
    thresholds = {"building": evaluator.threshold, "event": evaluator.change_threshold}
    if thresholds != {"building": 0.5, "event": 0.9610000252723694}:
        raise ValueError("precommitted task thresholds changed")
    model_hash, threshold_hash = canonical_hash(json.loads(evaluator.identity)), canonical_hash(thresholds)
    files = [Path(__file__), ROOT / "scripts/evaluate_pretrained_receipts.py", ROOT / "safesync/pretrained_ground.py",
             manifest_path, ROOT / "resources/pretrained_ground/calibration.json"]
    files += [ROOT / p for p in run_specs[0]["source_sha256"]]
    files += [run / name for run in runs for name in ["run_spec.json", "records.csv", "progress.json"]]
    overlap_path = ROOT / ("results/layered_tasks/confirmation_training_overlap.json" if args.confirmation else "resources/pretrained_ground/training_overlap.json")
    files.append(overlap_path)
    if args.confirmation:
        files.append(ROOT / "docs/LAYERED_CONFIRMATION_PROTOCOL.md")
    hashes = {relative(p): sha(p) for p in files}
    spec = {"schema": "layered-actual-receiver-tasks-v1", "scope": "confirmation" if args.confirmation else "development",
        "runs": [relative(r) for r in runs], "aoi_ids": [a["aoi_id"] for a in aois], "methods": METHODS,
        "budgets": args.budgets, "expected_outputs": expected, "thresholds": thresholds, "model_hash": model_hash,
        "threshold_hash": threshold_hash, "weight_sha256": evaluator.metadata["weight_sha256"], "source_hashes": hashes,
        "prediction_input": "actual LB01/LA01 LayeredState.render or SP03/IF06 PixelState RGB, cropped to manifest native dimensions",
        "current_sender_RGB_or_features_opened": False, "training_overlap": read(overlap_path),
        "event_target": "official current footprint track Id absent from first observation Id set; no receiver-staleness exclusion",
        "valid_mask": "current and first valid intersection, offline only; native padding invalid",
        "statistical_unit": "AOI, with seeds nested within AOI", "claim_scope": "fixed task-model communication distortion; no independent geographic generalization claim"}
    if (out / "run_spec.json").exists() and read(out / "run_spec.json") != spec:
        raise ValueError("task evaluation source/configuration changed; use a separate output")
    save_json(out / "run_spec.json", spec)
    snap = out / "source_snapshots"
    snap.mkdir(exist_ok=True)
    for path in files:
        destination = snap / relative(path).replace("/", "__")
        if not destination.exists():
            shutil.copyfile(path, destination)
    results, reference_maps, initial_receipts = [], {}, {}
    for budget in args.budgets:
        for method in METHODS:
            for aoi in aois:
                aid, first = aoi["aoi_id"], aoi["months"][0]
                month_map = {m["month"]: m for m in aoi["months"]}
                sequence = sorted(groups[(aid, method, budget)], key=lambda r: int(r["step"]))
                bootstrap_path = ROOT / sequence[0]["bootstrap_path"]
                bootstrap = bootstrap_path.read_bytes()
                source_spec = next(s for s in run_specs if int(sequence[0]["area_index"]) in s["area_indices"])
                config = source_spec["method_configs"].get(method)
                region = hashlib.sha256(aid.encode()).digest()[:16]
                width, height = first["dimensions"]
                receiver = ReplayedSequence(bootstrap, (height, width), region, config)
                if receiver.initial_ack != bootstrap_path.with_suffix(".ack").read_bytes():
                    raise ValueError("replayed bootstrap ACK differs from stored actual ACK")
                reference_key = canonical_hash({"aoi": aid, "native": [height, width], "initial_full_RGB": receiver.initial_sha,
                                                 "model_hash": model_hash, "threshold_hash": threshold_hash})
                if reference_key not in reference_maps:
                    reference_maps[reference_key], receipt = probability_from_receiver(evaluator, receiver.image()[:height, :width], out,
                        model_hash, threshold_hash, {"bootstrap": relative(bootstrap_path), "bootstrap_sha256": sha_bytes(bootstrap)})
                    initial_receipts[reference_key] = {**receipt, "aoi": aid, "bootstrap_path": relative(bootstrap_path),
                        "receiver_full_RGB_sha256": receiver.initial_sha, "bootstrap_packet_sha256": sha_bytes(bootstrap),
                        "bootstrap_down_bytes": len(bootstrap), "bootstrap_ack_bytes": len(receiver.initial_ack)}
                    save_json(out / "initial_receipts.json", initial_receipts)
                reference = reference_maps[reference_key]
                cumulative_down, cumulative_up = len(bootstrap), len(receiver.initial_ack)
                for source in sequence:
                    packet_path, step = ROOT / source["packet_path"], int(source["step"])
                    packet, actual_ack = packet_path.read_bytes(), packet_path.with_suffix(".ack").read_bytes()
                    if sha_bytes(packet) != source["packet_sha256"] or sha_bytes(actual_ack) != source["ack_sha256"]:
                        raise ValueError("actual input packet/ACK SHA differs from source trace")
                    rgb, ack, state_info = receiver.apply(packet, step)
                    decoded_sha = sha_bytes(rgb.tobytes())
                    if ack != actual_ack or decoded_sha != source["receiver_rgb_sha256"]:
                        raise ValueError("replayed receiver RGB/ACK differs from actual trace")
                    cumulative_down += len(packet)
                    cumulative_up += len(ack)
                    measured = {"down_bytes": len(packet), "up_bytes": len(ack), "reference_up_bytes": 0,
                        "bootstrap_down_bytes": len(bootstrap), "bootstrap_up_bytes": len(receiver.initial_ack),
                        "cumulative_down_bytes": cumulative_down, "cumulative_up_bytes": cumulative_up,
                        "sender_summary_bytes": state_info["receiver_cache_bytes"]}
                    if any(value != int(source[k]) for k, value in measured.items()) or len(packet) > budget:
                        raise ValueError("actual replay byte/cache accounting differs from source trace")
                    month = month_map[source["month"]]
                    target_path = packet_path.parents[2] / "gt" / aid / month["month"] / "new_track.png"
                    building, event, valid, gt_hashes = offline_masks(month, first, aid, target_path, args.confirmation)
                    if valid.shape != (height, width):
                        raise ValueError("offline labels differ from cropped native receiver grid")
                    if not np.array_equal(np.asarray(Image.open(target_path)) > 0, event):
                        raise ValueError("frozen track-event GT differs from transmission run")
                    key = canonical_hash({"packet": sha_bytes(packet), "bootstrap": sha_bytes(bootstrap), "receiver": decoded_sha,
                        "model": model_hash, "threshold": threshold_hash, "gt": gt_hashes, "source": hashes[relative(Path(__file__))],
                        "row_identity": row_identity(aid, method, budget, step)})
                    record_path = out / "records" / (key + ".json")
                    if record_path.exists():
                        row = read(record_path)
                        if sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
                            raise ValueError("saved receiver probability file changed")
                    else:
                        probability, receipt = probability_from_receiver(evaluator, rgb[:height, :width], out, model_hash, threshold_hash,
                            {"packet": relative(packet_path), "packet_sha256": sha_bytes(packet), "receiver_full_RGB_sha256": decoded_sha})
                        event_score = np.maximum(probability - reference, 0)
                        row = {"aoi": aid, "method": method, "budget": budget, "step": step, "month": month["month"],
                            "protocol": "LB01/LA01" if receiver.layered else "SP03/IF06", "packet_path": relative(packet_path),
                            "packet_sha256": sha_bytes(packet), "ack_sha256": sha_bytes(ack), "receiver_RGB_sha256_verified": decoded_sha,
                            "model_hash": model_hash, "threshold_hash": threshold_hash, "gt_sha256": gt_hashes,
                            "new_track_target_sha256": sha(target_path), "reference_receipt_key": reference_key,
                            "down_bytes": len(packet), "ack_bytes": len(ack), "reference_up_bytes": 0,
                            "bootstrap_down_bytes": len(bootstrap), "bootstrap_ack_bytes": len(receiver.initial_ack),
                            "cumulative_down_bytes": cumulative_down, "cumulative_up_bytes": cumulative_up,
                            "source_encoding_ms": float(source["compute_ms"]), "source_encoder_meta": json.loads(source["encoder_meta"]),
                            **state_info, **receipt,
                            **flat_counts("building", counts(probability >= evaluator.threshold, building, valid)),
                            **flat_counts("event", counts(event_score >= evaluator.change_threshold, event, valid)),
                            **flat_counts("initial_only_building", counts(reference >= evaluator.threshold, building, valid)),
                            **flat_counts("initial_only_event", counts(np.zeros_like(event, bool), event, valid))}
                        save_json(record_path, row)
                    if any(row[k] != v for k, v in row_identity(aid, method, budget, step).items()):
                        raise ValueError("cached metric row belongs to another method/budget/sequence")
                    results.append(row)
                    save_json(out / "progress.json", {"status": "in_progress", "completed": len(results), "expected": expected,
                        "aoi": aid, "method": method, "budget": budget, "updated_utc": now(), "all_receiver_RGB_and_ACK_match": True})
                save_json(out / "summary.partial.json", aggregate(results))
                write_csv(out / "records.csv", results)
                per = next(g for g in aggregate(results)["per_aoi_seed"] if (g["method"], g["budget"], g["aoi"]) == (method, budget, aid))
                print(json.dumps({"method": method, "budget": budget, "aoi": aid, "frames": per["frames"],
                    "building_f1": per["building"]["f1"], "event_f1": per["event"]["f1"], "outputs_done": len(results)}), flush=True)
        save_json(out / f"budget_{budget}_complete.json", {"outputs_done": len(results), "summary": aggregate(results)})
    if len(results) != expected:
        raise ValueError("incomplete fixed receipt cohort")
    identities = {(r["aoi"], r["method"], r["budget"], r["step"]) for r in results}
    if len(identities) != expected:
        raise ValueError("duplicated metric attribution across method/budget/sequence")
    summary = {**aggregate(results), "status": "complete", "outputs": len(results), "sequences": len(groups),
        "generated_utc": now(), "source_spec_sha256": sha(out / "run_spec.json"), "all_receiver_RGB_ACK_cache_history_match": True,
        "model_and_thresholds_unchanged": True, "independent_geographic_generalization_claim": False}
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {"status": "complete", "completed": len(results), "expected": expected,
        "records_sha256": sha(out / "records.csv"), "summary_sha256": sha(out / "summary.json"), "updated_utc": now()})
    print("complete", len(results), "actual receiver task outputs", flush=True)


if __name__ == "__main__":
    main()
