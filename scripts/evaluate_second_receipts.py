"""Evaluate only actually decoded SECOND pixel packets with frozen RGB heads.

No sender current features, official test pixels, head fitting, or routing occurs
here. Source producer files are read-only. Completed caches bind packet, initial
state, frozen head, threshold, and offline label/mask identities.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from safesync.data import label_array, verified_path
from safesync.features import FrozenEncoder
from safesync.pixel_wire import HEADER, PixelState, TransactionReceiver
from safesync.second_ground import ENCODER_CONFIG, SecondGroundEvaluator, sha_file

SCHEMA = "SECOND-actual-receiver-task-v1"
EXPECTED = 6912
ROWS_PER_PAIR = 108
SOURCE_FILES = ("scripts/evaluate_second_receipts.py", "safesync/second_ground.py",
                "safesync/features.py", "safesync/pixel_wire.py", "safesync/data.py")
TASK_FIELDS = ("task_schema", "source_record_sha256", "task_cache_key", "initial_packet_sha256",
               "initial_ack_sha256", "ground_head_sha256", "ground_metadata_sha256",
               "ground_threshold", "gt_binary_sha256", "gt_before_sha256", "gt_after_sha256",
               "task_valid_mask_sha256", "task_truth_sha256", "task_tokens_path", "task_tokens_sha256",
               "decoded_before_rgb_sha256", "decoded_current_rgb_sha256", "task_tp", "task_fp",
               "task_fn", "task_tn", "task_valid_pixels", "task_positive_pixels", "task_positive_pair",
               "task_f1", "task_iou", "task_ap", "task_ap_definition", "task_inference_ms")


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows, fields=None):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def safe_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("source path escapes the workspace")
    return path


def row_key(row):
    return (str(row["pair_id"]), int(row["heldout"]), row["method"], int(row["seed"]), int(row["budget"]))


def expected_keys(pair_id, config):
    return {(pair_id, h, method, seed, budget) for h in config["heldouts"]
            for method in config["methods"]
            for seed in (config["random_seeds"] if method == "pixel_random" else [0])
            for budget in config["budgets_total_bytes"]}


def read_source(path, spec):
    """Take one immutable CSV byte snapshot; ignore an unfinished final line."""
    blob = path.read_bytes()
    complete = blob.endswith(b"\n")
    if not complete:
        blob = blob[:blob.rfind(b"\n") + 1]
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
    fields, rows = reader.fieldnames, list(reader)
    if not fields or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("malformed complete source CSV row; refusing partial evidence")
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError("duplicate logical transaction in source CSV")
        seen.add(key)
        if row["pair_id"] not in spec["validation_pair_ids"]:
            raise ValueError("source transaction is outside the fixed validation pairs")
        groups[row["pair_id"]].append(row)
    selected = []
    for pair_id in spec["validation_pair_ids"]:
        group = groups[pair_id]
        if {row_key(r) for r in group} != expected_keys(pair_id, spec["config"]):
            break  # consume only the fixed, complete original-pair prefix
        selected.append((pair_id, group))
    return fields, selected, {"csv_sha256": digest(blob), "complete_final_line": complete,
                              "source_rows": len(rows), "complete_pair_prefix": len(selected)}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def exact_ap(labels, scores):
    positive = int(labels.sum())
    if not positive:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    tp = np.cumsum(labels[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(scores) - 1]
    group_tp = tp[ends]
    increment = np.diff(np.r_[0, group_tp])
    return float(np.sum(increment.astype(np.float64) * group_tp / (ends + 1)) / positive)


def task_metrics(truth, valid, probability, threshold):
    y, score = truth[valid], probability[valid]
    prediction = score >= threshold
    tp, fp = int((y & prediction).sum()), int((~y & prediction).sum())
    fn, tn = int((y & ~prediction).sum()), int((~y & ~prediction).sum())
    return {"task_tp": tp, "task_fp": fp, "task_fn": fn, "task_tn": tn,
            "task_valid_pixels": len(y), "task_positive_pixels": int(y.sum()),
            "task_positive_pair": bool(y.any()), "task_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "task_iou": ratio(tp, tp + fp + fn), "task_ap": exact_ap(y, score),
            "task_ap_definition": "exact native-pixel noninterpolated AP; equal scores grouped; no positive pixels=null"}


def offline_target(item, heldout):
    first = label_array(verified_path(ROOT, item, "label_before"))
    second = label_array(verified_path(ROOT, item, "label_after"))
    with Image.open(verified_path(ROOT, item, "binary_change")) as image:
        binary = np.asarray(image)
    if binary.shape != (512, 512) or not set(np.unique(binary)).issubset({0, 255}):
        raise ValueError("SECOND native GT_CD encoding/shape changed")
    truth = binary == 255
    valid = (first != heldout) & (second != heldout) & (first != 255) & (second != 255)
    identity = {"gt_binary_sha256": item["files"]["binary_change"]["sha256"],
                "gt_before_sha256": item["files"]["label_before"]["sha256"],
                "gt_after_sha256": item["files"]["label_after"]["sha256"],
                "task_valid_mask_sha256": digest(valid.tobytes()), "task_truth_sha256": digest(truth.tobytes())}
    return truth, valid, identity


def clone_receiver(state):
    return TransactionReceiver(PixelState(state.rgb.copy(), state.region, state.block,
                                         state.version, state.observed_at, state.statuses.copy()))


def initialize_pair(source, item):
    folder = source / "pairs" / item["id"]
    packet, ack = (folder / "initial.bin").read_bytes(), (folder / "initial.ack").read_bytes()
    costs = json.loads((folder / "initial_cost.json").read_text(encoding="utf-8"))
    if (len(packet) != costs["packet_bytes"] or len(ack) != costs["ack_bytes"]
            or digest(packet) != costs["packet_sha256"] or digest(ack) != costs["ack_sha256"]
            or sum(costs["components"].values()) != len(packet)):
        raise ValueError("initialization byte/hash accounting changed")
    region = hashlib.sha256(("SafeSync-SECOND-pixel:" + item["id"]).encode()).digest()[:16]
    receiver = TransactionReceiver(PixelState(np.zeros((512, 512, 3), np.uint8), region))
    actual_ack, _ = receiver.receive(packet)
    if actual_ack != ack:
        raise ValueError("initialization ACK replay differs")
    before = Image.open(verified_path(ROOT, item, "before")).convert("RGB")
    if receiver.state.rgb.tobytes() != before.tobytes():
        raise ValueError("initialization did not losslessly receive the actual before RGB")
    return receiver.state, {"initial_packet_sha256": digest(packet), "initial_ack_sha256": digest(ack),
                            "initial_down_bytes": len(packet), "initial_ack_bytes": len(ack),
                            "decoded_before_rgb_sha256": digest(receiver.state.rgb.tobytes())}


def replay_record(row, prior, initial, producer):
    if row["config_sha256"] != producer["config_sha256"] or row["source_bundle_sha256"] != producer["source_bundle_sha256"]:
        raise ValueError("source row does not match its frozen producer spec")
    if (row["backbone_weight_sha256"] != producer["encoder"]["weight_sha256"]
            or row["known_model_sha256"] != producer["known_model_sha256"][row["heldout"]]
            or row["score_file_sha256"] != producer["score_files_sha256"][f"{row['heldout']}:{row['pair_id']}"]):
        raise ValueError("source row model/score provenance differs from producer spec")
    packet, ack = safe_path(row["packet_path"]).read_bytes(), safe_path(row["ack_path"]).read_bytes()
    if (digest(packet) != row["packet_sha256"] or digest(ack) != row["ack_sha256"]
            or len(packet) != int(row["down_bytes"]) or len(ack) != int(row["ack_bytes"])
            or len(packet) + len(ack) != int(row["total_bytes"]) or len(packet) + len(ack) > int(row["budget"])):
        raise ValueError("current actual-byte budget, packet hash, or ACK hash differs")
    if len(packet) != sum(int(v) for k, v in row.items() if k.startswith("bytes_")):
        raise ValueError("current byte-component accounting differs")
    for key in ("initial_down_bytes", "initial_ack_bytes"):
        if initial[key] != int(row[key]):
            raise ValueError("row initialization cost differs from actual initial stream")
    receiver = clone_receiver(prior)
    actual_ack, _ = receiver.receive(packet)
    rgb = receiver.state.rgb
    if actual_ack != ack or digest(rgb.tobytes()) != row["receiver_sha256"]:
        raise ValueError("actual packet replay/receiver RGB hash differs")
    if receiver.state.digest().hex() != row["receiver_state_digest"]:
        raise ValueError("receiver state digest differs")
    return rgb


class ActualRGBInference:
    """Capture tokens from the same public RGB predict forward, without editing a head."""
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.captured = None
        self.checked = False
        self.hook = evaluator.head.register_forward_hook(self.capture)

    def capture(self, module, arguments, logits):
        self.captured = torch.sigmoid(logits.detach()).reshape(16, 16).cpu().numpy().copy()

    def predict(self, before, current):
        self.captured = None
        probability = self.evaluator.predict(before, current)
        if self.captured is None or self.captured.shape != (16, 16):
            raise ValueError("public RGB API did not execute exactly one expected head")
        if not self.checked:
            tensor = torch.from_numpy(self.captured).to(self.evaluator.device).reshape(1, 1, 16, 16)
            reconstructed = torch.nn.functional.interpolate(tensor, (512, 512), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            if not np.array_equal(reconstructed, probability):
                raise ValueError("saved token scores do not exactly reconstruct the actual RGB API output")
            self.checked = True
        return probability, self.captured.copy(), self.evaluator.last_timing["total_ms"]


def evaluate_record(row, rgb, before_rgb, initial, target, inference, out, spec):
    truth, valid, gt_identity = target
    evaluator = inference.evaluator
    binding = {"schema": SCHEMA, "packet_sha256": row["packet_sha256"],
               "initial_packet_sha256": initial["initial_packet_sha256"],
               "ground_head_sha256": evaluator.metadata["head_sha256"],
               "ground_metadata_sha256": spec["ground_models"][str(evaluator.heldout)]["metadata_sha256"],
               "ground_threshold": evaluator.threshold,
               "backbone_identity": evaluator.metadata["backbone_identity"], **gt_identity}
    key = digest(canonical(binding).encode())
    path, tokens_path = out / "cache" / f"{key}.json", out / "tokens" / f"{key}.npz"
    decoded_hash = digest(rgb.tobytes())
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached["binding"] != binding or cached["task_tokens_sha256"] != sha_file(tokens_path)
                or cached["decoded_current_rgb_sha256"] != decoded_hash):
            raise ValueError("completed task cache binding or token score integrity changed")
    else:
        probability, tokens, elapsed = inference.predict(before_rgb, Image.fromarray(rgb))
        metrics = task_metrics(truth, valid, probability, evaluator.threshold)
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tokens_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, token_probability=tokens.astype(np.float32))
        temporary.replace(tokens_path)
        cached = {"binding": binding, "task_cache_key": key, "task_schema": SCHEMA,
                  **{k: v for k, v in binding.items() if k in TASK_FIELDS}, **metrics,
                  "task_tokens_path": tokens_path.relative_to(ROOT).as_posix(),
                  "task_tokens_sha256": sha_file(tokens_path),
                  "decoded_current_rgb_sha256": decoded_hash,
                  "decoded_before_rgb_sha256": initial["decoded_before_rgb_sha256"],
                  "initial_ack_sha256": initial["initial_ack_sha256"], "task_inference_ms": elapsed}
        save_json(path, cached)
    result = {**row, **{k: v for k, v in cached.items() if k in TASK_FIELDS}}
    result["source_record_sha256"] = digest(canonical(row).encode())
    return result


def sanity_pair(source, item, prior, inferences, targets, out):
    path = out / "sanity" / f"{item['id']}.json"
    binding = {"initial_packet_sha256": sha_file(source / "pairs" / item["id"] / "initial.bin"),
               "full_packet_sha256": sha_file(source / "pairs" / item["id"] / "sanity_full_png.bin"),
               "models": {str(h): {"head_sha256": v.evaluator.metadata["head_sha256"],
                                    "threshold": v.evaluator.threshold, "target": targets[h][2]}
                          for h, v in inferences.items()}}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["binding"] != binding:
            raise ValueError("sanity cache initial/model/threshold/target binding changed")
        return cached["rows"]
    before = Image.fromarray(prior.rgb)
    packet = (source / "pairs" / item["id"] / "sanity_full_png.bin").read_bytes()
    receiver = clone_receiver(prior)
    ack, _ = receiver.receive(packet)
    if ack != (source / "pairs" / item["id"] / "sanity_full_png.ack").read_bytes():
        raise ValueError("full-PNG sanity ACK differs")
    rows = []
    for heldout, inference in inferences.items():
        truth, valid, _ = targets[heldout]
        expected = json.loads((ROOT / f"resources/second_ground_eval/heldout_{heldout}/validation_results.json").read_text(encoding="utf-8"))
        expected = next(r for r in expected["per_pair"] if r["pair_id"] == item["id"])
        probabilities = {}
        for condition, current in (("full_png", Image.fromarray(receiver.state.rgb)),
                                   ("old_state", before), ("dropped_all_updates", before.copy())):
            probability, _, _ = inference.predict(before, current)
            probabilities[condition] = probability
            metrics = task_metrics(truth, valid, probability, inference.evaluator.threshold)
            fixed = expected["full_current" if condition == "full_png" else condition]
            for new, old in (("task_tp", "tp"), ("task_fp", "fp"), ("task_fn", "fn"), ("task_tn", "tn")):
                if metrics[new] != fixed[old]:
                    raise ValueError("actual-RGB old/drop/full sanity differs from fixed-head report")
            if metrics["task_ap"] is not None and abs(metrics["task_ap"] - fixed["ap"]) > 1e-12:
                raise ValueError("actual-RGB sanity AP differs from frozen report")
            rows.append({"pair_id": item["id"], "heldout": heldout, "case": condition,
                         **metrics, "matches_fixed_head_report": True,
                         "decoded_current_rgb_sha256": digest(np.asarray(current).tobytes())})
        if not np.array_equal(probabilities["old_state"], probabilities["dropped_all_updates"]):
            raise ValueError("old and completely dropped actual RGB predictions differ")
    save_json(path, {"schema": SCHEMA, "binding": binding, "rows": rows})
    return rows


def pair_average(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["pair_id"], int(row["heldout"]), int(row["budget"]), row["method"])].append(row)
    fields = ["task_tp", "task_fp", "task_fn", "task_tn", "task_valid_pixels", "task_positive_pixels",
              "total_bytes", "down_bytes", "ack_bytes", "unknown_pixels", "known_pixels", "known_error_sum",
              "unknown_error_sum"] + [f"unknown_retained_pixels_{q}" for q in (25, 30, 35)]
    output = []
    for (pair, heldout, budget, method), rows in sorted(groups.items()):
        if len(rows) != (3 if method == "pixel_random" else 1):
            raise ValueError("pair/seed group is incomplete")
        average = {key: float(np.mean([float(r[key]) for r in rows])) for key in fields}
        aps = [r["task_ap"] for r in rows if r["task_ap"] is not None]
        tp, fp, fn = (average[k] for k in ("task_tp", "task_fp", "task_fn"))
        output.append({"pair_id": pair, "heldout": heldout, "budget": budget, "method": method,
                       "seed_count": len(rows), **average, "task_f1": ratio(2 * tp, 2 * tp + fp + fn),
                       "task_iou": ratio(tp, tp + fp + fn),
                       "mean_seed_exact_AP": float(np.mean(aps)) if aps else None})
    return output


def interval(values):
    finite = values[np.isfinite(values)]
    return tuple(map(float, np.quantile(finite, [.025, .975]))) if len(finite) else (None, None)


def f1_array(counts):
    denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(2 * counts[..., 0], denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def aggregate(records, out, config, prefix=""):
    means = pair_average(records)
    save_csv(out / f"{prefix}pair_means.csv", means)
    groups = defaultdict(list)
    for row in means:
        groups[(row["heldout"], row["budget"], row["method"])].append(row)
    summaries = []
    for (heldout, budget, method), rows in sorted(groups.items()):
        sums = {key: sum(r[key] for r in rows) for key in ("task_tp", "task_fp", "task_fn", "task_tn",
                "task_valid_pixels", "task_positive_pixels", "unknown_pixels", "known_pixels", "known_error_sum")}
        tp, fp, fn = (sums[k] for k in ("task_tp", "task_fp", "task_fn"))
        aps = [r["mean_seed_exact_AP"] for r in rows if r["mean_seed_exact_AP"] is not None]
        summary = {"heldout": heldout, "budget": budget, "method": method, "pairs": len(rows),
                   "task_positive_pairs": sum(r["task_positive_pixels"] > 0 for r in rows),
                   "unknown_positive_pairs": sum(r["unknown_pixels"] > 0 for r in rows), **sums,
                   "task_micro_f1": ratio(2 * tp, 2 * tp + fp + fn), "task_micro_iou": ratio(tp, tp + fp + fn),
                   "mean_positive_pair_exact_AP": float(np.mean(aps)) if aps else None,
                   "mean_actual_total_bytes": float(np.mean([r["total_bytes"] for r in rows])),
                   "source_known_mse": ratio(sums["known_error_sum"], sums["known_pixels"])}
        for q in (25, 30, 35):
            summary[f"unknown_coverage_{q}_micro"] = ratio(sum(r[f"unknown_retained_pixels_{q}"] for r in rows), sums["unknown_pixels"])
        summaries.append(summary)
    save_csv(out / f"{prefix}summary.csv", summaries)
    comparisons = []
    for heldout in config["heldouts"]:
        for budget in config["budgets_total_bytes"]:
            new = sorted(groups[(heldout, budget, "pixel_unknown")], key=lambda r: r["pair_id"])
            sample = np.random.default_rng(20261028).integers(0, len(new), (2000, len(new)))
            for baseline in ("pixel_change", "pixel_random", "unknown_only", "feature_change", "full_jpeg", "full_jpeg2000"):
                old = sorted(groups[(heldout, budget, baseline)], key=lambda r: r["pair_id"])
                if [r["pair_id"] for r in new] != [r["pair_id"] for r in old]:
                    raise ValueError("unpaired method comparison")
                n = np.array([[r[k] for k in ("task_tp", "task_fp", "task_fn")] for r in new])
                b = np.array([[r[k] for k in ("task_tp", "task_fp", "task_fn")] for r in old])
                delta_task = f1_array(n.sum(0)) - f1_array(b.sum(0))
                task_boot = f1_array(n[sample].sum(1)) - f1_array(b[sample].sum(1))
                denominator = np.array([r["unknown_pixels"] for r in new])
                if not np.array_equal(denominator, np.array([r["unknown_pixels"] for r in old])):
                    raise ValueError("unknown evaluation support differs between methods")
                difference = np.array([a["unknown_retained_pixels_30"] - o["unknown_retained_pixels_30"] for a, o in zip(new, old)])
                sampled_denominator = denominator[sample].sum(1)
                unknown_boot = np.divide(difference[sample].sum(1), sampled_denominator,
                                         out=np.full(2000, np.nan), where=sampled_denominator > 0)
                tl, th = interval(task_boot)
                ul, uh = interval(unknown_boot)
                comparisons.append({"heldout": heldout, "budget": budget, "method": "pixel_unknown", "baseline": baseline,
                    "pairs": len(new), "unknown_positive_pairs": int((denominator > 0).sum()),
                    "delta_task_micro_f1": float(delta_task) if np.isfinite(delta_task) else None,
                    "task_paired_bootstrap_low": tl, "task_paired_bootstrap_high": th,
                    "delta_unknown_coverage_30": ratio(difference.sum(), denominator.sum()),
                    "unknown_paired_bootstrap_low": ul, "unknown_paired_bootstrap_high": uh,
                    "unknown_bootstrap_undefined_replicates": int((~np.isfinite(unknown_boot)).sum()),
                    "delta_mean_actual_total_bytes": float(np.mean([a["total_bytes"] - o["total_bytes"] for a, o in zip(new, old)])),
                    "comparison_scope": "same budget cap; actual bytes reported, not exactly matched",
                    "statistical_unit": "original image pair; random seeds averaged inside pair; development evidence only"})
    save_csv(out / f"{prefix}comparisons.csv", comparisons)
    return summaries, comparisons


def prepare(source, out):
    producer_path = source / "run_spec.json"
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = {p["id"]: p for p in manifest["pairs"] if p["split"] == "validation"}
    if (len(items) != 64 or list(items) != producer["validation_pair_ids"]
            or any(p["source_split"] != "train" for p in manifest["pairs"])
            or sha_file(manifest_path) != producer["manifest_sha256"]):
        raise ValueError("producer data is outside the fixed 128/64 official-train development split")
    models = {}
    for h in (5, 3, 4):
        folder = ROOT / f"resources/second_ground_eval/heldout_{h}"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if not metadata["frozen"] or metadata["backbone_identity"] != producer["encoder"]:
            raise ValueError("fixed ground model and producer backbone identities differ")
        models[str(h)] = {"metadata_sha256": sha_file(folder / "metadata.json"),
                          "head_sha256": sha_file(folder / "head.npz"), "threshold": metadata["threshold"],
                          "frozen_utc": metadata["frozen_utc"]}
    source_hashes = {name: sha_file(ROOT / name) for name in SOURCE_FILES}
    spec = {"schema": SCHEMA, "producer_run": source.relative_to(ROOT).as_posix(),
            "producer_run_spec_sha256": sha_file(producer_path), "producer_spec": producer,
            "ground_models": models, "evaluation_source_sha256": source_hashes,
            "expected_records": EXPECTED, "expected_pairs": 64,
            "input": "each true initial.bin then current.bin decoded; actual supplied PIL RGB only enters frozen predict",
            "task_mask": "exclude heldout or semantic ignore at either time; remaining GT_CD 0 is binary no-change, not semantic known",
            "source_known_quality_mask": producer["config"]["known_mask"],
            "unknown_coverage_definition": producer["quality_metric"],
            "AP": "exact per-transaction pixel AP; reported aggregate is mean positive-pair AP, not pooled pixel AP",
            "score_storage": "float32[16,16] captured from same public RGB head forward; native bilinear scores exactly reconstructible",
            "undefined_metrics": "JSON null / CSV empty for zero denominator",
            "statistics": "sum TP/FP/FN after within-pair random-seed averaging; 2000 paired original-pair bootstrap, seed20261028",
            "official_test_pixels_read": 0}
    path = out / "run_spec.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != spec:
        raise ValueError("refusing to mix changed producer, evaluator source, or frozen model under one output run")
    save_json(path, spec)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, snapshot / name.replace("/", "__"))
    return producer, items, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/second_pixels/fixed_v1")
    parser.add_argument("--out", default="results/second_pixel_tasks/fixed_v1")
    parser.add_argument("--limit-pairs", type=int, default=0, choices=(0, 8))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll wait must be 1..60 seconds")
    source, out = safe_path(args.source), safe_path(args.out)
    if not out.is_relative_to((ROOT / "results/second_pixel_tasks").resolve()):
        raise ValueError("task output must stay under the owned second_pixel_tasks directory")
    out.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous completion assertion before examining this invocation.
    save_json(out / "progress.json", {"schema": SCHEMA, "status": "in_progress", "expected_records": EXPECTED,
               "current_records": 0, "utc": now(), "official_test_pixels_read": 0})
    producer, items, spec = prepare(source, out)
    encoder = FrozenEncoder(ROOT, ENCODER_CONFIG)
    inferences = {h: ActualRGBInference(SecondGroundEvaluator(ROOT, h, encoder=encoder)) for h in (5, 3, 4)}
    results, sanity, completed = [], [], set()
    while True:
        fields, pair_groups, source_info = read_source(source / "records.csv", producer)
        selected = pair_groups[:args.limit_pairs] if args.limit_pairs else pair_groups
        for pair_id, rows in selected:
            if pair_id in completed:
                continue
            item = items[pair_id]
            prior, initial = initialize_pair(source, item)
            before = Image.fromarray(prior.rgb)
            targets = {h: offline_target(item, h) for h in inferences}
            sanity.extend(sanity_pair(source, item, prior, inferences, targets, out))
            for row in rows:
                if row["before_sha256"] != item["files"]["before"]["sha256"] or row["after_sha256"] != item["files"]["after"]["sha256"]:
                    raise ValueError("production source image identity changed")
                rgb = replay_record(row, prior, initial, producer)
                results.append(evaluate_record(row, rgb, before, initial,
                               targets[int(row["heldout"])], inferences[int(row["heldout"])], out, spec))
            completed.add(pair_id)
            save_csv(out / "records.csv", results, fields + list(TASK_FIELDS))
            save_csv(out / "sanity.csv", sanity)
            aggregate(results, out, producer["config"])
            save_json(out / "progress.json", {"schema": SCHEMA, "status": "in_progress", "expected_records": EXPECTED,
                       "current_records": len(results), "completed_pairs": len(completed), "expected_pairs": 64,
                       "completed_pair_ids": [p for p in items if p in completed], "source_snapshot": source_info,
                       "records_csv_sha256": sha_file(out / "records.csv"), "official_test_pixels_read": 0, "utc": now()})
            print(f"SECOND actual receiver task {len(completed)}/64 pairs; {len(results)}/{EXPECTED} rows; all stream/ACK/RGB hashes match", flush=True)
            if len(completed) == 8:
                aggregate(results, out, producer["config"], prefix="pilot8_")
                shutil.copyfile(out / "records.csv", out / "pilot8_records.csv")
                print("PILOT8_COMPLETE actual decoded RGB tasks:864 rows; immutable heads/thresholds; see pilot8_comparisons.csv", flush=True)
        if len(results) == EXPECTED:
            _, final_groups, final_source = read_source(source / "records.csv", producer)
            final_rows = [r for _, group in final_groups for r in group]
            expected_map = {row_key(r): digest(canonical(r).encode()) for r in final_rows}
            actual_map = {row_key(r): r["source_record_sha256"] for r in results}
            if len(final_groups) != 64 or len(final_rows) != EXPECTED or expected_map != actual_map:
                raise ValueError("6912 outputs do not match current source rows one-to-one")
            if sha_file(source / "run_spec.json") != spec["producer_run_spec_sha256"]:
                raise ValueError("producer spec changed during receiver evaluation")
            for name, checksum in spec["evaluation_source_sha256"].items():
                if sha_file(ROOT / name) != checksum:
                    raise ValueError("evaluation source changed while this fixed run was active")
            save_json(out / "progress.json", {"schema": SCHEMA, "status": "complete", "expected_records": EXPECTED,
                       "current_records": len(results), "completed_pairs": 64, "expected_pairs": 64,
                       "source_snapshot": final_source, "records_csv_sha256": sha_file(out / "records.csv"),
                       "source_output_mapping_sha256": digest(canonical(sorted((str(k), v) for k, v in actual_map.items())).encode()),
                       "all_stream_ACK_receiver_hashes_match": True, "all_source_rows_matched": True,
                       "ground_models": spec["ground_models"], "official_test_pixels_read": 0, "utc": now()})
            print("COMPLETE:6912 actual bitstream receiver tasks, one-to-one source row/hash match", flush=True)
            break
        if args.limit_pairs or not args.follow:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
