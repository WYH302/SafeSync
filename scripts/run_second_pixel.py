"""Precommitted SECOND pixel-codec comparison; no official-test image access.

All routers have the same losslessly initialized historical RGB and quality
gate. Labels are used only after routing, for received-image evaluation.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, __version__ as pillow_version, features as pillow_features
from scipy.ndimage import label as connected_labels
from scipy.stats import rankdata

from safesync.data import read_pair
from safesync.reference import exact_mse, restore_coarse
from safesync.pixel_wire import (
    ACK, DEFERRED, STABLE, UPDATED, PixelState, TransactionReceiver,
    TransactionSender, encode_update,
)
from scripts.run_pixel_sequences import choose_packet, full_packet

CONFIG = {
    "protocol": "SECOND-pixel-SP03-fixed-v1", "heldouts": [5, 3, 4],
    "budgets_total_bytes": [4096, 8192, 16384, 32768],
    "block": 64, "coarse_side": 4, "roi_codec": "jpeg", "roi_quality": 60,
    "quality_gate_psnr": 25, "pixel_rank_weight": .75, "other_rank_weight": .25,
    "random_seeds": [1729, 2718, 31415], "coverage_psnr": [25, 30, 35],
    "rank_ties": "average; ranks divided by 64; rank all ROIs then zero non-gated ROIs",
    "score_pool": "16x16 token grid -> max over each 2x2 -> 8x8 native 64px ROI grid",
    "methods": ["feature_change", "pixel_change", "pixel_random", "pixel_unknown",
                "unknown_only", "full_jpeg", "full_jpeg2000"],
    "history": "same full losslessly decoded before RGB available to every sender",
    "budget": "current SP03 packet plus actual ACK; initialization separately counted; no padding",
    "known_mask": "both semantic labels in 1..6; neither label heldout",
    "unknown_mask": "current label heldout AND binary change AND both semantic labels in 1..6",
    "object_proxy": "8-connected components of unknown mask, no size filtering; not instance GT",
    "evaluation_split": "64 previously quality-inspected development-validation pairs; not final test",
    "bootstrap_replicates": 2000, "bootstrap_seed": 20261028,
}
SOURCE_FILES = [
    "scripts/run_second_pixel.py", "scripts/run_pixel_sequences.py",
    "safesync/pixel_wire.py", "safesync/reference.py", "safesync/data.py",
    "scripts/diagnose_unknown_scaled.py", "safesync/open_set.py", "safesync/scoring.py",
]
ACK_BYTES = ACK.size + 16


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, records):
    if not records:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def pool_scores(values):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (256,) or not np.isfinite(array).all():
        raise ValueError("Expected 256 finite frozen token scores")
    return array.reshape(8, 2, 8, 2).max(axis=(1, 3)).reshape(64)


def ranking_scores(method, pixel_mse, feature_change, unknown, gate, seed, pair_id):
    rank = lambda values: rankdata(values, method="average") / len(values)
    if method == "feature_change":
        score = rank(feature_change)
    elif method == "pixel_change":
        score = rank(pixel_mse)
    elif method == "pixel_unknown":
        score = .75 * rank(pixel_mse) + .25 * rank(unknown)
    elif method == "unknown_only":
        score = rank(unknown)
    elif method == "pixel_random":
        identity = f"SECOND-pixel-random:{seed}:{pair_id}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
        score = .75 * rank(pixel_mse) + .25 * rank(rng.random(len(pixel_mse)))
    else:
        raise ValueError(method)
    # A tied or zero novelty score cannot certify physical stability.
    score = np.where(gate, score, 0.0)
    if np.any(score[gate] <= 0) or np.any(score[~gate] != 0):
        raise AssertionError("Ranking was confused with the physical quality gate")
    return score


def evaluation_masks(old, current, changed, heldout):
    explicit = (old >= 1) & (old <= 6) & (current >= 1) & (current <= 6)
    unknown = (current == heldout) & changed & explicit
    known = explicit & (old != heldout) & (current != heldout)
    return known, unknown


def region_sums(array, block=64):
    height, width = array.shape
    return array.reshape(height // block, block, width // block, block).sum((1, 3)).reshape(-1)


def image_metrics(current, restored, known, unknown):
    error = np.mean(((current.astype(np.float64) - restored.astype(np.float64)) / 255) ** 2, axis=2)
    known_count, unknown_count = int(known.sum()), int(unknown.sum())
    result = {
        "global_error_sum": float(error.sum()), "global_pixels": int(error.size),
        "global_mse": float(error.mean()), "known_pixels": known_count,
        "known_error_sum": float(error[known].sum()),
        "known_mse": float(error[known].mean()) if known_count else None,
        "unknown_pixels": unknown_count, "unknown_error_sum": float(error[unknown].sum()),
        "unknown_mse": float(error[unknown].mean()) if unknown_count else None,
    }
    counts, errors = region_sums(unknown), region_sums(error * unknown)
    masked_mse = np.divide(errors, counts, out=np.full(64, np.inf), where=counts > 0)
    for threshold in CONFIG["coverage_psnr"]:
        retained = int(counts[masked_mse <= 10 ** (-threshold / 10)].sum())
        result[f"unknown_retained_pixels_{threshold}"] = retained
        result[f"unknown_coverage_{threshold}"] = retained / unknown_count if unknown_count else None
    return result


def fresh_receiver(prior):
    return TransactionReceiver(PixelState(prior.rgb.copy(), prior.region, prior.block,
                                         prior.version, prior.observed_at, prior.statuses.copy()))


def initialize(before, pair_id, folder):
    region = hashlib.sha256(("SafeSync-SECOND-pixel:" + pair_id).encode()).digest()[:16]
    receiver = TransactionReceiver(PixelState(np.zeros_like(before), region))
    sender = TransactionSender(receiver.state.version, receiver.state.digest())
    wire, costs = sender.propose(
        shape=before.shape[:2], block=64, region=region, observed_at=0,
        current_rgb=before, indices=np.arange(64), statuses=np.full(64, UPDATED, np.uint8), codec="png")
    ack, _ = receiver.receive(wire)
    sender.acknowledge(ack)
    if not np.array_equal(receiver.state.rgb, before) or len(wire) != sum(costs.values()):
        raise AssertionError("Initialization is not actual lossless PNG reception")
    (folder / "initial.bin").write_bytes(wire)
    (folder / "initial.ack").write_bytes(ack)
    save_json(folder / "initial_cost.json", {"packet_bytes": len(wire), "ack_bytes": len(ack),
              "components": costs, "packet_sha256": hashlib.sha256(wire).hexdigest(),
              "ack_sha256": hashlib.sha256(ack).hexdigest()})
    return receiver.state, len(wire), len(ack)


def transact(prior, current, method, budget, score, gate):
    receiver = fresh_receiver(prior)
    sender = TransactionSender(prior.version, prior.digest())
    start = time.perf_counter()
    if method in ("full_jpeg", "full_jpeg2000"):
        codec = "jpeg2000" if method == "full_jpeg2000" else "jpeg"
        wire, costs, selected, trials = full_packet(
            sender, current, prior.region, 1, budget - ACK_BYTES, codec)
    else:
        wire, costs, selected, trials = choose_packet(
            sender, current, prior.region, 1, score, 0.0, budget - ACK_BYTES,
            coarse_side=4, codec="jpeg", quality=60)
    encoding_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    receiver.expect(1)
    ack, _ = receiver.receive(wire)
    sender.acknowledge(ack)
    decoding_ms = (time.perf_counter() - start) * 1000
    if len(ack) != ACK_BYTES or len(wire) + len(ack) > budget or sum(costs.values()) != len(wire):
        raise AssertionError("Packet + ACK actual-byte budget/accounting failure")
    if sender.digest != receiver.state.digest():
        raise AssertionError("Acknowledged sender/receiver state mismatch")
    if method not in ("full_jpeg", "full_jpeg2000"):
        if not np.array_equal(receiver.state.statuses == STABLE, ~gate):
            raise AssertionError("Physical gate/status mismatch")
    prediction = receiver.state.rgb
    restored_hash = hashlib.sha256(prediction.tobytes()).hexdigest()
    # Replay uses only the decoded initialization state and wire, never current RGB.
    replay = fresh_receiver(prior)
    replay_ack, _ = replay.receive(wire)
    replay_hash = hashlib.sha256(replay.state.rgb.tobytes()).hexdigest()
    duplicate_ack, duplicate_updates = replay.receive(wire)
    if (replay_hash != restored_hash or replay_ack != ack or duplicate_ack != ack
            or duplicate_updates or replay.state.digest() != receiver.state.digest()):
        raise AssertionError("Saved-transaction replay/idempotence failure")
    details = {
        "down_bytes": len(wire), "ack_bytes": len(ack), "total_bytes": len(wire) + len(ack),
        "updated_rois": int(selected), "stable_rois": int((receiver.state.statuses == STABLE).sum()),
        "deferred_rois": int((receiver.state.statuses == DEFERRED).sum()),
        "physical_gate_rois": int(gate.sum()), "encoding_trials": trials,
        "encoding_ms": encoding_ms, "decoding_ack_ms": decoding_ms,
        "receiver_sha256": restored_hash, "receiver_state_digest": receiver.state.digest().hex(),
        "packet_sha256": hashlib.sha256(wire).hexdigest(), "ack_sha256": hashlib.sha256(ack).hexdigest(),
        "replay_passed": True, **{f"bytes_{key}": value for key, value in costs.items()},
    }
    return wire, ack, prediction.copy(), details


def sanity_images(prior, current, folder, gate):
    full, costs = encode_update(
        shape=current.shape[:2], block=64, region=prior.region, base_version=prior.version,
        base_digest=prior.digest(), observed_at=1, current_rgb=current,
        indices=np.arange(64), statuses=np.full(64, UPDATED, np.uint8), codec="png")
    receiver = fresh_receiver(prior)
    full_ack, _ = receiver.receive(full)
    if not np.array_equal(receiver.state.rgb, current):
        raise AssertionError("Full PNG sanity does not reproduce current pixels")
    (folder / "sanity_full_png.bin").write_bytes(full)
    (folder / "sanity_full_png.ack").write_bytes(full_ack)
    coarse, _ = encode_update(
        shape=current.shape[:2], block=64, region=prior.region, base_version=prior.version,
        base_digest=prior.digest(), observed_at=1, current_rgb=current,
        indices=[], statuses=np.where(gate, DEFERRED, STABLE).astype(np.uint8),
        codec="jpeg", quality=60, coarse_side=4)
    coarse_receiver = fresh_receiver(prior)
    coarse_ack, _ = coarse_receiver.receive(coarse)
    if not np.array_equal(coarse_receiver.state.rgb, restore_coarse(prior.rgb, current)):
        raise AssertionError("Sender quality gate differs from actual coarse decoding")
    (folder / "sanity_coarse_only.bin").write_bytes(coarse)
    (folder / "sanity_coarse_only.ack").write_bytes(coarse_ack)
    return {"empty": prior.rgb, "source_old": prior.rgb, "full_png": receiver.state.rgb,
            "coarse_only": coarse_receiver.state.rgb}


def aggregate(out, records, completed, expected):
    numeric = ["budget", "heldout", "seed", "unknown_pixels", "known_pixels", "global_pixels",
               "unknown_components_8", "total_bytes", "down_bytes", "ack_bytes", "updated_rois",
               "known_error_sum", "unknown_error_sum", "global_error_sum", "encoding_ms", "decoding_ack_ms"]
    numeric += [f"unknown_retained_pixels_{q}" for q in CONFIG["coverage_psnr"]]
    rows = []
    for record in records:
        row = dict(record)
        for key in numeric:
            row[key] = float(row[key])
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], int(row["seed"]))].append(row)
    summaries = []
    for (heldout, budget, method, seed), group in sorted(grouped.items()):
        positive = [row for row in group if row["unknown_pixels"] > 0]
        unknown_pixels = sum(row["unknown_pixels"] for row in group)
        known_pixels = sum(row["known_pixels"] for row in group)
        summary = dict(heldout=heldout, budget=budget, method=method, seed=seed, pairs=len(group),
                       positive_pairs=len(positive), unknown_components_8=int(sum(row["unknown_components_8"] for row in group)),
                       unknown_pixels=int(unknown_pixels), known_pixels=int(known_pixels),
                       mean_total_bytes=float(np.mean([r["total_bytes"] for r in group])),
                       mean_down_bytes=float(np.mean([r["down_bytes"] for r in group])),
                       mean_updated_rois=float(np.mean([r["updated_rois"] for r in group])),
                       known_mse=sum(r["known_error_sum"] for r in group) / known_pixels if known_pixels else None,
                       unknown_mse=sum(r["unknown_error_sum"] for r in group) / unknown_pixels if unknown_pixels else None,
                       global_mse=sum(r["global_error_sum"] for r in group) / sum(r["global_pixels"] for r in group),
                       mean_encoding_ms=float(np.mean([r["encoding_ms"] for r in group])),
                       mean_decoding_ack_ms=float(np.mean([r["decoding_ack_ms"] for r in group])))
        for threshold in CONFIG["coverage_psnr"]:
            key = f"unknown_retained_pixels_{threshold}"
            summary[f"unknown_coverage_{threshold}_micro"] = sum(r[key] for r in group) / unknown_pixels if unknown_pixels else None
            summary[f"unknown_coverage_{threshold}_macro_positive"] = float(np.mean([r[key] / r["unknown_pixels"] for r in positive])) if positive else None
        summaries.append(summary)
    save_csv(out / "summary.csv", summaries)
    # Pair random seeds first; they are repeated routings, not independent images.
    paired = defaultdict(list)
    for row in rows:
        paired[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    means = {key: {field: float(np.mean([r[field] for r in group])) for field in numeric}
             for key, group in paired.items()}
    comparisons = []
    for heldout in CONFIG["heldouts"]:
        for budget in CONFIG["budgets_total_bytes"]:
            ids = sorted(k[3] for k in means if k[:3] == (heldout, budget, "pixel_unknown"))
            for baseline in ("pixel_change", "pixel_random", "unknown_only", "feature_change", "full_jpeg", "full_jpeg2000"):
                new = [means[(heldout, budget, "pixel_unknown", key)] for key in ids]
                old = [means[(heldout, budget, baseline, key)] for key in ids]
                positive_ids = [i for i, r in enumerate(new) if r["unknown_pixels"] > 0]
                denominator = np.array([new[i]["unknown_pixels"] for i in positive_ids])
                difference = np.array([new[i]["unknown_retained_pixels_30"] - old[i]["unknown_retained_pixels_30"] for i in positive_ids])
                delta = float(difference.sum() / denominator.sum()) if len(denominator) else None
                low = high = None
                if len(denominator):
                    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
                    sample = rng.integers(0, len(denominator), (CONFIG["bootstrap_replicates"], len(denominator)))
                    bootstrap = difference[sample].sum(1) / denominator[sample].sum(1)
                    low, high = map(float, np.quantile(bootstrap, [.025, .975]))
                known_n = sum(r["known_pixels"] for r in new)
                comparisons.append(dict(heldout=heldout, budget=budget, method="pixel_unknown", baseline=baseline,
                    pairs=len(ids), positive_pairs=len(positive_ids), delta_unknown_coverage_30=delta,
                    paired_bootstrap_low=low, paired_bootstrap_high=high,
                    delta_known_mse=(sum(a["known_error_sum"]-b["known_error_sum"] for a,b in zip(new,old))/known_n) if known_n else None,
                    delta_mean_total_bytes=float(np.mean([a["total_bytes"]-b["total_bytes"] for a,b in zip(new,old)]))))
    save_csv(out / "comparisons.csv", comparisons)
    save_json(out / "progress.json", {"completed_pairs": len(completed), "expected_pairs": expected,
              "complete": len(completed) == expected, "records": len(records),
              "completed_pair_ids": completed, "official_test_pixels_read": 0})
    return summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8, choices=(0, 8), help="8=initial interface run; 0=all fixed 64")
    parser.add_argument("--out", default="results/second_pixels/fixed_v1")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / "results/second_pixels").resolve()):
        raise ValueError("Output must stay in the owned second_pixels directory")
    out.mkdir(parents=True, exist_ok=True)
    if not pillow_features.check("jpg_2000"):
        raise RuntimeError("JPEG2000 unavailable; cannot silently omit the fixed baseline")
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = [pair for pair in manifest["pairs"] if pair["split"] == "validation"]
    if len(pairs) != 64 or any(pair["source_split"] != "train" for pair in manifest["pairs"]):
        raise ValueError("Expected fixed 128/64 official-training development split")
    metadata_path = ROOT / "cache/second_scaled/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != sha_file(manifest_path):
        raise ValueError("Frozen features refer to another manifest")
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as data:
        if data["ids"].tolist() != [pair["id"] for pair in manifest["pairs"]]:
            raise ValueError("Frozen feature IDs differ from fixed data")
    sources = {name: sha_file(ROOT / name) for name in SOURCE_FILES}
    models = {str(h): sha_file(ROOT / f"results/unknown_scaled/known_{h}.npz") for h in CONFIG["heldouts"]}
    score_files = {f"{h}:{p['id']}": sha_file(ROOT / f"results/unknown_scaled/scores_{h}_{p['id']}.npz")
                   for h in CONFIG["heldouts"] for p in pairs}
    spec = dict(config=CONFIG, config_sha256=canonical_hash(CONFIG), source_sha256=sources,
                source_bundle_sha256=canonical_hash(sources), known_model_sha256=models,
                score_files_sha256=score_files, manifest_sha256=sha_file(manifest_path),
                feature_cache_sha256=sha_file(ROOT / "cache/second_scaled/features.npz"),
                feature_metadata_sha256=sha_file(metadata_path), encoder=metadata["encoder"]["identity"],
                validation_pair_ids=[p["id"] for p in pairs], initialization="actual full PNG SP03 packet + ACK, separately counted",
                environment={"python": platform.python_version(), "numpy": np.__version__, "Pillow": pillow_version,
                             "JPEG2000": pillow_features.version_codec("jpg_2000")},
                statistics_unit="original image pair; 3 random seeds averaged within pair for comparisons",
                quality_metric="unknown-pixel weighted ROI masked-PSNR coverage; not a per-pixel guarantee",
                timing_scope="cached scores; encoding search and receive/ACK only; identical generic transactions reused across folds",
                official_test_pixels_read=0)
    spec_path = out / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Frozen run inputs/source changed; do not mix runs")
    save_json(spec_path, spec)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in SOURCE_FILES:
        shutil.copyfile(ROOT / source, snapshot / source.replace("/", "__"))
    records, sanity_records, completed = [], [], []
    if (out / "records.csv").exists():
        records = list(csv.DictReader((out / "records.csv").open(encoding="utf-8")))
        sanity_records = list(csv.DictReader((out / "sanity.csv").open(encoding="utf-8")))
        completed = json.loads((out / "progress.json").read_text())["completed_pair_ids"]
    selected = pairs[:args.limit] if args.limit else pairs
    for item in selected:
        if item["id"] in completed:
            continue
        if any(sha_file(ROOT / name) != digest for name, digest in sources.items()):
            raise RuntimeError("Shared source changed during the fixed run; stopping")
        before_pil, after_pil, old_label, label, changed = read_pair(ROOT, item)
        before, current = np.asarray(before_pil), np.asarray(after_pil)
        if before.shape != (512, 512, 3) or current.shape != before.shape:
            raise ValueError("Unexpected native SECOND coordinates")
        folder = out / "pairs" / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        prior, initial_bytes, initial_ack = initialize(before, item["id"], folder)
        pixel_mse = exact_mse(current, restore_coarse(prior.rgb, current))
        gate = pixel_mse > 10 ** (-2.5)
        sanity = sanity_images(prior, current, folder, gate)
        generic = {}
        change_reference = None
        for heldout in CONFIG["heldouts"]:
            score_path = ROOT / f"results/unknown_scaled/scores_{heldout}_{item['id']}.npz"
            if sha_file(score_path) != score_files[f"{heldout}:{item['id']}"]:
                raise RuntimeError("Frozen novelty scores changed")
            with np.load(score_path, allow_pickle=False) as data:
                feature_change, unknown = pool_scores(data["change"]), pool_scores(data["unknown"])
            if change_reference is not None and not np.array_equal(change_reference, feature_change):
                raise ValueError("Feature-change baseline differs between heldout folds")
            change_reference = feature_change
            known_mask, unknown_mask = evaluation_masks(old_label, label, changed, heldout)
            components = int(connected_labels(unknown_mask, structure=np.ones((3, 3)))[1])
            for case, prediction in sanity.items():
                sanity_records.append(dict(pair_id=item["id"], heldout=heldout, case=case,
                    unknown_components_8=components, **image_metrics(current, prediction, known_mask, unknown_mask)))
            for method in CONFIG["methods"]:
                seeds = CONFIG["random_seeds"] if method == "pixel_random" else [0]
                for seed in seeds:
                    score = None if method.startswith("full_jpeg") else ranking_scores(
                        method, pixel_mse, feature_change, unknown, gate, seed, item["id"])
                    for budget in CONFIG["budgets_total_bytes"]:
                        key = (method, seed, budget)
                        shared = method not in ("pixel_unknown", "unknown_only")
                        if shared and key in generic:
                            wire, ack, prediction, details = generic[key]
                        else:
                            wire, ack, prediction, details = transact(prior, current, method, budget, score, gate)
                            if shared:
                                generic[key] = wire, ack, prediction, details
                        variant = method + (f"_{seed}" if seed else "")
                        transaction = folder / f"heldout_{heldout}" / f"{variant}_B{budget}"
                        transaction.mkdir(parents=True, exist_ok=True)
                        (transaction / "current.bin").write_bytes(wire)
                        (transaction / "ack.bin").write_bytes(ack)
                        records.append(dict(pair_id=item["id"], heldout=heldout, method=method, seed=seed,
                            budget=budget, packet_path=(transaction / "current.bin").relative_to(ROOT).as_posix(),
                            ack_path=(transaction / "ack.bin").relative_to(ROOT).as_posix(),
                            initial_down_bytes=initial_bytes, initial_ack_bytes=initial_ack,
                            before_sha256=item["files"]["before"]["sha256"], after_sha256=item["files"]["after"]["sha256"],
                            config_sha256=spec["config_sha256"], source_bundle_sha256=spec["source_bundle_sha256"],
                            backbone_weight_sha256=spec["encoder"]["weight_sha256"], known_model_sha256=models[str(heldout)],
                            score_file_sha256=score_files[f"{heldout}:{item['id']}"],
                            router_uses_unknown_model=method in ("pixel_unknown", "unknown_only"),
                            unknown_components_8=components, **details,
                            **image_metrics(current, prediction, known_mask, unknown_mask)))
        completed.append(item["id"])
        save_csv(out / "records.csv", records)
        save_csv(out / "sanity.csv", sanity_records)
        aggregate(out, records, completed, len(pairs))
        print(f"{len(completed)}/64 pair={item['id']} transactions={len(records)} all bytes <= caps; replay pass", flush=True)
    if any(sha_file(ROOT / name) != digest for name, digest in sources.items()):
        raise RuntimeError("Shared source changed at completion")
    summaries, comparisons = aggregate(out, records, completed, len(pairs))
    if len(completed) == 8 and args.limit == 8:
        for name in ("records.csv", "summary.csv", "comparisons.csv", "sanity.csv", "progress.json"):
            shutil.copyfile(out / name, out / ("pilot8_" + name))
    print(json.dumps({"completed_pairs": len(completed), "records": len(records),
        "config_sha256": spec["config_sha256"], "source_bundle_sha256": spec["source_bundle_sha256"],
        "pixel_unknown_vs_pixel_change": [r for r in comparisons if r["baseline"] == "pixel_change"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
