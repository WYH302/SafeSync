"""Factorial service/codec repair on the original 64 development pairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
from safesync import task_evidence_wire as wire
from safesync import full_image_codecs as whole
from safesync.product_quantizer import ProductQuantizer
from safesync.data import read_pair
from scripts.run_second_feature_mix import load_frozen

OUT = ROOT / "results/task_evidence/development_v1"
CAPS = (4096, 8192, 16384, 32768)
SERVICES = ("labels", "pq")
CODECS = (("jpeg", 60), ("webp", 70))
SEEDS = (1729, 2718, 31415)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", "utf-8")
    temporary.replace(path)


def specification():
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    source_names = ("scripts/run_task_evidence_development.py", "safesync/task_evidence_wire.py", "safesync/full_image_codecs.py", "safesync/product_quantizer.py")
    proposed = dict(schema="task-evidence-factorial-v1", pair_ids=[p["id"] for p in manifest["pairs"] if p["split"] == "validation"],
        services=SERVICES, evidence_codecs=CODECS, routes=["change", "mix", "random"], seeds=SEEDS, caps=CAPS,
        mixing=[.75, .25], full_controls=["jpeg420", "webp"], full_scales=[1, .75, .5, .375, .25],
        source_sha256={p: sha(ROOT / p) for p in source_names},
        frozen_feature_sha256=sha(ROOT / "cache/second_scaled/features.npz"),
        novelty_sha256=sha(ROOT / "results/unknown_recovery/scores_v1/complete.json"),
        books={str(h): sha(ROOT / f"results/product_quantizer/known_only_v1/h{h}_m8.npz") for h in (5, 3, 4)},
        prototypes={str(h): sha(ROOT / f"results/unknown_scaled/known_{h}.npz") for h in (5, 3, 4)},
        known_task="Same cosine prototype head; exact task-label service versus quantized-feature service",
        protocol="TE01, actual packet+60-byte ACK, exact reconstructed task-state XOR/absolute minimum, version/model/reference binding",
        attribution="Two service representations crossed with two fixed atlas codecs; same routes; full-image controls receive the same service",
        development_scope="Original 64 already inspected development pairs; earlier reserved-test outcomes diagnosed failure, this set selects repairs",
        timing="Pure codec payloads cached across budgets/services within source image; cached search timing is not deployment latency",
        expected_records=18816)
    proposed = json.loads(json.dumps(proposed))
    path = OUT / "run_spec.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != proposed:
            raise ValueError("Changed task/evidence repair")
    else:
        save(path, proposed)
        for source in source_names:
            destination = OUT / "source_snapshot" / source.replace("/", "__")
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes((ROOT / source).read_bytes())
    return proposed, manifest


def fit_atlas(prior, labels, current, ranking, codec, quality, cap, cache):
    order = np.argsort(-ranking, kind="stable")
    identity = hashlib.sha256(order.tobytes()).hexdigest()

    def make(count):
        ids = order[:count]
        key = codec, quality, identity, count
        if key not in cache:
            cache[key] = wire.encode_atlas(current, ids, codec=codec, quality=quality)
        return wire.encode(prior, labels, observation=1, payload=cache[key], indices=ids)

    best, costs = make(0)
    if len(best) + wire.ACK_BYTES > cap:
        raise ValueError("Mandatory task state exceeds the cap")
    selected, low, high = 0, 1, len(order)
    while low <= high:
        middle = (low + high) // 2
        packet, components = make(middle)
        if len(packet) + wire.ACK_BYTES <= cap:
            best, costs, selected = packet, components, middle
            low = middle + 1
        else:
            high = middle - 1
    for count in range(selected + 1, min(selected + 3, len(order)) + 1):
        packet, components = make(count)
        if len(packet) + wire.ACK_BYTES <= cap:
            best, costs, selected = packet, components, count
    return best, costs, selected


def fit_full(prior, labels, current, codec, cap):
    overhead, _ = wire.encode(prior, labels, observation=1)
    available = cap - len(overhead) - wire.ACK_BYTES
    # Only the standalone codec payload is nested in TE01. IF06's temporary
    # envelope is used by the unchanged sender-side search, then removed.
    packet, metadata = whole.fit(current, codec=codec, budget=available + whole.HEAD.size + 16,
        region=prior.region, base_version=prior.version, base_digest=bytes(16), observation=1)
    payload = packet[whole.HEAD.size:-16]
    if payload:
        encoded, costs = wire.encode(prior, labels, observation=1, payload=payload, image_mode=wire.FULL)
    else:
        encoded, costs = wire.encode(prior, labels, observation=1)
    return encoded, costs, metadata


def process(pair, features, emergence, helper, books, prototypes):
    before_pil, after_pil, old, label, changed = read_pair(ROOT, pair)
    before, current = np.asarray(before_pil), np.asarray(after_pil)
    change = 1 - np.sum(features[0] * features[1], axis=1)
    rank_change = rankdata(helper["pool_scores"](change)) / 64
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    png_stream = io.BytesIO()
    Image.fromarray(before).save(png_stream, format="PNG")
    bootstrap_payload = png_stream.getvalue()
    atlas_cache, whole_cache = {}, {}
    original_encode = whole.encode_payload

    def cached_full(image, codec, quality):
        key = image.size, codec, quality
        if key not in whole_cache:
            whole_cache[key] = original_encode(image, codec, quality)
        return whole_cache[key]

    whole.encode_payload = cached_full
    rows = []
    for heldout in (5, 3, 4):
        mean, classes = prototypes[heldout]
        known, unknown = helper["evaluation_masks"](old, label, changed, heldout)
        rank_unknown = rankdata(helper["pool_scores"](emergence[heldout])) / 64
        for service in SERVICES:
            book = books[heldout]
            if service == "labels":
                old_state = classes[np.argmax(features[0] @ mean.T, axis=1)].astype(np.uint8).reshape(16, 16)
                current_state = classes[np.argmax(features[1] @ mean.T, axis=1)].astype(np.uint8).reshape(16, 16)
                model = hashlib.sha256(mean.tobytes() + classes.tobytes() + b"task-labels-v1").digest()[:16]
            else:
                old_state, current_state = book.encode(features[0]), book.encode(features[1])
                model = hashlib.sha256(book.identity + mean.tobytes() + classes.tobytes()).digest()[:16]
            region = hashlib.sha256(("task-evidence:" + pair["id"]).encode()).digest()[:16]
            blank = wire.TaskEvidenceState(np.zeros_like(before), np.zeros_like(old_state), region, model)
            boot, boot_costs = wire.encode(blank, old_state, observation=0, payload=bootstrap_payload, image_mode=wire.FULL)
            prior, boot_ack = wire.receive(blank, boot)
            if not np.array_equal(prior.rgb, before) or not np.array_equal(prior.labels, old_state):
                raise AssertionError("Task/evidence bootstrap not exact")
            target = folder / f"h{heldout}_{service}"
            target.mkdir(exist_ok=True)
            (target / "bootstrap.bin").write_bytes(boot)
            (target / "bootstrap.ack").write_bytes(boot_ack)
            expected_confusion = None
            configurations = [(f"{codec}_{route}", codec, quality, route) for codec, quality in CODECS for route in ("change", "mix", "random")]
            configurations += [("full_" + codec, codec, 0, "full") for codec in ("jpeg420", "webp")]
            configurations += [("only", "none", 0, "none")]
            for name, codec, quality, route in configurations:
                for seed in (SEEDS if route == "random" else (0,)):
                    if route == "change":
                        ranking = rank_change
                    elif route == "mix":
                        ranking = .75 * rank_change + .25 * rank_unknown
                    elif route == "random":
                        key = hashlib.sha256(f"task-evidence-random:{pair['id']}:{seed}".encode()).digest()[:8]
                        ranking = np.random.default_rng(int.from_bytes(key, "little")).random(64) + .01
                    for cap in (CAPS if route != "none" else (0,)):
                        started = time.perf_counter()
                        if route == "full":
                            packet, costs, codec_meta = fit_full(prior, current_state, current, codec, cap)
                            selected = 64 if costs["image"] else 0
                        elif route == "none":
                            packet, costs = wire.encode(prior, current_state, observation=1)
                            selected, codec_meta = 0, {}
                        else:
                            packet, costs, selected = fit_atlas(prior, current_state, current, ranking, codec, quality, cap, atlas_cache)
                            codec_meta = dict(quality=quality, selection="largest tested fitting rank prefix; binary search and three-neighbor probe")
                        receiver = wire.TaskEvidenceReceiver(prior)
                        receiver.expect(1)
                        ack, _ = receiver.receive(packet)
                        state = receiver.state
                        repeat_ack, applied = receiver.receive(packet)
                        replay, replay_ack = wire.receive(prior, packet)
                        if (applied or ack != repeat_ack or ack != replay_ack or state.digest() != replay.digest()
                                or not np.array_equal(state.labels, current_state) or (cap and len(packet) + len(ack) > cap)):
                            raise AssertionError("Task/evidence replay, state service, or actual byte cap")
                        if service == "labels":
                            predicted = state.labels
                        else:
                            predicted = classes[np.argmax(book.decode(state.labels) @ mean.T, axis=1)].reshape(16, 16)
                        predicted = predicted.repeat(32, 0).repeat(32, 1)
                        matrix = np.bincount((label[known].astype(int) - 1) * 6 + predicted[known].astype(int) - 1, minlength=36).reshape(6, 6)
                        if expected_confusion is not None and not np.array_equal(matrix, expected_confusion):
                            raise AssertionError("Routing or codec altered the fixed task service")
                        expected_confusion = matrix
                        path = target / f"{name}_s{seed}_b{cap}.bin"
                        path.write_bytes(packet)
                        path.with_suffix(".ack").write_bytes(ack)
                        rows.append(dict(pair_id=pair["id"], heldout=heldout, service=service, method=service + "_" + name,
                            seed=seed, budget=cap, down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet) + len(ack),
                            bootstrap_down_bytes=len(boot), bootstrap_ack_bytes=len(boot_ack), selected_rois=selected,
                            packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                            receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),
                            receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
                            known_confusion=json.dumps(matrix.tolist()), components=json.dumps(costs, sort_keys=True),
                            codec_metadata=json.dumps(codec_meta, sort_keys=True), cached_pipeline_ms=(time.perf_counter() - started)*1000,
                            **helper["image_metrics"](current, state.rgb, known, unknown)))
    whole.encode_payload = original_encode
    if len(rows) != 294:
        raise AssertionError("Incomplete task/evidence factorial for one pair")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    spec, manifest = specification()
    if args.freeze_only:
        print(json.dumps(dict(status="locked_development", pairs=64, expected_records=18816)), flush=True)
        return
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    if args.finalize_only:
        rows = []
        for pair_id in spec["pair_ids"]:
            path = OUT / "pairs" / pair_id / "records.json"
            receipt = json.loads(path.with_name("complete.json").read_text("utf-8"))
            if receipt["spec_sha256"] != sha(OUT / "run_spec.json") or sha(path) != receipt["records_sha256"]:
                raise ValueError("Changed factorial pair output")
            rows.extend(json.loads(path.read_text("utf-8")))
        if len(rows) != spec["expected_records"]:
            raise AssertionError("Incomplete factorial matrix")
        helper["save_csv"](OUT / "records.csv", rows)
        save(OUT / "complete.json", dict(status="complete", records=len(rows), expected=18816,
            spec_sha256=sha(OUT / "run_spec.json"), records_sha256=sha(OUT / "records.csv")))
        return
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid worker assignment")
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    novelty, books, prototypes = {}, {}, {}
    for h in (5, 3, 4):
        with np.load(ROOT / f"results/unknown_recovery/scores_v1/scores_{h}.npz", allow_pickle=False) as stored:
            if stored["ids"].tolist() != ids:
                raise ValueError("Factorial score IDs differ")
            novelty[h] = stored["knn_emergence"]
        with np.load(ROOT / f"results/product_quantizer/known_only_v1/h{h}_m8.npz", allow_pickle=False) as stored:
            books[h] = ProductQuantizer(stored["centroids"])
        with np.load(ROOT / f"results/unknown_scaled/known_{h}.npz", allow_pickle=False) as stored:
            mean, labels = stored["means"], stored["labels"]
        prototypes[h] = mean / np.linalg.norm(mean, axis=1, keepdims=True), labels
    pairs = {p["id"]: p for p in manifest["pairs"]}
    assigned = spec["pair_ids"][args.worker_index::args.workers]
    for number, pair_id in enumerate(assigned, 1):
        folder = OUT / "pairs" / pair_id
        receipt_path = folder / "complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["spec_sha256"] != sha(OUT / "run_spec.json") or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed factorial pair")
            continue
        for source, expected in spec["source_sha256"].items():
            if sha(ROOT / source) != expected:
                raise ValueError("Factorial method changed during execution")
        index = ids.index(pair_id)
        rows = process(pairs[pair_id], features[index], {h: novelty[h][index] for h in (5, 3, 4)}, helper, books, prototypes)
        save(folder / "records.json", rows)
        save(receipt_path, dict(status="complete", records=294, spec_sha256=sha(OUT / "run_spec.json"), records_sha256=sha(folder / "records.json")))
        save(OUT / f"worker_{args.worker_index}_progress.json", dict(status="in_progress", completed_assignment=number, expected_pairs=len(assigned)))
        print(f"task/evidence worker {args.worker_index}: {number}/{len(assigned)} pairs complete", flush=True)
    save(OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", pairs=len(assigned), spec_sha256=sha(OUT / "run_spec.json")))


if __name__ == "__main__":
    main()
