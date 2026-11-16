"""Fresh confirmation of exact task states and current-context evidence."""
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
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync import task_evidence_wire as wire
from safesync import full_image_codecs as whole
from safesync.features import FrozenEncoder
from safesync.data import read_pair
from scripts.run_second_feature_mix import load_frozen
from scripts.run_task_evidence_development import fit_full, fit_atlas
from scripts.run_task_context_development import fit_context
from scripts.run_task_evidence_selector import quality
from scripts.run_locked_second_test import CONFIG, write_json, sha

OUT = ROOT / "results/task_evidence/fresh_context_v1"
DATA = ROOT / "data/second_task_context_test_v1"
LOCK = OUT / "method_lock.json"
RESERVATION = ROOT / "data/second_task_evidence_reservation_v1.json"
CAPS = (4096, 8192, 16384, 32768)
SEEDS = (1729, 2718, 31415)
SOURCE_NAMES = ("scripts/run_task_context_confirmation.py", "scripts/fetch_task_context_confirmation.py",
    "safesync/task_evidence_wire.py", "scripts/run_task_context_development.py",
    "scripts/run_task_evidence_development.py", "scripts/run_task_evidence_selector.py",
    "safesync/features.py", "safesync/data.py", "safesync/scoring.py", "safesync/full_image_codecs.py",
    "safesync/learned_codec.py", "safesync/entropy_bridge.py", "resources/learned_codec/entropy_server.py")


def lock_method():
    reserved = json.loads(RESERVATION.read_text("utf-8"))
    old_lock = json.loads((ROOT / "results/unknown_recovery/confirmation_lock_v1.json").read_text("utf-8"))
    proposed = dict(schema="fresh-task-context-v1", pair_ids=[p["id"] for p in reserved["reserved_pairs"]],
        reservation_sha256=sha(RESERVATION), encoder_config=CONFIG,
        model_weight_sha256=sha(ROOT / "resources/models/dinov2_vits14_pretrain.pth"),
        public_hyperprior_registry_sha256=sha(ROOT / "resources/learned_codec/public_downloads.json"),
        source_sha256={p: sha(ROOT / p) for p in SOURCE_NAMES},
        galleries=old_lock["galleries"], prototypes=old_lock["prototypes"],
        later_ground_head_sha256=sha(ROOT / "results/future_semantic_probe/v1/prototypes.npz"),
        development=dict(context_analysis_sha256=sha(ROOT / "results/task_evidence/context_development_v1/analysis.json"),
            use_analysis_sha256=sha(ROOT / "results/task_evidence/context_future_probe_v1/analysis.json")),
        heldouts=[5, 3, 4], caps=CAPS, seeds=SEEDS, novelty_k=10, mixture=[.75, .25], context_fraction=.5,
        roi_codec="WebP70 native atlas", context_codec="Full WebP search at half total cap; image bytes embedded once",
        action_quality="rank-weighted max(1 - native cell RGB MSE/0.001, 0); choose full WebP on ties",
        task_service="Exact current known-prototype class array; choose shorter absolute/XOR zlib stream from reconstructed reference",
        image_controls=["jpeg420", "jpeg2000", "webp", "public hyperprior q1..8 at five scales"],
        primary=dict(heldout=5, budget=8192, method="adaptive_mix",
            coverage_comparators=["adaptive_change", "full_webp", "hyperprior"],
            known_task="Identical decoded current task states across routes and image controls",
            later_class_f1_comparator="full_webp", later_class_f1_margin=-.02,
            decision="All three paired coverage 95% lower bounds exceed zero, the later-use 95% lower bound exceeds -0.02, and task states match the full-current service"),
        secondary="All three heldouts and four caps, 99.583333% descriptive intervals per contrast family",
        scope="Separate 256-pair reservation disjoint from all earlier acquired pairs. Development and earlier test diagnosed repairs. This method and joint criteria freeze before any new reserved image or label access.",
        expected_records=45312)
    proposed = json.loads(json.dumps(proposed))
    if LOCK.exists():
        stored = json.loads(LOCK.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != proposed:
            raise ValueError("Refusing to change the fresh context confirmation")
        return stored
    if DATA.exists() and any(DATA.rglob("*.png")):
        raise ValueError("Fresh lock attempted after reserved pixel access")
    proposed["locked_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(LOCK, proposed)
    for name in SOURCE_NAMES:
        path = OUT / "source_snapshot" / name.replace("/", "__")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return proposed


def prepare_pair(pair, encoder, galleries, prototypes):
    for key in ("before", "after"):
        item = pair["files"][key]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Fresh input image or label changed")
    before_pil = Image.open(ROOT / pair["files"]["before"]["path"]).convert("RGB")
    after_pil = Image.open(ROOT / pair["files"]["after"]["path"]).convert("RGB")
    before, current = np.asarray(before_pil), np.asarray(after_pil)
    f0, _ = encoder.cached(before_pil)
    f1, _ = encoder.cached(after_pil)
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "sender_features.npz", before=f0.astype(np.float16), current=f1.astype(np.float16))
    query = torch.nn.functional.normalize(torch.tensor(np.concatenate([f0, f1]), device=encoder.device), dim=1)
    stream = io.BytesIO()
    before_pil.save(stream, format="PNG")
    priors, tasks, emergence, common = {}, {}, {}, {}
    for h in (5, 3, 4):
        mean, classes = prototypes[h]
        old_task = classes[np.argmax(f0 @ mean.T, axis=1)].astype(np.uint8).reshape(16, 16)
        new_task = classes[np.argmax(f1 @ mean.T, axis=1)].astype(np.uint8).reshape(16, 16)
        model = hashlib.sha256(mean.tobytes() + classes.tobytes() + b"task-labels-v1").digest()[:16]
        region = hashlib.sha256(("task-evidence:"+pair["id"]).encode()).digest()[:16]
        blank = wire.TaskEvidenceState(np.zeros_like(before), np.zeros_like(old_task), region, model)
        bootstrap, _ = wire.encode(blank, old_task, observation=0, payload=stream.getvalue(), image_mode=wire.FULL)
        prior, ack = wire.receive(blank, bootstrap)
        if not np.array_equal(prior.rgb, before) or not np.array_equal(prior.labels, old_task):
            raise AssertionError("Fresh common bootstrap failed")
        boot_path = folder / f"h{h}_bootstrap.bin"
        boot_path.write_bytes(bootstrap)
        boot_path.with_suffix(".ack").write_bytes(ack)
        task_packet, _ = wire.encode(prior, new_task, observation=1)
        task_path = folder / f"h{h}_labels_only.bin"
        task_path.write_bytes(task_packet)
        _, task_ack = wire.receive(prior, task_packet)
        task_path.with_suffix(".ack").write_bytes(task_ack)
        neighbor = torch.topk(query @ galleries[h].T, k=10, dim=1, sorted=True).values[:, -1]
        novelty = torch.clamp(2 - 2*neighbor, min=0).cpu().numpy().reshape(2, 256)
        emergence[h] = np.maximum(novelty[1]-novelty[0], 0)
        priors[h], tasks[h] = prior, new_task
        common[str(h)] = dict(bootstrap_path=boot_path.relative_to(ROOT).as_posix(), bootstrap_sha256=sha(boot_path),
            labels_only_path=task_path.relative_to(ROOT).as_posix(), labels_only_sha256=sha(task_path),
            bootstrap_bytes=len(bootstrap)+len(ack), current_labels_sha256=hashlib.sha256(new_task.tobytes()).hexdigest())
    write_json(folder / "sender_inputs_ready.json", dict(status="complete", pair_id=pair["id"], lock_sha256=sha(LOCK),
        source_rgb_sha256={key: pair["files"][key]["sha256"] for key in ("before", "after")},
        features_sha256=sha(folder / "sender_features.npz"), common=common))
    return current, f0, f1, priors, tasks, emergence, common


def encode_pair(pair, encoder, galleries, prototypes, helper):
    current, f0, f1, priors, tasks, emergence, common = prepare_pair(pair, encoder, galleries, prototypes)
    folder = OUT / "pairs" / pair["id"]
    rank_change = rankdata(helper["pool_scores"](1-np.sum(f0*f1, axis=1)))/64
    full_cache, atlas_cache, context_cache = {}, {}, {}
    original_encode = whole.encode_payload

    def cached_full(image, codec, quality):
        key = image.size, codec, quality
        if key not in full_cache:
            full_cache[key] = original_encode(image, codec, quality)
        return full_cache[key]

    whole.encode_payload = cached_full
    pending = []

    def commit(h, cap, method, seed, packet, state, ack, metadata):
        if cap and len(packet)+len(ack) > cap:
            raise AssertionError("Fresh packet+ACK cap exceeded")
        if method != "old_only" and not np.array_equal(state.labels, tasks[h]):
            raise AssertionError("Decoded current task service differs")
        path = folder / f"h{h}_{method}_s{seed}_b{cap}.bin"
        path.write_bytes(packet)
        path.with_suffix(".ack").write_bytes(ack)
        row = dict(pair_id=pair["id"], heldout=h, method=method, seed=seed, budget=cap,
            down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet)+len(ack),
            bootstrap_bytes=common[str(h)]["bootstrap_bytes"], bootstrap_path=common[str(h)]["bootstrap_path"],
            packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
            receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),
            receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
            codec_metadata=json.dumps(metadata, sort_keys=True))
        pending.append((row, state))

    for h in (5, 3, 4):
        prior, task = priors[h], tasks[h]
        rank_unknown = rankdata(helper["pool_scores"](emergence[h]))/64
        for cap in CAPS:
            full_states = {}
            for codec in ("webp", "jpeg420", "jpeg2000"):
                packet, _, meta = fit_full(prior, task, current, codec, cap)
                state, ack = wire.receive(prior, packet)
                commit(h, cap, "full_"+codec, 0, packet, state, ack, meta)
                full_states[codec] = packet, state, ack
            full_packet, full_state, full_ack = full_states["webp"]
            base_packet, _, _ = fit_full(prior, task, current, "webp", cap//2)
            for route in ("change", "mix", "random"):
                for seed in (SEEDS if route == "random" else (0,)):
                    if route == "change":
                        ranking = rank_change
                    elif route == "mix":
                        ranking = .75*rank_change + .25*rank_unknown
                    else:
                        key = hashlib.sha256(f"task-evidence-random:{pair['id']}:{seed}".encode()).digest()[:8]
                        ranking = np.random.default_rng(int.from_bytes(key, "little")).random(64) + .01
                    packet, _, meta = fit_context(prior, task, current, ranking, cap, context_cache, base_packet)
                    state, ack = wire.receive(prior, packet)
                    replay, repeat_ack = wire.receive(prior, packet)
                    if state.digest() != replay.digest() or ack != repeat_ack:
                        raise AssertionError("Fresh context reconstruction differs on replay")
                    commit(h, cap, "context_"+route, seed, packet, state, ack, meta)
                    cq, fq = quality(current, state.rgb, ranking), quality(current, full_state.rgb, ranking)
                    selected = (packet, state, ack) if cq > fq else (full_packet, full_state, full_ack)
                    commit(h, cap, "adaptive_"+route, seed, *selected,
                        dict(action="context" if cq > fq else "full", context_quality=cq, full_quality=fq))
                    if route == "mix":
                        roi_packet, _, selected_rois = fit_atlas(prior, task, current, ranking, "webp", 70, cap, atlas_cache)
                        roi_state, roi_ack = wire.receive(prior, roi_packet)
                        commit(h, cap, "roi_mix", 0, roi_packet, roi_state, roi_ack, dict(quality=70, selected_rois=selected_rois))
        task_packet = (ROOT / common[str(h)]["labels_only_path"]).read_bytes()
        state, ack = wire.receive(prior, task_packet)
        commit(h, 0, "labels_only", 0, task_packet, state, ack, {})
        commit(h, 0, "old_only", 0, b"", prior, b"", dict(current_confirmation=False))
        stream = io.BytesIO()
        Image.fromarray(current).save(stream, format="PNG")
        packet, _ = wire.encode(prior, task, observation=1, payload=stream.getvalue(), image_mode=wire.FULL)
        state, ack = wire.receive(prior, packet)
        commit(h, 0, "full_current", 0, packet, state, ack, dict(lossless_current=True))
    whole.encode_payload = original_encode
    if len(pending) != 177:
        raise AssertionError("Incomplete fresh sender matrix")
    write_json(folder / "sender_commit.json", [r for r, _ in pending])
    # All online choices are committed before decoding evaluation labels.
    for key in ("label_before", "label_after", "binary_change"):
        item = pair["files"][key]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Fresh evaluation label changed")
    _, _, old, label, changed = read_pair(ROOT, pair)
    records = []
    for row, state in pending:
        known, unknown = helper["evaluation_masks"](old, label, changed, row["heldout"])
        prediction = state.labels.repeat(32, 0).repeat(32, 1)
        matrix = np.bincount((label[known].astype(int)-1)*6+prediction[known].astype(int)-1, minlength=36).reshape(6, 6)
        records.append(dict(**row, known_confusion=json.dumps(matrix.tolist()), **helper["image_metrics"](current, state.rgb, known, unknown)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    lock = lock_method()
    if args.freeze_only:
        print(json.dumps(dict(status="locked_before_fresh_pixels", pairs=256, primary=lock["primary"])), flush=True)
        return
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    if args.finalize_only:
        records = []
        for pair_id in lock["pair_ids"]:
            folder = OUT / "pairs" / pair_id
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed fresh completed pair")
            records.extend(json.loads((folder / "records.json").read_text("utf-8")))
        if len(records) != 45312:
            raise AssertionError("Incomplete fresh confirmation")
        helper["save_csv"](OUT / "records.csv", records)
        write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=45312, expected=45312,
            lock_sha256=sha(LOCK), records_sha256=sha(OUT / "records.csv"), completed_utc=datetime.now(timezone.utc).isoformat()))
        return
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid fresh worker assignment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    if encoder.metadata["identity"]["weight_sha256"] != lock["model_weight_sha256"]:
        raise ValueError("Frozen backbone changed")
    galleries, prototypes = {}, {}
    for h in (5, 3, 4):
        item = lock["galleries"][str(h)]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Fresh known-only gallery changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"], device=encoder.device)
        path = ROOT / f"results/unknown_scaled/known_{h}.npz"
        if sha(path) != lock["prototypes"][str(h)]:
            raise ValueError("Fresh task head changed")
        with np.load(path, allow_pickle=False) as stored:
            means, classes = stored["means"], stored["labels"]
        prototypes[h] = means/np.linalg.norm(means, axis=1, keepdims=True), classes
    assigned = lock["pair_ids"][args.worker_index::args.workers]
    done = set()
    for pair_id in assigned:
        folder = OUT / "pairs" / pair_id
        receipt_path = folder / "complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Fresh completed pair changed")
            done.add(pair_id)
    write_json(OUT / f"worker_{args.worker_index}_spec.json", dict(workers=args.workers, worker_index=args.worker_index,
        assignment="Frozen source-pair index modulo worker count", encoder=encoder.metadata, lock_sha256=sha(LOCK)))
    while len(done) < len(assigned):
        path = DATA / "manifest.json"
        manifest = json.loads(path.read_text("utf-8")) if path.exists() else {"pairs": [], "complete": False}
        available = {p["id"]: p for p in manifest["pairs"]}
        pending = [p for p in assigned if p not in done and p in available]
        if not pending:
            if manifest["complete"]:
                raise ValueError("Completed acquisition lacks fresh assigned pairs")
            time.sleep(15)
            continue
        for pair_id in pending:
            for name, expected in lock["source_sha256"].items():
                if sha(ROOT / name) != expected:
                    raise ValueError("Fresh method changed after lock")
            records = encode_pair(available[pair_id], encoder, galleries, prototypes, helper)
            folder = OUT / "pairs" / pair_id
            write_json(folder / "records.json", records)
            write_json(folder / "complete.json", dict(status="complete", records=177, lock_sha256=sha(LOCK),
                records_sha256=sha(folder / "records.json"), sender_commit_sha256=sha(folder / "sender_commit.json")))
            done.add(pair_id)
            write_json(OUT / f"worker_{args.worker_index}_progress.json", dict(status="in_progress", completed_pairs=len(done), expected_pairs=len(assigned)))
            print(f"fresh context worker {args.worker_index}: {len(done)}/{len(assigned)}", flush=True)
    write_json(OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", pairs=len(assigned), lock_sha256=sha(LOCK)))


if __name__ == "__main__":
    main()
