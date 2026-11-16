"""Test a current global context before native evidence, with a fixed half cap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
from safesync import task_evidence_wire as wire
from safesync.data import read_pair
from safesync import full_image_codecs as whole
from scripts.run_second_feature_mix import load_frozen
from scripts.run_task_evidence_development import fit_full
from scripts.run_task_evidence_selector import quality

BASE = ROOT / "results/task_evidence/development_v1"
OUT = ROOT / "results/task_evidence/context_development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2)+"\n", "utf-8")
    temporary.replace(path)


def image_payload(packet):
    header = wire.HEADER.unpack_from(packet)
    start = wire.HEADER.size + header[14] + header[11]*2 + header[15]
    return packet[start:-16]


def fit_context(prior, labels, current, ranking, cap, atlas_cache, base_packet):
    base = image_payload(base_packet)
    if not base:
        packet, costs, _ = fit_full(prior, labels, current, "webp", cap)
        return packet, costs, dict(context_fallback=True, refined_rois=0, base_bytes=0)
    order = np.argsort(-ranking, kind="stable")
    identity = hashlib.sha256(order.tobytes()).hexdigest()

    def make(count):
        key = identity, count
        if key not in atlas_cache:
            atlas_cache[key] = wire.encode_atlas(current, order[:count], codec="webp", quality=70)
        payload = wire.layered_payload(base, atlas_cache[key])
        return wire.encode(prior, labels, observation=1, payload=payload, indices=order[:count], image_mode=wire.LAYERED)

    best, costs = make(0)
    if len(best) + wire.ACK_BYTES > cap:
        raise ValueError("Context base and task state exceed cap")
    selected, low, high = 0, 1, 64
    while low <= high:
        middle = (low + high)//2
        packet, components = make(middle)
        if len(packet) + wire.ACK_BYTES <= cap:
            best, costs, selected = packet, components, middle
            low = middle + 1
        else:
            high = middle - 1
    for count in range(selected + 1, min(selected + 3, 64) + 1):
        packet, components = make(count)
        if len(packet) + wire.ACK_BYTES <= cap:
            best, costs, selected = packet, components, count
    return best, costs, dict(context_fallback=False, refined_rois=selected, base_bytes=len(base), base_total_cap=cap//2)


def run_pair(pair, features, novelty, helper):
    source = json.loads((BASE / "pairs" / pair["id"] / "records.json").read_text("utf-8"))
    lookup = {(int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in source}
    current = np.asarray(Image.open(ROOT / pair["files"]["after"]["path"]).convert("RGB"))
    change_rank = rankdata(helper["pool_scores"](1 - np.sum(features[0]*features[1], axis=1)))/64
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    atlas_cache, encoded_cache = {}, {}
    original_encode = whole.encode_payload

    def cached_full(image, codec, quality):
        key = image.size, codec, quality
        if key not in encoded_cache:
            encoded_cache[key] = original_encode(image, codec, quality)
        return encoded_cache[key]

    whole.encode_payload = cached_full
    pending = []
    for h in (5, 3, 4):
        base_folder = BASE / "pairs" / pair["id"] / f"h{h}_labels"
        boot = (base_folder / "bootstrap.bin").read_bytes()
        head = wire.HEADER.unpack_from(boot)
        blank = wire.TaskEvidenceState(np.zeros_like(current), np.zeros((16, 16), np.uint8), head[1], head[2])
        prior, boot_ack = wire.receive(blank, boot)
        service = lookup[(h, 0, "labels_only", 0)]
        current_service, _ = wire.receive(prior, (ROOT / service["packet_path"]).read_bytes())
        labels = current_service.labels
        unknown_rank = rankdata(helper["pool_scores"](novelty[h]))/64
        for cap in (4096, 8192, 16384, 32768):
            full = lookup[(h, cap, "labels_full_webp", 0)]
            full_packet = (ROOT / full["packet_path"]).read_bytes()
            full_state, full_ack = wire.receive(prior, full_packet)
            if cap == 4096:
                base_packet, _, _ = fit_full(prior, labels, current, "webp", cap//2)
            else:
                base_packet = (ROOT / lookup[(h, cap//2, "labels_full_webp", 0)]["packet_path"]).read_bytes()
            for route in ("change", "mix", "random"):
                for seed in ((1729, 2718, 31415) if route == "random" else (0,)):
                    if route == "change":
                        ranking = change_rank
                    elif route == "mix":
                        ranking = .75*change_rank + .25*unknown_rank
                    else:
                        key = hashlib.sha256(f"task-evidence-random:{pair['id']}:{seed}".encode()).digest()[:8]
                        ranking = np.random.default_rng(int.from_bytes(key, "little")).random(64) + .01
                    packet, costs, metadata = fit_context(prior, labels, current, ranking, cap, atlas_cache, base_packet)
                    state, ack = wire.receive(prior, packet)
                    replay, replay_ack = wire.receive(prior, packet)
                    if ack != replay_ack or state.digest() != replay.digest() or len(packet) + len(ack) > cap:
                        raise AssertionError("Context actual-byte or replay check")
                    if not np.array_equal(state.labels, labels) or not np.array_equal(full_state.labels, labels):
                        raise AssertionError("Context changed the exact task service")
                    cq, fq = quality(current, state.rgb, ranking), quality(current, full_state.rgb, ranking)
                    choices = [("context_"+route, packet, ack, state, "context"),
                               ("context_adaptive_"+route, packet if cq > fq else full_packet,
                                ack if cq > fq else full_ack, state if cq > fq else full_state,
                                "context" if cq > fq else "full")]
                    for method, selected, selected_ack, received, action in choices:
                        path = folder / f"h{h}_{method}_s{seed}_b{cap}.bin"
                        path.write_bytes(selected)
                        path.with_suffix(".ack").write_bytes(selected_ack)
                        meta = dict(pair_id=pair["id"], heldout=h, method="labels_"+method, seed=seed, budget=cap,
                            down_bytes=len(selected), ack_bytes=len(selected_ack), total_bytes=len(selected)+len(selected_ack),
                            packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                            bootstrap_path=(base_folder / "bootstrap.bin").relative_to(ROOT).as_posix(), bootstrap_bytes=len(boot)+len(boot_ack),
                            receiver_rgb_sha256=hashlib.sha256(received.rgb.tobytes()).hexdigest(),
                            selected_action=action, context_quality=cq, full_quality=fq, context_metadata=json.dumps(metadata, sort_keys=True))
                        pending.append((meta, received))
    whole.encode_payload = original_encode
    # Commit every decision before opening the evaluation labels.
    save(folder / "sender_commit.json", [r for r, _ in pending])
    _, _, old, label, changed = read_pair(ROOT, pair)
    rows = []
    for meta, state in pending:
        known, unknown = helper["evaluation_masks"](old, label, changed, meta["heldout"])
        prediction = state.labels.repeat(32, 0).repeat(32, 1)
        matrix = np.bincount((label[known].astype(int)-1)*6 + prediction[known].astype(int)-1, minlength=36).reshape(6, 6)
        rows.append(dict(**meta, known_confusion=json.dumps(matrix.tolist()), **helper["image_metrics"](current, state.rgb, known, unknown)))
    if len(rows) != 120:
        raise AssertionError("Incomplete context repair cohort")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/second_scaled/manifest.json").read_text("utf-8"))
    pair_ids = [p["id"] for p in manifest["pairs"] if p["split"] == "validation"]
    names = ("scripts/run_task_context_development.py", "scripts/run_task_evidence_development.py", "scripts/run_task_evidence_selector.py", "safesync/task_evidence_wire.py")
    spec = dict(source_sha256={p: sha(ROOT / p) for p in names}, source_candidates_sha256=sha(BASE / "records.csv"),
        manifest_sha256=sha(ROOT / "data/second_scaled/manifest.json"), pair_ids=pair_ids,
        base_fraction=.5, base="Full WebP fitted at half total cap, only its image bytes embedded in the final transaction",
        refinement="WebP70 rank-prefix atlas", selection="Same fixed 30dB bounded quality proxy; full-image tie preference",
        hypothesis="Stale unselected RGB harms the later semantic use; current global context should reduce that task loss",
        fixed="Known task service, backbone, novelty, ranks, heldouts, budgets and ground head",
        decision="Check both visual U30 and later-class F1 before fresh confirmation", expected_records=7680)
    spec_path = OUT / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text("utf-8")) != spec:
            raise ValueError("Changed context repair")
    else:
        save(spec_path, spec)
        save(OUT / "locked_before_run.json", dict(utc=datetime.now(timezone.utc).isoformat(), spec_sha256=sha(spec_path)))
    if args.freeze_only:
        print(json.dumps(dict(status="development_locked", expected=7680)), flush=True)
        return
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    if args.finalize_only:
        rows = []
        for pair_id in pair_ids:
            path = OUT / "pairs" / pair_id / "records.json"
            receipt = json.loads(path.with_name("complete.json").read_text("utf-8"))
            if sha(path) != receipt["records_sha256"] or receipt["spec_sha256"] != sha(spec_path):
                raise ValueError("Changed context cohort")
            rows.extend(json.loads(path.read_text("utf-8")))
        if len(rows) != 7680:
            raise AssertionError("Incomplete context repair")
        helper["save_csv"](OUT / "records.csv", rows)
        save(OUT / "complete.json", dict(status="complete", records=7680, records_sha256=sha(OUT / "records.csv"), spec_sha256=sha(spec_path)))
        return
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    novelty = {}
    for h in (5, 3, 4):
        with np.load(ROOT / f"results/unknown_recovery/scores_v1/scores_{h}.npz", allow_pickle=False) as stored:
            novelty[h] = stored["knn_emergence"]
    pairs = {p["id"]: p for p in manifest["pairs"]}
    assigned = pair_ids[args.worker_index::args.workers]
    for number, pair_id in enumerate(assigned, 1):
        folder = OUT / "pairs" / pair_id
        if (folder / "complete.json").exists():
            continue
        for name, expected in spec["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Context method changed during execution")
        index = ids.index(pair_id)
        rows = run_pair(pairs[pair_id], features[index], {h: novelty[h][index] for h in (5, 3, 4)}, helper)
        save(folder / "records.json", rows)
        save(folder / "complete.json", dict(status="complete", records=120, spec_sha256=sha(spec_path), records_sha256=sha(folder / "records.json")))
        save(OUT / f"worker_{args.worker_index}_progress.json", dict(completed_pairs=number, expected_pairs=len(assigned)))
        print(f"current-context worker {args.worker_index}: {number}/{len(assigned)}", flush=True)
    save(OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", pairs=len(assigned)))


if __name__ == "__main__":
    main()
