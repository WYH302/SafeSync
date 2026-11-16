"""Additional locked ablation of current distance versus temporal emergence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from safesync.data import read_pair
from safesync.features import FrozenEncoder
from scripts.run_second_feature_mix import load_frozen
from scripts.run_task_evidence_development import fit_full
from scripts.run_task_context_development import fit_context
from scripts.run_task_evidence_selector import quality
from scripts.evaluate_task_context_confirmation import evaluate
from scripts.run_locked_second_test import CONFIG, write_json, sha

MAIN = ROOT / "results/task_evidence/fresh_context_v1"
GROUND = ROOT / "results/task_evidence/fresh_context_ground_v1"
OUT = ROOT / "results/task_evidence/fresh_current_distance_v1"


def process(pair, galleries, helper):
    source = MAIN / "pairs" / pair["id"]
    ready = json.loads((source / "sender_inputs_ready.json").read_text("utf-8"))
    receipt = json.loads((source / "complete.json").read_text("utf-8"))
    if sha(source / "sender_commit.json") != receipt["sender_commit_sha256"]:
        raise ValueError("Current-distance source sender commit changed")
    committed = json.loads((source / "sender_commit.json").read_text("utf-8"))
    packet_hashes = {r["packet_path"]: r["packet_sha256"] for r in committed}
    if sha(source / "sender_features.npz") != ready["features_sha256"]:
        raise ValueError("Committed sender features changed")
    with np.load(source / "sender_features.npz", allow_pickle=False) as stored:
        f0, f1 = stored["before"].astype(np.float32), stored["current"].astype(np.float32)
    current = np.asarray(Image.open(ROOT / pair["files"]["after"]["path"]).convert("RGB"))
    rank_change = rankdata(helper["pool_scores"](1-np.sum(f0*f1, axis=1)))/64
    query = torch.nn.functional.normalize(torch.tensor(f1, device="cuda"), dim=1)
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    cache, full_cache = {}, {}
    original_encode = whole.encode_payload

    def cached_full(image, codec, quality):
        key = image.size, codec, quality
        if key not in full_cache:
            full_cache[key] = original_encode(image, codec, quality)
        return full_cache[key]

    whole.encode_payload = cached_full
    pending = []
    for h in (5, 3, 4):
        common = ready["common"][str(h)]
        boot_path, task_path = ROOT / common["bootstrap_path"], ROOT / common["labels_only_path"]
        if sha(boot_path) != common["bootstrap_sha256"] or sha(task_path) != common["labels_only_sha256"]:
            raise ValueError("Current-distance common packets changed")
        boot = boot_path.read_bytes()
        header = wire.HEADER.unpack_from(boot)
        blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), header[1], header[2])
        prior, _ = wire.receive(blank, boot)
        service, _ = wire.receive(prior, task_path.read_bytes())
        neighbor = torch.topk(query @ galleries[h].T, k=10, sorted=True).values[:, -1]
        distance = torch.clamp(2-2*neighbor, min=0).cpu().numpy()
        ranking = .75*rank_change + .25*rankdata(helper["pool_scores"](distance))/64
        for cap in (4096, 8192, 16384, 32768):
            full_path = source / f"h{h}_full_webp_s0_b{cap}.bin"
            if sha(full_path) != packet_hashes[full_path.relative_to(ROOT).as_posix()]:
                raise ValueError("Primary full-image candidate changed")
            full_packet = full_path.read_bytes()
            full_state, full_ack = wire.receive(prior, full_packet)
            if cap == 4096:
                base_packet, _, _ = fit_full(prior, service.labels, current, "webp", cap//2)
            else:
                base_packet = (source / f"h{h}_full_webp_s0_b{cap//2}.bin").read_bytes()
            packet, _, metadata = fit_context(prior, service.labels, current, ranking, cap, cache, base_packet)
            state, ack = wire.receive(prior, packet)
            cq, fq = quality(current, state.rgb, ranking), quality(current, full_state.rgb, ranking)
            choices = (("context_current_distance", packet, state, ack),
                ("adaptive_current_distance", packet if cq > fq else full_packet,
                 state if cq > fq else full_state, ack if cq > fq else full_ack))
            for method, selected, received, selected_ack in choices:
                if len(selected)+len(selected_ack) > cap or not np.array_equal(received.labels, service.labels):
                    raise AssertionError("Current-distance cap or common service failed")
                path = folder / f"h{h}_{method}_b{cap}.bin"
                path.write_bytes(selected)
                path.with_suffix(".ack").write_bytes(selected_ack)
                row = dict(pair_id=pair["id"], heldout=h, method=method, seed=0, budget=cap,
                    down_bytes=len(selected), ack_bytes=len(selected_ack), total_bytes=len(selected)+len(selected_ack),
                    bootstrap_bytes=common["bootstrap_bytes"], bootstrap_path=common["bootstrap_path"],
                    packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                    receiver_rgb_sha256=hashlib.sha256(received.rgb.tobytes()).hexdigest(),
                    receiver_task_sha256=hashlib.sha256(received.labels.tobytes()).hexdigest(),
                    codec_metadata=json.dumps(dict(context_quality=cq, full_quality=fq, context=metadata), sort_keys=True))
                pending.append((row, received))
    whole.encode_payload = original_encode
    write_json(folder / "sender_commit.json", [r for r, _ in pending])
    _, _, old, label, changed = read_pair(ROOT, pair)
    rows = []
    for row, state in pending:
        known, unknown = helper["evaluation_masks"](old, label, changed, row["heldout"])
        prediction = state.labels.repeat(32, 0).repeat(32, 1)
        matrix = np.bincount((label[known].astype(int)-1)*6+prediction[known].astype(int)-1, minlength=36).reshape(6, 6)
        rows.append(dict(**row, known_confusion=json.dumps(matrix.tolist()), **helper["image_metrics"](current, state.rgb, known, unknown)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    spec = dict(source_sha256=sha(Path(__file__)), primary_lock_sha256=sha(MAIN / "method_lock.json"),
        source_implementation_sha256={p: sha(ROOT / p) for p in ("safesync/task_evidence_wire.py", "scripts/run_task_context_development.py", "scripts/evaluate_task_context_confirmation.py")},
        difference="Replace positive temporal known-gallery distance increment with current distance, keeping k10, .75/.25 ranks, codec, context fraction and gate fixed",
        inference="Additional post-primary-lock ablation fixed before its own output; main numerical results remain uninspected and primary criteria unchanged",
        requested_by="One focused independent novelty review", expected_records=6144)
    spec_path = OUT / "ablation_lock.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != spec:
            raise ValueError("Current-distance ablation changed")
    else:
        spec["locked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(spec_path, spec)
    if args.freeze_only:
        print(json.dumps(dict(status="additional_ablation_locked", expected=6144)), flush=True)
        return
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    if args.finalize_only:
        rows = []
        for pair_id in lock["pair_ids"]:
            folder = OUT / "pairs" / pair_id
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed current-distance cohort")
            rows.extend(json.loads((folder / "records.json").read_text("utf-8")))
        if len(rows) != 6144:
            raise AssertionError("Incomplete current-distance ablation")
        helper["save_csv"](OUT / "records.csv", rows)
        write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=6144,
            records_sha256=sha(OUT / "records.csv"), ablation_lock_sha256=sha(spec_path)))
        return
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid ablation assignment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(ROOT / "results/future_semantic_probe/v1/prototypes.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    galleries = {}
    for h in (5, 3, 4):
        item = lock["galleries"][str(h)]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Prelocked current-distance gallery changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"], device="cuda")
    assigned = lock["pair_ids"][args.worker_index::args.workers]
    done = set()
    while len(done) < len(assigned):
        path = ROOT / "data/second_task_context_test_v1/manifest.json"
        manifest = json.loads(path.read_text("utf-8")) if path.exists() else {"pairs": []}
        pairs = {p["id"]: p for p in manifest["pairs"]}
        pending = [p for p in assigned if p not in done and p in pairs and (GROUND / "pairs" / p / "main/complete.json").exists()]
        if not pending:
            time.sleep(15)
            continue
        for pair_id in pending:
            folder = OUT / "pairs" / pair_id
            receipt_path = folder / "complete.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text("utf-8"))
                if sha(folder / "records.json") != receipt["records_sha256"] or receipt["ablation_lock_sha256"] != sha(spec_path):
                    raise ValueError("Changed current-distance output")
            else:
                rows = process(pairs[pair_id], galleries, helper)
                rows = evaluate(pairs[pair_id], "main", rows, encoder, means, classes, helper)
                if len(rows) != 24:
                    raise AssertionError("Incomplete current-distance pair")
                write_json(folder / "records.json", rows)
                write_json(receipt_path, dict(status="complete", records=24, records_sha256=sha(folder / "records.json"), ablation_lock_sha256=sha(spec_path)))
            done.add(pair_id)
            write_json(OUT / f"worker_{args.worker_index}_progress.json", dict(completed_pairs=len(done), expected_pairs=len(assigned)))
            print(f"current-distance worker {args.worker_index}: {len(done)}/{len(assigned)}", flush=True)
    write_json(OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", pairs=len(assigned)))


if __name__ == "__main__":
    main()
