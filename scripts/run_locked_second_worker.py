"""Disjoint execution workers for the unchanged, locked SECOND experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from scripts import run_locked_second_test as original
from safesync.features import FrozenEncoder
from scripts.run_second_feature_mix import load_frozen


def validate_pair(pair_id, lock):
    folder = original.OUT / "pairs" / pair_id
    path = folder / "complete.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text("utf-8"))
    if receipt["lock_sha256"] != original.sha(original.LOCK) or original.sha(folder / "records.json") != receipt["records_sha256"]:
        raise ValueError("Changed locked pair receipt")
    records = json.loads((folder / "records.json").read_text("utf-8"))
    if len(records) != 114 or pair_id not in lock["pair_ids"]:
        raise ValueError("Invalid locked pair layout")
    return records


def finish(lock):
    records = []
    for pair_id in lock["pair_ids"]:
        values = validate_pair(pair_id, lock)
        if values is None:
            raise ValueError(f"Reserved test pair remains pending: {pair_id}")
        records.extend(values)
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    helper["save_csv"](original.OUT / "records.csv", records)
    metadata = json.loads((original.OUT / "execution_workers.json").read_text("utf-8"))
    original.write_json(original.OUT / "complete.json", dict(status="complete", pairs=256, records=len(records), expected=29184,
        lock_sha256=original.sha(original.LOCK), source_sha256=original.sha(ROOT / "scripts/run_locked_second_test.py"),
        records_sha256=original.sha(original.OUT / "records.csv"), execution=metadata,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    original.write_json(original.OUT / "progress.json", dict(status="complete", completed_pairs=256, expected_pairs=256,
        records=len(records), expected_records=29184))
    print(json.dumps(dict(status="complete", pairs=256, records=len(records))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid disjoint worker assignment")
    lock = original.lock_method()
    for source, expected in lock["method_source_sha256"].items():
        if original.sha(ROOT / source) != expected:
            raise ValueError("Frozen method differs before worker execution")
    if args.finalize_only:
        finish(lock)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, original.CONFIG)
    execution = dict(worker_source_sha256=original.sha(Path(__file__)), workers=args.workers,
        assignment="Index in the original frozen pair list modulo worker count",
        method="Calls the unchanged score_pair from the locked driver; only execution scheduling changes",
        encoder=encoder.metadata)
    # Each worker retains its own metadata. Worker zero owns the shared receipt.
    original.write_json(original.OUT / f"worker_{args.worker_index}_spec.json", execution)
    if args.worker_index == 0:
        original.write_json(original.OUT / "execution_workers.json", execution)
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    wire, _ = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    galleries, prototypes, models = {}, {}, {}
    for heldout in original.HELDOUTS:
        item = lock["galleries"][str(heldout)]
        if original.sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Frozen gallery changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            galleries[heldout] = torch.tensor(stored["features"], device=encoder.device)
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as stored:
            mean, precision, labels = stored["means"], stored["precision"], stored["labels"]
        prototypes[heldout] = mean / np.linalg.norm(mean, axis=1, keepdims=True), labels
        models[heldout] = mean, precision
    manifest = json.loads((original.DATA / "manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    assigned = lock["pair_ids"][args.worker_index::args.workers]
    for number, pair_id in enumerate(assigned, 1):
        if validate_pair(pair_id, lock) is not None:
            continue
        for source, expected in lock["method_source_sha256"].items():
            if original.sha(ROOT / source) != expected:
                raise ValueError("Locked method changed during parallel execution")
        pair = pairs[pair_id]
        for item in pair["files"].values():
            if original.sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Reserved source changed")
        records = original.score_pair(pair, encoder, galleries, prototypes, models, wire, helper)
        if len(records) != 114:
            raise AssertionError("Incomplete original per-pair recipe")
        folder = original.OUT / "pairs" / pair_id
        original.write_json(folder / "records.json", records)
        original.write_json(folder / "complete.json", dict(status="complete", records=114,
            lock_sha256=original.sha(original.LOCK), records_sha256=original.sha(folder / "records.json"),
            input_sha256=__import__("hashlib").sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()))
        original.write_json(original.OUT / f"worker_{args.worker_index}_progress.json", dict(status="in_progress", last_pair_id=pair_id,
            completed_assignment=number, assignment_size=len(assigned)))
        print(f"locked worker {args.worker_index} assignment {number}/{len(assigned)} complete", flush=True)
    original.write_json(original.OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", assignment_size=len(assigned),
        worker_source_sha256=original.sha(Path(__file__)), method_lock_sha256=original.sha(original.LOCK)))


if __name__ == "__main__":
    main()
