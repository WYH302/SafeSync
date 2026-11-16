"""Disjoint source-pair scheduling for the frozen later-use ground assay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from scripts import run_future_semantic_assay as assay


def validate(pair_id, cohort, source_hash):
    folder = assay.OUT / "pairs" / pair_id / cohort
    path = folder / "complete.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text("utf-8"))
    if (receipt["source_receipt_sha256"] != source_hash or assay.sha(folder / "records.json") != receipt["records_sha256"]
            or receipt["assay_lock_sha256"] != assay.sha(assay.OUT / "method_lock.json")):
        raise ValueError("Later-use assay receipt changed")
    return json.loads((folder / "records.json").read_text("utf-8"))


def finish(spec):
    all_rows = []
    for pair_id in spec["pair_ids"]:
        for cohort in assay.SOURCES:
            source = assay.source_rows(cohort, pair_id)
            if source is None:
                raise ValueError("Future task source cohort remains incomplete")
            rows = validate(pair_id, cohort, source[1])
            if rows is None:
                raise ValueError("Future task output cohort remains incomplete")
            all_rows.extend(rows)
    if len(all_rows) != 46080:
        raise AssertionError("Incomplete later-use output matrix")
    _, helper = assay.load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    helper["save_csv"](assay.OUT / "records.csv", all_rows)
    assay.write_json(assay.OUT / "complete.json", dict(status="complete", records=46080, expected=46080,
        records_sha256=assay.sha(assay.OUT / "records.csv"), method_lock_sha256=assay.sha(assay.OUT / "method_lock.json"),
        source_sha256=spec["source_sha256"], execution_worker_sha256=assay.sha(Path(__file__)),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    assay.write_json(assay.OUT / "progress.json", dict(status="complete", completed_pair_cohorts=768,
        expected_pair_cohorts=768, records=46080, expected_records=46080))
    print(json.dumps(dict(status="complete", records=46080)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid source-pair worker assignment")
    lock_path = assay.OUT / "method_lock.json"
    spec = json.loads(lock_path.read_text("utf-8"))
    if assay.sha(ROOT / "scripts/run_future_semantic_assay.py") != spec["source_sha256"]:
        raise ValueError("The frozen future-task assay changed")
    if args.finalize_only:
        finish(spec)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = assay.FrozenEncoder(ROOT, assay.CONFIG)
    with np.load(assay.FIT / "prototypes.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    books = {}
    for h in (5, 3, 4):
        with np.load(ROOT / f"results/product_quantizer/known_only_v1/h{h}_m8.npz", allow_pickle=False) as stored:
            books[h] = assay.ProductQuantizer(stored["centroids"])
    wire, helper = assay.load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    manifest = json.loads((ROOT / "data/second_locked_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    assigned_ids = spec["pair_ids"][args.worker_index::args.workers]
    pending = {(p, cohort) for p in assigned_ids for cohort in assay.SOURCES}
    expected_per_pair = {"image": 102, "hybrid": 66, "learned": 12}
    assay.write_json(assay.OUT / f"worker_{args.worker_index}_spec.json", dict(worker_source_sha256=assay.sha(Path(__file__)),
        original_assay_sha256=spec["source_sha256"], assignment="Frozen source-pair index modulo worker count, all cohorts together",
        workers=args.workers, worker_index=args.worker_index, encoder=encoder.metadata))
    while pending:
        progressed = False
        for pair_id in assigned_ids:
            for cohort in assay.SOURCES:
                key = pair_id, cohort
                if key not in pending:
                    continue
                source = assay.source_rows(cohort, pair_id)
                if source is None:
                    continue
                source_rows, source_hash = source
                records = validate(pair_id, cohort, source_hash)
                if records is None:
                    if assay.sha(ROOT / "scripts/run_future_semantic_assay.py") != spec["source_sha256"]:
                        raise ValueError("Future-task recipe changed during execution")
                    records = assay.evaluate(cohort, pairs[pair_id], source_rows, encoder, wire, helper, books, means, classes)
                    if len(records) != expected_per_pair[cohort]:
                        raise AssertionError("Incomplete future-task cohort")
                    folder = assay.OUT / "pairs" / pair_id / cohort
                    assay.write_json(folder / "records.json", records)
                    assay.write_json(folder / "complete.json", dict(status="complete", records=len(records),
                        records_sha256=assay.sha(folder / "records.json"), source_receipt_sha256=source_hash,
                        assay_lock_sha256=assay.sha(lock_path)))
                pending.remove(key)
                progressed = True
                assay.write_json(assay.OUT / f"worker_{args.worker_index}_progress.json", dict(status="in_progress",
                    remaining_cohorts=len(pending), expected_cohorts=3 * len(assigned_ids)))
                print(f"future worker {args.worker_index}: {3*len(assigned_ids)-len(pending)}/{3*len(assigned_ids)} cohorts complete", flush=True)
        if pending and not progressed:
            time.sleep(15)
    assay.write_json(assay.OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", cohorts=3 * len(assigned_ids),
        worker_source_sha256=assay.sha(Path(__file__)), assay_lock_sha256=assay.sha(lock_path)))


if __name__ == "__main__":
    main()
