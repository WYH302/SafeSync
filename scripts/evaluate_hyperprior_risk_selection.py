"""Separate fixed-head evaluation of the exploratory codec-score intervention."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"results/layered_tasks/.deps"))
import numpy as np
import torch
from safesync.features import FrozenEncoder
from scripts.evaluate_task_context_confirmation import evaluate
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import CONFIG,sha,write_json

SOURCE = ROOT/"results/task_evidence/hyperprior_risk_development_v1"
OUT = ROOT/"results/task_evidence/hyperprior_risk_ground_v1"


def main():
    if "safesync.learned_codec" in sys.modules or "compressai" in sys.modules:
        raise RuntimeError("The ground evaluator must use its own fixed numerical process")
    spec_path = SOURCE/"run_spec.json"
    spec = json.loads(spec_path.read_text("utf-8"))
    manifest = json.loads((ROOT/"data/second_task_context_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]:p for p in manifest["pairs"]}
    proposed = dict(source_spec_sha256=sha(spec_path), source_sha256=sha(Path(__file__)),
        fixed_evaluator_sha256=sha(ROOT/"scripts/evaluate_task_context_confirmation.py"),
        later_head_sha256=sha(ROOT/"results/future_semantic_probe/v1/prototypes.npz"),
        scope="Exploratory source cohort, actual decoder PNG and TE01 packets, fixed all-class head, no codec imports")
    OUT.mkdir(parents=True,exist_ok=True)
    lock_path = OUT/"evaluation_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text("utf-8")) != proposed:
        raise ValueError("Changed exploratory ground evaluation")
    write_json(lock_path,proposed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT,CONFIG)
    with np.load(ROOT/"results/future_semantic_probe/v1/prototypes.npz",allow_pickle=False) as stored:
        means,classes = stored["means"],stored["labels"]
    _,helper = load_frozen(json.loads((ROOT/"results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    done,all_rows = set(),[]
    while len(done) < len(spec["pair_ids"]):
        progressed = False
        for pair_id in spec["pair_ids"]:
            if pair_id in done:
                continue
            source = SOURCE/"pairs"/pair_id
            if not (source/"complete.json").exists():
                continue
            receipt = json.loads((source/"complete.json").read_text("utf-8"))
            source_hash = sha(source/"records.json")
            if receipt["records_sha256"] != source_hash or receipt["sender_commit_sha256"] != sha(source/"sender_commit.json"):
                raise ValueError("Changed committed codec-score transaction")
            folder = OUT/"pairs"/pair_id
            if (folder/"complete.json").exists():
                completed = json.loads((folder/"complete.json").read_text("utf-8"))
                if completed["source_records_sha256"] != source_hash or completed["records_sha256"] != sha(folder/"records.json"):
                    raise ValueError("Changed completed score-only ground output")
                rows = json.loads((folder/"records.json").read_text("utf-8"))
            else:
                rows = evaluate(pairs[pair_id],"learned",json.loads((source/"records.json").read_text("utf-8")),encoder,means,classes,helper)
                if len(rows) != 60:
                    raise AssertionError("Incomplete exploratory pair evaluation")
                write_json(folder/"records.json",rows)
                write_json(folder/"complete.json",dict(status="complete",records=60,source_records_sha256=source_hash,
                    records_sha256=sha(folder/"records.json"),evaluation_lock_sha256=sha(lock_path)))
            all_rows.extend(rows)
            done.add(pair_id)
            progressed = True
            write_json(OUT/"progress.json",dict(completed_pairs=len(done),expected_pairs=64,records=len(all_rows)))
            print(f"score-only ground assay {len(done)}/64",flush=True)
        if not progressed:
            time.sleep(10)
    with (OUT/"records.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(OUT/"complete.json",dict(status="complete",pairs=64,records=len(all_rows),records_sha256=sha(OUT/"records.csv"),
        evaluation_lock_sha256=sha(lock_path),scope="Exploratory, previously inspected source cohort"))


if __name__ == "__main__":
    main()
