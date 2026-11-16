"""Separate fixed-head evaluation of actual TCM receiver reconstructions."""
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
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
import torch
from safesync.features import FrozenEncoder
from scripts.evaluate_task_context_confirmation import evaluate
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import sha, write_json

SOURCE = ROOT / "results/task_evidence/tcm_control_v1"
MAIN = ROOT / "results/task_evidence/residual_confirmation_v1"
OUT = ROOT / "results/task_evidence/tcm_ground_v1"


def main():
    if any(n.startswith("compressai") or n in ("safesync.learned_codec", "safesync.tcm_runtime") for n in sys.modules):
        raise RuntimeError("TCM evaluation must use an isolated task process")
    lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    control = json.loads((SOURCE / "method_lock.json").read_text("utf-8"))
    if sha(Path(__file__)) != control["source_sha256"]["scripts/evaluate_tcm_control.py"]:
        raise ValueError("Changed control ground source")
    prototype = ROOT / "results/future_semantic_probe/v1/prototypes.npz"
    if sha(prototype) != lock["later_ground_head_sha256"]:
        raise ValueError("Changed common later-use head")
    spec = dict(source_sha256=sha(Path(__file__)), control_lock_sha256=sha(SOURCE / "method_lock.json"),
        expected_records=3072, head_sha256=sha(prototype), inference="Hash-verified actual decoder PNG, common FP16 round-trip and common ground interface")
    spec_path = OUT / "evaluation_lock.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed TCM ground specification")
    write_json(spec_path, spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, lock["backbone"])

    def isolated_features(image):
        features, timing = encoder.extract(image)
        return features.astype(np.float16).astype(np.float32), timing

    encoder.cached = isolated_features
    with np.load(prototype, allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    pairs = {p["id"]: p for p in json.loads((ROOT / "data/second_residual_test_v1/manifest.json").read_text("utf-8"))["pairs"]}
    done, all_rows = set(), []
    while len(done) < 256:
        progressed = False
        for pair_id in lock["pair_ids"]:
            source = SOURCE / "pairs" / pair_id
            if pair_id in done or not (source / "complete.json").exists():
                continue
            receipt = json.loads((source / "complete.json").read_text("utf-8"))
            source_hash = sha(source / "records.json")
            if receipt["control_lock_sha256"] != spec["control_lock_sha256"] or receipt["records_sha256"] != source_hash:
                raise ValueError("Changed committed TCM pair")
            folder = OUT / "pairs" / pair_id
            if (folder / "complete.json").exists():
                saved = json.loads((folder / "complete.json").read_text("utf-8"))
                if saved["source_sha256"] != source_hash or saved["records_sha256"] != sha(folder / "records.json"):
                    raise ValueError("Changed completed TCM ground pair")
                rows = json.loads((folder / "records.json").read_text("utf-8"))
            else:
                for item in pairs[pair_id]["files"].values():
                    if sha(ROOT / item["path"]) != item["sha256"]:
                        raise ValueError("Changed ground labels or image")
                rows = evaluate(pairs[pair_id], "learned", json.loads((source / "records.json").read_text("utf-8")), encoder, means, classes, helper)
                if len(rows) != 12:
                    raise AssertionError("Incomplete TCM ground matrix")
                write_json(folder / "records.json", rows)
                write_json(folder / "complete.json", dict(status="complete", records=12, source_sha256=source_hash,
                    records_sha256=sha(folder / "records.json"), evaluation_lock_sha256=sha(spec_path)))
            all_rows.extend(rows)
            done.add(pair_id)
            progressed = True
            write_json(OUT / "progress.json", dict(completed_pairs=len(done), expected_pairs=256, records=len(all_rows)))
            print(f"TCM ground {len(done)}/256", flush=True)
        if not progressed:
            time.sleep(15)
    with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=len(all_rows), records_sha256=sha(OUT / "records.csv"),
        evaluation_lock_sha256=sha(spec_path), control_lock_sha256=spec["control_lock_sha256"]))


if __name__ == "__main__":
    main()
