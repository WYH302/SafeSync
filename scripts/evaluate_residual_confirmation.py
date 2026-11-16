"""Fixed-head ground evaluation of every actual new residual transaction."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"results/layered_tasks/.deps"))
import numpy as np
import torch
from safesync.features import FrozenEncoder
from scripts.evaluate_task_context_confirmation import evaluate
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import sha,write_json

SOURCE = ROOT/"results/task_evidence/residual_confirmation_v1"
DATA = ROOT/"data/second_residual_test_v1"
OUT = ROOT/"results/task_evidence/residual_confirmation_ground_v1"


def main():
    if "safesync.learned_codec" in sys.modules or "compressai" in sys.modules:
        raise RuntimeError("Ground tasks require a separate fixed numerical process")
    lock_path = SOURCE/"method_lock.json"
    lock = json.loads(lock_path.read_text("utf-8"))
    prototype = ROOT/"results/future_semantic_probe/v1/prototypes.npz"
    if sha(prototype) != lock["later_ground_head_sha256"]:
        raise ValueError("Changed locked later-use head")
    evaluation = dict(method_lock_sha256=sha(lock_path),source_sha256=sha(Path(__file__)),expected_records=29952,
        ground_head_sha256=sha(prototype),inference="Actual registered decoder PNG or actual full-image packet; same FP16 feature round-trip; no learned codec imports")
    OUT.mkdir(parents=True,exist_ok=True)
    spec_path = OUT/"evaluation_lock.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != evaluation:
        raise ValueError("Changed residual confirmation ground evaluation")
    write_json(spec_path,evaluation)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT,lock["backbone"])

    def isolated_features(image):
        values,timings = encoder.extract(image)
        return values.astype(np.float16).astype(np.float32),timings

    encoder.cached = isolated_features
    with np.load(prototype,allow_pickle=False) as stored:
        means,classes = stored["means"],stored["labels"]
    _,helper = load_frozen(json.loads((ROOT/"results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    done,all_rows = set(),[]
    while len(done) < 256:
        manifest_path = DATA/"manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {"pairs":[]}
        pairs = {p["id"]:p for p in manifest["pairs"]}
        progressed = False
        for pair_id in lock["pair_ids"]:
            source = SOURCE/"pairs"/pair_id
            if pair_id in done or pair_id not in pairs or not (source/"complete.json").exists():
                continue
            receipt = json.loads((source/"complete.json").read_text("utf-8"))
            source_hash = sha(source/"records.json")
            if (receipt["lock_sha256"] != sha(lock_path) or receipt["records_sha256"] != source_hash
                    or receipt["sender_commit_sha256"] != sha(source/"sender_commit.json")):
                raise ValueError("Changed committed residual source transactions")
            folder = OUT/"pairs"/pair_id
            if (folder/"complete.json").exists():
                saved = json.loads((folder/"complete.json").read_text("utf-8"))
                if saved["source_sha256"] != source_hash or saved["records_sha256"] != sha(folder/"records.json"):
                    raise ValueError("Changed completed residual ground pair")
                rows = json.loads((folder/"records.json").read_text("utf-8"))
            else:
                for item in pairs[pair_id]["files"].values():
                    if sha(ROOT/item["path"]) != item["sha256"]:
                        raise ValueError("Changed locked ground image or semantic label")
                source_rows = json.loads((source/"records.json").read_text("utf-8"))
                rows = evaluate(pairs[pair_id],"learned",source_rows,encoder,means,classes,helper)
                if len(rows) != 117:
                    raise AssertionError("Incomplete residual ground source pair")
                write_json(folder/"records.json",rows)
                write_json(folder/"complete.json",dict(status="complete",records=117,source_sha256=source_hash,
                    records_sha256=sha(folder/"records.json"),evaluation_lock_sha256=sha(spec_path)))
            all_rows.extend(rows)
            done.add(pair_id)
            progressed = True
            write_json(OUT/"progress.json",dict(completed_pairs=len(done),expected_pairs=256,records=len(all_rows)))
            print(f"locked residual ground {len(done)}/256",flush=True)
        if not progressed:
            time.sleep(10)
    with (OUT/"records.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(OUT/"complete.json",dict(status="complete",pairs=256,records=len(all_rows),records_sha256=sha(OUT/"records.csv"),
        evaluation_lock_sha256=sha(spec_path),method_lock_sha256=sha(lock_path)))


if __name__ == "__main__":
    main()
