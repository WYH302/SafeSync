"""Prepare only sender RGB/features/task states for the new locked cohort."""
from __future__ import annotations

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
from scripts import run_task_context_confirmation as prepared
from scripts.run_locked_second_test import sha,write_json

OUT = ROOT/"results/task_evidence/residual_confirmation_v1"
DATA = ROOT/"data/second_residual_test_v1"
LOCK = OUT/"method_lock.json"


def main():
    lock = json.loads(LOCK.read_text("utf-8"))
    prepared.OUT,prepared.LOCK = OUT,LOCK
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT,lock["backbone"])
    if encoder.metadata["identity"]["weight_sha256"] != lock["model_weight_sha256"]:
        raise ValueError("New confirmation backbone differs")

    def isolated_features(image):
        values,timings = encoder.extract(image)
        return values.astype(np.float16).astype(np.float32),timings

    # Preserve the same FP16 round-trip while avoiding shared-file cache races.
    encoder.cached = isolated_features
    galleries,prototypes = {},{}
    for h in (5,3,4):
        item = lock["galleries"][str(h)]
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed locked known-only gallery")
        with np.load(ROOT/item["path"],allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"],device=encoder.device)
        path = ROOT/f"results/unknown_scaled/known_{h}.npz"
        if sha(path) != lock["prototypes"][str(h)]:
            raise ValueError("Changed locked task head")
        with np.load(path,allow_pickle=False) as stored:
            means,classes = stored["means"],stored["labels"]
        prototypes[h] = means/np.linalg.norm(means,axis=1,keepdims=True),classes
    done = set()
    while len(done) < 256:
        manifest_path = DATA/"manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {"pairs":[]}
        pairs = {p["id"]:p for p in manifest["pairs"]}
        progressed = False
        for pair_id in lock["pair_ids"]:
            if pair_id in done or pair_id not in pairs:
                continue
            for relative,expected in lock["source_sha256"].items():
                if sha(ROOT/relative) != expected:
                    raise ValueError("Residual implementation changed after the new lock")
            ready_path = OUT/"pairs"/pair_id/"sender_inputs_ready.json"
            if ready_path.exists():
                ready = json.loads(ready_path.read_text("utf-8"))
                if ready["lock_sha256"] != sha(LOCK) or ready["features_sha256"] != sha(ready_path.parent/"sender_features.npz"):
                    raise ValueError("Changed prepared sender input")
            else:
                prepared.prepare_pair(pairs[pair_id],encoder,galleries,prototypes)
            done.add(pair_id)
            progressed = True
            write_json(OUT/"sender_preparation_progress.json",dict(completed_pairs=len(done),expected_pairs=256))
            print(f"new residual sender inputs {len(done)}/256",flush=True)
        if not progressed:
            time.sleep(10)
    write_json(OUT/"sender_preparation_complete.json",dict(status="complete",pairs=256,lock_sha256=sha(LOCK),encoder=encoder.metadata,
        scope="Current/before RGB and fixed calibrated resources only, no evaluation-label reads"))


if __name__ == "__main__":
    main()
