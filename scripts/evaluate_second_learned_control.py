"""Separate fixed ground task on actual received learned-codec PNGs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
import torch
from safesync.features import FrozenEncoder
from safesync.data import read_pair
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import CONFIG, LOCK, sha, write_json, semantic_confusion

SOURCE = ROOT / "results/second_learned/confirmation_v1"
OUT = ROOT / "results/second_learned/tasks_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if any(name.startswith("compressai") or name == "safesync.learned_codec" for name in sys.modules):
        raise ValueError("Learned codec imported into the separate ground evaluator")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    lock = json.loads(LOCK.read_text("utf-8"))
    prototypes = {}
    for heldout in (5, 3, 4):
        path = ROOT / f"results/unknown_scaled/known_{heldout}.npz"
        if sha(path) != lock["prototypes"][str(heldout)]:
            raise ValueError("Original fixed known-task prototype changed")
        with np.load(path, allow_pickle=False) as stored:
            mean, labels = stored["means"], stored["labels"]
        prototypes[heldout] = mean / np.linalg.norm(mean, axis=1, keepdims=True), labels
    wire, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    manifest_path = ROOT / "data/second_locked_test_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    spec = dict(source_sha256=sha(Path(__file__)), original_lock_sha256=sha(LOCK),
        learned_codec_lock_sha256=sha(SOURCE / "method_lock.json"), input_manifest_sha256=sha(manifest_path),
        prototypes=lock["prototypes"], encoder=encoder.metadata["identity"],
        decode_input="Actual received PNG receipts produced from LC09 and counted bootstrap, with packet/PNG/RGB hashes",
        codec_imported=False, fit_parameters=False)
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed learned-image ground experiment")
    write_json(spec_path, spec)
    completed, all_rows = set(), []
    while len(completed) < 256:
        progressed = False
        for pair_id in lock["pair_ids"]:
            if pair_id in completed:
                continue
            source_folder = SOURCE / "pairs" / pair_id
            source_receipt = source_folder / "complete.json"
            if not source_receipt.exists():
                continue
            original_receipt = json.loads(source_receipt.read_text("utf-8"))
            if (original_receipt["status"] != "complete" or original_receipt["records"] != 4
                    or sha(source_folder / "records.json") != original_receipt["records_sha256"]):
                raise ValueError("Incomplete or changed sender/receiver source receipt")
            target = OUT / "pairs" / pair_id
            result_path = target / "records.json"
            receipt_path = target / "complete.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text("utf-8"))
                if receipt["source_receipt_sha256"] != sha(source_receipt) or sha(result_path) != receipt["records_sha256"]:
                    raise ValueError("Changed ground-task receipt")
                rows = json.loads(result_path.read_text("utf-8"))
            else:
                _, after, old, current_label, changed = read_pair(ROOT, pairs[pair_id])
                current = np.asarray(after)
                rows, feature_cache = [], {}
                for item in json.loads((source_folder / "records.json").read_text("utf-8")):
                    if sha(ROOT / item["packet_path"]) != item["packet_sha256"] or sha(ROOT / item["received_png"]) != item["received_png_sha256"]:
                        raise ValueError("Received image or actual packet changed")
                    rgb = np.asarray(Image.open(ROOT / item["received_png"]).convert("RGB"))
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != item["receiver_rgb_sha256"]:
                        raise ValueError("Received RGB does not match decoder receipt")
                    for heldout in (5, 3, 4):
                        known, unknown = helper["evaluation_masks"](old, current_label, changed, heldout)
                        matrix = semantic_confusion(encoder, rgb, prototypes[heldout], current_label, known, feature_cache)
                        rows.append(dict(pair_id=pair_id, heldout=heldout, method="hyperprior", seed=0, budget=item["budget"],
                            down_bytes=item["down_bytes"], ack_bytes=item["ack_bytes"], total_bytes=item["total_bytes"],
                            initial_down_bytes=item["initial_down_bytes"], initial_ack_bytes=item["initial_ack_bytes"],
                            packet_path=item["packet_path"], packet_sha256=item["packet_sha256"],
                            receiver_rgb_sha256=item["receiver_rgb_sha256"], confusion=json.dumps(matrix.tolist()),
                            codec_metadata=json.dumps(item["codec_metadata"], sort_keys=True),
                            **helper["image_metrics"](current, rgb, known, unknown)))
                if len(rows) != 12:
                    raise AssertionError("Incomplete learned ground-task pair")
                write_json(result_path, rows)
                write_json(receipt_path, dict(status="complete", records=12, spec_sha256=sha(spec_path),
                    records_sha256=sha(result_path), source_receipt_sha256=sha(source_receipt)))
            all_rows.extend(rows)
            completed.add(pair_id)
            progressed = True
            write_json(OUT / "progress.json", dict(status="in_progress", completed_pairs=len(completed), expected_pairs=256,
                records=len(all_rows), expected_records=3072))
            print(f"SECOND hyperprior ground {len(completed)}/256, {len(all_rows)} task observations", flush=True)
        if len(completed) == 256:
            break
        if not progressed:
            if not args.follow:
                raise ValueError("Learned-codec source remains incomplete")
            time.sleep(15)
    helper["save_csv"](OUT / "records.csv", all_rows)
    write_json(OUT / "complete.json", dict(status="complete", records=len(all_rows), expected=3072, pairs=256,
        source_sha256=sha(Path(__file__)), spec_sha256=sha(spec_path), records_sha256=sha(OUT / "records.csv"),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
