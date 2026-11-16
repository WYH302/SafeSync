"""Separate ground evaluation of the fresh main and learned image receipts."""
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
import torch
from safesync.features import FrozenEncoder
from safesync.data import read_pair
from safesync import task_evidence_wire as wire
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import CONFIG, write_json, sha

MAIN = ROOT / "results/task_evidence/fresh_context_v1"
LEARNED = ROOT / "results/task_evidence/fresh_context_learned_v1"
OUT = ROOT / "results/task_evidence/fresh_context_ground_v1"
SOURCES = {"main": MAIN, "learned": LEARNED}


def source_pair(pair_id, cohort):
    folder = SOURCES[cohort] / "pairs" / pair_id
    receipt_path = folder / "complete.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if receipt["status"] != "complete" or sha(folder / "records.json") != receipt["records_sha256"]:
        raise ValueError("Fresh ground source receipt changed")
    return json.loads((folder / "records.json").read_text("utf-8")), sha(receipt_path)


def evaluate(pair, cohort, source, encoder, means, classes, helper):
    _, after, old, label, changed = read_pair(ROOT, pair)
    current = np.asarray(after)
    future_mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed
    priors, predictions = {}, {}
    result = []
    for row in source:
        heldout = int(row["heldout"])
        if heldout not in priors:
            packet = (ROOT / row["bootstrap_path"]).read_bytes()
            header = wire.HEADER.unpack_from(packet)
            blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), header[1], header[2])
            priors[heldout], _ = wire.receive(blank, packet)
        prior = priors[heldout]
        path = ROOT / row["packet_path"]
        if sha(path) != row["packet_sha256"] or path.stat().st_size != int(row["down_bytes"]):
            raise ValueError("Fresh actual packet or byte length changed")
        packet = path.read_bytes()
        if row["method"] == "old_only":
            state = prior
        elif cohort == "learned":
            png = ROOT / row["received_png"]
            if sha(png) != row["received_png_sha256"]:
                raise ValueError("Actual learned receiver PNG changed")
            rgb = np.asarray(Image.open(png).convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row["receiver_rgb_sha256"]:
                raise ValueError("Actual learned RGB changed")
            # The separate task process consumes the hash-bound decoder PNG.
            # Entropy reconstruction/replay belongs to the codec process.
            state, _ = wire.receive(prior, packet, learned_decoder=lambda base, payload: rgb)
        else:
            state, _ = wire.receive(prior, packet)
        identity = hashlib.sha256(state.rgb.tobytes()).hexdigest()
        if identity != row["receiver_rgb_sha256"] or hashlib.sha256(state.labels.tobytes()).hexdigest() != row["receiver_task_sha256"]:
            raise ValueError("Fresh received image or task state differs")
        known, unknown = helper["evaluation_masks"](old, label, changed, heldout)
        predicted_known = state.labels.repeat(32, 0).repeat(32, 1)
        known_matrix = np.bincount((label[known].astype(int)-1)*6+predicted_known[known].astype(int)-1, minlength=36).reshape(6, 6)
        if "known_confusion" in row and not np.array_equal(known_matrix, np.array(json.loads(row["known_confusion"]))):
            raise AssertionError("Independent fresh known-task counts differ")
        if identity not in predictions:
            features, _ = encoder.cached(Image.fromarray(state.rgb))
            prediction = classes[np.argmax(features @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
            predictions[identity] = np.bincount((label[future_mask].astype(int)-1)*6+prediction[future_mask].astype(int)-1, minlength=36).reshape(6, 6)
        keep = {key: row[key] for key in ("pair_id", "heldout", "method", "seed", "budget", "down_bytes", "ack_bytes", "total_bytes",
            "bootstrap_bytes", "packet_path", "packet_sha256", "receiver_rgb_sha256", "receiver_task_sha256")}
        quality_fields = helper["image_metrics"](current, state.rgb, known, unknown) if cohort == "learned" else {
            key: value for key, value in row.items() if key.startswith(("unknown_", "known_error_", "known_mse", "known_pixels", "global_"))}
        result.append(dict(**keep, cohort=cohort, known_confusion=json.dumps(known_matrix.tolist()),
            future_confusion=json.dumps(predictions[identity].tolist()), future_pixels=int(future_mask.sum()), **quality_fields))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    method_lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    prototype_path = ROOT / "results/future_semantic_probe/v1/prototypes.npz"
    if sha(prototype_path) != method_lock["later_ground_head_sha256"]:
        raise ValueError("Prelocked later ground head changed")
    spec = dict(source_sha256=sha(Path(__file__)), main_lock_sha256=sha(MAIN / "method_lock.json"),
        ground_head_sha256=sha(prototype_path), encoder_config=CONFIG,
        ground_use="Same all-class calibration head and explicit changed support as declared by the main joint acceptance criteria",
        isolation="No learned codec imports; consumes hash-verified receiver PNG for the learned control",
        expected_rows=48384, known_support="Both labels explicit 1..6, exclude heldout at either time", future_support="Both labels explicit 1..6 and binary changed true")
    spec_path = OUT / "evaluation_lock.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text("utf-8")) != spec:
            raise ValueError("Fresh ground wrapper changed")
    else:
        write_json(spec_path, spec)
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    if args.finalize_only:
        rows = []
        for pair_id in method_lock["pair_ids"]:
            for cohort in SOURCES:
                folder = OUT / "pairs" / pair_id / cohort
                receipt = json.loads((folder / "complete.json").read_text("utf-8"))
                source = source_pair(pair_id, cohort)
                if source is None or receipt["source_receipt_sha256"] != source[1] or sha(folder / "records.json") != receipt["records_sha256"]:
                    raise ValueError("Fresh ground cohort changed")
                rows.extend(json.loads((folder / "records.json").read_text("utf-8")))
        if len(rows) != 48384:
            raise AssertionError("Fresh ground matrix incomplete")
        helper["save_csv"](OUT / "records.csv", rows)
        write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=48384, expected=48384,
            records_sha256=sha(OUT / "records.csv"), evaluation_lock_sha256=sha(spec_path), completed_utc=datetime.now(timezone.utc).isoformat()))
        return
    if any(name.startswith("compressai") or name == "safesync.learned_codec" for name in sys.modules):
        raise ValueError("Learned codec imported into the fresh task process")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid ground source-pair assignment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(prototype_path, allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    assigned = method_lock["pair_ids"][args.worker_index::args.workers]
    pending = {(p, cohort) for p in assigned for cohort in SOURCES}
    while pending:
        manifest_path = ROOT / "data/second_task_context_test_v1/manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {"pairs": []}
        pairs = {p["id"]: p for p in manifest["pairs"]}
        progressed = False
        for pair_id in assigned:
            for cohort in SOURCES:
                key = pair_id, cohort
                if key not in pending or pair_id not in pairs:
                    continue
                source = source_pair(pair_id, cohort)
                if source is None:
                    continue
                source_values, source_hash = source
                folder = OUT / "pairs" / pair_id / cohort
                receipt_path = folder / "complete.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text("utf-8"))
                    if receipt["source_receipt_sha256"] != source_hash or sha(folder / "records.json") != receipt["records_sha256"]:
                        raise ValueError("Changed completed fresh ground cohort")
                else:
                    rows = evaluate(pairs[pair_id], cohort, source_values, encoder, means, classes, helper)
                    expected = 177 if cohort == "main" else 12
                    if len(rows) != expected:
                        raise AssertionError("Incomplete fresh ground source-pair cohort")
                    write_json(folder / "records.json", rows)
                    write_json(receipt_path, dict(status="complete", records=expected, evaluation_lock_sha256=sha(spec_path),
                        source_receipt_sha256=source_hash, records_sha256=sha(folder / "records.json")))
                pending.remove(key)
                progressed = True
                write_json(OUT / f"worker_{args.worker_index}_progress.json", dict(status="in_progress", remaining_cohorts=len(pending), expected_cohorts=2*len(assigned)))
                print(f"fresh ground worker {args.worker_index}: {2*len(assigned)-len(pending)}/{2*len(assigned)}", flush=True)
        if pending and not progressed:
            time.sleep(15)
    write_json(OUT / f"worker_{args.worker_index}_complete.json", dict(status="complete", cohorts=2*len(assigned), evaluation_lock_sha256=sha(spec_path)))


if __name__ == "__main__":
    main()
