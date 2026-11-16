"""Follow committed context packets with the unchanged later ground head."""
from __future__ import annotations

import argparse
import csv
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
import torch
from PIL import Image
from safesync.features import FrozenEncoder
from safesync.data import read_pair
from safesync import task_evidence_wire as wire
from scripts.run_locked_second_test import CONFIG

SOURCE = ROOT / "results/task_evidence/context_development_v1"
BASE = ROOT / "results/task_evidence/development_v1"
OUT = ROOT / "results/task_evidence/context_future_probe_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", "utf-8")


def f1(matrix):
    return 2*matrix[..., 4, 4] / (matrix[..., 4, :].sum(-1) + matrix[..., :, 4].sum(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(ROOT / "results/future_semantic_probe/v1/prototypes.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    manifest = json.loads((ROOT / "data/second_scaled/manifest.json").read_text("utf-8"))
    pairs = [p for p in manifest["pairs"] if p["split"] == "validation"]
    methods = ("labels_context_mix", "labels_context_adaptive_mix", "labels_context_adaptive_change", "labels_full_webp")
    done, records = set(), []
    while len(done) < 64:
        progressed = False
        for pair in pairs:
            if pair["id"] in done:
                continue
            receipt_path = SOURCE / "pairs" / pair["id"] / "complete.json"
            if not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text("utf-8"))
            path = receipt_path.with_name("records.json")
            if sha(path) != receipt["records_sha256"]:
                raise ValueError("Changed context source receipt")
            folder = OUT / "pairs" / pair["id"]
            completed_path = folder / "complete.json"
            if completed_path.exists():
                completed = json.loads(completed_path.read_text("utf-8"))
                if completed["source_receipt_sha256"] != sha(receipt_path) or sha(folder / "records.json") != completed["records_sha256"]:
                    raise ValueError("Changed context future probe")
                values = json.loads((folder / "records.json").read_text("utf-8"))
            else:
                rows = json.loads(path.read_text("utf-8"))
                rows += json.loads((BASE / "pairs" / pair["id"] / "records.json").read_text("utf-8"))
                selected = {(int(r["budget"]), r["method"]): r for r in rows
                    if int(r["heldout"]) == 5 and int(r["budget"]) in (4096, 8192) and r["method"] in methods}
                _, _, old, label, changed = read_pair(ROOT, pair)
                mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed
                boot = (BASE / "pairs" / pair["id"] / "h5_labels/bootstrap.bin").read_bytes()
                header = wire.HEADER.unpack_from(boot)
                blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), header[1], header[2])
                prior, _ = wire.receive(blank, boot)
                predictions, values = {}, []
                for cap in (4096, 8192):
                    for method in methods:
                        row = selected[(cap, method)]
                        packet_path = ROOT / row["packet_path"]
                        if sha(packet_path) != row["packet_sha256"]:
                            raise ValueError("Changed context probe packet")
                        state, _ = wire.receive(prior, packet_path.read_bytes())
                        identity = hashlib.sha256(state.rgb.tobytes()).hexdigest()
                        if identity != row["receiver_rgb_sha256"]:
                            raise ValueError("Context probe actual RGB differs")
                        if identity not in predictions:
                            features, _ = encoder.cached(Image.fromarray(state.rgb))
                            prediction = classes[np.argmax(features @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                            predictions[identity] = np.bincount((label[mask].astype(int)-1)*6+prediction[mask].astype(int)-1, minlength=36).reshape(6, 6)
                        values.append(dict(pair_id=pair["id"], budget=cap, method=method,
                            confusion=json.dumps(predictions[identity].tolist()), packet_sha256=row["packet_sha256"]))
                save(folder / "records.json", values)
                save(completed_path, dict(status="complete", records=8, source_receipt_sha256=sha(receipt_path), records_sha256=sha(folder / "records.json")))
            records.extend(values)
            done.add(pair["id"])
            progressed = True
            print(f"context later-use {len(done)}/64, {len(records)}/512", flush=True)
        if len(done) < 64 and not progressed:
            if not args.follow:
                raise ValueError("Context source remains incomplete")
            time.sleep(10)
    with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary, contrasts = [], []
    for cap in (4096, 8192):
        ids = sorted(done)
        values = {(r["pair_id"], r["method"]): np.array(json.loads(r["confusion"]), float) for r in records if r["budget"] == cap}
        base = np.stack([values[(p, "labels_full_webp")] for p in ids])
        draws = np.random.default_rng(20261104).multinomial(64, np.full(64, 1/64), size=10000)
        for method in methods:
            new = np.stack([values[(p, method)] for p in ids])
            summary.append(dict(budget=cap, method=method, later_class_f1=float(f1(new.sum(0)))))
            distribution = f1((draws @ new.reshape(64, 36)).reshape(-1, 6, 6)) - f1((draws @ base.reshape(64, 36)).reshape(-1, 6, 6))
            low, high = np.quantile(distribution, [.0125, .9875])
            contrasts.append(dict(budget=cap, method=method, comparator="labels_full_webp", delta_later_class_f1=float(f1(new.sum(0))-f1(base.sum(0))), low=float(low), high=float(high)))
    result = dict(status="complete", records=512, source_sha256=sha(Path(__file__)), records_sha256=sha(OUT / "records.csv"),
        classifier_sha256=sha(ROOT / "results/future_semantic_probe/v1/prototypes.npz"),
        source_spec_sha256=sha(SOURCE / "run_spec.json"), scope="64 development pairs, unchanged later ground head, building class, two caps, 97.5% descriptive intervals",
        summary=summary, comparisons=contrasts)
    save(OUT / "analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
