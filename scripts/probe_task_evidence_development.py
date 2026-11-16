"""Later-class use check before locking the repaired fresh confirmation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

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

OUT = ROOT / "results/task_evidence/selector_future_probe_v1"
BASE = ROOT / "results/task_evidence/development_v1"
SELECTOR = ROOT / "results/task_evidence/selector_development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(matrix):
    denominator = matrix[..., 4, :].sum(-1) + matrix[..., :, 4].sum(-1)
    return 2 * matrix[..., 4, 4] / denominator


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(ROOT / "results/future_semantic_probe/v1/prototypes.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    methods = ("labels_full_webp", "labels_webp_mix", "labels_adaptive_mix", "labels_adaptive_change")
    candidates = []
    for folder in (BASE, SELECTOR):
        candidates += [r for r in csv.DictReader((folder / "records.csv").open(encoding="utf-8"))
            if r["heldout"] == "5" and r["budget"] in ("4096", "8192") and r["method"] in methods]
    source = {(r["pair_id"], int(r["budget"]), r["method"]): r for r in candidates}
    manifest = json.loads((ROOT / "data/second_scaled/manifest.json").read_text("utf-8"))
    records = []
    for pair in [p for p in manifest["pairs"] if p["split"] == "validation"]:
        _, _, old, label, changed = read_pair(ROOT, pair)
        mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed
        boot = BASE / "pairs" / pair["id"] / "h5_labels/bootstrap.bin"
        raw = boot.read_bytes()
        head = wire.HEADER.unpack_from(raw)
        blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), head[1], head[2])
        prior, _ = wire.receive(blank, raw)
        predictions = {}
        for cap in (4096, 8192):
            for method in methods:
                row = source[(pair["id"], cap, method)]
                path = ROOT / row["packet_path"]
                if sha(path) != row["packet_sha256"]:
                    raise ValueError("Probe actual packet changed")
                state, _ = wire.receive(prior, path.read_bytes())
                identity = hashlib.sha256(state.rgb.tobytes()).hexdigest()
                if identity != row["receiver_rgb_sha256"]:
                    raise ValueError("Probe RGB differs from actual receiver")
                if identity not in predictions:
                    features, _ = encoder.cached(Image.fromarray(state.rgb))
                    prediction = classes[np.argmax(features @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                    predictions[identity] = np.bincount((label[mask].astype(int)-1)*6 + prediction[mask].astype(int)-1, minlength=36).reshape(6, 6)
                records.append(dict(pair_id=pair["id"], budget=cap, method=method,
                    confusion=json.dumps(predictions[identity].tolist()), packet_sha256=row["packet_sha256"]))
        print(f"later-use repair probe {pair['id']}: {len(records)}/512", flush=True)
    with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    comparisons, summary = [], []
    for cap in (4096, 8192):
        ids = sorted({r["pair_id"] for r in records})
        values = {(r["pair_id"], r["method"]): np.array(json.loads(r["confusion"]), float) for r in records if r["budget"] == cap}
        base = np.stack([values[(p, "labels_full_webp")] for p in ids])
        draw = np.random.default_rng(20261104).multinomial(64, np.full(64, 1/64), size=10000)
        for method in methods:
            new = np.stack([values[(p, method)] for p in ids])
            summary.append(dict(budget=cap, method=method, later_class_f1=float(f1(new.sum(0)))))
            distribution = f1((draw @ new.reshape(64, 36)).reshape(-1, 6, 6)) - f1((draw @ base.reshape(64, 36)).reshape(-1, 6, 6))
            low, high = np.quantile(distribution, [.0125, .9875])
            comparisons.append(dict(budget=cap, method=method, comparator="labels_full_webp",
                delta_later_class_f1=float(f1(new.sum(0))-f1(base.sum(0))), low=float(low), high=float(high)))
    result = dict(status="complete", records=512, source_sha256=sha(Path(__file__)), records_sha256=sha(OUT / "records.csv"),
        classifier_sha256=sha(ROOT / "results/future_semantic_probe/v1/prototypes.npz"),
        scope="64 development pairs, building target, 4/8 KiB, additional use endpoint before fresh-test lock; 97.5% descriptive two-cap intervals",
        summary=summary, comparisons=comparisons)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2)+"\n", "utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
