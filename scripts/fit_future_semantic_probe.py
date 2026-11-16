"""Calibration-only head for a later ground semantic use of retained RGB."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from safesync.data import read_pair, block_fraction

OUT = ROOT / "results/future_semantic_probe/v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    feature_path = ROOT / "cache/second_scaled/features.npz"
    with np.load(feature_path, allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float64), stored["ids"].tolist()
    if ids != [p["id"] for p in manifest["pairs"]]:
        raise ValueError("Future probe calibration feature identities differ")
    by_class = {label: [] for label in range(1, 7)}
    calibration_ids = []
    for pair, feature in zip(manifest["pairs"], features):
        if pair["split"] != "calibration":
            continue
        calibration_ids.append(pair["id"])
        _, _, old, current, _ = read_pair(ROOT, pair)
        for temporal, label in enumerate((old, current)):
            for cls in by_class:
                mask = block_fraction(label == cls, 16) >= .75
                by_class[cls].append(feature[temporal][mask])
    means, labels, counts, missing = [], [], {}, []
    for cls, arrays in by_class.items():
        values = np.concatenate(arrays)
        counts[str(cls)] = len(values)
        if not len(values):
            missing.append(cls)
            continue
        mean = values.mean(0)
        means.append(mean / np.linalg.norm(mean))
        labels.append(cls)
    path = OUT / "prototypes.npz"
    np.savez_compressed(path, means=np.stack(means), labels=np.array(labels, np.uint8))
    spec = dict(status="complete", source_sha256=sha(Path(__file__)), source_manifest_sha256=sha(manifest_path),
        source_features_sha256=sha(feature_path), prototypes_sha256=sha(path), calibration_pair_ids=calibration_ids,
        calibration_class_vectors=counts, missing_classes=missing,
        purpose="Fixed later ground semantic use, including operationally held-out classes; additional exploratory endpoint",
        separation="All-class calibration labels train only this ground evaluation head; no held-out class enters the sender gallery, codebooks, or routing",
        evaluation="Six-class cosine means on actual received RGB DINO features; held-out-class F1 on explicitly labeled changed support; all classes and caps retained",
        backbone="same frozen DINOv2-S/14, 448 input, 16x16 normalized pooled tokens, nearest class mean, native 32-pixel expansion",
        purity=.75, fitted_utc=datetime.now(timezone.utc).isoformat())
    (OUT / "fit_receipt.json").write_text(json.dumps(spec, indent=2) + "\n", "utf-8")
    print(json.dumps(dict(status="complete", vectors=counts, missing=missing)), flush=True)


if __name__ == "__main__":
    main()
