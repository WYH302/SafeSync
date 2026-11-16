"""Known-only kNN and temporal-novelty scoring on retained frozen features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safesync.data import block_fraction, read_pair

OUT = ROOT / "results/unknown_recovery/scores_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    feature_path = ROOT / "cache/second_scaled/features.npz"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    with np.load(feature_path, allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    if ids != [p["id"] for p in manifest["pairs"]]:
        raise ValueError("Feature ID mismatch")
    spec = dict(source_sha256=sha(Path(__file__)), manifest_sha256=sha(manifest_path),
                feature_sha256=sha(feature_path), heldouts=[5, 3, 4], k=10,
                gallery="calibration explicit known classes, purity .75, no heldout in either time",
                candidates=["mahalanobis", "mahalanobis_emergence", "knn", "knn_emergence"],
                references=["https://proceedings.mlr.press/v162/sun22d.html"],
                definitions="kNN squared unit-normalized distance to tenth known neighbor; emergence=max(current novelty - prior novelty,0)",
                official_test_pixels_read=0)
    path = OUT / "run_spec.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != spec:
        raise ValueError("Scorer specification changed")
    path.write_text(json.dumps(spec, indent=2) + "\n", "utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    labels = []
    for pair in manifest["pairs"]:
        _, _, old, current, _ = read_pair(ROOT, pair)
        labels.append((old, current))
    receipts = []
    for heldout in (5, 3, 4):
        bank = []
        for pair, f, (old, current) in zip(manifest["pairs"], features, labels):
            if pair["split"] != "calibration":
                continue
            excluded = block_fraction((old == heldout) | (current == heldout), 16) > 0
            for t, semantic in enumerate((old, current)):
                for known in range(1, 7):
                    if known != heldout:
                        pure = (block_fraction(semantic == known, 16) >= .75) & ~excluded
                        bank.append(f[t][pure])
        bank = np.concatenate(bank)
        if len(bank) < 10:
            raise ValueError("Insufficient known calibration gallery")
        gallery = torch.tensor(bank, device=device)
        gallery = torch.nn.functional.normalize(gallery, dim=1)
        flat = features.reshape(-1, features.shape[-1])
        distances = []
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0, len(flat), 1024):
            query = torch.tensor(flat[offset:offset + 1024], device=device)
            query = torch.nn.functional.normalize(query, dim=1)
            neighbor = torch.topk(query @ gallery.T, k=10, dim=1, sorted=True).values[:, -1]
            distances.append(torch.clamp(2 - 2 * neighbor, min=0).cpu().numpy())
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        knn = np.concatenate(distances).reshape(features.shape[:-1])
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as model:
            means, precision = model["means"], model["precision"]
        x = flat.astype(np.float64)
        px = x @ precision
        distance = np.sum(px * x, axis=1)[:, None] - 2 * px @ means.T + np.sum((means @ precision) * means, axis=1)
        mahalanobis = np.maximum(distance.min(axis=1), 0).reshape(features.shape[:-1]) / features.shape[-1]
        destination = OUT / f"scores_{heldout}.npz"
        np.savez_compressed(destination, ids=np.array(ids), mahalanobis=mahalanobis[:, 1],
                            mahalanobis_emergence=np.maximum(mahalanobis[:, 1] - mahalanobis[:, 0], 0),
                            knn=knn[:, 1], knn_emergence=np.maximum(knn[:, 1] - knn[:, 0], 0))
        receipts.append(dict(heldout=heldout, path=destination.relative_to(ROOT).as_posix(), sha256=sha(destination),
                             gallery_vectors=len(bank), gallery_float32_bytes=bank.nbytes,
                             batch_score_seconds=seconds, device=device, torch=torch.__version__))
        print(f"known-only scorers heldout={heldout}, gallery={len(bank)}, batched kNN={seconds:.2f}s", flush=True)
    (OUT / "complete.json").write_text(json.dumps(dict(status="complete", spec_sha256=sha(path), receipts=receipts), indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
