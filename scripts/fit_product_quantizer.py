"""Fit standard known-only PQ and measure its retained known-task utility."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from safesync.data import read_pair
from safesync.product_quantizer import ProductQuantizer

OUT = ROOT / "results/product_quantizer/known_only_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(matrix, heldout):
    denominator = matrix.sum(0) + matrix.sum(1)
    values = [2 * matrix[k, k] / denominator[k] for k in range(6) if k != heldout - 1 and denominator[k] > 0]
    return float(np.mean(values)) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    if ids != [p["id"] for p in manifest["pairs"]]:
        raise ValueError("Mismatched frozen feature IDs")
    spec = dict(source_sha256=sha(Path(__file__)), codec_sha256=sha(ROOT / "safesync/product_quantizer.py"),
                manifest_sha256=sha(manifest_path), features_sha256=sha(ROOT / "cache/second_scaled/features.npz"),
                subspaces=[8, 16], centroids_per_subspace=256, kmeans_seed=20261104,
                fitting="Frozen known-only calibration galleries, excluding heldout at both times",
                development_evaluation="Known-class cosine prototype task on previously inspected validation pairs",
                scope="Standard quantizer component assessment; formal test of other locked method continues independently")
    path = OUT / "run_spec.json"
    if path.exists() and json.loads(path.read_text("utf-8")) != spec:
        raise ValueError("Changed PQ component specification")
    path.write_text(json.dumps(spec, indent=2) + "\n", "utf-8")
    labels = {}
    for i, pair in enumerate(manifest["pairs"]):
        if pair["split"] == "validation":
            _, _, old, current, _ = read_pair(ROOT, pair)
            labels[i] = old, current
    report = []
    for heldout in (5, 3, 4):
        gallery_path = ROOT / f"results/unknown_recovery/locked_test_v1/galleries/known_{heldout}.npz"
        with np.load(gallery_path, allow_pickle=False) as stored:
            bank = stored["features"]
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as stored:
            means, classes = stored["means"], stored["labels"]
        means = means / np.linalg.norm(means, axis=1, keepdims=True)
        for parts in (8, 16):
            book_path = OUT / f"h{heldout}_m{parts}.npz"
            start = time.perf_counter()
            if book_path.exists():
                with np.load(book_path, allow_pickle=False) as stored:
                    quantizer = ProductQuantizer(stored["centroids"])
            else:
                quantizer = ProductQuantizer.fit(bank, parts)
                np.savez_compressed(book_path, centroids=quantizer.centroids)
            fit_seconds = time.perf_counter() - start
            matrices = defaultdict(lambda: np.zeros((6, 6), np.int64))
            frame_bytes, xor_bytes, errors = [], [], []
            for i, (old, current) in labels.items():
                f = features[i]
                encoded = quantizer.encode(f.reshape(-1, 384)).reshape(2, 256, parts)
                reconstructed = quantizer.decode(encoded[1])
                absolute = quantizer.frame(encoded[1])
                residual = quantizer.frame(encoded[1], encoded[0])
                for packet, prior in ((absolute, None), (residual, encoded[0])):
                    recovered, _ = quantizer.receive(packet, prior)
                    if not np.array_equal(recovered, encoded[1]):
                        raise AssertionError("Actual PQ frame decode mismatch")
                frame_bytes.append(len(absolute))
                xor_bytes.append(len(residual))
                errors.append(float(np.mean((reconstructed - f[1]) ** 2)))
                mask = (old >= 1) & (old <= 6) & (current >= 1) & (current <= 6) & (old != heldout) & (current != heldout)
                int8 = np.rint(f[1] * 127).clip(-127, 127).astype(np.int8).astype(np.float32) / 127
                for method, values in (("full_feature", f[1]), ("int8", int8), ("PQ", reconstructed)):
                    predicted = classes[np.argmax(values @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                    matrices[method] += np.bincount((current[mask].astype(int) - 1) * 6 + predicted[mask].astype(int) - 1, minlength=36).reshape(6, 6)
            values = dict(heldout=heldout, parts=parts, gallery_sha256=sha(gallery_path), gallery_vectors=len(bank),
                          codebook_path=book_path.relative_to(ROOT).as_posix(), codebook_sha256=sha(book_path),
                          codebook_raw_bytes=quantizer.centroids.nbytes, codebook_file_bytes=book_path.stat().st_size,
                          fit_seconds=fit_seconds, evaluation_pairs=len(labels),
                          mean_absolute_feature_frame_bytes=float(np.mean(frame_bytes)),
                          mean_xor_feature_frame_bytes=float(np.mean(xor_bytes)), mean_feature_mse=float(np.mean(errors)),
                          known_macro_f1={method: score(matrix, heldout) for method, matrix in matrices.items()},
                          confusion={method: matrix.tolist() for method, matrix in matrices.items()})
            report.append(values)
            print(json.dumps({k: values[k] for k in ("heldout", "parts", "fit_seconds", "mean_absolute_feature_frame_bytes", "mean_xor_feature_frame_bytes", "known_macro_f1")}), flush=True)
    (OUT / "complete.json").write_text(json.dumps(dict(status="complete", source_sha256=sha(Path(__file__)),
        spec_sha256=sha(path), rows=report), indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
