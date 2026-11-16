"""Known-calibrated product quantization and exact index-stream transport."""
from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np

HEADER = struct.Struct("<4sBIIIHH32s16s16sI")


def digest(data):
    return hashlib.sha256(data).digest()[:16]


class ProductQuantizer:
    def __init__(self, centroids):
        self.centroids = np.asarray(centroids, dtype=np.float32)
        if self.centroids.ndim != 3 or self.centroids.shape[1] != 256:
            raise ValueError("Expected M x 256 x subdimension centroids")
        if not np.isfinite(self.centroids).all():
            raise ValueError("Non-finite codebook")
        self.parts, _, self.subdim = self.centroids.shape
        self.identity = hashlib.sha256(self.centroids.astype("<f4").tobytes()).digest()

    @classmethod
    def fit(cls, features, parts, seed=20261104):
        from sklearn.cluster import MiniBatchKMeans
        values = np.asarray(features, np.float32)
        if values.ndim != 2 or values.shape[1] % parts or len(values) < 256:
            raise ValueError("Insufficient or incompatible PQ training samples")
        subdim = values.shape[1] // parts
        centroids = []
        for part in range(parts):
            model = MiniBatchKMeans(n_clusters=256, batch_size=1024, n_init=1,
                                   max_iter=50, random_state=seed + part, reassignment_ratio=.01)
            model.fit(values[:, part * subdim:(part + 1) * subdim])
            centroids.append(model.cluster_centers_)
        return cls(np.stack(centroids))

    def encode(self, features):
        values = np.asarray(features, np.float32)
        if values.ndim != 2 or values.shape[1] != self.parts * self.subdim:
            raise ValueError("Feature shape differs from PQ codebook")
        codes = np.empty((len(values), self.parts), np.uint8)
        for part, centers in enumerate(self.centroids):
            block = values[:, part * self.subdim:(part + 1) * self.subdim]
            distance = (block * block).sum(1)[:, None] + (centers * centers).sum(1) - 2 * block @ centers.T
            codes[:, part] = distance.argmin(1).astype(np.uint8)
        return codes

    def decode(self, codes):
        values = np.asarray(codes)
        if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != self.parts:
            raise ValueError("PQ indices have wrong dimensions or type")
        return np.concatenate([centers[values[:, part]] for part, centers in enumerate(self.centroids)], axis=1)

    def frame(self, current, prior=None, *, base=0, target=1, observation=1):
        current = np.asarray(current)
        if current.dtype != np.uint8 or current.ndim != 2 or current.shape[1] != self.parts:
            raise ValueError("Bad current PQ index array")
        residual = prior is not None
        if residual and (prior.shape != current.shape or prior.dtype != np.uint8):
            raise ValueError("PQ reference differs")
        raw = np.bitwise_xor(current, prior).tobytes() if residual else current.tobytes()
        payload = zlib.compress(raw, 6)
        body = HEADER.pack(b"PQF1", int(residual), base, target, observation, *current.shape,
                           self.identity, digest(prior.tobytes()) if residual else bytes(16),
                           digest(current.tobytes()), len(payload)) + payload
        return body + digest(body)

    def receive(self, packet, prior=None, *, expected_base=0):
        if len(packet) < HEADER.size + 16 or digest(packet[:-16]) != packet[-16:]:
            raise ValueError("PQ frame integrity failure")
        magic, residual, base, target, obs, cells, parts, identity, old_hash, new_hash, size = HEADER.unpack_from(packet)
        if magic != b"PQF1" or residual not in (0, 1) or identity != self.identity or parts != self.parts:
            raise ValueError("PQ model or schema mismatch")
        if base != expected_base or target != base + 1 or len(packet) != HEADER.size + size + 16:
            raise ValueError("PQ version or length mismatch")
        raw = zlib.decompress(packet[HEADER.size:-16])
        if len(raw) != cells * parts:
            raise ValueError("PQ decoded length differs")
        current = np.frombuffer(raw, np.uint8).reshape(cells, parts).copy()
        if residual:
            if prior is None or prior.shape != current.shape or digest(prior.tobytes()) != old_hash:
                raise ValueError("PQ reference mismatch")
            current ^= prior
        if digest(current.tobytes()) != new_hash:
            raise ValueError("PQ target digest mismatch")
        return current, dict(version=target, observation=obs)
