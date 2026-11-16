"""Fixed SECOND receiver-RGB binary-change heads, one per held-out class.

This module never reads task labels or sender-side current feature caches.
The internal feature interface is for fixed calibration/full-information replay.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .features import FrozenEncoder

ENCODER_CONFIG = {"seed": 20261028, "backbone_input": 448, "pool_patch_grid": 2,
                  "grid": 16, "feature_dim": 384}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_features(before, current) -> np.ndarray:
    """Identical FP16 round trip for original caches and decoded RGB inference."""
    before = np.asarray(before, np.float16).astype(np.float32)
    current = np.asarray(current, np.float16).astype(np.float32)
    if before.shape != (256, 384) or current.shape != before.shape:
        raise ValueError("SECOND head requires matching 16x16x384 frozen features")
    if not np.isfinite(before).all() or not np.isfinite(current).all():
        raise ValueError("nonfinite frozen features")
    return np.concatenate([before, current, np.abs(current - before)], axis=1)


def make_head(torch):
    return torch.nn.Sequential(torch.nn.Linear(1152, 64), torch.nn.GELU(),
                               torch.nn.Linear(64, 1))


class SecondGroundEvaluator:
    """Predict a native 512x512 score from supplied before/received-current RGB.

    The class-weighted sigmoid is a task score, not a calibrated posterior.
    Thresholds and all head parameters are fixed from calibration pairs only.
    Content caches are keyed by the actual RGB bytes plus backbone identity.
    """

    def __init__(self, root: str | Path, heldout: int, encoder: FrozenEncoder | None = None):
        import torch
        if heldout not in (5, 3, 4):
            raise ValueError("the fixed SECOND experiment has heldout folds 5, 3, 4")
        self.root, self.heldout = Path(root), int(heldout)
        self.torch, self.encoder = torch, encoder
        self.folder = self.root / "resources/second_ground_eval" / f"heldout_{heldout}"
        self.metadata = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        if not self.metadata.get("frozen") or self.metadata["heldout"] != heldout:
            raise ValueError("SECOND evaluator head/threshold are not frozen for this fold")
        if self.metadata["encoder_config"] != ENCODER_CONFIG:
            raise ValueError("SECOND evaluator preprocessing differs from the fixed configuration")
        checkpoint = self.folder / "head.npz"
        if sha_file(checkpoint) != self.metadata["head_sha256"]:
            raise ValueError("SECOND head checksum differs from frozen metadata")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = float(self.metadata["threshold"])
        self.head = make_head(torch).to(self.device)
        with np.load(checkpoint, allow_pickle=False) as stored:
            self.head.load_state_dict({key: torch.from_numpy(stored[key].copy()).to(self.device)
                                      for key in ("0.weight", "0.bias", "2.weight", "2.bias")})
            self.mean = torch.from_numpy(stored["feature_mean"].copy()).to(self.device)
            self.std = torch.from_numpy(stored["feature_std"].copy()).to(self.device)
        self.head.eval().requires_grad_(False)
        self.last_timing = None
        self._memory_features = {}

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = FrozenEncoder(self.root, ENCODER_CONFIG)
        if json.loads(self.encoder.identity) != self.metadata["backbone_identity"]:
            raise ValueError("RGB encoder identity differs from the frozen SECOND cache")
        return self.encoder

    def _predict_from_features(self, before, current) -> np.ndarray:
        """Internal cache replay only; public transmission evaluation uses predict."""
        torch = self.torch
        x = torch.from_numpy(pair_features(before, current)).to(self.device)
        with torch.inference_mode():
            scores = torch.sigmoid(self.head((x - self.mean) / self.std)).reshape(1, 1, 16, 16)
            scores = torch.nn.functional.interpolate(scores, size=(512, 512),
                                                      mode="bilinear", align_corners=False)
        return scores[0, 0].cpu().numpy().astype(np.float32)

    def _rgb_features(self, image: Image.Image):
        encoder = self._get_encoder()
        key = hashlib.sha256(encoder.identity.encode("utf-8") + image.tobytes()
                             + b"RGB:512x512").hexdigest()
        if key in self._memory_features:
            return self._memory_features[key], {"content_sha256": key, "cache": "memory"}
        folder = self.root / "resources/second_ground_eval/rgb_feature_cache"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                features = stored["features"].astype(np.float32)
            timing = {"content_sha256": key, "cache": "RGB-content-addressed disk"}
        else:
            features, timing = encoder.extract(image)
            features = features.astype(np.float16).astype(np.float32)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=features.astype(np.float16))
            temporary.replace(path)
            timing.update(content_sha256=key, cache="none; actual supplied RGB encoded")
        if features.shape != (256, 384) or not np.isfinite(features).all():
            raise ValueError("invalid content-addressed RGB feature cache")
        # Bounded RAM retention; disk cache remains reusable across heldout folds.
        if len(self._memory_features) >= 8:
            self._memory_features.pop(next(iter(self._memory_features)))
        self._memory_features[key] = features
        return features, timing

    def predict(self, before_PIL_RGB: Image.Image,
                received_current_PIL_RGB: Image.Image) -> np.ndarray:
        for image in (before_PIL_RGB, received_current_PIL_RGB):
            if not isinstance(image, Image.Image) or image.mode != "RGB":
                raise TypeError("SecondGroundEvaluator.predict requires two actual PIL RGB images")
            if image.size != (512, 512):
                raise ValueError("fixed SECOND evaluator requires native 512x512 RGB inputs")
        started = time.perf_counter()
        before, reference_timing = self._rgb_features(before_PIL_RGB)
        current, current_timing = self._rgb_features(received_current_PIL_RGB)
        output = self._predict_from_features(before, current)
        self.last_timing = {"total_ms": (time.perf_counter() - started) * 1000,
                            "before": reference_timing, "received_current": current_timing,
                            "input": "actual supplied RGB bytes only; no GT or sender cache lookup",
                            "native_shape": list(output.shape)}
        return output
