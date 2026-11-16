"""Frozen receiver-RGB pair evaluator for new building-ID footprints."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .features import FrozenEncoder
from .ground import GROUND_CONFIG, rgb_tiles


def pair_features(reference, current):
    """Same frozen FP16 token round trip for online RGB and cache replay."""
    reference = np.asarray(reference, np.float16).astype(np.float32)
    current = np.asarray(current, np.float16).astype(np.float32)
    if reference.shape != current.shape or reference.shape != (4096, 384):
        raise ValueError("pair evaluator requires matched 64x64 DINOv2-S tokens")
    return np.concatenate([reference, current, np.abs(current - reference)], axis=1)


def make_pair_head(torch):
    return torch.nn.Sequential(torch.nn.Linear(1152, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))


class GroundChangeEvaluator:
    """Predict new-building scores using only two actually supplied RGB images.

    The score is a sigmoid from class-weighted training, not a calibrated
    posterior probability. Use the saved calibration-only global threshold.
    """
    def __init__(self, root: str | Path, encoder: FrozenEncoder | None = None, *, _allow_unfrozen=False):
        import torch
        self.torch, self.root, self.encoder = torch, Path(root), encoder
        folder = self.root / "resources/ground_change_eval"
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if not self.metadata.get("frozen") and not _allow_unfrozen:
            raise ValueError("ground change calibration is incomplete")
        self.threshold = float(self.metadata["threshold"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = self.metadata["encoder_config"]
        checkpoint = folder / "head.npz"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.metadata["head_sha256"]:
            raise ValueError("pair head checksum differs from fixed metadata")
        self.head = make_pair_head(torch).to(self.device)
        with np.load(checkpoint, allow_pickle=False) as stored:
            self.head.load_state_dict({key: torch.from_numpy(stored[key].copy()).to(self.device)
                                      for key in ["0.weight", "0.bias", "2.weight", "2.bias"]}, strict=True)
            self.mean = torch.from_numpy(stored["feature_mean"].copy()).to(self.device)
            self.std = torch.from_numpy(stored["feature_std"].copy()).to(self.device)
        self.head.eval().requires_grad_(False)
        self.last_timing = None

    def _encoder(self):
        if self.encoder is None:
            self.encoder = FrozenEncoder(self.root, self.config)
        if json.loads(self.encoder.identity) != self.metadata["backbone_identity"]:
            raise ValueError("pair evaluator backbone/preprocessing identity differs")
        return self.encoder

    def tile_probabilities(self, reference, current):
        torch = self.torch
        features = pair_features(reference, current)
        with torch.inference_mode():
            tensor = torch.from_numpy(features).to(self.device)
            probabilities = torch.sigmoid(self.head((tensor - self.mean) / self.std)).reshape(1, 1, 64, 64)
            output = torch.nn.functional.interpolate(probabilities, size=(512, 512),
                                                      mode="bilinear", align_corners=False)
        return output[0, 0].cpu().numpy().astype(np.float32)

    def predict(self, reference: Image.Image, current: Image.Image):
        if not isinstance(reference, Image.Image) or not isinstance(current, Image.Image):
            raise TypeError("GroundChangeEvaluator.predict expects two PIL RGB images")
        if reference.size != current.size:
            raise ValueError("reference and received current RGB must share native dimensions")
        started = time.perf_counter()
        encoder = self._encoder()
        loaded = time.perf_counter()
        output = np.empty((reference.height, reference.width), np.float32)
        timings, reused = [], 0
        for first, second in zip(rgb_tiles(reference), rgb_tiles(current), strict=True):
            ref_features, ref_time = encoder.extract(first["image"])
            if first["image"].tobytes() == second["image"].tobytes():
                cur_features, cur_time = ref_features, None
                reused += 1
            else:
                cur_features, cur_time = encoder.extract(second["image"])
            scores = self.tile_probabilities(ref_features, cur_features)
            x, y, width, height = (first[key] for key in ["x", "y", "width", "height"])
            output[y:y+height, x:x+width] = scores[:height, :width]
            timings.append({"reference": ref_time, "current": cur_time})
        encoder.sync()
        self.last_timing = {"total_ms": (time.perf_counter() - started) * 1000,
                            "lazy_backbone_load_ms": (loaded - started) * 1000,
                            "native_size": list(reference.size), "tile_count": len(timings),
                            "identical_rgb_tiles_reused": reused, "feature_timings": timings,
                            "feature_cache_used": False, "input": "two supplied receiver RGB images only"}
        return output
