"""Fixed receiver-RGB building evaluator; no labels enter prediction."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .features import FrozenEncoder

GROUND_CONFIG = {"seed": 20261028, "backbone_input": 896, "pool_patch_grid": 1,
                 "grid": 64, "feature_dim": 384, "tile_side": 512, "head_hidden": 64}


def rgb_tiles(image: Image.Image, side: int = 512):
    """Yield row-major, black-padded native tiles and their unpadded extents."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    for y in range(0, image.height, side):
        for x in range(0, image.width, side):
            width, height = min(side, image.width - x), min(side, image.height - y)
            tile = Image.new("RGB", (side, side), (0, 0, 0))
            tile.paste(image.crop((x, y, x + width, y + height)), (0, 0))
            yield {"x": x, "y": y, "width": width, "height": height, "image": tile}


def soft_building_targets(building, valid, x, y, side=512, grid=64):
    """Area targets and valid-area loss weights; edge padding is never valid."""
    building, valid = np.asarray(building, bool), np.asarray(valid, bool)
    if building.shape != valid.shape or side % grid:
        raise ValueError("incompatible masks or target grid")
    height, width = min(side, building.shape[0] - y), min(side, building.shape[1] - x)
    pad_valid = np.zeros((side, side), bool)
    pad_building = np.zeros_like(pad_valid)
    pad_valid[:height, :width] = valid[y:y+height, x:x+width]
    pad_building[:height, :width] = building[y:y+height, x:x+width] & pad_valid[:height, :width]
    block = side // grid
    counts = pad_valid.reshape(grid, block, grid, block).sum(axis=(1, 3)).reshape(-1)
    positives = pad_building.reshape(grid, block, grid, block).sum(axis=(1, 3)).reshape(-1)
    targets = positives / np.maximum(counts, 1)
    return targets.astype(np.float32), (counts / (block * block)).astype(np.float32)


def feature_cache_path(root: Path, encoder: FrozenEncoder, tile: Image.Image):
    key = hashlib.sha256(encoder.identity.encode() + tile.tobytes() + str(tile.size).encode()).hexdigest()
    return root / "resources/ground_eval/features" / f"{key}.npz"


def cached_ground_features(root: Path, encoder: FrozenEncoder, tile: Image.Image):
    """Cache only image-derived frozen features, never targets or validity."""
    path = feature_cache_path(root, encoder, tile)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            return stored["features"].astype(np.float32), json.loads(str(stored["timings"])), path
    values, timings = encoder.extract(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=values.astype(np.float16), timings=json.dumps(timings))
    temporary.replace(path)
    return values.astype(np.float16).astype(np.float32), timings, path


def make_ground_head(torch, hidden=64):
    return torch.nn.Sequential(torch.nn.Linear(384, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))


class GroundEvaluator:
    """Load frozen metadata/head and predict native H×W probabilities from RGB.

    ``predict`` accepts only the supplied image. Masks/labels are exclusively
    offline fitting/evaluation inputs. A lazy encoder avoids an unused GPU copy.
    """
    def __init__(self, root: str | Path, encoder: FrozenEncoder | None = None, *, _allow_unfrozen=False):
        import torch
        self.torch, self.root = torch, Path(root)
        folder = self.root / "resources/ground_eval"
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if not self.metadata.get("frozen") and not _allow_unfrozen:
            raise ValueError("ground evaluator calibration is incomplete")
        self.config = self.metadata["encoder_config"]
        self.threshold = float(self.metadata["threshold"])
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head = make_ground_head(torch, self.config["head_hidden"]).to(self.device)
        checkpoint = folder / "head.npz"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.metadata["head_sha256"]:
            raise ValueError("ground head checksum differs from fixed metadata")
        with np.load(checkpoint, allow_pickle=False) as stored:
            state = {key: torch.from_numpy(stored[key].copy()).to(self.device)
                     for key in ["0.weight", "0.bias", "2.weight", "2.bias"]}
            self.mean = torch.from_numpy(stored["feature_mean"].copy()).to(self.device)
            self.std = torch.from_numpy(stored["feature_std"].copy()).to(self.device)
        self.head.load_state_dict(state, strict=True)
        self.head.eval().requires_grad_(False)
        self.last_timing = None

    def _encoder(self):
        if self.encoder is None:
            self.encoder = FrozenEncoder(self.root, self.config)
        expected = self.metadata["backbone_identity"]
        if json.loads(self.encoder.identity) != expected:
            raise ValueError("ground evaluator requires the exact frozen backbone/preprocessing identity")
        return self.encoder

    def tile_probabilities(self, features):
        """Head on image-derived tokens; used for both cache replay and RGB path."""
        torch = self.torch
        # Use the same explicit FP16 cache round trip in online and replay paths.
        features = np.asarray(features, dtype=np.float16).astype(np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(features).to(self.device)
            logits = self.head((tensor - self.mean) / self.std)
            grid = self.config["grid"]
            probabilities = torch.sigmoid(logits).reshape(1, 1, grid, grid)
            expanded = torch.nn.functional.interpolate(probabilities,
                size=(self.config["tile_side"], self.config["tile_side"]),
                mode="bilinear", align_corners=False)
        return expanded[0, 0].cpu().numpy().astype(np.float32)

    def predict(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError("GroundEvaluator.predict expects a PIL RGB image")
        started = time.perf_counter()
        encoder = self._encoder()
        loaded = time.perf_counter()
        output = np.empty((image.height, image.width), np.float32)
        timings = []
        for tile in rgb_tiles(image, self.config["tile_side"]):
            features, timing = encoder.extract(tile["image"])
            probabilities = self.tile_probabilities(features)
            x, y, width, height = (tile[k] for k in ["x", "y", "width", "height"])
            output[y:y+height, x:x+width] = probabilities[:height, :width]
            timings.append(timing)
        if encoder.device == "cuda":
            self.torch.cuda.synchronize()
        self.last_timing = {"total_ms": (time.perf_counter() - started) * 1000,
                            "lazy_backbone_load_ms": (loaded - started) * 1000,
                            "tile_count": len(timings), "feature_timings": timings,
                            "input_size": list(image.size), "input": "provided RGB only",
                            "feature_cache_used": False}
        return output
