"""Frozen official DINOv2-S feature extraction with content-addressed caches."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .scoring import normalize

WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


def download_weights(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    response = requests.get(WEIGHT_URL, stream=True, timeout=(20, 90))
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))
    received = 0
    with temporary.open("wb") as output:
        for chunk in response.iter_content(1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received % (10 * 1024 * 1024) < len(chunk):
                print(f"model weights: {received/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
    if total and received != total:
        raise RuntimeError("truncated pretrained checkpoint")
    temporary.replace(path)


class FrozenEncoder:
    def __init__(self, root: Path, config: dict):
        import torch
        self.torch, self.root, self.config = torch, root, config
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        start = time.perf_counter()
        weight = root / "resources/models/dinov2_vits14_pretrain.pth"
        download_weights(weight)
        weight_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
        source = root / "resources/dinov2"
        if not (source / "hubconf.py").exists():
            raise RuntimeError("clone official DINOv2 source into resources/dinov2 first")
        commit_file = source / "SOURCE_COMMIT.txt"
        commit = commit_file.read_text().strip() if commit_file.exists() else subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        self.model = torch.hub.load(str(source), "dinov2_vits14", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.identity = json.dumps({"name": "dinov2_vits14", "weight_sha256": weight_hash, "source_commit": commit, "input": config["backbone_input"], "pool": config["pool_patch_grid"], "resize": "PIL bicubic", "normalization": "ImageNet mean/std", "normalize_after_pool": True}, sort_keys=True)
        self.metadata = {"identity": json.loads(self.identity), "torch": torch.__version__, "cuda": torch.version.cuda, "device": self.device, "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None, "cold_load_seconds": time.perf_counter() - start, "backbone_frozen": not any(p.requires_grad for p in self.model.parameters()), "weight_source": WEIGHT_URL, "dinov3_access": "HTTP 401 GatedRepo; no account access used"}
        warm = Image.new("RGB", (512, 512))
        self.extract(warm)
        self.metadata["peak_allocated_gpu_bytes_after_warmup"] = torch.cuda.max_memory_allocated() if self.device == "cuda" else None

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def extract(self, image: Image.Image):
        torch = self.torch
        self.sync()
        start = time.perf_counter()
        side = self.config["backbone_input"]
        rgb = np.asarray(image.convert("RGB").resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        rgb = (rgb - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
        batch = torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
        self.sync()
        preprocessed = time.perf_counter()
        with torch.inference_mode():
            tokens = self.model.forward_features(batch)["x_norm_patchtokens"]
            patch_grid, group = side // 14, self.config["pool_patch_grid"]
            pooled = tokens.reshape(patch_grid // group, group, patch_grid // group, group, -1).mean(dim=(1,3)).reshape(-1, tokens.shape[-1])
        self.sync()
        inferred = time.perf_counter()
        output = normalize(pooled.float().cpu().numpy())
        finished = time.perf_counter()
        if output.shape != (self.config["grid"]**2, self.config["feature_dim"]):
            raise ValueError(f"unexpected frozen feature shape {output.shape}")
        return output, {"preprocessing_ms": (preprocessed-start)*1000, "backbone_ms": (inferred-preprocessed)*1000, "transfer_pool_normalize_ms": (finished-inferred)*1000, "total_ms": (finished-start)*1000}

    def cached(self, image: Image.Image):
        key = hashlib.sha256(self.identity.encode() + image.tobytes() + str(image.size).encode()).hexdigest()
        folder = self.root / "cache/features"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                values, timings = stored["features"].astype(np.float32), json.loads(str(stored["timings"]))
            return values, timings
        values, timings = self.extract(image)
        np.savez_compressed(path, features=values.astype(np.float16), timings=json.dumps(timings))
        return values.astype(np.float16).astype(np.float32), timings
