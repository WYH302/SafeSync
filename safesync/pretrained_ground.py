"""One fixed official SpaceNet7 winning-model component as an RGB evaluator."""
from __future__ import annotations

import collections.abc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image


class PretrainedGroundEvaluator:
    """Frozen motokimura exp_0200 Unet; supplied RGB only, no training.

    This is one author component, without ensemble, TTA, future-frame refinement,
    watershed or tracking. It is a communication-distortion task evaluator, not
    evidence of geographic generalization or a new unknown detector.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.folder = self.root / "resources/pretrained_ground"
        for path in [self.folder / ".deps", self.folder / ".legacy_deps", self.folder / "model_source"]:
            sys.path.insert(0, str(path))
        # timm 0.1.20 uses only this removed Python compatibility alias.
        if "torch._six" not in sys.modules:
            shim = types.ModuleType("torch._six")
            shim.container_abcs = collections.abc
            sys.modules["torch._six"] = shim
        import torch
        import torchvision
        import timm
        import cv2
        import segmentation_models_pytorch as smp
        if timm.__version__ != "0.1.20":
            raise ValueError("the author checkpoint requires the pinned timm 0.1.20 implementation")
        self.torch, self.cv2 = torch, cv2
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        source = json.loads((self.folder / "source_identity.json").read_text(encoding="utf-8"))
        architecture = json.loads((self.folder / "model_source_identity.json").read_text(encoding="utf-8"))
        receipt = json.loads((self.folder / "weight_receipt.json").read_text(encoding="utf-8"))
        checkpoint = self.folder / "model_best_exp_0200.pth"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("official checkpoint checksum changed")
        self.model = smp.Unet(encoder_name="timm-efficientnet-b3", encoder_weights=None,
                             decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None,
                             in_channels=3, classes=3, activation="sigmoid")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict({key.removeprefix("module."): value for key, value in saved.items()}, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.metadata = {
            "schema_version": "sn7-pretrained-ground-v1", "model": "motokimura exp_0200 single Unet EfficientNet-B3",
            "official_repository": source["repository"], "official_commit": source["commit"],
            "model_source_commit": architecture["commit"], "weight_sha256": receipt["sha256"],
            "weight_url": receipt["url"], "weight_bytes": receipt["bytes"],
            "preprocessing": {"RGB": "provided RGB only; drop any alpha via RGB conversion",
                              "padding": "center constant zero to 1024x1024, matching author PadIfNeeded",
                              "input_side": 3072, "scale": 3.0, "resize": "OpenCV INTER_LINEAR uint8 before normalization",
                              "mean_255": [0.485 * 255, 0.456 * 255, 0.406 * 255],
                              "std_255": [0.229 * 255, 0.224 * 255, 0.225 * 255],
                              "output": "OpenCV INTER_LINEAR 3072->1024, center crop to original HxW, footprint channel 0"},
            "building_threshold_default": 0.5, "change_score": "max(current building probability - reference building probability,0)",
            "model_frozen": not any(p.requires_grad for p in self.model.parameters()),
            "strict_loaded_state_tensors": len(saved), "parameters": sum(p.numel() for p in self.model.parameters()),
            "training_overlap": "Author trained five folds over the 60 official SN7 training AOIs; exp_0200 is fold0. Our four AOIs are official training AOIs, so potential/likely overlap must be disclosed. These are distortion-evaluation scores, never independent geographic generalization claims.",
            "omitted_author_components": ["other nine models", "TTA", "all-month/future-frame refinement", "watershed", "tracking"],
            "compatibility": "local torch._six.container_abcs alias for legacy timm; no numerical architecture/weight changes",
            "versions": {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
                         "cuda": torch.version.cuda, "timm": timm.__version__, "SMP_author_fork": smp.__version__,
                         "opencv": cv2.__version__, "numpy": np.__version__}}
        self.identity = json.dumps({key: self.metadata[key] for key in ["model", "official_commit", "model_source_commit", "weight_sha256", "preprocessing"]}, sort_keys=True)
        metadata_path = self.folder / "metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.cold_load_seconds = time.perf_counter() - started
        self.threshold, self.change_threshold = 0.5, 0.5
        calibration = self.folder / "calibration.json"
        if calibration.exists():
            fixed = json.loads(calibration.read_text(encoding="utf-8"))
            if fixed["weight_sha256"] != receipt["sha256"]:
                raise ValueError("calibration belongs to another pretrained checkpoint")
            self.threshold = float(fixed["building_threshold"])
            self.change_threshold = float(fixed["change_threshold"])
        self.last_timing = None

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def predict(self, image: Image.Image):
        if not isinstance(image, Image.Image):
            raise TypeError("predict expects the actual supplied PIL RGB image")
        if min(image.size) < 1 or max(image.size) > 1024:
            raise ValueError("author inference contract supports native SN7 images up to 1024x1024")
        self._sync()
        started = time.perf_counter()
        raw = np.asarray(image.convert("RGB"))
        height, width = raw.shape[:2]
        top, left = (1024 - height) // 2, (1024 - width) // 2
        padded = np.zeros((1024, 1024, 3), np.uint8)
        padded[top:top+height, left:left+width] = raw
        resized = self.cv2.resize(padded, (3072, 3072), interpolation=self.cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        # Preserve author's float64 constants, then _to_tensor float32 cast.
        normalized = ((resized - mean) / std).transpose(2, 0, 1).astype(np.float32)
        batch = self.torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        self._sync()
        prepared = time.perf_counter()
        with self.torch.inference_mode():
            predicted = self.model(batch)[0].permute(1, 2, 0).cpu().numpy()
        self._sync()
        inferred = time.perf_counter()
        restored = self.cv2.resize(predicted, (1024, 1024), interpolation=self.cv2.INTER_LINEAR)
        output = restored[top:top+height, left:left+width, 0].copy().astype(np.float32)
        self.last_timing = {"total_ms": (time.perf_counter() - started) * 1000,
                            "preprocessing_ms": (prepared - started) * 1000,
                            "model_and_output_transfer_ms": (inferred - prepared) * 1000,
                            "native_size": [width, height], "resized_input": [3072, 3072],
                            "device": self.device, "feature_cache_used": False,
                            "peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated() if self.device == "cuda" else None}
        if not np.isfinite(output).all() or output.min() < 0 or output.max() > 1:
            raise ValueError("invalid pretrained building probabilities")
        return output

    def cached_predict(self, image: Image.Image):
        started = time.perf_counter()
        rgb = image.convert("RGB")
        key = hashlib.sha256(self.identity.encode() + str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        path = self.folder / "predictions" / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                probabilities = saved["probabilities"].copy()
                inference_timing = json.loads(str(saved["timing"]))
            self.last_timing = {"operation": "cached_predict", "cache_hit": True,
                                "total_ms": (time.perf_counter() - started) * 1000,
                                "source_inference_timing": inference_timing}
            return probabilities
        probabilities = self.predict(rgb)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, probabilities=probabilities,
                            timing=json.dumps(self.last_timing), identity=self.identity)
        temporary.replace(path)
        return probabilities

    def predict_change(self, reference: Image.Image, current: Image.Image):
        if reference.size != current.size:
            raise ValueError("change inputs must share native dimensions")
        started = time.perf_counter()
        previous = self.predict(reference)
        reference_timing = self.last_timing
        identical = reference.convert("RGB").tobytes() == current.convert("RGB").tobytes()
        if identical:
            output, current_timing = np.zeros_like(previous), None
        else:
            output = np.maximum(self.predict(current) - previous, 0).astype(np.float32)
            current_timing = self.last_timing
        self.last_timing = {"operation": "predict_change", "total_ms": (time.perf_counter() - started) * 1000,
                            "reference_prediction": reference_timing, "current_prediction": current_timing,
                            "identical_receiver_RGB_inputs": identical, "feature_cache_used": False}
        return output
