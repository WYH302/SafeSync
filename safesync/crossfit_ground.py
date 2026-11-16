"""Author-fold OOF variant of the existing frozen SpaceNet RGB evaluator.

Only checkpoint selection/identity changes. The original evaluator and its
preprocessing/predict implementation remain untouched. Public best-epoch
weights used their validation folds for selection: these are not blind tests.
"""
from __future__ import annotations
import collections.abc
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

from .pretrained_ground import PretrainedGroundEvaluator


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CrossfitGroundEvaluator(PretrainedGroundEvaluator):
    def __init__(self, root, aoi_id, *, _allow_unfrozen=False):
        self.root = Path(root)
        resources = self.root / "resources/pretrained_ground_crossfit"
        shared = self.root / "resources/pretrained_ground"
        selection_path = resources / "fold_selection.json"
        selection = json.loads(selection_path.read_text("utf-8"))
        targets = {row["aoi_id"]:row for row in selection["targets"]}
        if aoi_id not in targets:
            raise ValueError("AOI outside the fixed crossfit plan")
        binding = targets[aoi_id]
        if _allow_unfrozen and binding["role"] != "calibration":
            raise ValueError("Only the two original calibration AOIs may use unfrozen thresholds")
        receipts = json.loads((resources / "download_receipts.json").read_text("utf-8"))
        if not receipts["complete"]:
            raise ValueError("Public crossfit checkpoints are not completely verified")
        by_model = {row["experiment"]:row for row in receipts["weights"]}
        experiment = binding["selected_experiment"]
        receipt = by_model[experiment]
        checkpoint = self.root / receipt["path"]
        if file_sha(checkpoint) != receipt["sha256"]:
            raise ValueError("Author-fold checkpoint bytes changed")
        cpu = json.loads((resources / "cpu_tensor_metadata.json").read_text("utf-8"))
        if not cpu["checkpoint_shapes_all_match_exp0200"]:
            raise ValueError("Author-fold architecture signature mismatch")
        calibration_path = resources / "calibration.json"
        if _allow_unfrozen:
            calibration, phase = None, "original_two_AOI_calibration_only"
        else:
            if not calibration_path.exists():
                raise ValueError("OOF event threshold must be frozen before confirmation")
            calibration = json.loads(calibration_path.read_text("utf-8"))
            if calibration["fold_selection_sha256"] != file_sha(selection_path):
                raise ValueError("OOF calibration refers to another fold selection")
            if calibration["weight_sha256_by_experiment"][experiment] != receipt["sha256"]:
                raise ValueError("OOF calibration refers to another checkpoint")
            phase = "frozen_OOF_actual_receiver_evaluation"
        for path in (shared / ".deps", shared / ".legacy_deps", shared / "model_source"):
            sys.path.insert(0, str(path))
        if "torch._six" not in sys.modules:
            shim = types.ModuleType("torch._six")
            shim.container_abcs = collections.abc
            sys.modules["torch._six"] = shim
        import torch
        import cv2
        import timm
        import segmentation_models_pytorch as smp
        if timm.__version__ != "0.1.20":
            raise ValueError("The author checkpoint requires pinned timm 0.1.20")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        self.torch, self.cv2 = torch, cv2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        self.model = smp.Unet(encoder_name="timm-efficientnet-b3", encoder_weights=None,
            decoder_channels=(256,128,64,32,16), decoder_attention_type=None,
            in_channels=3, classes=3, activation="sigmoid")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        loaded = self.model.load_state_dict({key.removeprefix("module."):value for key,value in saved.items()}, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.cold_load_seconds = time.perf_counter() - started
        original = json.loads((shared / "metadata.json").read_text("utf-8"))
        self.metadata = dict(schema_version="sn7-author-fold-OOF-ground-v1",
            model=f"motokimura {experiment} single Unet EfficientNet-B3",
            official_repository=original["official_repository"], official_commit=original["official_commit"],
            model_source_commit=original["model_source_commit"], weight_sha256=receipt["sha256"],
            weight_bytes=receipt["bytes"], weight_url=receipt["url"], author_validation_fold=binding["author_validation_fold"],
            preprocessing=original["preprocessing"], strict_loaded_state_tensors=len(saved),
            parameters=sum(parameter.numel() for parameter in self.model.parameters()),
            model_frozen=not any(parameter.requires_grad for parameter in self.model.parameters()),
            missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys),
            scope="expected excluded from author gradient-training AOIs; validation fold selected best epoch; no untouched-test claim",
            inherited_predict_source_sha256=file_sha(self.root / "safesync/pretrained_ground.py"),
            numerical_flags=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
            versions=dict(torch=torch.__version__, timm=timm.__version__, SMP=smp.__version__, opencv=cv2.__version__))
        self.threshold = .5
        self.change_threshold = .5 if calibration is None else float(calibration["change_threshold"])
        identity = {key:self.metadata[key] for key in ("model", "official_commit", "model_source_commit", "weight_sha256", "preprocessing")}
        identity.update(author_validation_fold=binding["author_validation_fold"], phase=phase,
            thresholds={"static":.5, "event":None if calibration is None else self.change_threshold},
            calibration_sha256=None if calibration is None else file_sha(calibration_path),
            fold_selection_sha256=file_sha(selection_path))
        self.identity = json.dumps(identity, sort_keys=True)
        self.folder = resources / "models" / experiment
        self.folder.mkdir(parents=True, exist_ok=True)
        metadata_path = self.folder / "metadata.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text("utf-8")) != self.metadata:
            raise ValueError("Existing isolated crossfit model metadata differs")
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        self.aoi_id, self.experiment, self.binding = aoi_id, experiment, binding
        self.last_timing = None

    def close(self):
        self.model.to("cpu")
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
