"""SECOND mirror adapter. Zero is non-change, not a known land-cover class."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {0: "non-change / semantic unlabelled", 1: "low vegetation", 2: "ground", 3: "tree", 4: "water", 5: "building", 6: "sports field"}
# ChangeMamba mirror's native indices differ from several other SECOND loaders.
PALETTE = np.array([[255,255,255],[0,128,0],[128,128,128],[0,255,0],[0,0,255],[128,0,0],[255,0,0]], dtype=np.uint8)


def load_manifest(root: Path):
    manifest = json.loads((root / "data/second_pilot/manifest.json").read_text(encoding="utf-8"))
    groups = [x["group_id"] for x in manifest["pairs"]]
    if len(groups) != len(set(groups)):
        raise ValueError("source image-pair groups overlap")
    if any(x["source_split"] != "train" for x in manifest["pairs"]):
        raise ValueError("official test data must remain untouched")
    return manifest


def verified_path(root: Path, item: dict, key: str):
    entry = item["files"][key]
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes workspace")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"data checksum changed: {path}")
    return path


def label_array(path: Path):
    with Image.open(path) as image:
        values = np.array(image)
    if values.ndim == 3:
        result = np.full(values.shape[:2], 255, np.uint8)
        for label, color in enumerate(PALETTE):
            result[np.all(values[..., :3] == color, axis=-1)] = label
        values = result
    if not set(np.unique(values)).issubset(set(range(7)) | {255}):
        raise ValueError(f"unrecognized semantic labels in {path}: {np.unique(values)}")
    return values.astype(np.uint8)


def block_fraction(mask: np.ndarray, grid: int):
    height, width = mask.shape
    if height % grid or width % grid:
        raise ValueError("image dimensions must be divisible by state grid")
    return mask.reshape(grid, height // grid, grid, width // grid).mean(axis=(1,3)).reshape(-1)


def read_pair(root: Path, item: dict):
    before = Image.open(verified_path(root, item, "before")).convert("RGB")
    after = Image.open(verified_path(root, item, "after")).convert("RGB")
    old_label = label_array(verified_path(root, item, "label_before"))
    label = label_array(verified_path(root, item, "label_after"))
    binary_raw = np.array(Image.open(verified_path(root, item, "binary_change")))
    if binary_raw.ndim != 2 or not set(np.unique(binary_raw)).issubset({0,1,255}):
        raise ValueError("unrecognized binary-change labels")
    changed = binary_raw > 0
    if not (old_label.shape == label.shape == changed.shape == (after.height, after.width)) or before.size != after.size:
        raise ValueError("image/label coordinate mismatch")
    return before, after, old_label, label, changed


def masks(old_label, label, changed, heldout, grid, known_purity):
    unknown_any = (label == heldout) | (old_label == heldout)
    valid_semantics = (label != 255) & (old_label != 255)
    current_known = (label > 0) & (label < 7) & (label != heldout)
    prior_known = (old_label > 0) & (old_label < 7) & (old_label != heldout)
    unknown_fraction = block_fraction(unknown_any, grid)
    return {
        "fit_current": (block_fraction(current_known, grid) >= known_purity) & (unknown_fraction == 0),
        "fit_before": (block_fraction(prior_known, grid) >= known_purity) & (unknown_fraction == 0),
        "unknown_pixels": (label == heldout) & changed & valid_semantics,
        "known_eval_pixels": (~unknown_any) & valid_semantics,
        "changed_units": block_fraction(changed, grid) >= 0.25,
        "known_eval_units": (unknown_fraction == 0) & (block_fraction(valid_semantics, grid) == 1),
        "diagnostic_unknown": block_fraction((label == heldout) & changed, grid) >= 0.25,
        "diagnostic_known": (block_fraction(current_known & changed, grid) >= 0.5) & (unknown_fraction == 0),
    }
