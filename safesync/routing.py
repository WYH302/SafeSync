"""Online routing has no label arguments; oracle selection is kept offline."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .wire import ROI_HEADER, TOKEN_ID, State, encode, minimum_bytes, quantize


def encode_rois(image: Image.Image, grid: int, codec="PNG", quality=80) -> dict[int, bytes]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    result = {}
    for index in range(grid**2):
        row, col = divmod(index, grid)
        crop = rgb.crop((col * width // grid, row * height // grid, (col + 1) * width // grid, (row + 1) * height // grid))
        stream = io.BytesIO()
        if codec == "PNG":
            crop.save(stream, format="PNG", optimize=False)
        elif codec == "JPEG":
            crop.save(stream, format="JPEG", quality=quality, subsampling=0)
        else:
            raise ValueError("unsupported ROI codec")
        result[index] = stream.getvalue()
    return result


def select_and_encode(state: State, current: np.ndarray, change: np.ndarray, evidence_priority: np.ndarray | None, roi_candidates: dict[int, bytes], stable: np.ndarray, budget: int, evidence_fraction: float):
    """Greedy rank selection under exact byte costs, with explicit fixed framing.

    Change-only passes evidence_priority=None. Random/uniform/unknown pass
    different priority arrays and otherwise use exactly the same code path.
    """
    remaining = budget - minimum_bytes(state)
    if remaining < 0:
        raise ValueError("budget cannot carry mandatory header and status manifest")
    token_cost = TOKEN_ID.size + state.codes.shape[1]
    selected, rois = set(), {}
    evidence_used = 0
    evidence_cap = int(remaining * evidence_fraction)
    if evidence_priority is not None:
        priorities = np.asarray(evidence_priority)
        if priorities.shape != (len(current),) or not np.all(np.isfinite(priorities)):
            raise ValueError("invalid evidence priority")
        order = np.lexsort((np.arange(len(priorities)), -priorities))
        for index in order:
            index = int(index)
            cost = token_cost + ROI_HEADER.size + len(roi_candidates[index])
            if cost <= evidence_cap - evidence_used and cost <= remaining:
                selected.add(index)
                rois[index] = roi_candidates[index]
                remaining -= cost
                evidence_used += cost
    order = np.lexsort((np.arange(len(change)), -np.asarray(change)))
    for index in order:
        if remaining < token_cost:
            break
        if int(index) not in selected:
            selected.add(int(index))
            remaining -= token_cost
    return encode(state, quantize(current), sorted(selected), rois, stable, budget)
