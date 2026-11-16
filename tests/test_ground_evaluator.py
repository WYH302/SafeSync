"""Small ground-evaluator contracts: observability, native tiling, and AP ties."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from safesync.ground import rgb_tiles, soft_building_targets


def test_edge_and_cloud_pixels_are_not_target_background():
    building = np.ones((6, 6), bool)
    valid = np.ones_like(building)
    valid[4, 4] = False
    target, weight = soft_building_targets(building, valid, x=4, y=4, side=8, grid=2)
    np.testing.assert_array_equal(target, [1, 0, 0, 0])
    np.testing.assert_array_equal(weight, [3/16, 0, 0, 0])


def test_native_tiles_preserve_nonmultiple_edges():
    image = Image.new("RGB", (514, 513), (17, 31, 59))
    tiles = list(rgb_tiles(image))
    assert [(t["width"], t["height"]) for t in tiles] == [(512, 512), (2, 512), (512, 1), (2, 1)]
    corner = np.asarray(tiles[-1]["image"])
    np.testing.assert_array_equal(corner[0, :2], [[17, 31, 59], [17, 31, 59]])
    assert not corner[1:].any() and not corner[:, 2:].any()


def test_exact_ap_groups_score_ties():
    path = Path(__file__).resolve().parents[1] / "scripts/train_ground_evaluator.py"
    spec = importlib.util.spec_from_file_location("train_ground_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    y = np.array([1, 0, 1, 0], bool)
    assert module.exact_average_precision(y, np.array([.9, .8, .8, .1])) == pytest.approx(5/6)
    assert module.exact_average_precision(y, np.full(4, .5)) == pytest.approx(.5)


def test_frozen_training_ids_use_only_calibration_aois():
    root = Path(__file__).resolve().parents[1]
    metadata = root / "resources/ground_eval/metadata.json"
    if not metadata.exists():
        pytest.skip("fitted optional evaluator is not present")
    fixed = json.loads(metadata.read_text(encoding="utf-8"))
    training = json.loads((metadata.parent / "training_ids.json").read_text(encoding="utf-8"))
    calibration, validation = set(fixed["calibration_aoi_ids"]), set(fixed["validation_aoi_ids"])
    assert fixed["frozen"] and len(calibration) == len(validation) == 2
    assert calibration.isdisjoint(validation)
    assert {row["aoi_id"] for row in training["tile_ids"]} == calibration
