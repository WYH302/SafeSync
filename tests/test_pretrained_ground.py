"""Cheap receiver-input and event-score contracts, without loading a GPU model."""
import numpy as np
from PIL import Image

from safesync.pretrained_ground import PretrainedGroundEvaluator


def test_pretrained_change_uses_only_the_supplied_pair_and_clips_decreases():
    reference = Image.new("RGB", (2, 2), (0, 0, 0))
    current = Image.new("RGB", (2, 2), (1, 1, 1))
    evaluator = object.__new__(PretrainedGroundEvaluator)
    evaluator.last_timing = None
    calls = []
    before = np.array([[.8, .1], [.2, .7]], np.float32)
    after = np.array([[.2, .4], [.6, .1]], np.float32)
    def supplied_only(image):
        calls.append(image)
        return before if image is reference else after
    evaluator.predict = supplied_only
    score = evaluator.predict_change(reference, current)
    np.testing.assert_allclose(score, [[0, .3], [.4, 0]], atol=1e-7)
    assert calls == [reference, current]
    assert not evaluator.last_timing["identical_receiver_RGB_inputs"]


def test_actual_old_rgb_after_deleting_every_update_has_zero_positive_difference():
    reference = Image.new("RGB", (3, 2), (17, 31, 59))
    evaluator = object.__new__(PretrainedGroundEvaluator)
    evaluator.last_timing = None
    calls = []
    def initial_only(image):
        calls.append(image)
        return np.full((2, 3), .9, np.float32)
    evaluator.predict = initial_only
    score = evaluator.predict_change(reference, reference.copy())
    assert not score.any()
    assert calls == [reference]
    assert evaluator.last_timing["identical_receiver_RGB_inputs"]
