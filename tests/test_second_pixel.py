"""Focused checks for evidence denominators, ROI alignment and routing gates."""
import unittest
import numpy as np

from scripts.run_second_pixel import evaluation_masks, image_metrics, pool_scores, ranking_scores


class SecondPixelTest(unittest.TestCase):
    def test_token_pool_matches_native_roi_indices(self):
        result = pool_scores(np.arange(256))
        self.assertEqual((result[0], result[1], result[8], result[-1]), (17, 19, 49, 255))

    def test_zero_unknown_does_not_certify_stability(self):
        gate = np.zeros(64, bool)
        gate[0] = True
        score = ranking_scores("unknown_only", np.zeros(64), np.zeros(64),
                               np.zeros(64), gate, 0, "test")
        self.assertAlmostEqual(score[0], 32.5 / 64)
        self.assertTrue(np.all(score[1:] == 0))

    def test_zero_ignore_and_prior_heldout_are_not_known(self):
        old = np.array([[0, 1, 5, 2], [1, 255, 3, 2]])
        current = np.array([[1, 5, 1, 255], [0, 5, 5, 1]])
        known, unknown = evaluation_masks(old, current, np.ones_like(old, bool), 5)
        self.assertEqual(np.argwhere(known).tolist(), [[1, 3]])
        self.assertEqual(np.argwhere(unknown).tolist(), [[0, 1], [1, 2]])

    def test_coverage_is_unknown_pixel_weighted_and_empty_is_missing(self):
        current = np.zeros((512, 512, 3), np.uint8)
        restored = current.copy()
        unknown = np.zeros((512, 512), bool)
        unknown[0, :20] = True
        unknown[0, 64] = True
        restored[0, 64] = 255
        result = image_metrics(current, restored, np.zeros_like(unknown), unknown)
        self.assertAlmostEqual(result["unknown_coverage_30"], 20 / 21)
        self.assertEqual(result["unknown_retained_pixels_30"], 20)
        self.assertAlmostEqual(result["unknown_mse"], 1 / 21)
        empty = image_metrics(current, restored, np.zeros_like(unknown), np.zeros_like(unknown))
        self.assertIsNone(empty["unknown_coverage_30"])
        self.assertIsNone(empty["unknown_mse"])
        self.assertIsNone(empty["known_mse"])


if __name__ == "__main__":
    unittest.main()
