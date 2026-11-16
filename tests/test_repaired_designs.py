"""Unit tests for the four targeted experimental-design repairs."""
from __future__ import annotations

import hashlib
import unittest
import zlib

import numpy as np

from safesync.certificates import HEAD, joint_rows, q16
from safesync.layered_cache import DetailCache, default_config
from safesync.layered_wire import parse_frame, to_layered
from safesync.pixel_wire import DEFERRED, STABLE, UPDATED
from safesync.reference import blocks, expand_means, means
from scripts.run_sn7_repaired_design_experiments import choose_packet_decoupled, streamed_joint_measure


class RepairedDesignsTests(unittest.TestCase):
    def test_sc02_header_seed_and_zero_matrix_streaming_equivalence(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        cfg = default_config("gaussian", 20261029)
        cache = DetailCache.from_rgb(rgb, cfg)

        magic, _, _, blk, h, w, header_seed, count, payload_len = HEAD.unpack(cache.wire[: HEAD.size])
        self.assertEqual(magic, b"SC02")
        self.assertEqual(header_seed, 20261029)
        self.assertEqual((h, w, blk), (256, 256, 64))
        self.assertEqual(len(cache.wire), HEAD.size + payload_len + 16)

        x = blocks(rgb, 64).astype(np.float32) / 255.0
        x_det = x - expand_means(means(x, 4), 64)
        mat = joint_rows(64, count * 3, header_seed)
        precomputed = x_det.reshape(len(x_det), -1) @ mat
        streamed, chunk_bytes = streamed_joint_measure(x_det, count * 3, header_seed, chunk_size=512)

        self.assertLessEqual(chunk_bytes, 512 * count * 3 * 8)
        self.assertLess(float(np.max(np.abs(precomputed - streamed))), 1e-13)
        self.assertTrue(np.array_equal(q16(precomputed), q16(streamed)))

    def test_decoupled_certificate_and_update_gating(self):
        rng = np.random.default_rng(101)
        rgb = rng.integers(20, 220, size=(256, 256, 3), dtype=np.uint8)
        region = hashlib.sha256(b"decoupled-test").digest()[:16]

        class DummySender:
            version = 1
            digest = hashlib.sha256(b"base-digest").digest()[:16]
            pending = None

        sender = DummySender()
        tau = 10 ** (-25.0 / 10.0)
        point = np.array([0.02, 0.015, 0.01, 0.008] + [0.001] * 12, dtype=np.float64)
        upper = np.array([0.12, 0.09, 0.06, 0.05] + [0.006] * 8 + [0.002] * 4, dtype=np.float64)

        wire, _, selected, _ = choose_packet_decoupled(
            sender, rgb, region, 1, point, point, upper, tau, tau, budget=16384, quality=60
        )
        parsed = parse_frame(to_layered(wire), 60)
        statuses = parsed["statuses"]
        self.assertEqual(selected, 4)
        self.assertTrue(np.all(statuses[:4] == UPDATED))
        self.assertTrue(np.all(statuses[4:12] == DEFERRED))
        self.assertTrue(np.all(statuses[12:16] == STABLE))

    def test_lossless_int8_feature_residual_encoding(self):
        rng = np.random.default_rng(2026)
        f0 = rng.standard_normal((256, 384), dtype=np.float32)
        f0 /= np.linalg.norm(f0, axis=1, keepdims=True)
        noise = 0.012 * rng.standard_normal((256, 384), dtype=np.float32)
        f1 = f0 + noise
        f1 /= np.linalg.norm(f1, axis=1, keepdims=True)

        q0 = np.clip(np.rint(f0 * 127.0), -127, 127).astype(np.int16)
        q1 = np.clip(np.rint(f1 * 127.0), -127, 127).astype(np.int16)
        diff16 = q1 - q0
        self.assertLessEqual(int(np.max(np.abs(diff16))), 127)

        diff8 = diff16.astype(np.int8)
        restored = q0 + diff8.astype(np.int16)
        self.assertTrue(np.array_equal(restored, q1))

        bytes_naive_int8 = len(zlib.compress(q1.astype(np.int8).tobytes(), level=6))
        bytes_res_int8 = len(zlib.compress(diff8.tobytes(), level=6))
        self.assertLess(bytes_res_int8, bytes_naive_int8 * 0.85)


if __name__ == "__main__":
    unittest.main()
