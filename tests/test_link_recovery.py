import unittest

import numpy as np

from scripts.run_real_link_recovery import digest, encode, receive


class LinkRecoveryTests(unittest.TestCase):
    def test_anchor_requires_same_session(self):
        previous = np.zeros((4, 8), np.int8)
        current = np.ones_like(previous)
        session = digest(b"area and frozen encoder A")
        packet = encode(previous, current, 0, 1, 0, True, session)
        with self.assertRaises(ValueError):
            receive(previous, 0, packet, None, None, digest(b"other area"))
        decoded, version, _, status = receive(previous, 0, packet, None, None, session)
        self.assertEqual((version, status), (1, "accepted"))
        np.testing.assert_array_equal(decoded, current)

    def test_closed_loop_residual_and_reference_rejection(self):
        rng = np.random.default_rng(41)
        state = np.zeros((4, 8), np.int8)
        session = digest(b"same region and encoder")
        for version in range(25):
            current = rng.integers(-127, 128, size=state.shape, dtype=np.int16).astype(np.int8)
            packet = encode(state, current, version, version + 1, version + 1, False, session)
            decoded, target, ack, status = receive(state, version, packet, None, None, session)
            self.assertEqual((target, status), (version + 1, "accepted"))
            np.testing.assert_array_equal(decoded, current)
            replay, _, _, status = receive(decoded, target, packet, packet, ack, session)
            self.assertEqual(status, "duplicate")
            np.testing.assert_array_equal(replay, current)
            _, _, _, status = receive(state, version + 1, packet, None, None, session)
            self.assertEqual(status, "rejected_reference")
            state = decoded


if __name__ == "__main__":
    unittest.main()
