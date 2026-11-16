"""Focused actual-codec checks for the residual image enhancement payload."""
import io
import struct
import unittest

import numpy as np
from PIL import Image

from safesync import residual_evidence as evidence


def png(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream,format="PNG")
    return stream.getvalue()


def decode_png(payload):
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))


class ResidualEvidenceTests(unittest.TestCase):
    def test_actual_codec_and_unselected_pixels(self):
        base = np.full((128,128,3),120,np.uint8)
        current = base.copy()
        current[:64,:64] = 170
        for mode in (evidence.ABSOLUTE,evidence.RESIDUAL):
            payload = evidence.encode(png(base),current,base,[0],mode=mode)
            restored = evidence.decode(payload,(128,128),decode_png)
            self.assertTrue(np.array_equal(restored[64:],base[64:]))
            self.assertTrue(np.array_equal(restored[:64,64:],base[:64,64:]))
            self.assertLess(np.abs(restored[:64,:64].astype(float)-170).mean(),8)

    def test_empty_update_uses_only_the_registered_base(self):
        base = np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3)
        payload = evidence.encode(png(base),np.zeros_like(base),base,[])
        self.assertTrue(np.array_equal(evidence.decode(payload,(128,128),decode_png),base))

    def test_invalid_lengths_and_duplicate_indices(self):
        base = np.zeros((128,128,3),np.uint8)
        with self.assertRaises(ValueError):
            evidence.encode(png(base),base,base,[0,0])
        payload = evidence.encode(png(base),base,base,[0])
        with self.assertRaises(ValueError):
            evidence.decode(payload[:-1],(128,128),decode_png)
        bad = bytearray(payload)
        struct.pack_into("<H",bad,evidence.HEADER.size,4)
        with self.assertRaises(ValueError):
            evidence.decode(bytes(bad),(128,128),decode_png)


if __name__ == "__main__":
    unittest.main()
