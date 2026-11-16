"""Targeted integrity checks for the joint feature/evidence frame."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from safesync.hybrid_wire import encode, receive
from safesync.product_quantizer import ProductQuantizer


def test_hybrid_feature_decode_and_session_checks():
    book = ProductQuantizer(np.zeros((8, 256, 48), np.float32))
    codes = np.zeros((256, 8), np.uint8)
    image = np.zeros((4, 4, 3), np.uint8)
    prior = SimpleNamespace(rgb=image, region=b"r" * 16, observed_at=0, digest=lambda: hashlib.sha256(image.tobytes()).digest()[:16])
    frame = encode(book.frame(codes), b"", prior.region, 1)
    features, restored, ack = receive(prior, frame, book, {})
    assert np.array_equal(features, book.decode(codes))
    assert np.array_equal(restored, image)
    assert len(ack) == 56
    with pytest.raises(ValueError, match="session"):
        receive(prior, encode(book.frame(codes), b"", b"s" * 16, 1), book, {})
    with pytest.raises(ValueError, match="integrity|checksum"):
        receive(prior, frame[:-1] + bytes([frame[-1] ^ 1]), book, {})
    prior.observed_at = 1
    with pytest.raises(ValueError, match="stale"):
        receive(prior, frame, book, {})


def test_pq_residual_exactness_and_reference_rejection():
    book = ProductQuantizer(np.zeros((8, 256, 48), np.float32))
    prior = np.zeros((256, 8), np.uint8)
    current = np.arange(256, dtype=np.uint8)[:, None].repeat(8, 1)
    packet = book.frame(current, prior, base=7, target=8, observation=13)
    restored, metadata = book.receive(packet, prior, expected_base=7)
    assert np.array_equal(restored, current)
    assert metadata == dict(version=8, observation=13)
    with pytest.raises(ValueError, match="reference"):
        book.receive(packet, np.ones_like(prior), expected_base=7)
    with pytest.raises(ValueError, match="version"):
        book.receive(packet, prior, expected_base=6)
