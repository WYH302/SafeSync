import io
import numpy as np
import pytest
from PIL import Image

from safesync.task_evidence_wire import (
    ACK_BYTES, DEFERRED, LEARNED, LAYERED, TaskEvidenceState, TaskEvidenceReceiver, encode, encode_atlas, layered_payload, receive,
)


def state():
    return TaskEvidenceState(np.zeros((128, 128, 3), np.uint8), np.zeros((16, 16), np.uint8), b"r" * 16, b"m" * 16)


def test_task_exactness_actual_bytes_and_missing_update_status():
    prior = state()
    labels = np.full((16, 16), 3, np.uint8)
    image = np.full((128, 128, 3), 96, np.uint8)
    payload = encode_atlas(image, [2], codec="webp", quality=70)
    packet, costs = encode(prior, labels, observation=0, payload=payload, indices=[2])
    receiver = TaskEvidenceReceiver(prior)
    receiver.expect(0)
    ack, applied = receiver.receive(packet)
    assert applied and len(ack) == ACK_BYTES
    assert len(packet) == sum(costs.values())
    assert np.array_equal(receiver.state.labels, labels)
    assert not np.any(receiver.state.rgb[:64])
    assert receiver.evidence_status.tolist() == [2, 2, 1, 2]
    duplicate_ack, applied = receiver.receive(packet)
    assert not applied and duplicate_ack == ack
    receiver.expect(1)
    assert not receiver.task_current
    assert np.all(receiver.evidence_status == DEFERRED)


def test_reference_and_integrity_rejection_preserve_state():
    prior = state()
    labels = np.ones((16, 16), np.uint8)
    packet, _ = encode(prior, labels, observation=0)
    receiver = TaskEvidenceReceiver(prior)
    with pytest.raises(ValueError, match="checksum"):
        receiver.receive(packet[:-1] + bytes([packet[-1] ^ 1]))
    assert receiver.state is prior
    other = state()
    other.model = b"x" * 16
    with pytest.raises(ValueError, match="reference|model"):
        receive(other, packet)
    restored, _ = receive(prior, packet)
    assert np.array_equal(restored.labels, labels)
    with pytest.raises(ValueError, match="reference"):
        receive(restored, packet)


def test_registered_full_decoder_is_required_and_bound_to_prior():
    prior = state()
    labels = np.ones((16, 16), np.uint8)
    packet, _ = encode(prior, labels, observation=0, payload=b"fixed entropy payload", image_mode=LEARNED)
    with pytest.raises(ValueError, match="registered decoder"):
        receive(prior, packet)

    def decoder(base, payload):
        assert base is prior
        assert payload == b"fixed entropy payload"
        return np.full_like(base.rgb, 12)

    restored, ack = receive(prior, packet, learned_decoder=decoder)
    assert np.all(restored.rgb == 12)
    assert np.all(restored.labels == 1)
    assert len(ack) == ACK_BYTES


def test_current_context_and_native_region_reconstruct_together():
    prior = state()
    labels = np.ones((16, 16), np.uint8)
    base = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 20, np.uint8)).save(base, format="PNG")
    current = np.full_like(prior.rgb, 96)
    atlas = encode_atlas(current, [2], codec="webp", quality=70)
    packet, costs = encode(prior, labels, observation=0, payload=layered_payload(base.getvalue(), atlas), indices=[2], image_mode=LAYERED)
    decoded, ack = receive(prior, packet)
    assert len(packet) == sum(costs.values())
    assert np.array_equal(decoded.labels, labels)
    assert np.all(decoded.rgb[:64] == 20)
    assert decoded.rgb[64:, :64].mean() > 80
    assert np.all(decoded.statuses == 1)
    assert len(ack) == ACK_BYTES
