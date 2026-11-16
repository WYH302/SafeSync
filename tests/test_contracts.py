import io

import numpy as np
import pytest
from PIL import Image

from safesync.data import block_fraction, label_array
from safesync.metrics import roc_auc_score
from safesync.image_wire import encode_image, receive_image
from safesync.routing import encode_rois, select_and_encode
from safesync.scoring import change_score, normalize, safety_gate
from safesync.wire import ReferenceMismatch, State, Status, WireError, encode, quantize, receive, signature


def example(seed=7):
    rng = np.random.default_rng(seed)
    old, now = normalize(rng.normal(size=(16, 8))), normalize(rng.normal(size=(16, 8)))
    state = State(quantize(old), 4, signature("unit-test"))
    return state, now


def test_roundtrip_and_actual_accounting():
    state, now = example()
    image = Image.new("RGB", (32,32), (10,50,90))
    rois = {2: encode_rois(image, 4)[2]}
    packet, counts = encode(state, quantize(now), [2,5], rois, np.zeros(16,bool), 1024)
    received = receive(state, packet)
    assert sum(counts.values()) == len(packet) == 1024
    assert np.array_equal(received.codes[[2,5]], quantize(now)[[2,5]])
    assert np.array_equal(received.codes[0], state.codes[0])
    assert received.statuses[0] == Status.DEFERRED
    assert np.array_equal(np.array(Image.open(io.BytesIO(received.rois[2]))), np.array(image.crop((16,0,24,8))))


@pytest.mark.parametrize("packet", [None, b""])
def test_no_stream_cannot_reveal_current_or_confirm_stability(packet):
    state, now = example()
    received = receive(state, packet)
    assert np.array_equal(received.codes, state.codes)
    assert not np.array_equal(received.codes, quantize(now))
    assert np.all(received.statuses == Status.STALE)
    assert received.version == state.version


def test_wrong_reference_and_model_are_rejected_atomically():
    state, now = example()
    packet, _ = encode(state, quantize(now), [0], {}, np.zeros(16,bool))
    incorrect = State(quantize(normalize(np.ones((16,8)))), 4, state.identity)
    original = incorrect.codes.copy()
    with pytest.raises(ReferenceMismatch):
        receive(incorrect, packet)
    assert np.array_equal(incorrect.codes, original)
    wrong_model = State(state.codes, 4, signature("other"))
    with pytest.raises(ReferenceMismatch):
        receive(wrong_model, packet)


def test_corruption_truncation_and_replay_rejected():
    state, now = example()
    packet, _ = encode(state, quantize(now), [0], {}, np.ones(16,bool))
    corrupted = bytearray(packet)
    corrupted[-6] ^= 1
    with pytest.raises(WireError):
        receive(state, bytes(corrupted))
    with pytest.raises(WireError):
        receive(state, packet[:-10])
    updated = receive(state, packet)
    with pytest.raises(ReferenceMismatch):
        receive(updated, packet)


def test_budget_starvation_never_becomes_stable():
    state, now = example()
    c, u = np.ones(16), np.ones(16)*3
    stable = safety_gate(c,u,.1,1.)
    image = Image.new("RGB", (32,32))
    packet, _ = select_and_encode(state,now,c,u,encode_rois(image,4),stable,160,.35)
    received = receive(state, packet)
    assert not np.any(received.statuses == Status.CONFIRMED_STABLE)
    assert np.any(received.statuses == Status.DEFERRED)


def test_change_confidence_drift_and_ack_all_participate_in_gate():
    assert not safety_gate([.5],[.1],.1,.2)[0]
    assert not safety_gate([.0],[.1],.1,.2,confidence=[.9])[0]
    assert not safety_gate([.0],[.1],.1,.2,drift=[.2])[0]
    assert not safety_gate([.0],[.1],.1,.2,acknowledgement_known=False)[0]
    assert safety_gate([.0],[.1],.1,.2)[0]


def test_anchor_can_recover_reference_mismatch():
    state, now = example()
    packet, _ = encode(state, quantize(now), list(range(16)), {}, np.zeros(16,bool), anchor=True)
    wrong = State(quantize(normalize(np.ones((16,8)))),4,state.identity)
    received = receive(wrong,packet)
    assert np.array_equal(received.codes, quantize(now))


def test_unknown_is_not_redefined_as_single_reference_distance():
    state, now = example()
    ref = normalize(state.features)
    distances = np.sum((normalize(now)-ref)**2,axis=-1)
    assert np.allclose(distances, 2*change_score(now,ref), atol=3e-7)


def test_closed_loop_delta_error_is_current_quantization_error():
    rng = np.random.default_rng(4)
    sender = receiver = np.zeros(8)
    maximum = 0.
    for feature in rng.normal(size=(1000,8)).cumsum(axis=0)*.03:
        residual = feature - sender
        encoded = np.round(residual/.01)*.01
        sender = sender + encoded
        receiver = receiver + encoded
        maximum = max(maximum, float(np.max(np.abs(receiver-feature))))
        assert np.allclose(receiver, sender)
    assert maximum <= .00500001


def test_zero_semantic_label_is_not_a_known_class(tmp_path):
    path = tmp_path / "label.png"
    Image.fromarray(np.zeros((32,32),np.uint8)).save(path)
    labels = label_array(path)
    assert np.all(labels == 0)
    assert not ((labels>0)&(labels<7)).any()
    assert np.array_equal(block_fraction(np.ones((32,32),bool),4),np.ones(16))


def test_source_palette_uses_mirror_indices(tmp_path):
    from safesync.data import CLASS_NAMES
    image=np.array([[[0,128,0],[0,255,0],[0,0,255]]],np.uint8)
    path=tmp_path/'colors.png'
    Image.fromarray(image).save(path)
    assert label_array(path).tolist()==[[1,3,4]]
    assert [CLASS_NAMES[i] for i in (1,3,4)]==['low vegetation','tree','water']


def test_auc_average_ranks_with_ties():
    assert roc_auc_score([False,False,True,True],[.1,.2,.2,.3])==.875
    assert roc_auc_score([False,True],[.1,.1])==.5


def test_image_payload_padding_and_budget_are_real():
    import zlib
    from safesync.wire import CRC
    state,now=example()
    image=Image.new('RGB',(32,32),(40,80,120))
    packet,counts=encode_image(state,image,80)
    frame=packet[:-CRC.size]+bytes(33)
    padded=frame+CRC.pack(zlib.crc32(frame))
    class Encoder:
        def cached(self,image):
            return now,{}
    restored,decoded,_=receive_image(state,padded,Encoder())
    assert decoded.size==image.size
    assert np.array_equal(restored.codes,quantize(now))
    assert len(padded)==sum(counts.values())+33


def test_anchor_is_bound_to_registered_region():
    state,now=example()
    packet,_=encode(state,quantize(now),list(range(16)),{},np.zeros(16,bool),anchor=True)
    other_region=State(state.codes,4,signature('another-geographic-region'))
    with pytest.raises(ReferenceMismatch):
        receive(other_region,packet)
