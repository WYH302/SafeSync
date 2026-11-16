import hashlib
import numpy as np
import pytest
from safesync.pixel_wire import PixelState,checksum
from safesync.fusion import (FusionSender,FusionReceiver,descriptor,ground_reply,
    receive_fusion,HISTORY_GOOD,CURRENT_POOR,HEAD)


def test_archive_query_roundtrip_and_ack_contract():
    # A stationary, textured observation is a protocol fixture, not an EO result.
    current=np.random.default_rng(414).integers(0,256,(128,128,3),dtype=np.uint8)
    region=hashlib.sha256(b"fixture").digest()[:16]
    state=PixelState(current.copy(),region)
    archive=current.copy()
    receiver=FusionReceiver(state,archive)
    sender=FusionSender(descriptor(state,checksum(archive.tobytes())))
    callback=lambda query,design,side,seed:ground_reply(query,receiver.state,design,side,seed,archive)
    packet,query,reply,timing=sender.propose(current,1,8192,callback,method="gaussian",psnr=30,seed=17)
    assert timing["reused_blocks"]>0
    assert len(query)>0 and len(reply)>0 and len(packet)+len(query)<=8192
    with pytest.raises(ValueError):sender.propose(current,2,8192,callback)
    receiver.expect(2)
    ack=receiver.receive(packet)
    assert not receiver.current_status().any()
    assert receiver.receive(packet)==ack
    sender.acknowledge(ack)
    assert sender.desc.digest==receiver.state.digest()
    assert np.any(receiver.state.statuses==HISTORY_GOOD)
    with pytest.raises(ValueError):sender.acknowledge(ack)
    with pytest.raises(ValueError):receive_fusion(state,packet,255-archive)


def test_no_current_payload_preserves_old_pixels_and_marks_poor():
    old=np.zeros((128,128,3),np.uint8)
    current=np.random.default_rng(91).integers(0,256,old.shape,dtype=np.uint8)
    state=PixelState(old,hashlib.sha256(b"empty-budget").digest()[:16])
    receiver=FusionReceiver(state)
    sender=FusionSender(descriptor(state))
    def never(*args):raise AssertionError("no reference should be requested")
    packet,query,reply,timing=sender.propose(current,1,HEAD.size+4+16,never,method="jpeg")
    assert query==reply==b"" and timing["chosen_quality"]==0
    sender.acknowledge(receiver.receive(packet))
    assert np.array_equal(receiver.state.rgb,old)
    assert np.all(receiver.state.statuses==CURRENT_POOR)


def test_full_reference_uses_its_own_decoder_not_simulator_copy():
    a=np.random.default_rng(12).integers(0,256,(128,128,3),dtype=np.uint8)
    state=PixelState(a,hashlib.sha256(b"full-cache").digest()[:16])
    cache=PixelState(a.copy(),state.region)
    sender=FusionSender(descriptor(state,checksum(a.tobytes())),cache,a.copy())
    receiver=FusionReceiver(state,a.copy())
    packet,query,reply,timing=sender.propose(a,1,8192,None,method="full_reference",psnr=30)
    assert query==reply==b""
    assert timing["sender_persistent_reference_bytes"]==a.nbytes*2
    ack=receiver.receive(packet);sender.acknowledge(ack)
    assert np.array_equal(sender.full_cache.rgb,receiver.state.rgb)
