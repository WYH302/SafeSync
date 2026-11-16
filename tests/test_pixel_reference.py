import hashlib
import numpy as np
import pytest
from safesync.reference import encode_reference,reference_score,bind_reference,unbind_reference,restore_coarse,exact_mse
from safesync.certificates import encode_certificate_reference,certificate_score
from safesync.pixel_wire import (PixelState,TransactionSender,TransactionReceiver,
                                encode_update,decode_update,UPDATED,STABLE,DEFERRED)


def fixture():
    a=np.random.default_rng(7).integers(0,256,(128,128,3),dtype=np.uint8)
    state=PixelState(a,hashlib.sha256(b"region-a").digest()[:16])
    return a,state


def test_equal_reference_bytes_and_small_self_distance():
    a,_=fixture()
    for side in (4,8,16):
        wires=[encode_reference(a,m,side) for m in ("thumbnail","projection","hybrid","dct_low","dct_spread")]
        assert len(set(map(len,wires)))==1
        for w in wires:
            assert np.max(reference_score(a,w))<2e-5
            bad=bytearray(w);bad[-1]^=1
            with pytest.raises(ValueError):reference_score(a,bytes(bad))


def test_pixel_stream_lossless_replay_and_ack_guard():
    a,state=fixture(); b=255-a
    sender=TransactionSender(state.version,state.digest())
    receiver=TransactionReceiver(state)
    args=dict(shape=a.shape[:2],block=64,region=state.region,observed_at=1,
              current_rgb=b,indices=[1],statuses=[STABLE,UPDATED,DEFERRED,DEFERRED],codec="png")
    packet,cost=sender.propose(**args)
    assert len(packet)==sum(cost.values())
    assert sender.digest==state.digest()
    with pytest.raises(ValueError):sender.propose(**args)
    receiver.expect(2)
    ack,patches=receiver.receive(packet)
    assert np.all(receiver.status()==0)
    assert np.array_equal(receiver.state.rgb[:64,64:],b[:64,64:])
    assert np.array_equal(receiver.state.rgb[64:],a[64:])
    duplicate,empty=receiver.receive(packet)
    assert duplicate==ack and empty==[]
    sender.acknowledge(ack)
    assert sender.version==receiver.state.version and sender.digest==receiver.state.digest()
    with pytest.raises(ValueError):sender.acknowledge(ack)
    with pytest.raises(ValueError):decode_update(receiver.state,packet)


def test_empty_update_cannot_create_current_pixels():
    a,state=fixture()
    p,c=encode_update(shape=a.shape[:2],block=64,region=state.region,
                     base_version=0,base_digest=state.digest(),observed_at=0,
                     current_rgb=None,indices=[],statuses=[DEFERRED]*4)
    after,ack,patches=decode_update(state,p)
    assert np.array_equal(after.rgb,a) and patches==[]
    wrong=PixelState(a,hashlib.sha256(b"other-region").digest()[:16])
    with pytest.raises(ValueError):decode_update(wrong,p)


def test_coarse_update_and_reference_binding():
    a,state=fixture();b=255-a
    packet,cost=encode_update(shape=a.shape[:2],block=64,region=state.region,
                     base_version=0,base_digest=state.digest(),observed_at=0,
                     current_rgb=b,indices=[],statuses=[STABLE]*4,coarse_side=4)
    decoded,_,_=decode_update(state,packet)
    assert np.array_equal(decoded.rgb,restore_coarse(a,b))
    wire=bind_reference(b"measurements",0,state.digest(),1)
    assert unbind_reference(wire,0,state.digest(),1)==b"measurements"
    with pytest.raises(ValueError):unbind_reference(wire,1,state.digest(),1)
    with pytest.raises(ValueError):unbind_reference(wire,0,state.digest(),2)


def test_dct_bound_covers_coarse_restoration_with_signed_details():
    a,_=fixture();b=255-a
    truth=exact_mse(b,restore_coarse(a,b))
    for side in (4,8,16):
        for method in ("dct_bound","hybrid_joint"):
            wire=encode_certificate_reference(a,method,side,seed=713,detail_side=4)
            point,upper,meta=certificate_score(b,wire)
            assert np.isfinite(upper).all() and np.min(upper)>=0
            if method=="dct_bound":assert np.all(truth<=upper+1e-6)
