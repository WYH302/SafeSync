import io,hashlib
import numpy as np
import pytest
from PIL import Image
from safesync.layered_cache import DetailCache,default_config
from safesync.layered_wire import LayeredSender,LayeredReceiver,parse_frame
from safesync.pixel_wire import encode_image
from safesync.reference import blocks,exact_mse


def test_local_cache_commit_matches_receiver_without_historical_pixels():
    rng=np.random.default_rng(73)
    before=rng.integers(0,256,(128,128,3),dtype=np.uint8)
    after=before.copy();after[:64,:64]=255-before[:64,:64]
    sender=LayeredSender();receiver=LayeredReceiver()
    boot,cost=sender.bootstrap(before,hashlib.sha256(b"area").digest()[:16],default_config())
    assert len(boot)==cost["config_bytes"]+sum(cost["components"].values())
    with pytest.raises(ValueError):sender.propose(after,1,8192)
    sender.acknowledge(receiver.receive(boot))
    assert np.array_equal(receiver.state.render(),before)
    assert sender.cache.wire==receiver.state.cache.wire
    for t in range(1,4):
        packet,meta=sender.propose(after,t,8192)
        assert len(packet)==sum(meta["cost"].values())<=8192
        assert meta["reference_up_bytes"]==0
        receiver.expect(t+1);ack=receiver.receive(packet)
        assert not receiver.status().any()
        assert receiver.receive(packet)==ack
        sender.acknowledge(ack)
        assert sender.cache.wire==receiver.state.cache.wire
        assert sender.history==receiver.state.history
        assert not hasattr(sender,"archive")
        with pytest.raises(ValueError):sender.acknowledge(ack)


def test_coarse_render_does_not_mutate_archived_detail():
    a=np.random.default_rng(7).integers(80,150,(128,128,3),dtype=np.uint8)
    s=LayeredSender();r=LayeredReceiver()
    boot,_=s.bootstrap(a,hashlib.sha256(b"coarse").digest()[:16],default_config("dct_bound"))
    s.acknowledge(r.receive(boot));saved=r.state.archive.copy()
    changed=np.clip(a.astype(int)+80,0,255).astype(np.uint8)
    packet,_=s.propose(changed,1,800)
    assert not parse_frame(packet)["patches"]
    s.acknowledge(r.receive(packet))
    assert np.array_equal(r.state.archive,saved)
    assert not np.array_equal(r.state.render(),a)


def test_independent_jpeg_blocks_have_placement_invariant_decodes():
    a=np.random.default_rng(11).integers(0,256,(128,256,3),dtype=np.uint8)
    with Image.open(io.BytesIO(encode_image(a,"jpeg",60))) as im:whole=np.array(im)
    original=blocks(a);decoded=blocks(whole)
    for idx in (0,2,4,7):
        with Image.open(io.BytesIO(encode_image(original[idx],"jpeg",60))) as im:single=np.array(im)
        assert np.array_equal(single,decoded[idx])
    # Reordering, atlas shape and optimized Huffman tables affect entropy bytes,
    # but not these 4:4:4, aligned 8x8 quantized block reconstructions.
    atlas=np.concatenate([original[7],original[2],original[0]],axis=0)
    with Image.open(io.BytesIO(encode_image(atlas,"jpeg",60))) as im:result=blocks(np.array(im))
    for j,idx in enumerate((7,2,0)):assert np.array_equal(result[j],decoded[idx])


def test_dct_bound_valid_for_rendered_state_and_stale_packets_fail():
    a=np.random.default_rng(8).integers(0,256,(128,128,3),dtype=np.uint8)
    s=LayeredSender();r=LayeredReceiver()
    boot,_=s.bootstrap(a,hashlib.sha256(b"dct").digest()[:16],default_config("dct_bound"))
    s.acknowledge(r.receive(boot))
    b=255-a;point,upper,_=s.cache.scores(b,1)
    packet,_=s.propose(b,1,8192)
    s.acknowledge(r.receive(packet));state=r.state
    # Bound refers to the pre-update archive rendered with current coarse means;
    # test a no-ROI subsequent rendering with the frozen archive explicitly.
    c=np.roll(b,1,axis=0);_,bound,_=s.cache.scores(c,2)
    from safesync.reference import restore_coarse
    truth=exact_mse(c,restore_coarse(state.archive,c))
    assert np.all(truth<=bound+1e-6)
    bad=bytearray(packet);bad[-1]^=1
    with pytest.raises(ValueError):r.receive(bytes(bad))


def test_full_archive_baseline_commits_its_own_decoded_archive():
    a=np.random.default_rng(84).integers(0,256,(128,128,3),dtype=np.uint8)
    s=LayeredSender(keep_full_archive=True);r=LayeredReceiver()
    boot,_=s.bootstrap(a,hashlib.sha256(b"full archive").digest()[:16],default_config("dct_bound"))
    s.acknowledge(r.receive(boot))
    packet,_=s.propose(255-a,1,8192,ranking="exact")
    saved=s.full_archive.copy()
    assert np.array_equal(saved,a)
    s.acknowledge(r.receive(packet))
    assert np.array_equal(s.full_archive,r.state.archive)
