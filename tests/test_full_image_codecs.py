import hashlib
import numpy as np
import pytest
from safesync.pixel_wire import PixelState,DEFERRED
from safesync.full_image_codecs import fit,receive,HEAD


def test_codec_resize_and_exact_actual_budget():
    a=np.random.default_rng(18).integers(0,256,(128,128,3),dtype=np.uint8)
    blank=PixelState(np.zeros_like(a),hashlib.sha256(b"full image").digest()[:16])
    for codec in ("jpeg420","jpeg2000","webp"):
        wire,meta=fit(a,codec=codec,budget=1300,region=blank.region,base_version=blank.version,base_digest=blank.digest(),observation=0)
        assert len(wire)==meta["bytes"]<=1300
        decoded,ack=receive(blank,wire)
        assert decoded.rgb.shape==a.shape and decoded.version==1 and len(ack)==60
        assert not np.array_equal(decoded.rgb,a)
        with pytest.raises(ValueError):receive(decoded,wire)
        bad=bytearray(wire);bad[-1]^=1
        with pytest.raises(ValueError):receive(blank,bytes(bad))


def test_zero_payload_is_deferred_not_a_free_image():
    a=np.zeros((128,128,3),np.uint8)
    blank=PixelState(a,hashlib.sha256(b"minimal").digest()[:16])
    wire,meta=fit(255-a,codec="jpeg420",budget=HEAD.size+16,region=blank.region,base_version=0,base_digest=blank.digest(),observation=0)
    decoded,_=receive(blank,wire)
    assert np.array_equal(decoded.rgb,a)
    assert np.all(decoded.statuses==DEFERRED)
