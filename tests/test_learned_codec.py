import hashlib

import numpy as np
import pytest

from safesync.learned_codec import (ACK_BYTES, HEAD, LEGACY_HEAD, Candidate, pack, parse,
                                   receive, select)
from safesync.pixel_wire import PixelState, checksum


def candidate(*, quality=1, mse=0.1, stream_bytes=20):
    return Candidate(quality, (64, 64), (64, 64), (64, 64), (1, 1),
                     "01" * 32, (b"y" * (stream_bytes - 8), b"z" * 8),
                     mse, 1.0, 1.0, "02" * 32)


def blank():
    region = hashlib.sha256(b"learned-codec-targeted-test").digest()[:16]
    return PixelState(np.zeros((64, 64, 3), np.uint8), region)


def test_actual_container_lengths_and_two_streams():
    prior, item = blank(), candidate(stream_bytes=100)
    wire = pack(item, region=prior.region, base_version=0,
                base_digest=prior.digest(), observation=0)
    parsed = parse(wire)
    assert len(wire) == HEAD.size + 100 + 16 == item.packet_bytes
    assert parsed["streams"] == item.streams
    assert parsed["weight_sha256"] is None
    assert parsed["model_registry"] == "bmshj2018-hyperprior-mse-v1/quality1"
    assert HEAD.size == 61
    assert parsed["native_shape"] == parsed["coded_shape"] == (64, 64)
    assert parsed["latent_shape"] == (1, 1)
    assert ACK_BYTES == 60
    damaged = bytearray(wire)
    damaged[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        parse(bytes(damaged))
    with pytest.raises(ValueError, match="checksum"):
        parse(wire[:-1])


def test_budget_uses_all_stream_header_and_checksum_bytes():
    prior = blank()
    small, large = candidate(mse=0.2), candidate(quality=2, mse=0.1, stream_bytes=200)
    wire, meta = select([large, small], budget=small.packet_bytes, region=prior.region,
        base_version=0, base_digest=prior.digest(), observation=0)
    assert len(wire) == small.packet_bytes
    assert meta["quality"] == 1
    assert meta["codec_trials"] == 2 and meta["fitting_trials"] == 1
    assert meta["entropy_stream_bytes"] + meta["header_bytes"] + meta["checksum_bytes"] == len(wire)
    with pytest.raises(ValueError, match="no candidate"):
        select([small], budget=small.packet_bytes - 1, region=prior.region,
               base_version=0, base_digest=prior.digest(), observation=0)


def test_receiver_checks_base_before_decoding_and_ack_is_counted():
    prior = blank()
    wire = pack(candidate(), region=prior.region, base_version=0,
                base_digest=prior.digest(), observation=0)

    class ReceiverOnlyDecoder:
        def __init__(self):
            self.called = 0

        def decode_packet(self, packet):
            self.called += 1
            return np.full((64, 64, 3), 17, np.uint8), parse(packet)

    decoder = ReceiverOnlyDecoder()
    state, ack = receive(prior, wire, decoder)
    assert decoder.called == 1 and state.version == 1
    assert state.rgb[0, 0].tolist() == [17, 17, 17]
    assert len(ack) == 60 and checksum(ack[:-16]) == ack[-16:]
    with pytest.raises(ValueError, match="wrong region/base"):
        receive(state, wire, decoder)
    assert decoder.called == 1


def test_shape_and_stream_length_metadata_are_verified():
    prior = blank()
    wire = pack(candidate(), region=prior.region, base_version=0,
                base_digest=prior.digest(), observation=0)
    fields = list(HEAD.unpack(wire[:HEAD.size]))
    fields[4] = 65  # Coded height cannot exceed the shared native height.
    body = HEAD.pack(*fields) + wire[HEAD.size:-16]
    with pytest.raises(ValueError, match="dimensions"):
        parse(body + checksum(body))
    fields = list(HEAD.unpack(wire[:HEAD.size]))
    fields[-1] += 1
    body = HEAD.pack(*fields) + wire[HEAD.size:-16]
    with pytest.raises(ValueError, match="stream layout"):
        parse(body + checksum(body))


def test_previous_lc08_packets_keep_their_complete_model_identity():
    prior, item = blank(), candidate()
    y,z = item.streams
    header = LEGACY_HEAD.pack(b"LC08",1,0,1,1,*item.native_shape,*item.coded_shape,
        *item.padded_shape,*item.latent_shape,0,1,0,prior.digest(),prior.region,
        bytes.fromhex(item.weight_sha256),len(y),len(z))
    body = header+y+z
    parsed = parse(body+checksum(body))
    assert parsed["weight_sha256"] == item.weight_sha256
    assert parsed["streams"] == item.streams
    assert parsed["header_bytes"] == 108 and parsed["wire_schema"] == "LC08"
