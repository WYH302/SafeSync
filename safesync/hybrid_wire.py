"""A joint current-feature and optional image-evidence transaction."""
from __future__ import annotations

import hashlib
import struct

HEADER = struct.Struct("<4sIII16s")


def digest(data):
    return hashlib.sha256(data).digest()[:16]


def encode(feature_packet, image_packet, region, observation):
    body = HEADER.pack(b"HY01", observation, len(feature_packet), len(image_packet), region) + feature_packet + image_packet
    return body + digest(body)


def receive(prior, packet, quantizer, image_helper):
    if len(packet) < HEADER.size + 16 or digest(packet[:-16]) != packet[-16:]:
        raise ValueError("Hybrid transaction checksum or length")
    magic, observation, nf, ni, region = HEADER.unpack_from(packet)
    if magic != b"HY01" or region != prior.region or len(packet) != HEADER.size + nf + ni + 16:
        raise ValueError("Hybrid session or body layout")
    if observation <= prior.observed_at:
        raise ValueError("Hybrid observation is stale")
    feature_packet = packet[HEADER.size:HEADER.size + nf]
    codes, metadata = quantizer.receive(feature_packet)
    if metadata["observation"] != observation:
        raise ValueError("Hybrid feature observation mismatch")
    if ni:
        receiver = image_helper["fresh_receiver"](prior)
        receiver.receive(packet[HEADER.size + nf:-16])
        if receiver.state.observed_at != observation:
            raise ValueError("Hybrid image observation mismatch")
        rgb, image_hash = receiver.state.rgb, receiver.state.digest()
    else:
        rgb, image_hash = prior.rgb.copy(), prior.digest()
    state_hash = digest(quantizer.identity + codes.tobytes() + image_hash)
    body = b"HA01" + struct.pack("<I", observation) + digest(packet) + state_hash
    return quantizer.decode(codes), rgb, body + digest(body)
