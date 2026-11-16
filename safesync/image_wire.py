"""A standard full-image baseline with a real header, identity and checksum."""
import io
import struct
import zlib

import numpy as np
from PIL import Image

from .wire import CRC, ReferenceMismatch, State, Status, WireError, quantize

IMAGE_HEADER = struct.Struct("<4sIII32s32s")


def encode_image(prior: State, image: Image.Image, quality: int):
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality, subsampling=0)
    payload = stream.getvalue()
    header = IMAGE_HEADER.pack(b"SI01", prior.version, prior.version+1, len(payload), prior.digest, prior.identity)
    frame = header + payload
    packet = frame + CRC.pack(zlib.crc32(frame))
    return packet, {"header": len(header), "image_payload": len(payload), "checksum": CRC.size, "manifest": 0, "indices": 0, "token_payload": 0, "evidence_headers": 0, "evidence_payload": 0, "padding": 0}


def receive_image(prior: State, packet: bytes, encoder):
    if len(packet) < IMAGE_HEADER.size + CRC.size or zlib.crc32(packet[:-CRC.size]) != CRC.unpack(packet[-CRC.size:])[0]:
        raise WireError("invalid image frame")
    magic, base, target, size, digest, identity = IMAGE_HEADER.unpack_from(packet)
    if magic != b"SI01" or base != prior.version or target != base+1 or digest != prior.digest or identity != prior.identity:
        raise ReferenceMismatch("image frame reference mismatch")
    if len(packet) < IMAGE_HEADER.size + size + CRC.size or any(packet[IMAGE_HEADER.size+size:-CRC.size]):
        raise WireError("invalid image payload size")
    image = Image.open(io.BytesIO(packet[IMAGE_HEADER.size:IMAGE_HEADER.size+size])).convert("RGB")
    features, timings = encoder.cached(image)
    received = State(quantize(features), prior.grid, prior.identity, target, np.full(prior.grid**2, Status.UPDATED, np.uint8))
    return received, image, timings


def encode_image_at_budget(prior: State, image: Image.Image, budget: int):
    """Pick the highest feasible quality found by a label-free binary search.

    Full native spatial resolution is retained. An infeasible q1 is reported,
    rather than silently violating the per-image budget or changing resolution.
    """
    low, high, best = 1, 95, None
    while low <= high:
        quality = (low+high)//2
        packet, counts = encode_image(prior,image,quality)
        if len(packet) <= budget:
            best = packet, counts, quality
            low = quality+1
        else:
            high = quality-1
    if best is None:
        return None
    packet, counts, quality = best
    padding = budget-len(packet)
    body = packet[:-CRC.size]+bytes(padding)
    counts['padding'] = padding
    return body+CRC.pack(zlib.crc32(body)),counts,quality
