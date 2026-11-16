"""Actual task-state and optional image-evidence transactions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import struct
import zlib

import numpy as np
from PIL import Image

HEADER = struct.Struct("<4s16s16sIIIHHHHHHBBIII16s16s")
ACK = struct.Struct("<4sII16s16s")
ACK_BYTES = ACK.size + 16
UPDATED, DEFERRED = 1, 2
ROI, FULL, LEARNED, LAYERED = 0, 1, 2, 3
LAYER_HEADER = struct.Struct("<4sII")


def checksum(data):
    return hashlib.sha256(data).digest()[:16]


@dataclass
class TaskEvidenceState:
    rgb: np.ndarray
    labels: np.ndarray
    region: bytes
    model: bytes
    version: int = 0
    observed_at: int = -1
    statuses: np.ndarray | None = None
    block: int = 64

    def __post_init__(self):
        h, w = self.rgb.shape[:2]
        if (self.rgb.dtype != np.uint8 or self.rgb.shape != (h, w, 3) or h % self.block or w % self.block
                or self.labels.dtype != np.uint8 or self.labels.ndim != 2
                or len(self.region) != 16 or len(self.model) != 16):
            raise ValueError("Invalid task/evidence state")
        if self.statuses is None:
            self.statuses = np.full(h * w // self.block ** 2, DEFERRED, np.uint8)
        if (self.statuses.dtype != np.uint8 or self.statuses.shape != (h * w // self.block ** 2,)
                or np.any((self.statuses != UPDATED) & (self.statuses != DEFERRED))):
            raise ValueError("Invalid task/evidence manifest")

    def digest(self):
        head = struct.pack("<IiHHHHH", self.version, self.observed_at, *self.rgb.shape[:2], self.block, *self.labels.shape)
        return checksum(self.region + self.model + head + self.rgb.tobytes() + self.labels.tobytes() + self.statuses.tobytes())


def label_payload(prior, current):
    values = np.asarray(current)
    if values.dtype != np.uint8 or values.shape != prior.labels.shape:
        raise ValueError("Task-state label shape or type")
    absolute = zlib.compress(values.tobytes(), 6)
    residual = zlib.compress(np.bitwise_xor(values, prior.labels).tobytes(), 6)
    return (1, residual) if len(residual) < len(absolute) else (0, absolute)


def encode(prior, labels, *, observation, payload=b"", indices=(), image_mode=ROI):
    if observation <= prior.observed_at or image_mode not in (ROI, FULL, LEARNED, LAYERED):
        raise ValueError("Task/evidence observation or image mode")
    h, w = prior.rgb.shape[:2]
    ids = np.asarray(indices, dtype="<u2")
    count = len(prior.statuses)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids >= count):
        raise ValueError("Task/evidence ROI index")
    if image_mode in (FULL, LEARNED) and (len(ids) or not payload):
        raise ValueError("Full image requires a payload and no ROI indices")
    if image_mode == ROI and bool(len(ids)) != bool(len(payload)):
        raise ValueError("ROI payload/index mismatch")
    status = np.full(count, DEFERRED, np.uint8)
    if image_mode in (FULL, LEARNED, LAYERED):
        status[:] = UPDATED
    else:
        status[ids] = UPDATED
    label_mode, encoded_labels = label_payload(prior, labels)
    header = HEADER.pack(b"TE01", prior.region, prior.model, prior.version, prior.version + 1, observation,
        h, w, prior.block, *labels.shape, len(ids), label_mode, image_mode,
        count, len(encoded_labels), len(payload), prior.digest(), checksum(labels.tobytes()))
    body = header + status.tobytes() + ids.tobytes() + encoded_labels + payload
    costs = dict(header=HEADER.size, manifest=count, indices=ids.nbytes, labels=len(encoded_labels),
        image=len(payload), checksum=16)
    packet = body + checksum(body)
    if len(packet) != sum(costs.values()):
        raise AssertionError("Task/evidence byte accounting")
    return packet, costs


def encode_atlas(current, ids, *, codec="webp", quality=70, block=64):
    ids = np.asarray(ids, int)
    if not len(ids):
        return b""
    columns = math.ceil(math.sqrt(len(ids)))
    rows = math.ceil(len(ids) / columns)
    atlas = np.zeros((rows * block, columns * block, 3), np.uint8)
    grid_width = current.shape[1] // block
    for slot, index in enumerate(ids):
        y, x = divmod(int(index), grid_width)
        ay, ax = divmod(slot, columns)
        atlas[ay * block:(ay + 1) * block, ax * block:(ax + 1) * block] = current[y * block:(y + 1) * block, x * block:(x + 1) * block]
    stream = io.BytesIO()
    image = Image.fromarray(atlas)
    if codec == "webp":
        image.save(stream, format="WEBP", quality=quality, method=4)
    elif codec == "jpeg":
        image.save(stream, format="JPEG", quality=quality, optimize=True, subsampling=0)
    else:
        raise ValueError("Unknown atlas codec")
    return stream.getvalue()


def layered_payload(base, atlas):
    if not base:
        raise ValueError("A current base image is required")
    return LAYER_HEADER.pack(b"LY01", len(base), len(atlas)) + base + atlas


def decode_full(payload, height, width):
    with Image.open(io.BytesIO(payload)) as coded:
        if not (0 < coded.width <= width and 0 < coded.height <= height):
            raise ValueError("Invalid full-image dimensions")
        return np.asarray(coded.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)).copy()


def paste_atlas(rgb, payload, ids, block):
    columns = math.ceil(math.sqrt(len(ids)))
    rows = math.ceil(len(ids) / columns)
    with Image.open(io.BytesIO(payload)) as coded:
        if coded.size != (columns * block, rows * block):
            raise ValueError("Atlas dimensions differ")
        atlas = np.asarray(coded.convert("RGB"))
    for slot, index in enumerate(ids):
        ay, ax = divmod(slot, columns)
        y, x = divmod(int(index), rgb.shape[1] // block)
        rgb[y * block:(y + 1) * block, x * block:(x + 1) * block] = atlas[ay * block:(ay + 1) * block, ax * block:(ax + 1) * block]


def receive(prior, packet, *, learned_decoder=None):
    if len(packet) < HEADER.size + 16 or checksum(packet[:-16]) != packet[-16:]:
        raise ValueError("Task/evidence checksum or truncation")
    values = HEADER.unpack_from(packet)
    magic, region, model, base, target, observation, h, w, block, lh, lw, count, lm, im, ns, nl, ni, old_hash, label_hash = values
    if (magic != b"TE01" or region != prior.region or model != prior.model or base != prior.version or target != base + 1
            or observation <= prior.observed_at or old_hash != prior.digest()):
        raise ValueError("Task/evidence reference, model, or observation mismatch")
    if ((h, w) != prior.rgb.shape[:2] or block != prior.block or (lh, lw) != prior.labels.shape
            or ns != len(prior.statuses) or count > ns or lm not in (0, 1) or im not in (ROI, FULL, LEARNED, LAYERED)
            or len(packet) != HEADER.size + ns + count * 2 + nl + ni + 16):
        raise ValueError("Task/evidence dimensions or payload layout")
    offset = HEADER.size
    status = np.frombuffer(packet[offset:offset + ns], np.uint8).copy()
    offset += ns
    ids = np.frombuffer(packet[offset:offset + count * 2], "<u2").copy()
    offset += count * 2
    if len(np.unique(ids)) != count or np.any(ids >= ns):
        raise ValueError("Task/evidence index layout")
    expected_status = np.full(ns, DEFERRED, np.uint8)
    expected_status[:] = UPDATED if im in (FULL, LEARNED, LAYERED) else DEFERRED
    expected_status[ids] = UPDATED
    if not np.array_equal(status, expected_status):
        raise ValueError("Task/evidence manifest mismatch")
    decoder = zlib.decompressobj()
    raw_labels = decoder.decompress(packet[offset:offset + nl], lh * lw + 1)
    if len(raw_labels) != lh * lw or not decoder.eof or decoder.unused_data:
        raise ValueError("Task-state decoded length")
    labels = np.frombuffer(raw_labels, np.uint8).reshape(lh, lw).copy()
    if lm:
        labels ^= prior.labels
    if checksum(labels.tobytes()) != label_hash:
        raise ValueError("Task-state reconstruction hash")
    offset += nl
    rgb = prior.rgb.copy()
    if im == LAYERED:
        payload = packet[offset:-16]
        if len(payload) < LAYER_HEADER.size:
            raise ValueError("Truncated context/refinement body")
        lm, nb, na = LAYER_HEADER.unpack_from(payload)
        if lm != b"LY01" or not nb or bool(na) != bool(count) or len(payload) != LAYER_HEADER.size + nb + na:
            raise ValueError("Context/refinement body layout")
        rgb = decode_full(payload[LAYER_HEADER.size:LAYER_HEADER.size + nb], h, w)
        if count:
            paste_atlas(rgb, payload[LAYER_HEADER.size + nb:], ids, block)
    elif im == LEARNED:
        if count or not ni or learned_decoder is None:
            raise ValueError("Learned image requires its registered decoder")
        rgb = np.asarray(learned_decoder(prior, packet[offset:-16]))
        if rgb.dtype != np.uint8 or rgb.shape != prior.rgb.shape:
            raise ValueError("Registered decoder returned invalid RGB")
    elif im == FULL:
        if count or not ni:
            raise ValueError("Invalid full-image body")
        rgb = decode_full(packet[offset:-16], h, w)
    elif count:
        if not ni:
            raise ValueError("Missing atlas")
        paste_atlas(rgb, packet[offset:-16], ids, block)
    elif ni:
        raise ValueError("Image payload without selected regions")
    state = TaskEvidenceState(rgb, labels, region, model, target, observation, status, block)
    body = ACK.pack(b"TA01", target, observation, checksum(packet), state.digest())
    return state, body + checksum(body)


class TaskEvidenceReceiver:
    def __init__(self, state, *, learned_decoder=None):
        self.state = state
        self.expected_observation = state.observed_at
        self.last_packet = None
        self.last_ack = None
        self.learned_decoder = learned_decoder

    def expect(self, observation):
        if observation < self.expected_observation:
            raise ValueError("Observation expectation moved backward")
        self.expected_observation = observation

    @property
    def task_current(self):
        return self.state.observed_at >= self.expected_observation

    @property
    def evidence_status(self):
        if not self.task_current:
            return np.full_like(self.state.statuses, DEFERRED)
        return self.state.statuses.copy()

    def receive(self, packet):
        if packet == self.last_packet:
            return self.last_ack, False
        state, ack = receive(self.state, packet, learned_decoder=self.learned_decoder)
        self.state, self.last_packet, self.last_ack = state, packet, ack
        return ack, True
