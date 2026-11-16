"""Actual absolute-token/ROI wire format and conservative state reconciliation.

The receiver API accepts only prior state and bytes. A digest checks shared
reference agreement, and makes no statement about the observed physical world.
"""
from __future__ import annotations

import hashlib
import io
import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from PIL import Image

HEADER = struct.Struct("<4sHHHHIIIII32s32s")
TOKEN_ID = struct.Struct("<H")
ROI_HEADER = struct.Struct("<HI")
CRC = struct.Struct("<I")
MAGIC = b"SS01"
ANCHOR_BASE = 2**32 - 1
CODEC_ID = hashlib.sha256(b"SafeSync-uniform-normalized-int8-v1").digest()


class Status(IntEnum):
    STALE = 0
    CONFIRMED_STABLE = 1
    UPDATED = 2
    DEFERRED = 3


class WireError(ValueError):
    pass


class ReferenceMismatch(WireError):
    pass


def quantize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1.00001:
        raise ValueError("codec requires finite, unit-normalized feature components")
    return np.rint(values * 127).clip(-127, 127).astype(np.int8)


def dequantize(codes: np.ndarray) -> np.ndarray:
    return np.asarray(codes, dtype=np.float32) / 127.0


def signature(backbone_identity: str) -> bytes:
    return hashlib.sha256(CODEC_ID + backbone_identity.encode("utf-8")).digest()


@dataclass
class State:
    codes: np.ndarray
    grid: int
    identity: bytes
    version: int = 0
    statuses: np.ndarray | None = None
    age: np.ndarray | None = None
    rois: dict[int, bytes] = field(default_factory=dict)

    def __post_init__(self):
        self.codes = np.array(self.codes, dtype=np.int8, copy=True)
        if self.codes.ndim != 2 or len(self.codes) != self.grid**2:
            raise ValueError("state shape does not match grid")
        if len(self.identity) != 32:
            raise ValueError("identity must be SHA256 bytes")
        self.statuses = np.full(len(self.codes), Status.STALE, np.uint8) if self.statuses is None else np.array(self.statuses, dtype=np.uint8, copy=True)
        self.age = np.zeros(len(self.codes), np.int32) if self.age is None else np.array(self.age, dtype=np.int32, copy=True)

    @property
    def features(self):
        return dequantize(self.codes)

    @property
    def digest(self):
        return hashlib.sha256(self.identity + self.codes.tobytes()).digest()


def minimum_bytes(state: State) -> int:
    return HEADER.size + len(state.codes) + CRC.size


def encode(state: State, current_codes: np.ndarray, token_ids: list[int], rois: dict[int, bytes], stable: np.ndarray, budget: int | None = None, anchor: bool = False):
    codes = np.asarray(current_codes)
    if codes.dtype != np.int8 or codes.shape != state.codes.shape:
        raise ValueError("current code shape/dtype mismatch")
    token_ids = sorted(set(int(x) for x in token_ids))
    if any(x < 0 or x >= len(codes) for x in token_ids):
        raise ValueError("out-of-range token ID")
    if not set(rois).issubset(token_ids):
        raise ValueError("every ROI action must include its current token")
    if anchor and len(token_ids) != len(codes):
        raise ValueError("anchor requires all current tokens")
    statuses = np.where(stable, Status.CONFIRMED_STABLE, Status.DEFERRED).astype(np.uint8)
    if statuses.shape != (len(codes),):
        raise ValueError("stable mask shape mismatch")
    statuses[token_ids] = Status.UPDATED
    parts = [statuses.tobytes()]
    for index in token_ids:
        parts += [TOKEN_ID.pack(index), codes[index].tobytes()]
    for index in sorted(rois):
        parts += [ROI_HEADER.pack(index, len(rois[index])), rois[index]]
    body = b"".join(parts)
    header = HEADER.pack(MAGIC, 1, state.grid, codes.shape[1], len(rois), ANCHOR_BASE if anchor else state.version, state.version + 1, len(token_ids), len(statuses), len(body), bytes(32) if anchor else state.digest, state.identity)
    used = len(header) + len(body) + CRC.size
    if budget is not None and used > budget:
        raise ValueError(f"encoded payload {used} exceeds frame budget {budget}")
    padding = 0 if budget is None else budget - used
    framed = header + body + bytes(padding)
    packet = framed + CRC.pack(zlib.crc32(framed))
    counts = {"header": HEADER.size, "manifest": len(statuses), "indices": TOKEN_ID.size * len(token_ids), "token_payload": codes.shape[1] * len(token_ids), "evidence_headers": ROI_HEADER.size * len(rois), "evidence_payload": sum(map(len, rois.values())), "padding": padding, "checksum": CRC.size}
    assert sum(counts.values()) == len(packet)
    return packet, counts


def receive(previous: State, packet: bytes | None) -> State:
    """Decode atomically. Lost/empty frames make status stale, preserving values."""
    if not packet:
        return State(previous.codes, previous.grid, previous.identity, previous.version, age=previous.age + 1)
    if len(packet) < HEADER.size + CRC.size:
        raise WireError("truncated header")
    if zlib.crc32(packet[:-CRC.size]) != CRC.unpack(packet[-CRC.size:])[0]:
        raise WireError("checksum mismatch")
    magic, schema, grid, dim, nr, base, target, nt, nm, body_size, digest, identity = HEADER.unpack_from(packet)
    if (magic, schema) != (MAGIC, 1):
        raise WireError("unsupported protocol")
    if grid != previous.grid or dim != previous.codes.shape[1] or identity != previous.identity:
        raise ReferenceMismatch("model/codec/grid identity mismatch")
    if target <= previous.version:
        raise ReferenceMismatch("replayed or out-of-order target version")
    if base != ANCHOR_BASE and (base != previous.version or digest != previous.digest or target != base + 1):
        raise ReferenceMismatch("base version/digest mismatch")
    if nm != grid**2 or nt > nm or nr > nt or body_size < nm or HEADER.size + body_size + CRC.size > len(packet):
        raise WireError("invalid record counts or body length")
    if any(packet[HEADER.size + body_size:-CRC.size]):
        raise WireError("nonzero padding")
    body = memoryview(packet)[HEADER.size:HEADER.size + body_size]
    statuses = np.frombuffer(body[:nm], dtype=np.uint8).copy()
    if np.any(statuses > Status.DEFERRED):
        raise WireError("invalid status")
    codes = previous.codes.copy()
    offset, ids = nm, set()
    for _ in range(nt):
        if offset + TOKEN_ID.size + dim > len(body):
            raise WireError("truncated token")
        index = TOKEN_ID.unpack_from(body, offset)[0]
        offset += TOKEN_ID.size
        if index >= nm or index in ids or statuses[index] != Status.UPDATED:
            raise WireError("invalid/duplicate token")
        ids.add(index)
        codes[index] = np.frombuffer(body[offset:offset + dim], dtype=np.int8)
        offset += dim
    if set(np.flatnonzero(statuses == Status.UPDATED)) != ids:
        raise WireError("updated manifest does not match payload")
    if base == ANCHOR_BASE and len(ids) != nm:
        raise WireError("incomplete anchor")
    rois = {}
    for _ in range(nr):
        if offset + ROI_HEADER.size > len(body):
            raise WireError("truncated ROI header")
        index, size = ROI_HEADER.unpack_from(body, offset)
        offset += ROI_HEADER.size
        if index not in ids or index in rois or offset + size > len(body):
            raise WireError("invalid ROI")
        content = bytes(body[offset:offset + size])
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as error:
            raise WireError("ROI cannot be decoded") from error
        rois[index] = content
        offset += size
    if offset != len(body):
        raise WireError("trailing body records")
    ages = previous.age + 1
    ages[statuses == Status.UPDATED] = 0
    return State(codes, grid, identity, target, statuses, ages, rois)
