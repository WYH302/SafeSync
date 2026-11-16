"""A real learned-base plus absolute/residual WebP-atlas image payload."""
from __future__ import annotations

import struct

import numpy as np

from . import task_evidence_wire as wire

HEADER = struct.Struct("<4sBBHII")
ABSOLUTE, RESIDUAL = 0, 1


def encode(base_packet, current, decoded_base, indices, *, mode=RESIDUAL, block=64):
    if (mode not in (ABSOLUTE,RESIDUAL) or current.dtype != np.uint8 or decoded_base.dtype != np.uint8
            or current.shape != decoded_base.shape or current.ndim != 3 or current.shape[2] != 3
            or not 0 < block <= 255 or current.shape[0] % block or current.shape[1] % block or not base_packet):
        raise ValueError("Invalid residual-evidence source or mode")
    ids = np.asarray(indices,dtype="<u2")
    count = current.shape[0]*current.shape[1]//block**2
    if ids.ndim != 1 or len(ids) > 65535 or len(ids) != len(np.unique(ids)) or np.any(ids >= count):
        raise ValueError("Invalid residual-evidence indices")
    image = current
    if mode == RESIDUAL:
        # Signed differences span[-255,255]. The fixed factor2 covers this range
        # in an8-bit codec and introduces at most1 RGB unit before atlas loss.
        difference = current.astype(np.int16)-decoded_base.astype(np.int16)
        image = np.clip(np.rint(difference/2+128),0,255).astype(np.uint8)
    atlas = wire.encode_atlas(image,ids,codec="webp",quality=70,block=block)
    return HEADER.pack(b"RE01",mode,block,len(ids),len(base_packet),len(atlas))+ids.tobytes()+base_packet+atlas


def decode(payload, native_shape, decode_base):
    if len(payload) < HEADER.size:
        raise ValueError("Truncated residual-evidence payload")
    magic,mode,block,n,base_length,atlas_length = HEADER.unpack_from(payload)
    h,w = native_shape
    if (magic != b"RE01" or mode not in (ABSOLUTE,RESIDUAL) or block <= 0 or h % block or w % block
            or not base_length or bool(n) != bool(atlas_length)
            or len(payload) != HEADER.size+2*n+base_length+atlas_length):
        raise ValueError("Malformed residual-evidence payload")
    ids = np.frombuffer(payload,dtype="<u2",count=n,offset=HEADER.size).copy()
    if len(ids) != len(np.unique(ids)) or np.any(ids >= h*w//block**2):
        raise ValueError("Residual-evidence index outside the image")
    start = HEADER.size+2*n
    base = decode_base(payload[start:start+base_length])
    if base.dtype != np.uint8 or base.shape != (h,w,3):
        raise ValueError("Registered base decoder returned a different image")
    if not n:
        return base.copy()
    atlas = payload[start+base_length:]
    if mode == ABSOLUTE:
        result = base.copy()
        wire.paste_atlas(result,atlas,ids,block)
        return result
    residual = np.full_like(base,128)
    wire.paste_atlas(residual,atlas,ids,block)
    return np.clip(base.astype(np.int16)+2*(residual.astype(np.int16)-128),0,255).astype(np.uint8)
