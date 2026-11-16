"""Compact, serialized reference measurements for receiver-relative updates.

All measurements are computed from the receiver's decoded RGB reference.
Random projections are existing estimators, not a newly proposed transform.
The hybrid removes block means before projection to avoid estimating their
energy twice. Reference traffic and downlink image traffic remain separate.
"""
from __future__ import annotations

import hashlib
import struct
from functools import lru_cache

import numpy as np
from scipy.fft import dctn


METHODS = {"thumbnail": 0, "projection": 1, "hybrid": 2,
           "dct_low": 3, "dct_spread": 4}
HEADER = struct.Struct("<4sBBHHHIII16s")
SCALE = 256.0
BOUND_HEADER=struct.Struct("<4sII16sI")


def bind_reference(payload,version,digest,observation):
    if len(digest)!=16:raise ValueError("invalid acknowledged digest")
    body=BOUND_HEADER.pack(b"SB02",version,observation,digest,len(payload))+payload
    return body+hashlib.sha256(body).digest()[:16]


def unbind_reference(wire,version,digest,observation):
    if len(wire)<BOUND_HEADER.size+16:raise ValueError("truncated reference binding")
    magic,v,t,d,n=BOUND_HEADER.unpack(wire[:BOUND_HEADER.size])
    if magic!=b"SB02" or (v,t,d)!=(version,observation,digest) or len(wire)!=BOUND_HEADER.size+n+16:
        raise ValueError("reference/ACK/observation mismatch")
    if hashlib.sha256(wire[:-16]).digest()[:16]!=wire[-16:]:raise ValueError("reference checksum mismatch")
    return wire[BOUND_HEADER.size:-16]


def blocks(rgb, block=64):
    a = np.asarray(rgb, dtype=np.uint8)
    h, w = a.shape[:2]
    if a.ndim != 3 or a.shape[2] != 3 or h % block or w % block:
        raise ValueError("RGB dimensions must be divisible by the block size")
    return a.reshape(h//block, block, w//block, block, 3).transpose(0,2,1,3,4).reshape(-1,block,block,3)


def means(x, side):
    n, h, w, c = x.shape
    if h % side or w % side:
        raise ValueError("mean grid must divide block dimensions")
    return x.reshape(n, side, h//side, side, w//side, c).mean((2,4))


def expand_means(x, block):
    return x.repeat(block//x.shape[1], 1).repeat(block//x.shape[2], 2)


@lru_cache(maxsize=12)
def projection_matrix(block, count, seed):
    rng = np.random.default_rng(seed)
    matrix = (rng.integers(0, 2, (block*block, count), dtype=np.int8)*2-1).astype(np.float32)/block
    matrix.setflags(write=False)
    return matrix


def project(x, count, seed):
    n, b, _, _ = x.shape
    return (x.transpose(0,3,1,2).reshape(n*3, b*b) @ projection_matrix(b,count,seed)).reshape(n,3,count).transpose(0,2,1)


def quantize(values):
    q = np.rint(values*SCALE)
    if np.any(np.abs(q) > 32767):
        raise ValueError("reference coefficient overflow")
    return q.astype("<i2")


def dct_indices(block, count, spread=False, seed=0):
    order = sorted(((i,j) for i in range(block) for j in range(block)), key=lambda ij:(sum(ij),ij[0]))
    if not spread:
        return np.array(order[:count], dtype=int)
    # Stratify over the fixed frequency ordering, rather than discarding all
    # high frequencies. One shared deterministic sample per stratum.
    rng = np.random.default_rng(seed)
    edges = np.linspace(0,block*block,count+1,dtype=int)
    return np.array([order[rng.integers(edges[k],edges[k+1])] for k in range(count)])


def encode_reference(rgb, method="hybrid", side=8, block=64, seed=20261028, version=0):
    if method not in METHODS or side not in (4,8,16) or block % side:
        raise ValueError("unsupported reference design")
    a = np.asarray(rgb, np.uint8)
    x = blocks(a,block).astype(np.float32)/255
    count = side*side//2
    if method == "thumbnail":
        payload = np.rint(means(x,side)*255).astype(np.uint8).tobytes()
    elif method == "projection":
        payload = quantize(project(x,count,seed)).tobytes()
    elif method == "hybrid":
        coarse = np.rint(means(x,side//2)*255).astype(np.uint8)
        residual = x-expand_means(coarse.astype(np.float32)/255,block)
        detail_count = (side*side-(side//2)**2)//2
        payload = coarse.tobytes()+quantize(project(residual,detail_count,seed)).tobytes()
    else:
        coords = dct_indices(block,count,method=="dct_spread",seed)
        coeff = dctn(x, axes=(1,2), norm="ortho")[:,coords[:,0],coords[:,1],:]
        payload = quantize(coeff).tobytes()
    expected = len(x)*side*side*3
    if len(payload) != expected:
        raise AssertionError("reference budget mismatch")
    identity = hashlib.sha256(a.tobytes()+struct.pack("<I",version)).digest()[:16]
    head = HEADER.pack(b"SR02",METHODS[method],side,block,a.shape[0],a.shape[1],seed,version,len(payload),identity)
    return head+payload+hashlib.sha256(head+payload).digest()[:16]


def reference_score(current_rgb, wire):
    if len(wire) < HEADER.size+16:
        raise ValueError("truncated reference")
    magic, kind, side, block, height, width, seed, version, size, identity = HEADER.unpack(wire[:HEADER.size])
    if magic != b"SR02" or kind not in METHODS.values() or side not in (4,8,16):
        raise ValueError("unknown reference format")
    if len(wire) != HEADER.size+size+16 or hashlib.sha256(wire[:-16]).digest()[:16] != wire[-16:]:
        raise ValueError("reference length/checksum mismatch")
    a = np.asarray(current_rgb,np.uint8)
    if a.shape != (height,width,3):
        raise ValueError("reference/current dimensions differ")
    x = blocks(a,block).astype(np.float32)/255
    payload = wire[HEADER.size:-16]
    count = side*side//2
    if kind == METHODS["thumbnail"]:
        old = np.frombuffer(payload,np.uint8).reshape(len(x),side,side,3).astype(np.float32)/255
        return np.mean((means(x,side)-old)**2,axis=(1,2,3))
    if kind == METHODS["hybrid"]:
        cside = side//2
        cbytes = len(x)*cside*cside*3
        old = np.frombuffer(payload[:cbytes],np.uint8).reshape(len(x),cside,cside,3).astype(np.float32)/255
        current = means(x,cside)
        coarse_energy = np.mean((current-old)**2,axis=(1,2,3))
        # Current high-pass must use its own coarse mean. The reference uses
        # a quantized mean; its small DC quantization energy is conservatively
        # included, and no full historical pixels are accessible here.
        residual = x-expand_means(current,block)
        count = (side*side-cside*cside)//2
        measured = np.frombuffer(payload[cbytes:],"<i2").reshape(len(x),count,3).astype(np.float32)/SCALE
        return coarse_energy+np.mean((project(residual,count,seed)-measured)**2,axis=(1,2))
    old = np.frombuffer(payload,"<i2").reshape(len(x),count,3).astype(np.float32)/SCALE
    if kind == METHODS["projection"]:
        return np.mean((project(x,count,seed)-old)**2,axis=(1,2))
    coords = dct_indices(block,count,kind==METHODS["dct_spread"],seed)
    new = dctn(x, axes=(1,2), norm="ortho")[:,coords[:,0],coords[:,1],:]
    delta2 = (new-old)**2
    if kind == METHODS["dct_low"]:
        return delta2.sum(axis=(1,2))/(block*block*3)
    edges = np.linspace(0,block*block,count+1,dtype=int)
    return (delta2*np.diff(edges)[None,:,None]).sum(axis=(1,2))/(block*block*3)


def exact_mse(current_rgb, prior_rgb, block=64):
    x, y = blocks(current_rgb,block).astype(np.float32), blocks(prior_rgb,block).astype(np.float32)
    return np.mean(((x-y)/255)**2,axis=(1,2,3))


def detail_thumbnail_score(current_rgb,wire,coarse_side=4):
    fields=HEADER.unpack(wire[:HEADER.size])
    magic,kind,side,b,h,w,seed,version,size,identity=fields
    if kind!=METHODS["thumbnail"] or side<coarse_side:
        raise ValueError("detail comparison requires a fine thumbnail")
    # Validate the wire through the common parser before using its payload.
    reference_score(current_rgb,wire)
    old=np.frombuffer(wire[HEADER.size:-16],np.uint8).reshape(-1,side,side,3).astype(np.float32)/255
    old_high=old-expand_means(means(old,coarse_side),side)
    x=blocks(current_rgb,b).astype(np.float32)/255
    coarse=means(x,coarse_side)
    now_high=means(x,side)-expand_means(coarse,side)
    coarse_error=np.mean((coarse-np.rint(coarse*255)/255)**2,axis=(1,2,3))
    return np.mean((now_high-old_high)**2,axis=(1,2,3))+coarse_error


def restore_coarse(prior_rgb,current_rgb,block=64,coarse_side=4):
    old=np.asarray(prior_rgb,np.uint8)
    current=np.asarray(current_rgb,np.uint8)
    h,w=old.shape[:2];factor=block//coarse_side
    means_old=old.reshape(h//factor,factor,w//factor,factor,3).mean((1,3))
    means_new=np.rint(current.reshape(h//factor,factor,w//factor,factor,3).mean((1,3)))
    return np.clip(np.rint(old.astype(np.float32)+(means_new-means_old).repeat(factor,0).repeat(factor,1)),0,255).astype(np.uint8)
