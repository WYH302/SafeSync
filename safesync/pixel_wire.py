"""Image-atlas update transactions with explicit manifests and counted ACKs.

The sender needs only the last acknowledged version/digest, not historical
pixels. Checksums detect accidental corruption; this is not authentication.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

UNKNOWN, STABLE, UPDATED, DEFERRED = 0, 1, 2, 3
HEADER = struct.Struct("<4sBBHHHIIII16s16sI")
ACK = struct.Struct("<4sII16s16s")
COARSE = struct.Struct("<BI")
CODECS = {"png":0,"jpeg":1,"jpeg2000":2,"webp":3}


def checksum(body):
    return hashlib.sha256(body).digest()[:16]


def encode_image(rgb, codec="jpeg", quality=90):
    stream=io.BytesIO()
    image=Image.fromarray(np.asarray(rgb,np.uint8))
    if codec=="jpeg": image.save(stream,format="JPEG",quality=int(quality),subsampling=0,optimize=True)
    elif codec=="jpeg2000": image.save(stream,format="JPEG2000",irreversible=True,quality_mode="rates",quality_layers=[float(quality)])
    elif codec=="webp": image.save(stream,format="WEBP",quality=int(quality),method=4)
    elif codec=="png": image.save(stream,format="PNG",compress_level=6)
    else: raise ValueError("unknown codec")
    return stream.getvalue()


@dataclass
class PixelState:
    rgb: np.ndarray
    region: bytes
    block: int=64
    version: int=0
    observed_at: int=-1
    statuses: np.ndarray|None=None

    def __post_init__(self):
        h,w,c=self.rgb.shape
        if c!=3 or h%self.block or w%self.block or len(self.region)!=16:
            raise ValueError("invalid state layout or region identity")
        if self.statuses is None:
            self.statuses=np.zeros((h//self.block)*(w//self.block),np.uint8)

    def digest(self):
        head=struct.pack("<Iii",self.version,self.observed_at,self.block)
        return checksum(self.region+head+self.rgb.tobytes()+self.statuses.tobytes())


def encode_update(*, shape, block, region, base_version, base_digest, observed_at,
                  current_rgb, indices, statuses, codec="jpeg", quality=90, transform=None,coarse_side=0):
    h,w=shape
    n=(h//block)*(w//block)
    ids=np.asarray(indices,dtype="<u2")
    status=np.asarray(statuses,np.uint8)
    if len(region)!=16 or len(base_digest)!=16 or len(status)!=n or np.any(status>3):
        raise ValueError("invalid update metadata")
    if len(set(ids.tolist()))!=len(ids) or np.any(ids>=n) or np.any(status[ids]!=UPDATED):
        raise ValueError("bad or unmanifested update index")
    if np.any(status==UPDATED) and set(np.flatnonzero(status==UPDATED))!=set(ids.tolist()):
        raise ValueError("manifest claims missing update content")
    payload=b""
    native_order=len(ids)==n and np.array_equal(ids,np.arange(n))
    columns=w//block if native_order else min(8,len(ids))
    if len(ids):
        source=np.asarray(current_rgb,np.uint8)
        if source.shape!=(h,w,3): raise ValueError("current image shape differs")
        atlas=np.zeros((((len(ids)+columns-1)//columns)*block,columns*block,3),np.uint8)
        for k,idx in enumerate(ids):
            y,x=divmod(int(idx),w//block)
            atlas[(k//columns)*block:(k//columns+1)*block,(k%columns)*block:(k%columns+1)*block]=source[y*block:(y+1)*block,x*block:(x+1)*block]
        payload=encode_image(atlas,codec,quality)
    # Six transmitted float32 coefficients. Applied identically by receiver;
    # identity transform is the default. No hidden photometric correction.
    affine=np.array([1,1,1,0,0,0] if transform is None else transform,dtype="<f4")
    if affine.shape!=(6,) or not np.isfinite(affine).all(): raise ValueError("bad affine")
    coarse_payload=b""
    if coarse_side:
        if block%coarse_side:raise ValueError("coarse side must divide block")
        factor=block//coarse_side
        source=np.asarray(current_rgb,np.uint8)
        small=np.rint(source.reshape(h//factor,factor,w//factor,factor,3).mean((1,3))).astype(np.uint8)
        coarse_payload=encode_image(small,"png")
    head=HEADER.pack(b"SP03",CODECS[codec],columns,block,h,w,base_version,base_version+1,
                     observed_at,len(ids),base_digest,region,len(payload))
    body=head+status.tobytes()+affine.tobytes()+COARSE.pack(coarse_side,len(coarse_payload))+ids.tobytes()+coarse_payload+payload
    return body+checksum(body),dict(header=HEADER.size,manifest=n,photometry=24,
                                   coarse_header=COARSE.size,coarse_image=len(coarse_payload),indices=len(ids)*2,image=len(payload),checksum=16)


def decode_update(prior, wire):
    if len(wire)<HEADER.size+40 or checksum(wire[:-16])!=wire[-16:]:
        raise ValueError("packet checksum/truncation")
    magic,codec,columns,block,h,w,base,target,obs,count,bd,region,size=HEADER.unpack(wire[:HEADER.size])
    if magic!=b"SP03" or codec not in CODECS.values() or block<=0 or h%block or w%block:
        raise ValueError("unknown schema/layout")
    if (h,w)!=prior.rgb.shape[:2] or block!=prior.block or region!=prior.region:
        raise ValueError("wrong region/layout")
    if base!=prior.version or target!=base+1 or bd!=prior.digest() or obs<=prior.observed_at:
        raise ValueError("wrong base or stale observation")
    n=(h//block)*(w//block)
    if count>n or not 0<=columns<=max(8,w//block) or (count and not columns):
        raise ValueError("packet size/index layout")
    offset=HEADER.size
    status=np.frombuffer(wire[offset:offset+n],np.uint8).copy();offset+=n
    affine=np.frombuffer(wire[offset:offset+24],"<f4");offset+=24
    coarse_side,coarse_bytes=COARSE.unpack(wire[offset:offset+COARSE.size]);offset+=COARSE.size
    expected=HEADER.size+n+24+COARSE.size+2*count+coarse_bytes+size+16
    if len(wire)!=expected or (coarse_side and block%coarse_side) or (bool(coarse_side)!=bool(coarse_bytes)):
        raise ValueError("coarse image layout/length")
    ids=np.frombuffer(wire[offset:offset+2*count],"<u2");offset+=2*count
    native_order=count==n and np.array_equal(ids,np.arange(n))
    if columns!=(w//block if native_order else min(8,count)):
        raise ValueError("noncanonical atlas layout")
    if not np.isfinite(affine).all() or np.any(status>3) or len(set(ids.tolist()))!=count or np.any(ids>=n):
        raise ValueError("invalid manifest/photometry")
    if set(np.flatnonzero(status==UPDATED))!=set(ids.tolist()):
        raise ValueError("manifest update/content mismatch")
    if (not count and size) or (count and not size): raise ValueError("invalid atlas payload")
    rgb=np.clip(np.rint(prior.rgb.astype(np.float32)*affine[:3]+affine[3:]),0,255).astype(np.uint8)
    if coarse_side:
        factor=block//coarse_side
        with Image.open(io.BytesIO(wire[offset:offset+coarse_bytes])) as im:
            coarse=np.array(im.convert("RGB"),np.float32)
        if coarse.shape!=(h//factor,w//factor,3):raise ValueError("decoded coarse dimensions differ")
        m=rgb.reshape(h//factor,factor,w//factor,factor,3).mean((1,3))
        rgb=np.clip(np.rint(rgb.astype(np.float32)+(coarse-m).repeat(factor,0).repeat(factor,1)),0,255).astype(np.uint8)
        offset+=coarse_bytes
    decoded=[]
    if count:
        with Image.open(io.BytesIO(wire[offset:offset+size])) as im:
            atlas=np.array(im.convert("RGB"))
        if atlas.shape!=(((count+columns-1)//columns)*block,columns*block,3):
            raise ValueError("decoded atlas dimensions differ")
        for k,idx in enumerate(ids):
            y,x=divmod(int(idx),w//block)
            patch=atlas[(k//columns)*block:(k//columns+1)*block,(k%columns)*block:(k%columns+1)*block]
            rgb[y*block:(y+1)*block,x*block:(x+1)*block]=patch
            decoded.append(patch.copy())
    state=PixelState(rgb,prior.region,block,target,obs,status)
    ack_body=ACK.pack(b"SA02",target,obs,checksum(wire),state.digest())
    return state,ack_body+checksum(ack_body),list(zip(ids.tolist(),decoded))


class TransactionSender:
    def __init__(self,version,digest):
        self.version,self.digest=version,digest
        self.pending=None

    def propose(self, **kwargs):
        if self.pending is not None: raise ValueError("ACK pending; queue observation")
        wire,cost=encode_update(base_version=self.version,base_digest=self.digest,**kwargs)
        self.pending=wire
        return wire,cost

    def acknowledge(self,ack):
        if self.pending is None or len(ack)!=ACK.size+16 or checksum(ack[:-16])!=ack[-16:]:
            raise ValueError("invalid/unexpected ACK")
        magic,target,obs,transaction,digest=ACK.unpack(ack[:-16])
        fields=HEADER.unpack(self.pending[:HEADER.size])
        if magic!=b"SA02" or target!=self.version+1 or obs!=fields[8] or transaction!=checksum(self.pending):
            raise ValueError("ACK does not match pending transaction")
        self.version,self.digest,self.pending=target,digest,None


class TransactionReceiver:
    def __init__(self,state):
        self.state=state
        self.last_wire=self.last_ack=None
        self.expected_at=-1

    def expect(self,obs): self.expected_at=max(obs,self.expected_at)

    def status(self):
        if self.state.observed_at<self.expected_at:
            return np.zeros_like(self.state.statuses)
        return self.state.statuses.copy()

    def receive(self,wire):
        if wire==self.last_wire:
            return self.last_ack,[]
        self.state,ack,decoded=decode_update(self.state,wire)
        self.last_wire,self.last_ack=wire,ack
        return ack,decoded
