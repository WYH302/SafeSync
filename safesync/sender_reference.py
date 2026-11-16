"""Pixel-free sender references for current full-frame task/evidence updates."""
from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from . import task_evidence_wire as wire
from .pixel_wire import PixelState

HEADER = struct.Struct("<4s16s16sIiHHHHHH16s16s")


@dataclass(frozen=True)
class _ImageDimensions:
    shape: tuple[int,int,int]


@dataclass(frozen=True)
class SenderReference:
    height: int
    width: int
    block: int
    region: bytes
    model: bytes
    version: int
    observed_at: int
    labels: np.ndarray
    statuses: np.ndarray
    state_digest: bytes
    pixel_digest: bytes

    def __post_init__(self):
        if (not 0 < self.height <= 65535 or not 0 < self.width <= 65535 or self.block <= 0
                or self.height % self.block or self.width % self.block
                or self.labels.dtype != np.uint8 or self.labels.ndim != 2
                or self.statuses.dtype != np.uint8 or self.statuses.shape != (self.height*self.width//self.block**2,)
                or np.any((self.statuses != wire.UPDATED) & (self.statuses != wire.DEFERRED))
                or any(len(value) != 16 for value in (self.region,self.model,self.state_digest,self.pixel_digest))):
            raise ValueError("Invalid compact sender reference")
        labels,statuses = self.labels.copy(),self.statuses.copy()
        labels.setflags(write=False)
        statuses.setflags(write=False)
        object.__setattr__(self,"labels",labels)
        object.__setattr__(self,"statuses",statuses)

    @classmethod
    def from_decoded_state(cls,state):
        pixel = PixelState(state.rgb,state.region,state.block,state.version,state.observed_at,state.statuses)
        return cls(*state.rgb.shape[:2],state.block,state.region,state.model,state.version,state.observed_at,
            state.labels,state.statuses,state.digest(),pixel.digest())

    @property
    def rgb(self):
        # The shared encoder reads only rgb.shape. Receiver decoding retains
        # the ordinary TaskEvidenceState with actual pixels on the ground.
        return _ImageDimensions((self.height,self.width,3))

    def digest(self):
        return self.state_digest

    def encode(self,labels,*,observation,payload,image_mode,indices=()):
        if image_mode not in (wire.FULL,wire.LEARNED,wire.LAYERED):
            raise ValueError("Pixel-free references support current full-frame image modes")
        return wire.encode(self,labels,observation=observation,payload=payload,image_mode=image_mode,indices=indices)

    def acknowledged(self,packet,ack,decoded_current,labels):
        if (len(packet) < wire.HEADER.size+16 or wire.checksum(packet[:-16]) != packet[-16:]
                or len(ack) != wire.ACK_BYTES or wire.checksum(ack[:-16]) != ack[-16:]):
            raise ValueError("Compact-reference transaction or acknowledgment checksum")
        head = wire.HEADER.unpack_from(packet)
        magic,region,model,base,target,observation,h,w,block,lh,lw,count,lm,im,ns,nl,ni,prior_digest,label_digest = head
        if (magic != b"TE01" or region != self.region or model != self.model or base != self.version
                or target != base+1 or observation <= self.observed_at or prior_digest != self.state_digest
                or (h,w,block) != (self.height,self.width,self.block) or (lh,lw) != self.labels.shape or (lh,lw) != labels.shape
                or labels.dtype != np.uint8 or wire.checksum(labels.tobytes()) != label_digest
                or ns != len(self.statuses) or count > ns or lm not in (0,1)
                or im not in (wire.FULL,wire.LEARNED,wire.LAYERED)
                or (im in (wire.FULL,wire.LEARNED) and (count or not ni))
                or len(packet) != wire.HEADER.size+ns+count*2+nl+ni+16):
            raise ValueError("Compact-reference transaction identity or current task")
        if not np.all(np.frombuffer(packet,np.uint8,count=ns,offset=wire.HEADER.size) == wire.UPDATED):
            raise ValueError("A current full-frame manifest must update every cell")
        status = np.full(len(self.statuses),wire.UPDATED,np.uint8)
        current = wire.TaskEvidenceState(decoded_current,labels,region,model,target,observation,status,block)
        expected = wire.ACK.pack(b"TA01",target,observation,wire.checksum(packet),current.digest())
        if ack[:-16] != expected:
            raise ValueError("Acknowledgment differs from the locally decoded current state")
        return self.from_decoded_state(current)

    def serialize(self):
        header = HEADER.pack(b"SRF1",self.region,self.model,self.version,self.observed_at,self.height,self.width,
            self.block,*self.labels.shape,len(self.statuses),self.state_digest,self.pixel_digest)
        body = header+self.labels.tobytes()+self.statuses.tobytes()
        return body+wire.checksum(body)

    @classmethod
    def deserialize(cls,encoded):
        if len(encoded) < HEADER.size+16 or wire.checksum(encoded[:-16]) != encoded[-16:]:
            raise ValueError("Compact-reference storage checksum")
        magic,region,model,version,observed,h,w,block,lh,lw,ns,state_digest,pixel_digest = HEADER.unpack_from(encoded)
        if magic != b"SRF1" or len(encoded) != HEADER.size+lh*lw+ns+16:
            raise ValueError("Compact-reference storage layout")
        offset = HEADER.size
        labels = np.frombuffer(encoded,np.uint8,count=lh*lw,offset=offset).copy().reshape(lh,lw)
        statuses = np.frombuffer(encoded,np.uint8,count=ns,offset=offset+lh*lw).copy()
        return cls(h,w,block,region,model,version,observed,labels,statuses,state_digest,pixel_digest)
