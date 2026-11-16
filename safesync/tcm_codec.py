"""Real-byte public TCM comparator in a registered standalone RGB container."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .learned_codec import Candidate,HEAD
from .task_evidence_wire import checksum
from .tcm_runtime import initialize,load_model

MAGIC = b"TC10"
SCALES = (1.,.75,.5,.375,.25)


class TCMCodec:
    def __init__(self,root:Path):
        import torch
        self.root,self.torch = root,torch
        self.bridge,self.model_class = initialize(root)
        torch.cuda.set_per_process_memory_fraction(.2)
        path = root/"resources/modern_codec/weights/registry.json"
        self.registry = json.loads(path.read_text("utf-8"))
        if self.registry["status"] != "complete" or len(self.registry["models"]) != 7:
            raise ValueError("The fixed complete public TCM menu is required")
        self.models,self.weights = {},{}
        self.runtime = dict(architecture="Authors' LIC-TCM",source_commit="6c6d7a364817a4c372bd210dbbcb175f9172caa0",
            torch=torch.__version__,cuda=torch.version.cuda,device="cuda",gpu=torch.cuda.get_device_name(),dtype="float32",tf32=False,
            entropy="Official CompressAI1.2.8 native rANS through local WSL bridge; buffered aggregation and exact prefix-replay API adapter",
            preprocessing="RGB/255, fixed Lanczos scales, zero right/bottom padding to128 multiples with minimum256, bicubic native reconstruction",
            training=False,gpu_memory_fraction=.2,minimum_padded_side=256)

    def model(self,slot):
        if slot not in self.models:
            item = next(r for r in self.registry["models"] if r["slot"] == slot)
            path = self.root/item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Registered public TCM checkpoint changed")
            model = load_model(self.root,self.model_class,path,item["channels_N"])
            self.models[slot] = model
            self.weights[slot] = dict(**item,parameter_count=sum(p.numel() for p in model.parameters()),
                backbone_operators_changed=False,requires_grad=any(p.requires_grad for p in model.parameters()))
        return self.models[slot]

    def prepare(self):
        for item in self.registry["models"]:
            self.model(item["slot"])
        return dict(runtime=self.runtime,weights=list(self.weights.values()))

    @staticmethod
    def padded_shape(coded):
        return tuple(max(256,math.ceil(n/128)*128) for n in coded)

    def _decode(self,slot,streams,latent_shape,coded_shape,native_shape):
        net = self.model(slot)
        with self.torch.inference_mode():
            image = net.decompress([[streams[0]],[streams[1]]],latent_shape)["x_hat"]
            expected = self.padded_shape(coded_shape)
            if tuple(image.shape) != (1,3,*expected):
                raise ValueError("TCM decoded padding differs")
            h,w = coded_shape
            rgb = image[0,:,:h,:w].clamp(0,1).mul(255).round().to(self.torch.uint8).permute(1,2,0).cpu().numpy()
        h,w = native_shape
        return np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BICUBIC))

    def encode_candidates(self,current):
        if current.dtype != np.uint8 or current.ndim != 3 or current.shape[2] != 3:
            raise ValueError("TCM expects uint8 RGB")
        h,w = current.shape[:2]
        candidates = []
        for item in self.registry["models"]:
            slot = item["slot"]
            net = self.model(slot)
            for scale in SCALES:
                coded = max(1,round(h*scale)),max(1,round(w*scale))
                ph,pw = self.padded_shape(coded)
                ch,cw = coded
                small = np.asarray(Image.fromarray(current).resize((cw,ch),Image.Resampling.LANCZOS))
                x = self.torch.from_numpy(small.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/255
                x = self.torch.nn.functional.pad(x,(0,pw-cw,0,ph-ch),value=0)
                self.torch.cuda.synchronize()
                start = time.perf_counter()
                with self.torch.inference_mode():
                    encoded = net.compress(x)
                self.torch.cuda.synchronize()
                compress_ms = (time.perf_counter()-start)*1000
                if len(encoded["strings"]) != 2 or any(len(group) != 1 for group in encoded["strings"]):
                    raise ValueError("TCM y/z stream layout differs")
                streams = tuple(bytes(group[0]) for group in encoded["strings"])
                latent = tuple(map(int,encoded["shape"]))
                if latent != (ph//64,pw//64):
                    raise ValueError("TCM latent shape differs from registered padding")
                start = time.perf_counter()
                rgb = self._decode(slot,streams,latent,coded,(h,w))
                self.torch.cuda.synchronize()
                decode_ms = (time.perf_counter()-start)*1000
                mse = float(np.mean(((current.astype(np.float64)-rgb)/255)**2))
                candidates.append(Candidate(slot,(h,w),coded,(ph,pw),latent,item["sha256"],streams,
                    mse,compress_ms,decode_ms,hashlib.sha256(rgb.tobytes()).hexdigest()))
        return candidates

    def pack(self,candidate,*,region,base_version,base_digest,observation):
        y,z = candidate.streams
        head = HEAD.pack(MAGIC,candidate.quality,*candidate.native_shape,*candidate.coded_shape,
            base_version,observation,base_digest,region,len(y),len(z))
        body = head+y+z
        return body+checksum(body)

    def select(self,candidates,*,budget,region,base_version,base_digest,observation):
        fitting = [c for c in candidates if c.packet_bytes <= budget]
        if not fitting:
            raise ValueError("No fixed TCM candidate fits the actual task-state envelope")
        best = min(fitting,key=lambda c:c.sender_pixel_mse)
        packet = self.pack(best,region=region,base_version=base_version,base_digest=base_digest,observation=observation)
        if len(packet) != best.packet_bytes or len(packet) > budget:
            raise AssertionError("TCM actual size differs from selection")
        item = next(r for r in self.registry["models"] if r["slot"] == best.quality)
        metadata = dict(best.metadata(),wire_schema="TC10",architecture="LIC-TCM",channels_N=item["channels_N"],
            lambda_mse=item["lambda_mse"],model_slot=best.quality,actual_candidates=len(candidates),fitting_candidates=len(fitting))
        return packet,metadata

    def decode_packet(self,packet):
        if len(packet) < HEAD.size+16 or checksum(packet[:-16]) != packet[-16:]:
            raise ValueError("TCM image packet checksum or length")
        magic,slot,h,w,ch,cw,base,observed,old_digest,region,ny,nz = HEAD.unpack_from(packet)
        if (magic != MAGIC or not 1 <= slot <= 7 or not 0 < ch <= h or not 0 < cw <= w
                or not ny or not nz or len(packet) != HEAD.size+ny+nz+16):
            raise ValueError("TCM registered model, dimensions or stream layout")
        streams = packet[HEAD.size:HEAD.size+ny],packet[HEAD.size+ny:-16]
        ph,pw = self.padded_shape((ch,cw))
        rgb = self._decode(slot,streams,(ph//64,pw//64),(ch,cw),(h,w))
        return rgb,dict(base_version=base,observation=observed,base_digest=old_digest,region=region,
            native_shape=(h,w),coded_shape=(ch,cw),model_slot=slot)
