"""Locally maintainable summaries of immutable detail archives.

An archive block changes only after an absolute, decoded ROI is acknowledged.
Coarse rendering never changes archived detail. This makes one fixed sketch
updatable from transmitted JPEG blocks without historical RGB at the sender.
"""
from __future__ import annotations
import hashlib
import numpy as np
from .certificates import HEAD,encode_certificate_reference,certificate_score,joint_rows
from .reference import blocks


def digest(data):return hashlib.sha256(data).digest()[:16]


class DetailCache:
    def __init__(self,wire,config):
        self.wire=wire;self.config=dict(config)
        fields=HEAD.unpack(wire[:HEAD.size])
        self.height,self.width=fields[4:6]
        self.block=fields[3];self.count=fields[7]
        self.n=(self.height//self.block)*(self.width//self.block)

    @classmethod
    def from_rgb(cls,rgb,config):
        if config.get("summary") not in ("gaussian","dct_bound") or config.get("side") not in (4,8,16) or config.get("block")!=64 or config.get("coarse_side")!=4:
            raise ValueError("unsupported detail-cache configuration")
        if not 0<config.get("alpha_per_area",0)<1 or not 1<=config.get("risk_horizon",0)<=10000:
            raise ValueError("invalid cache risk configuration")
        if config["summary"]=="gaussian":
            rows=config["side"]**2//2*3
            actual=hashlib.sha256(joint_rows(64,rows,config["seed"]).tobytes()).hexdigest()
            if config.get("projection_sha256")!=actual:raise ValueError("shared projection implementation differs")
        method="gaussian_joint" if config["summary"]=="gaussian" else "dct_bound"
        wire=encode_certificate_reference(rgb,method,config["side"],block=64,seed=config["seed"],detail_side=4)
        return cls(wire,config)

    def scores(self,current,observation):
        if self.config["summary"]=="gaussian":
            horizon=self.config["risk_horizon"]
            # Uniform over every block and every possible current/archive pair
            # in an exogenous finite observation sequence. JPEG archive outputs
            # must be independent of adaptive atlas placement (4:4:4, fixed q).
            pairs=horizon*(horizon+1)//2
            alpha=self.config["alpha_per_area"]/(self.n*pairs)
        else:alpha=.05 # DCT bound is deterministic; this argument is unused.
        point,upper,meta=certificate_score(current,self.wire,alpha)
        if observation>self.config["risk_horizon"] and self.config["summary"]=="gaussian":
            upper=np.full_like(upper,np.inf)
        return point,upper,dict(meta,fresh_seed_required=False,
                                scope="one fixed matrix, finite-candidate union over an exogenous observation sequence per area",
                                risk_horizon=self.config["risk_horizon"],alpha_per_area=self.config["alpha_per_area"],
                                per_candidate_tail_alpha=alpha if self.config["summary"]=="gaussian" else 0)

    def updated(self,patches):
        if not patches:return DetailCache(self.wire,self.config)
        ids=np.array([i for i,_ in patches],dtype=int)
        if len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=self.n):raise ValueError("bad cache update IDs")
        atlas=np.concatenate([p for _,p in patches],axis=0)
        sub=DetailCache.from_rgb(atlas,self.config)
        old=bytearray(self.wire[HEAD.size:-16]);new=sub.wire[HEAD.size:-16]
        coefficient_bytes=self.count*3*2
        if self.config["summary"]=="gaussian":
            for j,idx in enumerate(ids):
                old[idx*coefficient_bytes:(idx+1)*coefficient_bytes]=new[j*coefficient_bytes:(j+1)*coefficient_bytes]
        else:
            old_tail=self.n*coefficient_bytes;new_tail=len(ids)*coefficient_bytes
            for j,idx in enumerate(ids):
                old[idx*coefficient_bytes:(idx+1)*coefficient_bytes]=new[j*coefficient_bytes:(j+1)*coefficient_bytes]
                old[old_tail+idx*6:old_tail+(idx+1)*6]=new[new_tail+j*6:new_tail+(j+1)*6]
        body=self.wire[:HEAD.size]+bytes(old)
        return DetailCache(body+digest(body),self.config)


def default_config(summary="gaussian",seed=20261029,side=4):
    if summary not in ("gaussian","dct_bound") or side not in (4,8,16):raise ValueError("unsupported summary")
    config=dict(schema="layered-detail-v1",summary=summary,seed=int(seed),side=int(side),
                risk_horizon=32,alpha_per_area=.05,block=64,coarse_side=4,roi_codec="JPEG 4:4:4",roi_quality=60)
    if summary=="gaussian":config["projection_sha256"]=hashlib.sha256(joint_rows(64,side*side//2*3,seed).tobytes()).hexdigest()
    return config
