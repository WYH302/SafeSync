"""Layered RGB state: immutable last-ROI detail plus an explicit coarse field.

Checksummed, versioned simulation protocol. No authentication or real-radio
claim. Sender retains only DetailCache and metadata; ACK commits that cache.
"""
from __future__ import annotations
import io,json,struct
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from PIL import Image,JpegImagePlugin
from .layered_cache import DetailCache,digest
from .pixel_wire import HEADER,COARSE,ACK,encode_update,encode_image,STABLE,UPDATED,DEFERRED


def to_layered(wire,magic=b"LA01"):
    body=magic+wire[4:-16]
    return body+digest(body)


@lru_cache(maxsize=8)
def jpeg_tables(quality):
    payload=encode_image(np.zeros((8,8,3),np.uint8),"jpeg",quality)
    with Image.open(io.BytesIO(payload)) as im:return im.quantization


def parse_frame(wire,expected_quality=None):
    if len(wire)<HEADER.size+40 or digest(wire[:-16])!=wire[-16:]:raise ValueError("layered frame checksum/truncation")
    fields=HEADER.unpack(wire[:HEADER.size])
    magic,codec,cols,b,h,w,base,target,obs,n,bd,region,size=fields
    if magic not in (b"LA00",b"LA01") or codec not in (0,1) or b!=64 or h%b or w%b or h<=0 or w<=0:raise ValueError("layered layout/schema")
    count=(h//b)*(w//b);at=HEADER.size
    status=np.frombuffer(wire[at:at+count],np.uint8).copy();at+=count
    affine=np.frombuffer(wire[at:at+24],"<f4");at+=24
    if not np.array_equal(affine,np.array([1,1,1,0,0,0],np.float32)):raise ValueError("layered frame supports explicit coarse replacement only")
    cside,cbytes=COARSE.unpack(wire[at:at+COARSE.size]);at+=COARSE.size
    expected=HEADER.size+count+24+COARSE.size+2*n+cbytes+size+16
    if len(wire)!=expected or n>count or np.any(status>3):raise ValueError("layered length/manifest")
    ids=np.frombuffer(wire[at:at+2*n],"<u2");at+=2*n
    if len(set(ids.tolist()))!=n or np.any(ids>=count) or set(np.flatnonzero(status==UPDATED))!=set(ids.tolist()):raise ValueError("layered update IDs")
    native=n==count and np.array_equal(ids,np.arange(count))
    if cols!=(w//b if native else min(8,n)):raise ValueError("layered atlas columns")
    coarse=None
    if cside:
        if cside!=4 or not cbytes:raise ValueError("bad coarse grid")
        with Image.open(io.BytesIO(wire[at:at+cbytes])) as im:coarse=np.array(im.convert("RGB"))
        if coarse.shape!=(h//16,w//16,3):raise ValueError("coarse dimensions differ")
        at+=cbytes
    elif cbytes:raise ValueError("unexpected coarse payload")
    patches=[]
    if n:
        if not size:raise ValueError("missing absolute image content")
        with Image.open(io.BytesIO(wire[at:at+size])) as im:
            if (codec==0 and im.format!="PNG") or (codec==1 and im.format!="JPEG"):raise ValueError("codec mismatch")
            if codec==1 and (JpegImagePlugin.get_sampling(im)!=0 or im.mode!="RGB"):
                raise ValueError("archive JPEG must use three-channel 4:4:4")
            if codec==1 and expected_quality is not None and im.quantization!=jpeg_tables(expected_quality):
                raise ValueError("archive JPEG quantizer changed within session")
            atlas=np.array(im.convert("RGB"))
        if atlas.shape!=(((n+cols-1)//cols)*b,cols*b,3):raise ValueError("absolute atlas dimensions")
        for j,idx in enumerate(ids):
            p=atlas[(j//cols)*b:(j//cols+1)*b,(j%cols)*b:(j%cols+1)*b]
            patches.append((int(idx),p.copy()))
    elif size:raise ValueError("unexpected image content")
    return dict(magic=magic,codec=codec,block=b,shape=(h,w),base=base,target=target,observation=obs,
                base_digest=bd,region=region,statuses=status,coarse=coarse,patches=patches)


@dataclass
class LayeredState:
    archive:np.ndarray
    region:bytes
    cache:DetailCache
    coarse:np.ndarray|None=None
    version:int=1
    observation:int=0
    history:bytes=b""
    statuses:np.ndarray|None=None

    def render(self):
        if self.coarse is None:return self.archive.copy()
        h,w=self.archive.shape[:2]
        x=self.archive.astype(np.float64)
        mean=x.reshape(h//16,16,w//16,16,3).mean((1,3))
        return np.clip(np.rint(x+(self.coarse-mean).repeat(16,0).repeat(16,1)),0,255).astype(np.uint8)


def history_after(old,wire,cache):return digest(old+digest(wire)+digest(cache.wire))


def ack_for(wire,version,observation,history):
    body=ACK.pack(b"LK01",version,observation,digest(wire),history)
    return body+digest(body)


class LayeredReceiver:
    def __init__(self):
        self.state=None;self.last_wire=self.last_ack=None;self.expected_at=-1

    def expect(self,obs):self.expected_at=max(obs,self.expected_at)

    def receive(self,wire):
        if wire==self.last_wire:return self.last_ack
        if wire[:4]==b"LB01":
            if self.state is not None:raise ValueError("bootstrap replay")
            length=struct.unpack("<I",wire[4:8])[0]
            config=json.loads(wire[8:8+length].decode("utf-8"))
            frame=parse_frame(wire[8+length:])
            if frame["magic"]!=b"LA00" or frame["codec"]!=0 or frame["base"]!=0 or frame["target"]!=1 or frame["observation"]!=0 or frame["coarse"] is not None:raise ValueError("invalid bootstrap")
            h,w=frame["shape"];count=h*w//4096
            if len(frame["patches"])!=count:raise ValueError("bootstrap lacks complete evidence")
            archive=np.empty((h,w,3),np.uint8)
            for idx,patch in frame["patches"]:
                y,x=divmod(idx,w//64);archive[y*64:(y+1)*64,x*64:(x+1)*64]=patch
            cache=DetailCache.from_rgb(archive,config)
            history=history_after(b"\x00"*16,wire,cache)
            self.state=LayeredState(archive,frame["region"],cache,history=history,statuses=frame["statuses"])
        else:
            if self.state is None:raise ValueError("missing bootstrap")
            prior=self.state;frame=parse_frame(wire,prior.cache.config["roi_quality"])
            if frame["magic"]!=b"LA01" or frame["codec"]!=1 or frame["shape"]!=prior.archive.shape[:2] or frame["region"]!=prior.region:raise ValueError("wrong layered session")
            if frame["base"]!=prior.version or frame["target"]!=prior.version+1 or frame["base_digest"]!=prior.history or frame["observation"]<=prior.observation or frame["coarse"] is None:raise ValueError("stale/mismatched layered update")
            archive=prior.archive.copy();w=archive.shape[1]
            for idx,patch in frame["patches"]:
                y,x=divmod(idx,w//64);archive[y*64:(y+1)*64,x*64:(x+1)*64]=patch
            cache=prior.cache.updated(frame["patches"])
            history=history_after(prior.history,wire,cache)
            self.state=LayeredState(archive,prior.region,cache,frame["coarse"],frame["target"],frame["observation"],history,frame["statuses"])
        ack=ack_for(wire,self.state.version,self.state.observation,self.state.history)
        self.last_wire,self.last_ack=wire,ack
        return ack

    def status(self):
        if self.state is None:return None
        return self.state.statuses.copy() if self.state.observation>=self.expected_at else np.zeros_like(self.state.statuses)


class LayeredSender:
    def __init__(self,keep_full_archive=False):
        self.cache=None;self.version=0;self.history=b"\x00"*16;self.pending=None
        self.region=None;self.shape=None;self.observation=-1;self.provisional=None
        self.keep_full_archive=keep_full_archive;self.full_archive=None;self.provisional_archive=None

    def bootstrap(self,image,region,config):
        if self.pending is not None or self.version:raise ValueError("bootstrap already pending/committed")
        a=np.asarray(image,np.uint8);h,w=a.shape[:2];n=h*w//4096
        frame,cost=encode_update(shape=(h,w),block=64,region=region,base_version=0,base_digest=b"\x00"*16,
            observed_at=0,current_rgb=a,indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec="png")
        config_bytes=json.dumps(config,sort_keys=True,separators=(",",":")).encode("utf-8")
        wire=b"LB01"+struct.pack("<I",len(config_bytes))+config_bytes+to_layered(frame,b"LA00")
        # Derive the cache from the actual encoded content, not raw source pixels.
        decoded=parse_frame(wire[8+len(config_bytes):]);archive=np.empty_like(a)
        for idx,patch in decoded["patches"]:
            y,x=divmod(idx,w//64);archive[y*64:(y+1)*64,x*64:(x+1)*64]=patch
        cache=DetailCache.from_rgb(archive,config)
        self.region,self.shape=region,(h,w)
        self.pending=wire;self.provisional=(cache,history_after(self.history,wire,cache),1,0)
        if self.keep_full_archive:self.provisional_archive=archive
        return wire,dict(bootstrap_bytes=len(wire),config_bytes=8+len(config_bytes),components=cost)

    def propose(self,current,observation,budget,*,ranking="bound",random_seed=1729):
        if self.pending is not None or self.cache is None:raise ValueError("ACK pending or no bootstrap")
        if observation<=self.observation:raise ValueError("stale observation")
        from scripts.run_pixel_sequences import choose_packet
        if ranking=="exact":
            if self.full_archive is None:raise ValueError("exact baseline requires its charged full archive")
            from .reference import restore_coarse,exact_mse
            point=upper=exact_mse(current,restore_coarse(self.full_archive,current))
            meta=dict(kind="full archive exact RGB error",scope="full-cache baseline",fresh_seed_required=False)
        else:point,upper,meta=self.cache.scores(current,observation)
        tau=10**(-self.cache.config.get("gate_psnr",25)/10)
        if ranking=="random":
            rng=np.random.default_rng(random_seed)
            priority=np.where(upper>tau,tau+1+rng.random(len(upper)),0)
        elif ranking in ("bound","exact"):
            # After the declared statistical horizon, no block is certified;
            # retain the point estimate only for prioritizing overdue content.
            priority=np.where(np.isfinite(upper),upper,tau+1+point)
        else:raise ValueError("unknown ranking")
        class DescriptorOnly:pass
        descriptor=DescriptorOnly();descriptor.version=self.version;descriptor.digest=self.history;descriptor.pending=None
        frame,cost,selected,trials=choose_packet(descriptor,np.asarray(current,np.uint8),self.region,observation,priority,tau,budget,
                                               quality=self.cache.config["roi_quality"])
        wire=to_layered(frame);parsed=parse_frame(wire,self.cache.config["roi_quality"])
        cache=self.cache.updated(parsed["patches"])
        history=history_after(self.history,wire,cache)
        self.pending=wire;self.provisional=(cache,history,self.version+1,observation)
        if self.keep_full_archive:
            archive=self.full_archive.copy();width=self.shape[1]
            for idx,patch in parsed["patches"]:
                y,x=divmod(idx,width//64);archive[y*64:(y+1)*64,x*64:(x+1)*64]=patch
            self.provisional_archive=archive
        return wire,dict(cost=cost,selected=selected,encoding_trials=trials,cache_bytes=len(cache.wire),reference_up_bytes=0,
                         bound=upper,point=point,**meta)

    def acknowledge(self,wire):
        if self.pending is None or len(wire)!=ACK.size+16 or digest(wire[:-16])!=wire[-16:]:raise ValueError("bad/unexpected layered ACK")
        magic,version,obs,transaction,history=ACK.unpack(wire[:-16])
        cache,expected,v,t=self.provisional
        if magic!=b"LK01" or (version,obs,transaction,history)!=(v,t,digest(self.pending),expected):raise ValueError("layered ACK mismatch")
        self.cache,self.history,self.version,self.observation=cache,history,version,obs
        if self.keep_full_archive:self.full_archive=self.provisional_archive
        self.provisional_archive=None
        self.pending=self.provisional=None
