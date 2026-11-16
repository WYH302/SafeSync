"""Query compact history only for regions degraded by the current JPEG codec.

Uses standard JPEG as a complete current-image fallback. A quality bound, not
semantic GT, controls historical-detail substitution. Query/reply/ACK bytes and
the extra query round trip are explicit. This is a research application format.
"""
from __future__ import annotations
import hashlib,io,struct,time
from dataclasses import dataclass
import numpy as np
from PIL import Image
from .pixel_wire import PixelState,encode_image,checksum,ACK
from .reference import blocks,restore_coarse,exact_mse,bind_reference,unbind_reference
from .certificates import encode_certificate_reference,certificate_score,HEAD as CERT_HEAD
from .codec_reference import encode_codec_reference,codec_reference_score

HEAD=struct.Struct("<4sHHHIIII16s16s16sI")
REQUEST=struct.Struct("<4sHHHIII16s16s16s")
HISTORY_GOOD,CURRENT_GOOD,CURRENT_POOR=1,2,3


@dataclass(frozen=True)
class Descriptor:
    shape:tuple
    block:int
    region:bytes
    version:int
    digest:bytes
    archive_digest:bytes=b"\x00"*16


def descriptor(state,archive_digest=b"\x00"*16):
    return Descriptor(state.rgb.shape[:2],state.block,state.region,state.version,state.digest(),archive_digest)


def query_packet(desc,obs,indices):
    ids=np.asarray(indices,"<u2")
    h,w=desc.shape
    head=REQUEST.pack(b"SQ05",desc.block,h,w,desc.version,obs,len(ids),desc.digest,desc.region,desc.archive_digest)
    body=head+ids.tobytes()
    return body+checksum(body)


def parse_query(wire,state):
    if len(wire)<REQUEST.size+16 or checksum(wire[:-16])!=wire[-16:]:raise ValueError("bad query checksum")
    magic,b,h,w,v,obs,n,digest,region,archive_digest=REQUEST.unpack(wire[:REQUEST.size])
    if magic!=b"SQ05" or (b,h,w,v,digest,region)!=(state.block,*state.rgb.shape[:2],state.version,state.digest(),state.region):raise ValueError("query state mismatch")
    if len(wire)!=REQUEST.size+2*n+16 or obs<=state.observed_at:raise ValueError("stale or truncated query")
    ids=np.frombuffer(wire[REQUEST.size:-16],"<u2")
    if len(set(ids.tolist()))!=n or np.any(ids>=h*w//(b*b)):raise ValueError("bad query indices")
    return obs,ids,archive_digest


def ground_reply(query,state,method,side,seed,archive_rgb=None):
    obs,ids,archive_digest=parse_query(query,state)
    if not len(ids):raise ValueError("empty query")
    if archive_digest!=b"\x00"*16:
        if archive_rgb is None or checksum(np.asarray(archive_rgb,np.uint8).tobytes())!=archive_digest:raise ValueError("unknown evidence archive")
        source=archive_rgb
    else:source=state.rgb
    # Only queried historical blocks enter the reply. The sender never receives
    # the other historical pixels or an uncharged full reference.
    atlas=np.concatenate(blocks(source,state.block)[ids],axis=0)
    if method=="gaussian":
        core=encode_certificate_reference(atlas,"gaussian_joint",side,seed=seed,detail_side=4)
    elif method=="dct_bound":
        core=encode_certificate_reference(atlas,"dct_bound",side,seed=seed,detail_side=4)
    elif method in ("codec_jpeg","codec_jpeg2000"):
        cap=len(ids)*side*side*3+CERT_HEAD.size+16
        codec=method.removeprefix("codec_")
        try:core=encode_codec_reference(atlas,cap,codec)
        except ValueError:core=b"" # Declined reference is explicit and counted.
    else:raise ValueError("unsupported reference method")
    return checksum(query)+bind_reference(core,state.version,state.digest(),obs)


def jpeg_decode(payload,shape):
    with Image.open(io.BytesIO(payload)) as im:
        if im.format not in ("JPEG","JPEG2000"):raise ValueError("unsupported base image codec")
        rgb=np.array(im.convert("RGB"))
    if rgb.shape!=(*shape,3):raise ValueError("JPEG dimensions differ")
    return rgb


def make_fusion(desc,obs,jpeg,current,history_ids,status):
    h,w=desc.shape;b=desc.block
    ids=np.asarray(history_ids,"<u2")
    fine=blocks(current,b)[ids]
    if len(ids):coarse=np.rint(fine.reshape(len(ids),4,b//4,4,b//4,3).mean((2,4))).astype(np.uint8)
    else:coarse=np.empty((0,4,4,3),np.uint8)
    head=HEAD.pack(b"SF05",b,h,w,desc.version,desc.version+1,obs,len(ids),desc.digest,desc.region,desc.archive_digest,len(jpeg))
    body=head+np.asarray(status,np.uint8).tobytes()+ids.tobytes()+coarse.tobytes()+jpeg
    return body+checksum(body)


def receive_fusion(prior,wire,archive_rgb=None):
    if len(wire)<HEAD.size+16 or checksum(wire[:-16])!=wire[-16:]:raise ValueError("bad fusion checksum")
    magic,b,h,w,v,target,obs,n,digest,region,archive_digest,size=HEAD.unpack(wire[:HEAD.size])
    if magic!=b"SF05" or (b,h,w,v,digest,region)!=(prior.block,*prior.rgb.shape[:2],prior.version,prior.digest(),prior.region):raise ValueError("fusion reference mismatch")
    if target!=v+1 or obs<=prior.observed_at:raise ValueError("stale fusion transaction")
    count=h*w//(b*b)
    if len(wire)!=HEAD.size+count+2*n+48*n+size+16:raise ValueError("fusion length mismatch")
    at=HEAD.size;status=np.frombuffer(wire[at:at+count],np.uint8).copy();at+=count
    ids=np.frombuffer(wire[at:at+2*n],"<u2");at+=2*n
    coarse=np.frombuffer(wire[at:at+48*n],np.uint8).reshape(n,4,4,3);at+=48*n
    if len(set(ids.tolist()))!=n or np.any(ids>=count) or np.any((status<1)|(status>3)) or set(np.flatnonzero(status==HISTORY_GOOD))!=set(ids.tolist()):raise ValueError("fusion manifest mismatch")
    if archive_digest!=b"\x00"*16:
        if archive_rgb is None or checksum(np.asarray(archive_rgb,np.uint8).tobytes())!=archive_digest:raise ValueError("fusion archive identity mismatch")
        historical=archive_rgb
    else:historical=prior.rgb
    if not size:
        if n or np.any(status!=CURRENT_POOR):raise ValueError("empty current payload cannot confirm evidence")
        rgb=prior.rgb.copy()
    else:rgb=jpeg_decode(wire[at:at+size],(h,w))
    for k,idx in enumerate(ids):
        y,x=divmod(int(idx),w//b)
        old=historical[y*b:(y+1)*b,x*b:(x+1)*b].astype(np.float64)
        old_mean=old.reshape(4,b//4,4,b//4,3).mean((1,3))
        restored=np.clip(np.rint(old+(coarse[k]-old_mean).repeat(b//4,0).repeat(b//4,1)),0,255).astype(np.uint8)
        rgb[y*b:(y+1)*b,x*b:(x+1)*b]=restored
    state=PixelState(rgb,region,b,target,obs,status)
    ack_body=ACK.pack(b"SA04",target,obs,checksum(wire),state.digest())
    return state,ack_body+checksum(ack_body)


def choose_fusion(current,desc,obs,budget,reply_callback,*,method="gaussian",side=4,psnr=25,query_max=64,alpha=.05,seed=0,full_history=None,base_codec="jpeg",risk_scope="decision"):
    """Sender sees Descriptor, current RGB, and explicit query replies only.

    ``full_history`` is allowed only for the larger-cache full_reference baseline.
    A fresh seed is needed for each observation after the query is selected.
    Returns actual query/reply/final-image packets and measurable phase costs.
    """
    a=np.asarray(current,np.uint8);h,w=desc.shape;b=desc.block;n=h*w//(b*b);tau=10**(-psnr/10)
    if a.shape!=(h,w,3):raise ValueError("current shape differs")
    cost0=HEAD.size+n+16
    reserve=0 if method in ("jpeg","full_reference") else REQUEST.size+2*min(query_max,n)+16
    cache={};trials=0
    start=time.perf_counter()
    def encode(q):
        nonlocal trials
        if q not in cache:
            payload=encode_image(a,base_codec,q);image=jpeg_decode(payload,(h,w))
            cache[q]=(payload,exact_mse(a,image,b));trials+=1
        return cache[q]
    if base_codec not in ("jpeg","jpeg2000") or risk_scope not in ("decision","frame"):raise ValueError("unsupported operating mode")
    lo,hi=1,(95 if base_codec=="jpeg" else 1024);qmax=None
    while lo<=hi:
        q=(lo+hi)//2;jpeg,error=encode(q)
        fit=len(jpeg)+cost0+reserve<=budget
        if fit:qmax=q
        if base_codec=="jpeg":
            if fit:lo=q+1
            else:hi=q-1
        else:
            if fit:hi=q-1
            else:lo=q+1
    if qmax is None:
        packet=make_fusion(desc,obs,b"",a,[],np.full(n,CURRENT_POOR,np.uint8))
        if len(packet)>budget:raise ValueError("budget cannot hold even the deferred manifest")
        return packet,b"",b"",dict(server_reference_ms=0,sender_reference_score_ms=0,sender_total_ms=(time.perf_counter()-start)*1000,
            codec_trials=trials,chosen_quality=0,reference_bytes=0,query_bytes=0,down_bytes=len(packet),reused_blocks=0,
            query_round_trips=0,upper_objective=None,sender_persistent_reference_bytes=0 if full_history is None else full_history.nbytes)
    jpeg,errors=encode(qmax)
    query=reply=b"";server_ms=score_ms=0;hist_bound=np.full(n,np.inf)
    if method=="full_reference":
        if full_history is None:raise ValueError("full-reference baseline requires its charged full cache")
        hist_bound=exact_mse(a,restore_coarse(full_history,a),b)
    elif method!="jpeg":
        ids=np.flatnonzero(errors>tau)
        ids=ids[np.argsort(-errors[ids],kind="stable")][:query_max]
        if len(ids):
            query=query_packet(desc,obs,ids)
            t=time.perf_counter();reply=reply_callback(query,method,side,seed);server_ms=(time.perf_counter()-t)*1000
            if reply[:16]!=checksum(query):raise ValueError("reply/query mismatch")
            core=unbind_reference(reply[16:],desc.version,desc.digest,obs)
            if core:
                t=time.perf_counter();atlas=np.concatenate(blocks(a,b)[ids],axis=0)
                if method.startswith("codec_"):_,upper=codec_reference_score(atlas,core)
                else:_,upper,_=certificate_score(atlas,core,alpha=alpha/len(ids) if risk_scope=="frame" else alpha)
                score_ms=(time.perf_counter()-t)*1000;hist_bound[ids]=upper
    if method=="jpeg":choices=[qmax]
    elif base_codec=="jpeg":choices=sorted({1,qmax,max(1,qmax-1),max(1,qmax-2),max(1,qmax-4),max(1,qmax-8),max(1,qmax//2)},reverse=True)
    else:choices=sorted({qmax,qmax+1,qmax+2,qmax+4,qmax+8,qmax*2,qmax*4})
    best=None
    for q in choices:
        jpeg,error=encode(q)
        room=budget-len(query)-cost0-len(jpeg)
        if room<0:continue
        gain=error-hist_bound
        ids=np.flatnonzero((gain>0)&(hist_bound<=tau))
        ids=ids[np.argsort(-gain[ids],kind="stable")][:room//50]
        objective=float(error.sum()-gain[ids].sum())
        status=np.where(error<=tau,CURRENT_GOOD,CURRENT_POOR).astype(np.uint8);status[ids]=HISTORY_GOOD
        packet=make_fusion(desc,obs,jpeg,a,ids,status)
        if len(packet)+len(query)>budget:raise AssertionError("actual query+image budget exceeded")
        if best is None or objective<best[0]:best=(objective,packet,q,len(ids))
    if best is None:raise ValueError("no fusion candidate fits exact budget")
    objective,packet,quality,reused=best
    timing=dict(server_reference_ms=server_ms,sender_reference_score_ms=score_ms,
                sender_total_ms=(time.perf_counter()-start)*1000-server_ms,codec_trials=trials,chosen_quality=quality,
                reference_bytes=len(reply),query_bytes=len(query),down_bytes=len(query)+len(packet),
                reused_blocks=reused,query_round_trips=int(bool(query)),upper_objective=objective,
                sender_persistent_reference_bytes=0 if full_history is None else full_history.nbytes)
    return packet,query,reply,timing


class FusionReceiver:
    def __init__(self,state,archive_rgb=None):
        self.state=state;self.last_wire=self.last_ack=None;self.expected_at=-1
        self.archive_rgb=archive_rgb

    def expect(self,obs):self.expected_at=max(obs,self.expected_at)

    def receive(self,wire):
        if wire==self.last_wire:return self.last_ack
        self.state,ack=receive_fusion(self.state,wire,self.archive_rgb)
        self.last_wire,self.last_ack=wire,ack
        return ack

    def current_status(self):
        return self.state.statuses.copy() if self.state.observed_at>=self.expected_at else np.zeros_like(self.state.statuses)


class FusionSender:
    def __init__(self,desc,full_cache=None,archive_cache=None):
        self.desc=desc;self.full_cache=full_cache;self.pending=None;self.provisional=None
        self.archive_cache=archive_cache

    def propose(self,current,obs,budget,reply_callback,**kwargs):
        if self.pending is not None:raise ValueError("ACK pending; do not overtake transaction")
        history=self.archive_cache if self.archive_cache is not None else (None if self.full_cache is None else self.full_cache.rgb)
        result=choose_fusion(current,self.desc,obs,budget,reply_callback,full_history=history,**kwargs)
        self.pending=result[0]
        if self.full_cache is not None:
            self.provisional,_=receive_fusion(self.full_cache,self.pending,self.archive_cache)
            result[3]["sender_persistent_reference_bytes"]=self.full_cache.rgb.nbytes+(0 if self.archive_cache is None else self.archive_cache.nbytes)
        return result

    def acknowledge(self,wire):
        if self.pending is None or len(wire)!=ACK.size+16 or checksum(wire[:-16])!=wire[-16:]:raise ValueError("bad/unexpected ACK")
        magic,v,obs,transaction,digest=ACK.unpack(wire[:-16])
        fields=HEAD.unpack(self.pending[:HEAD.size])
        if magic!=b"SA04" or v!=self.desc.version+1 or obs!=fields[6] or transaction!=checksum(self.pending):raise ValueError("ACK transaction mismatch")
        if self.provisional is not None and digest!=self.provisional.digest():raise ValueError("sender decoded cache differs from ACK")
        self.desc=Descriptor(self.desc.shape,self.desc.block,self.desc.region,v,digest,self.desc.archive_digest)
        if self.provisional is not None:self.full_cache=self.provisional
        self.provisional=self.pending=None
