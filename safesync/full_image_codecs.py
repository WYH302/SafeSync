"""Compact full-image alternatives, including JPEG 4:2:0 and resizing.

No sparse-update indices are charged when a single whole image is sent. All
sources/headers/decoded resizing are explicit; the receiver has no current RGB.
Sender-side pixel-MSE selection uses its own current observation, never labels.
"""
from __future__ import annotations
import hashlib,io,struct,time
import numpy as np
from PIL import Image
from .pixel_wire import PixelState,checksum,ACK,UPDATED,DEFERRED

HEAD=struct.Struct("<4sBBHHHHIII16s16sI")
KINDS={"jpeg444":1,"jpeg420":2,"jpeg2000":3,"webp":4}


def encode_payload(image,codec,quality):
    output=io.BytesIO()
    if codec.startswith("jpeg") and codec!="jpeg2000":
        image.save(output,format="JPEG",quality=int(quality),subsampling=2 if codec=="jpeg420" else 0,optimize=True)
    elif codec=="jpeg2000":
        image.save(output,format="JPEG2000",irreversible=True,quality_mode="rates",quality_layers=[float(quality)])
    elif codec=="webp":image.save(output,format="WEBP",quality=int(quality),method=4)
    else:raise ValueError("unknown full-image codec")
    return output.getvalue()


def make_packet(payload,*,codec,scale_shape,native_shape,region,base_version,base_digest,observation):
    h,w=native_shape;ch,cw=scale_shape
    head=HEAD.pack(b"IF06",KINDS[codec],int(not payload),h,w,ch,cw,base_version,base_version+1,observation,base_digest,region,len(payload))
    body=head+payload
    return body+checksum(body)


def receive(prior,wire):
    if len(wire)<HEAD.size+16 or checksum(wire[:-16])!=wire[-16:]:raise ValueError("full-image checksum/length")
    magic,kind,empty,h,w,ch,cw,base,target,obs,digest,region,size=HEAD.unpack(wire[:HEAD.size])
    if magic!=b"IF06" or kind not in KINDS.values() or empty not in (0,1):raise ValueError("unknown full-image schema")
    if (h,w)!=prior.rgb.shape[:2] or region!=prior.region or base!=prior.version or digest!=prior.digest() or target!=base+1 or obs<=prior.observed_at:raise ValueError("full-image identity/version mismatch")
    if len(wire)!=HEAD.size+size+16 or bool(empty)!=bool(size==0) or not (0<ch<=h and 0<cw<=w):raise ValueError("full-image payload layout")
    if empty:
        rgb=prior.rgb.copy();status=np.full_like(prior.statuses,DEFERRED)
    else:
        with Image.open(io.BytesIO(wire[HEAD.size:-16])) as image:
            if image.size!=(cw,ch):raise ValueError("coded image dimensions differ")
            rgb=np.array(image.convert("RGB").resize((w,h),Image.Resampling.BICUBIC))
        status=np.full_like(prior.statuses,UPDATED)
    state=PixelState(rgb,prior.region,prior.block,target,obs,status)
    ack=ACK.pack(b"IA06",target,obs,checksum(wire),state.digest())
    return state,ack+checksum(ack)


def fit(current,*,codec,budget,region,base_version,base_digest,observation,allow_resize=True):
    a=np.asarray(current,np.uint8);h,w=a.shape[:2]
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError("expected RGB")
    started=time.perf_counter();best=None;trials=0
    factors=(1,.75,.5,.375,.25) if allow_resize else (1,)
    for factor in factors:
        cw,ch=max(1,round(w*factor)),max(1,round(h*factor))
        image=Image.fromarray(a).resize((cw,ch),Image.Resampling.LANCZOS)
        lo,hi=1,(2048 if codec=="jpeg2000" else 100 if codec=="webp" else 95)
        candidate=None
        while lo<=hi:
            q=(lo+hi)//2;payload=encode_payload(image,codec,q);trials+=1
            wire=make_packet(payload,codec=codec,scale_shape=(ch,cw),native_shape=(h,w),region=region,base_version=base_version,base_digest=base_digest,observation=observation)
            fits=len(wire)<=budget
            if fits:candidate=(wire,payload,q)
            if codec=="jpeg2000":
                if fits:hi=q-1
                else:lo=q+1
            else:
                if fits:lo=q+1
                else:hi=q-1
        if candidate is None:continue
        wire,payload,q=candidate
        with Image.open(io.BytesIO(payload)) as coded:
            rgb=np.array(coded.convert("RGB").resize((w,h),Image.Resampling.BICUBIC))
        mse=float(np.mean(((a.astype(np.float64)-rgb)/255)**2))
        if best is None or mse<best[0]:best=(mse,wire,q,ch,cw)
    if best is None:
        wire=make_packet(b"",codec=codec,scale_shape=(h,w),native_shape=(h,w),region=region,base_version=base_version,base_digest=base_digest,observation=observation)
        if len(wire)>budget:raise ValueError("even deferred header exceeds budget")
        best=(None,wire,0,h,w)
    mse,wire,q,ch,cw=best
    return wire,dict(bytes=len(wire),header_bytes=HEAD.size,checksum_bytes=16,quality=q,coded_height=ch,coded_width=cw,
                    codec_trials=trials,sender_pixel_mse=mse,sender_search_ms=(time.perf_counter()-started)*1000,
                    input="current RGB only; no semantic/validity labels",selection="lowest pixel MSE among tested resolutions and highest tested fitting qualities")
