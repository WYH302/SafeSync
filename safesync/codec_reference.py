"""Strong compact-reference baseline: image codec plus exact residual norms.

Uses the triangle inequality, not learned uncertainty. The receiver generates
the compact image and upward-rounded error norms; the sender decodes that
image and compares its current observation. Ground generation time and sender
decode/compare time should be reported separately.
"""
import io,struct,hashlib
import numpy as np
from PIL import Image
from .reference import blocks,means,expand_means
from .pixel_wire import encode_image
from .certificates import upward_half

HEAD=struct.Struct("<4sBHHHII")


def highpass(a):
    return a-expand_means(means(a,4),a.shape[1])


def encode_codec_reference(rgb,budget,codec="jpeg2000",block=64):
    original=np.asarray(rgb,np.uint8);h,w=original.shape[:2];n=h*w//(block*block)
    available=budget-HEAD.size-16-2*n
    if available<128:raise ValueError("reference budget cannot hold format/norms")
    target=highpass(blocks(original,block).astype(np.float64)/255)
    best=None
    for scale in (1,2,4):
        small=Image.fromarray(original).resize((w//scale,h//scale),Image.Resampling.BOX)
        lo,hi=1,(95 if codec=="jpeg" else 2048)
        fitted=None
        while lo<=hi:
            q=(lo+hi)//2
            payload=encode_image(small,codec,q)
            fit=len(payload)<=available
            if fit:fitted=payload
            if codec=="jpeg":
                if fit:lo=q+1
                else:hi=q-1
            else:
                if fit:hi=q-1
                else:lo=q+1
        if fitted is None:continue
        with Image.open(io.BytesIO(fitted)) as im:
            proxy=np.array(im.convert("RGB").resize((w,h),Image.Resampling.BICUBIC))
        error=highpass(blocks(proxy,block).astype(np.float64)/255)-target
        rms=np.sqrt(np.mean(error**2,axis=(1,2,3)))
        distortion=float(np.mean(rms**2))
        if best is None or distortion<best[0]:best=(distortion,scale,fitted,upward_half(rms))
    if best is None:raise ValueError("no compact reference codec configuration fits")
    _,scale,payload,norms=best
    head=HEAD.pack(b"CR02",scale,block,h,w,len(payload),n)
    body=head+norms.tobytes()+payload
    wire=body+hashlib.sha256(body).digest()[:16]
    if len(wire)>budget:raise AssertionError("codec reference exceeds exact budget")
    return wire


def codec_reference_score(current,wire):
    if len(wire)<HEAD.size+16 or hashlib.sha256(wire[:-16]).digest()[:16]!=wire[-16:]:raise ValueError("codec reference checksum/length")
    magic,scale,b,h,w,size,n=HEAD.unpack(wire[:HEAD.size])
    if magic!=b"CR02" or len(wire)!=HEAD.size+2*n+size+16:raise ValueError("codec reference format")
    a=np.asarray(current,np.uint8)
    if a.shape!=(h,w,3):raise ValueError("codec reference dimensions")
    norms=np.frombuffer(wire[HEAD.size:HEAD.size+2*n],"<f2").astype(np.float64)
    with Image.open(io.BytesIO(wire[HEAD.size+2*n:-16])) as im:
        proxy=np.array(im.convert("RGB").resize((w,h),Image.Resampling.BICUBIC))
    x=blocks(a,b).astype(np.float64)/255
    y=blocks(proxy,b).astype(np.float64)/255
    delta=highpass(x)-highpass(y)
    point=np.mean(delta**2,axis=(1,2,3))
    c=means(x,4);coarse_error=np.mean((c-np.rint(c*255)/255)**2,axis=(1,2,3))
    upper=(np.sqrt((np.sqrt(point)+norms)**2+coarse_error)+0.5/255)**2
    return point+coarse_error,upper
