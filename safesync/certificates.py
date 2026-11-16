"""Quantization-aware bounds on receiver-relative RGB mean squared error.

The Gaussian result is the standard chi-square norm estimate. Its scope is a
fixed current/reference pair independent of the fresh random matrix. It is a
pixel-MSE certificate, not a semantic or future-task guarantee. Reusing random
rows after sketch-dependent state selection would invalidate that premise.
"""
from __future__ import annotations
import hashlib
import struct
from functools import lru_cache
import numpy as np
from scipy.fft import dctn
from scipy.stats import chi2
from .reference import blocks, means, expand_means, dct_indices

HEAD=struct.Struct("<4sBBHHHIII")
Q=256.0


@lru_cache(maxsize=4)
def gaussian_rows(block,count,seed,channel):
    # Different independent rows for RGB; reuse across regions does not require
    # region independence for a per-decision bound, but seed must be fresh in time.
    rng=np.random.default_rng(np.random.SeedSequence([seed,channel]))
    result=rng.standard_normal((block*block,count),dtype=np.float32)/block
    result.setflags(write=False)
    return result


def gaussian_measure(x,count,seed):
    n,b,_,c=x.shape
    return np.stack([x[:,:,:,j].reshape(n,b*b)@gaussian_rows(b,count,seed,j) for j in range(c)],axis=-1)


@lru_cache(maxsize=2)
def joint_rows(block,count,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,7919]))
    result=rng.standard_normal((block*block*3,count),dtype=np.float32)/(block*np.sqrt(3))
    result.setflags(write=False)
    return result


def joint_measure(x,count,seed):
    return x.reshape(len(x),-1)@joint_rows(x.shape[1],count,seed)


def q16(a):
    q=np.rint(a*Q)
    if np.any(np.abs(q)>32767): raise ValueError("sketch overflow: transmit pixels instead")
    return q.astype("<i2")


def upward_half(a):
    q=np.asarray(a,dtype="<f2")
    return np.where(q.astype(np.float32)<a,np.nextafter(q,np.float16(np.inf)),q).astype("<f2")


def encode_certificate_reference(rgb,method="gaussian_hybrid",side=8,block=64,seed=0,detail_side=0):
    a=np.asarray(rgb,np.uint8)
    x=blocks(a,block).astype(np.float32)/255
    if detail_side not in (0,4):raise ValueError("detail side is fixed at four")
    if detail_side:x=x-expand_means(means(x,detail_side),block)
    count=side*side//2
    methods={"gaussian":0,"gaussian_hybrid":1,"dct_bound":2,"gaussian_joint":3,"hybrid_joint":4}
    if method not in methods or side not in (4,8,16):raise ValueError("unsupported certificate")
    if method in ("gaussian_hybrid","hybrid_joint"):
        coarse=means(x,side//2)
        detail=x-expand_means(coarse,block)
        count=(side*side-(side//2)**2)//2
        measure=joint_measure(detail,count*3,seed) if method=="hybrid_joint" else gaussian_measure(detail,count,seed)
        coarse_code=np.rint((coarse+1)*127.5) if detail_side else np.rint(coarse*255)
        payload=coarse_code.astype(np.uint8).tobytes()+q16(measure).tobytes()
    elif method in ("gaussian","gaussian_joint"):
        measure=joint_measure(x,count*3,seed) if method=="gaussian_joint" else gaussian_measure(x,count,seed)
        payload=q16(measure).tobytes()
    else:
        count-=1 # Three upward-rounded residual norms occupy the final 6 bytes.
        ij=dct_indices(block,count)
        coeff=dctn(x,axes=(1,2),norm="ortho")
        retained=coeff[:,ij[:,0],ij[:,1],:]
        coeff[:,ij[:,0],ij[:,1],:]=0
        tail_norm=np.sqrt(np.sum(coeff.astype(np.float64)**2,axis=(1,2)))
        payload=q16(retained).tobytes()+upward_half(tail_norm).tobytes()
    if len(payload)!=len(x)*side*side*3:raise AssertionError("wrong certificate budget")
    head=HEAD.pack(b"SC02",methods[method]+(8 if detail_side else 0),side,block,a.shape[0],a.shape[1],seed,count,len(payload))
    body=head+payload
    return body+hashlib.sha256(body).digest()[:16]


def certificate_score(current_rgb,wire,alpha=0.05):
    """Return (point MSE estimate, upper bound, metadata) in [0,1] RGB units.

    If upper <= tau, false acceptance of true MSE > tau has probability <=alpha
    over a fresh Gaussian sketch for any fixed pair (no distribution assumption).
    No such claim is made when the packet/observation depends on that seed.
    """
    if not 0<alpha<1 or len(wire)<HEAD.size+16:raise ValueError("bad alpha/reference")
    magic,method,side,b,h,w,seed,count,size=HEAD.unpack(wire[:HEAD.size])
    detail_side=4 if method>=8 else 0
    if detail_side:method-=8
    if magic!=b"SC02" or method not in (0,1,2,3,4) or len(wire)!=HEAD.size+size+16 or hashlib.sha256(wire[:-16]).digest()[:16]!=wire[-16:]:
        raise ValueError("certificate format/checksum")
    a=np.asarray(current_rgb,np.uint8)
    if a.shape!=(h,w,3):raise ValueError("reference dimensions differ")
    x=blocks(a,b).astype(np.float32)/255
    coarse_error=np.zeros(len(x),np.float64)
    if detail_side:
        m=means(x,detail_side)
        raw_mean=means(blocks(a,b).astype(np.float64),detail_side)
        coarse_error=np.mean(((raw_mean-np.rint(raw_mean))/255)**2,axis=(1,2,3))
        x=x-expand_means(m,b)
    def restore(point,upper,meta):
        if detail_side:
            point=point+coarse_error
            upper=(np.sqrt(upper+coarse_error)+0.5/255)**2
            meta=dict(meta,coarse_replacement_side=detail_side,includes_final_rounding=True)
        return point,upper,meta
    payload=wire[HEAD.size:-16]
    coarse_point=coarse_upper=np.zeros(len(x),np.float64)
    if method in (1,4):
        cside=side//2
        size_coarse=len(x)*cside*cside*3
        old=np.frombuffer(payload[:size_coarse],np.uint8).reshape(len(x),cside,cside,3).astype(np.float32)/255
        if detail_side:old=old*2-1
        now=means(x,cside)
        coarse_point=np.mean((now-old)**2,axis=(1,2,3))
        coarse_upper=(np.sqrt(coarse_point)+(1.0 if detail_side else 0.5)/255)**2
        x=x-expand_means(now,b)
        payload=payload[size_coarse:]
    if method in (3,4):
        rows=count*3
        old=np.frombuffer(payload,"<i2").reshape(len(x),rows).astype(np.float64)/Q
        delta=joint_measure(x,rows,seed).astype(np.float64)-old
        norm=np.sqrt(np.sum(delta**2,axis=1))
        point=coarse_point+np.mean(delta**2,axis=1)
        upper=coarse_upper+(norm+np.sqrt(rows)/(2*Q))**2/chi2.ppf(alpha,rows)
        return restore(point,upper,{"kind":"joint RGB probabilistic pixel-MSE", "alpha_per_decision":alpha,"fresh_seed_required":True,"rows":rows})
    if method in (0,1):
        old=np.frombuffer(payload,"<i2").reshape(len(x),count,3).astype(np.float32)/Q
        delta=gaussian_measure(x,count,seed).astype(np.float64)-old
        norm=np.sqrt(np.sum(delta**2,axis=1))
        # ||A d|| <= ||A current - Q(A prior)|| + sqrt(k)*qstep/2.
        qradius=np.sqrt(count)/(2*Q)
        lower_chi=chi2.ppf(alpha/3,count)
        detail_upper=np.mean((norm+qradius)**2/lower_chi,axis=-1)
        point=coarse_point+np.mean(delta**2,axis=(1,2))
        upper=coarse_upper+detail_upper
        return restore(point,upper,{"kind":"probabilistic pixel-MSE", "alpha_per_decision":alpha,"fresh_seed_required":True,"rows_per_channel":count})
    ij=dct_indices(b,count)
    coeff=dctn(x,axes=(1,2),norm="ortho")
    current=coeff[:,ij[:,0],ij[:,1],:]
    coeff[:,ij[:,0],ij[:,1],:]=0
    split=len(x)*count*3*2
    old=np.frombuffer(payload[:split],"<i2").reshape(len(x),count,3).astype(np.float64)/Q
    old_tail=np.frombuffer(payload[split:],"<f2").reshape(len(x),3).astype(np.float64)
    new_tail=np.sqrt(np.sum(coeff.astype(np.float64)**2,axis=(1,2)))
    delta_norm=np.sqrt(np.sum((current-old)**2,axis=1))
    low_upper=(delta_norm+np.sqrt(count)/(2*Q))**2
    upper=np.mean((low_upper+(new_tail+old_tail)**2)/(b*b),axis=1)
    point=np.mean(np.sum((current-old)**2,axis=1)/(b*b),axis=1)
    return restore(point,upper,{"kind":"deterministic DCT/tail norm bound", "alpha_per_decision":0,"fresh_seed_required":False,"retained_coefficients":count})
