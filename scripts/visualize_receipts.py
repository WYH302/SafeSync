"""Display received image evidence from a fixed set of original pair IDs."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from safesync.data import load_manifest,read_pair
from safesync.image_wire import IMAGE_HEADER
from safesync.wire import HEADER,State,receive

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=load_manifest(ROOT)
    pairs=[x for x in manifest['pairs'] if x['split']=='development_a'][:3]+[x for x in manifest['pairs'] if x['split']=='development_b'][:2]
    out=Image.new('RGB',(180*5,220*5+45),'white')
    draw=ImageDraw.Draw(out)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for index,title in enumerate(['Current reference','GT held-out buildings','Unknown received ROI','Random received ROI','JPEG received image']):
        draw.text((index*180+3,12),title,fill='black',font=font)
    folder=ROOT/'results/repair1_jpeg80'
    for row,pair in enumerate(pairs):
        _,after,_,label,changed=read_pair(ROOT,pair)
        initial=(folder/'bitstreams/initial'/f"{pair['id']}.bin").read_bytes()
        identity=HEADER.unpack_from(initial)[-1]
        state=receive(State(np.zeros((256,384),np.int8),16,identity),initial)
        mask=Image.fromarray((((label==5)&changed)*255).astype(np.uint8)).convert('RGB')
        images=[after,mask]
        for method,seed in [('change_unknown',0),('change_random',1729)]:
            packet=(folder/'bitstreams/h5'/f"{pair['id']}_{method}_b65536_s{seed}.bin").read_bytes()
            received=receive(state,packet)
            canvas=Image.new('RGB',(512,512),(215,215,215))
            for index,content in received.rois.items():
                y,x=divmod(index,16)
                canvas.paste(Image.open(io.BytesIO(content)).convert('RGB'),(x*32,y*32))
            images.append(canvas)
        packet=(folder/'bitstreams/h5'/f"{pair['id']}_jpeg_budget_b65536_s0.bin").read_bytes()
        size=IMAGE_HEADER.unpack_from(packet)[3]
        images.append(Image.open(io.BytesIO(packet[IMAGE_HEADER.size:IMAGE_HEADER.size+size])).convert('RGB'))
        for col,image in enumerate(images):
            out.paste(image.resize((180,180),Image.Resampling.NEAREST if col==1 else Image.Resampling.BICUBIC),(col*180,45+row*220))
        draw.text((4,45+row*220+184),f"{pair['id']}  64 KiB each; JPEG80 ROI, 35% reserve",fill='black',font=font)
    out.save(ROOT/'results/analysis/received_evidence.png')


if __name__=='__main__':
    main()
