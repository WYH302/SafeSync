"""Audit labels and create ten coordinate-aligned image-pair contact rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from safesync.data import CLASS_NAMES, PALETTE, block_fraction, load_manifest, masks, read_pair

ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'configs/mvp.json').read_text())
    manifest=load_manifest(ROOT)
    output=ROOT/'results/data_audit'
    output.mkdir(parents=True,exist_ok=True)
    heldouts=(5,3,4)
    counts={}
    findings=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    sheet=Image.new('RGB',(5*180,10*200+50),'white')
    draw=ImageDraw.Draw(sheet)
    for col,title in enumerate(['Before','Current','Semantic before','Semantic current','Binary change']):
        draw.text((col*180+6,12),title,fill='black',font=font)
    selected=manifest['pairs'][:5]+[x for x in manifest['pairs'] if x['split']=='development_a'][:3]+[x for x in manifest['pairs'] if x['split']=='development_b'][:2]
    for item in manifest['pairs']:
        before,after,old_label,label,changed=read_pair(ROOT,item)
        group=counts.setdefault(item['split'],{'pairs':0,'current_semantic_histogram':[0]*7,'prior_semantic_histogram':[0]*7,'binary_changed_pixels':0,'semantic_zero_pixels':0,'semantic_ignore_pixels':0,'known_fit_units':{str(x):0 for x in heldouts},'heldout_changed_pixels':{str(x):0 for x in heldouts}})
        group['pairs']+=1
        group['current_semantic_histogram']=(np.asarray(group['current_semantic_histogram'])+np.bincount(label[label<7].ravel(),minlength=7)).tolist()
        group['prior_semantic_histogram']=(np.asarray(group['prior_semantic_histogram'])+np.bincount(old_label[old_label<7].ravel(),minlength=7)).tolist()
        group['binary_changed_pixels']+=int(changed.sum())
        group['semantic_zero_pixels']+=int((label==0).sum())
        group['semantic_ignore_pixels']+=int((label==255).sum())
        annotated=(label>0)&(label<7)
        agree=float(np.mean(annotated==changed))
        if agree<.999:
            findings.append({'id':item['id'],'binary_vs_semantic_support_agreement':agree})
        for heldout in heldouts:
            m=masks(old_label,label,changed,heldout,config['grid'],config['known_purity'])
            group['known_fit_units'][str(heldout)]+=int(m['fit_current'].sum()+m['fit_before'].sum())
            group['heldout_changed_pixels'][str(heldout)]+=int(m['unknown_pixels'].sum())
        if item in selected:
            row=selected.index(item)
            images=[before,after,Image.fromarray(PALETTE[np.clip(old_label,0,6)]),Image.fromarray(PALETTE[np.clip(label,0,6)]),Image.fromarray((changed*255).astype(np.uint8)).convert('RGB')]
            for col,image in enumerate(images):
                sheet.paste(image.resize((180,180),Image.Resampling.NEAREST if col>=2 else Image.Resampling.BICUBIC),(col*180,50+row*200))
            draw.text((5,50+row*200+181),f"{item['id']} {item['split']}",fill='black',font=font)
    sheet.save(output/'ten_pairs.png')
    result={'class_mapping':CLASS_NAMES,'mapping_status':'native ChangeMamba grayscale IDs mapped through source GT_T2_COLORED pixels; source_color_check.json is authoritative verification','zero_rule':'non-change only; never fitted as a known semantic class','ignore_rule':'semantic 255 excluded; GT_CD 255 is binary change, not ignore','coordinate_rule':'512x512 native images, 16x16 grid, 32x32 native ROI; backbone resized separately','split_counts':counts,'binary_vs_semantic_support_mismatches':findings,'official_test_images_read':0,'geographic_independence':'not established: no geospatial grouping metadata in mirror','sample_rows':[x['id'] for x in selected]}
    (output/'label_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
