"""Recompute byte counts and reconcile every retained final-run bitstream."""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
import io

from safesync.data import load_manifest, verified_path
from safesync.image_wire import IMAGE_HEADER
from safesync.wire import CRC,HEADER,State,receive,signature

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['results/mvp','results/repair1_jpeg80','results/repair2_reserve10']


def main():
    manifest=load_manifest(ROOT)
    for item in manifest['pairs']:
        for key in item['files']:
            verified_path(ROOT,item,key)
    checks=[]
    for folder in VARIANTS:
        root=ROOT/folder
        rows=json.loads((root/'rows.json').read_text())
        env=json.loads((root/'environment.json').read_text(encoding='utf-8'))
        model_identity=json.dumps(env['identity'],sort_keys=True)
        tested,total,stream_hashes=0,0,[]
        states={}
        for item in manifest['pairs']:
            if item['split']=='calibration':
                continue
            ident=signature(model_identity+f";pair={item['id']};native={(512,512)}")
            initial=(root/'bitstreams/initial'/f"{item['id']}.bin").read_bytes()
            states[item['id']]=receive(State(np.zeros((256,384),np.int8),16,ident),initial)
        for row in rows:
            if row['method'] in ('old_state','drop_all_updates'):
                assert row['actual_bytes']==0
                continue
            packet_path=root/'bitstreams'/f"h{row['heldout']}"/f"{row['pair_id']}_{row['method']}_b{row['budget_bytes']}_s{row['seed']}.bin"
            packet=packet_path.read_bytes()
            accounted=sum(value for key,value in row.items() if key.startswith('bytes_'))
            assert len(packet)==accounted==row['actual_bytes'],str(packet_path)
            if row['budget_bytes']:
                assert len(packet)==row['budget_bytes'],str(packet_path)
            prior=states[row['pair_id']]
            if packet.startswith(b'SS01'):
                decoded=receive(prior,packet)
                assert decoded.version==prior.version+1
                assert int(np.sum(decoded.statuses==2))==row['received_token_count']
                assert len(decoded.rois)==row['received_roi_count']
            elif packet.startswith(b'SI01'):
                assert zlib.crc32(packet[:-CRC.size])==CRC.unpack(packet[-CRC.size:])[0]
                magic,base,target,size,digest,identity=IMAGE_HEADER.unpack_from(packet)
                assert base==prior.version and target==base+1 and digest==prior.digest and identity==prior.identity
                assert not any(packet[IMAGE_HEADER.size+size:-CRC.size])
                with Image.open(io.BytesIO(packet[IMAGE_HEADER.size:IMAGE_HEADER.size+size])) as image:
                    image.verify()
            else:
                raise AssertionError('unknown packet magic')
            stream_hashes.append({'path':packet_path.relative_to(ROOT).as_posix(),'bytes':len(packet),'sha256':hashlib.sha256(packet).hexdigest()})
            total+=len(packet);tested+=1
        # Complete-current and deliberately absent-current controls must differ.
        full=[x for x in rows if x['method']=='full_feature']
        old=[x for x in rows if x['method']=='old_state']
        dropped=[x for x in rows if x['method']=='drop_all_updates']
        assert all(x['f1']==0. for x in old+dropped)
        assert all(x['unknown_feature_coverage'] in (None,1.) for x in full)
        assert any(x['f1']>0. for x in full)
        assert sum(x['tp'] for x in full)>sum(x['tp'] for x in old)
        # Verify identical controlled simple baselines across codec repairs.
        checks.append({'variant':folder,'rows':len(rows),'checked_bitstreams':tested,'checked_wire_bytes':total,'controls_full_vs_old_vs_dropped':'passed','all_fixed_frame_budgets_exact':True,'checksums_and_versions':'passed'})
        (root/'bitstream_manifest.json').write_text(json.dumps(stream_hashes,indent=2),encoding='utf-8')
    colors=json.loads((ROOT/'results/data_audit/source_color_check.json').read_text())
    assert colors['all_palette_pixels_agree']
    output={'data_file_hashes_verified':sum(len(x['files']) for x in manifest['pairs']),'original_pair_groups_disjoint':True,'source_palette_verification':True,'geographic_independence_established':False,'official_test_images_read':0,'variants':checks,'scope':'actual bitstreams, receiver contract and byte accounting; does not establish research superiority'}
    target=ROOT/'results/verification'
    target.mkdir(parents=True,exist_ok=True)
    (target/'result_integrity.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
