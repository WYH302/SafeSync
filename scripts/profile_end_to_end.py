"""Fresh image-to-wire CPU/GPU timing, excluding frozen feature-cache reuse."""
from __future__ import annotations

import time
PROCESS_ENTRY=time.perf_counter()
import json
from pathlib import Path

import numpy as np
from PIL import Image

from safesync.features import FrozenEncoder
from safesync.data import load_manifest
from safesync.image_wire import encode_image_at_budget
from safesync.routing import encode_rois,select_and_encode
from safesync.scoring import KnownDistribution,change_score,safety_gate
from safesync.wire import State,HEADER,receive

ROOT=Path(__file__).resolve().parents[1]


def main():
    configs=[json.loads((ROOT/path).read_text()) for path in ('configs/mvp.json','configs/repair1_jpeg80.json','configs/repair2_reserve10.json')]
    encoder=FrozenEncoder(ROOT,configs[0])
    startup=time.perf_counter()-PROCESS_ENTRY
    manifest=load_manifest(ROOT)
    pairs=[x for x in manifest['pairs'] if x['split']=='development_a'][:10]
    rows=[]
    for config in configs:
        folder=ROOT/config.get('result_directory','results/mvp')
        calibration=json.loads((folder/'calibration.json').read_text())[0]
        with np.load(folder/'calibration_5.npz') as saved:
            distribution=KnownDistribution(saved['mean'],saved['variance'],saved['calibration_scores'])
        for pair in pairs:
            initial=(folder/'bitstreams/initial'/f"{pair['id']}.bin").read_bytes()
            identity=HEADER.unpack_from(initial)[-1]
            previous=receive(State(np.zeros((256,384),np.int8),16,identity),initial)
            start=time.perf_counter()
            image=Image.open(ROOT/pair['files']['after']['path']).convert('RGB')
            opened=time.perf_counter()
            features,timings=encoder.extract(image)
            extracted=time.perf_counter()
            change=change_score(features,previous.features)
            unknown=distribution.score(features)
            stable=safety_gate(change,unknown,calibration['silence_change_limit'],calibration['silence_unknown_limit'])
            scored=time.perf_counter()
            candidates=encode_rois(image,16,config['roi_codec'],config.get('roi_quality',80))
            cropped=time.perf_counter()
            packet,counts=select_and_encode(previous,features,change,unknown,candidates,stable,32768,config['evidence_budget_fraction'])
            ended=time.perf_counter()
            rows.append({'variant':config['protocol_version'],'pair_id':pair['id'],'input_io_ms':(opened-start)*1000,**timings,'risk_scoring_ms':(scored-extracted)*1000,'all_candidate_roi_encoding_ms':(cropped-scored)*1000,'route_and_serialize_ms':(ended-cropped)*1000,'image_to_wire_ms':(ended-start)*1000,'wire_bytes':len(packet),'scope':'warm batch1, fresh current backbone; historical state already in memory; all candidate ROI encoding included; no link time'})
            # Same native image and same budget for the simple image-codec route.
            start=time.perf_counter()
            image=Image.open(ROOT/pair['files']['after']['path']).convert('RGB')
            jpeg=encode_image_at_budget(previous,image,32768)
            ended=time.perf_counter()
            rows.append({'variant':config['protocol_version']+'_jpeg_budget','pair_id':pair['id'],'image_to_wire_ms':(ended-start)*1000,'wire_bytes':len(jpeg[0]) if jpeg else 0,'scope':'sender encoding search only; same native image; ground backbone cost reported in run CSV'})
    output={'startup_seconds_to_warm_encoder_ready':startup,'rows':rows,'scope':'micro-profile, 10 development images, no power or satellite throughput claim','hardware':encoder.metadata}
    (ROOT/'results/cost_profile.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'startup_seconds':startup,'rows':len(rows),'median_image_to_wire_ms':float(np.median([x['image_to_wire_ms'] for x in rows if not x['variant'].endswith('_jpeg_budget')]))},indent=2))


if __name__=='__main__':
    main()
