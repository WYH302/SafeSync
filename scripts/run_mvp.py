"""Run the predeclared receiver-only SECOND development pilot."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from safesync.data import CLASS_NAMES, block_fraction, load_manifest, masks, read_pair
from safesync.features import FrozenEncoder
from safesync.image_wire import encode_image, encode_image_at_budget, receive_image
from safesync.metrics import conditional_unknown_diagnostic, evaluate
from safesync.routing import encode_rois, select_and_encode
from safesync.scoring import KnownDistribution, change_score, choose_change_threshold, safety_gate
from safesync.wire import State, encode, quantize, receive, signature

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def uniform_priority(grid):
    coords = np.indices((grid,grid)).reshape(2,-1).T.astype(float)
    distances = np.sum((coords - (grid-1)/2)**2, axis=1)
    order, chosen = [], int(np.argmin(distances))
    minimum = np.full(len(coords), np.inf)
    for _ in range(len(coords)):
        order.append(chosen)
        minimum = np.minimum(minimum, np.sum((coords-coords[chosen])**2,axis=1))
        minimum[order] = -1
        chosen = int(np.argmax(minimum))
    priorities = np.zeros(len(coords))
    priorities[order] = np.arange(len(coords),0,-1)
    return priorities


def prepare(root, config, encoder, manifest):
    records=[]
    for index,item in enumerate(manifest["pairs"]):
        before,after,old_label,label,changed=read_pair(root,item)
        old_features,old_time=encoder.cached(before)
        current_features,current_time=encoder.cached(after)
        records.append({"item":item,"before":before,"after":after,"old_label":old_label,"label":label,"changed":changed,"old_features":old_features,"features":current_features,"before_timing":old_time,"current_timing":current_time})
        print(f"features {index+1}/{len(manifest['pairs'])} {item['id']} {current_features.shape}",flush=True)
    return records


def calibration(records,config,heldout):
    fitting,changes,truth,valid=[],[],[],[]
    for rec in records:
        if rec["item"]["split"]!="calibration":
            continue
        m=masks(rec['old_label'],rec['label'],rec['changed'],heldout,config['grid'],config['known_purity'])
        fitting.extend([rec['features'][m['fit_current']],rec['old_features'][m['fit_before']]])
        initial=State(quantize(rec['old_features']),config['grid'],signature("calibration"))
        changes.append(change_score(rec['features'],initial.features))
        truth.append(m['changed_units'])
        valid.append(m['known_eval_units'])
    values=np.concatenate(fitting)
    distribution=KnownDistribution.fit(values,config['variance_floor'])
    threshold=choose_change_threshold(np.concatenate(changes),np.concatenate(truth),np.concatenate(valid))
    return distribution,threshold,len(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/mvp.json')
    parser.add_argument('--heldouts',type=int,nargs='+',help='explicit semantic IDs; no label-dependent online routing')
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text(encoding='utf-8'))
    result_dir=ROOT/config.get('result_directory','results/mvp')
    result_dir.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(ROOT)
    dump(result_dir/'frozen_config.json',config)
    encoder=FrozenEncoder(ROOT,config)
    dump(result_dir/'environment.json',{**encoder.metadata,'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((ROOT/'data/second_pilot/manifest.json').read_bytes()).hexdigest(),'state_registry':'identity also binds original pair ID and native 512x512 coordinates'})
    records=prepare(ROOT,config,encoder,manifest)
    identity=signature(encoder.identity)
    rows,diagnostics,calibrations=[],[],[]
    spatial_priority=uniform_priority(config['grid'])
    for heldout in (args.heldouts or (config['primary_heldout'],config['secondary_heldout'])):
        distribution,threshold,fit_count=calibration(records,config,heldout)
        calibrations.append({'heldout':heldout,'class':CLASS_NAMES[heldout],'known_fit_tokens':fit_count,'change_threshold':threshold,'silence_change_limit':threshold/2,'silence_unknown_limit':distribution.known_quantile(.95),'known_supervision':'only explicitly annotated non-heldout calibration tokens; pooled Gaussian has no per-class targets; change threshold uses calibration known binary labels'})
        np.savez_compressed(result_dir/f'calibration_{heldout}.npz',mean=distribution.mean,variance=distribution.variance,calibration_scores=distribution.calibration_scores)
        diag_records={}
        for rec in records:
            item=rec['item']
            if item['split']=='calibration':
                continue
            m=masks(rec['old_label'],rec['label'],rec['changed'],heldout,config['grid'],config['known_purity'])
            region_identity=signature(encoder.identity+f";pair={item['id']};native={rec['after'].size}")
            state=State(quantize(rec['old_features']),config['grid'],region_identity)
            blank=State(np.zeros_like(state.codes),config['grid'],state.identity)
            initial_packet,initial_counts=encode(blank,state.codes,list(range(len(state.codes))),{},np.zeros(len(state.codes),bool),anchor=True)
            state=receive(blank,initial_packet)
            init_path=result_dir/'bitstreams'/'initial'/f"{item['id']}.bin"
            init_path.parent.mkdir(parents=True,exist_ok=True)
            init_path.write_bytes(initial_packet)
            score_start=time.perf_counter()
            change=change_score(rec['features'],state.features)
            unknown=distribution.score(rec['features'])
            stable=safety_gate(change,unknown,threshold/2,distribution.known_quantile(.95))
            score_ms=(time.perf_counter()-score_start)*1000
            start=time.perf_counter()
            roi_candidates=encode_rois(rec['after'],config['grid'],config['roi_codec'],config.get('roi_quality',80))
            candidate_roi_ms=(time.perf_counter()-start)*1000
            diag_records.setdefault(item['split'],[]).append((change,unknown,m['diagnostic_unknown'],m['diagnostic_known']))
            # Only this explicitly marked offline strategy can see ground truth.
            gt_priority=block_fraction(m['unknown_pixels'],config['grid'])
            methods=[('change_only',None,0),('change_uniform',spatial_priority,0),('change_unknown',unknown,0),('gt_diagnostic',gt_priority,0)]
            for seed in config['random_seeds']:
                rng=np.random.default_rng(int(hashlib.sha256(f"{seed}:{item['id']}".encode()).hexdigest()[:16],16))
                methods.append(('change_random',rng.random(len(change)),seed))
            # Initialization is paid once, equally for every temporal policy.
            init_bytes=len(initial_packet)
            def record(method,seed,budget,packet,counts,received,encode_ms,decode_ms,decoded_image=None):
                begin=time.perf_counter()
                metrics=evaluate(received,state,rec['features'],rec['after'],rec['changed'],m,threshold,config['minimum_token_cosine'],config['minimum_roi_psnr_db'],decoded_image)
                row={'heldout':heldout,'pair_id':item['id'],'split':item['split'],'method':method,'seed':seed,'budget_bytes':budget,'actual_bytes':len(packet) if packet is not None else 0,'initialization_bytes':init_bytes,'two_date_total_bytes':init_bytes+(len(packet) if packet is not None else 0),'pixels':rec['changed'].size,'score_ms':score_ms,'all_roi_candidates_encode_ms':candidate_roi_ms,'route_serialize_ms':encode_ms,'decode_ms':decode_ms,'offline_evaluate_ms':(time.perf_counter()-begin)*1000,'current_backbone_ms':rec['current_timing']['backbone_ms'],**{f'bytes_{k}':v for k,v in counts.items()},**metrics}
                rows.append(row)
                if packet is not None:
                    path=result_dir/'bitstreams'/f"h{heldout}"/f"{item['id']}_{method}_b{budget}_s{seed}.bin"
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(packet)
                    if len(packet)!=sum(counts.values()):
                        raise AssertionError('wire accounting mismatch')
            for budget in config['budgets_bytes']:
                for method,priority,seed in methods:
                    start=time.perf_counter()
                    packet,counts=select_and_encode(state,rec['features'],change,priority,roi_candidates,stable,budget,config['evidence_budget_fraction'])
                    encoded=time.perf_counter()
                    received=receive(state,packet)
                    decoded=time.perf_counter()
                    record(method,seed,budget,packet,counts,received,(encoded-start)*1000,(decoded-encoded)*1000)
            start=time.perf_counter()
            full_packet,full_counts=encode(state,quantize(rec['features']),list(range(len(change))),{},np.zeros(len(change),bool))
            encoded=time.perf_counter()
            full_received=receive(state,full_packet)
            decoded=time.perf_counter()
            record('full_feature',0,0,full_packet,full_counts,full_received,(encoded-start)*1000,(decoded-encoded)*1000)
            for control in ('old_state','drop_all_updates'):
                record(control,0,0,None,{},receive(state,None),0.,0.)
            for quality in (10,30,60,90):
                start=time.perf_counter()
                packet,counts=encode_image(state,rec['after'],quality)
                encoded=time.perf_counter()
                received,decoded_image,_=receive_image(state,packet,encoder)
                decoded=time.perf_counter()
                record(f'jpeg_q{quality}',0,0,packet,counts,received,(encoded-start)*1000,(decoded-encoded)*1000,decoded_image)
            for budget in config['budgets_bytes']:
                start=time.perf_counter()
                candidate=encode_image_at_budget(state,rec['after'],budget)
                encoded=time.perf_counter()
                if candidate is None:
                    packet,counts=encode(state,state.codes,[],{},np.zeros(len(change),bool),budget)
                    record('jpeg_budget',0,budget,packet,counts,receive(state,packet),(encoded-start)*1000,0.)
                    rows[-1]['jpeg_feasible']=False
                    rows[-1]['jpeg_selected_quality']=None
                    continue
                packet,counts,quality=candidate
                received,decoded_image,_=receive_image(state,packet,encoder)
                decoded=time.perf_counter()
                record('jpeg_budget',0,budget,packet,counts,received,(encoded-start)*1000,(decoded-encoded)*1000,decoded_image)
                rows[-1]['jpeg_selected_quality']=quality
                rows[-1]['jpeg_feasible']=True
            print(f"evaluated heldout={heldout} {item['split']} {item['id']}",flush=True)
            dump(result_dir/'rows.partial.json',rows)
        for split,items in diag_records.items():
            arrays=[np.concatenate([x[index] for x in items]) for index in range(4)]
            diagnostics.append({'heldout':heldout,'split':split,**conditional_unknown_diagnostic(*arrays)})
    dump(result_dir/'rows.json',rows)
    dump(result_dir/'calibration.json',calibrations)
    dump(result_dir/'unknown_diagnostic.json',diagnostics)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (result_dir/'per_pair_metrics.csv').open('w',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    dump(result_dir/'feature_timings.json',[{'pair_id':x['item']['id'],'split':x['item']['split'],'before':x['before_timing'],'current':x['current_timing']} for x in records])
    print(json.dumps({'rows':len(rows),'diagnostics':diagnostics,'result_dir':str(result_dir)},indent=2))


if __name__=='__main__':
    main()
