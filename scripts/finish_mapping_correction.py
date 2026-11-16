"""Run the intended tree class with verified ID3 and preserve ID4 diagnostics."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from safesync.data import CLASS_NAMES
from safesync.wire import HEADER,State,encode,receive

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=[('configs/mvp.json','results/mvp','png35'),('configs/repair1_jpeg80.json','results/repair1_jpeg80','jpeg80_35'),('configs/repair2_reserve10.json','results/repair2_reserve10','jpeg80_10')]


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    for config_path,folder,name in VARIANTS:
        original=ROOT/config_path
        config=json.loads(original.read_text())
        config['secondary_heldout']=3
        config['secondary_semantic_name']='tree'
        config['supplemental_heldout']=4
        config['mapping_correction']='Verified against native GT_T2_COLORED and maintainer inference source; ID3 is tree, ID4 water. See docs/DATA_MAPPING_CORRECTION.md.'
        dump(original,config)
        temp_config={**config,'result_directory':f'results/mapping_tree_{name}'}
        temp_path=ROOT/f'configs/mapping_tree_{name}.json'
        dump(temp_path,temp_config)
        print('tree ID3',name,flush=True)
        with (ROOT/f'results/mapping_tree_{name}_run.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.run_mvp','--config',str(temp_path.relative_to(ROOT)),'--heldouts','3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        target,source=ROOT/folder,ROOT/temp_config['result_directory']
        rows=[x for x in json.loads((target/'rows.json').read_text()) if x['heldout']!=3]
        # Bring older native-q1 infeasible controls under the same fixed-frame rule.
        for row in rows:
            if row['method']!='jpeg_budget_infeasible':
                continue
            initial=(target/'bitstreams/initial'/f"{row['pair_id']}.bin").read_bytes()
            identity=HEADER.unpack_from(initial)[-1]
            state=receive(State(np.zeros((256,384),np.int8),16,identity),initial)
            packet,counts=encode(state,state.codes,[],{},np.zeros(256,bool),row['budget_bytes'])
            row.update({'method':'jpeg_budget','actual_bytes':len(packet),'two_date_total_bytes':row['initialization_bytes']+len(packet),'deferred_token_count':256,'stale_token_count':0,'jpeg_feasible':False,'jpeg_selected_quality':None})
            row.update({f'bytes_{k}':v for k,v in counts.items()})
            path=target/'bitstreams'/f"h{row['heldout']}"/f"{row['pair_id']}_jpeg_budget_b{row['budget_bytes']}_s0.bin"
            path.write_bytes(packet)
        rows+=json.loads((source/'rows.json').read_text())
        dump(target/'rows.json',rows)
        fields=list(dict.fromkeys(k for x in rows for k in x))
        with (target/'per_pair_metrics.csv').open('w',encoding='utf-8',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for file in ('calibration.json','unknown_diagnostic.json'):
            combined=[x for x in json.loads((target/file).read_text()) if x['heldout']!=3]+json.loads((source/file).read_text())
            for x in combined:
                if 'class' in x:
                    x['class']=CLASS_NAMES[x['heldout']]
            dump(target/file,combined)
        shutil.copy2(source/'calibration_3.npz',target/'calibration_3.npz')
        shutil.copytree(source/'bitstreams/h3',target/'bitstreams/h3',dirs_exist_ok=True)
        dump(target/'final_protocol.json',{**config,'executed_heldouts':[5,3,4],'original_frozen_config_preserved':'frozen_config.json','tree_correction_run':str(source.relative_to(ROOT))})
        print('merged',name,len(rows),flush=True)


if __name__=='__main__':
    main()
