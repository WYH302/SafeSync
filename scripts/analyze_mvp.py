"""Image-grouped aggregation, paired bootstrap, and plots from real run files."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VARIANTS={'png35':'results/mvp','jpeg80_35':'results/repair1_jpeg80','jpeg80_10':'results/repair2_reserve10'}
METHODS=['change_only','change_random','change_uniform','change_unknown','gt_diagnostic','jpeg_budget']
COUNT_FIELDS=['tp','fp','fn','tn','unknown_pixels','usable_unknown_feature_pixels','usable_unknown_roi_pixels','false_stable_changed_pixels','changed_pixels']


def aggregate(records):
    counts={key:sum(row[key] for row in records) for key in COUNT_FIELDS}
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    unknown=counts['unknown_pixels']
    return {**counts,'f1':2*tp/max(1,2*tp+fp+fn),'iou':tp/max(1,tp+fp+fn),'unknown_roi_coverage':counts['usable_unknown_roi_pixels']/unknown if unknown else None,'unknown_feature_coverage':counts['usable_unknown_feature_pixels']/unknown if unknown else None,'actual_bytes_mean':float(np.mean([x['actual_bytes'] for x in records])),'payload_bytes_mean':float(np.mean([x['actual_bytes']-x.get('bytes_padding',0) for x in records])),'n_pairs':len(records),'n_unknown_positive_pairs':sum(x['unknown_pixels']>0 for x in records),'false_stable_rate':counts['false_stable_changed_pixels']/max(1,counts['changed_pixels'])}


def collapse_random(rows):
    by_pair=defaultdict(list)
    for row in rows:
        by_pair[(row['heldout'],row['split'],row['method'],row['budget_bytes'],row['pair_id'])].append(row)
    reduced=[]
    for _,items in by_pair.items():
        combined=items[0].copy()
        for key in set(key for x in items for key,value in x.items() if isinstance(value,(int,float)) and not isinstance(value,bool)):
            combined[key]=float(np.mean([x.get(key,0) for x in items]))
        reduced.append(combined)
    return reduced


def paired_interval(method,baseline,repetitions=1000,seed=20261028):
    base={x['pair_id']:x for x in baseline}
    selected=[x for x in method if x['pair_id'] in base]
    rng=np.random.default_rng(seed)
    fields=['tp','fp','fn','unknown_pixels','usable_unknown_feature_pixels','usable_unknown_roi_pixels']
    left=np.array([[row[k] for k in fields] for row in selected],float)
    right=np.array([[base[row['pair_id']][k] for k in fields] for row in selected],float)
    indices=rng.integers(0,len(selected),size=(repetitions,len(selected)))
    a,b=left[indices].sum(axis=1),right[indices].sum(axis=1)
    valid=(a[:,3]>0)&(b[:,3]>0)
    intervals={
        'roi_gain':a[valid,5]/a[valid,3]-b[valid,5]/b[valid,3],
        'feature_gain':a[valid,4]/a[valid,3]-b[valid,4]/b[valid,3],
        'known_f1_gain':2*a[:,0]/np.maximum(1,2*a[:,0]+a[:,1]+a[:,2])-2*b[:,0]/np.maximum(1,2*b[:,0]+b[:,1]+b[:,2])
    }
    return {key:{'ci95':[float(x) for x in np.quantile(values,[.025,.975])],'valid_replicates':len(values)} if len(values) else {'ci95':None,'valid_replicates':0} for key,values in intervals.items()}


def write_csv(path,records):
    keys=list(dict.fromkeys(k for x in records for k in x))
    with path.open('w',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=keys)
        writer.writeheader();writer.writerows(records)


def plot_variant(name,summary,rows,output):
    colors={'change_only':'#666666','change_random':'#E69F00','change_uniform':'#CC79A7','change_unknown':'#0072B2','gt_diagnostic':'#999999','jpeg_budget':'#009E73'}
    labels={'change_only':'Change-only','change_random':'Change + random','change_uniform':'Change + uniform','change_unknown':'Change + Unknown','gt_diagnostic':'GT diagnostic','jpeg_budget':'JPEG, matched frame'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for method in METHODS:
        data=sorted([x for x in summary if x['variant']==name and x['heldout']==5 and x['split']=='combined' and x['method']==method],key=lambda x:x['budget_bytes'])
        if not data:
            continue
        x=[a['actual_bytes_mean']/1024 for a in data]
        style='--' if method=='gt_diagnostic' else '-'
        for ax,key,scale in [(axes[0,0],'f1',1),(axes[0,1],'unknown_roi_coverage',100),(axes[1,0],'unknown_feature_coverage',100)]:
            ax.plot(x,[a[key]*scale for a in data],style,marker='o',markersize=4,color=colors[method],label=labels[method],linewidth=1.6)
    axes[0,0].set_ylabel('Receiver known-task F1')
    axes[0,1].set_ylabel('Usable unknown pixel-ROI coverage (%)')
    axes[1,0].set_ylabel('Usable unknown feature coverage (%)')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.set_xlabel('Actual on-wire KiB / image')
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=7,loc='lower right')
    parts=['header','manifest','indices','token_payload','evidence_headers','evidence_payload','padding','checksum']
    part_colors=['#CCCCCC','#BBBBBB','#999999','#56B4E9','#CC79A7','#E69F00','#EEEEEE','#000000']
    budgets=[8192,16384,32768,65536]
    bottom=np.zeros(4)
    for key,color in zip(parts,part_colors):
        values=[]
        for budget in budgets:
            selected=[r for r in rows if r['heldout']==5 and r['method']=='change_unknown' and r['budget_bytes']==budget]
            values.append(np.mean([r.get(f'bytes_{key}',0) for r in selected])/1024)
        axes[1,1].bar(np.arange(4),values,bottom=bottom,color=color,label=key.replace('_',' '))
        bottom+=values
    axes[1,1].set_xticks(np.arange(4),['8','16','32','64'])
    axes[1,1].set_xlabel('Frame budget (KiB)')
    axes[1,1].set_ylabel('Measured wire components (KiB)')
    axes[1,1].legend(fontsize=6,ncol=2,loc='upper left')
    fig.suptitle(f'Development pilot: {name}, held-out buildings, 32 source pairs',fontsize=12)
    fig.savefig(output/f'tradeoffs_{name}.png',dpi=180)
    fig.savefig(output/f'tradeoffs_{name}.pdf')
    plt.close(fig)


def main():
    output=ROOT/'results/analysis'
    output.mkdir(parents=True,exist_ok=True)
    summaries,comparisons=[],[]
    all_rows={}
    for name,folder in VARIANTS.items():
        source=json.loads((ROOT/folder/'rows.json').read_text())
        rows=collapse_random(source)
        all_rows[name]=rows
        for heldout in (5,3,4):
            for split in ('development_a','development_b','combined'):
                grouped=defaultdict(list)
                for row in rows:
                    if row['heldout']==heldout and (split=='combined' or row['split']==split):
                        grouped[(row['method'],row['budget_bytes'])].append(row)
                for (method,budget),items in grouped.items():
                    summaries.append({'variant':name,'heldout':heldout,'split':split,'method':method,'budget_bytes':budget,**aggregate(items)})
                for budget in (8192,16384,32768,65536):
                    ours=grouped[('change_unknown',budget)]
                    for baseline in ('change_only','change_random','change_uniform','jpeg_budget'):
                        other=grouped.get((baseline,budget))
                        if not other:
                            continue
                        a,b=aggregate(ours),aggregate(other)
                        gain=a['unknown_roi_coverage']-b['unknown_roi_coverage']
                        f1=a['f1']-b['f1']
                        interval=paired_interval(ours,other)
                        comparisons.append({'variant':name,'heldout':heldout,'split':split,'budget_bytes':budget,'baseline':baseline,'roi_gain_absolute':gain,'roi_miss_relative_reduction':gain/(1-b['unknown_roi_coverage']) if b['unknown_roi_coverage']<1 else None,'known_f1_gain':f1,'known_noninferior_at_002':f1>=-.02,'known_within_absolute_002':abs(f1)<=.02,'same_actual_frame_budget':a['actual_bytes_mean']==b['actual_bytes_mean']==budget,'n_pairs':len(ours),'n_unknown_positive_pairs':a['n_unknown_positive_pairs'],'bootstrap':interval})
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False),encoding='utf-8')
    (output/'comparisons.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(output/'summary.csv',summaries)
    flattened=[{**{k:v for k,v in x.items() if k!='bootstrap'},**{f'{k}_ci95':str(v['ci95']) for k,v in x['bootstrap'].items()}} for x in comparisons]
    write_csv(output/'comparisons.csv',flattened)
    for name in VARIANTS:
        plot_variant(name,summaries,all_rows[name],output)
    print(json.dumps({'summaries':len(summaries),'comparisons':len(comparisons),'bootstrap_unit':'original image pair; random seeds averaged within pair','output':str(output)},indent=2))


if __name__=='__main__':
    main()
