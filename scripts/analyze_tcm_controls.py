"""Complete-cohort comparisons for the additional fixed public TCM control."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.analyze_task_context_confirmation import later_counts, safe_ratio


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def complete(path, count, filename="records.csv"):
    receipt = json.loads((path / "complete.json").read_text("utf-8"))
    if receipt["records"] != count or receipt["records_sha256"] != sha(path / filename):
        raise ValueError("Incomplete or changed TCM comparator input")


def save(out, summary, comparisons, inputs, scope):
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summary), ("comparisons", comparisons)):
        with (out / (name+".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(status="complete", scope=scope, source_sha256=sha(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}, summary_sha256=sha(out / "summary.csv"),
        comparisons_sha256=sha(out / "comparisons.csv"), comparisons=comparisons)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", "utf-8")
    print(json.dumps(dict(scope=scope, primary=[r for r in comparisons if r.get("budget") == 8192 and r.get("heldout") == 5]
        if "SECOND" in scope else comparisons), indent=2), flush=True)


def second():
    base = ROOT / "results/task_evidence"
    source, control = base / "residual_confirmation_ground_v1", base / "tcm_ground_v1"
    complete(source, 29952)
    complete(control, 3072)
    complete(base / "tcm_control_v1", 3072, "records.json")
    rows = csv_rows(source / "records.csv")+csv_rows(control / "records.csv")
    methods = ("residual_current", "residual_change", "absolute_current", "hyperprior", "full_webp", "tcm")
    current = {(r["pair_id"],int(r["heldout"])):r["receiver_task_sha256"] for r in rows if r["method"] == "full_current"}
    selected = [r for r in rows if r["method"] in methods]
    values = {}
    for r in selected:
        h, cap = int(r["heldout"]), int(r["budget"])
        key = h,cap,r["method"],r["pair_id"]
        if key in values or r["receiver_task_sha256"] != current[r["pair_id"],h] or int(r["total_bytes"]) > cap:
            raise ValueError("TCM matrix duplicate, task identity or actual cap mismatch")
        values[key] = dict(support=float(r["unknown_pixels"]),retained=float(r["unknown_retained_pixels_30"]),
            future=later_counts(np.array(json.loads(r["future_confusion"]),float),h),bytes=int(r["total_bytes"]))
    ids = sorted({key[3] for key in values})
    if len(ids) != 256 or len(values) != 256*3*4*6:
        raise ValueError("Additional SECOND control cohort incomplete")
    draws = np.random.default_rng(20261108).multinomial(256,np.full(256,1/256),size=10000)
    summaries, contrasts = [], []
    for h in (5,3,4):
        for cap in (4096,8192,16384,32768):
            for method in methods:
                group = [values[h,cap,method,p] for p in ids]
                future = np.sum([r["future"] for r in group],axis=0)
                summaries.append(dict(heldout=h,budget=cap,method=method,pairs=256,
                    unknown_coverage_30=sum(r["retained"] for r in group)/sum(r["support"] for r in group),
                    later_class_f1=float(future[0]/future[1]),mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
            baseline = [values[h,cap,"tcm",p] for p in ids]
            for method in ("residual_current","residual_change","hyperprior","full_webp"):
                a = [values[h,cap,method,p] for p in ids]
                if any(x["support"] != y["support"] for x,y in zip(a,baseline)):
                    raise ValueError("Mismatched held-out support")
                support = np.array([r["support"] for r in a])
                delta = np.array([x["retained"]-y["retained"] for x,y in zip(a,baseline)])
                fa,fb = [np.stack([r["future"] for r in group]) for group in (a,baseline)]
                da,db = draws@fa,draws@fb
                ud = safe_ratio(draws@delta,draws@support)
                fd = safe_ratio(da[:,0],da[:,1])-safe_ratio(db[:,0],db[:,1])
                probability = .95 if h == 5 and cap == 8192 else 1-.05/12
                tail = (1-probability)/2
                ul,uh = np.nanquantile(ud,[tail,1-tail])
                fl,fh = np.nanquantile(fd,[tail,1-tail])
                contrasts.append(dict(heldout=h,budget=cap,method=method,comparator="tcm",probability=probability,
                    coverage_difference=float(delta.sum()/support.sum()),coverage_low=float(ul),coverage_high=float(uh),
                    later_f1_difference=float(fa.sum(0)[0]/fa.sum(0)[1]-fb.sum(0)[0]/fb.sum(0)[1]),later_low=float(fl),later_high=float(fh),
                    mean_method_bytes=float(np.mean([r["bytes"] for r in a])),mean_control_bytes=float(np.mean([r["bytes"] for r in baseline]))))
    inputs = [source / "records.csv",control / "records.csv",control / "complete.json",base / "tcm_control_v1/method_lock.json"]
    save(base / "tcm_analysis_v1",summaries,contrasts,inputs,"SECOND additional modern codec control, already acquired third cohort,10000 paired source-pair resamples, no new independent confirmation")


def sn7():
    ground = ROOT / "results/tcm_sn7/ground_v1"
    complete(ground,534)
    complete(ROOT / "results/tcm_sn7/transport_v1",534,"records.json")
    old = ROOT / "resources/pretrained_ground_crossfit/receipts_v1"
    sources = [old / f"area_{i}/records.csv" for i in range(8)]
    original = [r for path in sources for r in csv_rows(path)]
    controls = csv_rows(ground / "records.csv")
    reference = {(r["aoi"],r["month"],int(r["budget"])):r for r in original if r["method"] == "gaussian_20261029"}
    for row in controls:
        prior = reference[row["aoi"],row["month"],int(row["budget"])]
        fields = ("model_hash","threshold_hash","building_valid_pixels","building_positive_pixels","event_valid_pixels","event_positive_pixels")
        if any(row[field] != prior[field] for field in fields) or int(row["down_bytes"]) > int(row["budget"]) or int(row["ack_bytes"]) != 60:
            raise ValueError("Modern monthly comparator changed ground task, support, threshold or byte contract")
    methods = ("gaussian_20261029","dct_bound","full_cache_mse","full_cache_gate30","full_webp","learned_hyperprior_q1_8","tcm")
    groups = defaultdict(list)
    for row in original+controls:
        if row["method"] in methods:
            groups[row["method"],int(row["budget"]),row["aoi"]].append(row)
    values, areas = {}, sorted({r["aoi"] for r in controls})
    if len(areas) != 8:
        raise ValueError("Modern monthly comparison needs all eight AOIs")
    for key,group in groups.items():
        point = {}
        for task in ("building","event"):
            tp,fp,fn = (sum(int(r[task+"_"+name]) for r in group) for name in ("tp","fp","fn"))
            point[task+"_f1"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        point["mean_bytes"] = float(np.mean([int(r["down_bytes"])+int(r["ack_bytes"]) for r in group]))
        values[key] = point
    summaries,comparisons = [],[]
    draws = np.random.default_rng(20261108).integers(0,8,size=(10000,8))
    for cap in (32768,65536,131072):
        for method in methods:
            group = [values[method,cap,a] for a in areas]
            summaries.append(dict(method=method,budget=cap,AOIs=8,**{k:float(np.mean([r[k] for r in group])) for k in group[0]}))
        for method in ("gaussian_20261029","dct_bound","full_cache_mse","full_webp","learned_hyperprior_q1_8"):
            for task in ("building_f1","event_f1"):
                delta = np.array([values[method,cap,a][task]-values["tcm",cap,a][task] for a in areas])
                low,high = np.quantile(delta[draws].mean(1),[.05/6,1-.05/6])
                comparisons.append(dict(method=method,comparator="tcm",budget=cap,task=task,mean_difference=float(delta.mean()),
                    low=float(low),high=float(high),probability=1-.05/3,AOIs=8))
    inputs = sources+[ground / "records.csv",ground / "evaluation_lock.json",ROOT / "results/tcm_sn7/transport_v1/method_lock.json"]
    save(ROOT / "results/tcm_sn7/analysis_v1",summaries,comparisons,inputs,"SN7 additional modern comparator, same author-fold ground interface,10000 paired AOI resamples,three-cap-family intervals, post-hoc comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",choices=("second","sn7"))
    args = parser.parse_args()
    second() if args.study == "second" else sn7()
