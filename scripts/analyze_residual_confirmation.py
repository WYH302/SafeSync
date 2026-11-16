"""Prelocked intersection test from the complete third-cohort receiver records."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_task_context_confirmation import known_f1,later_counts,safe_ratio,scalar

SOURCE = ROOT/"results/task_evidence/residual_confirmation_v1"
GROUND = ROOT/"results/task_evidence/residual_confirmation_ground_v1"
OUT = ROOT/"results/task_evidence/residual_confirmation_analysis_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = json.loads((SOURCE/"method_lock.json").read_text("utf-8"))
    receipt = json.loads((GROUND/"complete.json").read_text("utf-8"))
    if receipt["records"] != 29952 or receipt["records_sha256"] != sha(GROUND/"records.csv"):
        raise ValueError("Residual confirmation needs the full unchanged matrix")
    rows = list(csv.DictReader((GROUND/"records.csv").open(encoding="utf-8")))
    current = {(int(r["heldout"]),r["pair_id"]):r["receiver_task_sha256"] for r in rows if r["method"] == "full_current"}
    task_exact = len(current) == 768 and all(r["method"] == "old_only" or r["receiver_task_sha256"] == current[int(r["heldout"]),r["pair_id"]] for r in rows)
    if not task_exact:
        raise AssertionError("Locked residual methods changed the task service")
    grouped = defaultdict(list)
    for row in rows:
        if int(row["budget"]) and int(row["total_bytes"]) > int(row["budget"]):
            raise AssertionError("Actual residual transaction exceeded its cap")
        grouped[int(row["heldout"]),int(row["budget"]),row["method"],row["pair_id"]].append(row)
    values = {}
    for key,group in grouped.items():
        values[key] = dict(known=np.mean([np.array(json.loads(r["known_confusion"]),float) for r in group],axis=0),
            future=np.mean([np.array(json.loads(r["future_confusion"]),float) for r in group],axis=0),
            support=float(group[0]["unknown_pixels"]),retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in group]),
            bytes=np.mean([float(r["total_bytes"]) for r in group]))
    ids = sorted(lock["pair_ids"])
    summaries = []
    for h,cap,method in sorted({k[:3] for k in values}):
        group = [values[h,cap,method,p] for p in ids]
        future = later_counts(np.sum([r["future"] for r in group],axis=0),h)
        support = sum(r["support"] for r in group)
        summaries.append(dict(heldout=h,budget=cap,method=method,pairs=256,unknown_pairs=sum(r["support"]>0 for r in group),
            unknown_pixels=support,unknown_coverage_30=scalar(safe_ratio(sum(r["retained"] for r in group),support)),
            later_class_f1=scalar(safe_ratio(future[0],future[1])),known_macro_f1=float(known_f1(np.sum([r["known"] for r in group],axis=0),h)),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
    draws = np.random.default_rng(20261107).multinomial(256,np.full(256,1/256),size=10000)

    def compare(h,cap,comparator,probability):
        a = [values[h,cap,"residual_current",p] for p in ids]
        b = [values[h,cap,comparator,p] for p in ids]
        if any(x["support"] != y["support"] or not np.array_equal(x["known"],y["known"]) for x,y in zip(a,b)):
            raise AssertionError("Locked residual support or known task differs")
        support = np.array([r["support"] for r in a])
        delta = np.array([x["retained"]-y["retained"] for x,y in zip(a,b)])
        u = safe_ratio(draws@delta,draws@support)
        fa = np.stack([later_counts(r["future"],h) for r in a])
        fb = np.stack([later_counts(r["future"],h) for r in b])
        ba,bb = draws@fa,draws@fb
        f = safe_ratio(ba[:,0],ba[:,1])-safe_ratio(bb[:,0],bb[:,1])
        tail = (1-probability)/2
        ul,uh = np.nanquantile(u,[tail,1-tail])
        fl,fh = np.nanquantile(f,[tail,1-tail])
        pa,pb = fa.sum(0),fb.sum(0)
        baseline_u = sum(r["retained"] for r in b)/support.sum() if support.sum() else np.nan
        difference = safe_ratio(delta.sum(),support.sum())
        return dict(heldout=h,budget=cap,method="residual_current",comparator=comparator,probability=probability,
            coverage_difference=scalar(difference),coverage_low=scalar(ul),coverage_high=scalar(uh),
            baseline_coverage_30=scalar(baseline_u),method_coverage_30=scalar(baseline_u+difference),
            relative_miss_reduction=scalar(difference/(1-baseline_u)) if baseline_u < 1 else None,
            later_class_f1_difference=scalar(safe_ratio(pa[0],pa[1])-safe_ratio(pb[0],pb[1])),later_low=scalar(fl),later_high=scalar(fh),
            known_task_exact=True,mean_method_bytes=float(np.mean([r["bytes"] for r in a])),mean_baseline_bytes=float(np.mean([r["bytes"] for r in b])),
            valid_coverage_draws=int(np.isfinite(u).sum()),valid_later_draws=int(np.isfinite(f).sum()))

    primary = {name:compare(5,8192,name,.95) for name in lock["primary"]["coverage_comparators"]}
    conditions = dict(task_exact=task_exact)
    conditions.update({"coverage_above_"+name:r["coverage_low"] is not None and r["coverage_low"] > 0 for name,r in primary.items()})
    for name in lock["primary"]["later_comparators"]:
        conditions["later_noninferior_"+name] = primary[name]["later_low"] is not None and primary[name]["later_low"] > -.02
    secondary = [compare(h,cap,name,1-.05/12) for h in (5,3,4) for cap in (4096,8192,16384,32768)
        for name in ("absolute_current","residual_change","residual_random","hyperprior","full_webp")]
    OUT.mkdir(parents=True,exist_ok=True)
    for name,group in (("summary",summaries),("comparisons",secondary)):
        with (OUT/(name+".csv")).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(group[0]))
            writer.writeheader()
            writer.writerows(group)
    paths = (SOURCE/"method_lock.json",SOURCE/"complete.json",GROUND/"complete.json",GROUND/"records.csv",ROOT/"data/second_residual_test_v1/complete.json")
    result = dict(status="complete",pairs=256,records=29952,source_sha256=sha(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},primary=primary,primary_conditions=conditions,
        primary_joint_pass=all(conditions.values()),summary_sha256=sha(OUT/"summary.csv"),comparisons_sha256=sha(OUT/"comparisons.csv"),
        statistics="10000 paired source-pair resamples; random repetitions averaged within pair; pooled held-out quality coverage and pooled later confusion counts",
        scope="The third preaccess-locked cohort, pair-ID disjoint. Geographic independence unknown. Operational holdout excludes sender calibration classes, with a public pretrained backbone.")
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n","utf-8")
    print(json.dumps(dict(primary_conditions=conditions,primary=primary),indent=2),flush=True)


if __name__ == "__main__":
    main()
