"""Paired factorial analysis, with random repetitions averaged within source pair."""
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

GROUND = ROOT/"results/task_evidence/residual_ground_v1"
SOURCE = ROOT/"results/task_evidence/residual_development_v1"
OUT = ROOT/"results/task_evidence/residual_analysis_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main():
    receipt = json.loads((GROUND/"complete.json").read_text("utf-8"))
    if receipt["records"] != 3072 or sha(GROUND/"records.csv") != receipt["records_sha256"]:
        raise ValueError("Incomplete residual development matrix")
    spec = json.loads((SOURCE/"run_spec.json").read_text("utf-8"))
    ids = sorted(spec["pair_ids"])
    values = rows(GROUND/"records.csv")
    values += [r for r in rows(ROOT/"results/task_evidence/fresh_context_ground_v1/records.csv")
        if r["pair_id"] in set(ids) and r["method"] == "full_webp"]
    grouped = defaultdict(list)
    for row in values:
        grouped[int(row["heldout"]),int(row["budget"]),row["method"],row["pair_id"]].append(row)
    per_pair = {}
    for key,group in grouped.items():
        if len({r["receiver_task_sha256"] for r in group}) != 1:
            raise AssertionError("Task states differ among random replicates")
        per_pair[key] = dict(task=group[0]["receiver_task_sha256"],
            known=np.mean([np.array(json.loads(r["known_confusion"]),float) for r in group],axis=0),
            future=np.mean([np.array(json.loads(r["future_confusion"]),float) for r in group],axis=0),
            support=float(group[0]["unknown_pixels"]),retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in group]),
            bytes=np.mean([float(r["total_bytes"]) for r in group]))
    summaries = []
    for h,cap,method in sorted({k[:3] for k in per_pair}):
        group = [per_pair[h,cap,method,p] for p in ids]
        support = sum(r["support"] for r in group)
        future = later_counts(np.sum([r["future"] for r in group],axis=0),h)
        summaries.append(dict(heldout=h,budget=cap,method=method,pairs=32,
            unknown_pairs=sum(r["support"]>0 for r in group),unknown_pixels=support,
            unknown_coverage_30=scalar(safe_ratio(sum(r["retained"] for r in group),support)),
            later_class_f1=scalar(safe_ratio(future[0],future[1])),
            known_macro_f1=float(known_f1(np.sum([r["known"] for r in group],axis=0),h)),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
    draws = np.random.default_rng(20261106).multinomial(32,np.full(32,1/32),size=10000)
    contrasts = []
    for h in (5,3,4):
        for cap in (4096,8192,16384,32768):
            a = [per_pair[h,cap,"residual_current",p] for p in ids]
            for comparator in ("absolute_current","residual_change","residual_random","hyperprior","full_webp"):
                b = [per_pair[h,cap,comparator,p] for p in ids]
                if any(x["task"] != y["task"] or not np.array_equal(x["known"],y["known"]) for x,y in zip(a,b)):
                    raise AssertionError("Residual factorial changed the current-task service")
                support = np.array([r["support"] for r in a])
                delta = np.array([x["retained"]-y["retained"] for x,y in zip(a,b)])
                u = safe_ratio(draws@delta,draws@support)
                fa = np.stack([later_counts(r["future"],h) for r in a])
                fb = np.stack([later_counts(r["future"],h) for r in b])
                ba,bb = draws@fa,draws@fb
                f = safe_ratio(ba[:,0],ba[:,1])-safe_ratio(bb[:,0],bb[:,1])
                ul,uh = np.nanquantile(u,[.025,.975])
                fl,fh = np.nanquantile(f,[.025,.975])
                pa,pb = fa.sum(0),fb.sum(0)
                contrasts.append(dict(heldout=h,budget=cap,method="residual_current",comparator=comparator,
                    coverage_difference=scalar(safe_ratio(delta.sum(),support.sum())),coverage_low=scalar(ul),coverage_high=scalar(uh),
                    later_class_f1_difference=scalar(safe_ratio(pa[0],pa[1])-safe_ratio(pb[0],pb[1])),later_low=scalar(fl),later_high=scalar(fh),
                    valid_coverage_draws=int(np.isfinite(u).sum()),valid_later_draws=int(np.isfinite(f).sum()),task_states_exact=True,
                    mean_method_bytes=float(np.mean([r["bytes"] for r in a])),mean_baseline_bytes=float(np.mean([r["bytes"] for r in b]))))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,group in (("summary",summaries),("comparisons",contrasts)):
        with (OUT/(name+".csv")).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(group[0]))
            writer.writeheader()
            writer.writerows(group)
    primary = [r for r in contrasts if r["heldout"] == 5 and r["budget"] == 8192]
    result = dict(status="complete",pairs=32,records=3072,source_sha256=sha(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in (SOURCE/"run_spec.json",GROUND/"complete.json",GROUND/"records.csv")},
        primary_diagnostic=primary,statistics="10000 paired source-pair resamples; three random seeds averaged within pair; unadjusted95% development intervals",
        scope="Exploratory32 previously inspected image pairs, all3 classes and4 caps retained, no independent confirmation claimed",
        summary_sha256=sha(OUT/"summary.csv"),comparisons_sha256=sha(OUT/"comparisons.csv"))
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n","utf-8")
    print(json.dumps(primary,indent=2),flush=True)


if __name__ == "__main__":
    main()
