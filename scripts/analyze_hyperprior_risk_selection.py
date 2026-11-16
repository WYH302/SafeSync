"""Exploratory paired diagnostics for score-only learned-codec selection."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_task_context_confirmation import known_f1,later_counts,safe_ratio,scalar

GROUND = ROOT/"results/task_evidence/hyperprior_risk_ground_v1"
SOURCE = ROOT/"results/task_evidence/hyperprior_risk_development_v1"
OUT = ROOT/"results/task_evidence/hyperprior_risk_analysis_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main():
    receipt = json.loads((GROUND/"complete.json").read_text("utf-8"))
    if receipt["records"] != 3840 or receipt["records_sha256"] != sha(GROUND/"records.csv"):
        raise ValueError("Incomplete or changed exploratory score intervention")
    spec = json.loads((SOURCE/"run_spec.json").read_text("utf-8"))
    ids = sorted(spec["pair_ids"])
    rows = read_rows(GROUND/"records.csv")
    original = read_rows(ROOT/"results/task_evidence/fresh_context_ground_v1/records.csv")
    rows += [r for r in original if r["pair_id"] in set(ids) and r["method"] == "full_webp"]
    values = {}
    for row in rows:
        key = int(row["heldout"]),int(row["budget"]),row["method"],row["pair_id"]
        if key in values:
            raise ValueError("Duplicate exploratory source-pair measurement")
        values[key] = dict(known=np.array(json.loads(row["known_confusion"]),float),
            future=np.array(json.loads(row["future_confusion"]),float), support=float(row["unknown_pixels"]),
            retained=float(row["unknown_retained_pixels_30"]), bytes=float(row["total_bytes"]),task=row["receiver_task_sha256"])
    summary = []
    for h,cap,method in sorted({k[:3] for k in values}):
        group = [values[h,cap,method,p] for p in ids]
        support = sum(r["support"] for r in group)
        f = later_counts(np.sum([r["future"] for r in group],axis=0),h)
        summary.append(dict(heldout=h,budget=cap,method=method,pairs=64,
            unknown_coverage_30=scalar(sum(r["retained"] for r in group)/support) if support else None,
            later_class_f1=scalar(safe_ratio(f[0],f[1])),
            known_macro_f1=float(known_f1(np.sum([r["known"] for r in group],axis=0),h)),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
    draws = np.random.default_rng(20261105).multinomial(64,np.full(64,1/64),size=10000)
    contrasts = []
    for h in (5,3,4):
        for cap in (4096,8192,16384,32768):
            for method in ("current_distance_quality","temporal_quality"):
                a = [values[h,cap,method,p] for p in ids]
                for comparator in ("mse","uniform_quality","change_quality","full_webp"):
                    b = [values[h,cap,comparator,p] for p in ids]
                    if any(x["task"] != y["task"] or not np.array_equal(x["known"],y["known"]) for x,y in zip(a,b)):
                        raise AssertionError("Score-only intervention changed the shared known task")
                    support = np.array([r["support"] for r in a])
                    delta = np.array([x["retained"]-y["retained"] for x,y in zip(a,b)])
                    sampled_support = draws@support
                    u = safe_ratio(draws@delta,sampled_support)
                    fa = np.stack([later_counts(r["future"],h) for r in a])
                    fb = np.stack([later_counts(r["future"],h) for r in b])
                    ba,bb = draws@fa,draws@fb
                    f = safe_ratio(ba[:,0],ba[:,1])-safe_ratio(bb[:,0],bb[:,1])
                    ul,uh = np.nanquantile(u,[.025,.975])
                    fl,fh = np.nanquantile(f,[.025,.975])
                    pa,pb = fa.sum(0),fb.sum(0)
                    contrasts.append(dict(heldout=h,budget=cap,method=method,comparator=comparator,
                        coverage_difference=scalar(safe_ratio(delta.sum(),support.sum())),coverage_low=scalar(ul),coverage_high=scalar(uh),
                        later_class_f1_difference=scalar(safe_ratio(pa[0],pa[1])-safe_ratio(pb[0],pb[1])),later_low=scalar(fl),later_high=scalar(fh),
                        task_states_exact=True,mean_method_bytes=float(np.mean([r["bytes"] for r in a])),
                        mean_baseline_bytes=float(np.mean([r["bytes"] for r in b]))))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,group in (("summary",summary),("comparisons",contrasts)):
        with (OUT/(name+".csv")).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(group[0]))
            writer.writeheader()
            writer.writerows(group)
    primary = [r for r in contrasts if r["heldout"] == 5 and r["budget"] == 8192]
    result = dict(status="complete",pairs=64,records=3840,source_sha256=sha(Path(__file__)),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in (SOURCE/"run_spec.json",GROUND/"complete.json",GROUND/"records.csv")},
        primary_diagnostic=primary,statistics="10000 paired source-pair resamples, descriptive unadjusted95% intervals",
        scope="Exploratory reuse of previously inspected source data. This cannot reverse the completed fresh-context primary failure.",
        summary_sha256=sha(OUT/"summary.csv"),comparisons_sha256=sha(OUT/"comparisons.csv"))
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n","utf-8")
    print(json.dumps(primary,indent=2),flush=True)


if __name__ == "__main__":
    main()
