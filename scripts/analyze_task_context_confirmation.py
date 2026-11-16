"""Joint prelocked inference from complete fresh receiver/ground records."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/task_evidence/fresh_context_v1"
GROUND = ROOT / "results/task_evidence/fresh_context_ground_v1"
ADDITIONAL = ROOT / "results/task_evidence/fresh_current_distance_v1"
OUT = ROOT / "results/task_evidence/fresh_analysis_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def known_f1(matrix, heldout):
    denominator = matrix.sum(-1)+matrix.sum(-2)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    score = np.divide(2*tp, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)
    score[..., heldout-1] = np.nan
    return np.nanmean(score, axis=-1)


def later_counts(matrix, heldout):
    index = heldout-1
    tp = matrix[index, index]
    return np.array([2*tp, matrix[index, :].sum()+matrix[:, index].sum()], float)


def safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(np.asarray(numerator, float), np.nan), where=denominator > 0)


def scalar(value):
    return float(value) if np.isfinite(value) else None


def load(folder, expected):
    receipt = json.loads((folder / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(folder / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Fresh joint inference requires complete unchanged records")
    rows = list(csv.DictReader((folder / "records.csv").open(encoding="utf-8")))
    if len(rows) != expected:
        raise ValueError("Fresh matrix size differs from the fixed experiment")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    original_rows = load(GROUND, 48384)
    additional_rows = load(ADDITIONAL, 6144)
    rows = original_rows + additional_rows
    current_hashes = {(int(r["heldout"]), r["pair_id"]): r["receiver_task_sha256"] for r in original_rows if r["method"] == "full_current"}
    task_exact = all(r["method"] == "old_only" or r["receiver_task_sha256"] == current_hashes[(int(r["heldout"]), r["pair_id"])] for r in rows)
    if not task_exact or len(current_hashes) != 768:
        raise AssertionError("A current-task service differs from the counted full-current control")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    per_pair = {}
    for key, group in grouped.items():
        per_pair[key] = dict(known=np.mean([np.array(json.loads(r["known_confusion"]), float) for r in group], axis=0),
            future=np.mean([np.array(json.loads(r["future_confusion"]), float) for r in group], axis=0),
            support=float(group[0]["unknown_pixels"]), retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in group]),
            retained25=np.mean([float(r["unknown_retained_pixels_25"]) for r in group]),
            retained35=np.mean([float(r["unknown_retained_pixels_35"]) for r in group]),
            bytes=np.mean([float(r["total_bytes"]) for r in group]))
    summaries = []
    for heldout, cap, method in sorted({k[:3] for k in per_pair}):
        group = [v for k, v in per_pair.items() if k[:3] == (heldout, cap, method)]
        if len(group) != 256:
            raise ValueError("A fresh method/fold/cap is missing source pairs")
        support = sum(r["support"] for r in group)
        future = later_counts(np.sum([r["future"] for r in group], axis=0), heldout)
        summaries.append(dict(heldout=heldout, budget=cap, method=method, pairs=256,
            unknown_pairs=sum(r["support"] > 0 for r in group), unknown_pixels=support,
            unknown_coverage_25=scalar(sum(r["retained25"] for r in group)/support) if support else None,
            unknown_coverage_30=scalar(sum(r["retained"] for r in group)/support) if support else None,
            unknown_coverage_35=scalar(sum(r["retained35"] for r in group)/support) if support else None,
            known_macro_f1=float(known_f1(np.sum([r["known"] for r in group], axis=0), heldout)),
            later_class_f1=scalar(safe_ratio(future[0], future[1])),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
    pair_draws = np.random.default_rng(20261104).multinomial(256, np.full(256, 1/256), size=10000)
    unknown_draws = {}

    def compare(heldout, cap, method, comparator, probability):
        ids = sorted(k[3] for k in per_pair if k[:3] == (heldout, cap, method))
        a = [per_pair[(heldout, cap, method, p)] for p in ids]
        b = [per_pair[(heldout, cap, comparator, p)] for p in ids]
        if len(ids) != 256 or any(x["support"] != y["support"] for x, y in zip(a, b)):
            raise ValueError("Fresh paired contrast has different support")
        known_equal = all(np.array_equal(x["known"], y["known"]) for x, y in zip(a, b))
        weights = np.array([r["support"] for r in a])
        active = weights > 0
        delta = np.array([x["retained"]-y["retained"] for x, y in zip(a, b)])[active]
        old_retained = sum(r["retained"] for r in b)
        old_u = old_retained/weights.sum() if weights.sum() else np.nan
        low = high = change = np.nan
        tail = (1-probability)/2
        if active.any():
            n = int(active.sum())
            if n not in unknown_draws:
                unknown_draws[n] = np.random.default_rng(20261104).multinomial(n, np.full(n, 1/n), size=10000)
            draws = unknown_draws[n]
            distribution = (draws@delta)/(draws@weights[active])
            low, high = np.quantile(distribution, [tail, 1-tail])
            change = delta.sum()/weights.sum()
        future_a = np.stack([later_counts(r["future"], heldout) for r in a])
        future_b = np.stack([later_counts(r["future"], heldout) for r in b])
        ba, bb = pair_draws@future_a, pair_draws@future_b
        future_distribution = safe_ratio(ba[:, 0], ba[:, 1])-safe_ratio(bb[:, 0], bb[:, 1])
        valid = np.isfinite(future_distribution)
        fl, fu = np.quantile(future_distribution[valid], [tail, 1-tail]) if valid.any() else (np.nan, np.nan)
        fa, fb = future_a.sum(0), future_b.sum(0)
        future_delta = safe_ratio(fa[0], fa[1])-safe_ratio(fb[0], fb[1])
        return dict(heldout=heldout, budget=cap, method=method, comparator=comparator, probability=probability,
            pairs=256, unknown_pairs=int(active.sum()), coverage_difference=scalar(change), coverage_low=scalar(low), coverage_high=scalar(high),
            baseline_coverage_30=scalar(old_u), method_coverage_30=scalar(old_u+change),
            relative_miss_reduction=scalar(change/(1-old_u)) if old_u < 1 else None,
            known_confusions_identical=known_equal,
            known_macro_f1_difference=float(known_f1(np.sum([r["known"] for r in a], axis=0), heldout)-known_f1(np.sum([r["known"] for r in b], axis=0), heldout)),
            later_class_f1_difference=scalar(future_delta), later_low=scalar(fl), later_high=scalar(fu), valid_future_bootstraps=int(valid.sum()),
            mean_method_bytes=float(np.mean([r["bytes"] for r in a])), mean_baseline_bytes=float(np.mean([r["bytes"] for r in b])))

    primary = {comparator: compare(5, 8192, "adaptive_mix", comparator, .95) for comparator in lock["primary"]["coverage_comparators"]}
    future_gate = primary["full_webp"]
    primary_conditions = dict(known_task_exact=task_exact,
        coverage_above_change=primary["adaptive_change"]["coverage_low"] is not None and primary["adaptive_change"]["coverage_low"] > 0,
        coverage_above_webp=primary["full_webp"]["coverage_low"] is not None and primary["full_webp"]["coverage_low"] > 0,
        coverage_above_hyperprior=primary["hyperprior"]["coverage_low"] is not None and primary["hyperprior"]["coverage_low"] > 0,
        later_use_noninferior=future_gate["later_low"] is not None and future_gate["later_low"] > lock["primary"]["later_class_f1_margin"])
    contrasts = []
    for h in (5, 3, 4):
        for cap in (4096, 8192, 16384, 32768):
            for comparator in ("adaptive_change", "adaptive_random", "full_webp", "full_jpeg420", "full_jpeg2000", "hyperprior", "roi_mix", "context_mix", "adaptive_current_distance"):
                contrasts.append(compare(h, cap, "adaptive_mix", comparator, 1-.05/12))
    for name, values in (("summary", summaries), ("comparisons", contrasts)):
        with (OUT/(name+".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    paths = [MAIN/"method_lock.json", MAIN/"complete.json", GROUND/"complete.json", GROUND/"records.csv",
        ADDITIONAL/"complete.json", ADDITIONAL/"records.csv", ROOT/"data/second_task_context_test_v1/complete.json"]
    result = dict(status="complete", source_sha256=sha(Path(__file__)), records=54528, pairs=256,
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}, primary=primary,
        primary_conditions=primary_conditions, primary_joint_pass=all(primary_conditions.values()),
        summary_sha256=sha(OUT/"summary.csv"), comparisons_sha256=sha(OUT/"comparisons.csv"),
        statistics="10000 paired source-image resamples; random seeds averaged within source pair. Unknown coverage pools labeled held-out pixels. Later F1 pools confusion counts, including explicitly labeled negative support.",
        inference_scope="The primary is the intersection of all locked conditions, with two-sided95% intervals per condition. Secondary12class/cap intervals per contrast are descriptive. Geographic independence of SECOND pairs is unavailable.",
        current_distance_scope="Additional locked ablation after the primary lock, before its own outcomes; does not replace the locked primary method or tests.")
    (OUT/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", "utf-8")
    print(json.dumps(dict(primary_conditions=primary_conditions, primary=primary), indent=2), flush=True)


if __name__ == "__main__":
    main()
