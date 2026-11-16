"""Summarize the complete factorial repair, retaining every budget and fold."""
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
OUT = ROOT / "results/task_evidence/development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(matrix, heldout):
    d = matrix.sum(-1) + matrix.sum(-2)
    t = np.diagonal(matrix, axis1=-2, axis2=-1)
    v = np.divide(2*t, d, out=np.full_like(t, np.nan), where=d > 0)
    v[..., heldout - 1] = np.nan
    return np.nanmean(v, axis=-1)


def main():
    receipt = json.loads((OUT / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(OUT / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Incomplete or changed factorial repair")
    grouped = defaultdict(list)
    records = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
    if len(records) != 18816:
        raise ValueError("Unexpected factorial size")
    for row in records:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    per_pair = {}
    for key, values in grouped.items():
        per_pair[key] = dict(matrix=np.mean([np.array(json.loads(r["known_confusion"]), float) for r in values], axis=0),
            unknown=float(values[0]["unknown_pixels"]), retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in values]),
            bytes=np.mean([float(r["total_bytes"]) for r in values]),
            label_bytes=np.mean([json.loads(r["components"])["labels"] for r in values]),
            image_bytes=np.mean([json.loads(r["components"])["image"] for r in values]))
    summaries, comparisons = [], []
    for key in sorted({k[:3] for k in per_pair}):
        h, cap, method = key
        group = [r for k, r in per_pair.items() if k[:3] == key]
        summaries.append(dict(heldout=h, budget=cap, method=method, pairs=len(group),
            known_macro_f1=float(f1(np.sum([r["matrix"] for r in group], axis=0), h)),
            unknown_coverage_30=float(sum(r["retained"] for r in group) / sum(r["unknown"] for r in group)),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group])),
            mean_service_payload_bytes=float(np.mean([r["label_bytes"] for r in group])),
            mean_image_payload_bytes=float(np.mean([r["image_bytes"] for r in group]))))
    contrasts = [("labels_webp_mix", b) for b in ("labels_webp_change", "labels_webp_random", "labels_full_webp", "labels_full_jpeg420", "labels_jpeg_mix", "pq_webp_mix", "pq_jpeg_mix")]
    contrasts += [("labels_jpeg_mix", "pq_jpeg_mix"), ("pq_webp_mix", "pq_jpeg_mix"), ("pq_webp_mix", "pq_webp_change")]
    for h in (5, 3, 4):
        for cap in (4096, 8192, 16384, 32768):
            for method, comparator in contrasts:
                ids = sorted(k[3] for k in per_pair if k[:3] == (h, cap, method))
                a = [per_pair[(h, cap, method, p)] for p in ids]
                b = [per_pair[(h, cap, comparator, p)] for p in ids]
                support = np.array([r["unknown"] for r in a])
                active = support > 0
                weights = support[active]
                delta = np.array([x["retained"] - y["retained"] for x, y in zip(a, b)])[active]
                draws = np.random.default_rng(20261104).multinomial(len(weights), np.full(len(weights), 1/len(weights)), size=10000)
                low, high = np.quantile((draws @ delta) / (draws @ weights), [.00625, .99375])
                ma = np.stack([r["matrix"] for r in a])
                mb = np.stack([r["matrix"] for r in b])
                identical = np.array_equal(ma, mb)
                if identical:
                    kl, ku = 0., 0.
                else:
                    resample = np.random.default_rng(20261104).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=10000)
                    boot = f1((resample @ ma.reshape(len(ids), 36)).reshape(-1, 6, 6), h) - f1((resample @ mb.reshape(len(ids), 36)).reshape(-1, 6, 6), h)
                    kl, ku = np.quantile(boot, [.00625, .99375])
                comparisons.append(dict(heldout=h, budget=cap, method=method, comparator=comparator,
                    delta_unknown_coverage_30=float(delta.sum()/weights.sum()), unknown_low=float(low), unknown_high=float(high),
                    delta_known_macro_f1=float(f1(ma.sum(0), h)-f1(mb.sum(0), h)), known_low=float(kl), known_high=float(ku),
                    known_confusions_identical=identical, interval_probability=.9875,
                    delta_mean_bytes=float(np.mean([x["bytes"]-y["bytes"] for x, y in zip(a, b)]))))
    for name, values in (("summary", summaries), ("comparisons", comparisons)):
        with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = dict(status="complete", source_sha256=sha(Path(__file__)), records_sha256=receipt["records_sha256"],
        summary_sha256=sha(OUT / "summary.csv"), comparisons_sha256=sha(OUT / "comparisons.csv"),
        scope="64 original development pairs. Four-cap 98.75% descriptive intervals per class/contrast; factorial attribution and method selection only.")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps([r for r in comparisons if r["method"] == "labels_webp_mix" and r["comparator"] == "labels_full_webp"], indent=2), flush=True)


if __name__ == "__main__":
    main()
