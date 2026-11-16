"""Locked source-pair statistics for the complete SECOND confirmation runs."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "results/unknown_recovery/locked_test_v1"
HYBRID = ROOT / "results/hybrid_pq/confirmation_v1"
LEARNED = ROOT / "results/second_learned/tasks_v1"
OUT = ROOT / "results/second_confirmation_analysis/v1"
CAPS = (4096, 8192, 16384, 32768)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(matrix, heldout):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    values = np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan, dtype=float), where=denominator > 0)
    values[..., heldout - 1] = np.nan
    return np.nanmean(values, axis=-1)


def load(folder, prefix, expected, matrix_key):
    receipt = json.loads((folder / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(folder / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Incomplete or changed confirmation records")
    rows = list(csv.DictReader((folder / "records.csv").open(encoding="utf-8")))
    if len(rows) != expected:
        raise ValueError("Unexpected confirmation record count")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["heldout"]), int(row["budget"]), prefix + row["method"], row["pair_id"])].append(row)
    result = {}
    for key, group in grouped.items():
        result[key] = dict(matrix=np.mean([np.array(json.loads(r[matrix_key]), float) for r in group], axis=0),
            retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in group]), support=float(group[0]["unknown_pixels"]),
            bytes=np.mean([float(r["total_bytes"]) for r in group]),
            known_error=np.mean([float(r["known_error_sum"]) for r in group]), known_support=float(group[0]["known_pixels"]))
    return result


def comparison(per_pair, heldout, budget, method, comparator, probability):
    ids = sorted(k[3] for k in per_pair if k[:3] == (heldout, budget, method))
    if len(ids) != 256:
        raise ValueError("A confirmation contrast is missing image pairs")
    a = [per_pair[(heldout, budget, method, p)] for p in ids]
    b = [per_pair[(heldout, budget, comparator, p)] for p in ids]
    if any(x["support"] != y["support"] or x["known_support"] != y["known_support"] for x, y in zip(a, b)):
        raise ValueError("Confirmation metric masks differ between methods")
    counts = np.array([r["support"] for r in a])
    active = counts > 0
    weights = counts[active]
    delta = np.array([x["retained"] - y["retained"] for x, y in zip(a, b)])[active]
    rng = np.random.default_rng(20261104)
    draw = rng.multinomial(len(weights), np.full(len(weights), 1 / len(weights)), size=10000)
    unknown_distribution = (draw @ delta) / (draw @ weights)
    matrix_a = np.stack([r["matrix"] for r in a])
    matrix_b = np.stack([r["matrix"] for r in b])
    identical = np.array_equal(matrix_a, matrix_b)
    if identical:
        known_distribution = np.zeros(10000)
    else:
        draws = np.random.default_rng(20261104).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=10000)
        boot_a = (draws @ matrix_a.reshape(len(ids), 36)).reshape(-1, 6, 6)
        boot_b = (draws @ matrix_b.reshape(len(ids), 36)).reshape(-1, 6, 6)
        known_distribution = macro_f1(boot_a, heldout) - macro_f1(boot_b, heldout)
    tail = (1 - probability) / 2
    u_low, u_high = np.quantile(unknown_distribution, [tail, 1 - tail])
    k_low, k_high = np.quantile(known_distribution, [tail, 1 - tail])
    return dict(heldout=heldout, budget=budget, method=method, comparator=comparator, pairs=len(ids), unknown_pairs=int(active.sum()),
        delta_unknown_coverage_30=float(delta.sum() / weights.sum()), unknown_low=float(u_low), unknown_high=float(u_high),
        delta_known_macro_f1=float(macro_f1(matrix_a.sum(0), heldout) - macro_f1(matrix_b.sum(0), heldout)),
        known_low=float(k_low), known_high=float(k_high), known_confusions_identical=identical,
        interval_probability=probability, supports_positive_coverage=bool(u_low > 0), supports_known_margin=bool(k_low > -.02),
        mean_new_bytes=float(np.mean([r["bytes"] for r in a])), mean_old_bytes=float(np.mean([r["bytes"] for r in b])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_pair = load(IMAGE, "image_", 29184, "confusion")
    per_pair.update(load(HYBRID, "hybrid_", 19200, "known_confusion"))
    input_files = [IMAGE / "complete.json", HYBRID / "complete.json", IMAGE / "records.csv", HYBRID / "records.csv"]
    if (LEARNED / "complete.json").exists():
        per_pair.update(load(LEARNED, "image_", 3072, "confusion"))
        input_files += [LEARNED / "complete.json", LEARNED / "records.csv"]
    summaries = []
    keys = sorted({k[:3] for k in per_pair})
    for heldout, budget, method in keys:
        group = [r for k, r in per_pair.items() if k[:3] == (heldout, budget, method)]
        if len(group) != 256:
            raise ValueError("Incomplete confirmation summary")
        matrix = np.sum([r["matrix"] for r in group], axis=0)
        summaries.append(dict(heldout=heldout, budget=budget, method=method, pairs=256,
            unknown_pairs=sum(r["support"] > 0 for r in group), known_macro_f1=float(macro_f1(matrix, heldout)),
            unknown_coverage_30=float(sum(r["retained"] for r in group) / sum(r["support"] for r in group)),
            known_mse=float(sum(r["known_error"] for r in group) / sum(r["known_support"] for r in group)),
            mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
    contrasts = []
    for heldout in (5, 3, 4):
        for cap in CAPS:
            for method, comparator in (("image_emergent_mix", "image_feature_change"),
                    ("hybrid_emergent_mix", "hybrid_change"), ("hybrid_emergent_mix", "hybrid_random"),
                    ("image_emergent_mix", "image_feature_random"), ("hybrid_emergent_mix", "image_full_jpeg420"),
                    ("hybrid_emergent_mix", "image_full_jpeg2000"), ("hybrid_emergent_mix", "image_full_webp")):
                contrasts.append(comparison(per_pair, heldout, cap, method, comparator, 1 - .05 / 12))
            if (heldout, cap, "image_hyperprior") in keys:
                contrasts.append(comparison(per_pair, heldout, cap, "hybrid_emergent_mix", "image_hyperprior", 1 - .05 / 12))
    primary = dict(
        original_image=comparison(per_pair, 5, 16384, "image_emergent_mix", "image_feature_change", .95),
        hybrid_extension=comparison(per_pair, 5, 8192, "hybrid_emergent_mix", "hybrid_change", .95),
        hybrid_two_protocol_sensitivity=comparison(per_pair, 5, 8192, "hybrid_emergent_mix", "hybrid_change", .975))
    for name, rows in (("summary", summaries), ("contrasts", contrasts)):
        with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(status="complete", source_sha256=sha(Path(__file__)), inputs={str(p.relative_to(ROOT)): sha(p) for p in input_files},
        original_method_lock_sha256=sha(ROOT / "results/unknown_recovery/confirmation_lock_v1.json"),
        hybrid_lock_sha256=sha(HYBRID / "method_lock.json"), primary=primary,
        statistics="10000 source-pair bootstrap resamples; random seeds averaged within source pair; coverage pools held-out pixels on explicitly labeled change support; known F1 pools confusion counts",
        interval_scope="Each primary uses its locked 95% interval. The hybrid secondary extension additionally reports 97.5% sensitivity for two protocol families. All class/cap contrasts have 99.583333% descriptive intervals; no correction across every method/comparator family.",
        access_disclosure=json.loads((HYBRID / "method_lock.json").read_text("utf-8"))["access_disclosure"],
        summary_sha256=sha(OUT / "summary.csv"), contrasts_sha256=sha(OUT / "contrasts.csv"))
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == "__main__":
    main()
