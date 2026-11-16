"""Exploratory later-use task statistics from complete receiver cohorts."""
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
OUT = ROOT / "results/future_semantic_probe/v1/confirmation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_f1(matrix, heldout):
    index = heldout - 1
    numerator = 2 * matrix[..., index, index]
    denominator = matrix[..., index, :].sum(-1) + matrix[..., :, index].sum(-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def main():
    receipt = json.loads((OUT / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(OUT / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Later-use ground assay incomplete or changed")
    groups = defaultdict(list)
    records = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
    if len(records) != 46080:
        raise ValueError("Unexpected later-use record count")
    for row in records:
        key = int(row["heldout"]), int(row["budget"]), row["cohort"] + "_" + row["method"], row["pair_id"]
        groups[key].append(np.array(json.loads(row["confusion"]), float))
    per_pair = {key: np.mean(values, axis=0) for key, values in groups.items()}
    summaries, comparisons = [], []
    for heldout, budget, method in sorted({key[:3] for key in per_pair}):
        ids = sorted(key[3] for key in per_pair if key[:3] == (heldout, budget, method))
        matrix = np.sum([per_pair[(heldout, budget, method, p)] for p in ids], axis=0)
        index = heldout - 1
        summaries.append(dict(heldout=heldout, budget=budget, method=method, pairs=len(ids),
            later_class_f1=float(class_f1(matrix, heldout)), true_positive=float(matrix[index, index]),
            false_positive=float(matrix[:, index].sum() - matrix[index, index]),
            false_negative=float(matrix[index, :].sum() - matrix[index, index]),
            target_pixels=float(matrix[index, :].sum()), prediction_pixels=float(matrix[:, index].sum())))
    contrasts = (("image_emergent_mix", "image_feature_change", None),
                 ("hybrid_emergent_mix", "hybrid_change", None),
                 ("hybrid_emergent_mix", "hybrid_random", None),
                 ("hybrid_emergent_mix", "hybrid_PQ_feature_only", 0),
                 ("hybrid_emergent_mix", "image_full_webp", None),
                 ("hybrid_emergent_mix", "learned_hyperprior", None))
    for heldout in (5, 3, 4):
        for cap in (4096, 8192, 16384, 32768):
            for method, comparator, fixed_cap in contrasts:
                ids = sorted(k[3] for k in per_pair if k[:3] == (heldout, cap, method))
                if len(ids) != 256:
                    raise ValueError("Incomplete later-use contrast")
                old_cap = cap if fixed_cap is None else fixed_cap
                a = np.stack([per_pair[(heldout, cap, method, p)] for p in ids])
                b = np.stack([per_pair[(heldout, old_cap, comparator, p)] for p in ids])
                draws = np.random.default_rng(20261104).multinomial(256, np.full(256, 1/256), size=10000)
                a_boot = (draws @ a.reshape(256, 36)).reshape(-1, 6, 6)
                b_boot = (draws @ b.reshape(256, 36)).reshape(-1, 6, 6)
                difference = class_f1(a_boot, heldout) - class_f1(b_boot, heldout)
                low, high = np.nanquantile(difference, [.05 / 24, 1 - .05 / 24])
                comparisons.append(dict(heldout=heldout, budget=cap, method=method, comparator=comparator,
                    comparator_budget=old_cap, delta_later_class_f1=float(class_f1(a.sum(0), heldout) - class_f1(b.sum(0), heldout)),
                    interval_low=float(low), interval_high=float(high), interval_probability=1-.05/12,
                    scope="Additional exploratory endpoint; descriptive 12 class/cap intervals per contrast family"))
    for name, rows in (("summary", summaries), ("comparisons", comparisons)):
        with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(status="complete", source_sha256=sha(Path(__file__)), records_sha256=receipt["records_sha256"],
        fit_receipt_sha256=sha(ROOT / "results/future_semantic_probe/v1/fit_receipt.json"),
        summary_sha256=sha(OUT / "summary.csv"), comparisons_sha256=sha(OUT / "comparisons.csv"),
        interpretation="Ground evaluation uses all-class calibration prototypes including held-out classes. Online routing and known-only quantizers remain unchanged. The feature-only diagnostic measures the same later class head directly on received PQ features.")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps([r for r in comparisons if r["heldout"] == 5 and r["budget"] == 8192], indent=2), flush=True)


if __name__ == "__main__":
    main()
