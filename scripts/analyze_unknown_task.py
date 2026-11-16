"""Paired confusion-count bootstrap for the appended receiver semantic probe."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/unknown_recovery/semantic_probe_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(matrix, heldout):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    score = np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)
    score[..., heldout - 1] = np.nan
    return np.nanmean(score, axis=-1)


def main():
    complete = json.loads((OUT / "complete.json").read_text("utf-8"))
    if complete["status"] != "complete" or sha(OUT / "records.csv") != complete["records_sha256"]:
        raise ValueError("Incomplete semantic probe")
    grouped = {}
    for r in csv.DictReader((OUT / "records.csv").open(encoding="utf-8")):
        key = int(r["heldout"]), int(r["budget"]), r["method"], r["pair_id"]
        grouped.setdefault(key, []).append(np.asarray(json.loads(r["confusion"]), dtype=np.float64))
    averaged = {key: np.mean(value, axis=0) for key, value in grouped.items()}
    comparisons = []
    for heldout in (5, 3, 4):
        for budget in (4096, 8192, 16384, 32768):
            for method in ("knn_mix", "knn_band", "emergent_mix", "emergent_band", "maha_band", "band_random"):
                ids = sorted(k[3] for k in averaged if k[:3] == (heldout, budget, method))
                new = np.stack([averaged[(heldout, budget, method, p)] for p in ids])
                old = np.stack([averaged[(heldout, budget, "feature_change", p)] for p in ids])
                sample = np.random.default_rng(20261104).integers(0, len(ids), (2000, len(ids)))
                bootstrap = f1(new[sample].sum(1), heldout) - f1(old[sample].sum(1), heldout)
                low, high = map(float, np.quantile(bootstrap, [.00625, .99375]))
                comparisons.append(dict(heldout=heldout, budget=budget, method=method, comparator="feature_change", pairs=len(ids),
                                        delta_known_macro_f1=float(f1(new.sum(0), heldout) - f1(old.sum(0), heldout)),
                                        ci_low=low, ci_high=high, noninferiority_margin=-.02,
                                        interval_supports_margin=low > -.02))
    with (OUT / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (OUT / "comparison_receipt.json").write_text(json.dumps(dict(status="complete", source_sha256=sha(Path(__file__)),
        inputs=dict(records_sha256=complete["records_sha256"], completion_sha256=sha(OUT / "complete.json")),
        output_sha256=sha(OUT / "comparisons.csv"), comparisons=len(comparisons),
        scope="Appended development probe; .02 internal tolerance, 98.75% four-budget descriptive intervals per class/method"), indent=2) + "\n", "utf-8")
    print(json.dumps([r for r in comparisons if r["method"] == "emergent_mix"], indent=2), flush=True)


if __name__ == "__main__":
    main()
