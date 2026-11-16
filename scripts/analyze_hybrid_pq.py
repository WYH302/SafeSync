"""Paired source-image analysis of the completed hybrid development experiment."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hybrid_pq/development_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((OUT / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(OUT / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Hybrid development records incomplete or changed")
    grouped = defaultdict(list)
    for row in csv.DictReader((OUT / "records.csv").open(encoding="utf-8")):
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    per_pair = {}
    for key, rows in grouped.items():
        per_pair[key] = dict(matrix=np.mean([np.array(json.loads(r["known_confusion"]), float) for r in rows], axis=0),
            retained=float(np.mean([float(r["unknown_retained_pixels_30"]) for r in rows])),
            support=float(rows[0]["unknown_pixels"]),
            bytes=float(np.mean([float(r["total_bytes"]) for r in rows])))
    summaries, comparisons = [], []
    for heldout in (5, 3, 4):
        for budget in (4096, 8192, 16384, 32768):
            for method in ("change", "emergent_mix", "emergent_only", "random"):
                ids = sorted(key[3] for key in per_pair if key[:3] == (heldout, budget, method))
                group = [per_pair[(heldout, budget, method, p)] for p in ids]
                support = np.array([r["support"] for r in group])
                retained = np.array([r["retained"] for r in group])
                matrix = np.sum([r["matrix"] for r in group], axis=0)
                denominator = matrix.sum(0) + matrix.sum(1)
                labels = np.array([i for i in range(6) if i != heldout - 1 and denominator[i] > 0])
                summaries.append(dict(heldout=heldout, budget=budget, method=method, pairs=len(ids),
                    unknown_pairs=int((support > 0).sum()), unknown_coverage_30=float(retained.sum() / support.sum()),
                    known_macro_f1=float(np.mean(2 * np.diag(matrix)[labels] / denominator[labels])),
                    mean_total_bytes=float(np.mean([r["bytes"] for r in group]))))
                if method in ("change", "random"):
                    continue
                for comparator in ("change", "random"):
                    old = [per_pair[(heldout, budget, comparator, p)] for p in ids]
                    if any(not np.array_equal(a["matrix"], b["matrix"]) for a, b in zip(group, old)):
                        raise AssertionError("Hybrid known service differs between routers")
                    active = support > 0
                    weights = support[active]
                    difference = (retained - np.array([r["retained"] for r in old]))[active]
                    draws = np.random.default_rng(20261104).integers(0, len(weights), (10000, len(weights)))
                    bootstrap = difference[draws].sum(1) / weights[draws].sum(1)
                    low, high = np.quantile(bootstrap, [.00625, .99375])
                    comparisons.append(dict(heldout=heldout, budget=budget, method=method, comparator=comparator,
                        delta_unknown_coverage_30=float(difference.sum() / weights.sum()),
                        interval_low=float(low), interval_high=float(high), interval_probability=.9875,
                        known_confusions_identical=True, delta_known_macro_f1=0.,
                        delta_mean_bytes=float(np.mean([a["bytes"] - b["bytes"] for a, b in zip(group, old)]))))
    for name, rows in (("summary", summaries), ("comparisons", comparisons)):
        with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "analysis.json").write_text(json.dumps(dict(status="complete", source_sha256=sha(Path(__file__)),
        records_sha256=receipt["records_sha256"], summary_sha256=sha(OUT / "summary.csv"),
        comparisons_sha256=sha(OUT / "comparisons.csv"),
        scope="64 previously inspected development pairs; 98.75% descriptive intervals within each four-budget class/method/comparator family",
        codebook_scope="393216 raw bytes per globally shared known-only codebook, recorded separately from updates",
        results=comparisons), indent=2) + "\n", "utf-8")
    print(json.dumps([r for r in comparisons if r["comparator"] == "change"], indent=2), flush=True)


if __name__ == "__main__":
    main()
