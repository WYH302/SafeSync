"""Source-pair statistics for the two-action development selector."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/task_evidence/selector_development_v1"
BASE = ROOT / "results/task_evidence/development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((OUT / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(OUT / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Incomplete selector")
    records = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
    if len(records) != 3840:
        raise ValueError("Incomplete selector matrix")
    records += [r for r in csv.DictReader((BASE / "records.csv").open(encoding="utf-8")) if r["service"] == "labels"]
    grouped = defaultdict(list)
    for row in records:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    per_pair = {}
    for key, rows in grouped.items():
        per_pair[key] = dict(unknown=float(rows[0]["unknown_pixels"]), retained=float(np.mean([float(r["unknown_retained_pixels_30"]) for r in rows])),
            bytes=float(np.mean([float(r["total_bytes"]) for r in rows])),
            full_fraction=float(np.mean([r.get("chosen_method") == "labels_full_webp" for r in rows])),
            confusion=json.loads(rows[0]["known_confusion"]))
    summaries, comparisons = [], []
    for h, b, method in sorted({k[:3] for k in per_pair}):
        values = [r for k, r in per_pair.items() if k[:3] == (h, b, method)]
        summaries.append(dict(heldout=h, budget=b, method=method, pairs=len(values),
            unknown_coverage_30=sum(r["retained"] for r in values)/sum(r["unknown"] for r in values),
            mean_total_bytes=float(np.mean([r["bytes"] for r in values])),
            full_image_fraction=float(np.mean([r["full_fraction"] for r in values]))))
    for h in (5, 3, 4):
        for b in (4096, 8192, 16384, 32768):
            for comparator in ("labels_full_webp", "labels_webp_mix", "labels_adaptive_change", "labels_adaptive_random"):
                ids = sorted(k[3] for k in per_pair if k[:3] == (h, b, "labels_adaptive_mix"))
                a = [per_pair[(h, b, "labels_adaptive_mix", p)] for p in ids]
                c = [per_pair[(h, b, comparator, p)] for p in ids]
                if any(x["confusion"] != y["confusion"] for x, y in zip(a, c)):
                    raise AssertionError("Selector changed the shared known service")
                w = np.array([r["unknown"] for r in a])
                active = w > 0
                weights = w[active]
                delta = np.array([x["retained"] - y["retained"] for x, y in zip(a, c)])[active]
                draw = np.random.default_rng(20261104).multinomial(len(weights), np.full(len(weights), 1/len(weights)), size=10000)
                low, high = np.quantile((draw @ delta)/(draw @ weights), [.00625, .99375])
                comparisons.append(dict(heldout=h, budget=b, method="labels_adaptive_mix", comparator=comparator,
                    delta_unknown_coverage_30=float(delta.sum()/weights.sum()), unknown_low=float(low), unknown_high=float(high),
                    known_confusions_identical=True, delta_mean_bytes=float(np.mean([x["bytes"]-y["bytes"] for x, y in zip(a, c)]))))
    for name, rows in (("summary", summaries), ("comparisons", comparisons)):
        with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "analysis.json").write_text(json.dumps(dict(status="complete", source_sha256=sha(Path(__file__)),
        records_sha256=receipt["records_sha256"], source_candidates_sha256=sha(BASE / "records.csv"),
        summary_sha256=sha(OUT / "summary.csv"), comparisons_sha256=sha(OUT / "comparisons.csv"),
        scope="64 development pairs; 98.75% descriptive four-cap intervals per class/contrast",
        metadata_clarification="The prototype flag sender_has_semantic_GT=False denotes non-use in the selection function. Its orchestration loaded pair labels and offline candidate metric records. The quality decision itself takes only RGB and score inputs."), indent=2)+"\n", "utf-8")
    print(json.dumps([r for r in comparisons if r["comparator"] in ("labels_full_webp", "labels_adaptive_change")], indent=2), flush=True)


if __name__ == "__main__":
    main()
