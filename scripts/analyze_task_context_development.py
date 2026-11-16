"""Evidence and task-gate summary for the completed current-context repair."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/task_evidence/context_development_v1"
BASE = ROOT / "results/task_evidence/development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((OUT / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(OUT / "records.csv") != receipt["records_sha256"]:
        raise ValueError("Incomplete current-context result")
    rows = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
    rows += [r for r in csv.DictReader((BASE / "records.csv").open(encoding="utf-8")) if r["method"] in ("labels_full_webp", "labels_full_jpeg420")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    per_pair = {key: dict(unknown=float(group[0]["unknown_pixels"]),
        retained=np.mean([float(r["unknown_retained_pixels_30"]) for r in group]), bytes=np.mean([float(r["total_bytes"]) for r in group]),
        matrix=json.loads(group[0]["known_confusion"]), full_fraction=np.mean([r.get("selected_action") == "full" for r in group])) for key, group in grouped.items()}
    summaries, contrasts = [], []
    for h, cap, method in sorted({k[:3] for k in per_pair}):
        values = [v for k, v in per_pair.items() if k[:3] == (h, cap, method)]
        summaries.append(dict(heldout=h, budget=cap, method=method, pairs=len(values),
            unknown_coverage_30=float(sum(v["retained"] for v in values)/sum(v["unknown"] for v in values)),
            mean_total_bytes=float(np.mean([v["bytes"] for v in values])),
            full_image_fraction=float(np.mean([v["full_fraction"] for v in values]))))
    for h in (5, 3, 4):
        for cap in (4096, 8192, 16384, 32768):
            for comparator in ("labels_full_webp", "labels_context_adaptive_change", "labels_context_adaptive_random", "labels_context_mix"):
                ids = sorted(k[3] for k in per_pair if k[:3] == (h, cap, "labels_context_adaptive_mix"))
                a = [per_pair[(h, cap, "labels_context_adaptive_mix", p)] for p in ids]
                b = [per_pair[(h, cap, comparator, p)] for p in ids]
                if any(x["matrix"] != y["matrix"] for x, y in zip(a, b)):
                    raise AssertionError("Context route changed the known-task service")
                support = np.array([x["unknown"] for x in a])
                active = support > 0
                weights = support[active]
                delta = np.array([x["retained"] - y["retained"] for x, y in zip(a, b)])[active]
                draw = np.random.default_rng(20261104).multinomial(len(weights), np.full(len(weights), 1/len(weights)), size=10000)
                low, high = np.quantile((draw@delta)/(draw@weights), [.00625, .99375])
                contrasts.append(dict(heldout=h, budget=cap, method="labels_context_adaptive_mix", comparator=comparator,
                    delta_unknown_coverage_30=float(delta.sum()/weights.sum()), low=float(low), high=float(high),
                    known_confusions_identical=True, delta_mean_bytes=float(np.mean([x["bytes"]-y["bytes"] for x, y in zip(a, b)]))))
    for name, records in (("summary", summaries), ("comparisons", contrasts)):
        with (OUT / (name+".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = dict(status="complete", source_sha256=sha(Path(__file__)), records_sha256=receipt["records_sha256"],
        summary_sha256=sha(OUT / "summary.csv"), comparisons_sha256=sha(OUT / "comparisons.csv"),
        scope="64 development pairs, four-cap 98.75% descriptive intervals per class/contrast", comparisons=contrasts)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2)+"\n", "utf-8")
    print(json.dumps([r for r in contrasts if r["heldout"] == 5 and r["comparator"] in ("labels_full_webp", "labels_context_adaptive_change")], indent=2), flush=True)


if __name__ == "__main__":
    main()
