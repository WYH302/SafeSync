"""Area-paired comparison of independent references and stateful recovery."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/link_recovery/fixed_reference_v1"
OLDER = ROOT / "results/link_recovery/real_features_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    group = defaultdict(list)
    for folder in (OUT, OLDER):
        receipt = json.loads((folder / "complete.json").read_text("utf-8"))
        if receipt["status"] != "complete" or sha(folder / "records.csv") != receipt["records_sha256"]:
            raise ValueError("Incomplete fixed-reference comparison")
        for row in csv.DictReader((folder / "per_aoi_seed.csv").open(encoding="utf-8")):
            if row["split"] == "confirmation":
                group[(row["profile"], row["policy"], row["aoi"])].append(row)
    metrics = ("mean_update_bytes", "mean_world_mse", "mean_age")
    area = {key: {m: float(np.mean([float(r[m]) for r in rows])) for m in metrics} for key, rows in group.items()}
    summaries, contrasts = [], []
    for profile in ("lossless", "iid10", "iid30", "burst"):
        for policy in ("delta_arq", "periodic_4", "periodic_8", "periodic_16", "risk_anchor", "fixed_reference", "absolute_refresh"):
            ids = sorted(k[2] for k in area if k[:2] == (profile, policy))
            summaries.append(dict(profile=profile, policy=policy, AOIs=len(ids),
                **{m: float(np.mean([area[(profile, policy, a)][m] for a in ids])) for m in metrics},
                extra_receiver_reference_bytes=98304 if policy == "fixed_reference" else 0))
        for comparator in ("risk_anchor", "delta_arq", "absolute_refresh"):
            ids = sorted(k[2] for k in area if k[:2] == (profile, comparator))
            row = dict(profile=profile, method="fixed_reference", comparator=comparator, AOIs=len(ids))
            draws = np.random.default_rng(20261104).integers(0, len(ids), (10000, len(ids)))
            for metric in metrics:
                a = np.array([area[(profile, "fixed_reference", p)][metric] for p in ids])
                b = np.array([area[(profile, comparator, p)][metric] for p in ids])
                low, high = np.quantile((a - b)[draws].mean(1), [.00625, .99375])
                row[metric] = dict(new_mean=float(a.mean()), comparator_mean=float(b.mean()),
                    paired_delta=float(np.mean(a - b)), low=float(low), high=float(high))
            contrasts.append(row)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    result = dict(status="complete", source_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (OUT / "complete.json", OLDER / "complete.json")},
        summary=summaries, contrasts=contrasts,
        statistics="Eight areas, three seeds averaged within each area, 98.75% descriptive four-profile intervals per contrast/metric",
        interpretation="Risk refresh has a rate/freshness trade-off. A fixed initial reference decodes fresh updates without waiting for feedback, using an extra 98304-byte receiver reference and longer residuals.")
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Independent-reference recovery controls on eight SN7 areas. Costs include emitted feedback; three seeds are nested within area. World error is in units of $10^{-5}$. Fixed-reference decoding retains an additional 96 KiB of quantized history.}",
        "\\label{tab:fixed-reference}", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llrrr}", "\\toprule", "Profile & Policy & KiB/update & Error & Age " + chr(92) * 2, "\\midrule"]
    for r in summaries:
        if r["profile"] != "lossless" and r["policy"] in ("risk_anchor", "fixed_reference", "absolute_refresh"):
            name = {"risk_anchor": "Risk refresh", "fixed_reference": "Fixed reference", "absolute_refresh": "Absolute current"}[r["policy"]]
            lines.append(f"{r['profile']} & {name} & {r['mean_update_bytes']/1024:.2f} & {r['mean_world_mse']*1e5:.2f} & {r['mean_age']:.3f} " + chr(92) * 2)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (ROOT / "paper/supplement/generated/recovery_fixed.tex").write_text("\n".join(lines), "utf-8")
    print(json.dumps(dict(status="complete", opportunities=6480, comparisons=len(contrasts))), flush=True)


if __name__ == "__main__":
    main()
