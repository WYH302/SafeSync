"""Cheap independent accounting/statistics check of the completed OOF cohort.

Reads the immutable per-frame CSV and completed summaries; no image inference,
packet generation, parameter selection, or alteration of the frozen run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "receipts_v1"
TASKS = ["building", "event", "initial_only_building", "initial_only_event"]
FIELDS = ["tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels", "predicted_positive_pixels"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left, right):
    if left is None or right is None:
        assert left is right, (left, right)
    else:
        assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right)


def rates(count):
    tp, fp, fn = (count[key] for key in ["tp", "fp", "fn"])
    divide = lambda a, b: a / b if b else None
    return dict(f1=divide(2 * tp, 2 * tp + fp + fn), iou=divide(tp, tp + fp + fn),
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn))


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    spec, summary, complete = [read(OUT / name) for name in ["run_spec.json", "summary.json", "complete.json"]]
    assert complete["status"] == summary["status"] == "complete"
    verified_artifacts = {(OUT / name).relative_to(ROOT).as_posix(): sha(OUT / name)
                          for name in ["run_spec.json", "summary.json", "complete.json", "paired_comparisons.json"]}
    for key, name in [("source_spec_sha256", "run_spec.json"), ("summary_sha256", "summary.json"),
                      ("paired_comparisons_sha256", "paired_comparisons.json")]:
        assert complete[key] == sha(OUT / name)
    manifest = read(ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json")
    all_groups, total, zero_valid_rows = {}, 0, 0
    for index, aoi in enumerate(manifest["aois"]):
        folder = OUT / f"area_{index}"
        marker, area_spec, area_summary = [read(folder / name) for name in ["progress.json", "run_spec.json", "summary.json"]]
        assert marker["status"] == "complete"
        for key, name in [("records_sha256", "records.csv"), ("summary_sha256", "summary.json"), ("spec_sha256", "run_spec.json")]:
            actual = sha(folder / name)
            assert marker[key] == actual
            verified_artifacts[(folder / name).relative_to(ROOT).as_posix()] = actual
        assert area_spec["parent_spec_sha256"] == sha(OUT / "run_spec.json")
        with (folder / "records.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(method, budget, step, month["month"]) for method in spec["methods"]
                    for budget in spec["budgets"] for step, month in enumerate(aoi["months"][1:], 1)}
        identities = {(r["method"], int(r["budget"]), int(r["step"]), r["month"]) for r in rows}
        assert identities == expected and len(rows) == len(expected) == marker["completed"]
        groups = defaultdict(lambda: {task: {field: 0 for field in FIELDS} for task in TASKS})
        binding = spec["fixed_fold_bindings"][index]
        for row in rows:
            assert row["aoi"] == aoi["aoi_id"]
            assert row["model_experiment"] == binding["selected_experiment"]
            assert int(row["author_validation_fold"]) == binding["author_validation_fold"]
            assert row["weight_sha256"] == spec["weight_sha256_by_experiment"][row["model_experiment"]]
            assert row["model_hash"] == area_spec["model_hash"] and row["threshold_hash"] == area_spec["threshold_hash"]
            assert row["part_of_original_blind_confirmation"] == "False"
            assert row["probability_path"].startswith(f"resources/pretrained_ground_crossfit/receipts_v1/area_{index}/probabilities/")
            assert int(row["down_bytes"]) <= int(row["budget"])
            zero_valid_rows += int(int(row["building_valid_pixels"]) == 0)
            key = row["method"], int(row["budget"]), row["aoi"]
            for task in TASKS:
                counts = {field: int(row[f"{task}_{field}"]) for field in FIELDS}
                assert sum(counts[k] for k in ["tp", "fp", "fn", "tn"]) == counts["valid_pixels"]
                assert counts["tp"] + counts["fn"] == counts["positive_pixels"]
                assert counts["tp"] + counts["fp"] == counts["predicted_positive_pixels"]
                for field, value in counts.items():
                    groups[key][task][field] += value
        area_lookup = {(g["method"], g["budget"], g["aoi"]): g for g in area_summary["per_aoi_seed"]}
        assert set(groups) == set(area_lookup)
        for key, counts in groups.items():
            for task in TASKS:
                for field, value in counts[task].items():
                    assert value == area_lookup[key][task][field]
                counts[task].update(rates(counts[task]))
                for field, value in rates(counts[task]).items():
                    same(value, area_lookup[key][task][field])
            all_groups[key] = counts
        total += len(rows)
    assert total == spec["expected_outputs"] == summary["outputs"] == complete["outputs"] == 4806
    assert summary["completed_areas"] == list(range(8))
    for item in summary["macro_aoi"]:
        assert item["AOIs"] == 8
        for task in TASKS:
            for metric in ["f1", "iou", "precision", "recall"]:
                same(item[task][metric], mean([all_groups[(item["method"], item["budget"], aid)][task][metric]
                                              for aid in spec["aoi_ids"]]))
    pairs = read(OUT / "paired_comparisons.json")
    assert len(pairs) == 12
    assert {(p["comparator"], p["budget"], p["task"]) for p in pairs} == {
        (method, cap, task) for method in ["full_webp", "learned_hyperprior_q1_8"]
        for cap in spec["budgets"] for task in ["building", "event"]}
    sample_ids = np.random.default_rng(20261029).integers(0, 8, (10000, 8))
    tail = (1 - (1 - .05 / 3)) / 2
    for comparison in pairs:
        assert comparison["paired_AOIs"] == 8 and comparison["bootstrap_resamples"] == 10000
        assert comparison["bootstrap_seed"] == 20261029
        same(comparison["confidence"], 1 - .05 / 3)
        differences = []
        for aid in spec["aoi_ids"]:
            left = all_groups[("gaussian_20261029", comparison["budget"], aid)][comparison["task"]]
            right = all_groups[(comparison["comparator"], comparison["budget"], aid)][comparison["task"]]
            assert left["valid_pixels"] == right["valid_pixels"] and left["positive_pixels"] == right["positive_pixels"] > 0
            differences.append(left["f1"] - right["f1"])
        values = np.asarray(differences)
        lo, hi = np.quantile(values[sample_ids].mean(axis=1), [tail, 1 - tail])
        same(comparison["difference_f1"], float(values.mean()))
        same(comparison["ci_low"], float(lo))
        same(comparison["ci_high"], float(hi))
    report = {"status": "passed", "outputs": total, "AOIs": 8, "zero_valid_frame_rows": zero_valid_rows,
              "checked": ["completion hashes", "full frame identities", "model/fold/threshold identities",
                          "native OOF cache path isolation", "per-frame support arithmetic", "AOI count-first rates",
                          "equal-AOI macro rates", "all 12 paired 10000-bootstrap contrasts"],
              "limits": "No new inference or repeat packet replay; no full probability-file rehash.",
              "source_spec_sha256": sha(OUT / "run_spec.json"), "verifier_sha256": sha(Path(__file__)),
              "checked_artifact_sha256": verified_artifacts,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "outputs", "AOIs", "zero_valid_frame_rows", "checked", "limits", "checked_utc"]}))


if __name__ == "__main__":
    main()
