"""Bounded known-only scorer/routing repair with real frozen SP03 transactions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safesync.data import read_pair
from scripts.run_second_feature_mix import load_frozen, initial_state

SCORES = ROOT / "results/unknown_recovery/scores_v1"
OLD = ROOT / "results/second_pixels/fixed_v1"
METHODS = ("knn_mix", "knn_band", "emergent_mix", "emergent_band", "maha_band", "band_random", "oracle_band", "oracle_all")
BUDGETS = (4096, 8192, 16384, 32768)
HELDOUTS = (5, 3, 4)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values):
    return rankdata(values, method="average") / len(values)


def priority(method, change, scores, unknown_counts, gate, seed, pair_id):
    change_rank = rank(change)
    bands = np.minimum(np.floor(change_rank * 8), 7)
    if method == "band_random":
        key = hashlib.sha256(f"unknown-band:{seed}:{pair_id}".encode()).digest()[:8]
        extra = np.random.default_rng(int.from_bytes(key, "little")).random(64)
    elif method.startswith("oracle"):
        extra = rank(unknown_counts)
    elif method == "maha_band":
        extra = rank(scores["mahalanobis"])
    elif method.startswith("emergent"):
        extra = rank(scores["knn_emergence"])
    else:
        extra = rank(scores["knn"])
    if method == "oracle_all":
        result = extra
    elif method.endswith("mix"):
        result = .75 * change_rank + .25 * extra
    else:
        result = bands + extra
    return np.where(gate, result, 0)


def summarize(out, rows, baseline, save_csv):
    # Average random seeds within each source pair before paired resampling.
    grouped = defaultdict(list)
    for row in rows + baseline:
        grouped[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    fields = ("unknown_pixels", "known_pixels", "unknown_retained_pixels_30", "known_error_sum", "total_bytes")
    per_pair = {key: {name: float(np.mean([float(r[name]) for r in group])) for name in fields}
                for key, group in grouped.items()}
    summaries = []
    comparisons = []
    for heldout in HELDOUTS:
        for budget in BUDGETS:
            for method in METHODS + ("feature_change", "full_jpeg", "full_jpeg2000"):
                ids = sorted(k[3] for k in per_pair if k[:3] == (heldout, budget, method))
                if not ids:
                    continue
                group = [per_pair[(heldout, budget, method, p)] for p in ids]
                positive = [r for r in group if r["unknown_pixels"] > 0]
                known_n = sum(r["known_pixels"] for r in group)
                unknown_n = sum(r["unknown_pixels"] for r in positive)
                summaries.append(dict(heldout=heldout, budget=budget, method=method, pairs=len(group),
                                      diagnostic_oracle=method.startswith("oracle"),
                                      unknown_coverage_30=sum(r["unknown_retained_pixels_30"] for r in positive) / unknown_n if unknown_n else None,
                                      known_mse=sum(r["known_error_sum"] for r in group) / known_n if known_n else None,
                                      mean_total_bytes=float(np.mean([r["total_bytes"] for r in group]))))
                if method not in METHODS:
                    continue
                for comparator in ("feature_change", "band_random"):
                    if method == comparator or any((heldout, budget, comparator, p) not in per_pair for p in ids):
                        continue
                    old = [per_pair[(heldout, budget, comparator, p)] for p in ids]
                    selected = [i for i, r in enumerate(group) if r["unknown_pixels"] > 0]
                    weights = np.array([group[i]["unknown_pixels"] for i in selected])
                    delta = np.array([group[i]["unknown_retained_pixels_30"] - old[i]["unknown_retained_pixels_30"] for i in selected])
                    low = high = value = None
                    if len(weights):
                        sample = np.random.default_rng(20261104).integers(0, len(weights), (2000, len(weights)))
                        distribution = delta[sample].sum(1) / weights[sample].sum(1)
                        # Four-cap family per scorer/class/comparator. Exploratory development data.
                        low, high = map(float, np.quantile(distribution, [.00625, .99375]))
                        value = float(delta.sum() / weights.sum())
                    comparisons.append(dict(heldout=heldout, budget=budget, method=method, comparator=comparator,
                                            diagnostic_oracle=method.startswith("oracle"), pairs=len(ids),
                                            delta_coverage_30=value, ci_low=low, ci_high=high,
                                            delta_known_mse=sum(a["known_error_sum"] - b["known_error_sum"] for a, b in zip(group, old)) / known_n if known_n else None,
                                            delta_mean_total_bytes=float(np.mean([a["total_bytes"] - b["total_bytes"] for a, b in zip(group, old)]))))
    save_csv(out / "summary.csv", summaries)
    save_csv(out / "comparisons.csv", comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "results/unknown_recovery/codec_v1")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    old_spec = json.loads((OLD / "run_spec.json").read_text("utf-8"))
    wire, helper = load_frozen(old_spec)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    all_pairs = manifest["pairs"]
    pairs = [p for p in all_pairs if p["split"] == "validation"]
    score_complete = json.loads((SCORES / "complete.json").read_text("utf-8"))
    data = {}
    for receipt in score_complete["receipts"]:
        path = ROOT / receipt["path"]
        if sha(path) != receipt["sha256"]:
            raise ValueError("Changed new novelty scores")
        with np.load(path, allow_pickle=False) as stored:
            if stored["ids"].tolist() != [p["id"] for p in all_pairs]:
                raise ValueError("Unknown scoring ID mismatch")
            data[receipt["heldout"]] = {name: stored[name] for name in ("knn", "knn_emergence", "mahalanobis")}
    spec = dict(source_sha256=sha(Path(__file__)), old_spec_sha256=sha(OLD / "run_spec.json"),
                score_receipt_sha256=sha(SCORES / "complete.json"), manifest_sha256=sha(manifest_path),
                methods=METHODS, budgets=BUDGETS, heldouts=HELDOUTS,
                bands=8, mixing_weights=[.75, .25], random_seeds=[1729, 2718, 31415],
                physical_gate_psnr=25, evidence_psnr=30, codec="unchanged frozen SP03 JPEG60, 64-pixel ROI atlas, coarse4",
                scope="Previously inspected 64 development pairs; known-only gallery; no official-test access",
                router_uses_gt="only the explicitly named oracle arms, used for headroom diagnosis",
                oracle_scope="unknown area prioritization, not an exact global optimum or task upper bound",
                comparison_ci="98.75% per comparison; 4 budgets within scorer/class/comparator; exploratory",
                limit=args.limit)
    spec_path = args.out / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != json.loads(json.dumps(spec)):
        raise ValueError("Changed repair experiment configuration")
    helper["save_json"](spec_path, spec)
    shutil.copyfile(Path(__file__), args.out / "source_snapshot.py")
    baseline = [r for r in csv.DictReader((OLD / "records.csv").open(encoding="utf-8"))
                if r["method"] in ("feature_change", "full_jpeg", "full_jpeg2000")]
    helper["save_csv"](args.out / "reference_records.csv", baseline)
    rows, completed = [], []
    if (args.out / "records.csv").exists():
        rows = list(csv.DictReader((args.out / "records.csv").open(encoding="utf-8")))
        completed = json.loads((args.out / "progress.json").read_text("utf-8"))["completed_pair_ids"]
    selected = pairs[:args.limit or None]
    for pair in selected:
        if pair["id"] in completed:
            continue
        before_pil, after_pil, old_label, label, changed = read_pair(ROOT, pair)
        before, current = np.asarray(before_pil), np.asarray(after_pil)
        folder = args.out / "pairs" / pair["id"]
        folder.mkdir(parents=True, exist_ok=True)
        prior, initial_down, initial_ack = initial_state(wire, folder, pair["id"], before)
        gate = helper["exact_mse"](current, helper["restore_coarse"](prior.rgb, current)) > 10 ** (-2.5)
        index = next(i for i, p in enumerate(all_pairs) if p["id"] == pair["id"])
        with np.load(OLD / "input_snapshot" / f"scores_5_{pair['id']}.npz", allow_pickle=False) as stored:
            change = helper["pool_scores"](stored["change"])
        for heldout in HELDOUTS:
            known_mask, unknown_mask = helper["evaluation_masks"](old_label, label, changed, heldout)
            unknown_counts = unknown_mask.reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1)
            scores = {key: helper["pool_scores"](value[index]) for key, value in data[heldout].items()}
            for method in METHODS:
                for seed in ((1729, 2718, 31415) if method == "band_random" else (0,)):
                    order = priority(method, change, scores, unknown_counts, gate, seed, pair["id"])
                    for budget in BUDGETS:
                        packet, ack, decoded, detail = helper["transact"](prior, current, method, budget, order, gate)
                        transaction = folder / f"h{heldout}_{method}_s{seed}_b{budget}"
                        transaction.mkdir(exist_ok=True)
                        (transaction / "current.bin").write_bytes(packet)
                        (transaction / "ack.bin").write_bytes(ack)
                        rows.append(dict(pair_id=pair["id"], heldout=heldout, method=method, seed=seed, budget=budget,
                                         router_uses_gt=method.startswith("oracle"), initial_down_bytes=initial_down,
                                         initial_ack_bytes=initial_ack,
                                         packet_path=(transaction / "current.bin").relative_to(ROOT).as_posix(),
                                         **detail, **helper["image_metrics"](current, decoded, known_mask, unknown_mask)))
        completed.append(pair["id"])
        helper["save_csv"](args.out / "records.csv", rows)
        helper["save_json"](args.out / "progress.json", dict(status="in_progress", completed_pair_ids=completed,
                            records=len(rows), expected_pairs=len(selected)))
        print(f"unknown repair {len(completed)}/{len(selected)} pairs, {len(rows)} actual decoded observations", flush=True)
    allowed_ids = set(completed)
    summarize(args.out, rows, [r for r in baseline if r["pair_id"] in allowed_ids], helper["save_csv"])
    expected = len(selected) * len(HELDOUTS) * len(BUDGETS) * (len(METHODS) + 2)
    if len(rows) != expected:
        raise AssertionError("Repair matrix record count differs")
    helper["save_json"](args.out / "complete.json", dict(status="complete", records=len(rows), expected=expected,
                        source_sha256=sha(Path(__file__)), records_sha256=sha(args.out / "records.csv"),
                        summary_sha256=sha(args.out / "summary.csv"), spec_sha256=sha(spec_path),
                        completed_pairs=len(completed), official_test_pixels_read=0))


if __name__ == "__main__":
    main()
