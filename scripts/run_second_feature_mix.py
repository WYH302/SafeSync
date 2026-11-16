"""One fixed feature-change controlled addition to the frozen SECOND experiment."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_labels
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safesync.data import read_pair

OLD = ROOT / "results/second_pixels/fixed_v1"
OUT = ROOT / "results/second_pixels/feature_mix_v1"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execute_functions(path, names, namespace):
    """Execute unchanged named functions from the hashed source snapshot only."""
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise RuntimeError("Frozen helper function inventory changed")
    code = compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec")
    exec(code, namespace)


def load_frozen(old_spec):
    snapshot = OLD / "source_snapshot"
    for source, expected in old_spec["source_sha256"].items():
        if sha_file(snapshot / source.replace("/", "__")) != expected:
            raise RuntimeError("Original source snapshot changed")
    loader = importlib.util.spec_from_file_location("feature_mix_frozen_wire", snapshot / "safesync__pixel_wire.py")
    wire = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = wire
    loader.loader.exec_module(wire)
    namespace = dict(np=np, hashlib=hashlib, json=json, csv=csv, time=time, rankdata=rankdata,
                     CONFIG=old_spec["config"], ACK_BYTES=wire.ACK.size + 16,
                     **{name: getattr(wire, name) for name in (
                         "PixelState", "TransactionReceiver", "TransactionSender", "encode_update",
                         "STABLE", "UPDATED", "DEFERRED")})
    execute_functions(snapshot / "safesync__reference.py", ["blocks", "exact_mse", "restore_coarse"], namespace)
    execute_functions(snapshot / "scripts__run_pixel_sequences.py", ["choose_packet", "full_packet"], namespace)
    execute_functions(snapshot / "scripts__run_second_pixel.py", [
        "pool_scores", "ranking_scores", "evaluation_masks", "region_sums", "image_metrics",
        "fresh_receiver", "transact", "sanity_images", "save_csv", "save_json"], namespace)
    return wire, namespace


def initial_state(wire, folder, pair_id, before):
    original = OLD / "pairs" / pair_id
    for name in ("initial.bin", "initial.ack", "initial_cost.json"):
        shutil.copyfile(original / name, folder / name)
    body = (folder / "initial.bin").read_bytes()
    fields = wire.HEADER.unpack(body[:wire.HEADER.size])
    state = wire.PixelState(np.zeros_like(before), fields[11], fields[3])
    receiver = wire.TransactionReceiver(state)
    ack, _ = receiver.receive(body)
    if ack != (folder / "initial.ack").read_bytes() or not np.array_equal(receiver.state.rgb, before):
        raise AssertionError("Original initialization bytes no longer match shared historical RGB")
    return receiver.state, len(body), len(ack)


def summarize(helpers, records, baseline, completed):
    save_csv, save_json = helpers["save_csv"], helpers["save_json"]
    relevant = records + [row for row in baseline if row["pair_id"] in completed]
    numeric = ["heldout", "budget", "seed", "unknown_pixels", "known_pixels", "global_pixels",
               "unknown_components_8", "total_bytes", "known_error_sum", "unknown_error_sum", "global_error_sum",
               "encoding_ms", "decoding_ack_ms", "unknown_retained_pixels_25",
               "unknown_retained_pixels_30", "unknown_retained_pixels_35"]
    groups, by_pair = defaultdict(list), defaultdict(list)
    for raw in relevant:
        row = dict(raw)
        for key in numeric:
            row[key] = float(row[key])
        groups[(int(row["heldout"]), int(row["budget"]), row["method"], int(row["seed"]))].append(row)
        by_pair[(int(row["heldout"]), int(row["budget"]), row["method"], row["pair_id"])].append(row)
    summaries = []
    for (heldout, budget, method, seed), rows in sorted(groups.items()):
        n_unknown = sum(row["unknown_pixels"] for row in rows)
        n_known = sum(row["known_pixels"] for row in rows)
        positives = [row for row in rows if row["unknown_pixels"] > 0]
        summary = dict(heldout=heldout, budget=budget, method=method, seed=seed, pairs=len(rows),
            positive_pairs=len(positives), unknown_pixels=int(n_unknown),
            unknown_components_8=int(sum(row["unknown_components_8"] for row in rows)),
            mean_total_bytes=float(np.mean([row["total_bytes"] for row in rows])),
            known_mse=sum(row["known_error_sum"] for row in rows) / n_known if n_known else None,
            unknown_mse=sum(row["unknown_error_sum"] for row in rows) / n_unknown if n_unknown else None,
            global_mse=sum(row["global_error_sum"] for row in rows) / sum(row["global_pixels"] for row in rows),
            mean_encoding_ms=float(np.mean([row["encoding_ms"] for row in rows])),
            mean_decoding_ack_ms=float(np.mean([row["decoding_ack_ms"] for row in rows])))
        for q in (25, 30, 35):
            retained = f"unknown_retained_pixels_{q}"
            summary[f"unknown_coverage_{q}_micro"] = sum(row[retained] for row in rows) / n_unknown if n_unknown else None
            summary[f"unknown_coverage_{q}_macro_positive"] = float(np.mean([row[retained] / row["unknown_pixels"] for row in positives])) if positives else None
        summaries.append(summary)
    save_csv(OUT / "summary.csv", summaries)
    mean = {key: {field: float(np.mean([row[field] for row in rows])) for field in numeric}
            for key, rows in by_pair.items()}
    comparisons = []
    for heldout in (5, 3, 4):
        for budget in (4096, 8192, 16384, 32768):
            ids = sorted(key[3] for key in mean if key[:3] == (heldout, budget, "feature_unknown"))
            for name in ("feature_change", "feature_random"):
                new = [mean[(heldout, budget, "feature_unknown", pair)] for pair in ids]
                old = [mean[(heldout, budget, name, pair)] for pair in ids]
                positive = [i for i, row in enumerate(new) if row["unknown_pixels"] > 0]
                weights = np.array([new[i]["unknown_pixels"] for i in positive])
                known_n = sum(row["known_pixels"] for row in new)
                result = dict(heldout=heldout, budget=budget, method="feature_unknown", baseline=name,
                    pairs=len(ids), positive_pairs=len(positive),
                    delta_known_mse=sum(a["known_error_sum"] - b["known_error_sum"] for a,b in zip(new,old)) / known_n if known_n else None,
                    delta_mean_total_bytes=float(np.mean([a["total_bytes"] - b["total_bytes"] for a,b in zip(new,old)])))
                for q in (25, 30, 35):
                    change = np.array([new[i][f"unknown_retained_pixels_{q}"] - old[i][f"unknown_retained_pixels_{q}"] for i in positive])
                    delta = low = high = None
                    if len(weights):
                        delta = float(change.sum() / weights.sum())
                        rng = np.random.default_rng(20261028)
                        sample = rng.integers(0, len(weights), (2000, len(weights)))
                        boot = change[sample].sum(1) / weights[sample].sum(1)
                        low, high = map(float, np.quantile(boot, [.025, .975]))
                    result[f"delta_unknown_coverage_{q}"] = delta
                    result[f"paired_bootstrap_low_{q}"] = low
                    result[f"paired_bootstrap_high_{q}"] = high
                comparisons.append(result)
    save_csv(OUT / "comparisons.csv", comparisons)
    save_json(OUT / "progress.json", dict(completed_pairs=len(completed), expected_pairs=64,
              complete=len(completed) == 64, records=len(records), completed_pair_ids=completed,
              reference_records=sum(row["pair_id"] in completed for row in baseline), official_test_pixels_read=0))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=(0, 8), default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    old_spec = json.loads((OLD / "run_spec.json").read_text("utf-8"))
    checked = json.loads((OLD / "verification.json").read_text("utf-8"))
    if not checked["passed"] or sha_file(OLD / "records.csv") != checked["records_sha256"]:
        raise RuntimeError("Original fixed baseline is not the verified result")
    wire, helper = load_frozen(old_spec)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    if sha_file(manifest_path) != old_spec["manifest_sha256"]:
        raise RuntimeError("Data changed since original fixed run")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    pairs = [pair for pair in manifest["pairs"] if pair["split"] == "validation"]
    if [pair["id"] for pair in pairs] != old_spec["validation_pair_ids"]:
        raise RuntimeError("Development validation prefix differs")
    config = dict(old_spec["config"])
    config.update(protocol="SECOND-feature-mix-SP03-fixed-v1", methods=["feature_unknown", "feature_random"],
        ranking_definitions={"feature_unknown": ".75 rank(feature_change)+.25 rank(unknown)",
                             "feature_random": ".75 rank(feature_change)+.25 independent random rank"},
        reason="Conditional diagnostic controlled DINO feature change; isolate unknown incremental value beyond that same change score.",
        shared_configuration_sha256=old_spec["config_sha256"], feature_rank_weight=.75, other_rank_weight=.25)
    config.pop("pixel_rank_weight")
    source = dict(old_spec["source_sha256"])
    source["scripts/run_second_feature_mix.py"] = sha_file(__file__)
    spec = dict(config=config, config_sha256=fingerprint(config), source_sha256=source,
        source_bundle_sha256=fingerprint(source), encoder=old_spec["encoder"],
        known_model_sha256=old_spec["known_model_sha256"], score_files_sha256=old_spec["score_files_sha256"],
        validation_pair_ids=old_spec["validation_pair_ids"], manifest_sha256=old_spec["manifest_sha256"],
        feature_cache_sha256=old_spec["feature_cache_sha256"], feature_metadata_sha256=old_spec["feature_metadata_sha256"],
        environment=old_spec["environment"], reference_run_spec_sha256=sha_file(OLD / "run_spec.json"),
        reference_records_sha256=sha_file(OLD / "records.csv"),
        frozen_function_source="results/second_pixels/fixed_v1/source_snapshot; unchanged functions selected by AST, no source rewriting",
        score_source="results/second_pixels/fixed_v1/input_snapshot; original hashed values",
        statistics_unit=old_spec["statistics_unit"], quality_metric=old_spec["quality_metric"],
        timing_scope=old_spec["timing_scope"], official_test_pixels_read=0)
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise RuntimeError("New fixed study inputs changed; refusing mixed run")
    helper["save_json"](spec_path, spec)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in old_spec["source_sha256"]:
        shutil.copyfile(OLD / "source_snapshot" / name.replace("/", "__"), snapshot / name.replace("/", "__"))
    shutil.copyfile(__file__, snapshot / "scripts__run_second_feature_mix.py")
    all_original = list(csv.DictReader((OLD / "records.csv").open(encoding="utf-8")))
    baseline = [row for row in all_original if row["method"] == "feature_change"]
    helper["save_csv"](OUT / "reference_records.csv", baseline)
    # Preflight the exact feature-controlled mixture without touching any data label.
    vector, reverse = np.arange(64), np.arange(64)[::-1]
    got = helper["ranking_scores"]("pixel_unknown", vector, vector, reverse, np.ones(64, bool), 0, "check")
    assert np.array_equal(got, .75 * rankdata(vector) / 64 + .25 * rankdata(reverse) / 64)
    records, sanity, completed = [], [], []
    if (OUT / "records.csv").exists():
        records = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
        sanity = list(csv.DictReader((OUT / "sanity.csv").open(encoding="utf-8")))
        completed = json.loads((OUT / "progress.json").read_text("utf-8"))["completed_pair_ids"]
    for pair in (pairs[:8] if args.limit else pairs):
        if pair["id"] in completed:
            continue
        if sha_file(__file__) != source["scripts/run_second_feature_mix.py"]:
            raise RuntimeError("Fixed new driver changed during execution")
        before_pil, current_pil, old_label, label, changed = read_pair(ROOT, pair)
        before, current = np.asarray(before_pil), np.asarray(current_pil)
        folder = OUT / "pairs" / pair["id"]
        folder.mkdir(parents=True, exist_ok=True)
        prior, initial_down, initial_ack = initial_state(wire, folder, pair["id"], before)
        mse = helper["exact_mse"](current, helper["restore_coarse"](prior.rgb, current))
        gate = mse > 10 ** (-2.5)
        sanity_images = helper["sanity_images"](prior, current, folder, gate)
        common, reference_change = {}, None
        for heldout in (5, 3, 4):
            score_path = OLD / "input_snapshot" / f"scores_{heldout}_{pair['id']}.npz"
            if sha_file(score_path) != old_spec["score_files_sha256"][f"{heldout}:{pair['id']}"]:
                raise RuntimeError("Archived scores changed")
            with np.load(score_path, allow_pickle=False) as data:
                change, unknown = helper["pool_scores"](data["change"]), helper["pool_scores"](data["unknown"])
            if reference_change is not None and not np.array_equal(reference_change, change):
                raise RuntimeError("Feature baseline differs across folds")
            reference_change = change
            known_mask, unknown_mask = helper["evaluation_masks"](old_label, label, changed, heldout)
            components = int(connected_labels(unknown_mask, np.ones((3, 3)))[1])
            for case, prediction in sanity_images.items():
                sanity.append(dict(pair_id=pair["id"], heldout=heldout, case=case, unknown_components_8=components,
                                   **helper["image_metrics"](current, prediction, known_mask, unknown_mask)))
            for method in ("feature_unknown", "feature_random"):
                for seed in ([0] if method == "feature_unknown" else [1729, 2718, 31415]):
                    # The frozen helper's first ranking input is now DINO change.
                    old_name = "pixel_unknown" if method == "feature_unknown" else "pixel_random"
                    score = helper["ranking_scores"](old_name, change, change, unknown, gate, seed, pair["id"])
                    for budget in (4096, 8192, 16384, 32768):
                        key = seed, budget
                        if method == "feature_random" and key in common:
                            packet, ack, prediction, detail = common[key]
                        else:
                            packet, ack, prediction, detail = helper["transact"](prior, current, method, budget, score, gate)
                            if method == "feature_random":
                                common[key] = packet, ack, prediction, detail
                        variant = method + (f"_{seed}" if seed else "")
                        transaction = folder / f"heldout_{heldout}" / f"{variant}_B{budget}"
                        transaction.mkdir(parents=True, exist_ok=True)
                        (transaction / "current.bin").write_bytes(packet)
                        (transaction / "ack.bin").write_bytes(ack)
                        records.append(dict(pair_id=pair["id"], heldout=heldout, method=method, seed=seed, budget=budget,
                            packet_path=(transaction / "current.bin").relative_to(ROOT).as_posix(),
                            ack_path=(transaction / "ack.bin").relative_to(ROOT).as_posix(),
                            initial_down_bytes=initial_down, initial_ack_bytes=initial_ack,
                            before_sha256=pair["files"]["before"]["sha256"], after_sha256=pair["files"]["after"]["sha256"],
                            config_sha256=spec["config_sha256"], source_bundle_sha256=spec["source_bundle_sha256"],
                            backbone_weight_sha256=spec["encoder"]["weight_sha256"],
                            known_model_sha256=spec["known_model_sha256"][str(heldout)],
                            score_file_sha256=spec["score_files_sha256"][f"{heldout}:{pair['id']}"],
                            router_uses_unknown_model=method == "feature_unknown", unknown_components_8=components,
                            **detail, **helper["image_metrics"](current, prediction, known_mask, unknown_mask)))
        completed.append(pair["id"])
        helper["save_csv"](OUT / "records.csv", records)
        helper["save_csv"](OUT / "sanity.csv", sanity)
        summarize(helper, records, baseline, completed)
        print(f"{len(completed)}/64 pair={pair['id']} new_transactions={len(records)} budget/replay pass", flush=True)
    comparisons = summarize(helper, records, baseline, completed)
    if args.limit == 8 and len(completed) == 8:
        for name in ("records.csv", "summary.csv", "comparisons.csv", "sanity.csv", "progress.json"):
            shutil.copyfile(OUT / name, OUT / ("pilot8_" + name))
    print(json.dumps(dict(completed_pairs=len(completed), records=len(records), config_sha256=spec["config_sha256"],
        comparisons_30=[{key: value for key, value in row.items() if key not in (
            "delta_unknown_coverage_25", "paired_bootstrap_low_25", "paired_bootstrap_high_25",
            "delta_unknown_coverage_35", "paired_bootstrap_low_35", "paired_bootstrap_high_35")} for row in comparisons]), indent=2))


if __name__ == "__main__":
    main()
