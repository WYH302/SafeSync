"""One fixed DINO-change/unknown mixing correction, evaluated from real packets.

The previous 6912-row run and its evaluator are read-only dependencies. Only
3072 new feature_unknown/feature_random transactions receive inference. Three
old verified baselines are reused by exact record/model/data bindings.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from scripts import evaluate_second_receipts as common
from safesync.features import FrozenEncoder
from safesync.second_ground import ENCODER_CONFIG, SecondGroundEvaluator, sha_file

SCHEMA = "SECOND-feature-mix-receiver-v1"
EXPECTED = 3072
SOURCE = ROOT / "results/second_pixels/feature_mix_v1"
OUT = ROOT / "results/second_pixel_tasks/feature_mix_v1"
OLD = ROOT / "results/second_pixel_tasks/fixed_v1"
BASELINES = ("feature_change", "unknown_only", "full_jpeg2000")
COMPARATORS = ("feature_random",) + BASELINES
SOURCE_FILES = ("scripts/evaluate_second_feature_mix.py", "scripts/evaluate_second_receipts.py",
                "safesync/second_ground.py", "safesync/features.py", "safesync/pixel_wire.py", "safesync/data.py")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def keys(pair_id, config):
    return {(pair_id, h, method, seed, budget) for h in config["heldouts"]
            for method in ("feature_unknown", "feature_random")
            for seed in (config["random_seeds"] if method == "feature_random" else [0])
            for budget in config["budgets_total_bytes"]}


def read_new_source(producer):
    path = SOURCE / "records.csv"
    if not path.exists():
        return [], [], {"source_rows": 0, "complete_pair_prefix": 0, "csv_sha256": None}
    blob = path.read_bytes()
    finished_line = blob.endswith(b"\n")
    if not finished_line:
        blob = blob[:blob.rfind(b"\n") + 1]
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
    fields, rows = reader.fieldnames, list(reader)
    if not fields or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("incomplete or malformed complete source CSV row")
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = common.row_key(row)
        if key in seen or row["pair_id"] not in producer["validation_pair_ids"]:
            raise ValueError("duplicate or unexpected new transaction")
        seen.add(key)
        groups[row["pair_id"]].append(row)
    complete = []
    for pair in producer["validation_pair_ids"]:
        group = groups[pair]
        if {common.row_key(r) for r in group} != keys(pair, producer["config"]):
            break
        complete.append((pair, group))
    return fields, complete, {"source_rows": len(rows), "complete_pair_prefix": len(complete),
                              "csv_sha256": common.digest(blob), "complete_final_line": finished_line}


def old_bindings():
    paths = [OLD / name for name in ("records.csv", "progress.json", "artifact_check.json", "run_spec.json")]
    paths += sorted((OLD / "source_snapshot").glob("*"))
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths if p.is_file()}


def prepare():
    producer = json.loads((SOURCE / "run_spec.json").read_text(encoding="utf-8"))
    old_spec = json.loads((OLD / "run_spec.json").read_text(encoding="utf-8"))
    marker = json.loads((OLD / "progress.json").read_text(encoding="utf-8"))
    check = json.loads((OLD / "artifact_check.json").read_text(encoding="utf-8"))
    if marker["status"] != "complete" or marker["current_records"] != 6912 or check["status"] != "pass":
        raise ValueError("old baseline run is not already complete and verified")
    if sha_file(OLD / "records.csv") != marker["records_csv_sha256"]:
        raise ValueError("old verified task records changed")
    old_producer = old_spec["producer_spec"]
    if producer["config"]["methods"] != ["feature_unknown", "feature_random"]:
        raise ValueError("only the precommitted feature_unknown/feature_random correction is allowed")
    if producer["config"]["feature_rank_weight"] != 0.75 or producer["config"]["other_rank_weight"] != 0.25:
        raise ValueError("the precommitted 0.75/0.25 weights must not be scanned or changed")
    controls = ("heldouts", "budgets_total_bytes", "block", "coarse_side", "roi_codec", "roi_quality",
                "quality_gate_psnr", "random_seeds", "coverage_psnr")
    for key in controls:
        if producer["config"][key] != old_producer["config"][key]:
            raise ValueError(f"the single control-variable correction changed {key}")
    for key in ("validation_pair_ids", "manifest_sha256", "encoder", "known_model_sha256", "score_files_sha256"):
        if producer[key] != old_producer[key]:
            raise ValueError(f"new source differs from fixed old data/model/score identity: {key}")
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha_file(manifest_path) != producer["manifest_sha256"] or any(p["source_split"] != "train" for p in manifest["pairs"]):
        raise ValueError("requires unchanged 128/64 official-training development data")
    items = {p["id"]: p for p in manifest["pairs"] if p["split"] == "validation"}
    if list(items) != producer["validation_pair_ids"] or len(items) != 64:
        raise ValueError("fixed validation IDs changed")
    _, old_rows = read_csv(OLD / "records.csv")
    old_rows = [r for r in old_rows if r["method"] in BASELINES]
    if len(old_rows) != 2304:
        raise ValueError("requires exactly 2304 already verified baseline task rows")
    models = old_spec["ground_models"]
    for h, model in models.items():
        folder = ROOT / f"resources/second_ground_eval/heldout_{h}"
        if sha_file(folder / "head.npz") != model["head_sha256"] or sha_file(folder / "metadata.json") != model["metadata_sha256"]:
            raise ValueError("a frozen head/threshold metadata changed")
    spec = {"schema": SCHEMA, "task_evaluation_schema": common.SCHEMA,
            "expected_records": EXPECTED, "expected_pairs": 64,
            "producer_run_spec_sha256": sha_file(SOURCE / "run_spec.json"), "producer_spec": producer,
            "ground_models": models, "old_read_only_bindings": old_bindings(),
            "evaluation_source_sha256": {p: sha_file(ROOT / p) for p in SOURCE_FILES},
            "baseline_methods": list(BASELINES), "baseline_rows_reused_without_inference": len(old_rows),
            "controlled_change": "0.75 rank(DINO feature change) + 0.25 rank(unknown or fixed-seed random)",
            "input": "actual initial/current packet replay and RGB-only frozen public API; no sender full-current features",
            "task_mask": old_spec["task_mask"], "unknown_quality_metric": producer["quality_metric"],
            "AP": old_spec["AP"], "undefined_metrics": "null JSON / empty CSV",
            "statistics": "random3seed confusion counts averaged inside original pair; paired original-pair bootstrap2000, seed20261028",
            "claim_scope": "one targeted development correction; all 3 folds and 4 caps reported; no final-test or geographic independence claim",
            "official_test_pixels_read": 0}
    path = OUT / "run_spec.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != spec:
        raise ValueError("refusing to mix changed sources/models or a second correction under this run")
    common.save_json(path, spec)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, snapshot / name.replace("/", "__"))
    common.save_csv(OUT / "baseline_reference.csv", old_rows)
    return producer, items, old_rows, spec


def pair_means(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["pair_id"], int(row["heldout"]), int(row["budget"]), row["method"])].append(row)
    fields = ["task_tp", "task_fp", "task_fn", "task_tn", "task_valid_pixels", "task_positive_pixels",
              "total_bytes", "down_bytes", "ack_bytes", "unknown_pixels", "known_pixels", "known_error_sum"]
    fields += [f"unknown_retained_pixels_{q}" for q in (25, 30, 35)]
    output = []
    for (pair, h, budget, method), rows in sorted(groups.items()):
        if len(rows) != (3 if method == "feature_random" else 1):
            raise ValueError("incomplete random seed or baseline group")
        values = {k: float(np.mean([float(r[k]) for r in rows])) for k in fields}
        aps = [float(r["task_ap"]) for r in rows if r["task_ap"] not in (None, "")]
        output.append({"pair_id": pair, "heldout": h, "budget": budget, "method": method,
                       "seed_count": len(rows), **values,
                       "mean_seed_exact_AP": float(np.mean(aps)) if aps else None})
    return output


def aggregate(new_rows, baseline_rows, config):
    pair_ids = {r["pair_id"] for r in new_rows}
    means = pair_means(new_rows + [r for r in baseline_rows if r["pair_id"] in pair_ids])
    common.save_csv(OUT / "pair_means.csv", means)
    groups = defaultdict(list)
    for row in means:
        groups[(row["heldout"], row["budget"], row["method"])].append(row)
    summaries, comparisons = [], []
    for (h, budget, method), rows in sorted(groups.items()):
        sums = {k: sum(r[k] for r in rows) for k in ("task_tp", "task_fp", "task_fn", "task_tn",
            "task_valid_pixels", "task_positive_pixels", "unknown_pixels", "known_pixels", "known_error_sum")}
        tp, fp, fn = (sums[k] for k in ("task_tp", "task_fp", "task_fn"))
        aps = [r["mean_seed_exact_AP"] for r in rows if r["mean_seed_exact_AP"] is not None]
        result = {"heldout": h, "budget": budget, "method": method, "pairs": len(rows),
                  "task_positive_pairs": sum(r["task_positive_pixels"] > 0 for r in rows),
                  "unknown_positive_pairs": sum(r["unknown_pixels"] > 0 for r in rows), **sums,
                  "task_micro_f1": common.ratio(2 * tp, 2 * tp + fp + fn),
                  "task_micro_iou": common.ratio(tp, tp + fp + fn),
                  "mean_positive_pair_exact_AP": float(np.mean(aps)) if aps else None,
                  "mean_actual_total_bytes": float(np.mean([r["total_bytes"] for r in rows])),
                  "source_known_mse": common.ratio(sums["known_error_sum"], sums["known_pixels"]),
                  "task_inference_origin": "new actual RGB" if method.startswith("feature_") and method != "feature_change" else "reused verified fixed_v1"}
        for q in (25, 30, 35):
            result[f"unknown_coverage_{q}_micro"] = common.ratio(sum(r[f"unknown_retained_pixels_{q}"] for r in rows), sums["unknown_pixels"])
        summaries.append(result)
    for h in config["heldouts"]:
        for budget in config["budgets_total_bytes"]:
            new = sorted(groups[(h, budget, "feature_unknown")], key=lambda r: r["pair_id"])
            sample = np.random.default_rng(20261028).integers(0, len(new), (2000, len(new)))
            n = np.array([[r[k] for k in ("task_tp", "task_fp", "task_fn")] for r in new])
            for baseline in COMPARATORS:
                old = sorted(groups[(h, budget, baseline)], key=lambda r: r["pair_id"])
                if [r["pair_id"] for r in new] != [r["pair_id"] for r in old]:
                    raise ValueError("original image-pair comparison is not paired")
                b = np.array([[r[k] for k in ("task_tp", "task_fp", "task_fn")] for r in old])
                for field in ("task_valid_pixels", "task_positive_pixels", "unknown_pixels"):
                    if [r[field] for r in new] != [r[field] for r in old]:
                        raise ValueError("new/reused baseline masks or positive support differ")
                delta = common.f1_array(n.sum(0)) - common.f1_array(b.sum(0))
                boot = common.f1_array(n[sample].sum(1)) - common.f1_array(b[sample].sum(1))
                low, high = common.interval(boot)
                byte_delta = np.array([a["total_bytes"] - o["total_bytes"] for a, o in zip(new, old)])
                bl, bh = common.interval(byte_delta[sample].mean(1))
                row = {"heldout": h, "budget": budget, "method": "feature_unknown", "baseline": baseline,
                       "pairs": len(new), "unknown_positive_pairs": sum(r["unknown_pixels"] > 0 for r in new),
                       "delta_task_micro_f1": float(delta) if np.isfinite(delta) else None,
                       "task_paired_bootstrap_low": low, "task_paired_bootstrap_high": high,
                       "delta_mean_actual_total_bytes": float(byte_delta.mean()),
                       "bytes_paired_bootstrap_low": bl, "bytes_paired_bootstrap_high": bh}
                denominator = np.array([r["unknown_pixels"] for r in new])
                den_boot = denominator[sample].sum(1)
                for q in (25, 30, 35):
                    difference = np.array([a[f"unknown_retained_pixels_{q}"] - o[f"unknown_retained_pixels_{q}"] for a, o in zip(new, old)])
                    ub = np.divide(difference[sample].sum(1), den_boot, out=np.full(2000, np.nan), where=den_boot > 0)
                    ul, uh = common.interval(ub)
                    row.update({f"delta_unknown_coverage_{q}": common.ratio(difference.sum(), denominator.sum()),
                                f"unknown_{q}_paired_bootstrap_low": ul, f"unknown_{q}_paired_bootstrap_high": uh,
                                f"unknown_{q}_undefined_bootstrap_replicates": int((~np.isfinite(ub)).sum())})
                row["scope"] = "same byte cap; actual bytes differ; original-pair developmental bootstrap, random seeds averaged within pair"
                comparisons.append(row)
    common.save_csv(OUT / "summary.csv", summaries)
    common.save_csv(OUT / "comparisons.csv", comparisons)
    return summaries, comparisons


def verify_artifacts(spec):
    fields, tasks = read_csv(OUT / "records.csv")
    source_fields, grouped, source_info = read_new_source(spec["producer_spec"])
    source_rows = [r for _, group in grouped for r in group]
    source_map = {common.row_key(r): r for r in source_rows}
    task_map = {common.row_key(r): r for r in tasks}
    expected = set().union(*(keys(p, spec["producer_spec"]["config"]) for p in spec["producer_spec"]["validation_pair_ids"]))
    if len(tasks) != EXPECTED or set(source_map) != set(task_map) or set(task_map) != expected:
        raise ValueError("3072 new task/source transactions do not match one-to-one")
    caches = set()
    for key, row in task_map.items():
        source_row = source_map[key]
        if {k: row[k] for k in source_fields} != source_row or row["source_record_sha256"] != common.digest(common.canonical(source_row).encode()):
            raise ValueError("new task original source fields/hash differ")
        if row["decoded_current_rgb_sha256"] != source_row["receiver_sha256"]:
            raise ValueError("task decoded RGB differs from production receipt")
        if sha_file(ROOT / row["packet_path"]) != row["packet_sha256"] or sha_file(ROOT / row["ack_path"]) != row["ack_sha256"]:
            raise ValueError("new packet or ACK hash changed")
        if (ROOT / row["packet_path"]).stat().st_size != int(row["down_bytes"]) or (ROOT / row["ack_path"]).stat().st_size != int(row["ack_bytes"]):
            raise ValueError("new actual packet/ACK byte lengths changed")
        tp, fp, fn, tn = (int(row[k]) for k in ("task_tp", "task_fp", "task_fn", "task_tn"))
        if tp + fp + fn + tn != int(row["task_valid_pixels"]) or tp + fn != int(row["task_positive_pixels"]):
            raise ValueError("invalid native-pixel confusion count")
        if (float(row["task_f1"]) if row["task_f1"] else None) != common.ratio(2 * tp, 2 * tp + fp + fn):
            raise ValueError("F1 or its null rule differs")
        cache_key = row["task_cache_key"]
        if cache_key not in caches:
            cache = json.loads((OUT / "cache" / f"{cache_key}.json").read_text(encoding="utf-8"))
            if cache_key != common.digest(common.canonical(cache["binding"]).encode()) or sha_file(ROOT / row["task_tokens_path"]) != row["task_tokens_sha256"]:
                raise ValueError("new complete score cache binding or token SHA changed")
            caches.add(cache_key)
    for path, checksum in spec["old_read_only_bindings"].items():
        if sha_file(ROOT / path) != checksum:
            raise ValueError("old 6912-row results, evaluator snapshot, or completion metadata changed")
    for path, checksum in spec["evaluation_source_sha256"].items():
        if sha_file(ROOT / path) != checksum:
            raise ValueError("evaluation source changed during the fixed correction")
    for h, model in spec["ground_models"].items():
        folder = ROOT / f"resources/second_ground_eval/heldout_{h}"
        if sha_file(folder / "head.npz") != model["head_sha256"] or sha_file(folder / "metadata.json") != model["metadata_sha256"]:
            raise ValueError("frozen model/head/threshold metadata changed")
    if sha_file(SOURCE / "run_spec.json") != spec["producer_run_spec_sha256"]:
        raise ValueError("new producer run spec changed")
    _, summary = read_csv(OUT / "summary.csv")
    _, comparisons = read_csv(OUT / "comparisons.csv")
    _, baselines = read_csv(OUT / "baseline_reference.csv")
    if len(summary) != 60 or len(comparisons) != 48 or len(baselines) != 2304:
        raise ValueError("missing fold/budget/method/baseline cells")
    result = {"schema": SCHEMA, "status": "pass", "utc": common.now(), "new_records": EXPECTED,
              "reused_old_baseline_rows": len(baselines), "unique_new_score_caches": len(caches),
              "all_original_source_rows_and_packet_ACK_RGB_hashes_match": True,
              "old_run_records_scripts_snapshots_unchanged": True, "frozen_models_unchanged": True,
              "summary_cells": len(summary), "comparison_cells": len(comparisons),
              "records_csv_sha256": sha_file(OUT / "records.csv"), "source_snapshot": source_info,
              "official_test_pixels_read": 0}
    common.save_json(OUT / "artifact_check.json", result)
    return result


def write_report(summaries, comparisons):
    lines = ["# SECOND 一次 Feature-change/Unknown 控制变量修正", "",
             "3072条新真实码流已全部评价；旧2304条已核验基线直接复用，原6912条及源码/快照未变。模型、阈值、数据、codec、gate、预算不变，没有扫描混合权重或新增Unknown算法。", "",
             "这里只修正到DINO feature-change作为混合底分：0.75 rank(c)+0.25 rank(u/random)。仍是开发验证，不是final test；以下完整报告所有三折四预算。random3seed先在原pair内平均混淆计数，CI是2000次配对原pair百分位bootstrap，没有地理独立性或多重比较保证。", "",
             "| heldout | budget | method | task F1 | U25 | U30 | U35 | mean actual bytes |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "null" if value is None else f"{value:.6f}"
    for r in summaries:
        lines.append(f"| {r['heldout']} | {r['budget']} | {r['method']} | {fmt(r['task_micro_f1'])} | {fmt(r['unknown_coverage_25_micro'])} | {fmt(r['unknown_coverage_30_micro'])} | {fmt(r['unknown_coverage_35_micro'])} | {r['mean_actual_total_bytes']:.2f} |")
    lines += ["", "完整48个比较及U25/U30/U35各自的配对bootstrap区间见comparisons.csv。分数保存为实际RGB公开API同次forward捕获的精确float32[16,16] token；每条AP精确，汇总AP为positive-pair平均，不冒充pooled pixel AP。mask沿用固定ground task，语义0仅作GT_CD真实no-change负例。", "",
              "实际预算上限相同，但字节有差异，不能自动称为相同实际码率。没有根据结果追加非劣容差。无正样本/零分母均为null。数据与模型、来源行、packet/ACK、receiver RGB和token SHA绑定见run_spec.json、records.csv、artifact_check.json。", "",
              "复现：`.venv/Scripts/python.exe -X utf8 -u scripts/evaluate_second_feature_mix.py --follow`；只检查保存产物：同命令使用`--verify`。旧基线不重新推模型；中断后缓存仍先核验新packet重放。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("poll wait must be 1..60 seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        spec = json.loads((OUT / "run_spec.json").read_text(encoding="utf-8"))
        print(json.dumps(verify_artifacts(spec), indent=2), flush=True)
        return
    common.save_json(OUT / "progress.json", {"schema": SCHEMA, "status": "in_progress", "expected_records": EXPECTED,
                     "current_records": 0, "utc": common.now(), "official_test_pixels_read": 0})
    while not (SOURCE / "run_spec.json").exists():
        if not args.follow:
            print("new producer spec not yet available; no source or baseline output changed", flush=True)
            return
        time.sleep(args.poll_seconds)
    producer, items, baselines, spec = prepare()
    encoder = FrozenEncoder(ROOT, ENCODER_CONFIG)
    inferences = {h: common.ActualRGBInference(SecondGroundEvaluator(ROOT, h, encoder=encoder)) for h in (5, 3, 4)}
    results, completed = [], set()
    old_by_pair_h = {(r["pair_id"], int(r["heldout"])): r for r in baselines}
    while True:
        fields, groups, source_info = read_new_source(producer)
        for pair_id, rows in groups:
            if pair_id in completed:
                continue
            item = items[pair_id]
            prior, initial = common.initialize_pair(SOURCE, item)
            before = Image.fromarray(prior.rgb)
            targets = {h: common.offline_target(item, h) for h in inferences}
            for h, target in targets.items():
                reference = old_by_pair_h[(pair_id, h)]
                for key, value in target[2].items():
                    if reference[key] != value:
                        raise ValueError("offline new/old target or task mask differs")
                if initial["initial_packet_sha256"] != reference["initial_packet_sha256"]:
                    raise ValueError("new/old initial received state packet differs")
            for row in rows:
                if row["before_sha256"] != item["files"]["before"]["sha256"] or row["after_sha256"] != item["files"]["after"]["sha256"]:
                    raise ValueError("new production source image identity differs")
                rgb = common.replay_record(row, prior, initial, producer)
                results.append(common.evaluate_record(row, rgb, before, initial,
                    targets[int(row["heldout"])], inferences[int(row["heldout"])], OUT, spec))
            completed.add(pair_id)
            common.save_csv(OUT / "records.csv", results, fields + list(common.TASK_FIELDS))
            aggregate(results, baselines, producer["config"])
            common.save_json(OUT / "progress.json", {"schema": SCHEMA, "status": "in_progress", "expected_records": EXPECTED,
                "current_records": len(results), "completed_pairs": len(completed), "expected_pairs": 64,
                "source_snapshot": source_info, "records_csv_sha256": sha_file(OUT / "records.csv"),
                "official_test_pixels_read": 0, "utc": common.now()})
            print(f"feature-mix actual receiver {len(completed)}/64 pairs, {len(results)}/{EXPECTED} new tasks; baseline IDs/masks/model/init matched", flush=True)
        if len(results) == EXPECTED:
            summaries, comparisons = aggregate(results, baselines, producer["config"])
            check = verify_artifacts(spec)
            write_report(summaries, comparisons)
            common.save_json(OUT / "progress.json", {"schema": SCHEMA, "status": "complete", "expected_records": EXPECTED,
                "current_records": len(results), "completed_pairs": 64, "expected_pairs": 64,
                "source_snapshot": check["source_snapshot"], "records_csv_sha256": check["records_csv_sha256"],
                "artifact_check_sha256": sha_file(OUT / "artifact_check.json"), "reused_verified_baseline_rows": 2304,
                "all_source_rows_matched": True, "all_packet_ACK_receiver_hashes_match": True,
                "old_run_unchanged": True, "ground_models": spec["ground_models"], "official_test_pixels_read": 0,
                "utc": common.now()})
            print("COMPLETE3072 new actual RGB tasks,2304 old verified baseline rows reused, all60 summary/48 comparison cells", flush=True)
            break
        if not args.follow:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
