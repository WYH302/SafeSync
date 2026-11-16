"""Post-confirmation gate30 full-history control, using the frozen RGB adapter.

This does not modify the original transmission/task scripts, specifications,
streams, or results. Gate30 was added after inspecting the initial confirmation.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.evaluate_layered_receipts as adapter
import numpy as np

OUT = ROOT / "results/full_cache_tasks/gate30"
INPUT = ROOT / "results/full_cache_controls/gate30"
MAIN = ROOT / "results/layered_tasks/confirmation_v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
FROZEN_ADAPTER_SHA = "f12572987ef30a3e1822a1d528fcef904f14605afae7bf34d3f3ebede887eaec"
LABELS = ["gaussian_20261029", "full_cache_gate25", "full_cache_gate30", "full_webp"]
BUDGETS = [32768, 65536, 131072]


def evaluate_area(index):
    if adapter.sha(ROOT / "scripts/evaluate_layered_receipts.py") != FROZEN_ADAPTER_SHA:
        raise ValueError("the original task adapter changed")
    source, target = INPUT / f"area_{index}", OUT / f"area_{index}"
    manifest, source_spec = adapter.read(MANIFEST), adapter.read(source / "run_spec.json")
    if source_spec["area_indices"] != [index]:
        raise ValueError("additional control source has a different AOI")
    control = {"schema": "post-confirmation-gate30-ground-task-v1", "control_gate_psnr": 30,
        "additional_control": True, "part_of_original_blind_confirmation": False,
        "history": "added after the initial eight-AOI gate25 confirmation because the original full-cache control often used less than its cap",
        "aoi_id": manifest["aois"][index]["aoi_id"], "area_index": index,
        "methods": ["full_cache_mse"], "budgets": BUDGETS,
        "model_and_threshold_changes": False, "task_thresholds": {"building": .5, "event": .9610000252723694},
        "weight_sha256": "8135967879f5e42635508251739e2e9e30e4a58e3abc5df9875340ebc0be7081",
        "entrypoint": adapter.relative(Path(__file__)), "entrypoint_sha256": adapter.sha(Path(__file__)),
        "frozen_adapter_sha256": FROZEN_ADAPTER_SHA, "source_spec_path": adapter.relative(source / "run_spec.json"),
        "source_spec_sha256": adapter.sha(source / "run_spec.json"), "manifest_sha256": adapter.sha(MANIFEST),
        "original_confirmation_summary_sha256": adapter.sha(MAIN / "summary.json"),
        "predict_input": "actual decoded LB01/LA01 receiver RGB only; labels/valid masks offline"}
    metadata = target / "additional_control_task_spec.json"
    if metadata.exists() and adapter.read(metadata) != control:
        raise ValueError("the additional control task specification changed")
    adapter.save_json(metadata, control)
    original_loader = adapter.load_run

    def checked_loader(run, manifest_view, budgets, allowed_indices):
        actual = adapter.read(run / "run_spec.json")
        config = actual["method_configs"]["full_cache_mse"]
        if actual["schema"] != "additional-full-cache-control-v1" or actual.get("control_gate_psnr") != 30:
            raise ValueError("this is not the explicitly additional gate30 control")
        if config["gate_psnr"] != 30 or actual["thresholds"]["gate_psnr"] != 30:
            raise ValueError("the actual detail/reference gate is not 30 dB")
        if actual["methods"] != ["full_cache_mse"] or actual["budgets"] != BUDGETS:
            raise ValueError("additional control methods/budgets changed")
        if actual["teacher_building_threshold"] != .5 or actual["teacher_event_threshold"] != .961:
            raise ValueError("the frozen teacher thresholds differ")
        # The original loader checks completion, all source hashes/snapshots,
        # manifest identity and fixed real month order. Only after those checks,
        # extend its in-memory file-list view with our own evaluation provenance.
        # This view is never written to the original transmission run_spec.
        verified, rows = original_loader(run, manifest_view, budgets, allowed_indices)
        evaluation_view = copy.deepcopy(verified)
        for path in [Path(__file__), metadata]:
            evaluation_view["source_sha256"][adapter.relative(path)] = adapter.sha(path)
        return evaluation_view, rows

    adapter.METHODS = ["full_cache_mse"]
    adapter.load_run = checked_loader
    adapter.OLD_CACHES.extend(MAIN / f"area_{i}" / "probabilities" for i in range(8))
    adapter.OLD_CACHES.extend(OUT / f"area_{i}" / "probabilities" for i in range(index))
    sys.argv = [str(__file__), "--confirmation", "--manifest", str(MANIFEST), "--runs", str(source), "--out", str(target)]
    adapter.main()


def read_source_groups():
    groups, hashes = {}, {}
    for index in range(8):
        controls = INPUT / f"area_{index}"
        original = ROOT / f"results/layered/confirmation_v1/area_{index}"
        for folder, additional in [(controls, True), (original, False)]:
            path = folder / "records.csv"
            progress = adapter.read(folder / "progress.json")
            if progress["status"] != "complete" or progress["records_sha256"] != adapter.sha(path):
                raise ValueError("comparison source is incomplete or changed")
            hashes[adapter.relative(path)] = adapter.sha(path)
            for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
                method = row["method"]
                label = "full_cache_gate30" if additional else "full_cache_gate25" if method == "full_cache_mse" else method
                if label in LABELS:
                    groups.setdefault((label, int(row["budget"]), row["aoi"]), []).append(row)
    return groups, hashes


def summarize():
    manifest = adapter.read(MANIFEST)
    aids = [a["aoi_id"] for a in manifest["aois"]]
    main = adapter.read(MAIN / "summary.json")
    groups, hashes = read_source_groups()
    combined = []
    for index in range(8):
        folder = OUT / f"area_{index}"
        summary, spec, progress = [adapter.read(folder / name) for name in ["summary.json", "run_spec.json", "progress.json"]]
        if progress["status"] != "complete" or progress["summary_sha256"] != adapter.sha(folder / "summary.json"):
            raise ValueError("additional task evaluation incomplete/changed")
        if spec["model_hash"] != main["model_hash"] or spec["threshold_hash"] != main["threshold_hash"]:
            raise ValueError("additional ground model identity differs")
        if spec["source_hashes"][adapter.relative(Path(__file__))] != adapter.sha(Path(__file__)):
            raise ValueError("additional evaluation wrapper changed")
        hashes.update({adapter.relative(folder / name): adapter.sha(folder / name) for name in
            ["summary.json", "records.csv", "run_spec.json", "additional_control_task_spec.json"]})
        combined.extend({**r, "label": "full_cache_gate30", "gate_psnr": 30} for r in summary["per_aoi_seed"])
    if sum(r["frames"] for r in combined) != 534:
        raise ValueError("expected all 534 additional receiver outputs")
    for row in main["per_aoi_seed"]:
        if row["method"] in ["gaussian_20261029", "full_cache_mse", "full_webp"]:
            label = "full_cache_gate25" if row["method"] == "full_cache_mse" else row["method"]
            combined.append({**row, "label": label, "gate_psnr": 25 if label != "full_webp" else None})
    for group in combined:
        rows = groups[(group["label"], group["budget"], group["aoi"])]
        if len(rows) != group["frames"]:
            raise ValueError("task and U30 comparison streams have different month counts")
        pixels = sum(int(r["new_track_psnr30_pixels"]) for r in rows)
        usable = sum(int(r["new_track_psnr30_usable_pixels"]) for r in rows)
        if pixels != group["event"]["positive_pixels"]:
            raise ValueError("same-stream U30 and task GT masks have different supports")
        if sum(int(r["down_bytes"]) + int(r["up_bytes"]) for r in rows) != group["actual_bidirectional_bytes"]:
            raise ValueError("U30 and task costs differ")
        group.update(new_track_U30_pixels=pixels, new_track_U30_usable_pixels=usable,
            new_track_U30=usable / pixels if pixels else None,
            sender_full_archive_bytes=max(int(r["sender_full_archive_bytes"]) for r in rows),
            sender_summary_bytes=max(int(r["sender_summary_bytes"]) for r in rows))
    support = {task: [aid for aid in aids if all(g[task]["positive_pixels"] > 0 and g[task]["valid_pixels"] > 0
        for g in combined if g["aoi"] == aid)] for task in ["building", "event"]}
    macro = []
    for budget in BUDGETS:
        for label in LABELS:
            members = [g for g in combined if (g["label"], g["budget"]) == (label, budget)]
            assert len(members) == 8
            item = {"method": label, "budget": budget, "geographic_AOIs": 8,
                "additional_control": label == "full_cache_gate30", "original_blind_confirmation": label != "full_cache_gate30",
                "mean_update_bidirectional_bytes": float(np.mean([g["mean_update_bidirectional_bytes"] for g in members])),
                "mean_amortized_bytes_per_update": float(np.mean([g["total_bytes_including_bootstrap"] / g["frames"] for g in members])),
                "mean_sender_full_archive_bytes": float(np.mean([g["sender_full_archive_bytes"] for g in members])),
                "mean_sender_summary_bytes": float(np.mean([g["sender_summary_bytes"] for g in members]))}
            for task in ["building", "event"]:
                selected = [g for g in members if g["aoi"] in support[task]]
                item[task + "_AOIs"] = len(selected)
                for metric in adapter.METRICS:
                    item[task + "_" + metric] = adapter.mean_defined([g[task][metric] for g in selected])
            item["new_track_U30"] = adapter.mean_defined([g["new_track_U30"] for g in members if g["aoi"] in support["event"]])
            macro.append(item)
    lookup = {(g["label"], g["budget"], g["aoi"]): g for g in combined}
    contrasts = []
    for comparator in ["gaussian_20261029", "full_webp", "full_cache_gate25"]:
        for budget in BUDGETS:
            for task in ["building", "event", "new_track_U30"]:
                ids = support["event" if task == "new_track_U30" else task]
                pairs = []
                for aid in ids:
                    a, b = lookup[("full_cache_gate30", budget, aid)], lookup[(comparator, budget, aid)]
                    av, bv = (a[task], b[task]) if task == "new_track_U30" else (a[task]["f1"], b[task]["f1"])
                    pairs.append({"aoi": aid, "gate30": av, "comparator": bv, "difference": av-bv})
                values = np.array([p["difference"] for p in pairs])
                draws = np.random.default_rng(20261029).integers(0, len(values), size=(10000, len(values)))
                ci = np.quantile(values[draws].mean(axis=1), [.05/6, 1-.05/6])
                contrasts.append({"method": "full_cache_gate30", "comparator": comparator, "budget": budget,
                    "metric": task, "difference_orientation": "additional full-cache gate30 minus comparator",
                    "difference": float(values.mean()), "CI98_333_lower": float(ci[0]), "CI98_333_upper": float(ci[1]),
                    "paired_AOIs": len(ids), "paired_AOI_ids": ids, "AOI_pairs": pairs})
    result = {"schema": "post-confirmation-full-cache-gate30-comparison-v1", "additional_outputs": 534,
        "additional_control": True, "part_of_original_blind_confirmation": False,
        "control_gate_psnr": 30, "parameter_search": False,
        "history": "one fixed control appended after original confirmation to address gate25 budget underuse",
        "source_hashes": {**hashes, adapter.relative(MAIN / "summary.json"): adapter.sha(MAIN / "summary.json"),
            adapter.relative(Path(__file__)): adapter.sha(Path(__file__))},
        "model_hash": main["model_hash"], "threshold_hash": main["threshold_hash"], "model_and_threshold_changes": False,
        "support_AOI_ids": support, "per_aoi": combined, "macro_aoi": macro, "paired_comparisons": contrasts,
        "bootstrap": "10000 paired equal-weight AOI resamples, seed20261029, nominal98.333333% per three-budget family per specified contrast/outcome; not corrected over all secondary contrasts",
        "training_overlap": main["training_overlap"], "independent_geographic_generalization_claim": False}
    adapter.save_json(OUT / "summary.json", result)
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    lines = ["# 追加完整历史 gate30 对照", "", "此对照在首轮确认结果后追加，不属于原始盲确认。仅固定一次gate30，没有继续扫门限；主方法、主3738帧、模型和任务阈值均未改变。", "",
        "| cap KiB | 方法 | 实际双向更新KiB | 含初始化摊销KiB/更新 | 建筑F1 | 事件F1 | 同码流新track U30 |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for r in macro:
        lines.append(f"| {r['budget']//1024} | {r['method']} | {r['mean_update_bidirectional_bytes']/1024:.2f} | "
            f"{r['mean_amortized_bytes_per_update']/1024:.2f} | {r['building_f1']:.4f} | {r['event_f1']:.4f} | {r['new_track_U30']:.4f} |")
    lines += ["", "表注：gate30行是追加对照；full-cache gate25为原设置。全部指标来自各自真实接收端，同一完整序列的任务GT与U30分母逐AOI核对一致。共同GT支持见summary.json；按AOI等权，月份不是独立样本。实际字节不同，不能把同cap写成严格相同码率。完整历史发送端另外保留原生补齐RGB（本设置3,145,728字节/地区），并保留同样摘要；主Gaussian不保存完整历史RGB，但其共享float64矩阵另计。", "",
        "统一任务模型为公开SN7 exp0200；按作者公开fold程序重建，8地区预计均属于该权重训练fold，原始训练日志不可用。这些是固定模型的通信失真测量，不证明任务模型地理泛化，也不证明任意未来任务安全。", "",
        "## 追加gate30 − 主Gaussian 的配对差值", "",
        "| cap KiB | 指标 | 绝对差 | 98.333%区间（三预算家族） |", "|---:|---|---:|---|"]
    for r in contrasts:
        if r["comparator"] == "gaussian_20261029":
            lines.append(f"| {r['budget']//1024} | {r['metric']} | {r['difference']:.5f} | [{r['CI98_333_lower']:.5f}, {r['CI98_333_upper']:.5f}] |")
    lines += ["", "配对区间为固定seed20261029、10,000次AOI重采样；每个指定对照/指标的三预算家族用名义98.333333%区间，不同时校正所有指标及次级对照。完整历史若更好，按此报告；不利用旧gate25少发字节制造方法优势。", ""]
    (OUT / "ADDITIONAL_CONTROL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    adapter.save_json(OUT / "progress.json", {"status": "complete", "completed_AOIs": 8, "additional_outputs": 534,
        "summary_sha256": adapter.sha(OUT / "summary.json"), "comparison_sha256": adapter.sha(OUT / "comparison.csv")})
    print("completed534 additional gate30 task outputs; original3738 unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-index", type=int, choices=range(8))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.area_index is not None:
        evaluate_area(args.area_index)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        launch = {"schema": "additional-gate30-task-launch-v1", "additional_control": True,
            "part_of_original_blind_confirmation": False, "control_gate_psnr": 30,
            "expected_outputs": 534, "entrypoint_sha256": adapter.sha(Path(__file__)), "adapter_sha256": FROZEN_ADAPTER_SHA,
            "model_changes": False, "threshold_changes": False, "AOI_order": list(range(8)), "budgets": BUDGETS}
        if (OUT / "launch_spec.json").exists() and adapter.read(OUT / "launch_spec.json") != launch:
            raise ValueError("additional-control entry identity changed")
        adapter.save_json(OUT / "launch_spec.json", launch)
        for i in range(8):
            while not (INPUT / f"area_{i}" / "progress.json").exists() or adapter.read(INPUT / f"area_{i}" / "progress.json")["status"] != "complete":
                time.sleep(15)
            adapter.save_json(OUT / "progress.json", {"status": "in_progress", "completed_AOIs": i, "area": i, "expected_outputs": 534})
            command = [str(ROOT / ".venv/Scripts/python.exe"), str(Path(__file__)), "--area-index", str(i)]
            with (OUT / f"area_{i}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"area_{i} {line}", end="", flush=True)
                if process.wait():
                    raise RuntimeError("additional task failed; its log is preserved")
    summarize()


if __name__ == "__main__":
    main()
