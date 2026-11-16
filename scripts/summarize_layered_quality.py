"""Aggregate frozen layered pixel-quality records without decoding or inference.

Only areas with a valid complete marker enter estimates. Seeds are nested in
AOIs; complete-confirmation intervals resample paired geographic AOIs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/layered_quality"
SCOPES = {
    "development": ("results/layered/development_v2", "data/spacenet7_pilot/development_manifest_v1.json", [2, 3]),
    "confirmation": ("results/layered/confirmation_v1", "data/spacenet7_pilot/confirmation_manifest_v1.json", list(range(8))),
}
THRESHOLDS = (25, 30, 35)
FAMILIES = ("gaussian", "dct_bound", "full_cache_mse", "random", "full_jpeg420", "full_jpeg2000", "full_webp")
CONTEXT = {"scope", "aoi", "area_index", "method", "family", "seed", "budget", "valid_psnr_infinite"}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def average(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def psnr(value):
    return -10 * math.log10(value) if value is not None and value > 0 else None


def interval_text(row, scale=1):
    def display(value):
        value = number(value)
        return "null" if value is None else f"{value*scale:.3f}"
    return f"{display(row.get('mean_difference'))} [{display(row.get('bootstrap_low'))}, {display(row.get('bootstrap_high'))}]"


def family(method):
    return "gaussian" if method.startswith("gaussian_") else "random" if method.startswith("random_") else method


def costs(row):
    meta = json.loads(row["encoder_meta"])
    down = int(row["down_bytes"])
    if "cost" in meta:
        parts = {key: int(value) for key, value in meta["cost"].items()}
        if sum(parts.values()) != down:
            raise ValueError("Recorded layered byte components do not sum to downlink")
    elif "header_bytes" in meta and "checksum_bytes" in meta:
        parts = {"header": int(meta["header_bytes"]), "checksum": int(meta["checksum_bytes"])}
        parts["image"] = down - sum(parts.values())
    else:
        parts = {"unclassified": down}
    return {"protocol_metadata": sum(parts.get(k, 0) for k in (
                "header", "manifest", "photometry", "coarse_header", "indices", "checksum")),
            "coarse_image": parts.get("coarse_image", 0), "image_payload": parts.get("image", 0),
            "unclassified": parts.get("unclassified", 0)}


def aggregate_sequence(rows):
    first = rows[0]
    rows = sorted(rows, key=lambda row: int(row["step"]))
    result = {"scope": first["scope"], "aoi": first["aoi"], "area_index": int(first["area_index"]),
              "method": first["method"], "family": family(first["method"]),
              "seed": int(first["method"].rsplit("_", 1)[1]) if family(first["method"]) in ("gaussian", "random") else None,
              "budget": int(first["budget"]), "frames": len(rows)}
    valid_pixels, weighted_error, bad_valid_mse = 0, 0.0, 0
    for row in rows:
        pixels = int(row["all_psnr30_pixels"])
        mse = number(row["valid_mse"])
        valid_pixels += pixels
        if pixels and mse is None:
            bad_valid_mse += 1
        elif pixels:
            weighted_error += pixels * mse
    result.update(valid_pixels=valid_pixels,
                  valid_mse=weighted_error / valid_pixels if valid_pixels and not bad_valid_mse else None,
                  valid_mse_invalid_frames=bad_valid_mse)
    result["valid_psnr_db"] = psnr(result["valid_mse"])
    result["valid_psnr_infinite"] = bool(valid_pixels and result["valid_mse"] == 0)
    for region in ("all", "building", "new_track"):
        for threshold in THRESHOLDS:
            key = f"{region}_psnr{threshold}"
            pixels = sum(int(row[f"{key}_pixels"]) for row in rows)
            usable = sum(int(row[f"{key}_usable_pixels"]) for row in rows)
            errors = sum(float(row[f"{key}_error_sum"]) for row in rows)
            if not 0 <= usable <= pixels or not math.isfinite(errors):
                raise ValueError("Nonfinite or inconsistent evidence counts")
            result[f"{key}_pixels"] = pixels
            result[f"{key}_usable_pixels"] = usable
            result[f"{key}_coverage"] = ratio(usable, pixels)
            result[f"{key}_mse"] = ratio(errors, pixels)
    for field in ("valid_blocks", "stable_blocks", "updated_blocks", "deferred_blocks", "false_stable_blocks"):
        result[field] = sum(int(row[field]) for row in rows)
    if (result["stable_blocks"] + result["updated_blocks"] + result["deferred_blocks"] != result["valid_blocks"]
            or result["false_stable_blocks"] > result["stable_blocks"]):
        raise ValueError("State counters do not match their valid-block denominators")
    result["stable_ratio"] = ratio(result["stable_blocks"], result["valid_blocks"])
    result["updated_ratio"] = ratio(result["updated_blocks"], result["valid_blocks"])
    result["deferred_ratio"] = ratio(result["deferred_blocks"], result["valid_blocks"])
    result["false_confirmation_rate"] = ratio(result["false_stable_blocks"], result["stable_blocks"])
    result["false_stable_fraction_all_valid"] = ratio(result["false_stable_blocks"], result["valid_blocks"])
    bootstrap = {(int(row["bootstrap_down_bytes"]), int(row["bootstrap_up_bytes"])) for row in rows}
    if len(bootstrap) != 1:
        raise ValueError("Bootstrap changed within a frozen sequence")
    boot_down, boot_up = next(iter(bootstrap))
    down, up = sum(int(row["down_bytes"]) for row in rows), sum(int(row["up_bytes"]) for row in rows)
    if int(rows[-1]["cumulative_down_bytes"]) != boot_down + down or int(rows[-1]["cumulative_up_bytes"]) != boot_up + up:
        raise ValueError("Cumulative bytes disagree with once-counted bootstrap")
    if any(int(row["down_bytes"]) > int(row["budget"]) or int(row["up_bytes"]) != 60 for row in rows):
        raise ValueError("Frozen downlink cap or actual ACK length differs")
    parts = [costs(row) for row in rows]
    result.update(update_down_bytes=down, update_ack_bytes=up,
                  reference_up_bytes=sum(int(row["reference_up_bytes"]) for row in rows),
                  update_total_bytes=down + up, bootstrap_down_bytes=boot_down,
                  bootstrap_ack_bytes=boot_up, bootstrap_total_bytes=boot_down + boot_up,
                  sequence_down_bytes=down + boot_down, sequence_up_bytes=up + boot_up,
                  sequence_total_bytes=down + up + boot_down + boot_up,
                  update_total_bytes_mean=(down + up) / len(rows),
                  sequence_bytes_per_update=(down + up + boot_down + boot_up) / len(rows),
                  sequence_bytes_per_observation=(down + up + boot_down + boot_up) / (len(rows) + 1))
    for name in ("protocol_metadata", "coarse_image", "image_payload", "unclassified"):
        total = sum(part[name] for part in parts)
        result[f"down_{name}_bytes"] = total
        result[f"down_{name}_bytes_mean"] = total / len(rows)
    for field in ("sender_summary_bytes", "sender_metadata_bytes", "sender_full_archive_bytes", "shared_projection_matrix_bytes"):
        values = [int(row[field]) for row in rows]
        result[f"{field}_max"] = max(values)
        result[f"{field}_mean"] = float(np.mean(values))
    result["sender_serialized_summary_metadata_bytes_max"] = max(
        int(row["sender_summary_bytes"]) + int(row["sender_metadata_bytes"]) for row in rows)
    timings = [value for row in rows if (value := number(row["compute_ms"])) is not None]
    result["compute_ms_mean"] = average(timings)
    result["compute_ms_p95"] = float(np.quantile(timings, .95)) if timings else None
    return result


def seed_average(single_rows):
    groups = defaultdict(list)
    for row in single_rows:
        groups[(row["scope"], row["aoi"], row["family"], row["budget"])].append(row)
    averaged = []
    for (scope, aoi, name, budget), rows in sorted(groups.items()):
        row = dict(scope=scope, aoi=aoi, area_index=rows[0]["area_index"], method=name, family=name,
                   seed_methods=";".join(sorted(r["method"] for r in rows)), seeds=len(rows), budget=budget)
        for key in rows[0]:
            if key not in CONTEXT:
                row[key] = average([r[key] for r in rows])
        row["valid_psnr_infinite"] = any(r["valid_psnr_infinite"] for r in rows)
        if row["valid_psnr_infinite"]:
            row["valid_psnr_db"] = None
        row["seeds_with_valid_pixels"] = sum(r["valid_mse"] is not None for r in rows)
        row["seeds_with_events"] = sum(r["new_track_psnr30_coverage"] is not None for r in rows)
        row["seeds_with_STABLE"] = sum(r["false_confirmation_rate"] is not None for r in rows)
        averaged.append(row)
    return averaged


def macro_rows(rows, expected_aois):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scope"], row["method"], row["budget"])].append(row)
    results = []
    for (scope, method, budget), group in sorted(groups.items()):
        result = dict(scope=scope, method=method, budget=budget, AOIs=len(group), expected_AOIs=expected_aois)
        for key, value in group[0].items():
            if key in CONTEXT or key in ("seed_methods",):
                continue
            if isinstance(value, (int, float)) or value is None:
                result[key] = average([row[key] for row in group])
        result["valid_psnr_infinite"] = any(row["valid_psnr_infinite"] for row in group)
        if result["valid_psnr_infinite"]:
            result["valid_psnr_db"] = None
        result["AOIs_valid_MSE"] = sum(row["valid_mse"] is not None for row in group)
        result["AOIs_finite_PSNR"] = sum(row["valid_psnr_db"] is not None for row in group)
        result["AOIs_events"] = sum(row["new_track_psnr30_coverage"] is not None for row in group)
        result["AOIs_STABLE"] = sum(row["false_confirmation_rate"] is not None for row in group)
        result["stable_blocks_seedmean_sum"] = sum(row["stable_blocks"] for row in group)
        result["valid_blocks_seedmean_sum"] = sum(row["valid_blocks"] for row in group)
        result["false_stable_blocks_seedmean_sum"] = sum(row["false_stable_blocks"] for row in group)
        results.append(result)
    return results


def paired_comparison(rows, complete, scope):
    indexed = {(r["aoi"], r["method"], r["budget"]): r for r in rows}
    comparisons, individual = [], []
    metrics = ("new_track_psnr25_coverage", "new_track_psnr30_coverage", "new_track_psnr35_coverage",
               "valid_psnr_db", "valid_mse", "update_total_bytes_mean", "sequence_bytes_per_update")
    for budget in sorted({r["budget"] for r in rows}):
        aois = sorted({r["aoi"] for r in rows if r["method"] == "gaussian" and r["budget"] == budget},
                      key=lambda aoi: indexed[(aoi, "gaussian", budget)]["area_index"])
        for metric in metrics:
            values = []
            baseline = []
            for aoi in aois:
                new = indexed[(aoi, "gaussian", budget)].get(metric)
                old = indexed.get((aoi, "full_webp", budget), {}).get(metric)
                delta = new - old if new is not None and old is not None else None
                individual.append(dict(scope=scope, aoi=aoi, area_index=indexed[(aoi, "gaussian", budget)]["area_index"],
                    budget=budget, metric=metric, method="gaussian", comparator="full_webp", gaussian=new, webp=old, difference=delta))
                if delta is not None:
                    values.append(delta)
                    baseline.append(old)
            low = high = None
            if complete and scope == "confirmation" and values:
                rng = np.random.default_rng(20261029)
                samples = rng.integers(0, len(values), (10000, len(values)))
                bootstrap = np.asarray(values)[samples].mean(1)
                low, high = map(float, np.quantile(bootstrap, [.05 / 3 / 2, 1 - .05 / 3 / 2]))
            difference, old_mean = average(values), average(baseline)
            comparisons.append(dict(scope=scope, method="gaussian", comparator="full_webp", budget=budget, metric=metric,
                paired_AOIs=len(values), mean_difference=difference, baseline_paired_mean=old_mean,
                gaussian_paired_mean=old_mean + difference if difference is not None else None,
                confidence=1 - .05 / 3, bootstrap_low=low, bootstrap_high=high,
                relative_coverage_change=difference / old_mean if difference is not None and old_mean and "coverage" in metric else None,
                relative_unusable_reduction=difference / (1 - old_mean) if difference is not None and old_mean is not None and old_mean < 1 and "coverage" in metric else None,
                inference_status="complete_preselected_3budget_intervals" if complete and scope == "confirmation" else "development_descriptive" if scope == "development" else "waiting_for_complete_confirmation"))
    return comparisons, individual


def collect(scope):
    input_root, manifest_relative, areas = SCOPES[scope]
    manifest_path = ROOT / manifest_relative
    manifest = json.loads(manifest_path.read_text("utf-8"))
    specs = {area: json.loads((ROOT / input_root / f"area_{area}/run_spec.json").read_text("utf-8"))
             for area in areas if (ROOT / input_root / f"area_{area}/run_spec.json").exists()}
    if not specs:
        raise ValueError(f"No frozen specification for {scope}")
    canonical = next(iter(specs.values()))
    methods, budgets = canonical["methods"], canonical["budgets"]
    expected = sum(len(manifest["aois"][i]["months"]) - 1 for i in areas) * len(methods) * len(budgets)
    all_rows, provenance, pending, errors = [], [], [], []
    for area in areas:
        folder = ROOT / input_root / f"area_{area}"
        aoi = manifest["aois"][area]
        updates = len(aoi["months"]) - 1
        wanted = updates * len(methods) * len(budgets)
        entry = dict(area_index=area, aoi=aoi["aoi_id"], expected_records=wanted)
        if area not in specs:
            entry["status"] = "missing_spec"
            provenance.append(entry); pending.append(area)
            continue
        spec = specs[area]
        if (spec["manifest_sha256"] != file_hash(manifest_path) or spec["methods"] != methods
                or spec["budgets"] != budgets or spec["source_sha256"] != canonical["source_sha256"]
                or spec["area_indices"] != [area] or spec["limit_months"] != 0
                or spec["aoi_ids"] != [a["aoi_id"] for a in manifest["aois"]]):
            raise ValueError("Frozen per-area scope/spec differs")
        entry.update(spec_sha256=file_hash(folder / "run_spec.json"), source_sha256=spec["source_sha256"])
        progress_path = folder / "progress.json"
        if not progress_path.exists():
            entry["status"] = "not_started"
            provenance.append(entry); pending.append(area)
            continue
        marker_body = progress_path.read_bytes()
        marker = json.loads(marker_body)
        entry.update(marker_sha256=digest(marker_body), marker=marker, status=marker["status"])
        if marker["status"] != "complete":
            provenance.append(entry); pending.append(area)
            continue
        body = (folder / "records.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8"))))
        entry["records_sha256"] = digest(body)
        if (marker.get("spec_sha256") != entry["spec_sha256"] or marker.get("records_sha256") != digest(body)
                or marker["expected"] != wanted or marker["records"] != wanted or len(rows) != wanted):
            entry["status"] = "invalid_complete_marker"
            errors.append(f"area {area}: complete marker/hash/count differs")
            provenance.append(entry)
            continue
        keys = [(row["method"], int(row["budget"]), int(row["step"])) for row in rows]
        intended = {(method, budget, t) for method in methods for budget in budgets for t in range(1, updates + 1)}
        if set(keys) != intended or len(set(keys)) != wanted:
            raise ValueError("Complete area has duplicate or missing frozen method/frame keys")
        if any(row["aoi"] != aoi["aoi_id"] or row["month"] != aoi["months"][int(row["step"])]["month"] for row in rows):
            raise ValueError("Complete records differ from frozen real-month order")
        all_rows.extend(rows)
        provenance.append(entry)
    groups = defaultdict(list)
    for row in all_rows:
        groups[(row["aoi"], row["method"], int(row["budget"]))].append(row)
    single = [aggregate_sequence(rows) for key, rows in sorted(groups.items())]
    averaged = seed_average(single)
    complete = not pending and not errors and len(all_rows) == expected
    result = dict(scope=scope, status="complete" if complete else "invalid" if errors else "in_progress",
                  expected_records=expected, consumed_records=len(all_rows), expected_AOIs=len(areas),
                  complete_AOIs=len({row["aoi"] for row in all_rows}), pending_area_indices=pending,
                  errors=errors, updates_in_frozen_manifest=sum(len(manifest["aois"][i]["months"]) - 1 for i in areas),
                  method_count=len(methods), budget_count=len(budgets), methods=methods, budgets=budgets,
                  manifest_path=manifest_relative, manifest_sha256=file_hash(manifest_path),
                  source_sha256=canonical["source_sha256"], runs=provenance)
    return result, single, averaged


def correct_unknown_counts():
    path = ROOT / "results/unknown_scaled/diagnostics.json"
    reports = json.loads(path.read_text("utf-8"))
    values = []
    for report in reports:
        model_path = ROOT / f"results/unknown_scaled/known_{report['heldout']}.npz"
        with np.load(model_path, allow_pickle=False) as model:
            labels = model["labels"].astype(int).tolist()
            if len(labels) != len(model["means"]):
                raise ValueError("Saved novelty model class inventory differs")
        counts = {int(key): int(value) for key, value in report["known_class_counts"].items()}
        if set(labels) != set(counts):
            raise ValueError("Postfit class counts differ from saved model labels")
        effective = sum(counts.values())
        values.append(dict(heldout=report["heldout"], fit_cells_attempted=report["fit_cells"],
            fit_cells_effective=effective, excluded_attempted_cells=report["fit_cells"] - effective,
            retained_class_counts=counts, saved_model_labels=labels, saved_model_sha256=file_hash(model_path)))
    save_json(OUT / "unknown_fit_counts.json", dict(diagnostics_sha256=file_hash(path), models=values,
        note="Original fit_cells is attempted input count; KnownClassDistance discards classes with fewer than five samples. Counts are the recorded postfit class counts, checked against saved NPZ labels; NPZ does not persist counts. Scores and models were not changed."))
    return values


def make_plots(folder, status, macro, individual):
    if status["status"] != "complete":
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "lines.linewidth": 1.5, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    colors = ["#0072B2", "#E69F00", "#000000", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    labels = ["Gaussian", "DCT + tail", "Full-cache MSE", "Random", "JPEG 4:2:0", "JPEG2000", "WebP"]
    plots = folder / "figures"; plots.mkdir(exist_ok=True)
    outputs = []
    for rate_field, suffix, xlabel in (("update_total_bytes_mean", "updates", "Actual update downlink + ACK (KiB/update)"),
            ("sequence_bytes_per_update", "amortized", "Actual bytes incl. bootstrap (KiB/update)")):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
        for method, color, marker, label in zip(FAMILIES, colors, markers, labels):
            rows = sorted((row for row in macro if row["method"] == method), key=lambda row: row["budget"])
            for axis, metric, scale in ((axes[0], "new_track_psnr30_coverage", 100), (axes[1], "valid_psnr_db", 1)):
                valid = [r for r in rows if r.get(metric) is not None]
                axis.plot([r[rate_field] / 1024 for r in valid], [r[metric] * scale for r in valid],
                          color=color, marker=marker, label=label, linewidth=2 if method == "gaussian" else 1.35)
                axis.set_xlabel(xlabel)
        axes[0].set_ylabel("New-ID evidence U30 (%)")
        axes[1].set_ylabel("Macro AOI PSNR (dB)")
        axes[0].set_ylim(-2, 102)
        axes[0].legend(loc="best", ncol=2)
        fig.suptitle(f"{status['scope'].capitalize()} — {status['complete_AOIs']} AOIs", fontsize=10)
        name = f"rate_quality_{suffix}"
        for extension in ("pdf", "png", "svg"):
            file = plots / f"{name}.{extension}"
            fig.savefig(file, bbox_inches="tight")
            outputs.append(file.relative_to(ROOT).as_posix())
        plt.close(fig)
    selected = [row for row in individual if row["metric"] == "new_track_psnr30_coverage"]
    if selected:
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True, constrained_layout=True)
        for axis, budget in zip(axes, status["budgets"]):
            rows = sorted((r for r in selected if r["budget"] == budget), key=lambda r: r["area_index"])
            valid = [r for r in rows if r["difference"] is not None]
            axis.axvline(0, color="#777777", linewidth=.8)
            axis.scatter([r["difference"] * 100 for r in valid], [r["area_index"] for r in valid], color="#0072B2", s=24)
            axis.set_title(f"Downlink cap {budget // 1024} KiB")
            axis.set_xlabel("Gaussian − WebP U30 (pp)")
            axis.set_yticks([r["area_index"] for r in rows], [f"AOI {r['area_index'] + 1}" for r in rows])
        axes[0].invert_yaxis()
        fig.suptitle(f"{status['scope'].capitalize()}: original AOI order", fontsize=10)
        for extension in ("pdf", "png", "svg"):
            file = plots / f"per_aoi_difference.{extension}"
            fig.savefig(file, bbox_inches="tight")
            outputs.append(file.relative_to(ROOT).as_posix())
        plt.close(fig)
    return outputs


def write_report(reports, unknown):
    text = "# 分层状态像素质量、真实字节与状态汇总\n\n"
    text += "本报告只读取冻结CSV和完成标记，不解码重跑、不重新推理任务网络、不选阈值或赢家。任务F1由独立工作流汇总。开发与确认结果分别保存，确认仅在各AOI完整标记、记录数和hash一致后纳入。未完成的地区不会作为零值补入宏平均。\n\n"
    text += "## 聚合口径\n\n每个AOI/方法/seed先累加跨月evidence usable/total，U25/30/35为累加计数之比；new-ID像素是重复月份的像素观测，不是独立建筑数。valid MSE以all_psnr30_pixels作为有效像素数加权。每seed每AOI先用加权MSE转PSNR，再在AOI内平均seed PSNR，最后对AOI等权平均dB；它不等于先合并全部像素再转PSNR。三Gaussian/三random seed先在同一AOI内平均，不把seed当新的地理样本；保留所有单seed原值。无valid或无事件的指标为null，宏平均只用有定义AOI并报告数量。\n\n"
    text += "下行cap为32/64/128 KiB，60-byte ACK另计，横轴使用实际down+ACK。bootstrap每序列只计一次；同时给出更新均值、含bootstrap每update摊销与每observation摊销。帧成本拆分协议元数据、coarse图像、ROI/全图载荷与ACK。STABLE和DEFERRED分母为至少99%有效的块观测；false confirmation分母为STABLE，0分母为null。没有STABLE时不能用零误确认宣称安全，也不能将STABLE子集结果扩大到整帧。\n\n"
    text += "## 内存数值实现说明\n\nsender_summary_bytes和sender_metadata_bytes是逻辑序列化状态，不是Python对象RSS；full_archive字段只记录数组数据字节。Gaussian与DCT均为48 B/block，DCT具体是7个q16系数/channel（42 B）加3个半精度尾范数（6 B）。不采用54 B/block估算。\n\n当前冻结运行中shared_projection_matrix_bytes实测为**2,359,296 B/seed**（12288×24×8），可跨AOI复用。float32随机draw经过含np.sqrt(3)的缩放后在当前NumPy实现中提升为float64；先前1,179,648 B估算是按float32计算，低估一半。LRU容量2最多保留**4,718,592 B矩阵数据**，单列为容量，不冒充测得总峰值RAM。这里未测Python对象、临时数组、codec工作区或进程RSS，也不覆盖旧审阅快照或修改冻结代码。\n\n"
    text += "单seed表的内存max是序列内最大值；family表平均seed最大值，macro表再平均AOI，不能把这些均值称为全局峰值。逐方法实际分量最大值另存memory_components.json。\n\n"
    text += "## 完成情况与结果\n\n"
    for scope, report in reports.items():
        status, macro = report["status"], report["macro"]
        text += f"### {scope}\n\n状态：**{status['status']}**；纳入{status['complete_AOIs']}/{status['expected_AOIs']}地区，{status['consumed_records']}/{status['expected_records']}帧。预期数从manifest的{status['updates_in_frozen_manifest']}次真实更新×{status['method_count']}方法×{status['budget_count']}预算计算。\n\n"
        if macro:
            text += "| 方法 | cap KiB | 实际down+ACK KiB/update | 含bootstrap KiB/update | U30 % | 有事件AOI | 宏PSNR dB | STABLE % | false/STABLE | DEFERRED % |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            fmt = lambda value, digits=3: "null" if value is None else f"{value:.{digits}f}"
            for row in sorted(macro, key=lambda r: (r["budget"], FAMILIES.index(r["method"]))):
                text += f"| {row['method']} | {row['budget']//1024} | {row['update_total_bytes_mean']/1024:.3f} | {row['sequence_bytes_per_update']/1024:.3f} | {fmt(None if row['new_track_psnr30_coverage'] is None else row['new_track_psnr30_coverage']*100)} | {row['AOIs_events']} | {fmt(row['valid_psnr_db'])} | {fmt(None if row['stable_ratio'] is None else row['stable_ratio']*100)} | {fmt(row['false_confirmation_rate'],6)} | {fmt(None if row['deferred_ratio'] is None else row['deferred_ratio']*100)} |\n"
            text += "\n"
        if scope == "development":
            for row in report["single"]:
                if row["method"] == "gaussian_20261029" and row["budget"] == 32768:
                    text += f"开发32 KiB，{row['aoi']} 的Gaussian仅{row['stable_blocks']}/{row['valid_blocks']}块观测为STABLE，false={row['false_stable_blocks']}；应同时保留这一低接受率事实。\n\n"
        if scope == "confirmation" and status["status"] == "complete":
            directory = OUT / Path(SCOPES[scope][0]).name
            for filename, label in (("paired_comparisons.csv", "3seed先AOI内平均"),
                                    ("paired_comparisons_single_seed.csv", "预选seed20261029，同任务F1实际stream")):
                comparisons = list(csv.DictReader((directory / filename).open(encoding="utf-8")))
                text += f"Gaussian − WebP，{label}；以下区间为预定98.333%配对AOI bootstrap区间：\n\n"
                text += "| cap KiB | U30差值pp [区间] | PSNR差值dB [区间] | 实际down+ACK均值差KiB/update | 配对AOI |\n|---|---:|---:|---:|---:|\n"
                for budget in status["budgets"]:
                    picked = {row["metric"]: row for row in comparisons if int(row["budget"]) == budget}
                    evidence, quality = picked["new_track_psnr30_coverage"], picked["valid_psnr_db"]
                    rate = picked["update_total_bytes_mean"]
                    rate_delta = number(rate["mean_difference"])
                    rate_label = "null" if rate_delta is None else f"{rate_delta/1024:.3f}"
                    text += f"| {budget//1024} | {interval_text(evidence,100)} | {interval_text(quality)} | {rate_label} | {evidence['paired_AOIs']} |\n"
                text += "\n"
        text += f"原始单seed、AOI内seed均值、AOI宏均值、逐AOI差值和来源hash见 results/layered_quality/{Path(SCOPES[scope][0]).name}/。\n\n"
    text += "## 固定推断边界\n\n主比较固定为Gaussian与WebP，不按AOI选codec。确认全部完成后，按AOI配对重采样10,000次、seed=20261029；每个指标的三预算使用98.333%区间（Bonferroni三预算家族）。这不是所有指标联合的家族保证。未完成时只保留描述性点值，区间为null；有效配对AOI数单列。开发结果不作为确认结论。所有其他固定方法与种子仍完整列出。\n\n"
    text += "质量图以真实down+ACK或包含一次bootstrap的摊销成本为横轴。全部方法共享同一个首月原始图像作为PNG初始化条件；LB01与SP03 bootstrap各自实际码流及ACK均只计一次，载荷格式不同不代表初始化信息免费。图rate_quality_updates不含初始化，rate_quality_amortized包含完整bootstrap摊销。\n\n像素质量的3seed均值不能解释为3seed任务F1已评价。任务F1只评价预选Gaussian20261029，因此F1+U30联合表必须使用per_seed_aoi.csv / macro_single_seed.csv以及paired_comparisons_single_seed.csv，保证同一实际stream。此单seed在看确认结果前已预定，不按显著性挑选，不替换3seed均值或稳健性图。compute_ms覆盖来源脚本计时段，未包含图像读取、GT构建、任务推理或进程启动，不能称为完整部署端到端耗时。\n\n"
    text += "## Unknown拟合计数修正（不改模型／评分）\n\n原diagnostics的fit_cells是尝试输入数；模型仅保留至少5单元的类别，实际有效数应对保存模型labels对应的postfit counts求和。\n\n| Heldout | attempted | effective/actual | 排除 |\n|---|---:|---:|---:|\n"
    for row in unknown:
        text += f"| {row['heldout']} | {row['fit_cells_attempted']} | {row['fit_cells_effective']} | {row['excluded_attempted_cells']} |\n"
    text += "\n后两折各1个尝试单元来自未达到模型类别支持下限的部分。NPZ不保存counts；这里把原postfit计数键与NPZ labels核对，不重新拟合模型。明细及模型／diagnostics SHA见unknown_fit_counts.json。\n\n复跑：` .venv-safesync/Scripts/python.exe scripts/summarize_layered_quality.py --scope all --plots `；等待生产完成可加`--follow`。唯一一次定向数值检查用`--self-check`，不执行额外实验或审计。\n"
    (ROOT / "docs/LAYERED_QUALITY_RESULTS.md").write_text(text, encoding="utf-8")


def self_check():
    assert ratio(0, 0) is None and average([None, float("nan"), 2.0]) == 2.0
    fixture = []
    for step, (pixels, mse) in enumerate(((10, .01), (30, .09)), 1):
        row = dict(scope="development", aoi="fixture", area_index=0, method="full_webp", budget=32768,
            step=step, valid_mse=mse, valid_blocks=2, stable_blocks=step-1, updated_blocks=3-step,
            deferred_blocks=0, false_stable_blocks=0, down_bytes=100, up_bytes=60, reference_up_bytes=0,
            bootstrap_down_bytes=20, bootstrap_up_bytes=60, cumulative_down_bytes=20+step*100,
            cumulative_up_bytes=60+step*60, sender_summary_bytes=0, sender_metadata_bytes=0,
            sender_full_archive_bytes=0, shared_projection_matrix_bytes=0, compute_ms=1,
            encoder_meta=json.dumps(dict(header_bytes=1, checksum_bytes=1)))
        for name in ("all", "building", "new_track"):
            for q in THRESHOLDS:
                n = pixels if name != "new_track" else 0
                row.update({f"{name}_psnr{q}_pixels": n, f"{name}_psnr{q}_usable_pixels": n//2,
                            f"{name}_psnr{q}_error_sum": n*mse})
        fixture.append(row)
    result = aggregate_sequence(fixture)
    assert abs(result["valid_mse"]-.07) < 1e-12
    assert result["sequence_total_bytes"] == 400 and result["sequence_bytes_per_update"] == 200
    assert result["new_track_psnr30_coverage"] is None and result["false_confirmation_rate"] == 0
    assert aggregate_sequence(fixture[:1])["false_confirmation_rate"] is None
    nested = []
    for aoi, values in (("A", [.2,.4,.6]), ("B", [.8,.8,.8])):
        for seed, value in enumerate(values):
            row = dict(result, aoi=aoi, method=f"gaussian_{seed}", family="gaussian", seed=seed,
                       new_track_psnr30_coverage=value)
            nested.append(row)
    actual = macro_rows(seed_average(nested), 2)[0]["new_track_psnr30_coverage"]
    assert abs(actual - .6) < 1e-12
    # A zero-STABLE sequence has no empirical conditional confirmation rate.
    assert ratio(0, 0) is None and ratio(0, 17) == 0
    rng = np.random.default_rng(20261029)
    boot = np.ones(8)[rng.integers(0, 8, (10000, 8))].mean(1)
    assert np.quantile(boot, .05 / 3 / 2) == 1
    print("targeted aggregation check passed: missingness, weighted MSE, seed/AOI nesting, zero-STABLE rate, paired bootstrap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "development", "confirmation"), default="all")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be 1..60")
    OUT.mkdir(parents=True, exist_ok=True)
    unknown = correct_unknown_counts()
    selected = list(SCOPES) if args.scope == "all" else [args.scope]
    previous = {}
    while True:
        reports = {}
        for scope in selected:
            status, single, averaged = collect(scope)
            macro = macro_rows(averaged, status["expected_AOIs"])
            comparisons, individual = paired_comparison(averaged, status["status"] == "complete", scope)
            fixed_seed_rows = [dict(row, method="gaussian", family="gaussian")
                               if row["method"] == "gaussian_20261029" else dict(row)
                               for row in single if row["method"] in ("gaussian_20261029", "full_webp")]
            fixed_comparisons, fixed_individual = paired_comparison(
                fixed_seed_rows, status["status"] == "complete", scope)
            for row in fixed_comparisons + fixed_individual:
                row["gaussian_seed"] = 20261029
                row["seed_aggregation"] = "one preselected actual stream; same as task-F1 evaluation"
            folder = OUT / Path(SCOPES[scope][0]).name
            folder.mkdir(exist_ok=True)
            signature = digest(json.dumps(status, sort_keys=True).encode())
            if previous.get(scope) != signature:
                save_csv(folder / "per_seed_aoi.csv", single)
                save_csv(folder / "per_aoi.csv", averaged)
                save_csv(folder / "macro_single_seed.csv", macro_rows(single, status["expected_AOIs"]))
                save_csv(folder / "macro.csv", macro)
                save_csv(folder / "paired_comparisons.csv", comparisons)
                save_csv(folder / "per_aoi_differences.csv", individual)
                save_csv(folder / "paired_comparisons_single_seed.csv", fixed_comparisons)
                save_csv(folder / "per_aoi_differences_single_seed.csv", fixed_individual)
                save_json(folder / "status.json", status)
                save_json(folder / "provenance.json", dict(summary_script_sha256=file_hash(__file__),
                    confirmation_protocol_sha256=file_hash(ROOT / "docs/LAYERED_CONFIRMATION_PROTOCOL.md"),
                    statistical_unit="AOI; seeds nested within AOI", bootstrap_seed=20261029, replicates=10000,
                    confidence=1-.05/3, inputs=status, memory_scope="serialized state and recorded array bytes, not measured RSS"))
                matrix_bytes = max((row["shared_projection_matrix_bytes_max"] for row in single), default=None)
                save_json(folder / "memory_components.json", dict(
                    state_fields="sender_summary_bytes/sender_metadata_bytes are logical serialized state, not Python RSS",
                    full_archive_field="sender_full_archive_bytes is array data only, not Python/RSS overhead",
                    method_required_matrix_array_bytes_recorded_max=matrix_bytes,
                    projection_matrix_runtime_dtype="float64 in frozen NumPy implementation",
                    projection_matrix_LRU_capacity=2,
                    matrix_data_LRU_capacity_bytes=matrix_bytes*2 if matrix_bytes is not None else None,
                    matrix_shared_across_AOIs=True, matrix_residency_actual_occupancy_not_measured=True,
                    DCT_coefficients_per_channel=7, DCT_q16_coefficient_bytes_per_block=42,
                    DCT_tail_norm_bytes_per_block=6, DCT_total_payload_bytes_per_block=48,
                    Gaussian_payload_bytes_per_block=48, process_RSS_measured=False,
                    observed_per_method_component_max={method: {
                        field: max(row[field+"_max"] for row in single if row["method"] == method)
                        for field in ("sender_summary_bytes", "sender_metadata_bytes", "sender_full_archive_bytes", "shared_projection_matrix_bytes")}
                        for method in sorted({row["method"] for row in single})},
                    transient_codec_arrays_and_object_overhead_excluded=True))
                if args.plots:
                    save_json(folder / "figures.json", make_plots(folder, status, macro, individual))
                print(f"{scope}: {status['status']} {status['consumed_records']}/{status['expected_records']} frames, {status['complete_AOIs']}/{status['expected_AOIs']} AOIs", flush=True)
                previous[scope] = signature
            reports[scope] = dict(status=status, single=single, macro=macro)
        write_report(reports, unknown)
        save_json(OUT / "status.json", {scope: report["status"]["status"] for scope, report in reports.items()})
        if all(report["status"]["status"] == "complete" for report in reports.values()):
            save_json(OUT / "artifact_manifest.json", dict(
                summary_script_sha256=file_hash(__file__),
                documentation_sha256=file_hash(ROOT / "docs/LAYERED_QUALITY_RESULTS.md"),
                unknown_fit_counts_sha256=file_hash(OUT / "unknown_fit_counts.json"),
                files={path.relative_to(ROOT).as_posix():file_hash(path)
                       for scope in selected for path in (OUT / Path(SCOPES[scope][0]).name).rglob("*")
                       if path.is_file()}, source_status={scope: report["status"] for scope, report in reports.items()}))
        if not args.follow or all(report["status"]["status"] == "complete" for report in reports.values()):
            break
        if any(report["status"]["status"] == "invalid" for report in reports.values()):
            raise RuntimeError("Invalid complete producer marker; no partial result relabelled complete")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
