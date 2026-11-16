"""Recompute original 20% rate savings vs Naive Delta and 30% relative unknown miss reduction across all definitions and cohorts."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import zlib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.analyze_task_context_confirmation import known_f1, later_counts, safe_ratio, scalar
from scripts.run_locked_second_test import sha, write_json
from scripts.run_real_link_recovery import CHECKSUM, FEEDBACK, HEADER, digest

OUT = ROOT / "results/original_targets_recomputed_v1"
CACHE = ROOT / "cache/sn7_frozen_v1"


def fnr_counts(matrix: np.ndarray, heldout: int) -> np.ndarray:
    idx = heldout - 1
    tp = float(matrix[idx, idx])
    gt = float(matrix[idx, :].sum())
    return np.array([gt - tp, gt], dtype=np.float64)


def q95(dist: np.ndarray) -> list[float | None]:
    valid = dist[np.isfinite(dist)]
    if len(valid) == 0:
        return [None, None]
    lo, hi = np.quantile(valid, [0.025, 0.975])
    return [scalar(lo), scalar(hi)]


def encode_cell_subset(
    active_mask: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    base: int,
    target: int,
    observation: int,
    residual: bool,
    session: bytes,
    use_int8_residual: bool = False,
) -> tuple[bytes, dict]:
    """Encode active cells (either absolute int8 tokens or int16/int8 residuals) with SR12 header + 32-byte active bitmap + zlib6."""
    bitmap = np.packbits(active_mask.astype(np.uint8)).tobytes()  # 256 bits = 32 bytes
    idx = np.flatnonzero(active_mask)
    if len(idx) == 0:
        raw_payload = b""
    elif not residual:
        raw_payload = current[idx].astype(np.int8).tobytes()
    else:
        diff = current[idx].astype(np.int16) - reference[idx].astype(np.int16)
        if use_int8_residual:
            if np.any(np.abs(diff) > 127):
                raise ValueError("Residual exceeds int8 range [-127, 127]")
            raw_payload = diff.astype(np.int8).tobytes()
        else:
            raw_payload = diff.astype("<i2").tobytes()
    compressed = zlib.compress(bitmap + raw_payload, 6)
    reconstructed = reference.copy()
    if len(idx) > 0:
        reconstructed[idx] = current[idx]
    mode_flag = int(not residual) if not use_int8_residual else (2 if residual else 1)
    body = (
        HEADER.pack(
            b"SR12",
            mode_flag,
            base,
            target,
            observation,
            *current.shape,
            session,
            digest(reference.tobytes()),
            digest(reconstructed.tobytes()),
            len(compressed),
        )
        + compressed
    )
    wire_pkt = body + CHECKSUM.pack(zlib.crc32(body))
    ack = FEEDBACK.pack(b"RF11", 1, target, observation, digest(wire_pkt), digest(reconstructed.tobytes()))
    overhead = HEADER.size + CHECKSUM.size + len(ack)
    return wire_pkt, {
        "total_bytes": len(wire_pkt) + len(ack),
        "down_bytes": len(wire_pkt),
        "ack_bytes": len(ack),
        "header_ack_overhead_bytes": overhead,
        "compressed_payload_bytes": len(compressed),
        "active_cells": int(len(idx)),
        "reconstructed": reconstructed,
    }


def compute_part_a_rate_savings() -> dict:
    """Part A: Recompute 20% rate savings under (A1) equal full-frame decoded features, (A2) Naive-DINO-Delta across thresholds tau, and (A3) equal task utility on SN7."""
    # A1: Load equal_decode_v3
    eq_rows = list(csv.DictReader((ROOT / "results/feature_rate/equal_decode_v3/per_aoi.csv").open(encoding="utf-8")))
    conf_eq = [r for r in eq_rows if r["split"] == "confirmation"]
    rng = np.random.default_rng(20261104)
    sample = rng.integers(0, len(conf_eq), (10000, len(conf_eq)))
    upd_vals = np.array([float(r["relative_update_saving"]) for r in conf_eq])
    seq_vals = np.array([float(r["relative_complete_sequence_saving"]) for r in conf_eq])
    a1_equal_decode = {
        "comparator": "Full-Feature (Absolute int8 tokens, zlib6)",
        "method": "Closed-Loop Reconstructed-Reference Residual Delta (int16 exact residual, zlib6)",
        "condition": "8 confirmation AOIs (181 updates), identical decoded int8 features at every step (D = 0 between streams)",
        "mean_full_feature_update_bytes": float(sum(int(r["absolute_update_bytes"]) for r in conf_eq) / sum(int(r["updates"]) for r in conf_eq)),
        "mean_delta_update_bytes": float(sum(int(r["delta_update_bytes"]) for r in conf_eq) / sum(int(r["updates"]) for r in conf_eq)),
        "relative_update_saving_mean": float(upd_vals.mean()),
        "relative_update_saving_ci95": [float(x) for x in np.quantile(upd_vals[sample].mean(1), [0.025, 0.975])],
        "relative_sequence_saving_mean": float(seq_vals.mean()),
        "relative_sequence_saving_ci95": [float(x) for x in np.quantile(seq_vals[sample].mean(1), [0.025, 0.975])],
        "per_aoi": conf_eq,
    }

    # A2: Direct comparison against Naive-DINO-Delta across thresholds tau
    # Verify max |q| and max |diff| across all 12 AOIs in SpaceNet 7
    complete = json.loads((CACHE / "complete.json").read_text("utf-8"))
    max_abs_q, max_abs_diff = 0, 0
    for area in complete["areas"]:
        with np.load(ROOT / area["path"], allow_pickle=False) as stored:
            f_all = stored["features"].astype(np.float32)
        q_all = np.rint(f_all * 127).clip(-127, 127).astype(np.int16)
        max_abs_q = max(max_abs_q, int(np.max(np.abs(q_all))))
        max_abs_diff = max(
            max_abs_diff,
            int(np.max(np.abs(q_all[1:] - q_all[:-1]))),
            int(np.max(np.abs(q_all - q_all[:1]))),
        )

    conf_areas = [a for a in complete["areas"] if "confirmation" in a["source_manifest"]]
    taus = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
    a2_by_tau = {}
    a2_repaired_int8_by_tau = {}
    a2_open_vs_closed_loop = {}
    for tau in taus:
        aoi_savings_vs_naive = []
        aoi_savings_vs_full = []
        aoi_savings_int8_vs_naive = []
        aoi_savings_int8_vs_full = []
        aoi_naive_bytes = []
        aoi_res_bytes = []
        aoi_res_int8_bytes = []
        aoi_full_bytes = []
        aoi_active_frac = []
        aoi_world_mse = []
        aoi_overhead_frac = []
        # Also track literal open-loop Naive-DINO-Delta: c_i = 1 - cos(f_{t,i}, f_{t-1,i}) > tau
        aoi_open_naive_bytes = []
        aoi_open_naive_mse = []
        for area in conf_areas:
            with np.load(ROOT / area["path"], allow_pickle=False) as stored:
                f = stored["features"].astype(np.float32)
            q = np.rint(f * 127).clip(-127, 127).astype(np.int8)
            session = digest((area["aoi"] + ":" + sha(CACHE / "run_spec.json")).encode())
            ref_naive = q[0].copy()
            ref_res = q[0].copy()
            ref_res_i8 = q[0].copy()
            ref_open_naive = q[0].copy()
            b_naive_sum, b_res_sum, b_res_i8_sum, b_full_sum, b_open_sum = 0, 0, 0, 0, 0
            act_sum = 0
            mse_list = []
            open_mse_list = []
            ovh_sum = 0
            n_steps = len(q) - 1
            for step in range(1, len(q)):
                # Full feature reference cost
                _, full_info = encode_cell_subset(np.ones(256, dtype=bool), q[step - 1], q[step], step, step + 1, step, False, session)
                b_full_sum += full_info["total_bytes"]

                # Closed-loop cosine drift between current feature f[step] and receiver reconstructed reference (ref / 127)
                ref_float = ref_res.astype(np.float32) / 127.0
                ref_norm = ref_float / np.maximum(np.linalg.norm(ref_float, axis=1, keepdims=True), 1e-8)
                cur_norm = f[step] / np.maximum(np.linalg.norm(f[step], axis=1, keepdims=True), 1e-8)
                c_i = 1.0 - np.sum(cur_norm * ref_norm, axis=1)
                active = c_i > tau if tau > 0.0 else np.ones(256, dtype=bool)

                _, naive_info = encode_cell_subset(active, ref_naive, q[step], step, step + 1, step, False, session)
                _, res_info = encode_cell_subset(active, ref_res, q[step], step, step + 1, step, True, session, use_int8_residual=False)
                _, res_i8_info = encode_cell_subset(active, ref_res_i8, q[step], step, step + 1, step, True, session, use_int8_residual=True)
                if not (np.array_equal(naive_info["reconstructed"], res_info["reconstructed"]) and np.array_equal(naive_info["reconstructed"], res_i8_info["reconstructed"])):
                    raise AssertionError("Naive-DINO-Delta and Residual-Delta reconstructed states must match")
                ref_naive = naive_info["reconstructed"]
                ref_res = res_info["reconstructed"]
                ref_res_i8 = res_i8_info["reconstructed"]

                b_naive_sum += naive_info["total_bytes"]
                b_res_sum += res_info["total_bytes"]
                b_res_i8_sum += res_i8_info["total_bytes"]
                ovh_sum += res_i8_info["header_ack_overhead_bytes"]
                act_sum += res_info["active_cells"]
                mse_list.append(float(np.mean((f[step].astype(np.float64) - ref_res.astype(np.float64) / 127.0) ** 2)))

                # Open-loop Naive-DINO-Delta (literal line 509: c_i = 1 - cos(f_t, f_{t-1}) > tau)
                prev_norm = f[step - 1] / np.maximum(np.linalg.norm(f[step - 1], axis=1, keepdims=True), 1e-8)
                c_open = 1.0 - np.sum(cur_norm * prev_norm, axis=1)
                act_open = c_open > tau if tau > 0.0 else np.ones(256, dtype=bool)
                _, open_info = encode_cell_subset(act_open, ref_open_naive, q[step], step, step + 1, step, False, session)
                ref_open_naive = open_info["reconstructed"]
                b_open_sum += open_info["total_bytes"]
                open_mse_list.append(float(np.mean((f[step].astype(np.float64) - ref_open_naive.astype(np.float64) / 127.0) ** 2)))

            aoi_naive_bytes.append(b_naive_sum / n_steps)
            aoi_res_bytes.append(b_res_sum / n_steps)
            aoi_res_int8_bytes.append(b_res_i8_sum / n_steps)
            aoi_full_bytes.append(b_full_sum / n_steps)
            aoi_savings_vs_naive.append(1.0 - b_res_sum / b_naive_sum)
            aoi_savings_vs_full.append(1.0 - b_res_sum / b_full_sum)
            aoi_savings_int8_vs_naive.append(1.0 - b_res_i8_sum / b_naive_sum)
            aoi_savings_int8_vs_full.append(1.0 - b_res_i8_sum / b_full_sum)
            aoi_active_frac.append(act_sum / (n_steps * 256.0))
            aoi_world_mse.append(float(np.mean(mse_list)))
            aoi_overhead_frac.append(ovh_sum / b_res_i8_sum)
            aoi_open_naive_bytes.append(b_open_sum / n_steps)
            aoi_open_naive_mse.append(float(np.mean(open_mse_list)))

        sav_n = np.array(aoi_savings_vs_naive)
        sav_f = np.array(aoi_savings_vs_full)
        sav_i8_n = np.array(aoi_savings_int8_vs_naive)
        sav_i8_f = np.array(aoi_savings_int8_vs_full)
        a2_by_tau[f"tau_{tau:.3f}"] = {
            "tau": tau,
            "mean_active_cell_fraction": float(np.mean(aoi_active_frac)),
            "mean_world_feature_mse": float(np.mean(aoi_world_mse)),
            "mean_full_feature_bytes": float(np.mean(aoi_full_bytes)),
            "mean_naive_dino_delta_bytes": float(np.mean(aoi_naive_bytes)),
            "mean_residual_delta_bytes": float(np.mean(aoi_res_bytes)),
            "mean_residual_delta_int8_bytes": float(np.mean(aoi_res_int8_bytes)),
            "mean_protocol_overhead_fraction": float(np.mean(aoi_overhead_frac)),
            "saving_residual_vs_naive_dino_delta_mean": float(sav_n.mean()),
            "saving_residual_vs_naive_dino_delta_ci95": [float(x) for x in np.quantile(sav_n[sample].mean(1), [0.025, 0.975])],
            "saving_residual_vs_full_feature_mean": float(sav_f.mean()),
            "saving_residual_vs_full_feature_ci95": [float(x) for x in np.quantile(sav_f[sample].mean(1), [0.025, 0.975])],
            "saving_int8_residual_vs_naive_dino_delta_mean": float(sav_i8_n.mean()),
            "saving_int8_residual_vs_naive_dino_delta_ci95": [float(x) for x in np.quantile(sav_i8_n[sample].mean(1), [0.025, 0.975])],
            "saving_int8_residual_vs_full_feature_mean": float(sav_i8_f.mean()),
            "saving_int8_residual_vs_full_feature_ci95": [float(x) for x in np.quantile(sav_i8_f[sample].mean(1), [0.025, 0.975])],
            "saving_naive_dino_delta_vs_full_feature_mean": float(np.mean([1.0 - n / f for n, f in zip(aoi_naive_bytes, aoi_full_bytes)])),
        }
        a2_repaired_int8_by_tau[f"tau_{tau:.3f}"] = {
            "tau": tau,
            "max_abs_q_coordinate": max_abs_q,
            "max_abs_residual_coordinate": max_abs_diff,
            "lossless_int8_verified": bool(max_abs_diff <= 127),
            "decoded_feature_mismatch_D_diff": 0.0,
            "mean_naive_dino_delta_bytes": float(np.mean(aoi_naive_bytes)),
            "mean_int8_residual_delta_bytes": float(np.mean(aoi_res_int8_bytes)),
            "saving_int8_residual_vs_naive_dino_delta_mean": float(sav_i8_n.mean()),
            "saving_int8_residual_vs_naive_dino_delta_ci95": [float(x) for x in np.quantile(sav_i8_n[sample].mean(1), [0.025, 0.975])],
            "saving_int8_residual_vs_full_feature_mean": float(sav_i8_f.mean()),
            "saving_int8_residual_vs_full_feature_ci95": [float(x) for x in np.quantile(sav_i8_f[sample].mean(1), [0.025, 0.975])],
            "passes_20pct_saving_vs_naive_target": bool(sav_i8_n.mean() >= 0.20),
        }
        a2_open_vs_closed_loop[f"tau_{tau:.3f}"] = {
            "tau": tau,
            "closed_loop_safesync_int8_bytes": float(np.mean(aoi_res_int8_bytes)),
            "closed_loop_world_mse": float(np.mean(aoi_world_mse)),
            "open_loop_naive_dino_delta_bytes": float(np.mean(aoi_open_naive_bytes)),
            "open_loop_world_mse": float(np.mean(aoi_open_naive_mse)),
            "open_loop_mse_inflation_ratio": float(np.mean(aoi_open_naive_mse) / max(np.mean(aoi_world_mse), 1e-12)),
        }

    # A3: Equal-task-utility interpolation on SpaceNet 7 (Author OOF Building F1 & Event F1)
    # Load SpaceNet 7 TCM analysis summary which includes gaussian_20261029, dct_bound, full_cache_mse, full_webp, learned_hyperprior_q1_8, tcm
    sn7_summary = list(csv.DictReader((ROOT / "results/tcm_sn7/analysis_v1/summary.csv").open(encoding="utf-8")))
    curves = defaultdict(list)
    for r in sn7_summary:
        curves[r["method"]].append(
            {
                "budget": int(r["budget"]),
                "bytes": float(r["mean_bytes"]),
                "building_f1": float(r["building_f1"]),
                "event_f1": float(r["event_f1"]),
            }
        )
    for m in curves:
        curves[m].sort(key=lambda x: x["bytes"])

    def interp_bytes_at_utility(pts: list[dict], metric: str, target_u: float) -> float | None:
        """Piecewise-linear interpolation of bytes as a function of utility strictly inside the measured range."""
        u_vals = [p[metric] for p in pts]
        b_vals = [p["bytes"] for p in pts]
        if target_u < min(u_vals) - 1e-9 or target_u > max(u_vals) + 1e-9:
            return None
        for i in range(len(pts) - 1):
            u0, u1 = u_vals[i], u_vals[i + 1]
            b0, b1 = b_vals[i], b_vals[i + 1]
            if min(u0, u1) - 1e-9 <= target_u <= max(u0, u1) + 1e-9:
                if abs(u1 - u0) < 1e-12:
                    return float(min(b0, b1))
                frac = np.clip((target_u - u0) / (u1 - u0), 0.0, 1.0)
                return float(b0 + frac * (b1 - b0))
        return None

    target_building_u = [0.35130, 0.37882, 0.39426, 0.41217, 0.42720, 0.43079]
    building_interp_table = []
    for u_val in target_building_u:
        b_gauss = interp_bytes_at_utility(curves["gaussian_20261029"], "building_f1", u_val)
        row_u = {"target_building_f1": u_val, "gaussian_bytes": b_gauss, "comparisons": {}}
        for comp in ("full_cache_mse", "full_webp", "learned_hyperprior_q1_8", "tcm", "dct_bound"):
            b_comp = interp_bytes_at_utility(curves[comp], "building_f1", u_val)
            if b_gauss is not None and b_comp is not None:
                saving = 1.0 - (b_gauss / b_comp)
                status = "interpolated_on_overlap"
            else:
                saving = None
                status = "N/A (non-overlapping utility range within 32-128 KiB caps)"
            row_u["comparisons"][comp] = {
                "comparator_bytes": b_comp,
                "relative_byte_saving_gaussian_vs_comparator": saving,
                "status": status,
            }
        building_interp_table.append(row_u)

    target_event_u = [0.14180, 0.15434, 0.16933, 0.17254, 0.18819]
    event_interp_table = []
    for u_val in target_event_u:
        b_gauss = interp_bytes_at_utility(curves["gaussian_20261029"], "event_f1", u_val)
        row_u = {"target_event_f1": u_val, "gaussian_bytes": b_gauss, "comparisons": {}}
        for comp in ("full_cache_mse", "full_webp", "learned_hyperprior_q1_8", "tcm", "dct_bound"):
            b_comp = interp_bytes_at_utility(curves[comp], "event_f1", u_val)
            if b_gauss is not None and b_comp is not None:
                saving = 1.0 - (b_gauss / b_comp)
                status = "interpolated_on_overlap"
            else:
                saving = None
                status = "N/A (non-overlapping utility range within 32-128 KiB caps)"
            row_u["comparisons"][comp] = {
                "comparator_bytes": b_comp,
                "relative_byte_saving_gaussian_vs_comparator": saving,
                "status": status,
            }
        event_interp_table.append(row_u)

    return {
        "A1_equal_decoded_feature_rate_control": a1_equal_decode,
        "A2_residual_delta_vs_naive_dino_delta_by_tau": a2_by_tau,
        "A2_repaired_int8_residual_vs_naive_dino_delta_by_tau": a2_repaired_int8_by_tau,
        "A2_closed_loop_vs_open_loop_naive_dino_delta": a2_open_vs_closed_loop,
        "A3_sn7_equal_utility_interpolation": {
            "measured_curves": curves,
            "building_f1_equal_utility": building_interp_table,
            "event_f1_equal_utility": event_interp_table,
        },
    }


def compute_part_b_unknown_miss_reduction() -> dict:
    """Part B: Recompute 30% relative unknown miss reduction across Cohorts 1, 2, and 3 under all three definitions."""
    draws = np.random.default_rng(20261107).multinomial(256, np.full(256, 1.0 / 256.0), size=10000)

    def eval_cohort_contrasts(
        per_pair: dict,
        pair_ids: list[str],
        heldout: int,
        cap: int,
        method: str,
        comparators: list[str],
    ) -> dict:
        a = [per_pair[(heldout, cap, method, pid)] for pid in pair_ids]
        sup = np.array([r["support"] for r in a], dtype=np.float64)
        ret_a = np.array([r["retained30"] for r in a], dtype=np.float64)
        kno_a = [r["known"] for r in a]
        fut_a = np.stack([later_counts(r["future"], heldout) for r in a]) if a[0]["future"] is not None else None
        fnr_a = np.stack([fnr_counts(r["future"], heldout) for r in a]) if a[0]["future"] is not None else None
        bytes_a = float(np.mean([r["bytes"] for r in a]))

        u30_a = float(safe_ratio(ret_a.sum(), sup.sum()))
        umr30_a = 1.0 - u30_a
        k_f1_a = float(known_f1(np.sum(kno_a, axis=0), heldout))
        l_f1_a = float(safe_ratio(fut_a[:, 0].sum(), fut_a[:, 1].sum())) if fut_a is not None else None
        l_fnr_a = float(safe_ratio(fnr_a[:, 0].sum(), fnr_a[:, 1].sum())) if fnr_a is not None else None

        res_comp = {}
        for comp in comparators:
            if (heldout, cap, comp, pair_ids[0]) not in per_pair:
                continue
            b = [per_pair[(heldout, cap, comp, pid)] for pid in pair_ids]
            ret_b = np.array([r["retained30"] for r in b], dtype=np.float64)
            kno_b = [r["known"] for r in b]
            bytes_b = float(np.mean([r["bytes"] for r in b]))
            u30_b = float(safe_ratio(ret_b.sum(), sup.sum()))
            umr30_b = 1.0 - u30_b
            k_f1_b = float(known_f1(np.sum(kno_b, axis=0), heldout))
            known_exact = all(np.array_equal(x, y) for x, y in zip(kno_a, kno_b))

            # UMR30 relative reduction: (C_new - C_base) / (1 - C_base) = (UMR_base - UMR_new) / UMR_base
            delta_ret = ret_a - ret_b
            u_diff_pt = float(safe_ratio(delta_ret.sum(), sup.sum()))
            u_diff_dist = safe_ratio(draws @ delta_ret, draws @ sup)
            c_a_dist = safe_ratio(draws @ ret_a, draws @ sup)
            c_b_dist = safe_ratio(draws @ ret_b, draws @ sup)
            rel_umr_pt = (u30_a - u30_b) / (1.0 - u30_b) if (1.0 - u30_b) > 0 else 0.0
            rel_umr_dist = safe_ratio(c_a_dist - c_b_dist, 1.0 - c_b_dist)

            entry = {
                "method": method,
                "comparator": comp,
                "heldout": heldout,
                "budget": cap,
                "mean_method_bytes": bytes_a,
                "mean_comparator_bytes": bytes_b,
                "known_macro_f1_method": k_f1_a,
                "known_macro_f1_comparator": k_f1_b,
                "known_confusions_identical": known_exact,
                "def1_unqualified_evidence_miss_UMR30": {
                    "baseline_u30": u30_b,
                    "method_u30": u30_a,
                    "baseline_umr30": umr30_b,
                    "method_umr30": umr30_a,
                    "abs_u30_diff": u_diff_pt,
                    "abs_u30_diff_ci95": q95(u_diff_dist),
                    "relative_umr30_reduction": rel_umr_pt,
                    "relative_umr30_reduction_ci95": q95(rel_umr_dist),
                    "meets_30pct_relative_reduction_target": bool(rel_umr_pt >= 0.30 and (q95(rel_umr_dist)[0] or 0.0) > 0.0),
                },
            }

            if fut_a is not None and b[0]["future"] is not None:
                fut_b = np.stack([later_counts(r["future"], heldout) for r in b])
                fnr_b = np.stack([fnr_counts(r["future"], heldout) for r in b])
                l_f1_b = float(safe_ratio(fut_b[:, 0].sum(), fut_b[:, 1].sum()))
                l_fnr_b = float(safe_ratio(fnr_b[:, 0].sum(), fnr_b[:, 1].sum()))
                ba_f, bb_f = draws @ fut_a, draws @ fut_b
                f1_diff_dist = safe_ratio(ba_f[:, 0], ba_f[:, 1]) - safe_ratio(bb_f[:, 0], bb_f[:, 1])

                fa_n, fb_n = draws @ fnr_a, draws @ fnr_b
                fnr_a_dist = safe_ratio(fa_n[:, 0], fa_n[:, 1])
                fnr_b_dist = safe_ratio(fb_n[:, 0], fb_n[:, 1])
                fnr_diff_dist = fnr_a_dist - fnr_b_dist
                rel_fnr_pt = (l_fnr_b - l_fnr_a) / l_fnr_b if l_fnr_b > 0 else 0.0
                rel_fnr_dist = safe_ratio(fnr_b_dist - fnr_a_dist, fnr_b_dist)

                entry["def2_semantic_probe_heldout_FNR"] = {
                    "baseline_later_f1": l_f1_b,
                    "method_later_f1": l_f1_a,
                    "later_f1_diff": l_f1_a - l_f1_b,
                    "later_f1_diff_ci95": q95(f1_diff_dist),
                    "baseline_heldout_fnr": l_fnr_b,
                    "method_heldout_fnr": l_fnr_a,
                    "abs_fnr_diff": l_fnr_a - l_fnr_b,
                    "abs_fnr_diff_ci95": q95(fnr_diff_dist),
                    "relative_fnr_reduction": rel_fnr_pt,
                    "relative_fnr_reduction_ci95": q95(rel_fnr_dist),
                    "meets_30pct_relative_reduction_target": bool(rel_fnr_pt >= 0.30 and (q95(rel_fnr_dist)[0] or 0.0) > 0.0),
                }
            res_comp[comp] = entry
        return res_comp

    # --- Load Cohort 1 (Reserved Test v1: unknown_recovery/locked_test_v1 + hybrid_pq/confirmation_v1 + future_semantic_probe/v1/confirmation) ---
    c1_img_rows = list(csv.DictReader((ROOT / "results/unknown_recovery/locked_test_v1/records.csv").open(encoding="utf-8")))
    c1_hyb_rows = list(csv.DictReader((ROOT / "results/hybrid_pq/confirmation_v1/records.csv").open(encoding="utf-8")))
    c1_lic_rows = list(csv.DictReader((ROOT / "results/second_learned/tasks_v1/records.csv").open(encoding="utf-8")))
    c1_fut_rows = list(csv.DictReader((ROOT / "results/future_semantic_probe/v1/confirmation/records.csv").open(encoding="utf-8")))
    fut_by_sha = {(int(r["heldout"]), r["receiver_rgb_sha256"]): np.array(json.loads(r["confusion"]), float) for r in c1_fut_rows if r["representation"] == "rgb"}

    c1_grouped = defaultdict(list)
    for r in c1_img_rows:
        c1_grouped[(int(r["heldout"]), int(r["budget"]), "image_" + r["method"], r["pair_id"])].append(
            {
                "known": np.array(json.loads(r["confusion"]), float),
                "future": fut_by_sha.get((int(r["heldout"]), r["receiver_rgb_sha256"])),
                "support": float(r["unknown_pixels"]),
                "retained30": float(r["unknown_retained_pixels_30"]),
                "bytes": float(r["total_bytes"]),
            }
        )
    for r in c1_hyb_rows:
        c1_grouped[(int(r["heldout"]), int(r["budget"]), "hybrid_" + r["method"], r["pair_id"])].append(
            {
                "known": np.array(json.loads(r["known_confusion"]), float),
                "future": fut_by_sha.get((int(r["heldout"]), r["receiver_rgb_sha256"])),
                "support": float(r["unknown_pixels"]),
                "retained30": float(r["unknown_retained_pixels_30"]),
                "bytes": float(r["total_bytes"]),
            }
        )
    for r in c1_lic_rows:
        c1_grouped[(int(r["heldout"]), int(r["budget"]), "hyperprior", r["pair_id"])].append(
            {
                "known": np.array(json.loads(r["confusion"]), float),
                "future": fut_by_sha.get((int(r["heldout"]), r["receiver_rgb_sha256"])),
                "support": float(r["unknown_pixels"]),
                "retained30": float(r["unknown_retained_pixels_30"]),
                "bytes": float(r["total_bytes"]),
            }
        )
    c1_per_pair = {}
    for k, grp in c1_grouped.items():
        c1_per_pair[k] = {
            "known": np.mean([x["known"] for x in grp], axis=0),
            "future": np.mean([x["future"] for x in grp], axis=0) if all(x["future"] is not None for x in grp) else None,
            "support": grp[0]["support"],
            "retained30": float(np.mean([x["retained30"] for x in grp])),
            "bytes": float(np.mean([x["bytes"] for x in grp])),
        }
    c1_pids = sorted({k[3] for k in c1_per_pair})

    # --- Load Cohort 2 (Fresh Context v1: fresh_context_ground_v1 + fresh_current_distance_v1) ---
    c2_rows = list(csv.DictReader((ROOT / "results/task_evidence/fresh_context_ground_v1/records.csv").open(encoding="utf-8")))
    c2_add = list(csv.DictReader((ROOT / "results/task_evidence/fresh_current_distance_v1/records.csv").open(encoding="utf-8")))
    c2_grouped = defaultdict(list)
    for r in c2_rows + c2_add:
        c2_grouped[(int(r["heldout"]), int(r["budget"]), r["method"], r["pair_id"])].append(
            {
                "known": np.array(json.loads(r["known_confusion"]), float),
                "future": np.array(json.loads(r["future_confusion"]), float),
                "support": float(r["unknown_pixels"]),
                "retained30": float(r["unknown_retained_pixels_30"]),
                "bytes": float(r["total_bytes"]),
            }
        )
    c2_per_pair = {
        k: {
            "known": np.mean([x["known"] for x in grp], axis=0),
            "future": np.mean([x["future"] for x in grp], axis=0),
            "support": grp[0]["support"],
            "retained30": float(np.mean([x["retained30"] for x in grp])),
            "bytes": float(np.mean([x["bytes"] for x in grp])),
        }
        for k, grp in c2_grouped.items()
    }
    c2_pids = sorted({k[3] for k in c2_per_pair})

    # --- Load Cohort 3 (Residual Confirmation v1 + TCM Control) ---
    c3_rows = list(csv.DictReader((ROOT / "results/task_evidence/residual_confirmation_ground_v1/records.csv").open(encoding="utf-8")))
    c3_tcm = list(csv.DictReader((ROOT / "results/task_evidence/tcm_ground_v1/records.csv").open(encoding="utf-8")))
    c3_grouped = defaultdict(list)
    for r in c3_rows + c3_tcm:
        c3_grouped[(int(r["heldout"]), int(r["budget"]), r["method"], r["pair_id"])].append(
            {
                "known": np.array(json.loads(r["known_confusion"]), float),
                "future": np.array(json.loads(r["future_confusion"]), float),
                "support": float(r["unknown_pixels"]),
                "retained30": float(r["unknown_retained_pixels_30"]),
                "bytes": float(r["total_bytes"]),
            }
        )
    c3_per_pair = {
        k: {
            "known": np.mean([x["known"] for x in grp], axis=0),
            "future": np.mean([x["future"] for x in grp], axis=0),
            "support": grp[0]["support"],
            "retained30": float(np.mean([x["retained30"] for x in grp])),
            "bytes": float(np.mean([x["bytes"] for x in grp])),
        }
        for k, grp in c3_grouped.items()
    }
    c3_pids = sorted({k[3] for k in c3_per_pair})

    return {
        "cohort_1_reserved_test_v1": {
            "primary_1a_image_emergent_h5_b16384": eval_cohort_contrasts(
                c1_per_pair,
                c1_pids,
                5,
                16384,
                "image_emergent_mix",
                ["image_feature_change", "image_full_webp", "hyperprior"],
            ),
            "primary_1b_hybrid_pq_emergent_h5_b8192": eval_cohort_contrasts(
                c1_per_pair,
                c1_pids,
                5,
                8192,
                "hybrid_emergent_mix",
                ["hybrid_change", "image_full_webp", "hyperprior"],
            ),
        },
        "cohort_2_fresh_task_context_v1": {
            "primary_2_adaptive_mix_h5_b8192": eval_cohort_contrasts(
                c2_per_pair,
                c2_pids,
                5,
                8192,
                "adaptive_mix",
                ["adaptive_change", "adaptive_current_distance", "full_webp", "hyperprior", "roi_mix"],
            ),
        },
        "cohort_3_residual_confirmation_v1": {
            "primary_3_residual_current_h5_b8192": eval_cohort_contrasts(
                c3_per_pair,
                c3_pids,
                5,
                8192,
                "residual_current",
                ["residual_change", "absolute_current", "absolute_change", "hyperprior", "full_webp", "tcm"],
            ),
            "encoding_effect_residual_change_h5_b8192": eval_cohort_contrasts(
                c3_per_pair,
                c3_pids,
                5,
                8192,
                "residual_change",
                ["absolute_change", "hyperprior", "full_webp", "tcm"],
            ),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Computing Part A: 20% Rate Savings vs Naive Delta...", flush=True)
    part_a = compute_part_a_rate_savings()
    print("Computing Part B: 30% Relative Unknown Miss Reduction across all cohorts and definitions...", flush=True)
    part_b = compute_part_b_unknown_miss_reduction()

    output = {
        "schema": "original-targets-recomputed-v1",
        "part_a_20pct_rate_savings": part_a,
        "part_b_30pct_unknown_miss_reduction": part_b,
    }
    write_json(OUT / "recomputed_targets.json", output)
    print("Saved results/original_targets_recomputed_v1/recomputed_targets.json successfully!", flush=True)


if __name__ == "__main__":
    main()
