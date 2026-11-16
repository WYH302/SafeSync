"""Complete 2x2 four-cell attribution and post-hoc Oracle / Ideal Selection diagnostic on SECOND."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch

from safesync import residual_evidence as enhancement
from safesync import task_evidence_wire as wire
from safesync.data import read_pair
from safesync.features import FrozenEncoder
from scripts.analyze_task_context_confirmation import known_f1, later_counts, safe_ratio, scalar
from scripts.run_locked_second_test import sha, write_json
from scripts.run_residual_evidence_development import fit, pool
from scripts.run_second_feature_mix import load_frozen
from scripts.run_task_evidence_selector import quality

SOURCE = ROOT / "results/task_evidence/residual_confirmation_v1"
GROUND_CSV = ROOT / "results/task_evidence/residual_confirmation_ground_v1/records.csv"
TCM_CSV = ROOT / "results/task_evidence/tcm_ground_v1/records.csv"
DATA = ROOT / "data/second_residual_test_v1"
OUT = ROOT / "results/task_evidence/factorial_oracle_diagnostic_v1"


def fnr_counts(matrix: np.ndarray, heldout: int) -> np.ndarray:
    """Return [FN, TP + FN] for the held-out class."""
    idx = heldout - 1
    tp = float(matrix[idx, idx])
    gt = float(matrix[idx, :].sum())
    return np.array([gt - tp, gt], dtype=np.float64)


def extract_payload(packet: bytes) -> bytes:
    """Extract the inner image payload from a TE01 packet."""
    values = wire.HEADER.unpack_from(packet)
    count, ns, nl = values[11], values[14], values[15]
    offset = wire.HEADER.size + ns + count * 2 + nl
    return packet[offset:-16]


def q95(dist: np.ndarray) -> tuple[float | None, float | None]:
    valid = dist[np.isfinite(dist)]
    if len(valid) == 0:
        return None, None
    lo, hi = np.quantile(valid, [0.025, 0.975])
    return scalar(lo), scalar(hi)


def bootstrap_contrast(
    draws: np.ndarray,
    ret_a: np.ndarray,
    ret_b: np.ndarray,
    sup: np.ndarray,
    fut_a: np.ndarray,
    fut_b: np.ndarray,
    fnr_a: np.ndarray,
    fnr_b: np.ndarray,
) -> dict:
    """Compute paired multinomial bootstrap differences and relative miss reductions."""
    delta_ret = ret_a - ret_b
    u_diff_dist = safe_ratio(draws @ delta_ret, draws @ sup)
    u_diff_pt = scalar(safe_ratio(delta_ret.sum(), sup.sum()))
    u_lo, u_hi = q95(u_diff_dist)

    # Relative UMR30 reduction: (C_a - C_b) / (1 - C_b)
    c_a_pt = scalar(safe_ratio(ret_a.sum(), sup.sum()))
    c_b_pt = scalar(safe_ratio(ret_b.sum(), sup.sum()))
    rel_umr_pt = scalar((c_a_pt - c_b_pt) / (1.0 - c_b_pt)) if (c_a_pt is not None and c_b_pt is not None and (1.0 - c_b_pt) > 0) else None
    c_a_dist = safe_ratio(draws @ ret_a, draws @ sup)
    c_b_dist = safe_ratio(draws @ ret_b, draws @ sup)
    rel_umr_dist = safe_ratio(c_a_dist - c_b_dist, 1.0 - c_b_dist)
    umr_lo, umr_hi = q95(rel_umr_dist)

    # Later F1 difference: F1(a) - F1(b)
    ba, bb = draws @ fut_a, draws @ fut_b
    f1_a_pt = scalar(safe_ratio(fut_a[:, 0].sum(), fut_a[:, 1].sum()))
    f1_b_pt = scalar(safe_ratio(fut_b[:, 0].sum(), fut_b[:, 1].sum()))
    f1_diff_pt = scalar(f1_a_pt - f1_b_pt) if (f1_a_pt is not None and f1_b_pt is not None) else None
    f1_diff_dist = safe_ratio(ba[:, 0], ba[:, 1]) - safe_ratio(bb[:, 0], bb[:, 1])
    f1_lo, f1_hi = q95(f1_diff_dist)

    # Held-out FNR difference: FNR(a) - FNR(b) and relative FNR reduction: (FNR(b) - FNR(a)) / FNR(b)
    fa, fb = draws @ fnr_a, draws @ fnr_b
    fnr_a_pt = scalar(safe_ratio(fnr_a[:, 0].sum(), fnr_a[:, 1].sum()))
    fnr_b_pt = scalar(safe_ratio(fnr_b[:, 0].sum(), fnr_b[:, 1].sum()))
    fnr_diff_pt = scalar(fnr_a_pt - fnr_b_pt) if (fnr_a_pt is not None and fnr_b_pt is not None) else None
    fnr_a_dist = safe_ratio(fa[:, 0], fa[:, 1])
    fnr_b_dist = safe_ratio(fb[:, 0], fb[:, 1])
    fnr_diff_dist = fnr_a_dist - fnr_b_dist
    fnr_lo, fnr_hi = q95(fnr_diff_dist)

    rel_fnr_pt = scalar((fnr_b_pt - fnr_a_pt) / fnr_b_pt) if (fnr_a_pt is not None and fnr_b_pt is not None and fnr_b_pt > 0) else None
    rel_fnr_dist = safe_ratio(fnr_b_dist - fnr_a_dist, fnr_b_dist)
    rfnr_lo, rfnr_hi = q95(rel_fnr_dist)

    return {
        "u30_diff": [u_diff_pt, u_lo, u_hi],
        "rel_umr_reduction": [rel_umr_pt, umr_lo, umr_hi],
        "later_f1_diff": [f1_diff_pt, f1_lo, f1_hi],
        "later_fnr_diff": [fnr_diff_pt, fnr_lo, fnr_hi],
        "rel_fnr_reduction": [rel_fnr_pt, rfnr_lo, rfnr_hi],
    }


def compute_factorial_analysis(rows: list[dict], pair_ids: list[str]) -> dict:
    ids = sorted(pair_ids)
    by_key = {(r["pair_id"], int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in rows}
    methods = [
        "absolute_change",
        "absolute_current",
        "residual_change",
        "residual_current",
        "hyperprior",
        "full_webp",
        "tcm",
    ]
    draws = np.random.default_rng(20261107).multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=10000)
    factorial_summary = {}
    for h in (5, 3, 4):
        for cap in (4096, 8192, 16384, 32768):
            cell_stats = {}
            arrays = {}
            for m in methods:
                if (ids[0], h, cap, m, 0) not in by_key:
                    continue
                recs = [by_key[(pid, h, cap, m, 0)] for pid in ids]
                ret = np.array([float(r["unknown_retained_pixels_30"]) for r in recs], dtype=np.float64)
                sup = np.array([float(r["unknown_pixels"]) for r in recs], dtype=np.float64)
                fut_mats = [np.array(json.loads(r["future_confusion"]), dtype=np.float64) for r in recs]
                kno_mats = [np.array(json.loads(r["known_confusion"]), dtype=np.float64) for r in recs]
                fut = np.stack([later_counts(mat, h) for mat in fut_mats])
                fnr_c = np.stack([fnr_counts(mat, h) for mat in fut_mats])
                tot_b = np.array([float(r["total_bytes"]) for r in recs], dtype=np.float64)

                u30 = scalar(safe_ratio(ret.sum(), sup.sum()))
                umr30 = scalar(1.0 - u30) if u30 is not None else None
                later_f1 = scalar(safe_ratio(fut[:, 0].sum(), fut[:, 1].sum()))
                later_fnr = scalar(safe_ratio(fnr_c[:, 0].sum(), fnr_c[:, 1].sum()))
                k_f1 = scalar(known_f1(np.sum(kno_mats, axis=0), h))
                arrays[m] = (ret, sup, fut, fnr_c, tot_b)
                cell_stats[m] = {
                    "u30": u30,
                    "umr30": umr30,
                    "later_f1": later_f1,
                    "later_fnr": later_fnr,
                    "known_macro_f1": k_f1,
                    "mean_total_bytes": scalar(tot_b.mean()),
                }

            A, B, C, D = (
                arrays["absolute_change"],
                arrays["absolute_current"],
                arrays["residual_change"],
                arrays["residual_current"],
            )
            HP, WP = arrays["hyperprior"], arrays["full_webp"]
            sup = A[1]

            # Interaction: (D - C) - (B - A)
            inter_delta = (D[0] - C[0]) - (B[0] - A[0])
            inter_pt = scalar(safe_ratio(inter_delta.sum(), sup.sum()))
            inter_dist = safe_ratio(draws @ inter_delta, draws @ sup)
            inter_lo, inter_hi = q95(inter_dist)

            ba_f = draws @ A[2]
            bb_f = draws @ B[2]
            bc_f = draws @ C[2]
            bd_f = draws @ D[2]
            inter_f1_dist = (safe_ratio(bd_f[:, 0], bd_f[:, 1]) - safe_ratio(bc_f[:, 0], bc_f[:, 1])) - (
                safe_ratio(bb_f[:, 0], bb_f[:, 1]) - safe_ratio(ba_f[:, 0], ba_f[:, 1])
            )
            if all(cell_stats[k]["later_f1"] is not None for k in ("residual_current", "residual_change", "absolute_current", "absolute_change")):
                inter_f1_pt = scalar(
                    (cell_stats["residual_current"]["later_f1"] - cell_stats["residual_change"]["later_f1"])
                    - (cell_stats["absolute_current"]["later_f1"] - cell_stats["absolute_change"]["later_f1"])
                )
            else:
                inter_f1_pt = None
            if1_lo, if1_hi = q95(inter_f1_dist)

            contrasts = {
                "B_minus_A_unknown_in_absolute": bootstrap_contrast(draws, B[0], A[0], sup, B[2], A[2], B[3], A[3]),
                "D_minus_C_unknown_in_residual": bootstrap_contrast(draws, D[0], C[0], sup, D[2], C[2], D[3], C[3]),
                "C_minus_A_residual_in_change": bootstrap_contrast(draws, C[0], A[0], sup, C[2], A[2], C[3], A[3]),
                "D_minus_B_residual_in_unknown": bootstrap_contrast(draws, D[0], B[0], sup, D[2], B[2], D[3], B[3]),
                "D_minus_HP_total_scheme_vs_hyperprior": bootstrap_contrast(draws, D[0], HP[0], sup, D[2], HP[2], D[3], HP[3]),
                "C_minus_HP_residual_change_vs_hyperprior": bootstrap_contrast(draws, C[0], HP[0], sup, C[2], HP[2], C[3], HP[3]),
                "D_minus_WebP": bootstrap_contrast(draws, D[0], WP[0], sup, D[2], WP[2], D[3], WP[3]),
                "interaction_DC_minus_BA": {
                    "u30_diff": [inter_pt, inter_lo, inter_hi],
                    "later_f1_diff": [float(inter_f1_pt), if1_lo, if1_hi],
                },
            }
            if "tcm" in arrays:
                TCM = arrays["tcm"]
                contrasts["D_minus_TCM"] = bootstrap_contrast(draws, D[0], TCM[0], sup, D[2], TCM[2], D[3], TCM[3])
                contrasts["TCM_minus_HP"] = bootstrap_contrast(draws, TCM[0], HP[0], sup, TCM[2], HP[2], TCM[3], HP[3])
                contrasts["TCM_minus_WebP"] = bootstrap_contrast(draws, TCM[0], WP[0], sup, TCM[2], WP[2], TCM[3], WP[3])

            factorial_summary[f"h{h}_b{cap}"] = {
                "cells": cell_stats,
                "contrasts": contrasts,
            }
    return factorial_summary


def run_oracle_and_selection_diagnostic(lock: dict, rows: list[dict]) -> dict:
    """Run Oracle / Ideal Selection diagnostic on the 256 confirmation pairs (h=5, cap=8192)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, lock["backbone"])

    def isolated_features(image):
        values, timings = encoder.extract(image)
        return values.astype(np.float16).astype(np.float32), timings

    encoder.cached = isolated_features
    prototype = ROOT / "results/future_semantic_probe/v1/prototypes.npz"
    with np.load(prototype, allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))

    with np.load(ROOT / lock["galleries"]["5"]["path"], allow_pickle=False) as stored:
        gallery_5 = torch.tensor(stored["features"], device="cuda")

    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    by_key = {(r["pair_id"], int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in rows}

    h = 5
    cap = 8192
    pair_ids = sorted(lock["pair_ids"])
    draws = np.random.default_rng(20261107).multinomial(len(pair_ids), np.full(len(pair_ids), 1.0 / len(pair_ids)), size=10000)

    diag_methods = [
        "base_4kib",
        "full_8kib_hp",
        "residual_change_raw_no_fallback",
        "residual_current_raw_no_fallback",
        "residual_pure_distance_raw",
        "residual_pure_distance_with_fallback",
        "oracle_residual_greedy",
        "oracle_hybrid_with_full_hp",
        "unconstrained_64cell_residual",
    ]
    u_rets = {m: [] for m in diag_methods}
    u_dens = []
    fut_cnts = {m: [] for m in diag_methods}
    fnr_cnts = {m: [] for m in diag_methods}
    total_bytes_map = {m: [] for m in diag_methods}

    jaccard_overlaps = []
    rank_diff_counts = []
    oracle_cell_counts = []
    positive_gain_cell_counts = []
    change_raw_cell_counts = []
    current_raw_cell_counts = []
    identical_raw_selections = 0
    identical_final_packets = 0

    for idx_p, pid in enumerate(pair_ids):
        pair = pairs[pid]
        folder = SOURCE / "pairs" / pid
        _, after, old, label, changed = read_pair(ROOT, pair)
        current = np.asarray(after)
        known, unknown = helper["evaluation_masks"](old, label, changed, h)
        future_mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed
        u_den = float(unknown.sum())
        u_dens.append(u_den)

        ready = json.loads((folder / "sender_inputs_ready.json").read_text("utf-8"))
        common = ready["common"][str(h)]
        boot = (ROOT / common["bootstrap_path"]).read_bytes()
        head = wire.HEADER.unpack_from(boot)
        blank = wire.TaskEvidenceState(
            np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), head[1], head[2]
        )
        prior, _ = wire.receive(blank, boot)
        service, _ = wire.receive(prior, (ROOT / common["labels_only_path"]).read_bytes())

        base_pkt = (folder / f"h{h}_hyperprior_s0_b4096.bin").read_bytes()
        base_inner = extract_payload(base_pkt)
        base_rgb = np.asarray(Image.open(folder / f"h{h}_hyperprior_s0_b4096.received.png").convert("RGB"))
        full_pkt = (folder / f"h{h}_hyperprior_s0_b8192.bin").read_bytes()
        full_rgb = np.asarray(Image.open(folder / f"h{h}_hyperprior_s0_b8192.received.png").convert("RGB"))

        rc_pkt = (folder / f"h{h}_residual_change_s0_b8192.bin").read_bytes()
        ru_pkt = (folder / f"h{h}_residual_current_s0_b8192.bin").read_bytes()
        if rc_pkt == ru_pkt:
            identical_final_packets += 1

        with np.load(folder / "sender_features.npz", allow_pickle=False) as stored:
            f0, f1_feat = stored["before"].astype(np.float32), stored["current"].astype(np.float32)
        change = rankdata(pool(1 - np.sum(f0 * f1_feat, axis=1))) / 64.0
        query = torch.nn.functional.normalize(torch.tensor(f1_feat, device="cuda"), dim=1)
        nearest = torch.topk(query @ gallery_5.T, k=10, sorted=True).values[:, -1]
        distance = torch.clamp(2 - 2 * nearest, min=0).cpu().numpy()
        pure_dist = rankdata(pool(distance)) / 64.0
        mixed = 0.75 * change + 0.25 * pure_dist

        pkt_rc, k_rc = fit(prior, service.labels, current, base_inner, base_rgb, change, enhancement.RESIDUAL, cap)
        pkt_ru, k_ru = fit(prior, service.labels, current, base_inner, base_rgb, mixed, enhancement.RESIDUAL, cap)
        pkt_pd, k_pd = fit(prior, service.labels, current, base_inner, base_rgb, pure_dist, enhancement.RESIDUAL, cap)

        change_raw_cell_counts.append(k_rc)
        current_raw_cell_counts.append(k_ru)

        set_rc = set(map(int, np.argsort(-change, kind="stable")[:k_rc]))
        set_ru = set(map(int, np.argsort(-mixed, kind="stable")[:k_ru]))
        if set_rc == set_ru:
            identical_raw_selections += 1
        union_len = len(set_rc | set_ru)
        jaccard_overlaps.append(len(set_rc & set_ru) / union_len if union_len > 0 else 1.0)
        rank_diff_counts.append(len(set_rc ^ set_ru))

        rgb_rc = enhancement.decode(extract_payload(pkt_rc), (512, 512), lambda p: base_rgb)
        rgb_ru = enhancement.decode(extract_payload(pkt_ru), (512, 512), lambda p: base_rgb)
        rgb_pd = enhancement.decode(extract_payload(pkt_pd), (512, 512), lambda p: base_rgb)

        enh_pd = quality(current, rgb_pd, pure_dist) > quality(current, full_rgb, pure_dist)
        rgb_pd_fb = rgb_pd if enh_pd else full_rgb
        bytes_pd_fb = (len(pkt_pd) + wire.ACK_BYTES) if enh_pd else (len(full_pkt) + wire.ACK_BYTES)

        all64_inner = enhancement.encode(base_inner, current, base_rgb, list(range(64)), mode=enhancement.RESIDUAL)
        all64_pkt, _ = wire.encode(prior, service.labels, observation=1, payload=all64_inner, image_mode=wire.LEARNED)
        rgb_all64 = enhancement.decode(all64_inner, (512, 512), lambda p: base_rgb)

        # Oracle Greedy selection: rank cells by true gain in PSNR>=30 unknown pixel count
        gains = np.zeros(64, dtype=np.float64)
        for c_idx in range(64):
            r, c = divmod(c_idx, 8)
            sl = (slice(r * 64, (r + 1) * 64), slice(c * 64, (c + 1) * 64))
            u_cell = unknown[sl]
            if not u_cell.any():
                gains[c_idx] = -1e6
                continue
            mse_b = np.mean((current[sl].astype(np.float32) - base_rgb[sl].astype(np.float32)) ** 2)
            mse_r = np.mean((current[sl].astype(np.float32) - rgb_all64[sl].astype(np.float32)) ** 2)
            psnr_b = 99.0 if mse_b <= 1e-12 else 10.0 * np.log10((255.0 ** 2) / mse_b)
            psnr_r = 99.0 if mse_r <= 1e-12 else 10.0 * np.log10((255.0 ** 2) / mse_r)
            pass_b = 1.0 if psnr_b >= 30.0 else 0.0
            pass_r = 1.0 if psnr_r >= 30.0 else 0.0
            gains[c_idx] = (pass_r - pass_b) * float(u_cell.sum()) + 1e-4 * float(u_cell.sum()) * (psnr_r - psnr_b)

        pos_cells = int(np.sum(gains > 1.0))
        positive_gain_cell_counts.append(pos_cells)
        pkt_oracle, k_oracle = fit(prior, service.labels, current, base_inner, base_rgb, gains, enhancement.RESIDUAL, cap)
        oracle_cell_counts.append(k_oracle)
        rgb_oracle = enhancement.decode(extract_payload(pkt_oracle), (512, 512), lambda p: base_rgb)

        candidates_rgb = {
            "base_4kib": (base_rgb, len(base_pkt) + wire.ACK_BYTES),
            "full_8kib_hp": (full_rgb, len(full_pkt) + wire.ACK_BYTES),
            "residual_change_raw_no_fallback": (rgb_rc, len(pkt_rc) + wire.ACK_BYTES),
            "residual_current_raw_no_fallback": (rgb_ru, len(pkt_ru) + wire.ACK_BYTES),
            "residual_pure_distance_raw": (rgb_pd, len(pkt_pd) + wire.ACK_BYTES),
            "residual_pure_distance_with_fallback": (rgb_pd_fb, bytes_pd_fb),
            "oracle_residual_greedy": (rgb_oracle, len(pkt_oracle) + wire.ACK_BYTES),
            "unconstrained_64cell_residual": (rgb_all64, len(all64_pkt) + wire.ACK_BYTES),
        }

        m_or = helper["image_metrics"](current, rgb_oracle, known, unknown)
        m_hp = helper["image_metrics"](current, full_rgb, known, unknown)
        if float(m_or["unknown_retained_pixels_30"]) >= float(m_hp["unknown_retained_pixels_30"]):
            candidates_rgb["oracle_hybrid_with_full_hp"] = (rgb_oracle, len(pkt_oracle) + wire.ACK_BYTES)
        else:
            candidates_rgb["oracle_hybrid_with_full_hp"] = (full_rgb, len(full_pkt) + wire.ACK_BYTES)

        pred_cache = {}
        for m_existing in ("hyperprior", "residual_change", "residual_current"):
            rec = by_key[(pid, h, cap, m_existing, 0)]
            pred_cache[rec["receiver_rgb_sha256"]] = np.array(json.loads(rec["future_confusion"]), dtype=np.float64)
        rec_4k = by_key[(pid, h, 4096, "hyperprior", 0)]
        pred_cache[rec_4k["receiver_rgb_sha256"]] = np.array(json.loads(rec_4k["future_confusion"]), dtype=np.float64)

        for m_name, (rgb_arr, b_len) in candidates_rgb.items():
            im_metrics = helper["image_metrics"](current, rgb_arr, known, unknown)
            u_rets[m_name].append(float(im_metrics["unknown_retained_pixels_30"]))
            total_bytes_map[m_name].append(float(b_len))
            rgb_sha = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
            if rgb_sha not in pred_cache:
                features, _ = encoder.cached(Image.fromarray(rgb_arr))
                prediction = classes[np.argmax(features @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                pred_cache[rgb_sha] = np.bincount(
                    (label[future_mask].astype(int) - 1) * 6 + prediction[future_mask].astype(int) - 1, minlength=36
                ).reshape(6, 6).astype(np.float64)
            fut_cnts[m_name].append(later_counts(pred_cache[rgb_sha], h))
            fnr_cnts[m_name].append(fnr_counts(pred_cache[rgb_sha], h))

        if (idx_p + 1) % 32 == 0:
            print(f"Oracle diagnostic progress: {idx_p + 1}/{len(pair_ids)}", flush=True)

    u_den_arr = np.array(u_dens, dtype=np.float64)
    oracle_summary = {}
    for m_name in diag_methods:
        ur_arr = np.array(u_rets[m_name], dtype=np.float64)
        fc_arr = np.stack(fut_cnts[m_name])
        fn_arr = np.stack(fnr_cnts[m_name])
        tb_arr = np.array(total_bytes_map[m_name], dtype=np.float64)
        u30_val = float(safe_ratio(ur_arr.sum(), u_den_arr.sum()))
        oracle_summary[m_name] = {
            "u30": u30_val,
            "umr30": 1.0 - u30_val,
            "later_f1": float(safe_ratio(fc_arr[:, 0].sum(), fc_arr[:, 1].sum())),
            "later_fnr": float(safe_ratio(fn_arr[:, 0].sum(), fn_arr[:, 1].sum())),
            "mean_total_bytes": float(tb_arr.mean()),
        }

    # Add comparisons of Oracle against Residual+Current, Full WebP, and TCM
    rec_ru = [by_key[(pid, h, cap, "residual_current", 0)] for pid in pair_ids]
    rec_wp = [by_key[(pid, h, cap, "full_webp", 0)] for pid in pair_ids]
    rec_tcm = [by_key[(pid, h, cap, "tcm", 0)] for pid in pair_ids]

    ru_ur = np.array([float(r["unknown_retained_pixels_30"]) for r in rec_ru], dtype=np.float64)
    ru_fc = np.stack([later_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_ru])
    ru_fn = np.stack([fnr_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_ru])

    wp_ur = np.array([float(r["unknown_retained_pixels_30"]) for r in rec_wp], dtype=np.float64)
    wp_fc = np.stack([later_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_wp])
    wp_fn = np.stack([fnr_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_wp])

    tcm_ur = np.array([float(r["unknown_retained_pixels_30"]) for r in rec_tcm], dtype=np.float64)
    tcm_fc = np.stack([later_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_tcm])
    tcm_fn = np.stack([fnr_counts(np.array(json.loads(r["future_confusion"]), float), h) for r in rec_tcm])

    or_ur = np.array(u_rets["oracle_residual_greedy"], dtype=np.float64)
    or_fc = np.stack(fut_cnts["oracle_residual_greedy"])
    or_fn = np.stack(fnr_cnts["oracle_residual_greedy"])

    oh_ur = np.array(u_rets["oracle_hybrid_with_full_hp"], dtype=np.float64)
    oh_fc = np.stack(fut_cnts["oracle_hybrid_with_full_hp"])
    oh_fn = np.stack(fnr_cnts["oracle_hybrid_with_full_hp"])

    oracle_contrasts = {
        "oracle_greedy_minus_residual_current": bootstrap_contrast(draws, or_ur, ru_ur, u_den_arr, or_fc, ru_fc, or_fn, ru_fn),
        "oracle_hybrid_minus_residual_current": bootstrap_contrast(draws, oh_ur, ru_ur, u_den_arr, oh_fc, ru_fc, oh_fn, ru_fn),
        "oracle_greedy_minus_full_webp": bootstrap_contrast(draws, or_ur, wp_ur, u_den_arr, or_fc, wp_fc, or_fn, wp_fn),
        "oracle_hybrid_minus_full_webp": bootstrap_contrast(draws, oh_ur, wp_ur, u_den_arr, oh_fc, wp_fc, oh_fn, wp_fn),
        "oracle_greedy_minus_tcm": bootstrap_contrast(draws, or_ur, tcm_ur, u_den_arr, or_fc, tcm_fc, or_fn, tcm_fn),
        "oracle_hybrid_minus_tcm": bootstrap_contrast(draws, oh_ur, tcm_ur, u_den_arr, oh_fc, tcm_fc, oh_fn, tcm_fn),
    }

    mechanism_stats = {
        "pairs": len(pair_ids),
        "identical_raw_selections": identical_raw_selections,
        "identical_final_packets": identical_final_packets,
        "mean_jaccard_overlap_rc_vs_ru": float(np.mean(jaccard_overlaps)),
        "median_jaccard_overlap_rc_vs_ru": float(np.median(jaccard_overlaps)),
        "mean_symmetric_diff_cells_rc_vs_ru": float(np.mean(rank_diff_counts)),
        "mean_change_raw_cells_fitted": float(np.mean(change_raw_cell_counts)),
        "mean_current_raw_cells_fitted": float(np.mean(current_raw_cell_counts)),
        "mean_oracle_cells_fitted": float(np.mean(oracle_cell_counts)),
        "mean_cells_needing_refinement_for_psnr30": float(np.mean(positive_gain_cell_counts)),
    }

    return {
        "methods": oracle_summary,
        "contrasts": oracle_contrasts,
        "mechanism": mechanism_stats,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((SOURCE / "method_lock.json").read_text("utf-8"))
    with GROUND_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if TCM_CSV.exists():
        with TCM_CSV.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))

    print("Computing 2x2 four-cell attribution across all classes and caps...", flush=True)
    factorial = compute_factorial_analysis(rows, lock["pair_ids"])

    print("Running Oracle / Ideal Selection diagnostic on 256 confirmation pairs (h=5, cap=8192)...", flush=True)
    oracle = run_oracle_and_selection_diagnostic(lock, rows)

    result = {
        "schema": "factorial-oracle-diagnostic-v1",
        "method_lock_sha256": sha(SOURCE / "method_lock.json"),
        "factorial_2x2": factorial,
        "oracle_diagnostic_h5_b8192": oracle,
    }
    write_json(OUT / "diagnostic.json", result)
    print("Saved diagnostic.json successfully!", flush=True)


if __name__ == "__main__":
    main()
