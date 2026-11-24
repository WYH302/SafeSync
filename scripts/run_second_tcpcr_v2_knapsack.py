"""Full Authoritative GPU Evaluation of SafeSync TC-PCR v2 (Exact Per-Cell Byte-Cost 0-1 Knapsack + WebP q=75).

Protocol locked strictly on the 32 disjoint SECOND development pairs (`residual_development_v1`, dev U30: 0.7554 -> 0.7672,
+1.18 pp with zero RE01 wire-format changes and zero extra header bytes), then evaluated identically on all 256 locked
Cohort 3 confirmation pairs across ALL 3 base codecs (WebP, Scale-Hyperprior, LIC-TCM) and ALL 3 held-out classes
(Building, Tree, Water) under the frozen RTX 3090 DINOv2-S/14 ViT semantic probe.

Computes complete receipt set:
  - Held-out U30, UMR reduction, Later F1, global FNR, raw FNR, mean serialized KiB, 95% paired bootstrap CIs
  - Unknown-over-Change 95% CIs across all 3 base codecs
  - Exact Refined-ROI Later F1 & FNR vs. same-ROI standalone base codecs across all 3 base codecs
  - 3-class macro FNR & U30
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import scipy  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch

from safesync.data import read_pair
from safesync.features import FrozenEncoder
from safesync import residual_evidence as enhancement
from safesync import task_evidence_wire as wire
from scripts.analyze_task_context_confirmation import later_counts, safe_ratio
from scripts.run_locked_second_test import write_json
from scripts.run_residual_evidence_development import pool
from scripts.run_second_feature_mix import load_frozen
from scripts.run_second_repaired_design_experiments import load_mahalanobis_scorer
from scripts.run_unknown_factorial_and_oracle_diagnostic import (
    extract_payload,
    fnr_counts,
)


def paired_bootstrap_ci(draws: np.ndarray, ur_a: np.ndarray, ur_b: np.ndarray, u_den: np.ndarray,
                        fc_a: np.ndarray, fc_b: np.ndarray, fn_a: np.ndarray, fn_b: np.ndarray) -> dict:
    d_f = draws.astype(np.float64)
    den_u = np.maximum(d_f @ u_den, 1.0)
    u_diff = ((d_f @ ur_a) - (d_f @ ur_b)) / den_u
    f1_a = (d_f @ fc_a[:, 0]) / np.maximum(d_f @ fc_a[:, 1], 1.0)
    f1_b = (d_f @ fc_b[:, 0]) / np.maximum(d_f @ fc_b[:, 1], 1.0)
    f1_diff = f1_a - f1_b
    fn_r_a = (d_f @ fn_a[:, 0]) / np.maximum(d_f @ fn_a[:, 1], 1.0)
    fn_r_b = (d_f @ fn_b[:, 0]) / np.maximum(d_f @ fn_b[:, 1], 1.0)
    fn_diff = fn_r_a - fn_r_b
    return {
        "u30_diff": float((ur_a.sum() - ur_b.sum()) / max(u_den.sum(), 1.0)),
        "u30_ci95": [float(np.percentile(u_diff, 2.5)), float(np.percentile(u_diff, 97.5))],
        "later_f1_diff": float(safe_ratio(fc_a[:, 0].sum(), fc_a[:, 1].sum()) - safe_ratio(fc_b[:, 0].sum(), fc_b[:, 1].sum())),
        "later_f1_ci95": [float(np.percentile(f1_diff, 2.5)), float(np.percentile(f1_diff, 97.5))],
        "fnr_diff": float(safe_ratio(fn_a[:, 0].sum(), fn_a[:, 1].sum()) - safe_ratio(fn_b[:, 0].sum(), fn_b[:, 1].sum())),
        "fnr_ci95": [float(np.percentile(fn_diff, 2.5)), float(np.percentile(fn_diff, 97.5))],
    }

OUT = ROOT / "results/task_evidence/second_tcpcr_v2_knapsack_v1"
SOURCE = ROOT / "results/task_evidence/residual_confirmation_v1"
TCM_DIR = ROOT / "results/task_evidence/tcm_control_v1"
DATA = ROOT / "data/second_residual_test_v1"
CLASS_NAMES = {5: "Building", 3: "Tree", 4: "Water"}
CAP = 8192
BASES = ("webp", "hyperprior", "tcm")
Q_RESID = 75  # Dev-locked on residual_development_v1 (32 disjoint pairs)


def encode_re01_q(base_packet: bytes, current: np.ndarray, decoded_base: np.ndarray, indices, q: int = Q_RESID) -> bytes:
    ids = np.asarray(indices, dtype="<u2")
    diff = current.astype(np.int16) - decoded_base.astype(np.int16)
    img = np.clip(np.rint(diff / 2 + 128), 0, 255).astype(np.uint8)
    atlas = wire.encode_atlas(img, ids, codec="webp", quality=q, block=64)
    return enhancement.HEADER.pack(b"RE01", 1, 64, len(ids), len(base_packet), len(atlas)) + ids.tobytes() + base_packet + atlas


def fit_re01_q(prior, labels, current, base_packet, base_rgb, ranking, cap: int, q: int = Q_RESID):
    order = np.argsort(-ranking, kind="stable")

    def make(count: int) -> bytes:
        inner = encode_re01_q(base_packet, current, base_rgb, order[:count], q=q)
        return wire.encode(prior, labels, observation=1, payload=inner, image_mode=wire.LEARNED)[0]

    best = make(0)
    if len(best) + wire.ACK_BYTES > cap:
        raise ValueError("Base packet + task state exceed cap")
    selected, lo, hi = 0, 1, 64
    while lo <= hi:
        mid = (lo + hi) // 2
        pkt = make(mid)
        if len(pkt) + wire.ACK_BYTES <= cap:
            best, selected, lo = pkt, mid, mid + 1
        else:
            hi = mid - 1
    for count in range(selected + 1, min(selected + 3, 64) + 1):
        pkt = make(count)
        if len(pkt) + wire.ACK_BYTES <= cap:
            best, selected = pkt, count
    return best, selected, order[:selected]


def main() -> None:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((SOURCE / "method_lock.json").read_text("utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, lock["backbone"])
    print(f"FrozenEncoder ready on {encoder.device} ({encoder.metadata.get('gpu')}) in {time.perf_counter()-t_start:.1f}s", flush=True)

    with np.load(ROOT / "results/future_semantic_probe/v1/prototypes.npz", allow_pickle=False) as st:
        eval_means, eval_classes = st["means"], st["labels"]

    cache_file = OUT / "probe_scores_cache.npz"
    feat_cache: dict[str, np.ndarray] = {}
    for cf in (
        ROOT / "results/task_evidence/second_repaired_design_v1/probe_scores_cache.npz",
        ROOT / "results/task_evidence/second_tcpcr_universal_v1/probe_scores_cache.npz",
        cache_file,
    ):
        if cf.exists():
            with np.load(cf, allow_pickle=False) as st:
                for k in st.files:
                    feat_cache[k] = st[k]
    print(f"Loaded {len(feat_cache)} cached DINOv2 probe score entries", flush=True)

    def get_scores(arr: np.ndarray) -> np.ndarray:
        sh = hashlib.sha256(arr.tobytes()).hexdigest()
        if sh not in feat_cache:
            vals, _ = encoder.extract(Image.fromarray(arr))
            vals = vals.astype(np.float16).astype(np.float32)
            feat_cache[sh] = (vals @ eval_means.T).astype(np.float32)
        return feat_cache[sh]

    pair_ids = sorted(lock["pair_ids"])
    draws = np.random.default_rng(20261107).multinomial(256, np.full(256, 1.0 / 256.0), size=10000)

    res_rows = list(csv.DictReader((ROOT / "results/task_evidence/residual_confirmation_ground_v1/records.csv").open(encoding="utf-8")))
    tcm_rows = list(csv.DictReader((ROOT / "results/task_evidence/tcm_ground_v1/records.csv").open(encoding="utf-8")))
    by_key = {(r["pair_id"], int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in res_rows + tcm_rows}

    results_by_class: dict[str, dict] = {}
    all_summary_rows = []
    refined_roi_rows = []

    for h in (5, 3, 4):
        cname = CLASS_NAMES[h]
        t_cls = time.perf_counter()
        means_n, maha_score = load_mahalanobis_scorer(h)
        print(f"\n=== [Class: {cname} (h={h})] Phase 1: Parallel CPU TC-PCR v2 (0-1 Knapsack + q75) Packing ===", flush=True)

        def process_pair_cpu(pid: str) -> dict:
            pair = pairs[pid]
            folder = SOURCE / "pairs" / pid
            tfolder = TCM_DIR / "pairs" / pid
            _, after, old, label, changed = read_pair(ROOT, pair)
            current = np.asarray(after)
            known, unknown = helper["evaluation_masks"](old, label, changed, h)
            future_mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed

            ready = json.loads((folder / "sender_inputs_ready.json").read_text("utf-8"))
            common_info = ready["common"][str(h)]
            boot = (ROOT / common_info["bootstrap_path"]).read_bytes()
            head = wire.HEADER.unpack_from(boot)
            blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), head[1], head[2])
            prior, _ = wire.receive(blank, boot)
            service, _ = wire.receive(prior, (ROOT / common_info["labels_only_path"]).read_bytes())

            with np.load(folder / "sender_features.npz", allow_pickle=False) as stored:
                f0, f1 = stored["before"].astype(np.float32), stored["current"].astype(np.float32)
            f0n = f0 / np.maximum(np.linalg.norm(f0, axis=1, keepdims=True), 1e-8)
            f1n = f1 / np.maximum(np.linalg.norm(f1, axis=1, keepdims=True), 1e-8)

            cos_d = (1.0 - np.sum(f0n * f1n, axis=1)).reshape(16, 16)
            s0 = np.max(f0n @ means_n.T, axis=1).reshape(16, 16)
            s1 = np.max(f1n @ means_n.T, axis=1).reshape(16, 16)
            m0 = maha_score(f0).reshape(16, 16)
            m1 = maha_score(f1).reshape(16, 16)
            m_em = (rankdata(np.maximum(m1 - m0, 0.0).reshape(-1)) / 256.0).reshape(16, 16)
            unk_tok = (1.0 - s1) + 1.5 * np.maximum(s0 - s1, 0.0) + 0.5 * m_em

            pix_diff = np.mean(np.abs(current.astype(np.float32) - prior.rgb.astype(np.float32)) / 255.0, axis=2)
            w_chg_pix = np.maximum(cos_d - 0.10, 0.0).repeat(32, 0).repeat(32, 1) * (0.2 + pix_diff)
            w_unk_pix = (np.maximum(cos_d - 0.10, 0.0) * (0.4 + unk_tok)).repeat(32, 0).repeat(32, 1) * (0.2 + pix_diff)
            chg_rank = rankdata(pool(cos_d.reshape(-1))) / 64.0
            unk_rank = rankdata(pool((cos_d * (0.4 + unk_tok)).reshape(-1))) / 64.0

            per_base = {}
            for base in BASES:
                if base == "webp":
                    b_pkt_path = folder / f"h{h}_full_webp_s0_b4096.bin"
                    b_rgb_path = folder / f"h{h}_full_webp_s0_b4096.received.png"
                    f_pkt_path = folder / f"h{h}_full_webp_s0_b8192.bin"
                    f_rgb_path = folder / f"h{h}_full_webp_s0_b8192.received.png"
                    locked_key = "full_webp"
                elif base == "hyperprior":
                    b_pkt_path = folder / f"h{h}_hyperprior_s0_b4096.bin"
                    b_rgb_path = folder / f"h{h}_hyperprior_s0_b4096.received.png"
                    f_pkt_path = folder / f"h{h}_hyperprior_s0_b8192.bin"
                    f_rgb_path = folder / f"h{h}_hyperprior_s0_b8192.received.png"
                    locked_key = "hyperprior"
                else:
                    b_pkt_path = tfolder / f"h{h}_b4096.bin"
                    b_rgb_path = tfolder / f"h{h}_b4096.received.png"
                    f_pkt_path = tfolder / f"h{h}_b8192.bin"
                    f_rgb_path = tfolder / f"h{h}_b8192.received.png"
                    locked_key = "tcm"

                base_inner = extract_payload(b_pkt_path.read_bytes())
                base_rgb = np.asarray(Image.open(b_rgb_path).convert("RGB"))
                full_pkt = f_pkt_path.read_bytes()
                full_rgb = np.asarray(Image.open(f_rgb_path).convert("RGB"))

                t_pack0 = time.perf_counter()
                all64_inner = encode_re01_q(base_inner, current, base_rgb, list(range(64)), q=Q_RESID)
                rgb_all64 = enhancement.decode(all64_inner, (512, 512), lambda p: base_rgb)
                err_b = np.mean(((current.astype(np.float64) - base_rgb.astype(np.float64)) / 255.0) ** 2, axis=2)
                err_r = np.mean(((current.astype(np.float64) - rgb_all64.astype(np.float64)) / 255.0) ** 2, axis=2)
                err_f = np.mean(((current.astype(np.float64) - full_rgb.astype(np.float64)) / 255.0) ** 2, axis=2)

                # Exact per-cell residual WebP byte cost for Rate-Distortion 0-1 Knapsack efficiency
                rimg = np.clip(np.rint((current.astype(np.int16) - base_rgb.astype(np.int16)) / 2 + 128), 0, 255).astype(np.uint8)
                cell_costs = np.array(
                    [len(wire.encode_atlas(rimg, np.array([j], dtype="<u2"), codec="webp", quality=Q_RESID, block=64)) for j in range(64)],
                    dtype=np.float64,
                )
                eff_denom = np.maximum(cell_costs - 40.0, 20.0)

                def eval_tcpcr_v2(w_pix: np.ndarray, base_rank: np.ndarray):
                    wc = w_pix.reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1)
                    eb = (w_pix * err_b).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    er = (w_pix * err_r).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    ebu = err_b.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1)
                    eru = err_r.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1)
                    cross = ((eb > 0.9e-3) & (er <= 1.15e-3)) | ((ebu > 0.9e-3) & (eru <= 1.15e-3))
                    good = (eb <= 0.85e-3) & (ebu <= 0.85e-3)
                    eff = wc / eff_denom
                    sc = np.where(
                        good,
                        -10.0 + base_rank,
                        np.where(
                            cross,
                            10.0 + rankdata(eff) / 64.0 + 0.5 * base_rank,
                            1.0 + 0.5 * rankdata(eff * np.maximum(eb - er, 0.0)) / 64.0 + 0.5 * base_rank,
                        ),
                    )
                    pkt, n_sel, sel_ids = fit_re01_q(prior, service.labels, current, base_inner, base_rgb, sc, CAP, q=Q_RESID)
                    pay = extract_payload(pkt)
                    rgb_cand = enhancement.decode(pay, (512, 512), lambda p: base_rgb)
                    err_c = np.mean(((current.astype(np.float64) - rgb_cand.astype(np.float64)) / 255.0) ** 2, axis=2)
                    ec = (w_pix * err_c).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    ef = (w_pix * err_f).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    qc = np.sum(wc * np.maximum(1.0 - ec / 1.1e-3, 0.0))
                    qf = np.sum(wc * np.maximum(1.0 - ef / 1.1e-3, 0.0))
                    pass_c = np.sum(wc * ((ec <= 1.05e-3) | (err_c.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1) <= 1.0e-3)))
                    pass_f = np.sum(wc * ((ef <= 1.05e-3) | (err_f.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1) <= 1.0e-3)))
                    use_re = bool(pass_c > pass_f or qc > qf * 1.02)
                    return rgb_cand, len(pkt), len(full_pkt), use_re, sel_ids

                rgb_c_cand, bytes_c, bytes_f, use_c, _ = eval_tcpcr_v2(w_chg_pix, chg_rank)
                rgb_u_cand, bytes_u, _, use_u, sel_u_ids = eval_tcpcr_v2(w_unk_pix, unk_rank)
                pack_ms = (time.perf_counter() - t_pack0) * 1000.0

                rec = by_key[(pid, h, CAP, locked_key, 0)]
                cm_st = np.array(json.loads(rec["future_confusion"]), float)
                st_ur = float(rec["unknown_retained_pixels_30"])
                st_km = 0.0 if not rec["known_mse"] else float(rec["known_mse"]) * float(known.sum())
                st_bytes = float(rec["down_bytes"]) + float(rec["ack_bytes"])

                im_c = helper["image_metrics"](current, rgb_c_cand, known, unknown) if use_c else None
                im_u = helper["image_metrics"](current, rgb_u_cand, known, unknown)

                roi_mask = np.zeros((512, 512), dtype=bool)
                for cid in sel_u_ids:
                    cy, cx = divmod(int(cid), 8)
                    roi_mask[cy * 64 : (cy + 1) * 64, cx * 64 : (cx + 1) * 64] = True

                per_base[base] = {
                    "st": (st_ur, st_km, cm_st, st_bytes, full_rgb),
                    "c": (use_c, rgb_c_cand if use_c else None, im_c, float(bytes_c + wire.ACK_BYTES) if use_c else st_bytes),
                    "u_raw": (True, rgb_u_cand, im_u, float(bytes_u + wire.ACK_BYTES)),
                    "u_sel_use_re": use_u,
                    "roi_mask": roi_mask,
                    "pack_ms": pack_ms,
                }

            return {
                "pid": pid,
                "u_den": float(unknown.sum()),
                "k_den": float(known.sum()),
                "label": label,
                "future_mask": future_mask,
                "per_base": per_base,
            }

        pair_results = []
        with ThreadPoolExecutor(max_workers=12) as pool_exec:
            for idx, res in enumerate(pool_exec.map(process_pair_cpu, pair_ids)):
                pair_results.append(res)
                if (idx + 1) % 64 == 0:
                    print(f"  [{cname}] CPU packed {idx+1}/256 pairs ({time.perf_counter()-t_cls:.1f}s)", flush=True)

        print(f"=== [Class: {cname} (h={h})] Phase 2: Frozen GPU DINOv2 Probe Scoring + Refined-ROI ===", flush=True)
        u_den, k_den = [], []
        arrays: dict[str, dict] = {}
        roi_accum: dict[str, dict] = {}
        for base in BASES:
            for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
                arrays[f"{base}_{mode}"] = {"ur": [], "km": [], "fc": [], "fn": [], "bytes": [], "re_count": 0, "pack_ms": []}
            roi_accum[base] = {"tcpcr_fc": np.zeros(2), "tcpcr_fn": np.zeros(2), "stand_fc": np.zeros(2), "stand_fn": np.zeros(2)}

        for pr in pair_results:
            u_den.append(pr["u_den"])
            k_den.append(pr["k_den"])
            label = pr["label"]
            future_mask = pr["future_mask"]

            def cm_of(rgb_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
                if not np.any(mask):
                    return np.zeros((6, 6), dtype=np.float64)
                scores = get_scores(rgb_arr)
                pred = eval_classes[np.argmax(scores, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                return np.bincount((label[mask].astype(int) - 1) * 6 + pred[mask].astype(int) - 1, minlength=36).reshape(6, 6).astype(np.float64)

            for base in BASES:
                pb = pr["per_base"][base]
                st_ur, st_km, cm_st, st_bytes, full_rgb = pb["st"]
                st_arr = arrays[f"{base}_standalone"]
                st_arr["ur"].append(st_ur)
                st_arr["km"].append(st_km)
                st_arr["fc"].append(later_counts(cm_st, h))
                st_arr["fn"].append(fnr_counts(cm_st, h))
                st_arr["bytes"].append(st_bytes)

                # Change selected
                use_c, rgb_c, im_c, b_c = pb["c"]
                dst_c = arrays[f"{base}_tcpcr_change"]
                if use_c:
                    cm_c = cm_of(rgb_c, future_mask)
                    dst_c["ur"].append(float(im_c["unknown_retained_pixels_30"]))
                    dst_c["km"].append(0.0 if im_c["known_mse"] is None else float(im_c["known_mse"]) * pr["k_den"])
                    dst_c["fc"].append(later_counts(cm_c, h))
                    dst_c["fn"].append(fnr_counts(cm_c, h))
                    dst_c["bytes"].append(b_c)
                    dst_c["re_count"] += 1
                else:
                    dst_c["ur"].append(st_ur)
                    dst_c["km"].append(st_km)
                    dst_c["fc"].append(later_counts(cm_st, h))
                    dst_c["fn"].append(fnr_counts(cm_st, h))
                    dst_c["bytes"].append(st_bytes)

                # Unknown raw & selected
                _, rgb_u, im_u, b_u = pb["u_raw"]
                cm_u = cm_of(rgb_u, future_mask)
                ur_u = float(im_u["unknown_retained_pixels_30"])
                km_u = 0.0 if im_u["known_mse"] is None else float(im_u["known_mse"]) * pr["k_den"]
                fc_u = later_counts(cm_u, h)
                fn_u = fnr_counts(cm_u, h)

                dst_raw = arrays[f"{base}_tcpcr_unknown_raw"]
                dst_raw["ur"].append(ur_u)
                dst_raw["km"].append(km_u)
                dst_raw["fc"].append(fc_u)
                dst_raw["fn"].append(fn_u)
                dst_raw["bytes"].append(b_u)
                dst_raw["re_count"] += 1
                dst_raw["pack_ms"].append(pb["pack_ms"])

                dst_u = arrays[f"{base}_tcpcr_unknown"]
                dst_u["pack_ms"].append(pb["pack_ms"])
                if pb["u_sel_use_re"]:
                    dst_u["ur"].append(ur_u)
                    dst_u["km"].append(km_u)
                    dst_u["fc"].append(fc_u)
                    dst_u["fn"].append(fn_u)
                    dst_u["bytes"].append(b_u)
                    dst_u["re_count"] += 1
                    # Refined ROI evaluation on selected pairs
                    rmask = future_mask & pb["roi_mask"]
                    if np.any(rmask):
                        cm_roi_u = cm_of(rgb_u, rmask)
                        cm_roi_s = cm_of(full_rgb, rmask)
                        roi_accum[base]["tcpcr_fc"] += np.array(later_counts(cm_roi_u, h), dtype=float)
                        roi_accum[base]["tcpcr_fn"] += np.array(fnr_counts(cm_roi_u, h), dtype=float)
                        roi_accum[base]["stand_fc"] += np.array(later_counts(cm_roi_s, h), dtype=float)
                        roi_accum[base]["stand_fn"] += np.array(fnr_counts(cm_roi_s, h), dtype=float)
                else:
                    dst_u["ur"].append(st_ur)
                    dst_u["km"].append(st_km)
                    dst_u["fc"].append(later_counts(cm_st, h))
                    dst_u["fn"].append(fnr_counts(cm_st, h))
                    dst_u["bytes"].append(st_bytes)

        u_den_a = np.array(u_den, float)
        k_den_a = np.array(k_den, float)
        rows_dict = {}
        contrasts = {}

        for base in BASES:
            for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
                key = f"{base}_{mode}"
                a = arrays[key]
                ur_a = np.array(a["ur"], float)
                km_a = np.array(a["km"], float)
                fc_a = np.array(a["fc"], float)
                fn_a = np.array(a["fn"], float)
                u30 = float(ur_a.sum() / max(u_den_a.sum(), 1.0))
                f1 = float(safe_ratio(fc_a[:, 0].sum(), fc_a[:, 1].sum()))
                fnr = float(safe_ratio(fn_a[:, 0].sum(), fn_a[:, 1].sum()))
                kmse = float(km_a.sum() / max(k_den_a.sum(), 1.0))
                kb = float(np.mean(a["bytes"])) / 1024.0
                row = {
                    "heldout_class": cname,
                    "method": key,
                    "u30": u30,
                    "later_f1": f1,
                    "fnr": fnr,
                    "known_mse": kmse,
                    "mean_total_KiB": kb,
                    "re_selected_pct": a["re_count"] / 256.0 * 100.0,
                    "mean_pack_ms": float(np.mean(a["pack_ms"])) if a["pack_ms"] else 0.0,
                }
                rows_dict[key] = row
                all_summary_rows.append(row)

            # Paired 95% CIs: unknown vs standalone AND unknown vs change
            for comp_name, other_mode in (("vs_standalone", "standalone"), ("unknown_over_change", "tcpcr_change")):
                a1 = arrays[f"{base}_tcpcr_unknown"]
                a2 = arrays[f"{base}_{other_mode}"]
                c = paired_bootstrap_ci(
                    draws,
                    np.array(a1["ur"], float), np.array(a2["ur"], float), u_den_a,
                    np.array(a1["fc"], float), np.array(a2["fc"], float),
                    np.array(a1["fn"], float), np.array(a2["fn"], float),
                )
                contrasts[f"{base}_{comp_name}"] = c

            ra = roi_accum[base]
            r_row = {
                "heldout_class": cname,
                "heldout_id": h,
                "base_codec": base,
                "tcpcr_v2_refined_roi_later_f1": float(safe_ratio(ra["tcpcr_fc"][0], ra["tcpcr_fc"][1])),
                "standalone_same_roi_later_f1": float(safe_ratio(ra["stand_fc"][0], ra["stand_fc"][1])),
                "tcpcr_v2_refined_roi_fnr": float(safe_ratio(ra["tcpcr_fn"][0], ra["tcpcr_fn"][1])),
                "standalone_same_roi_fnr": float(safe_ratio(ra["stand_fn"][0], ra["stand_fn"][1])),
            }
            refined_roi_rows.append(r_row)
            r_u = rows_dict[f"{base}_tcpcr_unknown"]
            r_r = rows_dict[f"{base}_tcpcr_unknown_raw"]
            r_s = rows_dict[f"{base}_standalone"]
            c_st = contrasts[f"{base}_vs_standalone"]
            c_uc = contrasts[f"{base}_unknown_over_change"]
            print(
                f"  [{cname:8s} | {base:10s}] U30: {r_s['u30']:.4f} -> {r_u['u30']:.4f} "
                f"({(r_u['u30']-r_s['u30'])*100:+.2f} pp, 95% CI [{c_st['u30_ci95'][0]*100:+.2f},{c_st['u30_ci95'][1]*100:+.2f}]) | "
                f"UoC: {c_uc['u30_diff']*100:+.2f} pp [{c_uc['u30_ci95'][0]*100:+.2f},{c_uc['u30_ci95'][1]*100:+.2f}] | "
                f"F1={r_u['later_f1']:.4f} (RefROI {r_row['tcpcr_v2_refined_roi_later_f1']:.4f} vs {r_row['standalone_same_roi_later_f1']:.4f}) | "
                f"FNR={r_u['fnr']:.4f} (raw {r_r['fnr']:.4f}, RefROI {r_row['tcpcr_v2_refined_roi_fnr']:.4f}) | KiB={r_u['mean_total_KiB']:.2f}",
                flush=True,
            )

        results_by_class[cname] = {"rows": rows_dict, "contrasts": contrasts}

    # Macro 3-class averages
    macro_3class = {}
    for base in BASES:
        for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
            key = f"{base}_{mode}"
            macro_3class[key] = {
                "u30_macro": float(np.mean([results_by_class[c]["rows"][key]["u30"] for c in ("Building", "Tree", "Water")])),
                "later_f1_macro": float(np.mean([results_by_class[c]["rows"][key]["later_f1"] for c in ("Building", "Tree", "Water")])),
                "fnr_macro": float(np.mean([results_by_class[c]["rows"][key]["fnr"] for c in ("Building", "Tree", "Water")])),
            }
    print("\n=== 3-CLASS MACRO SUMMARY ===")
    for k in ("webp_standalone", "webp_tcpcr_unknown", "hyperprior_standalone", "hyperprior_tcpcr_unknown", "tcm_standalone", "tcm_tcpcr_unknown", "tcm_tcpcr_unknown_raw"):
        m = macro_3class[k]
        print(f"  {k:24s}: macro_U30={m['u30_macro']:.4f} | macro_F1={m['later_f1_macro']:.4f} | macro_FNR={m['fnr_macro']:.4f}", flush=True)

    np.savez_compressed(cache_file, **feat_cache)
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(all_summary_rows)
    write_json(OUT / "refined_roi_summary.json", {"schema": "second-tcpcr-v2-refined-roi-v1", "status": "complete", "rows": refined_roi_rows})
    write_json(OUT / "analysis.json", {
        "schema": "second-tcpcr-v2-knapsack-v1",
        "status": "complete",
        "dev_locked_config": {"quality": Q_RESID, "ranking": "exact_per_cell_webp_byte_cost_01_knapsack", "wire_format": "RE01_unchanged"},
        "elapsed_s": time.perf_counter() - t_start,
        "results_by_class": results_by_class,
        "macro_3class": macro_3class,
    })
    print(f"\nSaved TC-PCR v2 complete GPU receipts in {time.perf_counter()-t_start:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
