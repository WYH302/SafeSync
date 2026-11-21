"""Universal Plug-and-Play Evaluation of SafeSync TC-PCR Across All Three Base Codecs (WebP, Scale-Hyperprior, LIC-TCM) and All Three Held-Out Classes (Building, Tree, Water) on SECOND Cohort 3 (256 Locked Pairs, 8-KiB Cap).

Zero model retraining, zero test-label peeking:
  - Uses identical frozen DINOv2-S/14 backbone, frozen Ledoit-Wolf known-class precision matrices,
    and frozen all-class ViT semantic probe.
  - Compares standalone 8-KiB codec (WebP, Hyperprior, LIC-TCM) against 4-KiB base + TC-PCR RE01
    residual refinement (total serialized transaction <= 8 KiB) across all 256 confirmation pairs.
  - Parallelizes pure-CPU PIL WebP70 residual packing across 8 threads (bit-identical) and reuses
    locked ground confusion counts whenever the selector keeps the 8-KiB full base packet.
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
from scripts.run_residual_evidence_development import fit, pool
from scripts.run_second_feature_mix import load_frozen
from scripts.run_second_repaired_design_experiments import load_mahalanobis_scorer
from scripts.run_unknown_factorial_and_oracle_diagnostic import (
    bootstrap_contrast,
    extract_payload,
    fnr_counts,
)

OUT = ROOT / "results/task_evidence/second_tcpcr_universal_v1"
SOURCE = ROOT / "results/task_evidence/residual_confirmation_v1"
TCM_DIR = ROOT / "results/task_evidence/tcm_control_v1"
DATA = ROOT / "data/second_residual_test_v1"
CLASS_NAMES = {5: "Building", 3: "Tree", 4: "Water"}
CAP = 8192
BASES = ("webp", "hyperprior", "tcm")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def isolated_features(image):
        values, timings = encoder.extract(image)
        return values.astype(np.float16).astype(np.float32), timings

    encoder.cached = isolated_features
    with np.load(ROOT / "results/future_semantic_probe/v1/prototypes.npz", allow_pickle=False) as st:
        eval_means, eval_classes = st["means"], st["labels"]

    cache_file = OUT / "probe_scores_cache.npz"
    prev_cache = ROOT / "results/task_evidence/second_repaired_design_v1/probe_scores_cache.npz"
    feat_cache: dict[str, np.ndarray] = {}
    for cf in (prev_cache, cache_file):
        if cf.exists():
            with np.load(cf, allow_pickle=False) as st:
                for k in st.files:
                    feat_cache[k] = st[k]
    print(f"Loaded {len(feat_cache)} cached DINOv2 probe score entries", flush=True)

    def get_scores(arr: np.ndarray) -> np.ndarray:
        sh = hashlib.sha256(arr.tobytes()).hexdigest()
        if sh not in feat_cache:
            feats, _ = encoder.cached(Image.fromarray(arr))
            feat_cache[sh] = (feats @ eval_means.T).astype(np.float32)
        return feat_cache[sh]

    pair_ids = sorted(lock["pair_ids"])
    draws = np.random.default_rng(20261107).multinomial(256, np.full(256, 1.0 / 256.0), size=10000)

    res_rows = list(csv.DictReader((ROOT / "results/task_evidence/residual_confirmation_ground_v1/records.csv").open(encoding="utf-8")))
    tcm_rows = list(csv.DictReader((ROOT / "results/task_evidence/tcm_ground_v1/records.csv").open(encoding="utf-8")))
    by_key = {(r["pair_id"], int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in res_rows + tcm_rows}

    results_by_class: dict[str, dict] = {}
    all_summary_rows = []

    for h in (5, 3, 4):
        cname = CLASS_NAMES[h]
        t_cls = time.perf_counter()
        means_n, maha_score = load_mahalanobis_scorer(h)
        print(f"\n=== [Class: {cname} (h={h})] Phase 1: Parallel CPU TC-PCR Packet Packing (256 pairs x 3 bases) ===", flush=True)

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

                all64_inner = enhancement.encode(base_inner, current, base_rgb, list(range(64)), mode=enhancement.RESIDUAL)
                rgb_all64 = enhancement.decode(all64_inner, (512, 512), lambda p: base_rgb)
                err_b = np.mean(((current.astype(np.float64) - base_rgb.astype(np.float64)) / 255.0) ** 2, axis=2)
                err_r = np.mean(((current.astype(np.float64) - rgb_all64.astype(np.float64)) / 255.0) ** 2, axis=2)
                err_f = np.mean(((current.astype(np.float64) - full_rgb.astype(np.float64)) / 255.0) ** 2, axis=2)

                def eval_tcpcr(w_pix: np.ndarray, base_rank: np.ndarray):
                    wc = w_pix.reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1)
                    eb = (w_pix * err_b).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    er = (w_pix * err_r).reshape(8, 64, 8, 64).sum((1, 3)).reshape(-1) / np.maximum(wc, 1e-9)
                    ebu = err_b.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1)
                    eru = err_r.reshape(8, 64, 8, 64).mean((1, 3)).reshape(-1)
                    cross = ((eb > 0.9e-3) & (er <= 1.15e-3)) | ((ebu > 0.9e-3) & (eru <= 1.15e-3))
                    good = (eb <= 0.85e-3) & (ebu <= 0.85e-3)
                    sc = np.where(
                        good,
                        -10.0 + base_rank,
                        np.where(
                            cross,
                            10.0 + rankdata(wc) / 64.0 + base_rank,
                            1.0 + 0.5 * rankdata(wc * np.maximum(eb - er, 0.0)) / 64.0 + 0.5 * base_rank,
                        ),
                    )
                    pkt, _ = fit(prior, service.labels, current, base_inner, base_rgb, sc, enhancement.RESIDUAL, CAP)
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
                    return rgb_cand, len(pkt), len(full_pkt), use_re

                rgb_c_cand, bytes_c, bytes_f, use_c = eval_tcpcr(w_chg_pix, chg_rank)
                rgb_u_cand, bytes_u, _, use_u = eval_tcpcr(w_unk_pix, unk_rank)

                rec = by_key[(pid, h, CAP, locked_key, 0)]
                cm_st = np.array(json.loads(rec["future_confusion"]), float)
                st_ur = float(rec["unknown_retained_pixels_30"])
                st_km = 0.0 if not rec["known_mse"] else float(rec["known_mse"]) * float(known.sum())
                st_bytes = float(rec["down_bytes"]) + float(rec["ack_bytes"])

                im_c = helper["image_metrics"](current, rgb_c_cand, known, unknown) if use_c else None
                im_u = helper["image_metrics"](current, rgb_u_cand, known, unknown)

                per_base[base] = {
                    "st": (st_ur, st_km, cm_st, st_bytes),
                    "c": (use_c, rgb_c_cand if use_c else None, im_c, float(bytes_c + wire.ACK_BYTES) if use_c else st_bytes),
                    "u_raw": (True, rgb_u_cand, im_u, float(bytes_u + wire.ACK_BYTES)),
                    "u_sel_use_re": use_u,
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
        with ThreadPoolExecutor(max_workers=8) as pool_exec:
            for idx, res in enumerate(pool_exec.map(process_pair_cpu, pair_ids)):
                pair_results.append(res)
                if (idx + 1) % 64 == 0:
                    print(f"  [{cname}] CPU packed {idx+1}/256 pairs ({time.perf_counter()-t_cls:.1f}s)", flush=True)

        print(f"=== [Class: {cname} (h={h})] Phase 2: Frozen GPU DINOv2 Probe Scoring ===", flush=True)
        u_den = []
        k_den = []
        arrays: dict[str, dict] = {}
        for base in BASES:
            for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
                arrays[f"{base}_{mode}"] = {"ur": [], "km": [], "fc": [], "fn": [], "bytes": [], "re_count": 0}

        for idx, pr in enumerate(pair_results):
            u_den.append(pr["u_den"])
            k_den.append(pr["k_den"])
            label = pr["label"]
            future_mask = pr["future_mask"]

            def cm_of(rgb_arr: np.ndarray) -> np.ndarray:
                scores = get_scores(rgb_arr)
                pred = eval_classes[np.argmax(scores, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                return (
                    np.bincount(
                        (label[future_mask].astype(int) - 1) * 6 + pred[future_mask].astype(int) - 1, minlength=36
                    )
                    .reshape(6, 6)
                    .astype(np.float64)
                )

            for base in BASES:
                pb = pr["per_base"][base]
                st_ur, st_km, cm_st, st_bytes = pb["st"]
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
                    cm_c = cm_of(rgb_c)
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
                cm_u = cm_of(rgb_u)
                ur_u = float(im_u["unknown_retained_pixels_30"])
                km_u = 0.0 if im_u["known_mse"] is None else float(im_u["known_mse"]) * pr["k_den"]
                fc_u = later_counts(cm_u, h)
                fn_u = fnr_counts(cm_u, h)

                dst_ur = arrays[f"{base}_tcpcr_unknown_raw"]
                dst_ur["ur"].append(ur_u)
                dst_ur["km"].append(km_u)
                dst_ur["fc"].append(fc_u)
                dst_ur["fn"].append(fn_u)
                dst_ur["bytes"].append(b_u)
                dst_ur["re_count"] += 1

                dst_us = arrays[f"{base}_tcpcr_unknown"]
                if pb["u_sel_use_re"]:
                    dst_us["ur"].append(ur_u)
                    dst_us["km"].append(km_u)
                    dst_us["fc"].append(fc_u)
                    dst_us["fn"].append(fn_u)
                    dst_us["bytes"].append(b_u)
                    dst_us["re_count"] += 1
                else:
                    dst_us["ur"].append(st_ur)
                    dst_us["km"].append(st_km)
                    dst_us["fc"].append(later_counts(cm_st, h))
                    dst_us["fn"].append(fnr_counts(cm_st, h))
                    dst_us["bytes"].append(st_bytes)

            if (idx + 1) % 64 == 0:
                print(f"  [{cname}] GPU scored {idx+1}/256 pairs ({time.perf_counter()-t_cls:.1f}s, cache={len(feat_cache)})", flush=True)

        u_den_arr = np.array(u_den, dtype=np.float64)
        k_den_arr = np.array(k_den, dtype=np.float64)
        cls_summary = {"rows": {}, "contrasts": {}}

        for base in BASES:
            for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
                key = f"{base}_{mode}"
                d = arrays[key]
                ur = np.array(d["ur"], dtype=np.float64)
                km = np.array(d["km"], dtype=np.float64)
                fc = np.stack(d["fc"])
                fn = np.stack(d["fn"])
                u30 = float(ur.sum() / u_den_arr.sum())
                row_obj = {
                    "heldout_class": cname,
                    "heldout_id": h,
                    "base_codec": base,
                    "mode": mode,
                    "u30": u30,
                    "umr30": 1.0 - u30,
                    "known_mse": float(km.sum() / k_den_arr.sum()),
                    "later_f1": float(safe_ratio(fc[:, 0].sum(), fc[:, 1].sum())),
                    "heldout_fnr": float(safe_ratio(fn[:, 0].sum(), fn[:, 1].sum())),
                    "mean_total_KiB": float(np.mean(d["bytes"]) / 1024.0),
                    "re01_selected_pairs": int(d["re_count"]),
                }
                cls_summary["rows"][key] = row_obj
                all_summary_rows.append(row_obj)

            def pack(k):
                d = arrays[k]
                return np.array(d["ur"], float), np.array(d["km"], float), np.stack(d["fc"]), np.stack(d["fn"])

            u_ur, _, u_fc, u_fn = pack(f"{base}_tcpcr_unknown")
            s_ur, _, s_fc, s_fn = pack(f"{base}_standalone")
            c_ur, _, c_fc, c_fn = pack(f"{base}_tcpcr_change")
            cls_summary["contrasts"][f"{base}_tcpcr_vs_standalone"] = bootstrap_contrast(
                draws, u_ur, s_ur, u_den_arr, u_fc, s_fc, u_fn, s_fn
            )
            cls_summary["contrasts"][f"{base}_unknown_vs_change"] = bootstrap_contrast(
                draws, u_ur, c_ur, u_den_arr, u_fc, c_fc, u_fn, c_fn
            )

        t_ur, _, t_fc, t_fn = pack("tcm_tcpcr_unknown")
        h_ur, _, h_fc, h_fn = pack("hyperprior_standalone")
        w_ur, _, w_fc, w_fn = pack("webp_standalone")
        ts_ur, _, ts_fc, ts_fn = pack("tcm_standalone")
        ht_ur, _, ht_fc, ht_fn = pack("hyperprior_tcpcr_unknown")
        cls_summary["contrasts"]["tcm_tcpcr_vs_hyperprior_standalone"] = bootstrap_contrast(
            draws, t_ur, h_ur, u_den_arr, t_fc, h_fc, t_fn, h_fn
        )
        cls_summary["contrasts"]["tcm_tcpcr_vs_webp_standalone"] = bootstrap_contrast(
            draws, t_ur, w_ur, u_den_arr, t_fc, w_fc, t_fn, w_fn
        )
        cls_summary["contrasts"]["hyperprior_tcpcr_vs_tcm_standalone"] = bootstrap_contrast(
            draws, ht_ur, ts_ur, u_den_arr, ht_fc, ts_fc, ht_fn, ts_fn
        )

        results_by_class[cname] = cls_summary
        np.savez_compressed(cache_file, **feat_cache)
        for base in BASES:
            st = cls_summary["rows"][f"{base}_standalone"]
            tp = cls_summary["rows"][f"{base}_tcpcr_unknown"]
            ct = cls_summary["contrasts"][f"{base}_tcpcr_vs_standalone"]
            print(
                f"  [{cname:8s} | {base:10s}] Standalone U30={st['u30']:.4f} (F1={st['later_f1']:.4f}, FNR={st['heldout_fnr']:.4f}) "
                f"-> TC-PCR U30={tp['u30']:.4f} (F1={tp['later_f1']:.4f}, FNR={tp['heldout_fnr']:.4f}, {tp['mean_total_KiB']:.2f} KiB) "
                f"| dU30={100*ct['u30_diff'][0]:+.2f} pp [{100*ct['u30_diff'][1]:+.2f}, {100*ct['u30_diff'][2]:+.2f}], UMR red={100*ct['rel_umr_reduction'][0]:+.1f}%",
                flush=True,
            )

    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        w.writeheader()
        w.writerows(all_summary_rows)

    macro_by_key: dict[str, dict] = {}
    for base in BASES:
        for mode in ("standalone", "tcpcr_change", "tcpcr_unknown", "tcpcr_unknown_raw"):
            key = f"{base}_{mode}"
            vals = [results_by_class[c]["rows"][key] for c in ("Building", "Tree", "Water")]
            macro_by_key[key] = {
                "u30_macro": float(np.mean([v["u30"] for v in vals])),
                "umr30_macro": float(np.mean([v["umr30"] for v in vals])),
                "later_f1_macro": float(np.mean([v["later_f1"] for v in vals])),
                "heldout_fnr_macro": float(np.mean([v["heldout_fnr"] for v in vals])),
                "mean_total_KiB": float(np.mean([v["mean_total_KiB"] for v in vals])),
            }

    write_json(OUT / "analysis.json", {
        "schema": "second-universal-tcpcr-v1",
        "status": "complete",
        "cohort": "SECOND Cohort 3 (256 locked pairs, 8-KiB transaction cap)",
        "results_by_class": results_by_class,
        "macro_3class": macro_by_key,
        "summary_csv_sha256": sha_file(OUT / "summary.csv"),
    })
    print(f"\nALL DONE in {time.perf_counter()-t_start:.1f}s -> {OUT / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
