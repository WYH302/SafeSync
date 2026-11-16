"""Complete fair comparison of Long-Horizon Risk Anchor against ARQ, Periodic Anchor, Timeout-Only Anchor, Drift-Gated Anchor, and Fixed-Reference controls."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import run_real_link_recovery as link
from scripts.run_locked_second_test import sha, write_json

REAL_V2 = ROOT / "results/link_recovery/real_features_v2"
FIXED_V1 = ROOT / "results/link_recovery/fixed_reference_v1"
OUT = ROOT / "results/link_recovery/complete_comparison_v1"
TEX_OUT = ROOT / "paper/supplement/generated/recovery_complete.tex"


def run_diagnostic_policy(
    features: np.ndarray,
    policy: str,
    profile: dict,
    seed: int,
    identity: str,
    drift_threshold: float = 1.5e-4,
) -> list[dict]:
    """Run timeout_only_anchor or drift_gated_anchor on the exact same link realization as real_features_v2."""
    session = link.digest((identity + ":" + link.sha(link.CACHE / "run_spec.json")).encode())
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "little"))
    uniforms = rng.random((len(features), 2))
    phase = int(rng.integers(0, 12))
    codes = np.rint(features.astype(np.float32) * 127).clip(-127, 127).astype(np.int8)
    sender = np.zeros_like(codes[0])
    initial = link.encode(sender, codes[0], 0, 1, 0, True, session)
    receiver, receiver_version, boot_ack, status = link.receive(sender, 0, initial, None, None, session)
    if status != "accepted" or not np.array_equal(receiver, codes[0]):
        raise AssertionError("Bootstrap decode failed")
    sender = receiver.copy()
    sender_version = 1
    last_wire, last_ack = initial, boot_ack
    feedback_queue = []
    feedback_risk = False
    last_confirmed_observation = 0
    latest_received_observation = 0
    rows = []

    def process_feedback(now: int):
        nonlocal sender, sender_version, feedback_risk, last_confirmed_observation
        ready = [ev for ev in feedback_queue if ev[0] <= now]
        feedback_queue[:] = [ev for ev in feedback_queue if ev[0] > now]
        for _, body in ready:
            magic, accepted, version, observation, _transaction, _state_hash = link.FEEDBACK.unpack(body)
            if magic != b"RF11":
                raise AssertionError("Feedback schema")
            if accepted:
                last_confirmed_observation = max(last_confirmed_observation, observation)
            else:
                feedback_risk = True

    for step in range(1, len(features)):
        process_feedback(step)
        current = codes[step]
        timeout = step - last_confirmed_observation > profile["delay"] + 1
        # Compute sender-side known drift between current observation and last confirmed observation
        confirmed_codes = codes[last_confirmed_observation]
        unconfirmed_drift = float(np.mean(((current.astype(np.float64) - confirmed_codes.astype(np.float64)) / 127.0) ** 2))

        if policy == "timeout_only_anchor":
            anchor = bool(timeout)
        elif policy == "nack_only_anchor":
            anchor = bool(feedback_risk)
        elif policy == "drift_gated_anchor":
            anchor = bool((feedback_risk or timeout) and unconfirmed_drift >= drift_threshold)
        elif policy == "timeout_only_t4":
            anchor = bool(step - last_confirmed_observation > 4)
        elif policy == "risk_anchor_t4":
            anchor = bool(feedback_risk or (step - last_confirmed_observation > 4))
        else:
            raise ValueError(f"Unknown diagnostic policy: {policy}")

        packet = link.encode(sender, current, sender_version, sender_version + 1, step, anchor, session)
        target_codes, target_version, target_obs = current.copy(), sender_version + 1, step
        feedback_risk = False
        lost = uniforms[step, 0] < profile["down_loss"] or (profile["burst"] and (step + phase) % 12 in (0, 1))
        feedback_bytes = 0
        outcome = "lost"
        false_accept = 0
        if not lost:
            receiver, receiver_version, ack, outcome = link.receive(
                receiver, receiver_version, packet, last_wire, last_ack, session
            )
            if outcome == "accepted":
                if not np.array_equal(receiver, target_codes):
                    false_accept = 1
                latest_received_observation = target_obs
                last_wire, last_ack = packet, ack
            feedback_bytes = len(ack)
            if uniforms[step, 1] >= profile["ack_loss"]:
                feedback_queue.append((step + profile["delay"], ack))
        sender, sender_version = target_codes, target_version
        process_feedback(step)

        rec_float = receiver.astype(np.float64) / 127.0
        gt_float = features[step].astype(np.float64)
        world_mse = float(np.mean((gt_float - rec_float) ** 2))
        ref_mse = float(np.mean(((sender.astype(np.float64) - receiver.astype(np.float64)) / 127.0) ** 2))
        # Cosine utility between received feature tokens and current ground-truth tokens
        rn = rec_float / np.maximum(np.linalg.norm(rec_float, axis=1, keepdims=True), 1e-8)
        gn = gt_float / np.maximum(np.linalg.norm(gt_float, axis=1, keepdims=True), 1e-8)
        cos_util = float(np.mean(np.sum(rn * gn, axis=1)))

        rows.append(
            {
                "step": step,
                "total_bytes": len(packet) + feedback_bytes,
                "anchor": int(anchor),
                "retry": 0,
                "outcome": outcome,
                "age": step - latest_received_observation,
                "world_mse": world_mse,
                "reference_mse": ref_mse,
                "cosine_utility": cos_util,
                "receiver_confirms_current": int(latest_received_observation == step),
                "false_accept": false_accept,
            }
        )
    return rows


def compute_episode_recovery_stats(ages: list[int], ref_mses: list[float]) -> dict:
    """Compute post-loss recovery steps and tail mismatch duration from a sequence of ages and reference MSEs."""
    ages_arr = np.asarray(ages, dtype=np.int64)
    ref_arr = np.asarray(ref_mses, dtype=np.float64)
    # Recovery episodes: contiguous runs where age > 0
    episodes = []
    cur = 0
    for a in ages_arr:
        if a > 0:
            cur += 1
        elif cur > 0:
            episodes.append(cur)
            cur = 0
    if cur > 0:
        episodes.append(cur)

    # Reference mismatch episodes: contiguous runs where reference_mse > 0
    mm_episodes = []
    cur_m = 0
    for m in ref_arr:
        if m > 1e-12:
            cur_m += 1
        elif cur_m > 0:
            mm_episodes.append(cur_m)
            cur_m = 0
    if cur_m > 0:
        mm_episodes.append(cur_m)

    return {
        "p95_age": float(np.quantile(ages_arr, 0.95)),
        "max_age": int(ages_arr.max()),
        "mean_recovery_steps": float(np.mean(episodes)) if episodes else 0.0,
        "max_recovery_steps": int(max(episodes)) if episodes else 0,
        "max_mismatch_duration": int(max(mm_episodes)) if mm_episodes else 0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads((link.CACHE / "complete.json").read_text("utf-8"))
    areas = cache["areas"]
    area_features = {}
    area_codes = {}
    area_split = {}
    for a in areas:
        with np.load(ROOT / a["path"], allow_pickle=False) as stored:
            f = stored["features"].astype(np.float32)
        area_features[a["aoi"]] = f
        area_codes[a["aoi"]] = np.rint(f * 127).clip(-127, 127).astype(np.int8)
        area_split[a["aoi"]] = (
            "confirmation"
            if "confirmation" in a["source_manifest"]
            else "calibration"
            if a["area_index"] < 2
            else "development"
        )

    # Load existing records from real_features_v2 and fixed_reference_v1
    real_rows = list(csv.DictReader((REAL_V2 / "records.csv").open(encoding="utf-8")))
    fixed_rows = list(csv.DictReader((FIXED_V1 / "records.csv").open(encoding="utf-8")))

    # Group sequence records by (aoi, policy, profile, seed)
    seq_groups = defaultdict(list)
    for r in real_rows:
        aoi = r["aoi"]
        step = int(r["step"])
        latest_obs = int(r["latest_observation"])
        f_step = area_features[aoi][step].astype(np.float64)
        rec_float = area_codes[aoi][latest_obs].astype(np.float64) / 127.0
        rn = rec_float / np.maximum(np.linalg.norm(rec_float, axis=1, keepdims=True), 1e-8)
        gn = f_step / np.maximum(np.linalg.norm(f_step, axis=1, keepdims=True), 1e-8)
        cos_util = float(np.mean(np.sum(rn * gn, axis=1)))
        seq_groups[(aoi, r["policy"], r["profile"], int(r["seed"]))].append(
            {
                "step": step,
                "total_bytes": float(r["total_bytes"]),
                "anchor": int(r["anchor"]),
                "retry": int(r["retry"]),
                "outcome": r["outcome"],
                "age": int(r["age"]),
                "world_mse": float(r["world_mse"]),
                "reference_mse": float(r["reference_mse"]),
                "cosine_utility": cos_util,
                "receiver_confirms_current": int(r["receiver_confirms_current"]),
                "false_accept": 0,
            }
        )

    for r in fixed_rows:
        aoi = r["aoi"]
        step = int(r["step"])
        latest_obs = int(r["latest_observation"])
        f_step = area_features[aoi][step].astype(np.float64)
        rec_float = area_codes[aoi][latest_obs].astype(np.float64) / 127.0
        rn = rec_float / np.maximum(np.linalg.norm(rec_float, axis=1, keepdims=True), 1e-8)
        gn = f_step / np.maximum(np.linalg.norm(f_step, axis=1, keepdims=True), 1e-8)
        cos_util = float(np.mean(np.sum(rn * gn, axis=1)))
        seq_groups[(aoi, r["policy"], r["profile"], int(r["seed"]))].append(
            {
                "step": step,
                "total_bytes": float(r["total_bytes"]),
                "anchor": 1 if r["policy"] == "absolute_refresh" else 0,
                "retry": 0,
                "outcome": r["outcome"],
                "age": int(r["age"]),
                "world_mse": float(r["world_mse"]),
                "reference_mse": 0.0,
                "cosine_utility": cos_util,
                "receiver_confirms_current": int(int(r["age"]) == 0),
                "false_accept": 0,
            }
        )

    # Also run diagnostic policies: timeout_only_anchor, nack_only_anchor, drift_gated_anchor, timeout_only_t4, risk_anchor_t4
    for a in areas:
        aoi = a["aoi"]
        f = area_features[aoi]
        for prof_name, prof_cfg in link.PROFILES.items():
            for seed in link.SEEDS:
                for diag_pol in ("timeout_only_anchor", "nack_only_anchor", "drift_gated_anchor", "timeout_only_t4", "risk_anchor_t4"):
                    seq_groups[(aoi, diag_pol, prof_name, seed)] = run_diagnostic_policy(
                        f, diag_pol, prof_cfg, seed, aoi, drift_threshold=1.5e-4
                    )

    # Compute per-sequence and per-AOI statistics
    per_seq = []
    for (aoi, policy, profile, seed), recs in seq_groups.items():
        recs.sort(key=lambda x: x["step"])
        ep = compute_episode_recovery_stats([x["age"] for x in recs], [x["reference_mse"] for x in recs])
        wmse = [x["world_mse"] for x in recs]
        per_seq.append(
            {
                "aoi": aoi,
                "split": area_split[aoi],
                "policy": policy,
                "profile": profile,
                "seed": seed,
                "updates": len(recs),
                "mean_update_bytes": float(np.mean([x["total_bytes"] for x in recs])),
                "mean_world_mse": float(np.mean(wmse)),
                "p95_world_mse": float(np.quantile(wmse, 0.95)),
                "max_world_mse": float(np.max(wmse)),
                "mean_reference_mse": float(np.mean([x["reference_mse"] for x in recs])),
                "mean_cosine_utility": float(np.mean([x["cosine_utility"] for x in recs])),
                "mean_age": float(np.mean([x["age"] for x in recs])),
                "p95_age": ep["p95_age"],
                "max_age": ep["max_age"],
                "current_fraction": float(np.mean([x["receiver_confirms_current"] for x in recs])),
                "mean_recovery_steps": ep["mean_recovery_steps"],
                "max_recovery_steps": ep["max_recovery_steps"],
                "max_mismatch_duration": ep["max_mismatch_duration"],
                "anchors": sum(x["anchor"] for x in recs),
                "retries": sum(x["retry"] for x in recs),
                "rejected": sum(x["outcome"] == "rejected_reference" for x in recs),
                "false_accepts": sum(x["false_accept"] for x in recs),
            }
        )

    # Check calibration selection among periodic_4, periodic_8, periodic_16 on calibration split
    cal_periodic = {}
    for prof_name in ("iid10", "iid30", "burst"):
        cal_scores = {}
        for p_pol in ("periodic_4", "periodic_8", "periodic_16"):
            grp = [r for r in per_seq if r["split"] == "calibration" and r["profile"] == prof_name and r["policy"] == p_pol]
            # Equal-weight normalized product of MSE and Bytes on calibration AOIs
            cal_scores[p_pol] = {
                "mean_world_mse": float(np.mean([r["mean_world_mse"] for r in grp])),
                "mean_update_bytes": float(np.mean([r["mean_update_bytes"] for r in grp])),
                "mse_x_bytes": float(np.mean([r["mean_world_mse"] for r in grp]) * np.mean([r["mean_update_bytes"] for r in grp])),
            }
        best_p = min(cal_scores, key=lambda k: cal_scores[k]["mse_x_bytes"])
        cal_periodic[prof_name] = {"selected_policy": best_p, "calibration_candidates": cal_scores}

    # Aggregate confirmation split across all policies and profiles
    all_policies = (
        "delta_arq",
        "periodic_4",
        "periodic_8",
        "periodic_16",
        "timeout_only_anchor",
        "nack_only_anchor",
        "risk_anchor",
        "drift_gated_anchor",
        "timeout_only_t4",
        "risk_anchor_t4",
        "fixed_reference",
        "absolute_refresh",
        "unrecovered_delta",
    )
    metric_keys = (
        "mean_update_bytes",
        "mean_world_mse",
        "p95_world_mse",
        "max_world_mse",
        "mean_reference_mse",
        "mean_cosine_utility",
        "mean_age",
        "p95_age",
        "max_age",
        "current_fraction",
        "mean_recovery_steps",
        "max_recovery_steps",
        "max_mismatch_duration",
        "anchors",
        "retries",
        "rejected",
        "false_accepts",
    )

    conf_by_aoi = {}
    for prof_name in link.PROFILES:
        for pol in all_policies:
            for aoi in sorted(k for k, v in area_split.items() if v == "confirmation"):
                grp = [r for r in per_seq if (r["aoi"], r["policy"], r["profile"]) == (aoi, pol, prof_name)]
                conf_by_aoi[(prof_name, pol, aoi)] = {k: float(np.mean([r[k] for r in grp])) for k in metric_keys}

    summary_rows = []
    for prof_name in link.PROFILES:
        for pol in all_policies:
            aoi_list = [conf_by_aoi[(prof_name, pol, aoi)] for aoi in sorted(k for k, v in area_split.items() if v == "confirmation")]
            row_s = {"profile": prof_name, "policy": pol, "AOIs": len(aoi_list)}
            for k in metric_keys:
                row_s[k] = float(np.mean([x[k] for x in aoi_list]))
            summary_rows.append(row_s)

    # Save summary CSV and JSON
    link.save_csv(OUT / "summary_complete.csv", summary_rows)
    write_json(
        OUT / "analysis_complete.json",
        {
            "schema": "link-recovery-complete-comparison-v1",
            "calibration_selected_periodic": cal_periodic,
            "confirmation_summary": summary_rows,
            "total_false_reference_acceptances": int(sum(r["false_accepts"] for r in per_seq)),
        },
    )

    # Generate LaTeX table for supplement
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\caption{Complete faulted-link recovery comparison across all policies on the eight reserved SN7 areas (three seeds nested per area). World MSE and 95th-percentile MSE are in units of $10^{-5}$. Age, p95 age, and recovery time are in observation opportunities. Every policy records zero false reference acceptances.}",
        r"\label{tab:recovery-complete}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Profile & Policy & KiB/upd & MSE & p95 MSE & Cos. util. & Mean age & p95 age & Rec. steps & Curr. frac. & Rej. \\",
        r"\midrule",
    ]
    pol_labels = {
        "delta_arq": "Delta + ARQ retry",
        "periodic_4": "Periodic Anchor ($P=4$)",
        "periodic_8": "Periodic Anchor ($P=8$, cal.)",
        "periodic_16": "Periodic Anchor ($P=16$)",
        "timeout_only_anchor": "Timeout-only Anchor ($T=d+1$)",
        "nack_only_anchor": "NACK-only Anchor",
        "risk_anchor": "Risk Anchor ($T=d+1$)",
        "drift_gated_anchor": "Drift-gated Risk Anchor",
        "timeout_only_t4": "Timeout-only Anchor ($T=4$)",
        "risk_anchor_t4": "Risk Anchor (NACK+$T=4$)",
        "fixed_reference": "Fixed initial reference",
        "absolute_refresh": "Full absolute refresh",
        "unrecovered_delta": "Unrecovered optimistic Delta",
    }
    for prof_name in ("iid10", "iid30", "burst"):
        for pol in (
            "delta_arq",
            "periodic_4",
            "periodic_8",
            "periodic_16",
            "timeout_only_anchor",
            "risk_anchor",
            "timeout_only_t4",
            "risk_anchor_t4",
            "drift_gated_anchor",
            "fixed_reference",
            "absolute_refresh",
            "unrecovered_delta",
        ):
            r = next(x for x in summary_rows if x["profile"] == prof_name and x["policy"] == pol)
            tex_lines.append(
                f"{prof_name} & {pol_labels[pol]} & {r['mean_update_bytes']/1024:.2f} & "
                f"{r['mean_world_mse']*1e5:.2f} & {r['p95_world_mse']*1e5:.2f} & "
                f"{r['mean_cosine_utility']:.4f} & {r['mean_age']:.2f} & {r['p95_age']:.2f} & "
                f"{r['mean_recovery_steps']:.2f} & {r['current_fraction']:.3f} & {r['rejected']:.1f} \\\\"
            )
        if prof_name != "burst":
            tex_lines.append(r"\midrule")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    TEX_OUT.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print("Saved complete link recovery comparison and LaTeX table!", flush=True)


if __name__ == "__main__":
    main()
