"""Exploratory zero-tolerance feature constraint on already verified image packets."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync.features import FrozenEncoder
from scripts.run_task_evidence_selector import quality
from scripts.run_residual_evidence_development import pool
from scripts.run_locked_second_test import sha,write_json

SOURCE = ROOT/"results/task_evidence/residual_confirmation_v1"
GROUND = ROOT/"results/task_evidence/residual_confirmation_ground_v1"
DATA = ROOT/"data/second_residual_test_v1"
OUT = ROOT/"results/task_evidence/feature_gate_development_v1"
CANDIDATES = ("full_webp","hyperprior","residual_change")


def main():
    lock = json.loads((SOURCE/"method_lock.json").read_text("utf-8"))
    ids = sorted(lock["pair_ids"],key=lambda p:hashlib.sha256(("feature-gate-development:"+p).encode()).hexdigest())[:64]
    spec = dict(pair_ids=ids,source_sha256=sha(Path(__file__)),source_method_lock_sha256=sha(SOURCE/"method_lock.json"),
        encoder_config=lock["backbone"],model_weight_sha256=lock["model_weight_sha256"],
        candidates=CANDIDATES,caps=lock["caps"],heldouts=lock["heldouts"],feature_tolerance=0.,
        feature_error="Mean squared L2 difference of the256 normalized384-D source/candidate tokens, with the same FP16 round-trip",
        quality="Change-rank-weighted max(1-native64-pixel cell RGB MSE/0.001,0); WebP wins ties",
        policies=["quality_only","feature_gate"],
        scope="64 already inspected third-cohort pairs, chosen by metadata hash. Exploratory repair after the failed seven-condition test. Existing actual packets are selected; new neural compression and independent samples are not claimed.",
        diagnosis="Residual evidence improves visual coverage, while the independent confirmation leaves the WebP later-use constraint unsatisfied. Preserve the backbone representation relative to the same-budget reference.",
        unknown_module="Omitted after its matched residual/change contrast failed to establish an increment")
    spec = json.loads(json.dumps(spec))
    OUT.mkdir(parents=True,exist_ok=True)
    spec_path = OUT/"run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed zero-tolerance feature-gate development")
    write_json(spec_path,spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT,lock["backbone"])
    manifest = json.loads((DATA/"manifest.json").read_text("utf-8"))
    pairs = {p["id"]:p for p in manifest["pairs"]}
    decisions = []
    for pair_id in ids:
        folder = SOURCE/"pairs"/pair_id
        receipt = json.loads((folder/"complete.json").read_text("utf-8"))
        if receipt["records_sha256"] != sha(folder/"records.json"):
            raise ValueError("Changed source sender decisions")
        sender_rows = json.loads((folder/"records.json").read_text("utf-8"))
        lookup = {(r["heldout"],r["budget"],r["method"]):r for r in sender_rows if r["seed"] == 0}
        ready = json.loads((folder/"sender_inputs_ready.json").read_text("utf-8"))
        if ready["features_sha256"] != sha(folder/"sender_features.npz"):
            raise ValueError("Changed committed source features")
        with np.load(folder/"sender_features.npz",allow_pickle=False) as stored:
            f0,f1 = stored["before"].astype(np.float32),stored["current"].astype(np.float32)
        item = pairs[pair_id]["files"]["after"]
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed sender current image")
        current = np.asarray(Image.open(ROOT/item["path"]).convert("RGB"))
        ranking = rankdata(pool(1-np.sum(f0*f1,axis=1)))/64
        cached = {}
        for h in lock["heldouts"]:
            for cap in lock["caps"]:
                options = []
                for method in CANDIDATES:
                    row = lookup[h,cap,method]
                    path,png = ROOT/row["packet_path"],ROOT/row["received_png"]
                    if sha(path) != row["packet_sha256"] or sha(png) != row["received_png_sha256"]:
                        raise ValueError("Changed real candidate packet or decoder PNG")
                    identity = row["receiver_rgb_sha256"]
                    if identity not in cached:
                        rgb = np.asarray(Image.open(png).convert("RGB"))
                        if hashlib.sha256(rgb.tobytes()).hexdigest() != identity:
                            raise ValueError("Candidate registered-decoder RGB changed")
                        feature,timing = encoder.extract(Image.fromarray(rgb))
                        feature = feature.astype(np.float16).astype(np.float32)
                        cached[identity] = dict(feature_error=float(np.mean(np.sum((f1-feature)**2,axis=1))),
                            rgb_quality=quality(current,rgb,ranking),uncached_feature_ms=timing["total_ms"])
                    options.append(dict(source=row,**cached[identity]))
                reference = next(r for r in options if r["source"]["method"] == "full_webp")
                for policy in ("quality_only","feature_gate"):
                    eligible = options if policy == "quality_only" else [r for r in options if r["feature_error"] <= reference["feature_error"]]
                    selected = max(eligible,key=lambda r:(r["rgb_quality"],r["source"]["method"] == "full_webp"))
                    source = selected["source"]
                    decisions.append(dict(pair_id=pair_id,heldout=h,budget=cap,method=policy,selected_method=source["method"],
                        packet_path=source["packet_path"],packet_sha256=source["packet_sha256"],receiver_rgb_sha256=source["receiver_rgb_sha256"],
                        receiver_task_sha256=source["receiver_task_sha256"],feature_error=selected["feature_error"],reference_feature_error=reference["feature_error"],
                        rgb_quality=selected["rgb_quality"],selected_feature_ms=selected["uncached_feature_ms"]))
        print(f"fixed feature-gate decisions {ids.index(pair_id)+1}/64",flush=True)
    # No semantic GT or receiver metric table is opened before this commit.
    write_json(OUT/"sender_commit.json",decisions)
    source_ground_receipt = json.loads((GROUND/"complete.json").read_text("utf-8"))
    if source_ground_receipt["records_sha256"] != sha(GROUND/"records.csv"):
        raise ValueError("Changed source ground observations")
    measured = list(csv.DictReader((GROUND/"records.csv").open(encoding="utf-8")))
    measured = {(r["pair_id"],int(r["heldout"]),int(r["budget"]),r["method"]):r for r in measured if r["seed"] == "0"}
    result = []
    for choice in decisions:
        row = measured[choice["pair_id"],choice["heldout"],choice["budget"],choice["selected_method"]]
        if any(row[k] != choice[k] for k in ("packet_sha256","receiver_rgb_sha256","receiver_task_sha256")):
            raise ValueError("Selected packet differs from its measured receiver input")
        result.append(dict(row,method=choice["method"],selected_method=choice["selected_method"],feature_error=choice["feature_error"],
            reference_feature_error=choice["reference_feature_error"],rgb_quality=choice["rgb_quality"]))
    with (OUT/"records.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    write_json(OUT/"complete.json",dict(status="complete",pairs=64,records=len(result),records_sha256=sha(OUT/"records.csv"),
        sender_commit_sha256=sha(OUT/"sender_commit.json"),source_ground_sha256=sha(GROUND/"records.csv"),spec_sha256=sha(spec_path),
        feature_condition_exact=all(r["method"] != "feature_gate" or r["feature_error"] <= r["reference_feature_error"] for r in decisions),
        scope="Reuses hash-identical actual receiver measurements only after sender choices commit; exploratory64-pair cohort"))


if __name__ == "__main__":
    main()
