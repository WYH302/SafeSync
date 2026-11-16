"""Later ground semantic use of actual RGB and a feature-only diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
import torch
from safesync.data import read_pair
from safesync.features import FrozenEncoder
from safesync.full_image_codecs import receive as full_receive
from safesync.product_quantizer import ProductQuantizer
from safesync.hybrid_wire import receive as hybrid_receive
from scripts.run_second_feature_mix import load_frozen
from scripts.run_locked_second_test import CONFIG, LOCK, write_json, sha

OUT = ROOT / "results/future_semantic_probe/v1/confirmation"
FIT = ROOT / "results/future_semantic_probe/v1"
SOURCES = {"image": ROOT / "results/unknown_recovery/locked_test_v1",
           "hybrid": ROOT / "results/hybrid_pq/confirmation_v1",
           "learned": ROOT / "results/second_learned/tasks_v1"}
IMAGE_METHODS = {"feature_change", "emergent_mix", "feature_random", "full_jpeg420", "full_jpeg2000", "full_webp", "full_current", "old_only"}
HYBRID_METHODS = {"change", "emergent_mix", "random", "PQ_only"}


def source_rows(cohort, pair_id):
    folder = SOURCES[cohort] / "pairs" / pair_id
    receipt_path = folder / "complete.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if receipt["status"] != "complete" or sha(folder / "records.json") != receipt["records_sha256"]:
        raise ValueError("Changed source pair for the future ground assay")
    rows = json.loads((folder / "records.json").read_text("utf-8"))
    if cohort == "image":
        rows = [r for r in rows if r["method"] in IMAGE_METHODS]
    elif cohort == "hybrid":
        rows = [r for r in rows if r["method"] in HYBRID_METHODS]
    return rows, sha(receipt_path)


def prior_state(cohort, pair_id, wire):
    folder = SOURCES[cohort] / "pairs" / pair_id
    if cohort == "learned":
        folder = ROOT / "results/second_learned/confirmation_v1/pairs" / pair_id
    packet = (folder / "initial.bin").read_bytes()
    region = hashlib.sha256(("SafeSync-SECOND-pixel:" + pair_id).encode()).digest()[:16]
    receiver = wire.TransactionReceiver(wire.PixelState(np.zeros((512, 512, 3), np.uint8), region))
    ack, _ = receiver.receive(packet)
    if ack != (folder / "initial.ack").read_bytes():
        raise AssertionError("Future assay bootstrap replay differs")
    return receiver.state


def evaluate(cohort, pair, rows, encoder, wire, helper, books, means, classes):
    _, _, old, label, changed = read_pair(ROOT, pair)
    mask = (old >= 1) & (old <= 6) & (label >= 1) & (label <= 6) & changed
    prior = prior_state(cohort, pair["id"], wire)
    predictions = {}
    result = []

    def confusion(features):
        prediction = classes[np.argmax(features @ means.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
        return np.bincount((label[mask].astype(int) - 1) * 6 + prediction[mask].astype(int) - 1, minlength=36).reshape(6, 6)

    for row in rows:
        packet_path = ROOT / row["packet_path"]
        if sha(packet_path) != row["packet_sha256"]:
            raise ValueError("Future assay actual input packet changed")
        packet = packet_path.read_bytes()
        features = None
        if cohort == "learned":
            original_folder = ROOT / "results/second_learned/confirmation_v1/pairs" / pair["id"]
            image_path = original_folder / f"b{row['budget']}.received.png"
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
        elif cohort == "hybrid":
            features, rgb, _ = hybrid_receive(prior, packet, books[int(row["heldout"])], helper)
        elif row["method"] == "old_only":
            rgb = prior.rgb
        elif row["method"] in {"full_jpeg420", "full_jpeg2000", "full_webp"}:
            state, _ = full_receive(prior, packet)
            rgb = state.rgb
        else:
            receiver = helper["fresh_receiver"](prior)
            receiver.receive(packet)
            rgb = receiver.state.rgb
        identity = hashlib.sha256(rgb.tobytes()).hexdigest()
        if identity != row["receiver_rgb_sha256"]:
            raise AssertionError("Future assay fresh RGB differs from source receipt")
        if identity not in predictions:
            image_features, _ = encoder.cached(Image.fromarray(rgb))
            predictions[identity] = confusion(image_features)
        common = dict(pair_id=pair["id"], cohort=cohort, heldout=int(row["heldout"]), seed=int(row["seed"]),
            budget=int(row["budget"]), total_bytes=int(row["total_bytes"]), packet_sha256=row["packet_sha256"],
            receiver_rgb_sha256=identity, evaluated_pixels=int(mask.sum()))
        result.append(dict(**common, method=row["method"], representation="received_RGB",
            confusion=json.dumps(predictions[identity].tolist())))
        if cohort == "hybrid" and row["method"] == "PQ_only":
            if hashlib.sha256(features.tobytes()).hexdigest() != row["receiver_features_sha256"]:
                raise AssertionError("Future task feature-only diagnostic differs from received codes")
            result.append(dict(**common, method="PQ_feature_only", representation="received_PQ_features",
                confusion=json.dumps(confusion(features).tolist())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text("utf-8"))
    fit = json.loads((FIT / "fit_receipt.json").read_text("utf-8"))
    if sha(FIT / "prototypes.npz") != fit["prototypes_sha256"]:
        raise ValueError("Fixed later ground head changed")
    spec = dict(source_sha256=sha(Path(__file__)), fit_receipt_sha256=sha(FIT / "fit_receipt.json"),
        earlier_lock_sha256=sha(LOCK), hybrid_lock_sha256=sha(SOURCES["hybrid"] / "method_lock.json"),
        pair_ids=lock["pair_ids"], image_methods=sorted(IMAGE_METHODS), hybrid_methods=sorted(HYBRID_METHODS),
        ground_fit="All-class calibration labels only, including classes held out of the online sender fitting",
        support="Both temporal labels explicit 1..6 and binary change true; all six semantic classes retained",
        endpoint="Held-out-class F1 from six-class counts on actual received RGB; shared all-class prototype head",
        feature_diagnostic="The same later head on received PQ-only features, to test whether visual evidence adds value beyond the latent",
        interpretation="Additional exploratory use assay. It does not change any routing, codebook, or primary acceptance test.")
    spec = json.loads(json.dumps(spec))
    spec_path = OUT / "method_lock.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != spec:
            raise ValueError("Changed later-use assay definition")
    else:
        spec["locked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(spec_path, spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(FIT / "prototypes.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    books = {}
    for h in (5, 3, 4):
        with np.load(ROOT / f"results/product_quantizer/known_only_v1/h{h}_m8.npz", allow_pickle=False) as stored:
            books[h] = ProductQuantizer(stored["centroids"])
    wire, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    manifest = json.loads((ROOT / "data/second_locked_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    complete, all_records = set(), []
    expected_per_pair = {"image": 102, "hybrid": 66, "learned": 12}
    while len(complete) < 768:
        progressed = False
        for pair_id in lock["pair_ids"]:
            for cohort in SOURCES:
                key = pair_id, cohort
                if key in complete:
                    continue
                source = source_rows(cohort, pair_id)
                if source is None:
                    continue
                source_values, receipt_hash = source
                folder = OUT / "pairs" / pair_id / cohort
                receipt_path = folder / "complete.json"
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text("utf-8"))
                    if receipt["source_receipt_sha256"] != receipt_hash or sha(folder / "records.json") != receipt["records_sha256"]:
                        raise ValueError("Changed later-use task receipt")
                    records = json.loads((folder / "records.json").read_text("utf-8"))
                else:
                    records = evaluate(cohort, pairs[pair_id], source_values, encoder, wire, helper, books, means, classes)
                    if len(records) != expected_per_pair[cohort]:
                        raise AssertionError("Incomplete later-use cohort for one source pair")
                    write_json(folder / "records.json", records)
                    write_json(receipt_path, dict(status="complete", records=len(records), records_sha256=sha(folder / "records.json"),
                        source_receipt_sha256=receipt_hash, assay_lock_sha256=sha(spec_path)))
                all_records.extend(records)
                complete.add(key)
                progressed = True
                write_json(OUT / "progress.json", dict(status="in_progress", completed_pair_cohorts=len(complete), expected_pair_cohorts=768,
                    records=len(all_records), expected_records=46080))
                print(f"later ground task {len(complete)}/768 pair-cohorts, {len(all_records)} observations", flush=True)
        if len(complete) == 768:
            break
        if not progressed:
            if not args.follow:
                raise ValueError("Later-use source cohorts remain incomplete")
            time.sleep(15)
    helper["save_csv"](OUT / "records.csv", all_records)
    write_json(OUT / "complete.json", dict(status="complete", records=len(all_records), expected=46080,
        records_sha256=sha(OUT / "records.csv"), method_lock_sha256=sha(spec_path),
        source_sha256=sha(Path(__file__)), completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
