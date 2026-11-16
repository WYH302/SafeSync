"""Strong IF06 image-codec controls with fixed downlink/total-cost contracts.

Only the sender fitting function opens current source RGB. Ground prediction
accepts only the PixelState reconstructed from actual initial/IF06 packets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, features, __version__ as pillow_version

from safesync.full_image_codecs import fit, receive
from safesync.pixel_wire import ACK, PixelState, TransactionReceiver, checksum
from safesync.pretrained_ground import PretrainedGroundEvaluator
from scripts.evaluate_pretrained_receipts import (canonical_hash, counts, flat_counts, offline_gt,
                                                 rates, save_json, sha, sha_bytes)

CODECS = ["jpeg420", "jpeg2000", "webp"]
CAPS = [32768, 65536]
CONTRACTS = ["A_downcap", "B_equal_total"]
ACK_BYTES = 60
OLD_RUN = ROOT / "results/pixel_sequences/development_side4"
OLD_TASK = ROOT / "results/pretrained_receipts/development_side4"
DEFAULT_OUT = ROOT / "results/full_codec_controls/development_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def descriptor_from_ack(ack, wire, expected_magic):
    if len(ack) != ACK_BYTES or checksum(ack[:-16]) != ack[-16:]:
        raise ValueError("actual ACK length/checksum differs")
    magic, version, observation, transaction, digest = ACK.unpack(ack[:-16])
    if magic != expected_magic or transaction != checksum(wire):
        raise ValueError("ACK does not acknowledge this actual packet")
    return {"version": version, "digest": digest, "observation": observation}


def cost_budget(contract, cap, gaussian):
    if contract == "A_downcap":
        return cap, cap + ACK_BYTES
    target_total = int(gaussian["down_bytes"]) + int(gaussian["up_bytes"])
    return target_total - ACK_BYTES, target_total


def sender_encode(path, expected_sha, *, codec, budget, region, descriptor, observation):
    """Sender-only source boundary; current pixels do not escape this function."""
    if sha(path) != expected_sha:
        raise ValueError("current sender source PNG checksum differs from frozen manifest")
    with Image.open(path) as image:
        current = np.asarray(image.convert("RGB"))
    wire, meta = fit(current, codec=codec, budget=budget, region=region,
                     base_version=descriptor["version"], base_digest=descriptor["digest"],
                     observation=observation, allow_resize=True)
    return wire, meta


def receiver_probability(evaluator, rgb, out, model_hash, threshold_hash, origin):
    """Only actual decoded pixels accepted; may reuse an identical old receipt."""
    rgb = np.asarray(rgb, np.uint8)
    input_sha = sha_bytes(rgb.tobytes())
    key = canonical_hash({"decoded_native_RGB_sha256": input_sha, "shape": list(rgb.shape),
                          "model_hash": model_hash, "threshold_hash": threshold_hash})
    own = out / "probabilities" / (key + ".npz")
    old = OLD_TASK / "probabilities" / (key + ".npz")
    path = own if own.exists() else old if old.exists() else own
    cache_hit = path.exists()
    if cache_hit:
        with np.load(path, allow_pickle=False) as saved:
            probability = saved["probability"].copy()
            if str(saved["input_rgb_sha256"]) != input_sha or str(saved["model_hash"]) != model_hash or str(saved["threshold_hash"]) != threshold_hash:
                raise ValueError("actual reconstructed-image cache identity differs")
        timing = None
    else:
        probability = evaluator.predict(Image.fromarray(rgb))
        timing = evaluator.last_timing
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, probability=probability, input_rgb_sha256=input_sha,
                            model_hash=model_hash, threshold_hash=threshold_hash,
                            origin=json.dumps(origin, ensure_ascii=False))
        temporary.replace(path)
    if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
        raise ValueError("invalid native receiver probability map")
    return probability, {"prediction_input_RGB_sha256": input_sha,
        "probability_path": path.relative_to(ROOT).as_posix(), "probability_file_sha256": sha(path),
        "probability_array_sha256": sha_bytes(probability.tobytes()), "probability_cache_hit": cache_hit,
        "cache_origin": "previous actual receipt" if path == old else "this actual receipt run",
        "prediction_inference_timing": timing}


def aggregate(rows, complete=False):
    fields = ["tp", "fp", "fn", "tn", "valid_pixels", "positive_pixels", "predicted_positive_pixels"]
    prefixes = ["building", "event", "initial_only_building", "initial_only_event"]
    groups = {}
    for row in rows:
        key = (row["contract"], row["codec"], row["cap"], row["aoi"])
        group = groups.setdefault(key, {"contract": key[0], "codec": key[1], "cap": key[2], "aoi": key[3],
            "frames": 0, "down_bytes": 0, "up_bytes": 0, "target_total_bytes": 0, "packet_budget_bytes": 0,
            "total_slack_bytes": 0, "encoding_trials": 0, "sender_search_ms": 0.0, "receiver_ms": 0.0,
            "initial_down_bytes": row["initial_down_bytes"], "initial_ack_bytes": row["initial_ack_bytes"],
            **{prefix: {field: 0 for field in fields} for prefix in prefixes}})
        group["frames"] += 1
        for field in ["down_bytes", "up_bytes", "target_total_bytes", "packet_budget_bytes", "total_slack_bytes", "encoding_trials", "sender_search_ms", "receiver_ms"]:
            group[field] += row[field]
        for prefix in prefixes:
            for field in fields:
                group[prefix][field] += row[prefix + "_" + field]
    for group in groups.values():
        for prefix in prefixes:
            group[prefix].update(rates(group[prefix]))
        group["actual_total_including_initial"] = group["down_bytes"] + group["up_bytes"] + group["initial_down_bytes"] + group["initial_ack_bytes"]
    macro = []
    for contract in CONTRACTS:
        for codec in CODECS:
            for cap in CAPS:
                members = [g for g in groups.values() if (g["contract"], g["codec"], g["cap"]) == (contract, codec, cap)]
                if not members:
                    continue
                item = {"contract": contract, "codec": codec, "cap": cap, "aois_present": len(members),
                        "expected_aois": 2, "frames": sum(g["frames"] for g in members),
                        "actual_down_bytes": sum(g["down_bytes"] for g in members),
                        "actual_up_bytes": sum(g["up_bytes"] for g in members),
                        "target_total_bytes": sum(g["target_total_bytes"] for g in members),
                        "actual_total_slack_bytes": sum(g["total_slack_bytes"] for g in members),
                        "sender_search_ms": sum(g["sender_search_ms"] for g in members)}
                for prefix in prefixes:
                    item[prefix] = {}
                    for metric in ["f1", "iou", "precision", "recall"]:
                        values = [g[prefix][metric] for g in members if g[prefix][metric] is not None]
                        item[prefix][metric] = float(np.mean(values)) if values else None
                        item[prefix][metric + "_defined_aoi_count"] = len(values)
                macro.append(item)
    return {"status": "complete" if complete else "in_progress", "actual_outputs": len(rows),
            "expected_outputs": 540, "statistical_unit": "AOI, aggregate supports before macro rates",
            "zero_denominator_rule": "null; no-event images never score 100%",
            "per_aoi": list(groups.values()), "macro_aoi": macro}


def write_csv(path, rows):
    if not rows:
        return
    fields = [key for key, value in rows[0].items() if not isinstance(value, dict) and key != "prediction_inference_timing"]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "resources/pretrained_ground/dataset_manifest.json"
    manifest = read(manifest_path)
    aois = manifest["aois"][2:4]
    gaussian = {(r["aoi"], int(r["budget"]), r["month"]): r for r in
                csv.DictReader((OLD_RUN / "records.csv").open(encoding="utf-8", newline=""))
                if r["aoi"] in {a["aoi_id"] for a in aois} and r["method"] == "gaussian_detail" and int(r["budget"]) in CAPS}
    if len(gaussian) != 90 or sum(len(a["months"]) - 1 for a in aois) != 45:
        raise ValueError("the fixed two-AOI / 45-update cohort or Gaussian cost trace differs")
    evaluator = PretrainedGroundEvaluator(ROOT)
    thresholds = {"building": evaluator.threshold, "event": evaluator.change_threshold}
    if thresholds != {"building": 0.5, "event": 0.9610000252723694}:
        raise ValueError("pretrained task thresholds changed")
    model_hash, threshold_hash = canonical_hash(json.loads(evaluator.identity)), canonical_hash(thresholds)
    source_paths = [Path(__file__), ROOT / "safesync/full_image_codecs.py", ROOT / "safesync/pixel_wire.py",
        ROOT / "safesync/pretrained_ground.py", ROOT / "scripts/evaluate_pretrained_receipts.py",
        manifest_path, OLD_RUN / "records.csv", OLD_RUN / "run_spec.json",
        ROOT / "resources/pretrained_ground/calibration.json"]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    snapshots = out / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for path in source_paths:
        shutil.copyfile(path, snapshots / path.relative_to(ROOT).as_posix().replace("/", "__"))
    spec = {"codecs": CODECS, "contracts": CONTRACTS, "caps": CAPS, "aoi_ids": [a["aoi_id"] for a in aois],
            "expected_outputs": 540, "expected_sequences": 24, "months_per_control": 45,
            "A": "packet <= nominal cap; actual 60-byte ACK separately counted",
            "B": "per-frame target=old Gaussian actual down+up; full codec packet budget=target-60-byte ACK",
            "selection": "current sender pixel MSE only; 5 fixed scales; highest fitting quality at each tested scale",
            "scales": [1, .75, .5, .375, .25], "allow_resize": True,
            "sender_history_RGB_bytes": 0, "sender_state": "region, last ACK version and digest only; no old RGB",
            "thresholds": thresholds, "threshold_hash": threshold_hash, "model_hash": model_hash,
            "weight_sha256": evaluator.metadata["weight_sha256"], "source_hashes": hashes,
            "Pillow": pillow_version, "codec_versions": {k: features.version(k) for k in ["jpg", "jpg_2000", "webp"]},
            "model_training_overlap": read(ROOT / "resources/pretrained_ground/training_overlap.json"),
            "scope": "development controls only; old Gaussian router hash absent; no independent geographic generalization",
            "expanded_confirmation_images_or_labels_read": False,
            "source_current_RGB_scope": "sender encoding and its pixel MSE only; never ground prediction"}
    spec_path = out / "run_spec.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("full-codec control specification changed; cannot mix frozen results")
    save_json(spec_path, spec)
    save_json(out / "in_progress.json", {"status": "in_progress", "updated_utc": now(), "actual_outputs": 0, "expected_outputs": 540})
    results, initial_probabilities, initial_receipts = [], {}, {}
    for contract in CONTRACTS:
        for codec in CODECS:
            for cap in CAPS:
                for aoi in aois:
                    aid, first = aoi["aoi_id"], aoi["months"][0]
                    width, height = first["dimensions"]
                    shape = (math.ceil(height/64)*64, math.ceil(width/64)*64, 3)
                    region = hashlib.sha256(aid.encode()).digest()[:16]
                    initial_wire = (OLD_RUN / aid / "initial.bin").read_bytes()
                    bootstrap = TransactionReceiver(PixelState(np.zeros(shape, np.uint8), region))
                    initial_ack, _ = bootstrap.receive(initial_wire)
                    state = bootstrap.state
                    descriptor = descriptor_from_ack(initial_ack, initial_wire, b"SA02")
                    initial_sha = sha_bytes(state.rgb.tobytes())
                    if aid not in initial_probabilities:
                        probability, receipt = receiver_probability(evaluator, state.rgb[:height, :width], out, model_hash,
                            threshold_hash, {"initial_packet_sha256": sha_bytes(initial_wire), "scope": "actual SP03 initialization"})
                        initial_probabilities[aid] = probability
                        initial_receipts[aid] = {**receipt, "initial_packet_sha256": sha_bytes(initial_wire),
                            "initial_receiver_SHA256": initial_sha, "initial_down_bytes": len(initial_wire), "initial_ack_bytes": len(initial_ack)}
                        save_json(out / "initial_receipts.json", initial_receipts)
                        atomic_bytes(out / "initial" / aid / "initial.bin", initial_wire)
                        atomic_bytes(out / "initial" / aid / "initial.ack", initial_ack)
                    elif initial_receipts[aid]["initial_receiver_SHA256"] != initial_sha:
                        raise ValueError("initial state differs among the frozen full-codec controls")
                    reference = initial_probabilities[aid]
                    for step, month in enumerate(aoi["months"][1:], start=1):
                        started = time.perf_counter()
                        old = gaussian[(aid, cap, month["month"])]
                        packet_budget, total_target = cost_budget(contract, cap, old)
                        folder = out / "packets" / contract / codec / str(cap) / aid
                        packet_path = folder / f"{step:03d}.bin"
                        meta_path = folder / f"{step:03d}.meta.json"
                        encoder_identity = {"codec": codec, "contract": contract, "cap": cap,
                            "packet_budget": packet_budget, "source_png_sha256": month["derived_sha256"]["rgb_path"],
                            "region": region.hex(), "base_version": descriptor["version"],
                            "base_digest": descriptor["digest"].hex(), "observation": step,
                            "encoder_source_sha256": hashes["safesync/full_image_codecs.py"]}
                        if packet_path.exists() and meta_path.exists():
                            stored = read(meta_path)
                            if stored["encoder_identity"] != encoder_identity:
                                raise ValueError("cached packet identity differs")
                            packet, meta = packet_path.read_bytes(), stored["meta"]
                            if sha_bytes(packet) != stored["packet_sha256"]:
                                raise ValueError("saved IF06 packet changed")
                        else:
                            packet, meta = sender_encode(ROOT / month["rgb_path"], month["derived_sha256"]["rgb_path"],
                                codec=codec, budget=packet_budget, region=region, descriptor=descriptor, observation=step)
                            atomic_bytes(packet_path, packet)
                            save_json(meta_path, {"encoder_identity": encoder_identity, "packet_sha256": sha_bytes(packet), "meta": meta})
                        received_at = time.perf_counter()
                        state, ack = receive(state, packet)
                        receiver_ms = (time.perf_counter() - received_at) * 1000
                        descriptor = descriptor_from_ack(ack, packet, b"IA06")
                        if descriptor["digest"] != state.digest() or len(ack) != ACK_BYTES:
                            raise ValueError("received state and actual ACK descriptor differ")
                        actual_total = len(packet) + len(ack)
                        if len(packet) > packet_budget or actual_total > total_target:
                            raise ValueError("actual full-codec bytes exceed the fixed cost contract")
                        atomic_bytes(packet_path.with_suffix(".ack"), ack)
                        building, event, valid, gt_hashes = offline_gt(month, first, aid)
                        receiver_sha = sha_bytes(state.rgb.tobytes())
                        row_key = canonical_hash({"packet_sha": sha_bytes(packet), "initial_sha": initial_sha,
                            "model_hash": model_hash, "threshold_hash": threshold_hash,
                            "GT": gt_hashes, "evaluation_source": hashes["scripts/run_full_codec_controls.py"]})
                        row_path = out / "records" / (row_key + ".json")
                        if row_path.exists():
                            row = read(row_path)
                            if row["receiver_sha256"] != receiver_sha or sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
                                raise ValueError("cached actual-receiver task record differs")
                        else:
                            probability, receipt = receiver_probability(evaluator, state.rgb[:height, :width], out,
                                model_hash, threshold_hash, {"packet_path": packet_path.relative_to(ROOT).as_posix(),
                                    "packet_sha256": sha_bytes(packet), "receiver_SHA256": receiver_sha})
                            change = np.maximum(probability - reference, 0)
                            row = {"aoi": aid, "month": month["month"], "step": step, "codec": codec, "contract": contract,
                                "cap": cap, "packet_budget_bytes": packet_budget, "target_total_bytes": total_target,
                                "gaussian_actual_down_bytes": int(old["down_bytes"]), "gaussian_actual_up_bytes": int(old["up_bytes"]),
                                "packet_path": packet_path.relative_to(ROOT).as_posix(), "packet_sha256": sha_bytes(packet),
                                "ack_path": packet_path.with_suffix(".ack").relative_to(ROOT).as_posix(), "ack_sha256": sha_bytes(ack),
                                "down_bytes": len(packet), "up_bytes": len(ack), "actual_total_bytes": actual_total,
                                "packet_slack_bytes": packet_budget-len(packet), "total_slack_bytes": total_target-actual_total,
                                "initial_down_bytes": len(initial_wire), "initial_ack_bytes": len(initial_ack),
                                "initial_packet_sha256": sha_bytes(initial_wire), "receiver_sha256": receiver_sha,
                                "source_png_sha256_sender_only": month["derived_sha256"]["rgb_path"],
                                "sender_history_RGB_bytes": 0, "sender_descriptor_bytes": 36,
                                "quality": meta["quality"], "coded_height": meta["coded_height"], "coded_width": meta["coded_width"],
                                "scale_x": meta["coded_width"]/width, "scale_y": meta["coded_height"]/height,
                                "encoding_trials": meta["codec_trials"], "sender_search_ms": meta["sender_search_ms"],
                                "sender_pixel_mse": meta["sender_pixel_mse"], "receiver_ms": receiver_ms,
                                "frame_wall_ms": (time.perf_counter()-started)*1000,
                                "model_hash": model_hash, "threshold_hash": threshold_hash, "GT_sha256": gt_hashes,
                                **receipt,
                                **flat_counts("building", counts(probability >= evaluator.threshold, building, valid)),
                                **flat_counts("event", counts(change >= evaluator.change_threshold, event, valid)),
                                **flat_counts("initial_only_building", counts(reference >= evaluator.threshold, building, valid)),
                                **flat_counts("initial_only_event", counts(np.zeros_like(event, bool), event, valid))}
                            save_json(row_path, row)
                        results.append(row)
                        write_csv(out / "records.csv", results)
                        save_json(out / "in_progress.json", {"status": "in_progress", "updated_utc": now(),
                            "actual_outputs": len(results), "expected_outputs": 540, "contract": contract,
                            "codec": codec, "cap": cap, "aoi": aid, "month": month["month"]})
                        if len(results) % 5 == 0 or len(results) == 1:
                            print(f"IF06 {len(results)}/540 {contract} {codec} cap={cap} {aid} {month['month']}", flush=True)
                    save_json(out / "summary.partial.json", aggregate(results))
                partial = aggregate(results)
                group_result = next(g for g in partial["macro_aoi"] if (g["contract"], g["codec"], g["cap"]) == (contract, codec, cap))
                save_json(out / "summary.partial.json", partial)
                print("complete fixed 45-frame control", json.dumps(group_result), flush=True)
    if len(results) != 540:
        raise ValueError("incomplete fixed full-codec cohort")
    summary = aggregate(results, complete=True)
    summary.update(created_utc=now(), all_actual_IF06_packets_replayed=True,
                   all_cost_contracts_satisfied=True, fixed_model_and_thresholds_unchanged=True,
                   original_gaussian_router_hash_absent=True, independent_geographic_generalization_claim=False)
    save_json(out / "summary.json", summary)
    save_json(out / "complete.json", {"status": "complete", "updated_utc": now(), "actual_outputs": 540,
        "expected_outputs": 540, "record_csv_sha256": sha(out / "records.csv"), "summary_sha256": sha(out / "summary.json")})
    (out / "in_progress.json").unlink(missing_ok=True)
    print("all 540 fixed IF06 codec controls complete", flush=True)


if __name__ == "__main__":
    main()
