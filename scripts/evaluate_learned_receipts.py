"""Independent ground-model process for fresh packet-decoded LC09 PNG receipts.

Never imports the learned codec or CompressAI: their numerical flags must not
contaminate the fixed ground evaluator or its existing probability cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_layered_receipts as common
from safesync.pixel_wire import ACK, UPDATED, PixelState, checksum, decode_update
from safesync.pretrained_ground import PretrainedGroundEvaluator
import numpy as np
from PIL import Image

RUN = ROOT / "results/learned_codec/confirmation_q1_8_v1"
OUT = ROOT / "results/learned_tasks/confirmation_q1_8_v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
MAIN = ROOT / "results/layered_tasks/confirmation_v1"
FULL = ROOT / "results/full_cache_tasks/gate30"
CAPS = [32768, 65536, 131072]
METHOD = "learned_hyperprior_q1_8"
# Metadata parsing only, copied from the explicitly fixed LC09 wire schema.
# No codec implementation or network operators are imported into this process.
LC09_HEAD = struct.Struct("<4sBHHHHII16s16sII")


def field(value, *keys):
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    raise ValueError("fresh replay receipt lacks required field: " + "/".join(keys))


def path_from(value, *, within=None):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or (within is not None and not path.is_relative_to(within.resolve())):
        raise ValueError("receipt artifact path leaves its declared decoded-output directory")
    return path


def require_no_codec_imports():
    forbidden = [name for name in sys.modules if name == "safesync.learned_codec" or name.startswith("compressai")
                 or name.startswith("safesync.entropy_bridge")]
    if forbidden:
        raise ValueError("codec imported into the ground numerical process: " + str(forbidden))


def numerical_flags(torch):
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision()}


def assert_ground_flags(flags):
    expected = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
        "cudnn_benchmark": False, "cudnn_deterministic": False,
        "deterministic_algorithms": False, "float32_matmul_precision": "highest"}
    if flags != expected:
        raise ValueError("ground runtime differs from the original fresh-process torch2.8 defaults; do not reuse caches")


def checked_png(path, expected_file_sha, expected_pixel_sha, shape):
    if common.sha(path) != expected_file_sha:
        raise ValueError("fresh receiver PNG file SHA differs")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("fresh receiver image must be lossless three-channel RGB PNG")
        rgb = np.array(image)
    if rgb.dtype != np.uint8 or rgb.shape != tuple(shape) or common.sha_bytes(rgb.tobytes()) != expected_pixel_sha:
        raise ValueError("fresh receiver PNG pixel SHA/native geometry differs")
    return rgb


def load_source(run, aoi, index):
    spec_path = run / "run_spec.json"
    spec = common.read(spec_path)
    if spec["wire_schema"] != "LC09" or spec["qualities"] != list(range(1, 9)) or spec["caps"] != CAPS:
        raise ValueError("the fixed all-q1..8 LC09 candidate/cap protocol changed")
    if spec["scales"] != [1.0, .75, .5, .375, .25] or spec["limit_months"] != 0 or not spec["confirmation"]:
        raise ValueError("LIC comparison is truncated or uses another fixed scale grid")
    if spec["expected_outputs"] != 534 or spec["manifest_sha256"] != common.sha(MANIFEST):
        raise ValueError("LIC uses another geographic/month cohort")
    frozen = common.read(run / "frozen_before_pixels.json")
    if frozen["spec_sha256"] != common.sha(spec_path) or frozen["status"] != "frozen":
        raise ValueError("LIC input specification was not frozen as recorded")
    source_hashes = {common.relative(spec_path): common.sha(spec_path),
        common.relative(run / "frozen_before_pixels.json"): common.sha(run / "frozen_before_pixels.json")}
    for name, expected in spec["source_sha256"].items():
        if common.sha(ROOT / name) != expected or common.sha(run / "source_snapshot" / name.replace("/", "__")) != expected:
            raise ValueError("LIC frozen source differs: " + name)
        source_hashes[name] = expected
    assets = run / "asset_manifest.snapshot.json"
    if common.sha(assets) != spec["asset_manifest_sha256"]:
        raise ValueError("LIC common pretrained model registry differs")
    source_hashes[common.relative(assets)] = common.sha(assets)
    aid, fresh = aoi["aoi_id"], run / "fresh_replay"
    marker_path = fresh / "complete_aois" / (aid + ".json")
    marker = common.read(marker_path)
    expected = 3 * (len(aoi["months"]) - 1)
    if marker["status"] != "complete" or marker["aoi"] != aid or marker["expected_outputs"] != expected:
        raise ValueError("fresh replay AOI is not completely committed")
    if marker["spec_sha256"] != common.sha(spec_path):
        raise ValueError("fresh replay marker refers to another LIC run")
    receipt_path = path_from(marker["receipt_path"], within=fresh)
    if common.sha(receipt_path) != marker["receipt_sha256"]:
        raise ValueError("fresh replay AOI receipt SHA differs")
    receipt = common.read(receipt_path)
    if receipt["status"] != "complete" or receipt["aoi"] != aid or receipt["spec_sha256"] != common.sha(spec_path):
        raise ValueError("fresh replay receipt identity differs")
    if receipt["source_current_or_candidate_RGB_opened"] is not False or receipt["GT_validity_or_ground_model_opened"] is not False:
        raise ValueError("fresh replay does not meet its packet-only decoder boundary")
    if marker["all_RGB_and_ACK_hashes_match"] is not True or receipt["actual_outputs"] != expected:
        raise ValueError("fresh replay has not verified the complete packet cohort")
    rows = receipt["records"]
    if len(rows) != expected:
        raise ValueError("fresh replay receipt omits a fixed packet")
    for cap in CAPS:
        subset = sorted([r for r in rows if r["cap"] == cap], key=lambda r: r["step"])
        if [(r["month"], r["step"]) for r in subset] != [(m["month"], i) for i, m in enumerate(aoi["months"][1:], 1)]:
            raise ValueError("fresh replay reorders, duplicates or omits real months")
        if any(r["aoi"] != aid or r["wire_schema"] != "LC09" or not r["fresh_matches_recorded_receiver"]
               or not r["fresh_matches_recorded_ack"] for r in subset):
            raise ValueError("a packet lacks independent fresh receiver/ACK verification")
    source_hashes.update({common.relative(marker_path): common.sha(marker_path), common.relative(receipt_path): common.sha(receipt_path)})
    return spec, receipt, rows, source_hashes


def initialization(run, aoi, receipt, row):
    initial = receipt["initial"]
    fresh = run / "fresh_replay"
    packet_path = path_from(field(initial, "bootstrap_path", "initial_packet_path", "packet_path"))
    packet = packet_path.read_bytes()
    if common.sha_bytes(packet) != row["initial_packet_sha256"]:
        raise ValueError("actual bootstrap hash differs from fresh receipt")
    w, h = aoi["months"][0]["dimensions"]
    ph, pw = math.ceil(h / 64) * 64, math.ceil(w / 64) * 64
    region = hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
    state, ack, _ = decode_update(PixelState(np.zeros((ph, pw, 3), np.uint8), region), packet)
    if len(packet) != row["initial_down_bytes"] or len(ack) != row["initial_ack_bytes"]:
        raise ValueError("actual bootstrap/ACK costs differ")
    ack_path = path_from(field(initial, "ack_path"), within=fresh)
    if ack != ack_path.read_bytes() or common.sha_bytes(ack) != row["initial_ack_sha256"] or common.sha_bytes(ack) != initial["ack_sha256"]:
        raise ValueError("independently decoded SP03 initial ACK differs from fresh/recorded ACK")
    if state.digest().hex() != initial["state_digest"] or common.sha_bytes(packet) != initial["bootstrap_sha256"]:
        raise ValueError("SP03 initialization state digest/bootstrap identity differs")
    if common.sha_bytes(state.rgb.tobytes()) != row["initial_rgb_sha256"]:
        raise ValueError("fresh initial padded RGB differs from independent SP03 bootstrap decoding")
    native_path = path_from(field(initial, "fresh_initial_native_png_path", "native_png_path", "fresh_native_png_path"), within=fresh)
    padded_path = path_from(field(initial, "fresh_initial_padded_png_path", "padded_png_path", "fresh_padded_png_path"), within=fresh)
    native = checked_png(native_path, field(initial, "fresh_initial_native_png_sha256", "native_png_sha256", "fresh_native_png_sha256"),
        row["initial_native_rgb_sha256"], (h, w, 3))
    padded = checked_png(padded_path, field(initial, "fresh_initial_padded_png_sha256", "padded_png_sha256", "fresh_padded_png_sha256"),
        row["initial_rgb_sha256"], (ph, pw, 3))
    if not np.array_equal(padded, state.rgb) or not np.array_equal(native, padded[:h, :w]):
        raise ValueError("initial receiver native/padded cropping differs")
    return state, native, {"bootstrap_path": common.relative(packet_path), "bootstrap_packet_sha256": common.sha_bytes(packet),
        "bootstrap_down_bytes": len(packet), "bootstrap_ack_bytes": len(ack), "bootstrap_ack_sha256": common.sha_bytes(ack),
        "fresh_native_png_path": common.relative(native_path), "fresh_padded_png_path": common.relative(padded_path),
        "actual_initial_native_RGB_sha256": common.sha_bytes(native.tobytes())}


def receiver_png_state(run, row, prior, month):
    packet_path, fresh = path_from(row["packet_path"]), run / "fresh_replay"
    wire = packet_path.read_bytes()
    if common.sha_bytes(wire) != row["packet_sha256"] or len(wire) != row["down_bytes"] or len(wire) > row["cap"]:
        raise ValueError("actual LIC packet hash/byte cap differs")
    if len(wire) < LC09_HEAD.size + 16 or checksum(wire[:-16]) != wire[-16:]:
        raise ValueError("LC09 input byte checksum/truncation")
    magic, quality, h, w, ch, cw, base, observation, bd, region, ny, nz = LC09_HEAD.unpack(wire[:LC09_HEAD.size])
    if magic != b"LC09" or quality not in range(1, 9) or len(wire) != LC09_HEAD.size + ny + nz + 16:
        raise ValueError("LC09 actual header or entropy-stream layout differs")
    if not (0 < ch <= h and 0 < cw <= w and h % 64 == w % 64 == 0 and ny > 0 and nz > 0):
        raise ValueError("LC09 coded geometry/entropy streams differ")
    if (quality, ch, cw, ny, nz) != (row["quality"], row["coded_height"], row["coded_width"], row["y_stream_bytes"], row["z_stream_bytes"]):
        raise ValueError("LC09 actual candidate metadata differs from the committed fresh receipt")
    if (h, w) != prior.rgb.shape[:2] or region != prior.region or base != prior.version or bd != prior.digest() or observation != row["step"]:
        raise ValueError("LC09 region/base version/history differs")
    nw, nh = month["dimensions"]
    if (row["source_native_width"], row["source_native_height"]) != (nw, nh):
        raise ValueError("LIC source native dimensions differ from the fixed manifest")
    native_path = path_from(row["fresh_native_png_path"], within=fresh)
    padded_path = path_from(row["fresh_padded_png_path"], within=fresh)
    native = checked_png(native_path, row["fresh_native_png_sha256"], row["fresh_native_rgb_sha256"], (nh, nw, 3))
    padded = checked_png(padded_path, row["fresh_padded_png_sha256"], row["fresh_padded_rgb_sha256"], (h, w, 3))
    if not np.array_equal(native, padded[:nh, :nw]) or row["fresh_native_rgb_sha256"] != row["decoded_native_rgb_sha256"] or row["fresh_padded_rgb_sha256"] != row["receiver_rgb_sha256"]:
        raise ValueError("fresh LIC native crop differs from original actual receiver")
    state = PixelState(padded, prior.region, prior.block, base+1, observation, np.full_like(prior.statuses, UPDATED))
    ack_path = path_from(row["fresh_ack_path"], within=fresh)
    ack = ack_path.read_bytes()
    expected_body = ACK.pack(b"LA09", state.version, observation, checksum(wire), state.digest())
    if ack != expected_body + checksum(expected_body) or common.sha_bytes(ack) != row["fresh_ack_sha256"]:
        raise ValueError("fresh LA09 ACK does not bind packet and decoded padded RGB")
    if common.sha_bytes(ack) != row["ack_sha256"] or state.digest().hex() != row["fresh_state_digest"]:
        raise ValueError("fresh replay state/ACK differs from committed raw receiver")
    if ack != path_from(row["ack_path"]).read_bytes():
        raise ValueError("fresh ACK bytes differ from the actual raw-run ACK file")
    if len(ack) != row["up_bytes"] or len(wire) + len(ack) != row["actual_total_bytes"]:
        raise ValueError("actual learned packet/ACK accounting differs")
    return state, native, {"packet_path": common.relative(packet_path), "packet_sha256": common.sha_bytes(wire),
        "ack_sha256": common.sha_bytes(ack), "fresh_state_digest": state.digest().hex(),
        "fresh_native_png_path": common.relative(native_path), "fresh_native_png_sha256": common.sha(native_path),
        "fresh_padded_png_path": common.relative(padded_path), "fresh_padded_png_sha256": common.sha(padded_path),
        "fresh_padded_RGB_sha256": common.sha_bytes(padded.tobytes()), "quality": quality,
        "coded_height": ch, "coded_width": cw, "y_stream_bytes": ny, "z_stream_bytes": nz}


def evaluate_area(run, out, index):
    require_no_codec_imports()
    manifest = common.read(MANIFEST)
    aoi = manifest["aois"][index]
    source_spec, receipt, source_rows, hashes = load_source(run, aoi, index)
    import torch
    before = numerical_flags(torch)
    assert_ground_flags(before)
    evaluator = PretrainedGroundEvaluator(ROOT)
    flags = numerical_flags(torch)
    assert_ground_flags(flags)
    require_no_codec_imports()
    model_hash = common.canonical_hash(json.loads(evaluator.identity))
    thresholds = {"building": evaluator.threshold, "event": evaluator.change_threshold}
    threshold_hash = common.canonical_hash(thresholds)
    main = common.read(MAIN / "summary.json")
    if model_hash != main["model_hash"] or threshold_hash != main["threshold_hash"]:
        raise ValueError("ground teacher/model thresholds changed")
    if evaluator.metadata["versions"] != common.read(ROOT / "resources/pretrained_ground/metadata.json")["versions"]:
        raise ValueError("ground runtime versions differ from the original evaluator")
    files = [Path(__file__), ROOT / "scripts/evaluate_layered_receipts.py", ROOT / "scripts/evaluate_pretrained_receipts.py",
        ROOT / "safesync/pixel_wire.py", ROOT / "safesync/pretrained_ground.py", MANIFEST,
        ROOT / "resources/pretrained_ground/calibration.json", MAIN / "summary.json"]
    hashes.update({common.relative(p): common.sha(p) for p in files})
    spec = {"schema": "independent-LC09-ground-task-v1", "additional_control": True, "part_of_original_blind_confirmation": False,
        "aoi_ids": [aoi["aoi_id"]], "area_index": index, "method": METHOD, "budgets": CAPS,
        "expected_outputs": 3 * (len(aoi["months"]) - 1), "run": common.relative(run),
        "model_hash": model_hash, "threshold_hash": threshold_hash, "thresholds": thresholds,
        "numerical_isolation": {"codec_imported": False, "fresh_windows_process": True,
            "torch_flags_before_model": before, "torch_flags_after_model": flags, "versions": evaluator.metadata["versions"]},
        "source_hashes": hashes, "source_current_RGB_or_features_opened": False,
        "prediction_input": "native PNG independently fresh decoded from bootstrap+LC09 byte strings+public model registry, validated against padded RGB and LA09 ACK",
        "training_overlap": main["training_overlap"], "independent_geographic_generalization_claim": False}
    if (out / "run_spec.json").exists() and common.read(out / "run_spec.json") != spec:
        raise ValueError("independent learned ground-task input identity changed")
    common.save_json(out / "run_spec.json", spec)
    snapshots = out / "source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for name in hashes:
        path = ROOT / name
        destination = snapshots / name.replace("/", "__")
        if path.is_file() and not destination.exists():
            shutil.copyfile(path, destination)
    common.OLD_CACHES.extend(MAIN / f"area_{i}" / "probabilities" for i in range(8))
    common.OLD_CACHES.extend(FULL / f"area_{i}" / "probabilities" for i in range(8))
    results, reference_maps, initial_receipts = [], {}, {}
    for cap in CAPS:
        sequence = sorted([r for r in source_rows if r["cap"] == cap], key=lambda r: r["step"])
        state, native, initial_receipt = initialization(run, aoi, receipt, sequence[0])
        reference_key = common.canonical_hash({"aoi": aoi["aoi_id"], "initial_native_RGB": initial_receipt["actual_initial_native_RGB_sha256"],
            "model_hash": model_hash, "threshold_hash": threshold_hash})
        if reference_key not in reference_maps:
            reference_maps[reference_key], prediction_receipt = common.probability_from_receiver(evaluator, native, out,
                model_hash, threshold_hash, {"bootstrap": initial_receipt["bootstrap_path"], "fresh_receiver": True})
            initial_receipts[reference_key] = {**initial_receipt, **prediction_receipt}
            common.save_json(out / "initial_receipts.json", initial_receipts)
        reference = reference_maps[reference_key]
        cumulative_down, cumulative_up = initial_receipt["bootstrap_down_bytes"], initial_receipt["bootstrap_ack_bytes"]
        for source, month in zip(sequence, aoi["months"][1:]):
            state, native, wire_receipt = receiver_png_state(run, source, state, month)
            cumulative_down += source["down_bytes"]
            cumulative_up += source["up_bytes"]
            event_path = ROOT / f"results/layered/confirmation_v1/area_{index}/gt" / aoi["aoi_id"] / month["month"] / "new_track.png"
            building, event, valid, gt_hashes = common.offline_masks(month, aoi["months"][0], aoi["aoi_id"], event_path, True)
            key = common.canonical_hash({"row_identity": common.row_identity(aoi["aoi_id"], METHOD, cap, source["step"]),
                "packet": source["packet_sha256"], "RGB": wire_receipt["fresh_native_png_sha256"], "GT": gt_hashes,
                "model": model_hash, "threshold": threshold_hash, "source": common.sha(Path(__file__))})
            record_path = out / "records" / (key + ".json")
            if record_path.exists():
                row = common.read(record_path)
                if common.sha(ROOT / row["probability_path"]) != row["probability_file_sha256"]:
                    raise ValueError("saved ground probability artifact changed")
            else:
                probability, prediction_receipt = common.probability_from_receiver(evaluator, native, out, model_hash, threshold_hash,
                    {**wire_receipt, "input_boundary": "independent fresh LC09 receiver PNG only"})
                event_score = np.maximum(probability-reference, 0)
                row = {"aoi": aoi["aoi_id"], "method": METHOD, "budget": cap, "step": source["step"], "month": month["month"],
                    "additional_control": True, "part_of_original_blind_confirmation": False,
                    "model_hash": model_hash, "threshold_hash": threshold_hash, "gt_sha256": gt_hashes,
                    "down_bytes": source["down_bytes"], "ack_bytes": source["up_bytes"], "reference_up_bytes": 0,
                    "bootstrap_down_bytes": initial_receipt["bootstrap_down_bytes"], "bootstrap_ack_bytes": initial_receipt["bootstrap_ack_bytes"],
                    "cumulative_down_bytes": cumulative_down, "cumulative_up_bytes": cumulative_up,
                    "source_encoding_ms": source["sender_search_ms_shared"],
                    "sender_search_time_shared_across_caps": True, "reference_receipt_key": reference_key,
                    **wire_receipt, **prediction_receipt,
                    **common.flat_counts("building", common.counts(probability >= evaluator.threshold, building, valid)),
                    **common.flat_counts("event", common.counts(event_score >= evaluator.change_threshold, event, valid)),
                    **common.flat_counts("initial_only_building", common.counts(reference >= evaluator.threshold, building, valid)),
                    **common.flat_counts("initial_only_event", common.counts(np.zeros_like(event, bool), event, valid))}
                common.save_json(record_path, row)
            if any(row[k] != value for k, value in common.row_identity(aoi["aoi_id"], METHOD, cap, source["step"]).items()):
                raise ValueError("saved learned task row has another cap/method identity")
            results.append(row)
            common.save_json(out / "progress.json", {"status": "in_progress", "completed": len(results),
                "expected": spec["expected_outputs"], "area_index": index, "cap": cap, "month": month["month"]})
        common.write_csv(out / "records.csv", results)
        common.save_json(out / "summary.partial.json", common.aggregate(results))
        print("LIC ground", index, cap, len(sequence), "native receiver outputs; no codec imported", flush=True)
    require_no_codec_imports()
    assert_ground_flags(numerical_flags(torch))
    if len(results) != spec["expected_outputs"] or len({(r["budget"], r["step"]) for r in results}) != len(results):
        raise ValueError("independent learned task cohort incomplete/duplicated")
    summary = {**common.aggregate(results), "status": "complete", "outputs": len(results),
        "additional_control": True, "part_of_original_blind_confirmation": False,
        "all_fresh_PNG_packet_ACK_state_bindings_verified": True, "codec_imported_in_ground_process": False,
        "numerical_flags_unchanged": True, "model_and_thresholds_unchanged": True}
    common.save_json(out / "summary.json", summary)
    common.save_json(out / "progress.json", {"status": "complete", "completed": len(results), "expected": len(results),
        "records_sha256": common.sha(out / "records.csv"), "summary_sha256": common.sha(out / "summary.json")})


def summarize(out):
    manifest = common.read(MANIFEST)
    aids = [a["aoi_id"] for a in manifest["aois"]]
    main, full = common.read(MAIN / "summary.json"), common.read(FULL / "summary.json")
    combined, hashes, new_outputs = [], {}, 0
    for index, aid in enumerate(aids):
        folder = out / f"area_{index}"
        progress, spec, summary = [common.read(folder / name) for name in ["progress.json", "run_spec.json", "summary.json"]]
        if progress["status"] != "complete" or summary["status"] != "complete" or spec["aoi_ids"] != [aid]:
            raise ValueError("the independent LIC ground-task cohort is not complete")
        if progress["records_sha256"] != common.sha(folder / "records.csv") or progress["summary_sha256"] != common.sha(folder / "summary.json"):
            raise ValueError("committed learned task counts changed")
        if spec["model_hash"] != main["model_hash"] or spec["threshold_hash"] != main["threshold_hash"]:
            raise ValueError("learned control uses a different ground model or threshold")
        if spec["source_hashes"][common.relative(Path(__file__))] != common.sha(Path(__file__)):
            raise ValueError("LIC ground-task entry changed across AOIs")
        new_outputs += summary["outputs"]
        combined.extend({**g, "label": METHOD, "additional_control": True} for g in summary["per_aoi_seed"])
        hashes.update({common.relative(folder / name): common.sha(folder / name) for name in ["progress.json", "run_spec.json", "summary.json", "records.csv"]})
    if new_outputs != 534:
        raise ValueError("expected all534 additional LC09 receiver task outputs")
    for g in main["per_aoi_seed"]:
        if g["method"] in ["gaussian_20261029", "full_webp"]:
            combined.append({**g, "label": g["method"], "additional_control": False})
    combined.extend({**g, "label": "full_cache_gate30", "additional_control": True}
                    for g in full["per_aoi"] if g["label"] == "full_cache_gate30")
    labels = ["gaussian_20261029", "full_webp", "full_cache_gate30", METHOD]
    support = {task: [] for task in ["building", "event"]}
    for task in support:
        for aid in aids:
            members = [g for g in combined if g["aoi"] == aid]
            counts = {(g[task]["valid_pixels"], g[task]["positive_pixels"]) for g in members}
            if len(counts) != 1:
                raise ValueError("shared GT/valid supports differ across controls")
            valid, positive = next(iter(counts))
            if valid > 0 and positive > 0:
                support[task].append(aid)
    macro = []
    for cap in CAPS:
        for label in labels:
            members = [g for g in combined if (g["label"], g["budget"]) == (label, cap)]
            if len(members) != 8:
                raise ValueError("a comparison method omits a geographic AOI")
            row = {"method": label, "cap": cap, "geographic_AOIs": 8,
                "additional_control": label in [METHOD, "full_cache_gate30"],
                "actual_mean_update_bidirectional_bytes": float(np.mean([g["mean_update_bidirectional_bytes"] for g in members])),
                "actual_mean_amortized_bidirectional_bytes_per_update": float(np.mean([g["total_bytes_including_bootstrap"] / g["frames"] for g in members]))}
            for task in support:
                group = [g for g in members if g["aoi"] in support[task]]
                row[task + "_supported_AOIs"] = len(group)
                for metric in common.METRICS:
                    row[task + "_" + metric] = common.mean_defined([g[task][metric] for g in group])
            macro.append(row)
    lookup = {(g["label"], g["budget"], g["aoi"]): g for g in combined}
    contrasts = []
    for comparator in labels[:-1]:
        for cap in CAPS:
            for task, ids in support.items():
                pairs = [{"aoi": aid, "learned_f1": lookup[(METHOD, cap, aid)][task]["f1"],
                    "comparator_f1": lookup[(comparator, cap, aid)][task]["f1"]} for aid in ids]
                values = np.array([p["learned_f1"] - p["comparator_f1"] for p in pairs])
                if len(values):
                    draws = np.random.default_rng(20261029).integers(0, len(values), size=(10000, len(values)))
                    ci = np.quantile(values[draws].mean(axis=1), [.05/6, 1-.05/6]).tolist()
                else:
                    ci = [None, None]
                contrasts.append({"method": METHOD, "comparator": comparator, "cap": cap, "task": task,
                    "difference_orientation": "learned minus comparator", "macro_F1_difference": float(values.mean()) if len(values) else None,
                    "paired_AOIs": len(ids), "paired_AOI_ids": ids, "AOI_pairs": pairs,
                    "CI98_333_lower": ci[0], "CI98_333_upper": ci[1]})
    hashes.update({common.relative(MAIN / "summary.json"): common.sha(MAIN / "summary.json"),
        common.relative(FULL / "summary.json"): common.sha(FULL / "summary.json"),
        common.relative(Path(__file__)): common.sha(Path(__file__))})
    result = {"schema": "additional-independent-LC09-ground-comparison-v1", "status": "complete", "additional_outputs": 534,
        "additional_control": True, "part_of_original_blind_confirmation": False, "source_hashes": hashes,
        "model_hash": main["model_hash"], "threshold_hash": main["threshold_hash"], "model_and_threshold_changes": False,
        "codec_imported_in_ground_process": False, "source_current_RGB_or_features_opened": False,
        "support_AOI_ids": support, "per_aoi": combined, "macro_aoi": macro, "paired_comparisons": contrasts,
        "bootstrap": "10000 paired equal-weight AOI resamples, seed20261029; nominal98.333333% intervals per three-cap family per specified contrast/outcome, not simultaneous over all secondary contrasts",
        "LIC_quality_evaluation": "separate root-owned offline process at results/learned_quality/confirmation_v1; sender MSE is not substituted for task/evidence metrics",
        "training_overlap": main["training_overlap"], "independent_geographic_generalization_claim": False}
    common.save_json(out / "summary.json", result)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    lines = ["# 追加learned codec：独立接收端任务评价", "", "LIC是原始确认之后追加的强对照，不是原始盲确认。全部534个LC09包经独立codec进程重放，再在全新ground进程评价；没有导入CompressAI/learned_codec或复用其TF32/deterministic设置。", "",
        "| cap KiB | 方法 | 实际双向更新KiB | 含bootstrap摊销KiB/更新 | 建筑F1 | 新track事件F1 |",
        "|---:|---|---:|---:|---:|---:|"]
    for r in macro:
        lines.append(f"| {r['cap']//1024} | {r['method']} | {r['actual_mean_update_bidirectional_bytes']/1024:.2f} | "
            f"{r['actual_mean_amortized_bidirectional_bytes_per_update']/1024:.2f} | {r['building_f1']:.4f} | {r['event_f1']:.4f} |")
    lines += ["", "表注：actual downlink含真实LC09头、y/z ANS字节和checksum；另计60字节ACK，初始化每AOI/cap只加一次。ground只获得已核验fresh native PNG，native裁剪与padded RGB、bootstrap、LC09版本链及LA09 ACK状态digest均校验。40个固定候选为官方q1–8与五尺度，按发送端当前RGB MSE选择，未用GT或任务模型选质量。", "",
        "公开exp0200任务模型与0.5/0.961阈值不变；八AOI按作者公开fold重建预计均在其训练fold，原始训练日志不可用，故仅作统一模型的通信失真测量，不证明任务模型独立地理泛化。事件GT为相对首月官方track Id差集，valid=current∩first离线使用；所有方法固定共同GT正例AOI支持。空有效区为null但字节照计，无GT误报原始计数保留，不算100%。", "",
        "## LIC − Gaussian 配对差值", "", "| cap KiB | 任务 | F1绝对差 | 98.333%区间（三cap家族） |", "|---:|---|---:|---|"]
    for r in contrasts:
        if r["comparator"] == "gaussian_20261029":
            lines.append(f"| {r['cap']//1024} | {r['task']} | {r['macro_F1_difference']:.5f} | [{r['CI98_333_lower']:.5f}, {r['CI98_333_upper']:.5f}] |")
    lines += ["", "8地区等权配对，10,000次重采样seed20261029；各指定对照/任务对三个cap作名义98.333333%区间，不声称覆盖所有次级比较。全部预算保留；同cap不等于严格相同实际字节。", "",
        "U25/30/35与PSNR由单独质量进程从相同fresh接收PNG计算，不在ground进程读取source-current，也不把sender的全图MSE当成未知证据覆盖。", ""]
    (out / "LEARNED_TASK_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    common.save_json(out / "progress.json", {"status": "complete", "completed_AOIs": 8, "additional_outputs": 534,
        "summary_sha256": common.sha(out / "summary.json"), "comparison_sha256": common.sha(out / "comparison.csv")})
    print("completed all534 independent fresh-PNG learned task outputs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--area-index", type=int, choices=range(8))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if args.area_index is not None:
        evaluate_area(run, out / f"area_{args.area_index}", args.area_index)
        return
    if args.summarize_only:
        summarize(out)
        return
    manifest = common.read(MANIFEST)
    out.mkdir(parents=True, exist_ok=True)
    for index, aoi in enumerate(manifest["aois"]):
        marker = run / "fresh_replay/complete_aois" / (aoi["aoi_id"] + ".json")
        while not marker.exists():
            common.save_json(out / "progress.json", {"status": "waiting_independent_fresh_replay", "completed_AOIs": index,
                "aoi": aoi["aoi_id"], "expected_outputs": 534, "complete": False})
            time.sleep(15)
        command = [str(ROOT / ".venv/Scripts/python.exe"), str(Path(__file__)), "--run", str(run), "--out", str(out), "--area-index", str(index)]
        with (out / f"area_{index}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"area_{index} {line}", end="", flush=True)
            if process.wait():
                raise RuntimeError("independent learned ground task failed; log preserved")
    summarize(out)


if __name__ == "__main__":
    main()
