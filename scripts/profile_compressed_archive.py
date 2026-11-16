"""Measure one fixed lossless PNG representation of frozen gate30 archives.

CPU-only receiver replay. This measures historical archive pixels, never render()
output, and does not change routing, packets, models, or the frozen experiments.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, __version__ as pillow_version, features
from safesync.layered_wire import LayeredReceiver

BASE = ROOT / "results/full_cache_controls/gate30"
TASK = ROOT / "results/full_cache_tasks/gate30"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
CAPS = (32768, 65536, 131072)
SOURCE_NAMES = ["scripts/profile_compressed_archive.py", "safesync/layered_wire.py",
                "safesync/layered_cache.py", "safesync/pixel_wire.py", "safesync/certificates.py", "safesync/reference.py"]


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_save(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def snapshot(state, identity):
    archive = state.archive
    if archive.dtype != np.uint8 or archive.shape != (1024, 1024, 3):
        raise ValueError("unexpected frozen archive shape or dtype")
    digest = sha_bytes(archive.tobytes())
    stream = io.BytesIO()
    wall, cpu = time.perf_counter(), time.process_time()
    Image.fromarray(archive).save(stream, format="PNG", optimize=False, compress_level=6)
    encode_cpu_ms = (time.process_time() - cpu) * 1000
    encode_wall_ms = (time.perf_counter() - wall) * 1000
    png = stream.getvalue()
    wall, cpu = time.perf_counter(), time.process_time()
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("lossless archive changed format/mode")
        restored = np.array(image)
    decode_cpu_ms = (time.process_time() - cpu) * 1000
    decode_wall_ms = (time.perf_counter() - wall) * 1000
    exact = np.array_equal(archive, restored)
    if not exact or sha_bytes(restored.tobytes()) != digest:
        raise ValueError("stored archive does not exactly recover historical RGB")
    metadata = {"archive_format": "RGB PNG", "region_hex": state.region.hex(),
                "history_hex": state.history.hex(), "version": state.version,
                "observation": state.observation, "config": state.cache.config}
    metadata_bytes = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    result = {**identity, "archive_RGB_sha256": digest, "raw_archive_bytes": int(archive.nbytes),
              "PNG_bytes": len(png), "PNG_sha256": sha_bytes(png), "decoded_exact": exact,
              "PNG_state_metadata_json_bytes": len(metadata_bytes), "existing_fullcache_summary_bytes": len(state.cache.wire),
              "current_coarse_array_bytes_excluded": 0 if state.coarse is None else int(state.coarse.nbytes),
              "encode_CPU_ms": encode_cpu_ms, "decode_CPU_ms": decode_cpu_ms,
              "encode_wall_ms": encode_wall_ms, "decode_wall_ms": decode_wall_ms}
    return result, png, metadata_bytes


def prepare(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new output directory; existing profile is never overwritten")
    manifest = read(MANIFEST)
    if len(manifest["aois"]) != 8 or sum(len(a["months"]) - 1 for a in manifest["aois"]) != 178:
        raise ValueError("wrong confirmation cohort")
    inputs = {relative(MANIFEST): sha(MANIFEST)}
    plans = []
    for index, aoi in enumerate(manifest["aois"]):
        area = BASE / f"area_{index}"
        progress, spec = read(area / "progress.json"), read(area / "run_spec.json")
        if progress["status"] != "complete" or spec["control_gate_psnr"] != 30 or spec["methods"] != ["full_cache_mse"]:
            raise ValueError("requires the complete own-bootstrap gate30 stream")
        if progress["records_sha256"] != sha(area / "records.csv") or progress["spec_sha256"] != sha(area / "run_spec.json"):
            raise ValueError("frozen transport binding differs")
        for name, digest in spec["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("frozen transport source changed: " + name)
        task_progress = read(TASK / f"area_{index}/progress.json")
        task_path = TASK / f"area_{index}/records.csv"
        if task_progress["status"] != "complete" or task_progress["records_sha256"] != sha(task_path):
            raise ValueError("original receiver archive-hash evidence incomplete")
        task_lookup = {(int(r["budget"]), int(r["step"])): r for r in rows(task_path)}
        for path in (area / "progress.json", area / "run_spec.json", area / "records.csv", task_path, TASK / f"area_{index}/progress.json"):
            inputs[relative(path)] = sha(path)
        source_rows = rows(area / "records.csv")
        for cap in CAPS:
            sequence = sorted([r for r in source_rows if int(r["budget"]) == cap], key=lambda r: int(r["step"]))
            if [r["month"] for r in sequence] != [m["month"] for m in aoi["months"][1:]]:
                raise ValueError("gate30 month sequence differs")
            plans.append((index, aoi, cap, sequence, task_lookup))
    spec = {"schema": "fixed-lossless-history-PNG-profile-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "additional representation sensitivity of frozen gate30 receiver.archive; no rerouting",
            "codec": {"format": "PNG", "mode": "RGB uint8", "optimize": False, "compress_level": 6, "codec_search": False},
            "AOIs": 8, "caps": list(CAPS), "updates": 534, "bootstraps": 24, "expected_snapshots": 558,
            "source_sha256": {name: sha(ROOT / name) for name in SOURCE_NAMES}, "input_sha256": inputs,
            "runtime": {"python": sys.version, "numpy": np.__version__, "Pillow": pillow_version,
                        "zlib": features.version("zlib"), "machine": platform.machine(), "platform": platform.platform(),
                        "thread_env": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
            "timing": "one encode/decode per snapshot; CPU process_time and wall perf_counter; no warmup or encoder search",
            "timing_exclusions": "hash/equality checks and filesystem persistence excluded from encode/decode; packet read/replay separate",
            "state_scope": "historical PNG only; current/coarse, summaries, metadata, provisional state and runtime allocations separate",
            "source_pixels_or_models_opened": False, "render_called": False, "GPU_used": False}
    out.mkdir(parents=True, exist_ok=True)
    save(out / "run_spec.json", spec)
    (out / "source_snapshot").mkdir()
    for name in SOURCE_NAMES:
        shutil.copyfile(ROOT / name, out / "source_snapshot" / name.replace("/", "__"))
    return spec, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/sender_profile/compressed_archive_v1")
    args = parser.parse_args()
    out = args.out.resolve()
    spec, plans = prepare(out)
    start = time.perf_counter()
    records, area_summaries = [], []
    for index, aoi, cap, sequence, task_lookup in plans:
        decoder = LayeredReceiver()
        bootstrap_path = ROOT / sequence[0]["bootstrap_path"]
        packets = [(0, aoi["months"][0]["month"], bootstrap_path, None)]
        packets += [(int(r["step"]), r["month"], ROOT / r["packet_path"], r) for r in sequence]
        sequence_records = []
        for step, month, packet_path, source in packets:
            tick = time.perf_counter()
            packet = packet_path.read_bytes()
            packet_read_ms = (time.perf_counter() - tick) * 1000
            if packet[:4] != (b"LB01" if step == 0 else b"LA01"):
                raise ValueError("wrong own-bootstrap/update schema")
            if source and sha_bytes(packet) != source["packet_sha256"]:
                raise ValueError("gate30 packet changed")
            decoder.expect(step)
            wall, cpu = time.perf_counter(), time.process_time()
            ack = decoder.receive(packet)
            replay_cpu_ms = (time.process_time() - cpu) * 1000
            replay_wall_ms = (time.perf_counter() - wall) * 1000
            expected_ack = source["ack_sha256"] if source else sha(packet_path.with_suffix(".ack"))
            if sha_bytes(ack) != expected_ack:
                raise ValueError("replayed frozen ACK differs")
            identity = {"area_index": index, "aoi": aoi["aoi_id"], "budget": cap, "step": step, "month": month,
                        "bootstrap": step == 0, "packet_path": relative(packet_path), "packet_sha256": sha_bytes(packet),
                        "ACK_sha256": sha_bytes(ack), "packet_read_wall_ms": packet_read_ms,
                        "packet_replay_CPU_ms": replay_cpu_ms, "packet_replay_wall_ms": replay_wall_ms,
                        "source_sender_metadata_bytes": int(sequence[0]["sender_metadata_bytes"]),
                        "source_shared_projection_matrix_bytes": int(sequence[0]["shared_projection_matrix_bytes"])}
            record, png, metadata = snapshot(decoder.state, identity)
            if source and record["archive_RGB_sha256"] != task_lookup[(cap, step)]["receiver_archive_RGB_sha256"]:
                raise ValueError("historical archive differs from original task receiver evidence")
            if step == len(sequence):
                final = out / f"area_{index}" / f"cap_{cap}"
                final.mkdir(parents=True, exist_ok=True)
                (final / "final_archive.png").write_bytes(png)
                (final / "final_state.json").write_bytes(metadata)
            records.append(record)
            sequence_records.append(record)
        updates = [r for r in sequence_records if not r["bootstrap"]]
        area_summaries.append({"area_index": index, "aoi": aoi["aoi_id"], "budget": cap, "updates": len(updates),
            "raw_archive_bytes": updates[0]["raw_archive_bytes"], "bootstrap_PNG_bytes": sequence_records[0]["PNG_bytes"],
            "update_PNG_bytes_mean": statistics.mean(r["PNG_bytes"] for r in updates),
            "update_PNG_bytes_min": min(r["PNG_bytes"] for r in updates), "update_PNG_bytes_max": max(r["PNG_bytes"] for r in updates),
            "final_PNG_bytes": updates[-1]["PNG_bytes"],
            **{name + "_mean": statistics.mean(r[name] for r in updates) for name in
               ("encode_CPU_ms", "decode_CPU_ms", "encode_wall_ms", "decode_wall_ms", "packet_replay_CPU_ms", "packet_replay_wall_ms", "PNG_state_metadata_json_bytes")},
            "existing_fullcache_summary_bytes": updates[0]["existing_fullcache_summary_bytes"],
            "source_sender_metadata_bytes": updates[0]["source_sender_metadata_bytes"]})
        save(out / "progress.json", {"status": "in_progress", "sequences": len(area_summaries), "snapshots": len(records), "expected_snapshots": 558})
        print(json.dumps({"area": index, "cap": cap, "snapshots": len(records), "mean_PNG_bytes": area_summaries[-1]["update_PNG_bytes_mean"]}), flush=True)
    if len(records) != 558 or len(area_summaries) != 24:
        raise ValueError("profile cohort incomplete")
    for name, digest in {**spec["source_sha256"], **spec["input_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError("bound source/input changed during profiling: " + name)
    csv_save(out / "records.csv", records)
    csv_save(out / "per_aoi.csv", area_summaries)
    cap_rows = []
    for cap in CAPS:
        members = [r for r in area_summaries if r["budget"] == cap]
        cap_rows.append({"budget": cap, "AOIs": 8, "sum_AOI_mean_PNG_bytes": sum(r["update_PNG_bytes_mean"] for r in members),
            "sum_AOI_final_PNG_bytes": sum(r["final_PNG_bytes"] for r in members),
            "sum_raw_archive_bytes": sum(r["raw_archive_bytes"] for r in members),
            "sum_AOI_mean_state_metadata_json_bytes": sum(r["PNG_state_metadata_json_bytes_mean"] for r in members),
            "existing_fullcache_summary_bytes_eight_AOIs": sum(r["existing_fullcache_summary_bytes"] for r in members),
            "source_sender_metadata_bytes_eight_AOIs": sum(r["source_sender_metadata_bytes"] for r in members),
            "Gaussian_eight_summaries_plus_one_matrix_bytes": 8 * 12328 + 2359296,
            "aggregate_is_simultaneous_peak": False})
    csv_save(out / "per_cap.csv", cap_rows)
    updates = [r for r in records if not r["bootstrap"]]
    summary = {"status": "complete", "AOIs": 8, "caps": list(CAPS), "snapshots": 558, "update_snapshots": 534,
        "bootstrap_snapshots": 24, "all_decoded_archives_exact": True,
        "all_update_archive_hashes_match_original_task_receipts": True,
        "historical_state": "receiver.archive before coarse render; full-cache sender retains identical committed ROI history",
        "no_render_no_rerouting_no_inference": True, "PNG_codec": spec["codec"], "per_aoi": area_summaries, "per_cap": cap_rows,
        "update_timing": {name: {"mean": statistics.mean(r[name] for r in updates), "min": min(r[name] for r in updates),
                                  "max": max(r[name] for r in updates)} for name in ("encode_CPU_ms", "decode_CPU_ms", "encode_wall_ms", "decode_wall_ms", "packet_replay_CPU_ms", "packet_replay_wall_ms")},
        "decode_only_scope": "one PNG-to-RGB materialization; separate from reconstructing/re-encoding new archive",
        "encode_store_scope": "one RGB-to-PNG encode per changed archive; encode timing excludes final-file persistence",
        "historical_PNG_excludes": ["current RGB", "current coarse field", "summary", "metadata", "provisional state", "decoded working RGB", "runtime allocations"],
        "metadata_scope": "explicit canonical UTF8 JSON sidecar counted separately; source sender metadata and DCT summary separately reported",
        "aggregate_scope": "sum of each AOI temporal mean within one cap; not a simultaneous peak-memory measurement",
        "observed_wall_seconds": time.perf_counter() - start,
        "profile_outputs_are_not_algorithm_persistent_state": True}
    save(out / "summary.json", summary)
    artifact_hashes = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "progress.json"}
    artifact_sizes = {name: (out / name).stat().st_size for name in artifact_hashes}
    save(out / "complete.json", {"status": "complete", "snapshots": 558, "update_snapshots": 534, "AOIs": 8,
        "source_spec_sha256": sha(out / "run_spec.json"), "summary_sha256": sha(out / "summary.json"),
        "records_sha256": sha(out / "records.csv"), "artifact_sha256": artifact_hashes,
        "artifact_bytes": artifact_sizes, "receipt_files_are_not_state_memory": True})
    save(out / "progress.json", {"status": "complete", "sequences": 24, "snapshots": 558, "expected_snapshots": 558})
    print(json.dumps({"status": "complete", "wall_seconds": summary["observed_wall_seconds"], "per_cap": cap_rows, "update_timing": summary["update_timing"]}), flush=True)


if __name__ == "__main__":
    main()
