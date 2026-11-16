"""Freeze and acquire a pair-disjoint SECOND calibration/validation extension.

Only selected official-training ZIP members are requested. Official-test pixels
are never requested; their precommitted IDs are metadata-only reservations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import struct
import threading
import time
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import requests
from PIL import Image

from fetch_second import RangeFile, URL

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/second_scaled"
PILOT = ROOT / "data/second_pilot"
PROTOCOL = "SafeSync-SECOND-confirm-20261028-v1"
SEED_PREFIX = "SafeSync-confirm-20261028:"
CALIBRATION_COUNT = 128
VALIDATION_COUNT = 64
RESERVED_TEST_COUNT = 256
FOLDERS = (("T1", "before"), ("T2", "after"),
           ("GT_T1", "label_before"), ("GT_T2", "label_after"),
           ("GT_CD", "binary_change"))
CLASSES = {0: "unchanged_or_semantically_unlabelled", 1: "low_vegetation",
           2: "ground", 3: "trees", 4: "water", 5: "buildings", 6: "sports_field"}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class PoliteRangeFile(RangeFile):
    """Reuse the pilot seek/read adapter, but never write its caches."""

    def __init__(self, size: int, requests_per_second: float):
        super().__init__(URL, size)
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.interval = 1.0 / requests_per_second
        self.local = threading.local()
        self.request_count = 0
        self.cache_reused_bytes = 0
        self.allowed_training_ranges = []

    def throttle(self):
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_request)
            self.next_request = scheduled + self.interval
        time.sleep(max(0.0, scheduled - now))

    def get(self, start: int, end: int) -> bytes:
        for cached_start, body in self.cache:
            if cached_start <= start and end < cached_start + len(body):
                return body[start-cached_start:end-cached_start+1]
        if not any(a <= start <= end <= b for a, b in self.allowed_training_ranges):
            raise RuntimeError(f"Refusing non-selected-training byte range {start}-{end}")
        if end - start + 1 > 196608:
            return b"".join(self.get(a, min(end, a + 196607))
                            for a in range(start, end + 1, 196608))
        cache_name = f"{start}-{end}.bin"
        local_cache = TARGET / "ranges" / cache_name
        for cached in (local_cache, PILOT / "ranges" / cache_name):
            if cached.exists() and cached.stat().st_size == end - start + 1:
                with self.lock:
                    self.cache_reused_bytes += end - start + 1
                return cached.read_bytes()
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers["User-Agent"] = "SafeSync-public-research/1.0"
        for attempt in range(6):
            try:
                self.throttle()
                with self.local.session.get(
                    URL, headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 90), stream=True,
                ) as response:
                    with self.lock:
                        self.request_count += 1
                    if response.status_code in (429, 500, 502, 503, 504):
                        retry = response.headers.get("Retry-After", "")
                        delay = min(60, float(retry) if retry.isdigit() else 5 * (attempt + 1))
                        print(f"HTTP {response.status_code}: retry in {delay}s", flush=True)
                        time.sleep(delay)
                        continue
                    if response.status_code != 206:
                        raise RuntimeError(f"HTTP {response.status_code}; refusing full archive")
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Unexpected HTTP Content-Range")
                    body = response.content
                if len(body) != end - start + 1:
                    raise RuntimeError("Truncated HTTP range")
                with self.lock:
                    self.transferred += len(body)
                local_cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = local_cache.with_suffix(".tmp")
                temporary.write_bytes(body)
                os.replace(temporary, local_cache)
                return body
            except (requests.RequestException, RuntimeError) as error:
                print(f"Range {start}-{end} attempt {attempt+1}: {error}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(30, 2 * (attempt + 1)))
        raise RuntimeError("HTTP retry budget exhausted")


def load_inventory(remote):
    """Parse saved ZIP metadata. No image or label data is read here."""
    tail = (PILOT / "zip_tail.bin").read_bytes()
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise RuntimeError("Saved ZIP end record missing")
    fields = struct.unpack_from("<4s4H2IH", tail, offset)
    directory_size, directory_start = fields[5], fields[6]
    directory = (PILOT / "zip_directory.bin").read_bytes()
    if len(directory) != remote.size - directory_start:
        raise RuntimeError("Saved central-directory size does not match archive")
    remote.cache = [(directory_start, directory)]
    with zipfile.ZipFile(remote) as archive:
        entries = archive.infolist()
    infos = {entry.filename: entry for entry in entries if not entry.is_dir()}
    ordered_offsets = sorted({entry.header_offset for entry in entries} | {directory_start})
    ends = {a: b - 1 for a, b in zip(ordered_offsets, ordered_offsets[1:])}
    metadata = {"directory_start": directory_start, "directory_bytes": directory_size,
                "saved_directory_sha256": hashlib.sha256(directory).hexdigest(),
                "saved_tail_sha256": hashlib.sha256(tail).hexdigest()}
    return infos, ends, metadata


def frozen_selection(infos, pilot_manifest, archive_item, directory_metadata):
    pilot_ids = {pair["id"] for pair in pilot_manifest["pairs"]}
    if len(pilot_ids) != 64:
        raise RuntimeError("Expected 64 unique exploratory pilot IDs")
    train = [member for member in infos
             if member.startswith("SECOND/train/T1/") and member.endswith(".png")
             and PurePosixPath(member).stem not in pilot_ids]
    rank = lambda member: hashlib.sha256(
        (SEED_PREFIX + PurePosixPath(member).name).encode()).hexdigest()
    train.sort(key=rank)
    required = CALIBRATION_COUNT + VALIDATION_COUNT
    if len(train) < required:
        raise RuntimeError("Insufficient non-pilot official-training pairs")
    selected = []
    for index, member in enumerate(train[:required]):
        name = PurePosixPath(member).name
        members = {key: member.replace("/T1/", f"/{folder}/") for folder, key in FOLDERS}
        if any(path not in infos for path in members.values()):
            raise RuntimeError(f"Incomplete pair in ZIP: {name}")
        selected.append({"id": PurePosixPath(name).stem, "filename": name,
                         "source_split": "train", "selection_rank": index,
                         "split": "calibration" if index < CALIBRATION_COUNT else "validation",
                         "selection_hash": rank(member), "members": members})
    payload = {"protocol": PROTOCOL, "source": URL,
               "archive_bytes": archive_item["size"],
               "archive_checksum": archive_item.get("checksum"),
               "selection_rule": "SHA256(UTF8('SafeSync-confirm-20261028:' + filename)), ascending",
               "excluded_exploratory_ids": sorted(pilot_ids),
               "calibration_pairs": CALIBRATION_COUNT, "validation_pairs": VALIDATION_COUNT,
               "selection_is_label_blind": True, "pairs": selected,
               "zip_metadata": directory_metadata}
    payload["selection_sha256"] = digest_json(payload)
    path = TARGET / "selection_before_label_access.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Refusing to change previously frozen pair selection")
    else:
        atomic_json(path, payload)
    test = sorted((member for member in infos
                   if member.startswith("SECOND/test/T1/") and member.endswith(".png")
                   and PurePosixPath(member).stem not in pilot_ids), key=rank)
    reserved = {"protocol": PROTOCOL, "source_split": "test", "pixels_downloaded": 0,
                "selection_rule": payload["selection_rule"],
                "available_official_test_pairs": len(test),
                "reserved_pairs": [{"id": PurePosixPath(member).stem,
                                    "filename": PurePosixPath(member).name,
                                    "selection_hash": rank(member)}
                                   for member in test[:RESERVED_TEST_COUNT]],
                "restriction": "Metadata-only reservation. No official test pixels may be read until method lock."}
    reserved["reservation_sha256"] = digest_json(reserved)
    reserved_path = TARGET / "reserved_official_test.json"
    if reserved_path.exists() and json.loads(reserved_path.read_text(encoding="utf-8")) != reserved:
        raise RuntimeError("Refusing to change official-test reservation")
    atomic_json(reserved_path, reserved)
    return payload


def extract_member(member, local, infos, ends, remote):
    info = infos[member]
    if not member.startswith("SECOND/train/"):
        raise RuntimeError("Only official-training pixels may be extracted")
    if local.exists():
        content = local.read_bytes()
        if len(content) == info.file_size and zlib.crc32(content) == info.CRC:
            return content
    block = remote.get(info.header_offset, ends[info.header_offset])
    if block[:4] != b"PK\x03\x04":
        raise RuntimeError("Invalid local ZIP header")
    filename_length, extra_length = struct.unpack_from("<HH", block, 26)
    actual_name = block[30:30+filename_length].decode("utf-8")
    if actual_name != member:
        raise RuntimeError("ZIP local and central filenames disagree")
    start = 30 + filename_length + extra_length
    compressed = block[start:start+info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise RuntimeError(f"Unsupported ZIP compression {info.compress_type}")
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise RuntimeError(f"ZIP size/CRC mismatch: {member}")
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(".png.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, local)
    return content


def audit_pair(pair):
    arrays = {}
    shapes, modes, histograms = {}, {}, {}
    for key, metadata in pair["files"].items():
        with Image.open(ROOT / metadata["path"]) as image:
            arrays[key] = np.asarray(image)
            shapes[key] = list(arrays[key].shape)
            modes[key] = image.mode
    spatial_shapes = {tuple(array.shape[:2]) for array in arrays.values()}
    if len(spatial_shapes) != 1:
        raise RuntimeError(f"Mismatched native coordinates/sizes: {pair['id']}")
    for key in ("before", "after"):
        if arrays[key].dtype != np.uint8 or arrays[key].ndim != 3 or arrays[key].shape[2] != 3:
            raise RuntimeError(f"Unexpected RGB image format: {pair['id']} {key}")
    for key in ("label_before", "label_after", "binary_change"):
        label = arrays[key]
        values, counts = np.unique(label, return_counts=True)
        allowed = {0, 255} if key == "binary_change" else set(CLASSES)
        if label.ndim != 2 or not set(values.tolist()) <= allowed:
            raise RuntimeError(f"Unexpected native labels: {pair['id']} {key} {values}")
        histograms[key] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    changed = arrays["binary_change"] == 255
    before_labelled, after_labelled = arrays["label_before"] > 0, arrays["label_after"] > 0
    result = {"id": pair["id"], "split": pair["split"], "shapes": shapes,
              "modes": modes, "histograms": histograms,
              "native_pixel_coordinates_preserved": True,
              "coordinate_check_scope": "same native dimensions and paired filename; no georegistration accuracy claim",
              "changed_pixels": int(changed.sum()), "total_pixels": int(changed.size),
              "semantic_before_outside_change": int((before_labelled & ~changed).sum()),
              "semantic_after_outside_change": int((after_labelled & ~changed).sum()),
              "change_without_either_semantic_label": int((changed & ~before_labelled & ~after_labelled).sum())}
    atomic_json(TARGET / "audit_pairs" / f"{pair['id']}.json", result)
    return result


def manifest_payload(selection, completed, remote, started, complete=False):
    return {"protocol": PROTOCOL, "source": URL, "complete": complete,
            "archive_bytes": selection["archive_bytes"],
            "archive_checksum": selection["archive_checksum"],
            "selection_sha256": selection["selection_sha256"],
            "selection": selection["selection_rule"],
            "pairs_expected": CALIBRATION_COUNT + VALIDATION_COUNT,
            "pairs_completed": len(completed),
            "split_counts_completed": dict(Counter(pair["split"] for pair in completed.values())),
            "downloaded_bytes_this_run": remote.transferred,
            "http_requests_this_run": remote.request_count,
            "range_cache_reused_bytes_this_run": remote.cache_reused_bytes,
            "run_started_utc": started,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "official_test_pixels_downloaded": 0,
            "class_mapping": CLASSES,
            "binary_change_labels": {"0": "unchanged", "255": "changed (not ignore)"},
            "geographic_independence": "Original image pairs are disjoint from each other and the exploratory pilot; geographic independence is unknown because this mirror provides no geographic grouping.",
            "pairs": [completed[pair["id"]] for pair in selection["pairs"]
                      if pair["id"] in completed]}


def aggregate_audit(audits, selection):
    result = {"protocol": PROTOCOL, "selection_sha256": selection["selection_sha256"],
              "pairs_audited": len(audits), "class_mapping": CLASSES,
              "all_native_coordinate_checks_passed": True,
              "official_test_pixels_read": 0, "splits": {}}
    for split in ("calibration", "validation"):
        members = [audit for audit in audits.values() if audit["split"] == split]
        totals = {key: Counter() for key in ("label_before", "label_after", "binary_change")}
        for audit in members:
            for key in totals:
                totals[key].update(audit["histograms"][key])
        result["splits"][split] = {"pairs": len(members),
                                  "histograms": {key: dict(value) for key, value in totals.items()},
                                  "native_shapes": dict(Counter(
                                      "x".join(map(str, audit["shapes"]["before"]))
                                      for audit in members)),
                                  **{key: sum(audit[key] for audit in members) for key in (
                                      "total_pixels", "changed_pixels", "semantic_before_outside_change",
                                      "semantic_after_outside_change", "change_without_either_semantic_label")}}
    atomic_json(TARGET / "label_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Acquire only the fixed selection prefix, without changing the selection")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--requests-per-second", type=float, default=1.6)
    args = parser.parse_args()
    if not 0 < args.requests_per_second <= 2.0:
        parser.error("requests-per-second must be >0 and <=2")
    if args.limit is not None and not 0 < args.limit <= CALIBRATION_COUNT + VALIDATION_COUNT:
        parser.error("limit must be 1..192")
    TARGET.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    record = json.loads((PILOT / "source_record.json").read_text(encoding="utf-8"))
    archive_item = next(item for item in record["files"] if item["key"] == "SECOND.zip")
    if archive_item["size"] != 3837645776 or archive_item.get("checksum") != "md5:0f1dae03304bd7a05a9bea9a792a0c89":
        raise RuntimeError("Unexpected public source archive identity")
    pilot_manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    if pilot_manifest["source"] != URL or pilot_manifest["archive_bytes"] != archive_item["size"]:
        raise RuntimeError("Pilot metadata belongs to another source archive")
    remote = PoliteRangeFile(archive_item["size"], args.requests_per_second)
    infos, ends, metadata = load_inventory(remote)
    selection = frozen_selection(infos, pilot_manifest, archive_item, metadata)
    atomic_json(TARGET / "source_provenance.json", {
        "source": URL, "zenodo_record": "https://zenodo.org/records/14037770",
        "archive": archive_item, "metadata_reused_read_only_from": "data/second_pilot/source_record.json",
        "source_record_sha256": hashlib.sha256((PILOT / "source_record.json").read_bytes()).hexdigest(),
        "preprocessed_mirror": "ChangeMamba SECOND mirror; native index mapping differs from original RGB palette",
        "metadata_only_zip_cache": metadata,
        "archive_md5_status": "Published source identity recorded, not independently verified because the complete ZIP is not downloaded",
        "per_file_integrity": "ZIP entry length and CRC32 verified; SHA256 recorded for each decoded file",
        "official_test_pixels_requested": 0})
    print(f"Frozen {CALIBRATION_COUNT} calibration + {VALIDATION_COUNT} validation pairs; "
          f"excluded {len(selection['excluded_exploratory_ids'])} pilot IDs. "
          f"Reserved {RESERVED_TEST_COUNT} test IDs metadata-only.", flush=True)
    if args.plan_only:
        return
    all_members = [member for pair in selection["pairs"] for member in pair["members"].values()]
    remote.allowed_training_ranges = [(infos[member].header_offset, ends[infos[member].header_offset])
                                      for member in all_members]
    def fetch_pair(selected):
        pair = {"id": selected["id"], "group_id": selected["id"],
                "split": selected["split"], "source_split": "train",
                "selection_rank": selected["selection_rank"], "files": {}}

        def fetch_file(folder_key):
            folder, key = folder_key
            member = selected["members"][key]
            local = TARGET / folder / selected["filename"]
            content = extract_member(member, local, infos, ends, remote)
            return key, {"path": local.relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256(content).hexdigest(),
                         "bytes": len(content), "zip_crc32": f"{infos[member].CRC:08x}",
                         "archive_member": member}

        # At most six network calls are in flight across two pairs. The shared
        # request-start limiter still caps the server load at 1.6 requests/sec.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as file_pool:
            pair["files"].update(file_pool.map(fetch_file, FOLDERS))
        return pair, audit_pair(pair)

    completed, audits = {}, {}
    previous_path = TARGET / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("selection_sha256") != selection["selection_sha256"]:
            raise RuntimeError("Existing manifest uses another frozen selection")
        selections_by_id = {pair["id"]: pair for pair in selection["pairs"]}
        for pair in previous["pairs"]:
            selected = selections_by_id.get(pair["id"])
            audit_path = TARGET / "audit_pairs" / f"{pair['id']}.json"
            if selected is None or not audit_path.exists():
                continue
            valid = True
            for folder, key in FOLDERS:
                local = TARGET / folder / selected["filename"]
                info = infos[selected["members"][key]]
                content = local.read_bytes() if local.exists() else b""
                if (len(content) != info.file_size or zlib.crc32(content) != info.CRC
                        or hashlib.sha256(content).hexdigest() != pair["files"][key]["sha256"]):
                    valid = False
                    break
            if valid:
                completed[pair["id"]] = pair
                audits[pair["id"]] = json.loads(audit_path.read_text(encoding="utf-8"))
    atomic_json(TARGET / "manifest.json", manifest_payload(
        selection, completed, remote, started,
        len(completed) == CALIBRATION_COUNT + VALIDATION_COUNT))
    selected_jobs = selection["pairs"][:args.limit] if args.limit else selection["pairs"]
    selected_jobs = [pair for pair in selected_jobs if pair["id"] not in completed]
    expected_this_run = len(completed) + len(selected_jobs)
    print(f"Reused {len(completed)} verified complete pairs; acquiring {len(selected_jobs)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_pair, selected): selected for selected in selected_jobs}
        for future in concurrent.futures.as_completed(pending):
            selected = pending[future]
            try:
                pair, audit = future.result()
            except Exception as error:
                atomic_json(TARGET / "last_error.json", {"id": selected["id"],
                            "exception": type(error).__name__, "message": str(error),
                            "utc": datetime.now(timezone.utc).isoformat()})
                for outstanding in pending:
                    outstanding.cancel()
                raise
            completed[pair["id"]], audits[pair["id"]] = pair, audit
            atomic_json(TARGET / "manifest.json", manifest_payload(
                selection, completed, remote, started, len(completed) == CALIBRATION_COUNT + VALIDATION_COUNT))
            print(f"{len(completed)}/{expected_this_run} {pair['id']} {pair['split']} "
                  f"downloaded={remote.transferred / 1e6:.1f}MB requests={remote.request_count}", flush=True)
    aggregate_audit(audits, selection)
    print(f"Finished: {len(completed)} pairs, {remote.transferred} network bytes; "
          "official test pixel requests=0", flush=True)


if __name__ == "__main__":
    main()
