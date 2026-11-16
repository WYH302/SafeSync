"""Acquire the previously reserved SECOND test subset after explicit method lock."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_second_scaled as source

TARGET = ROOT / "data/second_locked_test_v1"
LOCK = ROOT / "results/unknown_recovery/confirmation_lock_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(member, destination, infos, ends, remote):
    if not member.startswith("SECOND/test/"):
        raise ValueError("Test acquisition accepts the locked official-test whitelist")
    info = infos[member]
    if destination.exists():
        content = destination.read_bytes()
        if len(content) == info.file_size and zlib.crc32(content) == info.CRC:
            return content
    block = remote.get(info.header_offset, ends[info.header_offset])
    if block[:4] != b"PK\x03\x04":
        raise ValueError("ZIP local header mismatch")
    name_len, extra_len = struct.unpack_from("<HH", block, 26)
    if block[30:30 + name_len].decode("utf-8") != member:
        raise ValueError("ZIP member identity mismatch")
    start = 30 + name_len + extra_len
    compressed = block[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise ValueError("ZIP entry CRC or length mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text("utf-8"))
    for name, expected in lock["method_source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Locked method source changed before test acquisition")
    TARGET.mkdir(parents=True, exist_ok=True)
    source.TARGET = TARGET
    reservation_path = ROOT / "data/second_scaled/reserved_official_test.json"
    if sha(reservation_path) != lock["reservation_file_sha256"]:
        raise ValueError("Changed fixed test reservation")
    reservation = json.loads(reservation_path.read_text("utf-8"))
    selection = reservation["reserved_pairs"]
    if [p["id"] for p in selection] != lock["pair_ids"] or len(selection) != 256:
        raise ValueError("Changed locked test identities")
    record = json.loads((source.PILOT / "source_record.json").read_text("utf-8"))
    archive = next(r for r in record["files"] if r["key"] == "SECOND.zip")
    remote = source.PoliteRangeFile(archive["size"], 1.6)
    infos, ends, metadata = source.load_inventory(remote)
    members = [f"SECOND/test/{folder}/{p['filename']}" for p in selection for folder, _ in source.FOLDERS]
    if any(name not in infos for name in members):
        raise ValueError("Reserved test pair lacks a required ZIP member")
    remote.allowed_training_ranges = [(infos[name].header_offset, ends[infos[name].header_offset]) for name in members]
    plan = dict(method_lock_sha256=sha(LOCK), reservation_sha256=sha(reservation_path), pairs=256,
                expected_files=len(members), source_bytes=sum(infos[m].file_size for m in members),
                source=source.URL, inventory=metadata, acquisition_source_sha256=sha(Path(__file__)),
                scope="Previously reserved public official-test subset after method lock; no geographical independence assertion")
    plan_path = TARGET / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text("utf-8")) != plan:
        raise ValueError("Changed acquisition plan")
    source.atomic_json(plan_path, plan)
    print(json.dumps({k: plan[k] for k in ("pairs", "expected_files", "source_bytes")}), flush=True)
    if args.plan_only:
        return
    manifest_path = TARGET / "manifest.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    completed = {p["id"]: p for p in previous.get("pairs", [])}

    def fetch_pair(selected):
        pair = dict(id=selected["id"], group_id=selected["id"], split="confirmation", source_split="test", files={})

        def fetch_file(folder_key):
            folder, key = folder_key
            member = f"SECOND/test/{folder}/{selected['filename']}"
            path = TARGET / folder / selected["filename"]
            content = extract(member, path, infos, ends, remote)
            return key, dict(path=path.relative_to(ROOT).as_posix(), bytes=len(content),
                             sha256=hashlib.sha256(content).hexdigest(), archive_member=member,
                             zip_crc32=f"{infos[member].CRC:08x}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            pair["files"].update(pool.map(fetch_file, source.FOLDERS))
        source.audit_pair(pair)
        return pair

    remaining = [p for p in selection if p["id"] not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch_pair, p) for p in remaining]
        for future in concurrent.futures.as_completed(futures):
            pair = future.result()
            completed[pair["id"]] = pair
            source.atomic_json(manifest_path, dict(complete=len(completed) == 256,
                method_lock_sha256=sha(LOCK), expected_pairs=256, pairs_completed=len(completed),
                updated_utc=datetime.now(timezone.utc).isoformat(),
                pairs=[completed[p["id"]] for p in selection if p["id"] in completed]))
            print(f"locked SECOND test {len(completed)}/256 pairs", flush=True)
    if len(completed) != 256:
        raise ValueError("Incomplete locked acquisition")
    for pair in completed.values():
        for item in pair["files"].values():
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Changed acquired test file")
    source.atomic_json(TARGET / "complete.json", dict(status="complete", pairs=256, files=1280,
        manifest_sha256=sha(manifest_path), plan_sha256=sha(plan_path), source_sha256=sha(Path(__file__)),
        method_lock_sha256=sha(LOCK), transferred_bytes_this_run=remote.transferred,
        requests_this_run=remote.request_count))


if __name__ == "__main__":
    main()
