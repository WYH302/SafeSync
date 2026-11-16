"""Acquire a separate pre-reserved public subset after the repaired-method lock."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_second_scaled as source
from fetch_locked_second_test import extract

TARGET = ROOT / "data/second_task_context_test_v1"
LOCK = ROOT / "results/task_evidence/fresh_context_v1/method_lock.json"
RESERVATION = ROOT / "data/second_task_evidence_reservation_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = json.loads(LOCK.read_text("utf-8"))
    for path, expected in lock["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Fresh method source differs before acquisition")
    if sha(RESERVATION) != lock["reservation_sha256"]:
        raise ValueError("Fresh reservation changed")
    selection = json.loads(RESERVATION.read_text("utf-8"))["reserved_pairs"]
    if [p["id"] for p in selection] != lock["pair_ids"] or len(selection) != 256:
        raise ValueError("Fresh locked IDs differ")
    TARGET.mkdir(parents=True, exist_ok=True)
    source.TARGET = TARGET
    original = json.loads((source.PILOT / "source_record.json").read_text("utf-8"))
    archive = next(r for r in original["files"] if r["key"] == "SECOND.zip")
    remote = source.PoliteRangeFile(archive["size"], 1.6)
    infos, ends, metadata = source.load_inventory(remote)
    members = [f"SECOND/test/{folder}/{p['filename']}" for p in selection for folder, _ in source.FOLDERS]
    if any(m not in infos for m in members):
        raise ValueError("Fresh member is missing")
    remote.allowed_training_ranges = [(infos[m].header_offset, ends[infos[m].header_offset]) for m in members]
    plan = dict(lock_sha256=sha(LOCK), reservation_sha256=sha(RESERVATION), pairs=256, files=1280,
        source_bytes=sum(infos[m].file_size for m in members), metadata=metadata, source=source.URL,
        acquisition_source_sha256=sha(Path(__file__)))
    plan_path = TARGET / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text("utf-8")) != plan:
        raise ValueError("Fresh acquisition plan changed")
    source.atomic_json(plan_path, plan)
    print(json.dumps(dict(status="fresh_acquisition_started", pairs=256, source_bytes=plan["source_bytes"])), flush=True)
    path = TARGET / "manifest.json"
    previous = json.loads(path.read_text("utf-8")) if path.exists() else {}
    complete = {p["id"]: p for p in previous.get("pairs", [])}

    def fetch_pair(selected):
        pair = dict(id=selected["id"], group_id=selected["id"], split="confirmation", source_split="test", files={})

        def fetch_one(folder_key):
            folder, key = folder_key
            member = f"SECOND/test/{folder}/{selected['filename']}"
            destination = TARGET / folder / selected["filename"]
            content = extract(member, destination, infos, ends, remote)
            return key, dict(path=destination.relative_to(ROOT).as_posix(), bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(), archive_member=member, zip_crc32=f"{infos[member].CRC:08x}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            pair["files"].update(executor.map(fetch_one, source.FOLDERS))
        source.audit_pair(pair)
        return pair

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fetch_pair, p) for p in selection if p["id"] not in complete]
        for future in concurrent.futures.as_completed(futures):
            pair = future.result()
            complete[pair["id"]] = pair
            source.atomic_json(path, dict(complete=len(complete) == 256, lock_sha256=sha(LOCK), expected_pairs=256,
                pairs_completed=len(complete), updated_utc=datetime.now(timezone.utc).isoformat(),
                pairs=[complete[p["id"]] for p in selection if p["id"] in complete]))
            print(f"fresh locked SECOND {len(complete)}/256 pairs", flush=True)
    source.atomic_json(TARGET / "complete.json", dict(status="complete", pairs=256, files=1280,
        manifest_sha256=sha(path), plan_sha256=sha(plan_path), lock_sha256=sha(LOCK), requests=remote.request_count,
        transferred_bytes=remote.transferred))


if __name__ == "__main__":
    main()
