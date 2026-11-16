"""Metadata-only reservation of fresh pairs for the task/evidence repair."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_second_scaled as source

OUT = ROOT / "data/second_task_evidence_reservation_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads((source.PILOT / "source_record.json").read_text("utf-8"))
    archive = next(r for r in record["files"] if r["key"] == "SECOND.zip")
    remote = source.PoliteRangeFile(archive["size"], 1.6)
    infos, _, metadata = source.load_inventory(remote)
    earlier_paths = [ROOT / "data/second_pilot/manifest.json", ROOT / "data/second_scaled/manifest.json",
                     ROOT / "data/second_locked_test_v1/manifest.json"]
    excluded = {p["id"] for path in earlier_paths for p in json.loads(path.read_text("utf-8"))["pairs"]}
    members = [name for name in infos if name.startswith("SECOND/test/T1/") and name.endswith(".png")
               and PurePosixPath(name).stem not in excluded]
    rank = lambda name: hashlib.sha256(("SafeSync-task-state-20261104:" + PurePosixPath(name).name).encode()).hexdigest()
    members.sort(key=rank)
    selected = [dict(id=PurePosixPath(name).stem, filename=PurePosixPath(name).name, selection_hash=rank(name)) for name in members[:256]]
    proposed = dict(schema="task-evidence-fresh-reservation-v1", source_split="test", pixels_accessed=0,
        selection_rule="SHA256('SafeSync-task-state-20261104:' + filename), ascending after excluding all acquired IDs",
        metadata=metadata, previous_input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in earlier_paths},
        excluded_pair_ids=sorted(excluded), remaining_complete_pairs=len(members), reserved_pairs=selected,
        purpose="Fresh confirmation if the task-state/atlas repair passes development comparisons",
        restriction="No reserved image or label access before its separate method lock")
    if len(selected) != 256 or set(p["id"] for p in selected) & excluded:
        raise ValueError("Fresh reservation is incomplete or overlaps acquired input")
    if OUT.exists():
        stored = json.loads(OUT.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "reserved_utc"} != proposed:
            raise ValueError("Refusing to change fresh reserved pairs")
    else:
        proposed["reserved_utc"] = datetime.now(timezone.utc).isoformat()
        source.atomic_json(OUT, proposed)
    print(json.dumps(dict(status="metadata_reserved", pairs=256, pixels_accessed=0,
        public_pixel_requests=remote.request_count, remaining_candidates=len(members))), flush=True)


if __name__ == "__main__":
    main()
