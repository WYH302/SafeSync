"""Reserve another source-pair-disjoint cohort using public ZIP metadata only."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import fetch_second_scaled as source

OUT = ROOT/"data/second_residual_reservation_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((source.PILOT/"source_record.json").read_text("utf-8"))
    archive = next(r for r in original["files"] if r["key"] == "SECOND.zip")
    remote = source.PoliteRangeFile(archive["size"],1.6)
    infos,_,metadata = source.load_inventory(remote)
    previous = [ROOT/f"data/{name}/manifest.json" for name in
        ("second_pilot","second_scaled","second_locked_test_v1","second_task_context_test_v1")]
    excluded = {p["id"] for path in previous for p in json.loads(path.read_text("utf-8"))["pairs"]}
    names = [name for name in infos if name.startswith("SECOND/test/T1/") and name.endswith(".png") and PurePosixPath(name).stem not in excluded]

    def rank(name):
        return hashlib.sha256(("SafeSync-residual-confirmation-20261104:"+PurePosixPath(name).name).encode()).hexdigest()

    names.sort(key=rank)
    selected = [dict(id=PurePosixPath(name).stem,filename=PurePosixPath(name).name,selection_hash=rank(name)) for name in names[:256]]
    proposed = dict(schema="residual-confirmation-reservation-v1",source_split="test",pixels_accessed=0,
        selection_rule="SHA256('SafeSync-residual-confirmation-20261104:' + filename), excluding every acquired SECOND pair ID",
        metadata=metadata,previous_input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in previous},
        excluded_pair_ids=sorted(excluded),remaining_candidates=len(names),reserved_pairs=selected,
        scope="A third confirmation cohort, pair-ID disjoint; geographic independence unavailable",public_pixel_requests=remote.request_count)
    if len(selected) != 256 or {p["id"] for p in selected} & excluded:
        raise ValueError("Residual reservation overlaps acquired data or is incomplete")
    if OUT.exists():
        stored = json.loads(OUT.read_text("utf-8"))
        if {k:v for k,v in stored.items() if k != "reserved_utc"} != proposed:
            raise ValueError("Refusing to replace the residual reservation")
    else:
        proposed["reserved_utc"] = datetime.now(timezone.utc).isoformat()
        source.atomic_json(OUT,proposed)
    print(json.dumps(dict(status="metadata_reserved",pairs=256,pixel_requests=remote.request_count,remaining=len(names))),flush=True)


if __name__ == "__main__":
    main()
