"""Fixed public learned-image control on the reserved SECOND image pairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from safesync.learned_codec import HyperpriorCodec, receive, select, ACK_BYTES
from scripts.run_learned_codec import cached_candidates
from scripts.run_second_pixel import initialize
from scripts.run_locked_second_test import LOCK as ORIGINAL_LOCK, write_json, sha

OUT = ROOT / "results/second_learned/confirmation_v1"
RESOURCES = ROOT / "resources/learned_codec"
CAPS = (4096, 8192, 16384, 32768)


def lock_method():
    previous = json.loads(ORIGINAL_LOCK.read_text("utf-8"))
    source_names = ("scripts/run_second_learned_control.py", "safesync/learned_codec.py", "safesync/entropy_bridge.py",
                    "resources/learned_codec/entropy_server.py", "scripts/run_learned_codec.py", "safesync/pixel_wire.py")
    proposed = dict(schema="SECOND-learned-image-v1", architecture="bmshj2018-hyperprior", metric="mse",
        qualities=list(range(1, 9)), scales=[1., .75, .5, .375, .25], caps=CAPS,
        pair_ids=previous["pair_ids"], original_lock_sha256=sha(ORIGINAL_LOCK),
        selection="Lowest sender RGB MSE among actual encoded fixed candidates fitting packet+60-byte ACK cap",
        source_sha256={p: sha(ROOT / p) for p in source_names},
        public_registry_sha256=sha(RESOURCES / "public_downloads.json"),
        native_extensions_sha256=sha(RESOURCES / "native_extensions_probe.json"),
        scope="Additional strong codec control fixed before reserved numerical outcomes were inspected; all 256 pairs and all caps retained",
        labels_used_for_encoding=False, training_performed=False)
    proposed = json.loads(json.dumps(proposed))
    path = OUT / "method_lock.json"
    if path.exists():
        stored = json.loads(path.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != proposed:
            raise ValueError("Changed SECOND learned-codec control")
        return stored
    proposed["locked_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, proposed)
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    lock = lock_method()
    if args.freeze_only:
        print(json.dumps(dict(status="locked", pairs=256, caps=CAPS)), flush=True)
        return
    codec = HyperpriorCodec(RESOURCES, device="cuda", threads=2, entropy_bridge=True)
    assets = codec.prepare()
    write_json(OUT / "assets.json", assets)
    manifest = json.loads((ROOT / "data/second_locked_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    all_rows = []
    for pair_id in lock["pair_ids"]:
        folder = OUT / "pairs" / pair_id
        receipt_path = folder / "complete.json"
        for source, expected in lock["source_sha256"].items():
            if sha(ROOT / source) != expected:
                raise ValueError("Learned-codec control source changed after lock")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["lock_sha256"] != sha(OUT / "method_lock.json") or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed learned-codec pair receipt")
            rows = json.loads((folder / "records.json").read_text("utf-8"))
        else:
            folder.mkdir(parents=True, exist_ok=True)
            pair = pairs[pair_id]
            for key in ("before", "after"):
                item = pair["files"][key]
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Changed sender RGB input")
            before = np.asarray(Image.open(ROOT / pair["files"]["before"]["path"]).convert("RGB"))
            current = np.asarray(Image.open(ROOT / pair["files"]["after"]["path"]).convert("RGB"))
            prior, initial_down, initial_ack = initialize(before, pair_id, folder)
            identity = dict(lock_sha256=sha(OUT / "method_lock.json"), source_rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                            runtime=codec.runtime, weights=assets["weights"])
            candidates, search_ms, _ = cached_candidates(codec, current, folder / "candidates", identity)
            rows = []
            for cap in CAPS:
                packet, metadata = select(candidates, budget=cap - ACK_BYTES, region=prior.region,
                    base_version=prior.version, base_digest=prior.digest(), observation=1)
                started = time.perf_counter()
                state, ack = receive(prior, packet, codec)
                receiver_ms = (time.perf_counter() - started) * 1000
                replay, replay_ack = receive(prior, packet, codec)
                if (not np.array_equal(state.rgb, replay.rgb) or ack != replay_ack
                        or len(packet) + len(ack) > cap or len(ack) != ACK_BYTES):
                    raise AssertionError("Learned-image replay or actual cap violation")
                path = folder / f"b{cap}.bin"
                path.write_bytes(packet)
                path.with_suffix(".ack").write_bytes(ack)
                png = path.with_suffix(".received.png")
                Image.fromarray(state.rgb).save(png)
                rows.append(dict(pair_id=pair_id, method="hyperprior", budget=cap,
                    initial_down_bytes=initial_down, initial_ack_bytes=initial_ack,
                    down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet) + len(ack),
                    packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                    received_png=png.relative_to(ROOT).as_posix(), received_png_sha256=sha(png),
                    receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),
                    complete_search_ms=search_ms, selected_receiver_ms=receiver_ms,
                    codec_metadata=metadata))
            write_json(folder / "records.json", rows)
            write_json(receipt_path, dict(status="complete", records=4, lock_sha256=sha(OUT / "method_lock.json"),
                records_sha256=sha(folder / "records.json")))
        all_rows.extend(rows)
        write_json(OUT / "progress.json", dict(status="in_progress", completed_pairs=len(all_rows) // 4,
            expected_pairs=256, records=len(all_rows), expected_records=1024))
        print(f"SECOND hyperprior {len(all_rows)//4}/256, {len(all_rows)} actual received packets", flush=True)
    write_json(OUT / "records.json", all_rows)
    write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=len(all_rows), expected=1024,
        lock_sha256=sha(OUT / "method_lock.json"), records_sha256=sha(OUT / "records.json"), assets_sha256=sha(OUT / "assets.json"),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
