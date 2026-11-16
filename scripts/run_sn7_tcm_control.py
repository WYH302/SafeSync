"""Modern fixed-menu image control on the existing eight-AOI monthly cohort."""
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
from safesync.tcm_codec import TCMCodec, SCALES
from safesync.pixel_wire import PixelState, ACK, UPDATED, checksum
from scripts.run_learned_codec import cached_candidates, initial_state, open_sender_rgb
from scripts.run_locked_second_test import sha, write_json

OUT = ROOT / "results/tcm_sn7/transport_v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"
CAPS = (32768, 65536, 131072)


def receive(prior, packet, codec):
    rgb, meta = codec.decode_packet(packet)
    if (meta["region"] != prior.region or meta["base_version"] != prior.version or meta["base_digest"] != prior.digest()
            or meta["observation"] <= prior.observed_at or rgb.shape != prior.rgb.shape):
        raise ValueError("TCM monthly reference or geometry mismatch")
    state = PixelState(rgb, prior.region, prior.block, prior.version+1, meta["observation"], np.full_like(prior.statuses, UPDATED))
    ack = ACK.pack(b"TA10", state.version, state.observed_at, checksum(packet), state.digest())
    return state, ack+checksum(ack)


def freeze():
    names = ("scripts/run_sn7_tcm_control.py", "scripts/evaluate_sn7_tcm_control.py", "safesync/tcm_codec.py", "safesync/tcm_runtime.py",
        "safesync/learned_codec.py", "safesync/entropy_bridge.py", "safesync/pixel_wire.py", "scripts/run_learned_codec.py",
        "resources/learned_codec/entropy_server.py", "resources/modern_codec/tcm_source/models/tcm.py")
    spec = dict(source_sha256={n:sha(ROOT/n) for n in names}, manifest_sha256=sha(MANIFEST),
        weights_registry_sha256=sha(ROOT / "resources/modern_codec/weights/registry.json"), caps=list(CAPS), scales=list(SCALES),
        checkpoints=7, candidates_per_frame=35, expected_aois=8, expected_updates=178, expected_records=534,
        selection="Minimum actual-decoded padded native RGB MSE among fixed fitting TC10 packets, source RGB only",
        cost_contract="Actual downlink packet at most cap, plus separately counted60-byte ACK and shared paid lossless initialization",
        scope="Post-hoc modern comparator on the already inspected main cohort. No new independent geography or untouched-test claim.",
        gpu_memory_fraction=.45, source_training=False)
    path = OUT / "method_lock.json"
    if path.exists():
        if {k:v for k,v in json.loads(path.read_text("utf-8")).items() if k != "locked_utc"} != spec:
            raise ValueError("Monthly TCM control changed")
    else:
        write_json(path, dict(**spec, locked_utc=datetime.now(timezone.utc).isoformat()))
    return sha(path)


def run_area(index, aoi, codec, assets, lock_hash):
    folder = OUT / f"area_{index}"
    initial, initial_meta = initial_state(aoi, ROOT / "results/pixel_sequences/development_side4", OUT)
    states, rows = {cap:initial for cap in CAPS}, []
    for step, month in enumerate(aoi["months"][1:], 1):
        target = folder / f"step_{step:03d}"
        if (target / "complete.json").exists():
            receipt = json.loads((target / "complete.json").read_text("utf-8"))
            if receipt["control_lock_sha256"] != lock_hash or receipt["records_sha256"] != sha(target / "records.json"):
                raise ValueError("Changed monthly TCM checkpoint")
            group = json.loads((target / "records.json").read_text("utf-8"))
            if len(group) != 3:
                raise ValueError("Incomplete resumed monthly TCM output")
            for row in group:
                image = ROOT / row["padded_png"]
                if sha(image) != row["padded_png_sha256"] or sha(ROOT / row["packet_path"]) != row["packet_sha256"]:
                    raise ValueError("Changed resumed monthly receiver")
                rgb = np.asarray(Image.open(image).convert("RGB"))
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["receiver_rgb_sha256"]:
                    raise ValueError("Changed reconstructed monthly pixels")
                prior = states[row["cap"]]
                states[row["cap"]] = PixelState(rgb, prior.region, prior.block, row["target_version"], step, np.full_like(prior.statuses, UPDATED))
            rows.extend(group)
            continue
        current = open_sender_rgb(month)
        identity = dict(control_lock_sha256=lock_hash, source_png_sha256=month["derived_sha256"]["rgb_path"], runtime=codec.runtime, weights=assets["weights"])
        candidates, search_ms, _ = cached_candidates(codec, current, target / "candidates", identity)
        target.mkdir(parents=True, exist_ok=True)
        group = []
        for cap in CAPS:
            prior = states[cap]
            packet, metadata = codec.select(candidates, budget=cap, region=prior.region, base_version=prior.version,
                base_digest=prior.digest(), observation=step)
            path = target / f"cap_{cap}.bin"
            path.write_bytes(packet)
            started = time.perf_counter()
            state, ack = receive(prior, path.read_bytes(), codec)
            receiver_ms = (time.perf_counter()-started)*1000
            if len(packet) > cap or len(ack) != 60 or hashlib.sha256(state.rgb.tobytes()).hexdigest() != metadata["sender_decoded_rgb_sha256"]:
                raise AssertionError("Monthly TCM packet replay or cost mismatch")
            path.with_suffix(".ack").write_bytes(ack)
            padded = path.with_suffix(".padded.png")
            Image.fromarray(state.rgb).save(padded)
            w, h = month["dimensions"]
            native = path.with_suffix(".decoded.png")
            Image.fromarray(state.rgb[:h,:w]).save(native)
            states[cap] = state
            group.append(dict(aoi=aoi["aoi_id"], area_index=index, month=month["month"], step=step, method="tcm", cap=cap,
                down_bytes=len(packet), up_bytes=len(ack), actual_total_bytes=len(packet)+len(ack), target_version=state.version,
                packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path), ack_sha256=hashlib.sha256(ack).hexdigest(),
                padded_png=padded.relative_to(ROOT).as_posix(), padded_png_sha256=sha(padded),
                decoded_path=native.relative_to(ROOT).as_posix(), decoded_png_sha256=sha(native),
                receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(), decoded_native_rgb_sha256=hashlib.sha256(state.rgb[:h,:w].tobytes()).hexdigest(),
                sender_search_ms_shared=search_ms, receiver_ms=receiver_ms, **initial_meta, codec_metadata=metadata))
        write_json(target / "records.json", group)
        write_json(target / "complete.json", dict(status="complete", records=3, records_sha256=sha(target / "records.json"), control_lock_sha256=lock_hash))
        rows.extend(group)
        print(f"SN7 TCM area{index} update{step}/{len(aoi['months'])-1}", flush=True)
    write_json(folder / "records.json", rows)
    write_json(folder / "complete.json", dict(status="complete", records=len(rows), aoi=aoi["aoi_id"], records_sha256=sha(folder / "records.json"), control_lock_sha256=lock_hash))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    args = parser.parse_args()
    lock_hash = freeze()
    if args.lock_only:
        print(lock_hash, flush=True)
        return
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    if args.finalize_only:
        rows = []
        for index in range(8):
            folder = OUT / f"area_{index}"
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if receipt["control_lock_sha256"] != lock_hash or receipt["records_sha256"] != sha(folder / "records.json"):
                raise ValueError("Monthly TCM source changed")
            rows.extend(json.loads((folder / "records.json").read_text("utf-8")))
        if len(rows) != 534:
            raise AssertionError("Monthly TCM matrix incomplete")
        write_json(OUT / "records.json", rows)
        write_json(OUT / "complete.json", dict(status="complete", aois=8, updates=178, records=534, records_sha256=sha(OUT / "records.json"), control_lock_sha256=lock_hash))
        return
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid area worker assignment")
    codec = TCMCodec(ROOT)
    codec.torch.cuda.set_per_process_memory_fraction(.45)
    codec.runtime = dict(codec.runtime, gpu_memory_fraction=.45)
    assets = codec.prepare()
    write_json(OUT / f"assets_worker{args.worker_index}.json", assets)
    for index in range(args.worker_index, 8, args.workers):
        run_area(index, manifest["aois"][index], codec, assets, lock_hash)


if __name__ == "__main__":
    main()
