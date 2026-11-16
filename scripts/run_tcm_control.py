"""Fixed seven-checkpoint TCM control with the common decoded task service."""
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
from safesync import task_evidence_wire as wire
from safesync.pixel_wire import PixelState
from safesync.tcm_codec import TCMCodec, SCALES
from scripts.run_learned_codec import cached_candidates
from scripts.run_locked_second_test import sha, write_json

MAIN = ROOT / "results/task_evidence/residual_confirmation_v1"
OUT = ROOT / "results/task_evidence/tcm_control_v1"
DATA = ROOT / "data/second_residual_test_v1"


def specification():
    names = ("scripts/run_tcm_control.py", "scripts/evaluate_tcm_control.py", "safesync/tcm_codec.py",
        "safesync/tcm_runtime.py", "safesync/learned_codec.py", "safesync/entropy_bridge.py",
        "safesync/task_evidence_wire.py", "resources/learned_codec/entropy_server.py",
        "resources/modern_codec/tcm_source/models/tcm.py")
    return dict(source_sha256={n: sha(ROOT / n) for n in names},
        plan_sha256=sha(OUT / "plan.json"), amendment_sha256=sha(OUT / "padding_amendment.json"),
        weights_registry_sha256=sha(ROOT / "resources/modern_codec/weights/registry.json"),
        common_lock_sha256=sha(MAIN / "method_lock.json"), data_manifest_sha256=sha(DATA / "manifest.json"),
        scales=list(SCALES), checkpoints=7, candidates_per_pair=35, expected_pairs=256, expected_records=3072,
        selection="Minimum sender current RGB MSE among actual fitting candidates, including TE01 task state and60-byte ACK",
        scope="Additional comparator on the already acquired third cohort. Original menu preceded residual numerical inspection. Implementation finalized afterward. No new independent confirmation.")


def encode_pair(pair, codec, assets, control_hash):
    pair_id = pair["id"]
    folder = OUT / "pairs" / pair_id
    folder.mkdir(parents=True, exist_ok=True)
    ready_path = MAIN / "pairs" / pair_id / "sender_inputs_ready.json"
    ready = json.loads(ready_path.read_text("utf-8"))
    if ready["lock_sha256"] != sha(MAIN / "method_lock.json"):
        raise ValueError("Changed common task-state source")
    after = pair["files"]["after"]
    if sha(ROOT / after["path"]) != after["sha256"]:
        raise ValueError("Changed sender current RGB")
    current = np.asarray(Image.open(ROOT / after["path"]).convert("RGB"))
    identity = dict(control_lock_sha256=control_hash, source_rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
        runtime=codec.runtime, weights=assets["weights"])
    candidates, search_ms, _ = cached_candidates(codec, current, folder / "candidates", identity)
    if len(candidates) != 35:
        raise AssertionError("Incomplete fixed TCM candidate menu")
    rows, decoded = [], {}
    for heldout in (5, 3, 4):
        common = ready["common"][str(heldout)]
        boot_path, label_path = ROOT / common["bootstrap_path"], ROOT / common["labels_only_path"]
        if sha(boot_path) != common["bootstrap_sha256"] or sha(label_path) != common["labels_only_sha256"]:
            raise ValueError("Changed common source packets")
        boot = boot_path.read_bytes()
        head = wire.HEADER.unpack_from(boot)
        blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), head[1], head[2])
        prior, _ = wire.receive(blank, boot)
        service, _ = wire.receive(prior, label_path.read_bytes())
        empty, _ = wire.encode(prior, service.labels, observation=1)
        pixel_prior = PixelState(prior.rgb, prior.region, prior.block, prior.version, prior.observed_at, prior.statuses)

        def decode_image(base, payload):
            key = hashlib.sha256(payload).hexdigest()
            if key not in decoded:
                decoded[key] = codec.decode_packet(payload)
            rgb, meta = decoded[key]
            if (meta["region"] != base.region or meta["base_version"] != base.version
                    or meta["base_digest"] != pixel_prior.digest() or meta["observation"] != 1):
                raise ValueError("TCM nested reference mismatch")
            return rgb

        for cap in (4096, 8192, 16384, 32768):
            inner, metadata = codec.select(candidates, budget=cap-len(empty)-wire.ACK_BYTES, region=prior.region,
                base_version=prior.version, base_digest=pixel_prior.digest(), observation=1)
            packet, costs = wire.encode(prior, service.labels, observation=1, payload=inner, image_mode=wire.LEARNED)
            receiver = wire.TaskEvidenceReceiver(prior, learned_decoder=decode_image)
            ack, _ = receiver.receive(packet)
            state = receiver.state
            duplicate_ack, applied = receiver.receive(packet)
            replay, replay_ack = wire.receive(prior, packet, learned_decoder=decode_image)
            rgb_hash = hashlib.sha256(state.rgb.tobytes()).hexdigest()
            if (applied or duplicate_ack != ack or replay_ack != ack or replay.digest() != state.digest()
                    or len(packet)+len(ack) > cap or sum(costs.values()) != len(packet)
                    or not np.array_equal(state.labels, service.labels) or rgb_hash != metadata["sender_decoded_rgb_sha256"]):
                raise AssertionError("TCM actual budget, replay, task or reconstruction differs")
            path = folder / f"h{heldout}_b{cap}.bin"
            path.write_bytes(packet)
            path.with_suffix(".ack").write_bytes(ack)
            png = path.with_suffix(".received.png")
            Image.fromarray(state.rgb).save(png)
            rows.append(dict(pair_id=pair_id, heldout=heldout, method="tcm", seed=0, budget=cap,
                down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet)+len(ack),
                bootstrap_bytes=common["bootstrap_bytes"], bootstrap_path=common["bootstrap_path"],
                packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                receiver_rgb_sha256=rgb_hash, receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
                received_png=png.relative_to(ROOT).as_posix(), received_png_sha256=sha(png),
                complete_search_ms=search_ms, components=costs, codec_metadata=metadata))
    write_json(folder / "records.json", rows)
    write_json(folder / "complete.json", dict(status="complete", records=12, records_sha256=sha(folder / "records.json"),
        control_lock_sha256=control_hash, sender_inputs_sha256=sha(ready_path)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    lock_path = OUT / "method_lock.json"
    spec = specification()
    if lock_path.exists():
        if {k: v for k, v in json.loads(lock_path.read_text("utf-8")).items() if k != "locked_utc"} != spec:
            raise ValueError("TCM frozen control changed")
    else:
        write_json(lock_path, dict(**spec, locked_utc=datetime.now(timezone.utc).isoformat()))
    control_hash = sha(lock_path)
    if args.lock_only:
        print(control_hash, flush=True)
        return
    lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in json.loads((DATA / "manifest.json").read_text("utf-8"))["pairs"]}
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid disjoint worker assignment")
    if args.finalize_only:
        rows = []
        for pair_id in lock["pair_ids"]:
            folder = OUT / "pairs" / pair_id
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if receipt["control_lock_sha256"] != control_hash or receipt["records_sha256"] != sha(folder / "records.json"):
                raise ValueError("Completed TCM pair changed")
            rows.extend(json.loads((folder / "records.json").read_text("utf-8")))
        if len(rows) != 3072:
            raise AssertionError("Incomplete TCM matrix")
        write_json(OUT / "records.json", rows)
        write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=len(rows), records_sha256=sha(OUT / "records.json"),
            control_lock_sha256=control_hash, completed_utc=datetime.now(timezone.utc).isoformat()))
        return
    codec = TCMCodec(ROOT)
    assets = codec.prepare()
    write_json(OUT / f"assets_worker{args.worker_index}.json", assets)
    assigned = lock["pair_ids"][args.worker_index::args.workers]
    for index, pair_id in enumerate(assigned):
        folder = OUT / "pairs" / pair_id
        if (folder / "complete.json").exists():
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if receipt["control_lock_sha256"] != control_hash or receipt["records_sha256"] != sha(folder / "records.json"):
                raise ValueError("Changed resumable TCM pair")
        else:
            encode_pair(pairs[pair_id], codec, assets, control_hash)
        print(f"TCM worker{args.worker_index} {index+1}/{len(assigned)} complete", flush=True)


if __name__ == "__main__":
    main()
