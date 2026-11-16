"""Public hyperprior with the exact same decoded task-state service."""
from __future__ import annotations

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
from safesync import learned_codec as learned
from safesync.pixel_wire import PixelState
from scripts.run_learned_codec import cached_candidates
from scripts.run_locked_second_test import write_json, sha

MAIN = ROOT / "results/task_evidence/fresh_context_v1"
OUT = ROOT / "results/task_evidence/fresh_context_learned_v1"
RESOURCES = ROOT / "resources/learned_codec"


def main():
    method_lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    for relative in ("safesync/learned_codec.py", "safesync/entropy_bridge.py", "resources/learned_codec/entropy_server.py", "safesync/task_evidence_wire.py"):
        if sha(ROOT / relative) != method_lock["source_sha256"][relative]:
            raise ValueError("Prelocked learned-control implementation changed")
    if sha(RESOURCES / "public_downloads.json") != method_lock["public_hyperprior_registry_sha256"]:
        raise ValueError("Prelocked public weights registry changed")
    spec = dict(wrapper_sha256=sha(Path(__file__)), main_lock_sha256=sha(MAIN / "method_lock.json"),
        qualities=list(range(1, 9)), scales=[1, .75, .5, .375, .25], caps=method_lock["caps"],
        decision="Lowest sender current RGB MSE among actual fitting LC09 candidates after TE01 task-state/header and 60-byte ACK costs",
        online_inputs="Current RGB and hash-verified common bootstrap/current task-state packets; no semantic evaluation labels",
        scope="Separate control wrapper lock; network implementation, public registry, quality/scale grid already bound by the main preaccess lock")
    spec_path = OUT / "method_lock.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != spec:
            raise ValueError("Changed learned control wrapper")
    else:
        spec["locked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(spec_path, spec)
    codec = learned.HyperpriorCodec(RESOURCES, device="cuda", threads=2, entropy_bridge=True)
    assets = codec.prepare()
    write_json(OUT / "assets.json", assets)

    def decode_image(prior, packet):
        pixel_prior = PixelState(prior.rgb, prior.region, prior.block, prior.version, prior.observed_at, prior.statuses)
        parsed = learned.parse(packet)
        if (parsed["region"] != prior.region or parsed["base_version"] != prior.version
                or parsed["base_digest"] != pixel_prior.digest() or parsed["observation"] != 1):
            raise ValueError("Nested learned-image reference mismatch")
        rgb, _ = codec.decode_packet(packet)
        return rgb

    done, all_rows = set(), []
    for pair_id in method_lock["pair_ids"]:
        folder = OUT / "pairs" / pair_id
        if (folder / "complete.json").exists():
            receipt = json.loads((folder / "complete.json").read_text("utf-8"))
            if receipt["control_lock_sha256"] != sha(spec_path) or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed learned task-state control")
            all_rows.extend(json.loads((folder / "records.json").read_text("utf-8")))
            done.add(pair_id)
    while len(done) < 256:
        available_path = ROOT / "data/second_task_context_test_v1/manifest.json"
        available = json.loads(available_path.read_text("utf-8")) if available_path.exists() else {"pairs": []}
        pairs = {p["id"]: p for p in available["pairs"]}
        pending = [p for p in method_lock["pair_ids"] if p not in done and p in pairs and (MAIN / "pairs" / p / "sender_inputs_ready.json").exists()]
        if not pending:
            time.sleep(15)
            continue
        for pair_id in pending:
            folder = OUT / "pairs" / pair_id
            folder.mkdir(parents=True, exist_ok=True)
            ready_path = MAIN / "pairs" / pair_id / "sender_inputs_ready.json"
            ready = json.loads(ready_path.read_text("utf-8"))
            if ready["lock_sha256"] != sha(MAIN / "method_lock.json"):
                raise ValueError("Common task states belong to another method")
            after = pairs[pair_id]["files"]["after"]
            if sha(ROOT / after["path"]) != after["sha256"]:
                raise ValueError("Learned-control current image changed")
            current = np.asarray(Image.open(ROOT / after["path"]).convert("RGB"))
            identity = dict(control_lock_sha256=sha(spec_path), source_rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                runtime=codec.runtime, weights=assets["weights"])
            candidates, search_ms, _ = cached_candidates(codec, current, folder / "candidates", identity)
            rows = []
            for h in (5, 3, 4):
                common = ready["common"][str(h)]
                bootstrap = ROOT / common["bootstrap_path"]
                label_only = ROOT / common["labels_only_path"]
                if sha(bootstrap) != common["bootstrap_sha256"] or sha(label_only) != common["labels_only_sha256"]:
                    raise ValueError("Common counted task-state packets changed")
                boot_packet = bootstrap.read_bytes()
                header = wire.HEADER.unpack_from(boot_packet)
                blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16, 16), np.uint8), header[1], header[2])
                prior, _ = wire.receive(blank, boot_packet)
                task_state, _ = wire.receive(prior, label_only.read_bytes())
                empty, _ = wire.encode(prior, task_state.labels, observation=1)
                pixel_prior = PixelState(prior.rgb, prior.region, prior.block, prior.version, prior.observed_at, prior.statuses)
                for cap in method_lock["caps"]:
                    available_bytes = cap - len(empty) - wire.ACK_BYTES
                    inner, metadata = learned.select(candidates, budget=available_bytes, region=prior.region,
                        base_version=prior.version, base_digest=pixel_prior.digest(), observation=1)
                    packet, costs = wire.encode(prior, task_state.labels, observation=1, payload=inner, image_mode=wire.LEARNED)
                    receiver = wire.TaskEvidenceReceiver(prior, learned_decoder=decode_image)
                    ack, _ = receiver.receive(packet)
                    state = receiver.state
                    duplicate_ack, applied = receiver.receive(packet)
                    replay, replay_ack = wire.receive(prior, packet, learned_decoder=decode_image)
                    if (applied or duplicate_ack != ack or replay_ack != ack or replay.digest() != state.digest()
                            or len(packet)+len(ack) > cap or len(packet) != sum(costs.values())
                            or not np.array_equal(state.labels, task_state.labels)):
                        raise AssertionError("Learned task-state stream replay, task identity, or byte budget")
                    path = folder / f"h{h}_b{cap}.bin"
                    path.write_bytes(packet)
                    path.with_suffix(".ack").write_bytes(ack)
                    png = path.with_suffix(".received.png")
                    Image.fromarray(state.rgb).save(png)
                    rows.append(dict(pair_id=pair_id, heldout=h, method="hyperprior", seed=0, budget=cap,
                        down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet)+len(ack),
                        bootstrap_bytes=common["bootstrap_bytes"], bootstrap_path=common["bootstrap_path"],
                        packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                        receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),
                        receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
                        received_png=png.relative_to(ROOT).as_posix(), received_png_sha256=sha(png),
                        complete_search_ms=search_ms, components=costs, codec_metadata=metadata))
            if len(rows) != 12:
                raise AssertionError("Incomplete learned control pair")
            write_json(folder / "records.json", rows)
            write_json(folder / "complete.json", dict(status="complete", records=12, records_sha256=sha(folder / "records.json"),
                control_lock_sha256=sha(spec_path), common_sender_receipt_sha256=sha(ready_path)))
            done.add(pair_id)
            all_rows.extend(rows)
            write_json(OUT / "progress.json", dict(status="in_progress", completed_pairs=len(done), expected_pairs=256, records=len(all_rows), expected_records=3072))
            print(f"fresh common-task hyperprior {len(done)}/256, {len(all_rows)} actual streams", flush=True)
    write_json(OUT / "records.json", all_rows)
    write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=3072,
        control_lock_sha256=sha(spec_path), records_sha256=sha(OUT / "records.json"), assets_sha256=sha(OUT / "assets.json"),
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
