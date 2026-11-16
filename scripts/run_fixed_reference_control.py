"""Independent-update controls for the real-feature link recovery experiment."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_real_link_recovery as link

OUT = ROOT / "results/link_recovery/fixed_reference_v1"
POLICIES = ("fixed_reference", "absolute_refresh")


def receive_fixed(initial, current, version, packet, session):
    if len(packet) < link.HEADER.size + 4 or zlib.crc32(packet[:-4]) != link.CHECKSUM.unpack(packet[-4:])[0]:
        raise ValueError("Fixed-reference checksum or length")
    magic, anchor, base, target, observation, cells, dims, context, old_hash, new_hash, size = link.HEADER.unpack_from(packet)
    if (magic != b"SR12" or anchor != 0 or context != session or base != 1
            or initial.shape != (cells, dims) or len(packet) != link.HEADER.size + size + 4
            or old_hash != link.digest(initial.tobytes())):
        raise ValueError("Fixed-reference schema, session, or immutable base")
    if target <= version or target != observation + 1:
        raise ValueError("Fixed-reference stale or wrong observation")
    raw = zlib.decompress(packet[link.HEADER.size:-4])
    if len(raw) != initial.size * 2:
        raise ValueError("Fixed-reference residual shape")
    restored = initial.astype(np.int16) + np.frombuffer(raw, "<i2").reshape(initial.shape)
    if np.any(restored < -127) or np.any(restored > 127):
        raise ValueError("Fixed-reference residual overflow")
    restored = restored.astype(np.int8)
    if link.digest(restored.tobytes()) != new_hash:
        raise ValueError("Fixed-reference target hash")
    ack = link.FEEDBACK.pack(b"RF11", 1, target, observation, link.digest(packet), new_hash)
    return restored, target, ack


def run(features, policy, profile, seed, identity, packets):
    session = link.digest((identity + ":" + link.sha(link.CACHE / "run_spec.json")).encode())
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "little"))
    uniforms = rng.random((len(features), 2))
    phase = int(rng.integers(0, 12))
    codes = np.rint(features.astype(np.float32) * 127).clip(-127, 127).astype(np.int8)
    initial = link.encode(np.zeros_like(codes[0]), codes[0], 0, 1, 0, True, session)
    current, version, ack, status = link.receive(np.zeros_like(codes[0]), 0, initial, None, None, session)
    if status != "accepted":
        raise AssertionError("Common bootstrap failed")
    bootstrap_bytes = len(initial) + len(ack)
    link.save_packet(packets, initial)
    link.save_packet(packets, ack)
    latest = 0
    records = []
    for step in range(1, len(features)):
        absolute = policy == "absolute_refresh"
        packet = link.encode(codes[0], codes[step], 1, step + 1, step, absolute, session)
        identifier = link.save_packet(packets, packet)
        lost = uniforms[step, 0] < profile["down_loss"] or (profile["burst"] and (step + phase) % 12 in (0, 1))
        feedback_bytes = 0
        if not lost:
            if absolute:
                current, version, ack, status = link.receive(current, version, packet, None, None, session)
                if status != "accepted":
                    raise AssertionError("Independent absolute update was rejected")
            else:
                current, version, ack = receive_fixed(codes[0], current, version, packet, session)
            if not np.array_equal(current, codes[step]):
                raise AssertionError("Independent update did not restore exact current codes")
            latest = step
            feedback_bytes = len(ack)
            link.save_packet(packets, ack)
        records.append(dict(step=step, down_bytes=len(packet), feedback_bytes=feedback_bytes,
            total_bytes=len(packet) + feedback_bytes, outcome="lost" if lost else "accepted",
            latest_observation=latest, age=step - latest, receiver_version=version,
            world_mse=float(np.mean((features[step].astype(np.float64) - current.astype(np.float64) / 127) ** 2)),
            quantization_mse=float(np.mean((features[step].astype(np.float64) - codes[step].astype(np.float64) / 127) ** 2)),
            extra_receiver_reference_bytes=codes[0].nbytes if not absolute else 0,
            packet_sha256=identifier, bootstrap_bytes=bootstrap_bytes,
            scope="Same counted feedback as earlier controls; neither encoder requires feedback to encode a fresh current update"))
    return records


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    packets = OUT / "packets"
    packets.mkdir(exist_ok=True)
    areas = json.loads((link.CACHE / "complete.json").read_text("utf-8"))["areas"]
    spec = dict(policies=POLICIES, profiles=link.PROFILES, seeds=link.SEEDS,
        source_sha256=link.sha(Path(__file__)), codec_sha256=link.sha(ROOT / "scripts/run_real_link_recovery.py"),
        cache_spec_sha256=link.sha(link.CACHE / "run_spec.json"),
        input_sha256={a["path"]: a["sha256"] for a in areas},
        rationale="Strong simple independent-update controls added before interpreting the Anchor contribution",
        fixed_reference="Retain the common decoded bootstrap at both endpoints; every update is a direct exact residual from it",
        feedback="ACK bytes counted even though independent-update encoders do not depend on ACK arrivals",
        extra_receiver_memory_bytes=256 * 384,
        assumptions="Same source, quantizer, codec, bootstrap, random losses, monthly opportunities as real_features_v2")
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != json.loads(json.dumps(spec)):
        raise ValueError("Changed fixed-reference experiment")
    link.save_json(spec_path, spec)
    (OUT / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    all_records = []
    for area in areas:
        if link.sha(ROOT / area["path"]) != area["sha256"]:
            raise ValueError("Frozen feature cache changed")
        with np.load(ROOT / area["path"], allow_pickle=False) as stored:
            features = stored["features"].astype(np.float32)
        split = "confirmation" if "confirmation" in area["source_manifest"] else "calibration" if area["area_index"] < 2 else "development"
        for name, profile in link.PROFILES.items():
            for seed in link.SEEDS:
                for policy in POLICIES:
                    rows = run(features, policy, profile, seed, area["aoi"], packets)
                    for row in rows:
                        row.update(aoi=area["aoi"], split=split, profile=name, seed=seed, policy=policy)
                    all_records.extend(rows)
        link.save_csv(OUT / "records.csv", all_records)
        print(f"fixed-reference {area['aoi']} {len(all_records)} actual opportunities", flush=True)
    groups = defaultdict(list)
    for row in all_records:
        groups[(row["aoi"], row["split"], row["profile"], row["seed"], row["policy"])].append(row)
    summaries = []
    for (aoi, split, profile, seed, policy), rows in groups.items():
        summaries.append(dict(aoi=aoi, split=split, profile=profile, seed=seed, policy=policy,
            mean_update_bytes=float(np.mean([r["total_bytes"] for r in rows])),
            mean_world_mse=float(np.mean([r["world_mse"] for r in rows])),
            mean_age=float(np.mean([r["age"] for r in rows])),
            extra_receiver_reference_bytes=rows[0]["extra_receiver_reference_bytes"]))
    link.save_csv(OUT / "per_aoi_seed.csv", summaries)
    expected = sum(len(a["source_images"]) - 1 for a in areas) * 4 * 3 * 2
    if len(all_records) != expected:
        raise AssertionError("Incomplete independent-update controls")
    link.save_json(OUT / "complete.json", dict(status="complete", records=len(all_records), expected=expected,
        spec_sha256=link.sha(spec_path), records_sha256=link.sha(OUT / "records.csv"),
        per_aoi_seed_sha256=link.sha(OUT / "per_aoi_seed.csv"), seconds=time.perf_counter() - start))


if __name__ == "__main__":
    main()
