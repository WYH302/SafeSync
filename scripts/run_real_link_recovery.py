"""Real-monthly-feature transport under predeclared simulated link faults.

All policies share source features, quantizer, zlib codec, link random numbers,
and bootstrap. A correct ACK/retransmission baseline is included. Packet payloads
are actually encoded/decoded, with reference/version checks and byte accounting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import zlib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/sn7_frozen_v1"
HEADER = struct.Struct("<4sBIIIHH16s16s16sI")
FEEDBACK = struct.Struct("<4sBII16s16s")
CHECKSUM = struct.Struct("<I")
POLICIES = ("delta_arq", "periodic_4", "periodic_8", "periodic_16", "risk_anchor", "unrecovered_delta")
PROFILES = {
    "lossless": dict(down_loss=0., ack_loss=0., delay=0, burst=False),
    "iid10": dict(down_loss=.1, ack_loss=.1, delay=1, burst=False),
    "iid30": dict(down_loss=.3, ack_loss=.1, delay=2, burst=False),
    "burst": dict(down_loss=0., ack_loss=.1, delay=1, burst=True),
}
SEEDS = (1729, 2718, 31415)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).digest()[:16]


def save_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", "utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def encode(reference, current, base, target, observation, anchor, session):
    # Residuals use int16 so exact differences between int8 endpoints cannot wrap.
    payload = current.tobytes() if anchor else (current.astype(np.int16) - reference).astype("<i2").tobytes()
    compressed = zlib.compress(payload, 6)
    body = HEADER.pack(b"SR12", int(anchor), base, target, observation, *current.shape, session,
                       bytes(16) if anchor else digest(reference.tobytes()), digest(current.tobytes()), len(compressed)) + compressed
    return body + CHECKSUM.pack(zlib.crc32(body))


def receive(reference, version, wire, last_wire, last_ack, session):
    if wire == last_wire:
        return reference, version, last_ack, "duplicate"
    if len(wire) < HEADER.size + CHECKSUM.size or zlib.crc32(wire[:-4]) != CHECKSUM.unpack(wire[-4:])[0]:
        raise ValueError("Bad checksum or truncated transport")
    magic, anchor, base, target, obs, cells, dims, context, old_hash, new_hash, size = HEADER.unpack_from(wire)
    if magic != b"SR12" or anchor not in (0, 1) or context != session or reference.shape != (cells, dims) or len(wire) != HEADER.size + size + 4:
        raise ValueError("Bad transport schema or dimensions")
    if target <= version or (not anchor and (base != version or target != base + 1 or old_hash != digest(reference.tobytes()))):
        feedback = FEEDBACK.pack(b"RF11", 0, version, obs, digest(wire), digest(reference.tobytes()))
        return reference, version, feedback, "rejected_reference"
    raw = zlib.decompress(wire[HEADER.size:-4])
    expected = cells * dims * (1 if anchor else 2)
    if len(raw) != expected:
        raise ValueError("Bad decompressed length")
    if anchor:
        current = np.frombuffer(raw, np.int8).reshape(reference.shape).copy()
    else:
        summed = reference.astype(np.int16) + np.frombuffer(raw, "<i2").reshape(reference.shape)
        if np.any(summed < -127) or np.any(summed > 127):
            raise ValueError("Residual overflow")
        current = summed.astype(np.int8)
    if digest(current.tobytes()) != new_hash:
        raise ValueError("Decoded current state hash mismatch")
    feedback = FEEDBACK.pack(b"RF11", 1, target, obs, digest(wire), new_hash)
    return current, target, feedback, "accepted"


def save_packet(folder, wire):
    identifier = hashlib.sha256(wire).hexdigest()
    path = folder / (identifier + ".bin")
    if not path.exists():
        path.write_bytes(wire)
    return identifier


def run_sequence(features, policy, profile, seed, identity, packets):
    session = digest((identity + ":" + sha(CACHE / "run_spec.json")).encode())
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "little"))
    uniforms = rng.random((len(features), 2))
    phase = int(rng.integers(0, 12))
    codes = np.rint(features.astype(np.float32) * 127).clip(-127, 127).astype(np.int8)
    sender = np.zeros_like(codes[0])
    initial = encode(sender, codes[0], 0, 1, 0, True, session)
    receiver, receiver_version, boot_ack, status = receive(sender, 0, initial, None, None, session)
    if status != "accepted" or not np.array_equal(receiver, codes[0]):
        raise AssertionError("Bootstrap decode failed")
    sender = receiver.copy()
    sender_version = 1
    last_wire, last_ack = initial, boot_ack
    pending = None
    feedback_queue = []
    feedback_risk = False
    last_confirmed_observation = 0
    latest_received_observation = 0
    rows = []
    save_packet(packets, initial)

    def process_feedback(now):
        nonlocal pending, sender, sender_version, feedback_risk, last_confirmed_observation
        ready = [event for event in feedback_queue if event[0] <= now]
        feedback_queue[:] = [event for event in feedback_queue if event[0] > now]
        for _, body in ready:
            magic, accepted, version, observation, transaction, state_hash = FEEDBACK.unpack(body)
            if magic != b"RF11":
                raise AssertionError("Feedback schema")
            if accepted:
                last_confirmed_observation = max(last_confirmed_observation, observation)
            if policy == "delta_arq" and pending is not None and accepted and transaction == digest(pending[0]):
                if version != pending[2] or state_hash != digest(pending[1].tobytes()):
                    raise AssertionError("ACK does not bind pending reconstruction")
                sender, sender_version = pending[1], pending[2]
                pending = None
            elif not accepted:
                feedback_risk = True

    for step in range(1, len(features)):
        process_feedback(step)
        current = codes[step]
        timeout = step - last_confirmed_observation > profile["delay"] + 1
        period = int(policy.split("_")[1]) if policy.startswith("periodic_") else None
        anchor = (period is not None and step % period == 0) or (policy == "risk_anchor" and (feedback_risk or timeout))
        retry = policy == "delta_arq" and pending is not None
        if retry:
            packet, target_codes, target_version, target_obs = pending
            anchor = False
        else:
            packet = encode(sender, current, sender_version, sender_version + 1, step, anchor, session)
            target_codes, target_version, target_obs = current.copy(), sender_version + 1, step
            if policy == "delta_arq":
                pending = (packet, target_codes, target_version, target_obs)
        feedback_risk = False
        identifier = save_packet(packets, packet)
        lost = uniforms[step, 0] < profile["down_loss"] or (profile["burst"] and (step + phase) % 12 in (0, 1))
        feedback_bytes = 0
        outcome = "lost"
        if not lost:
            receiver, receiver_version, ack, outcome = receive(receiver, receiver_version, packet, last_wire, last_ack, session)
            if outcome == "accepted":
                latest_received_observation = target_obs
                last_wire, last_ack = packet, ack
            feedback_bytes = len(ack)
            save_packet(packets, ack)
            if uniforms[step, 1] >= profile["ack_loss"]:
                feedback_queue.append((step + profile["delay"], ack))
        if policy != "delta_arq":
            sender, sender_version = target_codes, target_version
        process_feedback(step)
        world_mse = float(np.mean((features[step].astype(np.float64) - receiver.astype(np.float64) / 127) ** 2))
        reference_mse = float(np.mean(((sender.astype(np.float64) - receiver.astype(np.float64)) / 127) ** 2))
        rows.append(dict(step=step, down_bytes=len(packet), feedback_bytes=feedback_bytes,
                         total_bytes=len(packet) + feedback_bytes, anchor=int(anchor), retry=int(retry),
                         outcome=outcome, latest_observation=latest_received_observation,
                         age=step - latest_received_observation, receiver_version=receiver_version,
                         sender_version=sender_version, world_mse=world_mse, reference_mse=reference_mse,
                         quantization_mse=float(np.mean((features[step].astype(np.float64) - current.astype(np.float64) / 127) ** 2)),
                         packet_sha256=identifier, bootstrap_sha256=hashlib.sha256(initial).hexdigest(),
                         session=session.hex(), receiver_sha256=digest(receiver.tobytes()).hex(),
                         receiver_confirms_current=int(latest_received_observation == step)))
    return rows, len(initial) + len(boot_ack)


def aggregate(out, records, area_split):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["aoi"], row["policy"], row["profile"], row["seed"])].append(row)
    per_area = []
    for (aoi, policy, profile, seed), group in grouped.items():
        per_area.append(dict(aoi=aoi, split=area_split[aoi], policy=policy, profile=profile, seed=seed,
                             updates=len(group), mean_update_bytes=float(np.mean([r["total_bytes"] for r in group])),
                             mean_world_mse=float(np.mean([r["world_mse"] for r in group])),
                             mean_reference_mse=float(np.mean([r["reference_mse"] for r in group])),
                             mean_age=float(np.mean([r["age"] for r in group])),
                             current_fraction=float(np.mean([r["receiver_confirms_current"] for r in group])),
                             anchors=sum(r["anchor"] for r in group), retries=sum(r["retry"] for r in group),
                             rejected=sum(r["outcome"] == "rejected_reference" for r in group)))
    save_csv(out / "per_aoi_seed.csv", per_area)
    macro = []
    for split in ("calibration", "development", "confirmation"):
        for profile in PROFILES:
            for policy in POLICIES:
                group = [r for r in per_area if (r["split"], r["profile"], r["policy"]) == (split, profile, policy)]
                if group:
                    macro.append(dict(split=split, profile=profile, policy=policy,
                                      AOIs=len({r["aoi"] for r in group}),
                                      **{key: float(np.mean([r[key] for r in group])) for key in (
                                          "mean_update_bytes", "mean_world_mse", "mean_reference_mse", "mean_age", "current_fraction", "anchors", "retries", "rejected")}))
    save_csv(out / "macro.csv", macro)
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/link_recovery/real_features_v2")
    parser.add_argument("--limit-areas", type=int, default=0)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    packets = args.out / "packets"
    packets.mkdir(exist_ok=True)
    cache = json.loads((CACHE / "complete.json").read_text("utf-8"))
    areas = cache["areas"][:args.limit_areas or None]
    spec = dict(schema="real-feature-link-v2", policies=POLICIES, profiles=PROFILES, seeds=SEEDS,
                source_sha256=sha(Path(__file__)), cache_spec_sha256=sha(CACHE / "run_spec.json"),
                input_sha256={a["path"]: a["sha256"] for a in areas},
                limit_areas=args.limit_areas,
                session="16-byte identity binds geographic AOI and frozen feature-cache specification",
                assumptions="Monthly source features; one transmission opportunity per observation; feedback delays are observation opportunities, not measured radio latency; reliable common bootstrap",
                quantizer="normalized feature int8 nearest 1/127; exact int16 closed-loop differences; zlib6",
                recovery_scope="ARQ current samples may wait behind an immutable pending transaction; optimistic policies detect/reject bad references; weak no-recovery arm is diagnostic",
                calibration_selection="All periods retained; no policy/period selected on confirmation")
    spec_path = args.out / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != json.loads(json.dumps(spec)):
        raise ValueError("Changed experiment specification; use another output")
    save_json(spec_path, spec)
    shutil.copyfile(Path(__file__), args.out / "source_snapshot.py")
    all_records = []
    area_split = {}
    start = time.perf_counter()
    for area in areas:
        if sha(ROOT / area["path"]) != area["sha256"]:
            raise ValueError("Feature input changed")
        with np.load(ROOT / area["path"], allow_pickle=False) as stored:
            features = stored["features"].astype(np.float32)
        split = "confirmation" if "confirmation" in area["source_manifest"] else "calibration" if area["area_index"] < 2 else "development"
        area_split[area["aoi"]] = split
        for profile, config in PROFILES.items():
            for seed in SEEDS:
                for policy in POLICIES:
                    rows, bootstrap_bytes = run_sequence(features, policy, config, seed, area["aoi"], packets)
                    for row in rows:
                        row.update(aoi=area["aoi"], policy=policy, profile=profile, seed=seed, bootstrap_bytes=bootstrap_bytes)
                    if profile == "lossless" and (any(r["age"] for r in rows) or any(r["reference_mse"] for r in rows)):
                        raise AssertionError("Lossless closed-loop numerical recovery failed")
                    all_records.extend(rows)
        save_csv(args.out / "records.csv", all_records)
        save_json(args.out / "progress.json", dict(status="in_progress", completed_areas=len(area_split), expected_areas=len(areas), records=len(all_records)))
        print(f"real-link {len(area_split)}/{len(areas)} AOIs, {len(all_records)} actual decoded opportunities", flush=True)
    macro = aggregate(args.out, all_records, area_split)
    expected = sum(len(a["source_images"]) - 1 for a in areas) * len(PROFILES) * len(SEEDS) * len(POLICIES)
    if expected != len(all_records):
        raise AssertionError("Incomplete fault matrix")
    save_json(args.out / "complete.json", dict(status="complete", records=len(all_records), expected=expected,
               source_sha256=sha(Path(__file__)), spec_sha256=sha(spec_path), records_sha256=sha(args.out / "records.csv"),
               macro_sha256=sha(args.out / "macro.csv"), seconds=time.perf_counter() - start,
               complete_utc=datetime.now(timezone.utc).isoformat(), macro=macro))
    print(json.dumps({"status": "complete", "records": len(all_records), "seconds": time.perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
