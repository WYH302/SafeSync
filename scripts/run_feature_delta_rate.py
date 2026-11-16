"""Equal-decoded-feature actual-rate control for correct closed-loop Delta."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_real_link_recovery import digest, encode, receive, save_csv, save_json, sha


def main():
    cache = ROOT / "cache/sn7_frozen_v1"
    out = ROOT / "results/feature_rate/equal_decode_v3"
    out.mkdir(parents=True, exist_ok=True)
    complete = json.loads((cache / "complete.json").read_text("utf-8"))
    spec = dict(source_sha256=sha(Path(__file__)), codec_source_sha256=sha(ROOT / "scripts/run_real_link_recovery.py"),
                cache_sha256=sha(cache / "complete.json"),
                claim="Same decoded int8 frozen features at every observation, compare complete versus reconstructed-reference residual coding",
                entropy_codec="same zlib6; full int8 or exact int16 residuals; actual SR12 session/header/checksum/feedback included",
                link="no loss, immediate actual ACK, shared charged bootstrap",
                utility_scope="feature equality; no separate RGB evidence or semantic safety claim")
    spec_path = out / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed equal-decode study")
    save_json(spec_path, spec)
    import shutil
    shutil.copyfile(Path(__file__), out / "source_snapshot.py")
    shutil.copyfile(ROOT / "scripts/run_real_link_recovery.py", out / "codec_snapshot.py")
    rows, areas = [], []
    for area in complete["areas"]:
        source = ROOT / area["path"]
        if sha(source) != area["sha256"]:
            raise ValueError("Changed cached features")
        with np.load(source, allow_pickle=False) as stored:
            f = stored["features"].astype(np.float32)
        q = np.rint(f * 127).clip(-127, 127).astype(np.int8)
        states = {"absolute": np.zeros_like(q[0]), "delta": np.zeros_like(q[0])}
        session = digest((area["aoi"] + ":" + sha(cache / "run_spec.json")).encode())
        boot = encode(states["delta"], q[0], 0, 1, 0, True, session)
        initial, _, ack, _ = receive(states["delta"], 0, boot, None, None, session)
        states = {name: initial.copy() for name in states}
        folder = out / area["aoi"]
        folder.mkdir(exist_ok=True)
        (folder / "initial.bin").write_bytes(boot)
        for step in range(1, len(q)):
            result = dict(aoi=area["aoi"], step=step,
                          split="confirmation" if "confirmation" in area["source_manifest"] else "calibration" if area["area_index"] < 2 else "development",
                          bootstrap_bytes=len(boot) + len(ack),
                          quantization_mse=float(np.mean((f[step].astype(np.float64) - q[step].astype(np.float64) / 127) ** 2)))
            for method in states:
                packet = encode(states[method], q[step], step, step + 1, step, method == "absolute", session)
                decoded, _, feedback, status = receive(states[method], step, packet, None, None, session)
                if status != "accepted" or not np.array_equal(decoded, q[step]):
                    raise AssertionError("Equal-decoding control failed")
                (folder / f"{method}_{step:03d}.bin").write_bytes(packet)
                result[method + "_bytes"] = len(packet) + len(feedback)
                states[method] = decoded
            result["receiver_features_identical"] = bool(np.array_equal(states["absolute"], states["delta"]))
            rows.append(result)
        group = [r for r in rows if r["aoi"] == area["aoi"]]
        absolute, delta = sum(r["absolute_bytes"] for r in group), sum(r["delta_bytes"] for r in group)
        areas.append(dict(aoi=area["aoi"], split=group[0]["split"], updates=len(group),
                          absolute_update_bytes=absolute, delta_update_bytes=delta,
                          bootstrap_bytes=len(boot) + len(ack),
                          relative_update_saving=1 - delta / absolute,
                          relative_complete_sequence_saving=1 - (delta + len(boot) + len(ack)) / (absolute + len(boot) + len(ack))))
    save_csv(out / "records.csv", rows)
    save_csv(out / "per_aoi.csv", areas)
    macro = []
    for split in ("calibration", "development", "confirmation"):
        group = [r for r in areas if r["split"] == split]
        macro.append(dict(split=split, AOIs=len(group),
                          mean_relative_update_saving=float(np.mean([r["relative_update_saving"] for r in group])),
                          mean_relative_complete_sequence_saving=float(np.mean([r["relative_complete_sequence_saving"] for r in group]))))
    if len(rows) != 270:
        raise AssertionError("Incomplete observation count")
    save_json(out / "complete.json", dict(status="complete", observations=len(rows), expected=270,
               source_sha256=sha(Path(__file__)), spec_sha256=sha(spec_path),
               records_sha256=sha(out / "records.csv"), all_features_equal=all(r["receiver_features_identical"] for r in rows), macro=macro))
    print(json.dumps(macro), flush=True)


if __name__ == "__main__":
    main()
