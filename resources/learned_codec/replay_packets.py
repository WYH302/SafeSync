"""Isolated full-sequence bootstrap+LC09 receiver producing actual ground PNGs.

The only pixel inputs are counted bootstrap/update packet bytes. No sender
current/candidate images, validity labels, GT or ground model are opened.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from safesync.learned_codec import HyperpriorCodec, receive
from safesync.pixel_wire import HEADER as BOOT_HEAD, PixelState, decode_update


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def pixel_files(folder, rgb, *, width, height):
    folder.parent.mkdir(parents=True, exist_ok=True)
    full = folder.with_suffix(".padded.png")
    Image.fromarray(rgb).save(full)
    native_rgb = rgb[:height, :width]
    native = folder.with_suffix(".native.png") if native_rgb.shape != rgb.shape else full
    if native != full:
        Image.fromarray(native_rgb).save(native)
    return {"padded_png_path": full.relative_to(ROOT).as_posix(),
        "padded_png_sha256": sha(full.read_bytes()), "padded_rgb_sha256": sha(rgb.tobytes()),
        "native_png_path": native.relative_to(ROOT).as_posix(),
        "native_png_sha256": sha(native.read_bytes()), "native_rgb_sha256": sha(native_rgb.tobytes()),
        "native_width": width, "native_height": height,
        "padded_width": rgb.shape[1], "padded_height": rgb.shape[0]}


def source_rows(folder, aid, spec, follow):
    marker = folder / "complete_aois" / (aid + ".json")
    if marker.exists():
        info = read(marker)
        path = ROOT / info["receipt_path"]
        if info["status"] != "complete" or sha(path.read_bytes()) != info["receipt_sha256"]:
            raise ValueError("uncommitted/corrupt source AOI receipt")
        if info["spec_sha256"] != sha((folder / "run_spec.json").read_bytes()):
            raise ValueError("source AOI frozen specification differs")
        result = read(path)["records"]
        if len(result) != info["expected_outputs"]:
            raise ValueError("source AOI output count differs")
        return result
    if follow:
        return None
    if read(folder / "summary.json")["status"] != "complete":
        raise ValueError("source run is incomplete")
    return [r for r in read(folder / "records.json") if r["aoi"] == aid]


def replay_aoi(folder, aid, rows, codec, spec_hash):
    target = folder / "fresh_replay"
    first = rows[0]
    bootstrap_path = folder / "initial" / aid / "initial.bin"
    bootstrap = bootstrap_path.read_bytes()
    if sha(bootstrap) != first["initial_packet_sha256"]:
        raise ValueError("actual bootstrap identity differs")
    fields = BOOT_HEAD.unpack(bootstrap[:BOOT_HEAD.size])
    block, height, width, region = fields[3], fields[4], fields[5], fields[11]
    blank = PixelState(np.zeros((height, width, 3), np.uint8), region, block)
    initial, initial_ack, _ = decode_update(blank, bootstrap)
    expected_ack = bootstrap_path.with_suffix(".ack").read_bytes()
    if initial_ack != expected_ack or sha(initial.rgb.tobytes()) != first["initial_rgb_sha256"]:
        raise ValueError("fresh bootstrap RGB/ACK differs")
    iw = first.get("initial_native_width", first["source_native_width"])
    ih = first.get("initial_native_height", first["source_native_height"])
    initial_pngs = pixel_files(target / "initial" / aid / "initial", initial.rgb, width=iw, height=ih)
    if initial_pngs["native_rgb_sha256"] != first["initial_native_rgb_sha256"]:
        raise ValueError("fresh native bootstrap RGB differs")
    init_ack_path = target / "initial" / aid / "initial.ack"
    init_ack_path.write_bytes(initial_ack)
    initial_receipt = {**initial_pngs, "bootstrap_path": bootstrap_path.relative_to(ROOT).as_posix(),
        "bootstrap_sha256": sha(bootstrap), "down_bytes": len(bootstrap), "ack_bytes": len(initial_ack),
        "ack_path": init_ack_path.relative_to(ROOT).as_posix(), "ack_sha256": sha(initial_ack),
        "state_digest": initial.digest().hex()}
    initial_receipt.update({"fresh_initial_" + k: v for k,v in initial_pngs.items()})
    states = {cap: initial for cap in {r["cap"] for r in rows}}
    records = []
    for row in sorted(rows, key=lambda r: (r["step"], r["cap"])):
        packet = (ROOT / row["packet_path"]).read_bytes()
        if sha(packet) != row["packet_sha256"] or len(packet) != row["down_bytes"] or len(packet) > row["cap"]:
            raise ValueError("actual saved packet identity/size/cap differs")
        started = time.perf_counter()
        state, ack = receive(states[row["cap"]], packet, codec)
        codec.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if sha(state.rgb.tobytes()) != row["receiver_rgb_sha256"]:
            raise ValueError("fresh sequence receiver RGB differs")
        if sha(ack) != row["ack_sha256"] or ack != (ROOT / row["ack_path"]).read_bytes() or len(ack) != 60:
            raise ValueError("fresh sequence ACK differs")
        prefix = target / "decoded" / str(row["cap"]) / aid / f"{row['step']:03d}"
        pngs = pixel_files(prefix, state.rgb, width=row["source_native_width"], height=row["source_native_height"])
        if pngs["native_rgb_sha256"] != row["decoded_native_rgb_sha256"]:
            raise ValueError("fresh native receiver RGB differs")
        ack_path = prefix.with_suffix(".ack")
        ack_path.write_bytes(ack)
        states[row["cap"]] = state
        records.append({**row, **{"fresh_" + k: v for k,v in pngs.items()},
            "fresh_initial_native_png_path": initial_pngs["native_png_path"],
            "fresh_initial_padded_png_path": initial_pngs["padded_png_path"],
            "fresh_initial_native_png_sha256": initial_pngs["native_png_sha256"],
            "fresh_initial_native_rgb_sha256": initial_pngs["native_rgb_sha256"],
            "fresh_initial_padded_png_sha256": initial_pngs["padded_png_sha256"],
            "fresh_initial_padded_rgb_sha256": initial_pngs["padded_rgb_sha256"],
            "fresh_ack_path": ack_path.relative_to(ROOT).as_posix(), "fresh_ack_sha256": sha(ack),
            "fresh_state_digest": state.digest().hex(), "fresh_matches_recorded_receiver": True,
            "fresh_matches_recorded_ack": True, "fresh_sequence_decode_ms": elapsed})
    receipt = {"status": "complete", "aoi": aid, "actual_outputs": len(records),
        "expected_outputs": len(rows), "spec_sha256": spec_hash, "initial": initial_receipt,
        "records": records, "codec_runtime": codec.runtime,
        "decoder_inputs": "actual counted bootstrap + LC09 bytes + public weights/config only",
        "source_current_or_candidate_RGB_opened": False, "GT_validity_or_ground_model_opened": False}
    path = target / "aoi_receipts" / (aid + ".json")
    save(path, receipt)
    save(target / "complete_aois" / (aid + ".json"), {"status": "complete", "aoi": aid,
        "actual_outputs": len(records), "expected_outputs": len(rows),
        "spec_sha256": spec_hash, "receipt_path": path.relative_to(ROOT).as_posix(),
        "receipt_sha256": sha(path.read_bytes()), "all_RGB_and_ACK_hashes_match": True})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/learned_codec/confirmation_q1_8_v1")
    parser.add_argument("--follow", action="store_true", help="follow committed AOIs in a separately running sender/receiver")
    parser.add_argument("--deadline-seconds", type=int, default=5400)
    args = parser.parse_args()
    folder = ROOT / args.run
    deadline = time.monotonic() + args.deadline_seconds
    while not (folder / "run_spec.json").exists():
        if not args.follow or time.monotonic() >= deadline:
            raise TimeoutError("frozen run specification not available")
        time.sleep(2)
    spec = read(folder / "run_spec.json")
    spec_hash = sha((folder / "run_spec.json").read_bytes())
    codec = HyperpriorCodec(ROOT / "resources/learned_codec", entropy_bridge=True)
    completed = {}
    while len(completed) < len(spec["aoi_ids"]):
        changed = False
        for aid in spec["aoi_ids"]:
            if aid in completed:
                continue
            marker = folder / "fresh_replay/complete_aois" / (aid + ".json")
            if marker.exists():
                stored = read(marker)
                path = ROOT / stored["receipt_path"]
                if stored["spec_sha256"] != spec_hash or sha(path.read_bytes()) != stored["receipt_sha256"]:
                    raise ValueError("saved fresh replay receipt identity differs")
                receipt = read(path)
            else:
                rows = source_rows(folder, aid, spec, args.follow)
                if rows is None:
                    continue
                receipt = replay_aoi(folder, aid, rows, codec, spec_hash)
            completed[aid] = receipt
            changed = True
            all_rows = [r for item in completed.values() for r in item["records"]]
            result = {"status": "passed" if len(completed) == len(spec["aoi_ids"]) else "in_progress",
                "actual_outputs": len(all_rows), "expected_outputs": spec["expected_outputs"],
                "completed_aois": list(completed), "records": all_rows,
                "runtime": codec.runtime, "spec_sha256": spec_hash,
                "new_windows_process": True, "new_local_WSL_entropy_host": True,
                "all_RGB_and_ACK_hashes_match": True,
                "decoder_inputs": "actual bootstrap+LC09 packets and public models only",
                "source_RGB_or_labels_opened": False, "ground_model_loaded": False,
                "bridge_stats": codec.bridge.stats}
            save(folder / "fresh_replay_receipt.json", result)
            save(folder / "fresh_replay/progress.json", {k:v for k,v in result.items() if k not in ("records","bridge_stats")})
            print(f"fresh AOI complete {aid}: {len(all_rows)}/{spec['expected_outputs']} actual images; RGB+ACK matched", flush=True)
        if len(completed) == len(spec["aoi_ids"]):
            break
        if not args.follow or time.monotonic() >= deadline:
            raise TimeoutError("not all committed AOIs became available before replay deadline")
        if not changed:
            time.sleep(10)
    if sum(len(x["records"]) for x in completed.values()) != spec["expected_outputs"]:
        raise ValueError("fresh full-sequence replay cohort incomplete")
    codec.bridge.close()
    print(json.dumps({"status": "passed", "packets": spec["expected_outputs"],
        "all_RGB_and_ACK_hashes_match": True, "source_RGB_or_labels_opened": False}))


if __name__ == "__main__":
    main()
