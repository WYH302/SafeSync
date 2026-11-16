"""Real-bitstream pretrained image control, with sender-only pixel-MSE fitting.

Default: original development AOI index 2, initialization plus two updates.
This runner never opens task/validity labels and does no ground-model inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from safesync.learned_codec import (ACK_BYTES, HEAD, QUALITIES, SCALES, WIRE_MAGIC, Candidate,
                                   HyperpriorCodec, file_sha256, receive, select, sha256)
from safesync.pixel_wire import UPDATED, PixelState, decode_update, encode_update

RESOURCES = ROOT / "resources/learned_codec"


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pad(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    return np.pad(a, ((0, -h % 64), (0, -w % 64), (0, 0)))


def open_sender_rgb(month: dict) -> np.ndarray:
    path = ROOT / month["rgb_path"]
    if file_sha256(path) != month["derived_sha256"]["rgb_path"]:
        raise ValueError("sender PNG differs from frozen manifest")
    with Image.open(path) as image:
        return pad(np.asarray(image.convert("RGB")))


def prepare_assets(codec: HyperpriorCodec) -> dict:
    info = codec.prepare()
    import compressai
    source = RESOURCES / "source" / "compressai-1.2.8.tar.gz"
    if not source.exists():
        metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/compressai/1.2.8/json"))
        item = next(f for f in metadata["urls"] if f["filename"] == source.name)
        data = urllib.request.urlopen(item["url"]).read()
        if sha256(data) != item["digests"]["sha256"]:
            raise ValueError("official PyPI source archive SHA256 differs")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        save_json(source.with_suffix(".download.json"), {"url": item["url"],
            "sha256": sha256(data), "bytes": len(data)})
    source_info = read_json(source.with_suffix(".download.json"))
    if file_sha256(source) != source_info["sha256"]:
        raise ValueError("saved official source archive changed")
    package = (RESOURCES / "source/vendor/compressai" if codec.bridge is not None
               else Path(compressai.__file__).parent)
    source_files = ["models/google.py", "zoo/image.py", "zoo/pretrained.py",
                    "entropy_models/entropy_models.py"]
    installed_sources = {}
    for name in source_files:
        path = package / name
        target = RESOURCES / "source" / name.replace("/", "__")
        shutil.copyfile(path, target)
        installed_sources[name] = file_sha256(path)
    info.update(source_archive=source_info, installed_source_sha256=installed_sources,
                source_home="https://github.com/InterDigitalInc/CompressAI",
                source_version="1.2.8", quality_grid=list(QUALITIES), scales=list(SCALES),
                training_performed=False, likelihood_rate_used=False,
                decode_input="actual rANS y/z bytes and model configuration")
    if codec.bridge is not None:
        native_probe = read_json(RESOURCES / "native_extensions_probe.json")
        for meta in native_probe["extensions"].values():
            binary = RESOURCES / "linux_wheel/compressai" / meta["file"]
            if file_sha256(binary) != meta["sha256"]:
                raise ValueError("official native entropy extension differs from validated binary")
        info["local_entropy_bridge"] = {"identity": codec.bridge.identity,
            "extensions": native_probe["extensions"],
            "implementation_source_sha256": {name: file_sha256(ROOT / name) for name in (
                "safesync/entropy_bridge.py", "resources/learned_codec/entropy_server.py")},
            "wheel_sha256": file_sha256(RESOURCES / "wheelhouse/compressai-1.2.8-cp310-cp310-manylinux_2_34_x86_64.whl"),
            "standard_compressai_installation": False,
            "network_operators_modified": False, "arithmetic_coder_reimplemented": False}
        info["official_loaded_source_sha256"] = {
            str(Path(module.__file__).relative_to(package)): file_sha256(Path(module.__file__))
            for name, module in sys.modules.items()
            if name.startswith("compressai.") and getattr(module, "__file__", None)
            and Path(module.__file__).is_relative_to(package)}
    save_json(RESOURCES / "asset_manifest.json", info)
    return info


def initial_state(aoi: dict, initial_root: Path, out: Path) -> tuple[PixelState, dict]:
    first = aoi["months"][0]
    image = open_sender_rgb(first)
    region = hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
    blank = PixelState(np.zeros_like(image), region)
    old = initial_root / aoi["aoi_id"] / "initial.bin"
    if old.exists():
        wire = old.read_bytes()
        origin = old.relative_to(ROOT).as_posix()
    else:
        h, w = image.shape[:2]
        n = h * w // 4096
        wire, _ = encode_update(shape=(h, w), block=64, region=region, base_version=0,
            base_digest=blank.digest(), observed_at=0, current_rgb=image, indices=np.arange(n),
            statuses=np.full(n, UPDATED, np.uint8), codec="png")
        origin = "generated shared lossless SP03 PNG initialization; fully charged"
    state, ack, _ = decode_update(blank, wire)
    if not np.array_equal(state.rgb, image) or len(ack) != ACK_BYTES:
        raise ValueError("shared paid initialization does not recover first PNG")
    folder = out / "initial" / aoi["aoi_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "initial.bin").write_bytes(wire)
    (folder / "initial.ack").write_bytes(ack)
    native_width, native_height = first["dimensions"]
    native_rgb = state.rgb[:native_height, :native_width]
    Image.fromarray(native_rgb).save(folder / "decoded.png")
    return state, {"initial_down_bytes": len(wire), "initial_ack_bytes": len(ack),
        "initial_packet_sha256": sha256(wire), "initial_ack_sha256": sha256(ack),
        "initial_rgb_sha256": sha256(state.rgb.tobytes()), "initial_origin": origin,
        "initial_native_rgb_sha256": sha256(native_rgb.tobytes()),
        "initial_native_width": native_width, "initial_native_height": native_height,
        "initial_decoded_path": (folder / "decoded.png").relative_to(ROOT).as_posix()}


def cached_candidates(codec, current, folder, identity) -> tuple[list[Candidate], float, bool]:
    meta_path = folder / "candidates.json"
    if meta_path.exists():
        stored = read_json(meta_path)
        if stored["identity"] != identity:
            raise ValueError("candidate checkpoint identity differs; use a new output folder")
        candidates = []
        for item in stored["candidates"]:
            streams = tuple((folder / item[key]).read_bytes() for key in ("y_file", "z_file"))
            if [sha256(stream) for stream in streams] != item["stream_sha256"]:
                raise ValueError("saved candidate entropy stream changed")
            candidates.append(Candidate(item["quality"], tuple(item["native_shape"]),
                tuple(item["coded_shape"]), tuple(item["padded_shape"]), tuple(item["latent_shape"]),
                item["weight_sha256"], streams, item["sender_pixel_mse"],
                item["sender_compress_ms"], item["sender_decode_ms"], item["decoded_rgb_sha256"]))
        return candidates, stored["sender_search_ms"], True
    started = time.perf_counter()
    candidates = codec.encode_candidates(current)
    search_ms = (time.perf_counter() - started) * 1000
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for k, candidate in enumerate(candidates):
        y_name, z_name = f"{k:02d}.y.bin", f"{k:02d}.z.bin"
        (folder / y_name).write_bytes(candidate.streams[0])
        (folder / z_name).write_bytes(candidate.streams[1])
        records.append({"quality": candidate.quality, "native_shape": candidate.native_shape,
            "coded_shape": candidate.coded_shape, "padded_shape": candidate.padded_shape,
            "latent_shape": candidate.latent_shape, "weight_sha256": candidate.weight_sha256,
            "sender_pixel_mse": candidate.sender_pixel_mse,
            "sender_compress_ms": candidate.sender_compress_ms,
            "sender_decode_ms": candidate.sender_decode_ms,
            "decoded_rgb_sha256": candidate.decoded_rgb_sha256,
            "y_file": y_name, "z_file": z_name,
            "stream_sha256": [sha256(stream) for stream in candidate.streams],
            "actual_packet_bytes": candidate.packet_bytes})
    save_json(meta_path, {"identity": identity, "sender_search_ms": search_ms,
                         "candidates": records})
    return candidates, search_ms, False


def write_records(out: Path, rows: list[dict]) -> None:
    save_json(out / "records.json", rows)
    if rows:
        temporary = out / "records.tmp.csv"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out / "records.csv")


def run(args, codec, assets) -> None:
    manifest_path = ROOT / args.manifest
    manifest = read_json(manifest_path)
    if "confirmation" in args.manifest and not args.confirmation:
        raise ValueError("expanded AOIs require explicit --confirmation")
    if args.confirmation and args.limit_months:
        raise ValueError("confirmation sequences must not be truncated")
    aois = [manifest["aois"][i] for i in args.area_indices]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    source_names = ["safesync/learned_codec.py", "scripts/run_learned_codec.py", "safesync/pixel_wire.py"]
    if codec.bridge is not None:
        source_names += ["safesync/entropy_bridge.py", "resources/learned_codec/entropy_server.py"]
    source_names += ["resources/learned_codec/replay_packets.py"]
    source_hashes = {name: file_sha256(ROOT / name) for name in source_names}
    spec = {"codec": "bmshj2018_hyperprior", "metric": "mse", "qualities": list(QUALITIES),
        "scales": list(SCALES), "caps": args.caps, "aoi_ids": [a["aoi_id"] for a in aois],
        "limit_months": args.limit_months, "manifest": args.manifest,
        "manifest_sha256": file_sha256(manifest_path), "source_sha256": source_hashes,
        "asset_manifest_sha256": file_sha256(RESOURCES / "asset_manifest.json"),
        "cost_contract": "actual LC08 packet <= cap; actual 60-byte ACK separately charged",
        "selection": "decoded native RGB MSE on sender current only; all fixed candidates",
        "sender_search_shared_across_caps": True, "task_labels_or_valid_masks_opened": False,
        "ground_task_inference_performed": False, "training_performed": False,
        "initialization": "shared lossless PNG image information, actual packet and ACK charged",
        "header_bytes": HEAD.size, "checksum_bytes": 16, "ack_bytes": ACK_BYTES,
        "wire_schema": WIRE_MAGIC.decode(), "weight_registry_is_common_predeployment": True,
        "runtime": codec.runtime, "weights": assets["weights"],
        "region_geometry": {a["aoi_id"]: {"initial_native_dimensions": a["months"][0]["dimensions"],
            "months": [{"month": m["month"], "dimensions": m["dimensions"]}
                       for m in a["months"][:args.limit_months or None]]} for a in aois},
        "confirmation": args.confirmation, "expected_outputs": sum(
            len(a["months"][:args.limit_months or None]) - 1 for a in aois) * len(args.caps)}
    spec_path = out / "run_spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("frozen learned-codec run specification differs; use a new folder")
    save_json(spec_path, spec)
    save_json(out / "asset_manifest.snapshot.json", assets)
    save_json(out / "public_downloads.snapshot.json", read_json(RESOURCES / "public_downloads.json"))
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in source_hashes:
        shutil.copyfile(ROOT / name, snapshot / name.replace("/", "__"))
    for name in assets.get("official_loaded_source_sha256", {}):
        shutil.copyfile(RESOURCES / "source/vendor/compressai" / name,
                        snapshot / ("official__" + name.replace("\\", "__").replace("/", "__")))
    if codec.bridge is not None:
        for meta in assets["local_entropy_bridge"]["extensions"].values():
            shutil.copyfile(RESOURCES / "linux_wheel/compressai" / meta["file"],
                            snapshot / ("official_native__" + meta["file"]))
    save_json(out / "frozen_before_pixels.json", {"status": "frozen", "spec_sha256": file_sha256(spec_path),
        "manifest_sha256": spec["manifest_sha256"], "asset_manifest_sha256": spec["asset_manifest_sha256"],
        "source_sha256": source_hashes, "runtime": codec.runtime,
        "official_loaded_source_sha256": assets.get("official_loaded_source_sha256", {}),
        "weights": assets["weights"], "no_selected_source_RGB_opened_before_this_marker": True})
    records, search_times = [], []
    for aoi in aois:
        aid = aoi["aoi_id"]
        initial, initial_meta = initial_state(aoi, ROOT / args.initial_root, out)
        aoi_start = len(records)
        states = {cap: initial for cap in args.caps}
        for step, month in enumerate(aoi["months"][1:args.limit_months or None], 1):
            current = open_sender_rgb(month)
            identity = {"source_png_sha256": month["derived_sha256"]["rgb_path"],
                "source_sha256": source_hashes, "weights": assets["weights"],
                "runtime": codec.runtime, "qualities": list(QUALITIES), "scales": list(SCALES)}
            candidates, search_ms, cache_hit = cached_candidates(codec, current,
                out / "sender_candidates" / aid / month["month"], identity)
            search_times.append(search_ms)
            for cap in args.caps:
                prior = states[cap]
                wire, meta = select(candidates, budget=cap, region=prior.region,
                    base_version=prior.version, base_digest=prior.digest(), observation=step)
                folder = out / "packets" / str(cap) / aid
                folder.mkdir(parents=True, exist_ok=True)
                packet = folder / f"{step:03d}.bin"
                packet.write_bytes(wire)
                # Receive persisted bytes, after sender choice; no current RGB is passed.
                started = time.perf_counter()
                state, ack = receive(prior, packet.read_bytes(), codec)
                codec.synchronize()
                receiver_ms = (time.perf_counter() - started) * 1000
                if len(wire) > cap or len(ack) != ACK_BYTES:
                    raise ValueError("actual learned codec packet/ACK cost contract differs")
                actual_rgb_sha = sha256(state.rgb.tobytes())
                if actual_rgb_sha != meta["sender_decoded_rgb_sha256"]:
                    raise ValueError("actual receiver image differs from sender entropy roundtrip")
                packet.with_suffix(".ack").write_bytes(ack)
                decoded_path = folder / f"{step:03d}.decoded.png"
                native_width, native_height = month["dimensions"]
                native_rgb = state.rgb[:native_height, :native_width]
                Image.fromarray(native_rgb).save(decoded_path)
                states[cap] = state
                row = {"aoi": aid, "month": month["month"], "step": step,
                    "codec": "bmshj2018_hyperprior", "method": "learned_hyperprior_q1_8", "cap": cap, **meta,
                    "down_bytes": len(wire), "up_bytes": len(ack),
                    "actual_total_bytes": len(wire) + len(ack), "packet_slack_bytes": cap - len(wire),
                    "packet_path": packet.relative_to(ROOT).as_posix(), "packet_sha256": sha256(wire),
                    "ack_path": packet.with_suffix(".ack").relative_to(ROOT).as_posix(),
                    "ack_sha256": sha256(ack), "decoded_path": decoded_path.relative_to(ROOT).as_posix(),
                    "decoded_png_sha256": file_sha256(decoded_path), "receiver_rgb_sha256": actual_rgb_sha,
                    "decoded_native_rgb_sha256": sha256(native_rgb.tobytes()),
                    "source_native_width": native_width, "source_native_height": native_height,
                    "receiver_ms": receiver_ms, "sender_search_ms_shared": search_ms,
                    "candidate_cache_hit": cache_hit, "source_png_sha256_sender_only": identity["source_png_sha256"],
                    "sender_history_RGB_bytes": 0, "sender_descriptor_bytes": 36, **initial_meta}
                records.append(row)
                if args.limit_months or step == 1 or step % 5 == 0 or step == len(aoi["months"]) - 1:
                    print(f"{WIRE_MAGIC.decode()} {len(records)}/{spec['expected_outputs']} cap={cap} {aid} {month['month']} "
                          f"q={meta['quality']} {meta['coded_width']}x{meta['coded_height']} "
                          f"{len(wire)} B PSNR={meta['sender_psnr_db']:.3f} dB", flush=True)
            write_records(out, records)
            save_json(out / "progress.json", {"status": "in_progress", "actual_outputs": len(records),
                "expected_outputs": spec["expected_outputs"], "aoi": aid, "month": month["month"]})
        aoi_rows = records[aoi_start:]
        expected_aoi = (len(aoi["months"][:args.limit_months or None]) - 1) * len(args.caps)
        if len(aoi_rows) != expected_aoi:
            raise ValueError("incomplete AOI receiver output cohort")
        receipt_path = out / "aoi_receipts" / (aid + ".json")
        save_json(receipt_path, {"status": "complete", "aoi": aid, "expected_outputs": expected_aoi,
            "spec_sha256": file_sha256(spec_path), "initial": initial_meta, "records": aoi_rows})
        save_json(out / "complete_aois" / (aid + ".json"), {"status": "complete", "aoi": aid,
            "actual_outputs": len(aoi_rows), "expected_outputs": expected_aoi,
            "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": file_sha256(receipt_path), "spec_sha256": file_sha256(spec_path)})
        print(f"AOI receiver complete {aid} {len(aoi_rows)} packets; committed receipt", flush=True)
    if len(records) != spec["expected_outputs"]:
        raise ValueError("incomplete learned codec run")
    summary = {"status": "complete", "actual_outputs": len(records),
        "aoi_count": len(aois), "updated_frames": len(search_times),
        "mean_sender_search_seconds_per_frame_all_caps": float(np.mean(search_times)) / 1000,
        "max_sender_search_seconds_per_frame_all_caps": max(search_times) / 1000,
        "mean_receiver_ms": float(np.mean([r["receiver_ms"] for r in records])),
        "all_actual_packets_fit_caps": True, "all_actual_ack_lengths": ACK_BYTES,
        "same_runtime_sender_receiver_hashes_match": True,
        "gpu_peak_reserved_bytes": codec.torch.cuda.max_memory_reserved() if codec.device.type == "cuda" else 0,
        "task_metrics": "not computed in this sender/receiver control", "per_cap": []}
    for cap in args.caps:
        rows = [r for r in records if r["cap"] == cap]
        summary["per_cap"].append({"cap": cap, "frames": len(rows),
            "down_bytes": sum(r["down_bytes"] for r in rows), "up_bytes": sum(r["up_bytes"] for r in rows),
            "mean_psnr_db": float(np.mean([r["sender_psnr_db"] for r in rows])),
            "initial_down_bytes_once_per_aoi": sum(next(r["initial_down_bytes"] for r in rows if r["aoi"] == a["aoi_id"]) for a in aois),
            "initial_up_bytes_once_per_aoi": ACK_BYTES * len(aois)})
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {"status": "complete", "actual_outputs": len(records),
        "expected_outputs": spec["expected_outputs"]})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--decode-packet", type=Path)
    parser.add_argument("--decode-output", type=Path)
    parser.add_argument("--manifest", default="data/spacenet7_pilot/development_manifest_v1.json")
    parser.add_argument("--area-indices", type=int, nargs="+", default=[2])
    parser.add_argument("--limit-months", type=int, default=3)
    parser.add_argument("--caps", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--initial-root", default="results/pixel_sequences/development_side4")
    parser.add_argument("--out", default="results/learned_codec/development_q1_8_v1")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--entropy-bridge", action="store_true", help="Windows GPU network source with local WSL official C++ entropy RPC")
    args = parser.parse_args()
    codec = HyperpriorCodec(RESOURCES, device=args.device, entropy_bridge=args.entropy_bridge)
    if args.decode_packet:
        if not args.decode_output:
            raise ValueError("--decode-output is required")
        rgb, _ = codec.decode_packet(args.decode_packet.read_bytes())
        args.decode_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.decode_output)
        print(json.dumps({"decoded_rgb_sha256": sha256(rgb.tobytes()), "shape": list(rgb.shape)}))
        return
    assets = prepare_assets(codec)
    if args.prepare_only:
        print(json.dumps(assets, indent=2), flush=True)
        return
    run(args, codec, assets)


if __name__ == "__main__":
    main()
