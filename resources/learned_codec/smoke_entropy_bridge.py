"""One 64x64 real official-network + official-rANS feasibility test."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from safesync.learned_codec import Candidate, HyperpriorCodec, pack, receive, sha256
from safesync.pixel_wire import PixelState


def main():
    out = ROOT / "results/learned_codec/bridge_smoke64_v1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    codec = HyperpriorCodec(ROOT / "resources/learned_codec", entropy_bridge=True)
    bridge_start_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    net = codec.model(1)
    prepare_ms = (time.perf_counter() - started) * 1000

    def forward_forbidden(*args, **kwargs):
        raise AssertionError("the top-level forward reconstruction must not be used")
    net.forward = forward_forbidden
    manifest = json.loads((ROOT / "data/spacenet7_pilot/development_manifest_v1.json").read_text())
    month = manifest["aois"][2]["months"][1]
    source = ROOT / month["rgb_path"]
    if sha256(source.read_bytes()) != month["derived_sha256"]["rgb_path"]:
        raise ValueError("source identity differs")
    with Image.open(source) as image:
        a = np.asarray(image.convert("RGB").crop((480, 480, 544, 544)))
    x = codec.torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).to("cuda").float().div_(255)
    codec.synchronize()
    started = time.perf_counter()
    with codec.torch.inference_mode():
        encoded = net.compress(x)
    codec.synchronize()
    compress_ms = (time.perf_counter() - started) * 1000
    streams = tuple(bytes(group[0]) for group in encoded["strings"])
    latent_shape = tuple(encoded["shape"])
    weight_sha = codec.weights[1]["sha256"]
    started = time.perf_counter()
    rgb = codec._decode(quality=1, streams=streams, latent_shape=latent_shape,
        padded_shape=(64, 64), coded_shape=(64, 64), native_shape=(64, 64), weight_sha256=weight_sha)
    codec.synchronize()
    decode_ms = (time.perf_counter() - started) * 1000
    mse = float(np.mean(((a.astype(np.float64) - rgb) / 255) ** 2))
    candidate = Candidate(1, (64, 64), (64, 64), (64, 64), latent_shape,
        weight_sha, streams, mse, compress_ms, decode_ms, sha256(rgb.tobytes()))
    region = hashlib.sha256(b"learned-bridge-64-only").digest()[:16]
    blank = PixelState(np.zeros_like(a), region)
    packet = pack(candidate, region=region, base_version=0, base_digest=blank.digest(), observation=0)
    (out / "sample.bin").write_bytes(packet)
    state, ack = receive(blank, (out / "sample.bin").read_bytes(), codec)
    if not np.array_equal(state.rgb, rgb):
        raise ValueError("persisted-packet receiver differs from entropy roundtrip")
    (out / "sample.ack").write_bytes(ack)
    Image.fromarray(state.rgb).save(out / "sample.decoded.png")
    with codec.torch.inference_mode():
        repeat = net.compress(x)
        changed = net.compress(1 - x)
    deterministic = repeat["strings"] == encoded["strings"]
    if not deterministic or changed["strings"] == encoded["strings"]:
        raise ValueError("real entropy input-identity/determinism check failed")
    changed_streams = tuple(bytes(group[0]) for group in changed["strings"])
    changed_rgb = codec._decode(quality=1, streams=changed_streams, latent_shape=tuple(changed["shape"]),
        padded_shape=(64, 64), coded_shape=(64, 64), native_shape=(64, 64), weight_sha256=weight_sha)
    if np.array_equal(changed_rgb, rgb):
        raise ValueError("distinct input produced identical decoded RGB")
    loaded = {}
    for name, module in sys.modules.items():
        path = getattr(module, "__file__", None)
        if name.startswith("compressai.") and path and Path(path).is_file():
            loaded[name] = {"source": str(Path(path).relative_to(ROOT)),
                            "sha256": sha256(Path(path).read_bytes())}
    result = {"status": "passed", "scope": "one 64x64 crop, official quality1, feasibility only",
        "runtime": codec.runtime, "weight": codec.weights[1], "bridge_identity": codec.bridge.identity,
        "bridge_stats": codec.bridge.stats, "source_png": month["rgb_path"],
        "source_png_sha256": month["derived_sha256"]["rgb_path"], "crop_xyxy": [480,480,544,544],
        "input_rgb_sha256": sha256(a.tobytes()), "decoded_rgb_sha256": sha256(rgb.tobytes()),
        "packet_sha256": sha256(packet), "packet_bytes": len(packet), "ack_bytes": len(ack),
        "y_bytes": len(streams[0]), "z_bytes": len(streams[1]), "hyperlatent_shape": latent_shape,
        "mse": mse, "psnr_db": -10*np.log10(mse), "bridge_start_ms": bridge_start_ms,
        "bridge_start_ms_scope": "entire codec constructor including Python imports and bridge startup",
        "model_prepare_ms": prepare_ms, "compress_ms_including_local_entropy_RPC": compress_ms,
        "decode_ms_including_local_entropy_RPC": decode_ms,
        "top_level_forward_forbidden": True, "identical_input_entropy_bytes_deterministic": deterministic,
        "changed_input_entropy_and_decoded_RGB_differ": True,
        "persisted_LC08_packet_receiver_RGB_matches": True,
        "model_training": net.training, "requires_grad_any": any(p.requires_grad for p in net.parameters()),
        "official_loaded_source_files": loaded,
        "gpu_peak_reserved_bytes": codec.torch.cuda.max_memory_reserved(),
        "confirmation_pixels_read": False, "task_masks_or_predictions_used": False}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ("status","packet_bytes","ack_bytes","psnr_db",
        "bridge_start_ms","model_prepare_ms","compress_ms_including_local_entropy_RPC",
        "decode_ms_including_local_entropy_RPC","gpu_peak_reserved_bytes")}, indent=2))
    codec.bridge.close()


if __name__ == "__main__":
    main()
