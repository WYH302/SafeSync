"""Frozen CompressAI scale-hyperprior control with real rANS byte streams.

LC09 is a standalone RGB image container plus the same region/version contract
as the image controls. Its receiver needs only this packet, registered public
weights and the pinned numerical runtime to reconstruct the image. No source
RGB, estimated likelihood bit rate or floating-point latent is transmitted.
"""
from __future__ import annotations

import hashlib
import math
import platform
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from .pixel_wire import ACK, UPDATED, PixelState, checksum

ARCHITECTURE = "bmshj2018-hyperprior"
QUALITIES = tuple(range(1, 9))
SCALES = (1.0, 0.75, 0.5, 0.375, 0.25)
# Fixed schema implies official ScaleHyperprior/MSE/rANS, right/bottom padding
# to 64, z shape=padding/64 and target=base+1. Quality selects exact common
# pretrained registry weights; their full SHA256 is verified out of band.
WIRE_MAGIC = b"LC09"
HEAD = struct.Struct("<4sBHHHHII16s16sII")
LEGACY_HEAD = struct.Struct("<4sBBBBHHHHHHHHIII16s16s32sII")
CHECKSUM_BYTES = 16
ACK_BYTES = ACK.size + CHECKSUM_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes())


@dataclass(frozen=True)
class Candidate:
    quality: int
    native_shape: tuple[int, int]
    coded_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    latent_shape: tuple[int, int]
    weight_sha256: str
    streams: tuple[bytes, bytes]
    sender_pixel_mse: float
    sender_compress_ms: float
    sender_decode_ms: float
    decoded_rgb_sha256: str

    @property
    def packet_bytes(self) -> int:
        return HEAD.size + sum(map(len, self.streams)) + CHECKSUM_BYTES

    def metadata(self) -> dict:
        return {
            "quality": self.quality,
            "native_height": self.native_shape[0], "native_width": self.native_shape[1],
            "coded_height": self.coded_shape[0], "coded_width": self.coded_shape[1],
            "padded_height": self.padded_shape[0], "padded_width": self.padded_shape[1],
            "latent_height": self.latent_shape[0], "latent_width": self.latent_shape[1],
            "weight_sha256": self.weight_sha256,
            "y_stream_bytes": len(self.streams[0]), "z_stream_bytes": len(self.streams[1]),
            "entropy_stream_bytes": sum(map(len, self.streams)),
            "header_bytes": HEAD.size, "checksum_bytes": CHECKSUM_BYTES,
            "wire_schema": WIRE_MAGIC.decode(),
            "bytes": self.packet_bytes, "sender_pixel_mse": self.sender_pixel_mse,
            "sender_psnr_db": -10 * math.log10(self.sender_pixel_mse) if self.sender_pixel_mse else None,
            "sender_compress_ms": self.sender_compress_ms,
            "sender_decode_ms": self.sender_decode_ms,
            "sender_decoded_rgb_sha256": self.decoded_rgb_sha256,
        }


def pack(candidate: Candidate, *, region: bytes, base_version: int,
         base_digest: bytes, observation: int) -> bytes:
    if len(region) != 16 or len(base_digest) != 16:
        raise ValueError("LC08 region/base digest must be 16 bytes")
    y, z = candidate.streams
    head = HEAD.pack(WIRE_MAGIC, candidate.quality,
                     *candidate.native_shape, *candidate.coded_shape,
                     base_version, observation, base_digest, region, len(y), len(z))
    body = head + y + z
    return body + checksum(body)


def parse(wire: bytes) -> dict:
    if len(wire) < HEAD.size + CHECKSUM_BYTES or checksum(wire[:-16]) != wire[-16:]:
        raise ValueError("LC08 checksum/truncation")
    magic = wire[:4]
    if magic == b"LC08":
        if len(wire) < LEGACY_HEAD.size + CHECKSUM_BYTES:
            raise ValueError("legacy LC08 header truncated")
        (magic, arch, metric, quality, coder, h, w, ch, cw, ph, pw, zh, zw,
         base, target, observation, base_digest, region, weights, ny, nz) = LEGACY_HEAD.unpack(wire[:LEGACY_HEAD.size])
        weight_sha = weights.hex()
        header_bytes = LEGACY_HEAD.size
    elif magic == WIRE_MAGIC:
        (magic, quality, h, w, ch, cw, base, observation, base_digest, region, ny, nz) = HEAD.unpack(wire[:HEAD.size])
        arch, metric, coder = 1, 0, 1
        ph, pw = math.ceil(ch / 64) * 64, math.ceil(cw / 64) * 64
        zh, zw = ph // 64, pw // 64
        target, weight_sha, header_bytes = base + 1, None, HEAD.size
    else:
        raise ValueError("unknown learned image schema")
    if (arch, metric, coder) != (1, 0, 1) or quality not in QUALITIES:
        raise ValueError("unknown learned model/metric/quality/coder")
    if not (0 < ch <= h and 0 < cw <= w and h % 64 == w % 64 == 0):
        raise ValueError("LC08 native/coded dimensions differ")
    if (ph, pw) != (math.ceil(ch / 64) * 64, math.ceil(cw / 64) * 64):
        raise ValueError("LC08 padding must be zero on right and bottom to multiple 64")
    if (zh, zw) != (ph // 64, pw // 64) or not ny or not nz:
        raise ValueError("LC08 entropy shape/streams differ")
    if target != base + 1 or len(wire) != header_bytes + ny + nz + CHECKSUM_BYTES:
        raise ValueError("LC08 version/stream layout differs")
    offset = header_bytes
    return {"quality": quality, "native_shape": (h, w), "coded_shape": (ch, cw),
            "padded_shape": (ph, pw), "latent_shape": (zh, zw),
            "base_version": base, "target_version": target, "observation": observation,
            "base_digest": base_digest, "region": region, "weight_sha256": weight_sha,
            "wire_schema": magic.decode(), "header_bytes": header_bytes,
            "model_registry": f"bmshj2018-hyperprior-mse-v1/quality{quality}",
            "streams": (wire[offset:offset + ny], wire[offset + ny:-CHECKSUM_BYTES]),
            "y_stream_bytes": ny, "z_stream_bytes": nz}


class HyperpriorCodec:
    """Inference-only official model; all candidates use actual compress/decompress."""

    def __init__(self, resources: Path, *, device: str = "cuda", threads: int = 2,
                 gpu_memory_fraction: float = 0.075, entropy_bridge: bool = False):
        self.bridge, self.scale_hyperprior = None, None
        if entropy_bridge:
            from .entropy_bridge import install_local_bridge
            self.bridge, self.scale_hyperprior = install_local_bridge(Path(resources))
        import compressai
        import torch
        import scipy

        if compressai.__version__ != "1.2.8" or torch.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError("this control is pinned to CompressAI 1.2.8 and torch 2.8.0")
        self.torch, self.compressai = torch, compressai
        self.resources = Path(resources)
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        self.resources.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(threads)
        # A runner constructs one codec. Inter-op count cannot change after use.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("the fixed GPU codec requires CUDA")
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, self.device)
        compressai.set_entropy_coder("ans")
        self.models, self.weights = {}, {}
        self.runtime = {
            "architecture": ARCHITECTURE, "metric": "mse", "entropy_coder": "ans",
            "python": platform.python_version(), "numpy": np.__version__, "Pillow": pillow_version,
            "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "compressai": compressai.__version__,
            "device": str(self.device), "dtype": "float32", "tf32": False,
            "cudnn_deterministic": True, "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "gpu_memory_fraction": gpu_memory_fraction,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "cross_device_decode_supported": False,
            "implementation": "Windows official network source + local WSL official C++ entropy RPC" if entropy_bridge else "standard CompressAI package",
        }

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def model(self, quality: int):
        if quality not in QUALITIES:
            raise ValueError("only the fixed complete public quality grid 1..8 is permitted")
        if quality not in self.models:
            from compressai.zoo.pretrained import load_pretrained
            if self.bridge is not None:
                import json
                assets = json.loads((self.resources / "public_downloads.json").read_text())
                registered = next((item for item in assets["weights"] if item["quality"] == quality), None)
                if registered is None:
                    raise RuntimeError(f"official quality {quality} is not yet downloaded/registered; complete the fixed 1..8 asset set before running")
                url = registered["url"]
            else:
                from compressai.zoo import bmshj2018_hyperprior
                from compressai.zoo.image import model_urls
                url = model_urls[ARCHITECTURE]["mse"][quality]
            path = self.resources / "weights" / url.rsplit("/", 1)[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                self.torch.hub.download_url_to_file(url, str(path), progress=True)
            digest = file_sha256(path)
            if self.bridge is not None and digest != registered["sha256"]:
                raise ValueError("predeployed common model full SHA256 differs")
            # Official model URLs embed an eight-character SHA256 prefix.
            expected_prefix = path.name.split("-")[-1].split(".")[0]
            if not digest.startswith(expected_prefix):
                raise ValueError("official pretrained weight SHA256 prefix differs")
            state = self.torch.load(path, map_location="cpu", weights_only=True)
            state = load_pretrained(state)
            if self.bridge is not None:
                net = self.scale_hyperprior.from_state_dict(state)
            else:
                net = bmshj2018_hyperprior(quality=quality, metric="mse", pretrained=False)
                net.load_state_dict(state)
            net.eval().requires_grad_(False)
            net.update(force=True)
            cdf_hash = hashlib.sha256()
            for name, tensor in net.named_buffers():
                if any(key in name for key in ("_quantized_cdf", "_cdf_length", "_offset")):
                    cdf_hash.update(name.encode())
                    cdf_hash.update(tensor.cpu().numpy().tobytes())
            self.weights[quality] = {"quality": quality, "url": url,
                "file": path.name, "sha256": digest, "bytes": path.stat().st_size,
                "entropy_cdf_sha256": cdf_hash.hexdigest(), "training": False,
                "parameter_count": sum(p.numel() for p in net.parameters()),
                "requires_grad_any": any(p.requires_grad for p in net.parameters())}
            self.models[quality] = net.to(self.device)
        return self.models[quality]

    def prepare(self) -> dict:
        for quality in QUALITIES:
            self.model(quality)
        return {"runtime": self.runtime, "weights": list(self.weights.values())}

    def _decode(self, *, quality: int, streams: tuple[bytes, bytes], latent_shape,
                padded_shape, coded_shape, native_shape, weight_sha256: str) -> np.ndarray:
        net = self.model(quality)
        if self.weights[quality]["sha256"] != weight_sha256:
            raise ValueError("LC08 registered pretrained weight hash differs")
        with self.torch.inference_mode():
            decoded = net.decompress([[streams[0]], [streams[1]]], latent_shape)["x_hat"]
            if tuple(decoded.shape) != (1, 3, *padded_shape):
                raise ValueError("LC08 decoded entropy dimensions differ")
            ch, cw = coded_shape
            rgb = (decoded[0, :, :ch, :cw].clamp(0, 1).mul(255).round()
                   .to(self.torch.uint8).permute(1, 2, 0).cpu().numpy())
        h, w = native_shape
        return np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC))

    def decode_packet(self, wire: bytes) -> tuple[np.ndarray, dict]:
        """Standalone reconstruction; accepts no prior image or sender object."""
        parsed = parse(wire)
        if parsed["weight_sha256"] is None:
            self.model(parsed["quality"])
            parsed["weight_sha256"] = self.weights[parsed["quality"]]["sha256"]
        rgb = self._decode(**{key: parsed[key] for key in (
            "quality", "streams", "latent_shape", "padded_shape", "coded_shape",
            "native_shape", "weight_sha256")})
        return rgb, parsed

    def encode_candidates(self, current: np.ndarray) -> list[Candidate]:
        a = np.asarray(current)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("expected uint8 RGB")
        h, w = a.shape[:2]
        if h % 64 or w % 64:
            raise ValueError("shared native image must be padded to multiple 64")
        candidates = []
        for quality in QUALITIES:
            net = self.model(quality)
            for scale in SCALES:
                ch, cw = max(1, round(h * scale)), max(1, round(w * scale))
                ph, pw = math.ceil(ch / 64) * 64, math.ceil(cw / 64) * 64
                small = np.asarray(Image.fromarray(a).resize((cw, ch), Image.Resampling.LANCZOS))
                tensor = self.torch.from_numpy(small.copy()).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(self.device, self.torch.float32).div_(255)
                tensor = self.torch.nn.functional.pad(tensor, (0, pw - cw, 0, ph - ch), value=0)
                self.synchronize()
                started = time.perf_counter()
                with self.torch.inference_mode():
                    encoded = net.compress(tensor)
                self.synchronize()
                compress_ms = (time.perf_counter() - started) * 1000
                strings = encoded["strings"]
                if len(strings) != 2 or any(len(group) != 1 for group in strings):
                    raise ValueError("LC08 requires exactly one y and one z byte stream")
                streams = tuple(bytes(group[0]) for group in strings)
                shape = tuple(map(int, encoded["shape"]))
                if shape != (ph // 64, pw // 64):
                    raise ValueError("official hyperprior latent shape differs")
                started = time.perf_counter()
                rgb = self._decode(quality=quality, streams=streams, latent_shape=shape,
                    padded_shape=(ph, pw), coded_shape=(ch, cw), native_shape=(h, w),
                    weight_sha256=self.weights[quality]["sha256"])
                self.synchronize()
                decode_ms = (time.perf_counter() - started) * 1000
                mse = float(np.mean(((a.astype(np.float64) - rgb) / 255.0) ** 2))
                candidates.append(Candidate(quality, (h, w), (ch, cw), (ph, pw), shape,
                    self.weights[quality]["sha256"], streams, mse, compress_ms, decode_ms,
                    sha256(rgb.tobytes())))
        return candidates


def select(candidates: list[Candidate], *, budget: int, region: bytes,
           base_version: int, base_digest: bytes, observation: int) -> tuple[bytes, dict]:
    fitting = [candidate for candidate in candidates if candidate.packet_bytes <= budget]
    if not fitting:
        raise ValueError("no candidate from the fixed quality/scale grid fits this actual packet budget")
    # Stable grid order breaks exact MSE ties, never labels or downstream tasks.
    best = min(fitting, key=lambda candidate: candidate.sender_pixel_mse)
    wire = pack(best, region=region, base_version=base_version,
                base_digest=base_digest, observation=observation)
    if len(wire) != best.packet_bytes or len(wire) > budget:
        raise ValueError("LC08 serialized size differs from budget selection")
    return wire, {**best.metadata(), "codec_trials": len(candidates),
        "fitting_trials": len(fitting), "input": "current sender RGB only; no masks, GT or task outputs",
        "selection": "lowest decoded native RGB pixel MSE among all fitting fixed candidates"}


def receive(prior: PixelState, wire: bytes, codec: HyperpriorCodec) -> tuple[PixelState, bytes]:
    parsed = parse(wire)
    if (parsed["native_shape"] != prior.rgb.shape[:2] or parsed["region"] != prior.region
            or parsed["base_version"] != prior.version or parsed["base_digest"] != prior.digest()
            or parsed["observation"] <= prior.observed_at):
        raise ValueError("LC08 wrong region/base state or stale observation")
    rgb, _ = codec.decode_packet(wire)
    status = np.full_like(prior.statuses, UPDATED)
    state = PixelState(rgb, prior.region, prior.block, parsed["target_version"],
                       parsed["observation"], status)
    ack_magic = b"LA09" if parsed["wire_schema"] == "LC09" else b"LA08"
    ack = ACK.pack(ack_magic, state.version, state.observed_at, checksum(wire), state.digest())
    return state, ack + checksum(ack)
