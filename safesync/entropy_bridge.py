"""Local Windows-Torch / WSL-official-rANS bridge; no neural operator changes."""
from __future__ import annotations

from array import array
import atexit
import hashlib
import importlib
import json
from pathlib import Path, PureWindowsPath
import struct
import subprocess
import sys
import threading
import time
import types

FRAME = struct.Struct("<4sBI")


class EntropyBridge:
    def __init__(self, resources: Path):
        self.resources = Path(resources).resolve()
        script = PureWindowsPath(self.resources / "entropy_server.py")
        wsl_script = "/mnt/" + script.drive[0].lower() + "/" + "/".join(script.parts[1:])
        self.stderr = (self.resources / "entropy_server.stderr.log").open("ab")
        self.process = subprocess.Popen(["wsl.exe", "-d", "Astra-Ubuntu-22.04", "--", "python3", wsl_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
        self.table_keys, self.lock = set(), threading.Lock()
        self.stats = {"calls": {}, "request_bytes": 0, "response_bytes": 0, "wall_ms": 0.0,
                      "table_uploads": 0}
        self.identity = json.loads(self.request(0, b""))
        atexit.register(self.close)

    def _read_exact(self, size):
        result = bytearray()
        while len(result) < size:
            part = self.process.stdout.read(size - len(result))
            if not part:
                raise RuntimeError("local entropy host closed; see entropy_server.stderr.log")
            result.extend(part)
        return bytes(result)

    def request(self, op, body):
        started = time.perf_counter()
        with self.lock:
            timeout = threading.Timer(30, self.process.kill)
            timeout.start()
            try:
                frame = memoryview(FRAME.pack(b"EB01", op, len(body)) + body)
                while frame:
                    written = self.process.stdin.write(frame)
                    if not written:
                        raise RuntimeError("local entropy host input closed")
                    frame = frame[written:]
                self.process.stdin.flush()
                magic, status, size = FRAME.unpack(self._read_exact(FRAME.size))
                result = self._read_exact(size)
            finally:
                timeout.cancel()
        if magic != b"EB01" or status:
            raise RuntimeError("local entropy host: " + result.decode(errors="replace"))
        self.stats["calls"][str(op)] = self.stats["calls"].get(str(op), 0) + 1
        self.stats["request_bytes"] += FRAME.size + len(body)
        self.stats["response_bytes"] += FRAME.size + len(result)
        self.stats["wall_ms"] += (time.perf_counter() - started) * 1000
        return result

    def table(self, cdfs, lengths, offsets):
        rows, columns = len(cdfs), len(cdfs[0])
        data = (struct.pack("<II", rows, columns) + array("i", (x for row in cdfs for x in row)).tobytes()
                + array("i", lengths).tobytes() + array("i", offsets).tobytes())
        key = hashlib.sha256(data).digest()
        if key not in self.table_keys:
            if self.request(1, key + data) != key:
                raise RuntimeError("local entropy CDF cache acknowledgement differs")
            self.table_keys.add(key)
            self.stats["table_uploads"] += 1
        return key

    def encode(self, symbols, indexes, cdfs, lengths, offsets):
        key = self.table(cdfs, lengths, offsets)
        if len(symbols) != len(indexes):
            raise ValueError("symbol/index count differs")
        body = key + struct.pack("<I", len(symbols)) + array("i", symbols).tobytes() + array("i", indexes).tobytes()
        return self.request(2, body)

    def decode(self, stream, indexes, cdfs, lengths, offsets):
        key = self.table(cdfs, lengths, offsets)
        body = key + struct.pack("<II", len(indexes), len(stream)) + array("i", indexes).tobytes() + bytes(stream)
        values = array("i")
        values.frombytes(self.request(3, body))
        if len(values) != len(indexes):
            raise ValueError("official decoded symbol count differs")
        return values.tolist()

    def pmf(self, probabilities, precision):
        body = struct.pack("<II", precision, len(probabilities)) + array("d", probabilities).tobytes()
        values = array("i")
        values.frombytes(self.request(4, body))
        return values.tolist()

    def close(self):
        if self.process.poll() is None:
            try:
                self.request(255, b"")
                self.process.wait(timeout=5)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                self.process.kill()
        if not self.stderr.closed:
            self.stderr.close()


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    if "." in name:
        parent, leaf = name.rsplit(".", 1)
        setattr(sys.modules[parent], leaf, module)
    return module


def install_local_bridge(resources: Path):
    resources = Path(resources).resolve()
    root = resources.parents[1]
    for dependency in (root / "resources/pretrained_ground/.deps", root / "results/layered_tasks/.deps"):
        if dependency.exists():
            sys.path.insert(0, str(dependency))
    source = resources / "source/vendor/compressai"
    if "compressai" in sys.modules:
        raise RuntimeError("start the isolated bridge in a clean Python process")
    bridge = EntropyBridge(resources)
    package = _namespace("compressai", source)
    package.__version__ = "1.2.8"
    package.available_entropy_coders = lambda: ["ans"]
    package.get_entropy_coder = lambda: "ans"

    def set_coder(name):
        if name != "ans":
            raise ValueError("this local bridge supports only official rANS")
    package.set_entropy_coder = set_coder
    ans = types.ModuleType("compressai.ans")

    class RansEncoder:
        encode_with_indexes = lambda self, *args, **kwargs: bridge.encode(*args, **kwargs)

    class RansDecoder:
        decode_with_indexes = lambda self, *args, **kwargs: bridge.decode(*args, **kwargs)

    class BufferedRansEncoder:
        def __init__(self):
            raise NotImplementedError("this scoped bridge supports ScaleHyperprior's stateless rANS only")

    ans.RansEncoder, ans.RansDecoder, ans.BufferedRansEncoder = RansEncoder, RansDecoder, BufferedRansEncoder
    package.ans = sys.modules["compressai.ans"] = ans
    cxx = types.ModuleType("compressai._CXX")
    cxx.pmf_to_quantized_cdf = bridge.pmf
    package._CXX = sys.modules["compressai._CXX"] = cxx
    registry = _namespace("compressai.registry", source / "registry")
    official_registry = importlib.import_module("compressai.registry.torch")
    registry.register_model = official_registry.register_model
    entropy = _namespace("compressai.entropy_models", source / "entropy_models")
    official_entropy = importlib.import_module("compressai.entropy_models.entropy_models")
    for name in ("EntropyBottleneck", "EntropyModel", "GaussianConditional", "GaussianMixtureConditional"):
        setattr(entropy, name, getattr(official_entropy, name))
    latent = _namespace("compressai.latent_codecs", source / "latent_codecs")
    latent.LatentCodec = importlib.import_module("compressai.latent_codecs.base").LatentCodec
    _namespace("compressai.models", source / "models")
    google = importlib.import_module("compressai.models.google")
    _namespace("compressai.zoo", source / "zoo")
    return bridge, google.ScaleHyperprior
