"""WSL stdlib-only RPC host for unmodified official CompressAI C++ extensions.

Only local stdin/stdout are used. No sockets, Torch, arrays-as-JSON or shell
arguments are used for symbols/indexes/CDF tables.
"""
from array import array
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import sys

FRAME = struct.Struct("<4sBI")
U32 = struct.Struct("<I")
ROOT = Path(__file__).resolve().parent


def load(name):
    path = next((ROOT / "linux_wheel/compressai").glob(name + ".*.so"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANS, CXX = load("ans"), load("_CXX")
TABLES = {}


def ints(data):
    values = array("i")
    values.frombytes(data)
    return values.tolist()


def read_exact(size):
    parts = bytearray()
    while len(parts) < size:
        block = sys.stdin.buffer.read(size - len(parts))
        if not block:
            raise EOFError
        parts.extend(block)
    return bytes(parts)


def operation(op, body):
    if op == 0:
        return json.dumps({"extension_host": "WSL Python stdlib", "torch_imported": "torch" in sys.modules,
                           "coder": "official CompressAI 1.2.8 C++ rANS"}).encode()
    if op == 1:
        key, data = body[:32], body[32:]
        if hashlib.sha256(data).digest() != key:
            raise ValueError("CDF cache key differs")
        rows, columns = struct.unpack_from("<II", data)
        count = rows * columns
        flat = ints(data[8:8 + count * 4])
        cdfs = [flat[k * columns:(k + 1) * columns] for k in range(rows)]
        lengths = ints(data[8 + count * 4:8 + (count + rows) * 4])
        offsets = ints(data[8 + (count + rows) * 4:])
        if len(lengths) != rows or len(offsets) != rows:
            raise ValueError("CDF table dimensions differ")
        TABLES[key] = (cdfs, lengths, offsets)
        return key
    if op == 2:
        key, count = body[:32], U32.unpack_from(body, 32)[0]
        if len(body) != 36 + 8 * count:
            raise ValueError("encoder symbol/index byte layout differs")
        symbols = ints(body[36:36 + 4 * count])
        indexes = ints(body[36 + 4 * count:])
        return ANS.RansEncoder().encode_with_indexes(symbols, indexes, *TABLES[key])
    if op == 3:
        key = body[:32]
        count, stream_bytes = struct.unpack_from("<II", body, 32)
        if len(body) != 40 + 4 * count + stream_bytes:
            raise ValueError("decoder stream/index byte layout differs")
        indexes = ints(body[40:40 + 4 * count])
        stream = body[40 + 4 * count:]
        values = ANS.RansDecoder().decode_with_indexes(stream, indexes, *TABLES[key])
        return array("i", values).tobytes()
    if op == 4:
        precision, count = struct.unpack_from("<II", body)
        values = array("d")
        values.frombytes(body[8:])
        if len(values) != count:
            raise ValueError("PMF byte layout differs")
        return array("i", CXX.pmf_to_quantized_cdf(values.tolist(), precision)).tobytes()
    if op == 255:
        return b""
    raise ValueError("unknown entropy RPC operation")


while True:
    try:
        magic, op, size = FRAME.unpack(read_exact(FRAME.size))
    except EOFError:
        break
    if magic != b"EB01" or size > 128 * 1024 * 1024:
        raise ValueError("bad local entropy RPC frame")
    body = read_exact(size)
    try:
        result, status = operation(op, body), 0
    except Exception as error:
        result, status = (type(error).__name__ + ": " + str(error)).encode(), 1
    sys.stdout.buffer.write(FRAME.pack(b"EB01", status, len(result)) + result)
    sys.stdout.buffer.flush()
    if op == 255:
        break
