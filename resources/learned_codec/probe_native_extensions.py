"""WSL stdlib-only dependency/link and real ANS roundtrip probe."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
native = root / "linux_wheel/compressai"
modules, info = {}, {}
for name in ("ans", "_CXX"):
    path = next(native.glob(name + ".*.so"))
    links = subprocess.check_output(["ldd", str(path)], text=True)
    if "libtorch" in links or "not found" in links:
        raise RuntimeError(f"native extension dependency unavailable: {links}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modules[name] = module
    info[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "ldd": links, "loaded_without_torch": True}
cdf = modules["_CXX"].pmf_to_quantized_cdf([0.5, 0.5], 16)
symbols, indexes = [0, 1, 0, 0, 1, 1, 0, 1], [0] * 8
stream = modules["ans"].RansEncoder().encode_with_indexes(symbols, indexes, [cdf], [len(cdf)], [0])
decoded = modules["ans"].RansDecoder().decode_with_indexes(stream, indexes, [cdf], [len(cdf)], [0])
if decoded != symbols:
    raise RuntimeError("official native ANS roundtrip differs")
result = {"status": "passed", "python": sys.version, "extensions": info,
          "torch_imported": "torch" in sys.modules, "cdf": cdf,
          "actual_ans_stream_bytes": len(stream), "symbols_equal": decoded == symbols}
(root / "native_extensions_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
