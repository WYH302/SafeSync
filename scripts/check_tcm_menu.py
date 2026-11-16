"""One already-inspected development image validates the fixed TCM menu."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from safesync.tcm_codec import TCMCodec
from scripts.run_locked_second_test import write_json, sha


def main():
    pair = next(p for p in json.loads((ROOT / "data/second_task_context_test_v1/manifest.json").read_text("utf-8"))["pairs"] if p["id"] == "05306")
    item = pair["files"]["after"]
    if sha(ROOT / item["path"]) != item["sha256"]:
        raise ValueError("Development RGB changed")
    current = np.asarray(Image.open(ROOT / item["path"]).convert("RGB"))
    codec = TCMCodec(ROOT)
    started = time.perf_counter()
    codec.prepare()
    candidates = codec.encode_candidates(current)
    if len(candidates) != 35:
        raise AssertionError("Fixed menu is incomplete")
    checks = []
    for budget in (3800, 7900, 16000, 32000):
        packet, metadata = codec.select(candidates, budget=budget, region=b"r"*16, base_version=0, base_digest=b"d"*16, observation=1)
        rgb, _ = codec.decode_packet(packet)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != metadata["sender_decoded_rgb_sha256"]:
            raise AssertionError("Independent selected packet reconstruction differs")
        checks.append(dict(budget=budget, bytes=len(packet), slot=metadata["model_slot"]))
    receipt = dict(status="passed", pair_id=pair["id"], candidates=len(candidates), checkpoints=7,
        fixed_scales=5, actual_packet_replay_checks=checks, seconds=time.perf_counter()-started,
        source_sha256=sha(Path(__file__)), registry_sha256=sha(ROOT / "resources/modern_codec/weights/registry.json"))
    write_json(ROOT / "resources/modern_codec/menu_verification.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
