"""Cache the existing frozen DINOv2 encoder on all retained real SN7 months."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safesync.features import FrozenEncoder

CONFIG = dict(seed=20261104, backbone_input=448, pool_patch_grid=2, grid=16, feature_dim=384)
MANIFESTS = ("data/spacenet7_pilot/development_manifest_v1.json",
             "data/spacenet7_pilot/confirmation_manifest_v1.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", "utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "cache/sn7_frozen_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    areas = []
    for manifest in MANIFESTS:
        data = json.loads((ROOT / manifest).read_text("utf-8"))
        for index, area in enumerate(data["aois"]):
            areas.append(dict(area, source_manifest=manifest, area_index=index))
    spec = dict(config=CONFIG, source_sha256=sha(Path(__file__)),
                feature_source_sha256=sha(ROOT / "safesync/features.py"),
                manifests={name: sha(ROOT / name) for name in MANIFESTS},
                expected_areas=len(areas), expected_months=sum(len(a["months"]) for a in areas),
                scope="real monthly observations, resized 448 pixels; protocol feature distortion only",
                label_inputs_to_encoder=False)
    spec_path = args.out / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed frozen feature specification")
    write_json(spec_path, spec)
    encoder = None
    receipts = []
    for area in areas:
        path = args.out / (area["aoi_id"] + ".npz")
        receipt_path = path.with_suffix(".json")
        source = [{"path": m["rgb_path"], "sha256": sha(ROOT / m["rgb_path"]), "month": m["month"]}
                  for m in area["months"]]
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["source_images"] != source or receipt["sha256"] != sha(path):
                raise ValueError("Changed feature cache or image identity")
        else:
            if encoder is None:
                encoder = FrozenEncoder(ROOT, CONFIG)
            features, timings = [], []
            for item in source:
                with Image.open(ROOT / item["path"]) as image:
                    feature, timing = encoder.extract(image)
                features.append(feature.astype(np.float16))
                timings.append(timing)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=np.stack(features),
                                months=np.array([m["month"] for m in source]))
            temporary.replace(path)
            receipt = dict(aoi=area["aoi_id"], source_manifest=area["source_manifest"],
                           area_index=area["area_index"], source_images=source,
                           path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                           shape=[len(source), 256, 384], encoder=encoder.metadata, timings=timings)
            write_json(receipt_path, receipt)
        receipts.append(receipt)
        write_json(args.out / "progress.json", dict(status="in_progress", completed_areas=len(receipts),
                   expected_areas=len(areas), completed_months=sum(len(r["source_images"]) for r in receipts)))
        print(f"frozen real SN7 features {len(receipts)}/{len(areas)} areas, {len(source)} months", flush=True)
    write_json(args.out / "complete.json", dict(status="complete", spec_sha256=sha(spec_path),
               completed_utc=datetime.now(timezone.utc).isoformat(), areas=receipts,
               total_months=sum(len(r["source_images"]) for r in receipts)))


if __name__ == "__main__":
    main()
