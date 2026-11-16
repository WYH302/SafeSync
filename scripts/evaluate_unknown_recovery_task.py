"""Known-class semantic probe on actual receiver RGB from the repair study.

The frozen DINO encoder and explicit known-only calibration prototypes are shared
by every method. This exploratory probe is separate from RGB evidence coverage.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
from safesync.data import read_pair
from safesync.features import FrozenEncoder
from scripts.run_second_feature_mix import load_frozen

RUN = ROOT / "results/unknown_recovery/codec_v1"
OLD = ROOT / "results/second_pixels/fixed_v1"
OUT = ROOT / "results/unknown_recovery/semantic_probe_v1"
CONFIG = dict(seed=20261104, backbone_input=448, pool_patch_grid=2, grid=16, feature_dim=384)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(matrix, heldout):
    chosen = [i for i in range(6) if i != heldout - 1]
    tp = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = [2 * tp[i] / denom[i] for i in chosen if denom[i] > 0]
    return dict(known_macro_f1=float(np.mean(f1)) if f1 else None,
                known_micro_accuracy=float(tp.sum() / matrix.sum()) if matrix.sum() else None,
                valid_known_pixels=float(matrix.sum()), evaluated_classes=len(f1))


def main():
    if not (RUN / "complete.json").exists():
        raise ValueError("Wait for the complete repair matrix")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"] if p["split"] == "validation"}
    observations = [r for r in csv.DictReader((RUN / "records.csv").open(encoding="utf-8")) if r["router_uses_gt"] == "False"]
    observations += [r for r in csv.DictReader((OLD / "records.csv").open(encoding="utf-8")) if r["method"] in ("feature_change", "full_jpeg", "full_jpeg2000")]
    spec = dict(source_sha256=sha(Path(__file__)), run_complete_sha256=sha(RUN / "complete.json"),
                original_records_sha256=sha(OLD / "records.csv"), manifest_sha256=sha(manifest_path),
                encoder_config=CONFIG, feature_source_sha256=sha(ROOT / "safesync/features.py"),
                prototypes={str(h): sha(ROOT / f"results/unknown_scaled/known_{h}.npz") for h in (5, 3, 4)},
                classifier="cosine nearest normalized known-class calibration mean; hard token labels expanded to native 32-pixel cells",
                mask="both temporal labels explicit; exclude heldout in either time; classes1..6 evaluated including unsupported class6",
                expected_records=len(observations) + 64 * 3 * 2,
                scope="Appended exploratory known-class task probe on development data; no neural fitting or unknown-label calibration")
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed semantic probe")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", "utf-8")
    wire, helper = load_frozen(json.loads((OLD / "run_spec.json").read_text("utf-8")))
    encoder = FrozenEncoder(ROOT, CONFIG)
    prototypes = {}
    for heldout in (5, 3, 4):
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as stored:
            mean, labels = stored["means"], stored["labels"]
        prototypes[heldout] = (mean / np.linalg.norm(mean, axis=1, keepdims=True), labels)
    rows = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8"))) if (OUT / "records.csv").exists() else []
    existing = {(r["pair_id"], int(r["heldout"]), r["method"], int(r["budget"]), int(r["seed"])) for r in rows}
    for index, (pair_id, pair) in enumerate(pairs.items()):
        before_pil, current_pil, old, current, changed = read_pair(ROOT, pair)
        initial = (OLD / "pairs" / pair_id / "initial.bin").read_bytes()
        fields = wire.HEADER.unpack(initial[:wire.HEADER.size])
        blank = wire.PixelState(np.zeros_like(np.asarray(before_pil)), fields[11], fields[3])
        base = wire.TransactionReceiver(blank)
        base.receive(initial)
        items = [r for r in observations if r["pair_id"] == pair_id]
        items += [dict(pair_id=pair_id, heldout=h, method=method, budget=0, seed=0)
                  for h in (5, 3, 4) for method in ("full_current", "old_only")]
        masks = {h: helper["evaluation_masks"](old, current, changed, h)[0] for h in (5, 3, 4)}
        for item in items:
            heldout, budget, seed = int(item["heldout"]), int(item["budget"]), int(item["seed"])
            key = pair_id, heldout, item["method"], budget, seed
            if key in existing:
                continue
            if item["method"] == "full_current":
                rgb = np.asarray(current_pil)
                source_hash = pair["files"]["after"]["sha256"]
            elif item["method"] == "old_only":
                rgb = base.state.rgb
                source_hash = pair["files"]["before"]["sha256"]
            else:
                packet = (ROOT / item["packet_path"]).read_bytes()
                if hashlib.sha256(packet).hexdigest() != item["packet_sha256"]:
                    raise ValueError("Changed saved current transaction")
                receiver = helper["fresh_receiver"](base.state)
                receiver.receive(packet)
                rgb = receiver.state.rgb
                source_hash = item["packet_sha256"]
                if hashlib.sha256(rgb.tobytes()).hexdigest() != item["receiver_sha256"]:
                    raise AssertionError("Probe input differs from actual receiver")
            feature, _ = encoder.cached(Image.fromarray(rgb))
            mean, classes = prototypes[heldout]
            predicted = classes[np.argmax(feature @ mean.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
            mask = masks[heldout]
            matrix = np.bincount(((current[mask].astype(int) - 1) * 6 + predicted[mask].astype(int) - 1), minlength=36).reshape(6, 6)
            rows.append(dict(pair_id=pair_id, heldout=heldout, method=item["method"], budget=budget, seed=seed,
                             input_sha256=source_hash, receiver_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                             confusion=json.dumps(matrix.tolist(), separators=(",", ":")), **metrics(matrix, heldout)))
            existing.add(key)
        helper["save_csv"](OUT / "records.csv", rows)
        helper["save_json"](OUT / "progress.json", dict(status="in_progress", completed_pairs=index + 1, expected_pairs=64,
                            records=len(rows), expected_records=spec["expected_records"]))
        print(f"receiver known-task {index+1}/64 pairs, {len(rows)} scored inputs", flush=True)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(int(r["heldout"]), int(r["budget"]), r["method"], r["pair_id"])].append(np.array(json.loads(r["confusion"]), dtype=np.float64))
    pooled = defaultdict(lambda: np.zeros((6, 6), np.float64))
    for (heldout, budget, method, _), matrices in grouped.items():
        pooled[(heldout, budget, method)] += np.mean(matrices, axis=0)
    summary = [dict(heldout=h, budget=b, method=m, **metrics(matrix, h), confusion=json.dumps(matrix.tolist()))
               for (h, b, m), matrix in sorted(pooled.items())]
    helper["save_csv"](OUT / "summary.csv", summary)
    if len(rows) != spec["expected_records"]:
        raise AssertionError("Incomplete known-task probe")
    helper["save_json"](OUT / "complete.json", dict(status="complete", records=len(rows), expected=spec["expected_records"],
                        source_sha256=sha(Path(__file__)), spec_sha256=sha(spec_path), records_sha256=sha(OUT / "records.csv"),
                        summary_sha256=sha(OUT / "summary.csv"), encoder=encoder.metadata))


if __name__ == "__main__":
    main()
