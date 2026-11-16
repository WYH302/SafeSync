"""Lock and execute one fixed SECOND test experiment with actual receiver tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
import torch

from safesync.data import block_fraction, read_pair
from safesync.features import FrozenEncoder
from safesync.scoring import change_score
from safesync.full_image_codecs import fit as full_fit, receive as full_receive
from scripts.run_second_feature_mix import load_frozen
from scripts.run_second_pixel import initialize
from scripts.evaluate_unknown_recovery_task import metrics as task_metrics

OUT = ROOT / "results/unknown_recovery/locked_test_v1"
LOCK = ROOT / "results/unknown_recovery/confirmation_lock_v1.json"
DATA = ROOT / "data/second_locked_test_v1"
HELDOUTS = (5, 3, 4)
BUDGETS = (4096, 8192, 16384, 32768)
CONFIG = dict(seed=20261104, backbone_input=448, pool_patch_grid=2, grid=16, feature_dim=384)
METHODS = ("feature_change", "emergent_mix", "feature_random", "maha_mix", "full_jpeg420", "full_jpeg2000", "full_webp")
SOURCES = ("scripts/run_locked_second_test.py", "scripts/fetch_locked_second_test.py", "scripts/run_second_pixel.py",
           "scripts/run_second_feature_mix.py", "scripts/evaluate_unknown_recovery_task.py", "safesync/features.py",
           "safesync/data.py", "safesync/scoring.py", "safesync/pixel_wire.py", "safesync/full_image_codecs.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", "utf-8")
    temporary.replace(path)


def lock_method():
    reservation = ROOT / "data/second_scaled/reserved_official_test.json"
    reserved = json.loads(reservation.read_text("utf-8"))["reserved_pairs"]
    train_manifest = ROOT / "data/second_scaled/manifest.json"
    manifest = json.loads(train_manifest.read_text("utf-8"))
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    if ids != [p["id"] for p in manifest["pairs"]] or set(ids) & {p["id"] for p in reserved}:
        raise ValueError("Calibration IDs or held-out reservation overlap")
    gallery_folder = OUT / "galleries"
    gallery_folder.mkdir(parents=True, exist_ok=True)
    gallery_hashes = {}
    for heldout in HELDOUTS:
        path = gallery_folder / f"known_{heldout}.npz"
        if not path.exists():
            values = []
            for pair, f in zip(manifest["pairs"], features):
                if pair["split"] != "calibration":
                    continue
                _, _, old, current, _ = read_pair(ROOT, pair)
                excluded = block_fraction((old == heldout) | (current == heldout), 16) > 0
                for temporal, semantic in enumerate((old, current)):
                    for label in range(1, 7):
                        if label != heldout:
                            keep = (block_fraction(semantic == label, 16) >= .75) & ~excluded
                            values.append(f[temporal][keep])
            bank = np.concatenate(values)
            bank /= np.linalg.norm(bank, axis=1, keepdims=True)
            np.savez_compressed(path, features=bank)
        gallery_hashes[str(heldout)] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    proposed = dict(schema="SECOND-independent-fixed-v1", methods=METHODS, heldouts=HELDOUTS, budgets=BUDGETS,
        k=10, mixing_weights=[.75, .25], physical_gate_psnr=25, usable_evidence_psnr=30,
        codec="SP03 JPEG60 ROI atlas+coarse4; full-image IF06 JPEG420/JP2/WebP with five fixed scales",
        random_seeds=[1729, 2718, 31415], encoder_config=CONFIG,
        primary=dict(heldout=5, budget=16384, method="emergent_mix", comparator="feature_change",
                     unknown_coverage_difference=0, known_macro_f1_noninferiority_margin=-.02,
                     decision="Both primary paired 95% intervals must exceed their respective margins"),
        secondary_interval="99.583333% two-sided for each of 12 class/cap contrasts against feature_change",
        selection_basis="Method/class/cap chosen using the completed development repair and known-task probe; fixed before reserved test pixels",
        reservation_file_sha256=sha(reservation), pair_ids=[p["id"] for p in reserved],
        calibration_manifest_sha256=sha(train_manifest), calibration_feature_sha256=sha(ROOT / "cache/second_scaled/features.npz"),
        galleries=gallery_hashes, prototypes={str(h): sha(ROOT / f"results/unknown_scaled/known_{h}.npz") for h in HELDOUTS},
        method_source_sha256={name: sha(ROOT / name) for name in SOURCES},
        ground_task="fixed cosine known-class calibration prototypes on actual receiver DINO features; native 32-pixel expansion",
        split_scope="256 pre-reserved official-test pairs; geographic independence unavailable")
    proposed = json.loads(json.dumps(proposed))
    if LOCK.exists():
        old = json.loads(LOCK.read_text("utf-8"))
        if {k: v for k, v in old.items() if k != "locked_utc"} != proposed:
            raise ValueError("Refusing to change locked test method")
        return old
    proposed["locked_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(LOCK, proposed)
    return proposed


def semantic_confusion(encoder, image, prototype, label, mask, feature_cache):
    identity = hashlib.sha256(image.tobytes()).hexdigest()
    if identity not in feature_cache:
        feature_cache[identity] = encoder.cached(Image.fromarray(image))[0]
    mean, classes = prototype
    prediction = classes[np.argmax(feature_cache[identity] @ mean.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
    return np.bincount((label[mask].astype(int) - 1) * 6 + prediction[mask].astype(int) - 1, minlength=36).reshape(6, 6)


def score_pair(pair, encoder, galleries, prototypes, known_models, wire, helper):
    before_pil, after_pil, old_label, label, changed = read_pair(ROOT, pair)
    before, current = np.asarray(before_pil), np.asarray(after_pil)
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    prior, boot_down, boot_ack = initialize(before, pair["id"], folder)
    f0, _ = encoder.cached(before_pil)
    f1, _ = encoder.cached(after_pil)
    device = encoder.device
    query = torch.nn.functional.normalize(torch.tensor(np.concatenate([f0, f1]), device=device), dim=1)
    change = helper["pool_scores"](change_score(f1, f0))
    gate = helper["exact_mse"](current, helper["restore_coarse"](prior.rgb, current)) > 10 ** (-2.5)
    sanity = helper["sanity_images"](prior, current, folder, gate)
    decoded_features, common_whole = {}, {}
    result = []
    for heldout in HELDOUTS:
        neighbor = torch.topk(query @ galleries[heldout].T, k=10, dim=1, sorted=True).values[:, -1]
        novelty = torch.clamp(2 - 2 * neighbor, min=0).cpu().numpy().reshape(2, 256)
        emergence = helper["pool_scores"](np.maximum(novelty[1] - novelty[0], 0))
        mean, precision = known_models[heldout]
        x = f1.astype(np.float64)
        px = x @ precision
        md = np.maximum((np.sum(px * x, axis=1)[:, None] - 2 * px @ mean.T + np.sum((mean @ precision) * mean, axis=1)).min(1), 0) / 384
        mahalanobis = helper["pool_scores"](md)
        known_mask, unknown_mask = helper["evaluation_masks"](old_label, label, changed, heldout)
        for method in METHODS + ("full_current", "old_only"):
            seeds = (1729, 2718, 31415) if method == "feature_random" else (0,)
            budgets = BUDGETS if method in METHODS else (0,)
            for seed in seeds:
                if method == "feature_change":
                    ranking = helper["ranking_scores"]("feature_change", change, change, mahalanobis, gate, seed, pair["id"])
                elif method in ("emergent_mix", "maha_mix", "feature_random"):
                    old_name = "pixel_random" if method == "feature_random" else "pixel_unknown"
                    extra = emergence if method == "emergent_mix" else mahalanobis
                    ranking = helper["ranking_scores"](old_name, change, change, extra, gate, seed, pair["id"])
                for budget in budgets:
                    detail = {}
                    if method == "full_current":
                        rgb = sanity["full_png"]
                        packet = (folder / "sanity_full_png.bin").read_bytes()
                        ack = (folder / "sanity_full_png.ack").read_bytes()
                    elif method == "old_only":
                        rgb, packet, ack = prior.rgb, b"", b""
                    elif method.startswith("full_"):
                        key = method, budget
                        if key not in common_whole:
                            packet, detail = full_fit(current, codec=method.removeprefix("full_"), budget=budget - 60,
                                region=prior.region, base_version=prior.version, base_digest=prior.digest(), observation=1)
                            received, ack = full_receive(prior, packet)
                            replay, replay_ack = full_receive(prior, packet)
                            if replay_ack != ack or not np.array_equal(replay.rgb, received.rgb):
                                raise AssertionError("Whole-image receiver replay differs")
                            common_whole[key] = packet, ack, received.rgb, detail
                        packet, ack, rgb, detail = common_whole[key]
                    else:
                        packet, ack, rgb, detail = helper["transact"](prior, current, method, budget, ranking, gate)
                    if budget and len(packet) + len(ack) > budget:
                        raise AssertionError("Actual current stream exceeds total cap")
                    name = f"h{heldout}_{method}_s{seed}_b{budget}"
                    packet_path = folder / (name + ".bin")
                    packet_path.write_bytes(packet)
                    packet_path.with_suffix(".ack").write_bytes(ack)
                    confusion = semantic_confusion(encoder, rgb, prototypes[heldout], label, known_mask, decoded_features)
                    record = dict(pair_id=pair["id"], heldout=heldout, method=method, seed=seed, budget=budget,
                        initial_down_bytes=boot_down, initial_ack_bytes=boot_ack, down_bytes=len(packet), ack_bytes=len(ack),
                        total_bytes=len(packet) + len(ack), packet_path=packet_path.relative_to(ROOT).as_posix(),
                        packet_sha256=hashlib.sha256(packet).hexdigest(), receiver_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        confusion=json.dumps(confusion.tolist(), separators=(",", ":")), **task_metrics(confusion, heldout),
                        **helper["image_metrics"](current, rgb, known_mask, unknown_mask),
                        codec_metadata=json.dumps(detail, sort_keys=True))
                    result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--wait-for-data", action="store_true")
    args = parser.parse_args()
    lock = lock_method()
    if args.freeze_only:
        print(json.dumps(dict(status="locked_before_test", pairs=len(lock["pair_ids"]), primary=lock["primary"])), flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    old_spec = json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8"))
    wire, helper = load_frozen(old_spec)
    encoder = FrozenEncoder(ROOT, CONFIG)
    galleries, prototypes, known_models = {}, {}, {}
    for heldout in HELDOUTS:
        item = lock["galleries"][str(heldout)]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Locked gallery changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            galleries[heldout] = torch.tensor(stored["features"], device=encoder.device)
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as stored:
            mean, precision, labels = stored["means"], stored["precision"], stored["labels"]
        known_models[heldout] = mean, precision
        prototypes[heldout] = mean / np.linalg.norm(mean, axis=1, keepdims=True), labels
    completed = set()
    for pair_id in lock["pair_ids"]:
        receipt_path = OUT / "pairs" / pair_id / "complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or sha(receipt_path.with_name("records.json")) != receipt["records_sha256"]:
                raise ValueError("Changed completed test pair")
            completed.add(pair_id)
    while len(completed) < len(lock["pair_ids"]):
        manifest_path = DATA / "manifest.json"
        available = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {"pairs": [], "complete": False}
        available_map = {p["id"]: p for p in available["pairs"]}
        pending = [p for p in lock["pair_ids"] if p in available_map and p not in completed]
        if not pending:
            if not args.wait_for_data or available["complete"]:
                raise ValueError("Locked test input is incomplete")
            time.sleep(20)
            continue
        for pair_id in pending:
            for source, expected in lock["method_source_sha256"].items():
                if sha(ROOT / source) != expected:
                    raise ValueError("Method changed after test access")
            pair = available_map[pair_id]
            for item in pair["files"].values():
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Changed test pixels or labels")
            records = score_pair(pair, encoder, galleries, prototypes, known_models, wire, helper)
            if len(records) != 114:
                raise AssertionError("Incomplete locked per-pair matrix")
            folder = OUT / "pairs" / pair_id
            write_json(folder / "records.json", records)
            write_json(folder / "complete.json", dict(status="complete", records=len(records), lock_sha256=sha(LOCK),
                       records_sha256=sha(folder / "records.json"), input_sha256=hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest()))
            completed.add(pair_id)
            write_json(OUT / "progress.json", dict(status="in_progress", completed_pairs=len(completed), expected_pairs=256,
                       records=114 * len(completed), expected_records=29184))
            print(f"locked official test {len(completed)}/256 pairs, {114*len(completed)} receiver observations", flush=True)
    records = []
    for pair_id in lock["pair_ids"]:
        records.extend(json.loads((OUT / "pairs" / pair_id / "records.json").read_text("utf-8")))
    helper["save_csv"](OUT / "records.csv", records)
    write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=len(records), expected=29184,
               lock_sha256=sha(LOCK), source_sha256=sha(Path(__file__)), records_sha256=sha(OUT / "records.csv"),
               encoder=encoder.metadata, completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
