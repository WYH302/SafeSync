"""Lock a hybrid extension before inspecting the reserved-test outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from scipy.stats import rankdata
import torch
from safesync.data import read_pair
from safesync.features import FrozenEncoder
from safesync.scoring import change_score
from safesync.product_quantizer import ProductQuantizer
from safesync.hybrid_wire import encode, receive, HEADER
from scripts.run_second_feature_mix import load_frozen
from scripts.run_second_pixel import initialize
from scripts.run_locked_second_test import CONFIG, LOCK as EARLIER_LOCK, write_json, sha

OUT = ROOT / "results/hybrid_pq/confirmation_v1"
LOCK = OUT / "method_lock.json"
DATA = ROOT / "data/second_locked_test_v1"
BUDGETS = (4096, 8192, 16384, 32768)
METHODS = ("change", "emergent_mix", "emergent_only", "random", "PQ_only")
SOURCES = ("scripts/run_hybrid_pq_confirmation.py", "safesync/hybrid_wire.py", "safesync/product_quantizer.py",
           "safesync/features.py", "safesync/scoring.py", "safesync/data.py", "scripts/run_second_feature_mix.py",
           "scripts/run_second_pixel.py", "scripts/run_locked_second_test.py")


def lock_method():
    earlier = json.loads(EARLIER_LOCK.read_text("utf-8"))
    development = ROOT / "results/hybrid_pq/development_v2"
    if not (development / "analysis.json").exists():
        raise ValueError("Hybrid development analysis must precede its extension lock")
    books = {str(h): dict(path=f"results/product_quantizer/known_only_v1/h{h}_m8.npz",
                         sha256=sha(ROOT / f"results/product_quantizer/known_only_v1/h{h}_m8.npz")) for h in (5, 3, 4)}
    proposed = dict(schema="hybrid-reserved-extension-v1", methods=METHODS, budgets=BUDGETS, heldouts=[5, 3, 4],
        encoder_config=CONFIG, books=books, galleries=earlier["galleries"], prototypes=earlier["prototypes"],
        pair_ids=earlier["pair_ids"], earlier_lock_sha256=sha(EARLIER_LOCK),
        development_records_sha256=sha(development / "records.csv"), development_analysis_sha256=sha(development / "analysis.json"),
        source_sha256={p: sha(ROOT / p) for p in SOURCES},
        primary=dict(heldout=5, budget=8192, method="emergent_mix", comparator="change",
            decision="Positive paired 95% coverage difference and identical known-task confusion counts between routers"),
        secondary="99.583333% descriptive paired intervals for all 12 class/cap emergence-mixture contrasts",
        route="0.75 change rank + 0.25 positive temporal tenth-neighbor novelty rank; all 64 cells eligible",
        feature_service="Complete current known-only M8 PQ indices in every route, same codebook and decoder",
        evidence="Optional JPEG60 SP03 atlas without coarse field, actual HY01 combined bytes and 56-byte joint ACK",
        random_seeds=[1729, 2718, 31415], codebook_raw_bytes=393216,
        initialization="Same actual before-PNG initialization; one globally shared codebook reported separately",
        access_disclosure="Reserved pixels were already downloaded and the earlier image-only experiment was running. Only progress counts were inspected. This hybrid method was selected using 64 development pairs and locked before any reserved-test numerical outcomes were inspected. This is a secondary extension on the same reserved set.")
    proposed = json.loads(json.dumps(proposed))
    if LOCK.exists():
        stored = json.loads(LOCK.read_text("utf-8"))
        if {k: v for k, v in stored.items() if k != "locked_utc"} != proposed:
            raise ValueError("Refusing to change the hybrid confirmation lock")
        return stored
    proposed["locked_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(LOCK, proposed)
    return proposed


def score_pair(pair, encoder, wire, helper, books, galleries, prototypes):
    before_pil, after_pil, old, label, changed = read_pair(ROOT, pair)
    before, current = np.asarray(before_pil), np.asarray(after_pil)
    folder = OUT / "pairs" / pair["id"]
    folder.mkdir(parents=True, exist_ok=True)
    prior, initial_down, initial_ack = initialize(before, pair["id"], folder)
    f0, _ = encoder.cached(before_pil)
    f1, _ = encoder.cached(after_pil)
    rank_change = rankdata(helper["pool_scores"](change_score(f1, f0))) / 64
    query = torch.nn.functional.normalize(torch.tensor(np.concatenate([f0, f1]), device=encoder.device), dim=1)
    result = []
    for heldout in (5, 3, 4):
        neighbor = torch.topk(query @ galleries[heldout].T, k=10, dim=1, sorted=True).values[:, -1]
        novelty = torch.clamp(2 - 2 * neighbor, min=0).cpu().numpy().reshape(2, 256)
        emergence = helper["pool_scores"](np.maximum(novelty[1] - novelty[0], 0))
        rank_unknown = rankdata(emergence) / 64
        quantizer = books[heldout]
        codes = quantizer.encode(f1)
        feature_packet = quantizer.frame(codes)
        mean, classes = prototypes[heldout]
        known, unknown = helper["evaluation_masks"](old, label, changed, heldout)
        common_confusion = None
        for method in METHODS:
            seeds = (1729, 2718, 31415) if method == "random" else (0,)
            budgets = BUDGETS if method != "PQ_only" else (0,)
            for seed in seeds:
                if method == "change":
                    ranking = rank_change
                elif method == "emergent_mix":
                    ranking = .75 * rank_change + .25 * rank_unknown
                elif method == "emergent_only":
                    ranking = rank_unknown
                elif method == "random":
                    key = hashlib.sha256(f"hybrid-random:{pair['id']}:{seed}".encode()).digest()[:8]
                    ranking = np.random.default_rng(int.from_bytes(key, "little")).random(64) + .01
                for budget in budgets:
                    if budget:
                        remaining = budget - len(feature_packet) - HEADER.size - 16 - 56
                        sender = wire.TransactionSender(prior.version, prior.digest())
                        image_packet, _, _, _ = helper["choose_packet"](
                            sender, current, prior.region, 1, ranking, 0., remaining, coarse_side=0, codec="jpeg", quality=60)
                    else:
                        image_packet = b""
                    packet = encode(feature_packet, image_packet, prior.region, 1)
                    restored_features, rgb, ack = receive(prior, packet, quantizer, helper)
                    replay_f, replay_rgb, replay_ack = receive(prior, packet, quantizer, helper)
                    if (ack != replay_ack or not np.array_equal(restored_features, replay_f)
                            or not np.array_equal(rgb, replay_rgb) or (budget and len(packet) + len(ack) > budget)):
                        raise AssertionError("Hybrid confirmation replay or budget failure")
                    prediction = classes[np.argmax(restored_features @ mean.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                    matrix = np.bincount((label[known].astype(int) - 1) * 6 + prediction[known].astype(int) - 1, minlength=36).reshape(6, 6)
                    if common_confusion is not None and not np.array_equal(matrix, common_confusion):
                        raise AssertionError("Hybrid known service changed between routes")
                    common_confusion = matrix
                    name = f"h{heldout}_{method}_s{seed}_b{budget}"
                    path = folder / (name + ".bin")
                    path.write_bytes(packet)
                    path.with_suffix(".ack").write_bytes(ack)
                    result.append(dict(pair_id=pair["id"], heldout=heldout, method=method, seed=seed, budget=budget,
                        initial_down_bytes=initial_down, initial_ack_bytes=initial_ack, codebook_raw_bytes=quantizer.centroids.nbytes,
                        down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet) + len(ack),
                        feature_bytes=len(feature_packet), image_bytes=len(image_packet), joint_header_checksum_bytes=HEADER.size + 16,
                        packet_path=path.relative_to(ROOT).as_posix(), packet_sha256=sha(path),
                        receiver_features_sha256=hashlib.sha256(restored_features.tobytes()).hexdigest(),
                        receiver_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        known_confusion=json.dumps(matrix.tolist()), **helper["image_metrics"](current, rgb, known, unknown)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    lock = lock_method()
    if args.freeze_only:
        print(json.dumps(dict(status="locked_secondary_extension", primary=lock["primary"])), flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    spec = json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8"))
    wire, helper = load_frozen(spec)
    books, galleries, prototypes = {}, {}, {}
    for h in (5, 3, 4):
        item = lock["books"][str(h)]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Locked PQ book changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            books[h] = ProductQuantizer(stored["centroids"])
        item = lock["galleries"][str(h)]
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Locked gallery changed")
        with np.load(ROOT / item["path"], allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"], device=encoder.device)
        path = ROOT / f"results/unknown_scaled/known_{h}.npz"
        if sha(path) != lock["prototypes"][str(h)]:
            raise ValueError("Locked task prototype changed")
        with np.load(path, allow_pickle=False) as stored:
            mean, labels = stored["means"], stored["labels"]
        prototypes[h] = mean / np.linalg.norm(mean, axis=1, keepdims=True), labels
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    all_records = []
    for pair_id in lock["pair_ids"]:
        folder = OUT / "pairs" / pair_id
        receipt_path = folder / "complete.json"
        for source, expected in lock["source_sha256"].items():
            if sha(ROOT / source) != expected:
                raise ValueError("Hybrid method changed after extension lock")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or sha(folder / "records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed hybrid pair")
            records = json.loads((folder / "records.json").read_text("utf-8"))
        else:
            for item in pairs[pair_id]["files"].values():
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Changed reserved image or label")
            records = score_pair(pairs[pair_id], encoder, wire, helper, books, galleries, prototypes)
            if len(records) != 75:
                raise AssertionError("Incomplete hybrid pair")
            write_json(folder / "records.json", records)
            write_json(receipt_path, dict(status="complete", records=75, lock_sha256=sha(LOCK), records_sha256=sha(folder / "records.json")))
        all_records.extend(records)
        write_json(OUT / "progress.json", dict(status="in_progress", completed_pairs=len(all_records) // 75,
            expected_pairs=256, records=len(all_records), expected_records=19200))
        print(f"hybrid confirmation {len(all_records)//75}/256, {len(all_records)} received observations", flush=True)
    helper["save_csv"](OUT / "records.csv", all_records)
    write_json(OUT / "complete.json", dict(status="complete", pairs=256, records=len(all_records), expected=19200,
        lock_sha256=sha(LOCK), records_sha256=sha(OUT / "records.csv"), encoder=encoder.metadata,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
