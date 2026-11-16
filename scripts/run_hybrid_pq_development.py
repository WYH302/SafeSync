"""Joint known-feature service and RGB evidence under an actual shared cap."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from safesync.data import read_pair
from safesync.product_quantizer import ProductQuantizer
from safesync.hybrid_wire import encode as compose, receive as decode, HEADER
from scripts.run_second_feature_mix import load_frozen, initial_state

OUT = ROOT / "results/hybrid_pq/development_v2"
OLD = ROOT / "results/second_pixels/fixed_v1"
SCORE = ROOT / "results/unknown_recovery/scores_v1"
METHODS = ("change", "emergent_mix", "emergent_only", "random", "PQ_only")
BUDGETS = (4096, 8192, 16384, 32768)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(matrix, heldout):
    denom = matrix.sum(0) + matrix.sum(1)
    values = [2 * matrix[i, i] / denom[i] for i in range(6) if i != heldout - 1 and denom[i] > 0]
    return float(np.mean(values)) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/second_scaled/manifest.json").read_text("utf-8"))
    pairs = [p for p in manifest["pairs"] if p["split"] == "validation"]
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    wire, helper = load_frozen(json.loads((OLD / "run_spec.json").read_text("utf-8")))
    books, scores, prototypes = {}, {}, {}
    for heldout in (5, 3, 4):
        with np.load(ROOT / f"results/product_quantizer/known_only_v1/h{heldout}_m8.npz", allow_pickle=False) as stored:
            books[heldout] = ProductQuantizer(stored["centroids"])
        with np.load(SCORE / f"scores_{heldout}.npz", allow_pickle=False) as stored:
            if stored["ids"].tolist() != ids:
                raise ValueError("Hybrid frozen score IDs differ")
            scores[heldout] = stored["knn_emergence"]
        with np.load(ROOT / f"results/unknown_scaled/known_{heldout}.npz", allow_pickle=False) as stored:
            mean, classes = stored["means"], stored["labels"]
        prototypes[heldout] = mean / np.linalg.norm(mean, axis=1, keepdims=True), classes
    sources = ["scripts/run_hybrid_pq_development.py", "safesync/product_quantizer.py", "safesync/hybrid_wire.py"]
    spec = dict(methods=METHODS, budgets=BUDGETS, PQ_subspaces=8, centroid_count=256,
                source_sha256={name: sha(ROOT / name) for name in sources},
                known_task="All routes decode the same complete current PQ representation",
                image_routing="All 64 cells eligible; untransmitted RGB remains deferred; no semantic stability assertion",
                evidence="SP03 JPEG60 ROI atlas without mandatory coarse image, within remaining packet bytes",
                scope="Hypothesis repair after development scorer/MSE trade-off; original 64 development pairs",
                initialization="shared actual before PNG; known-only codebook deployment bytes recorded separately",
                gallery_source_sha256=sha(SCORE / "complete.json"), PQ_fit_sha256=sha(ROOT / "results/product_quantizer/known_only_v1/complete.json"))
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != json.loads(json.dumps(spec)):
        raise ValueError("Changed hybrid experiment")
    helper["save_json"](spec_path, spec)
    for name in sources:
        snapshot = OUT / "source_snapshot" / name.replace("/", "__")
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    records = []
    completed = []
    if (OUT / "records.csv").exists():
        records = list(csv.DictReader((OUT / "records.csv").open(encoding="utf-8")))
        completed = json.loads((OUT / "progress.json").read_text("utf-8"))["completed_pair_ids"]
    for pair in pairs:
        if pair["id"] in completed:
            continue
        before_pil, after_pil, old, current_label, changed = read_pair(ROOT, pair)
        before, current = np.asarray(before_pil), np.asarray(after_pil)
        folder = OUT / "pairs" / pair["id"]
        folder.mkdir(parents=True, exist_ok=True)
        prior, boot_down, boot_ack = initial_state(wire, folder, pair["id"], before)
        index = ids.index(pair["id"])
        with np.load(OLD / "input_snapshot" / f"scores_5_{pair['id']}.npz", allow_pickle=False) as stored:
            change = helper["pool_scores"](stored["change"])
        rank_change = rankdata(change) / 64
        for heldout in (5, 3, 4):
            quantizer = books[heldout]
            codes = quantizer.encode(features[index, 1])
            feature_packet = quantizer.frame(codes)
            novelty = helper["pool_scores"](scores[heldout][index])
            rank_unknown = rankdata(novelty) / 64
            mean, classes = prototypes[heldout]
            known_mask, unknown_mask = helper["evaluation_masks"](old, current_label, changed, heldout)
            reference_matrix = None
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
                        if method == "PQ_only":
                            image_packet = b""
                        else:
                            # The complete current feature service is mandatory.
                            # RGB has no mandatory coarse layer, so a small cap
                            # can still hold selected full-resolution evidence.
                            remaining = budget - len(feature_packet) - HEADER.size - 16 - 56
                            sender = wire.TransactionSender(prior.version, prior.digest())
                            image_packet, _, _, _ = helper["choose_packet"](
                                sender, current, prior.region, 1, ranking, 0., remaining,
                                coarse_side=0, codec="jpeg", quality=60)
                        packet = compose(feature_packet, image_packet, prior.region, 1)
                        received_features, rgb, ack = decode(prior, packet, quantizer, helper)
                        repeat_f, repeat_rgb, repeat_ack = decode(prior, packet, quantizer, helper)
                        if ack != repeat_ack or not np.array_equal(received_features, repeat_f) or not np.array_equal(rgb, repeat_rgb):
                            raise AssertionError("Hybrid fresh receiver replay failed")
                        if budget and len(packet) + len(ack) > budget:
                            raise AssertionError("Joint actual-byte cap exceeded")
                        predicted = classes[np.argmax(received_features @ mean.T, axis=1)].reshape(16, 16).repeat(32, 0).repeat(32, 1)
                        matrix = np.bincount((current_label[known_mask].astype(int) - 1) * 6 + predicted[known_mask].astype(int) - 1, minlength=36).reshape(6, 6)
                        if reference_matrix is not None and not np.array_equal(matrix, reference_matrix):
                            raise AssertionError("Image routing changed the known-feature service")
                        reference_matrix = matrix
                        transaction = folder / f"h{heldout}_{method}_s{seed}_b{budget}"
                        transaction.mkdir(exist_ok=True)
                        (transaction / "current.bin").write_bytes(packet)
                        (transaction / "ack.bin").write_bytes(ack)
                        records.append(dict(pair_id=pair["id"], heldout=heldout, method=method, seed=seed, budget=budget,
                            down_bytes=len(packet), ack_bytes=len(ack), total_bytes=len(packet) + len(ack),
                            feature_bytes=len(feature_packet), image_bytes=len(image_packet),
                            joint_header_checksum_bytes=HEADER.size + 16,
                            bootstrap_down_bytes=boot_down, bootstrap_ack_bytes=boot_ack,
                            codebook_raw_bytes=quantizer.centroids.nbytes,
                            packet_path=(transaction / "current.bin").relative_to(ROOT).as_posix(),
                            packet_sha256=sha(transaction / "current.bin"),
                            receiver_features_sha256=hashlib.sha256(received_features.tobytes()).hexdigest(),
                            receiver_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            known_confusion=json.dumps(matrix.tolist()), known_macro_f1=f1(matrix, heldout),
                            **helper["image_metrics"](current, rgb, known_mask, unknown_mask)))
        completed.append(pair["id"])
        helper["save_csv"](OUT / "records.csv", records)
        helper["save_json"](OUT / "progress.json", dict(status="in_progress", completed_pair_ids=completed, expected_pairs=64,
                            records=len(records), expected_records=4800))
        print(f"hybrid PQ {len(completed)}/64 pairs, {len(records)} joint receiver observations", flush=True)
    if len(records) != 4800:
        raise AssertionError("Incomplete hybrid development matrix")
    helper["save_json"](OUT / "complete.json", dict(status="complete", records=len(records), expected=4800,
        spec_sha256=sha(spec_path), records_sha256=sha(OUT / "records.csv"),
        all_known_outputs_identical_within_pair_and_fold=True))


if __name__ == "__main__":
    main()
