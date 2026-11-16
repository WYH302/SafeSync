"""Sender-visible quality choice between actual ROI and full-image packets."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from safesync import task_evidence_wire as wire
from safesync.data import read_pair
from scripts.run_second_feature_mix import load_frozen

SOURCE = ROOT / "results/task_evidence/development_v1"
OUT = ROOT / "results/task_evidence/selector_development_v1"
TAU = 10 ** -3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", "utf-8")


def quality(current, received, ranking):
    error = ((current.astype(np.float64) - received.astype(np.float64))/255) ** 2
    cell_error = error.reshape(8, 64, 8, 64, 3).mean((1, 3, 4)).reshape(-1)
    value = np.maximum(1 - cell_error / TAU, 0)
    return float(np.dot(ranking, value) / np.sum(ranking))


def main():
    receipt = json.loads((SOURCE / "complete.json").read_text("utf-8"))
    if receipt["status"] != "complete" or sha(SOURCE / "records.csv") != receipt["records_sha256"]:
        raise ValueError("The factorial candidate set is incomplete")
    manifest = json.loads((ROOT / "data/second_scaled/manifest.json").read_text("utf-8"))
    with np.load(ROOT / "cache/second_scaled/features.npz", allow_pickle=False) as stored:
        features, ids = stored["features"].astype(np.float32), stored["ids"].tolist()
    source_rows = list(csv.DictReader((SOURCE / "records.csv").open(encoding="utf-8")))
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[row["pair_id"]].append(row)
    novelty = {}
    for h in (5, 3, 4):
        with np.load(ROOT / f"results/unknown_recovery/scores_v1/scores_{h}.npz", allow_pickle=False) as stored:
            novelty[h] = stored["knn_emergence"]
    _, helper = load_frozen(json.loads((ROOT / "results/second_pixels/fixed_v1/run_spec.json").read_text("utf-8")))
    spec = dict(source_sha256=sha(Path(__file__)), wire_sha256=sha(ROOT / "safesync/task_evidence_wire.py"),
        candidates_sha256=receipt["records_sha256"], quality_psnr=30, candidate_methods=["labels_webp_change", "labels_webp_mix", "labels_webp_random", "labels_full_webp"],
        quality="sum ranking_i * max(1 - actual native RGB cell MSE_i / 0.001, 0) / sum ranking_i",
        decision="Choose full image on ties; choices use only current RGB, decoded common history and actual decoded candidate RGB",
        scope="Development repair of diffuse-content failures; no semantic labels enter the selector",
        metadata="Full and ROI candidates already include the exact same decoded known-task label service")
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise ValueError("Changed quality selector")
    save(spec_path, spec)
    (OUT / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    results = []
    for pair in [p for p in manifest["pairs"] if p["split"] == "validation"]:
        index = ids.index(pair["id"])
        _, after, _, _, _ = read_pair(ROOT, pair)
        current = np.asarray(after)
        change_rank = rankdata(helper["pool_scores"](1 - np.sum(features[index, 0]*features[index, 1], axis=1)))/64
        rows = grouped[pair["id"]]
        lookup = {(int(r["heldout"]), int(r["budget"]), r["method"], int(r["seed"])): r for r in rows}
        for h in (5, 3, 4):
            unknown_rank = rankdata(helper["pool_scores"](novelty[h][index]))/64
            first = next(r for r in rows if int(r["heldout"]) == h and r["service"] == "labels")
            boot = ROOT / first["packet_path"]
            boot = boot.parent / "bootstrap.bin"
            packet = boot.read_bytes()
            header = wire.HEADER.unpack_from(packet)
            blank = wire.TaskEvidenceState(np.zeros_like(current), np.zeros((16, 16), np.uint8), header[1], header[2])
            prior, _ = wire.receive(blank, packet)
            for cap in (4096, 8192, 16384, 32768):
                full = lookup[(h, cap, "labels_full_webp", 0)]
                full_state, _ = wire.receive(prior, (ROOT / full["packet_path"]).read_bytes())
                for route in ("change", "mix", "random"):
                    for seed in ((1729, 2718, 31415) if route == "random" else (0,)):
                        roi = lookup[(h, cap, "labels_webp_" + route, seed)]
                        roi_state, _ = wire.receive(prior, (ROOT / roi["packet_path"]).read_bytes())
                        if route == "change":
                            ranking = change_rank
                        elif route == "mix":
                            ranking = .75*change_rank + .25*unknown_rank
                        else:
                            key = hashlib.sha256(f"task-evidence-random:{pair['id']}:{seed}".encode()).digest()[:8]
                            ranking = np.random.default_rng(int.from_bytes(key, "little")).random(64) + .01
                        fq, rq = quality(current, full_state.rgb, ranking), quality(current, roi_state.rgb, ranking)
                        chosen = roi if rq > fq else full
                        if chosen["known_confusion"] != full["known_confusion"]:
                            raise AssertionError("Action choice changed the known service")
                        record = {key: value for key, value in chosen.items() if key not in ("cached_pipeline_ms", "codec_metadata")}
                        record.update(method="labels_adaptive_" + route, seed=seed, chosen_method=chosen["method"],
                            roi_proxy_quality=rq, full_proxy_quality=fq, sender_has_source_RGB=True, sender_has_semantic_GT=False)
                        results.append(record)
        print(f"quality selector {pair['id']}, {len(results)} actual chosen transactions", flush=True)
    if len(results) != 3840:
        raise AssertionError("Incomplete quality-selector matrix")
    helper["save_csv"](OUT / "records.csv", results)
    save(OUT / "complete.json", dict(status="complete", records=len(results), expected=3840,
        records_sha256=sha(OUT / "records.csv"), spec_sha256=sha(spec_path), source_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    main()
