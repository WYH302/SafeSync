"""Uncached primary-cap latency using the confirmed sender/receiver entry points."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync import task_evidence_wire as wire
from safesync.features import FrozenEncoder
from scripts.run_locked_second_test import CONFIG, write_json, sha
from scripts.run_task_evidence_development import fit_full
from scripts.run_task_context_development import fit_context
from scripts.run_task_evidence_selector import quality

SOURCE = ROOT / "results/task_evidence/fresh_context_v1"
OUT = ROOT / "results/task_evidence/latency_v1"


def pool(values):
    return np.asarray(values).reshape(8, 2, 8, 2).mean((1, 3)).reshape(-1)


def main():
    # No data-selection, model fitting or cache-based inference occurs here.
    if not (SOURCE / "complete.json").exists():
        raise RuntimeError("Complete confirmation before timing its implementation")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((SOURCE / "method_lock.json").read_text("utf-8"))
    manifest = json.loads((ROOT / "data/second_task_context_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]: p for p in manifest["pairs"]}
    ids = sorted(lock["pair_ids"], key=lambda p: hashlib.sha256(("latency:"+p).encode()).hexdigest())[:16]
    sources = ["scripts/profile_task_context.py", "safesync/features.py", "safesync/task_evidence_wire.py",
        "scripts/run_task_evidence_development.py", "scripts/run_task_context_development.py",
        "scripts/run_task_evidence_selector.py", "safesync/full_image_codecs.py"]
    spec = dict(pair_ids=ids, repeats=3, cap=8192, heldout=5, source_sha256={p:sha(ROOT/p) for p in sources},
        source_lock_sha256=sha(SOURCE/"method_lock.json"),
        entry_point="Uncached current-image read, acknowledged state read, FrozenEncoder.extract, current task head, route, real codec search, TE01 serialization and receive",
        shared_history="Identical acknowledged RGB and label bootstrap and FP16 history features for every method",
        cache="No prior experiment codec or feature cache is used in timed current inference or codec search. An atlas cache is confined to one new candidate search.",
        scope="Local single-process microbenchmark after other experiment GPU workers exit. Disk page cache and ordinary OS scheduling are uncontrolled. Network and radio delays excluded.")
    spec_path = OUT / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text("utf-8")) != spec:
        raise RuntimeError("Refusing to replace a different latency specification")
    write_json(spec_path, spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder(ROOT, CONFIG)
    with np.load(ROOT / "results/unknown_scaled/known_5.npz", allow_pickle=False) as stored:
        means, classes = stored["means"], stored["labels"]
    means = means / np.linalg.norm(means, axis=1, keepdims=True)
    path = ROOT / lock["galleries"]["5"]["path"]
    with np.load(path, allow_pickle=False) as stored:
        gallery_values = stored["features"].astype(np.float32)
    gallery = torch.tensor(gallery_values, device=encoder.device)
    records = []
    for pair_id in ids:
        pair = pairs[pair_id]
        source = SOURCE / "pairs" / pair_id
        for repeat in range(3):
            start = time.perf_counter()
            image = Image.open(ROOT / pair["files"]["after"]["path"]).convert("RGB")
            current = np.asarray(image).copy()
            input_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            with np.load(source / "sender_features.npz", allow_pickle=False) as stored:
                old_feature = stored["before"].astype(np.float32)
            blank = wire.TaskEvidenceState(np.zeros((512, 512, 3), np.uint8), np.zeros((16,16), np.uint8),
                hashlib.sha256(("task-evidence:"+pair_id).encode()).digest()[:16],
                hashlib.sha256(means.tobytes()+classes.tobytes()+b"task-labels-v1").digest()[:16])
            prior, _ = wire.receive(blank, (source / "h5_bootstrap.bin").read_bytes())
            state_ms = (time.perf_counter()-start)*1000
            feature, timings = encoder.extract(image)
            # FP16 storage round-trip matches the representation used in confirmation.
            feature = feature.astype(np.float16).astype(np.float32)
            start = time.perf_counter()
            labels = classes[np.argmax(feature@means.T, axis=1)].astype(np.uint8).reshape(16,16)
            task_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            change = rankdata(pool(1-np.sum(old_feature*feature, axis=1)))/64
            change_ms = (time.perf_counter()-start)*1000
            encoder.sync()
            start = time.perf_counter()
            query = torch.nn.functional.normalize(torch.tensor(np.concatenate([old_feature, feature]), device=encoder.device), dim=1)
            nearest = torch.topk(query@gallery.T, k=10, dim=1, sorted=True).values[:,-1]
            distance = torch.clamp(2-2*nearest, min=0).cpu().numpy().reshape(2,256)
            novelty = rankdata(pool(np.maximum(distance[1]-distance[0], 0)))/64
            encoder.sync()
            novelty_ms = (time.perf_counter()-start)*1000
            for method in ("full_webp", "adaptive_change", "adaptive_mix"):
                start = time.perf_counter()
                full, _, _ = fit_full(prior, labels, current, "webp", 8192)
                full_state, full_ack = wire.receive(prior, full)
                full_ms = (time.perf_counter()-start)*1000
                context_ms = selection_ms = 0.
                packet, received, ack = full, full_state, full_ack
                if method != "full_webp":
                    ranking = change if method == "adaptive_change" else .75*change+.25*novelty
                    start = time.perf_counter()
                    base, _, _ = fit_full(prior, labels, current, "webp", 4096)
                    contextual, _, _ = fit_context(prior, labels, current, ranking, 8192, {}, base)
                    contextual_state, contextual_ack = wire.receive(prior, contextual)
                    context_ms = (time.perf_counter()-start)*1000
                    start = time.perf_counter()
                    if quality(current, contextual_state.rgb, ranking) > quality(current, full_state.rgb, ranking):
                        packet, received, ack = contextual, contextual_state, contextual_ack
                    selection_ms = (time.perf_counter()-start)*1000
                if len(packet)+len(ack) > 8192 or not np.array_equal(received.labels, labels):
                    raise AssertionError("Timed normal path changed its task contract")
                sender_ms = full_ms+context_ms+selection_ms+task_ms
                route_ms = 0. if method == "full_webp" else change_ms
                if method == "adaptive_mix":
                    route_ms += novelty_ms
                records.append(dict(pair_id=pair_id, repeat=repeat, method=method,
                    input_read_ms=input_ms, state_read_ms=state_ms, **timings, task_head_ms=task_ms,
                    route_ms=route_ms, full_codec_and_decode_ms=full_ms, context_codec_and_decode_ms=context_ms,
                    selection_ms=selection_ms, sender_core_ms=sender_ms+route_ms,
                    image_to_packet_and_decode_ms=input_ms+state_ms+timings["total_ms"]+sender_ms+route_ms,
                    total_bytes=len(packet)+len(ack), packet_sha256=hashlib.sha256(packet).hexdigest()))
        print(f"uncached local timing {ids.index(pair_id)+1}/16", flush=True)
    with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = ("total_ms", "route_ms", "sender_core_ms", "image_to_packet_and_decode_ms")
    summary = {}
    for method in ("full_webp", "adaptive_change", "adaptive_mix"):
        group = [r for r in records if r["method"] == method]
        summary[method] = {k:dict(median=float(np.median([r[k] for r in group])),
            p90=float(np.quantile([r[k] for r in group], .9))) for k in metrics}
    write_json(OUT / "complete.json", dict(status="complete", pairs=16, records=len(records),
        spec_sha256=sha(spec_path), records_sha256=sha(OUT/"records.csv"), encoder=encoder.metadata,
        python=sys.version, platform=platform.platform(), gallery_vectors=len(gallery_values), gallery_bytes=gallery_values.nbytes,
        old_rgb_bytes=512*512*3, old_feature_fp16_bytes=256*384*2, task_array_bytes=256,
        summary=summary, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if encoder.device == "cuda" else None))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
