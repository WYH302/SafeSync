"""Exploratory score-only intervention on the unchanged real hyperprior grid."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync import task_evidence_wire as wire
from safesync import learned_codec as learned
from safesync.pixel_wire import PixelState
from scripts.run_learned_codec import cached_candidates
from scripts.run_locked_second_test import sha, write_json

MAIN = ROOT / "results/task_evidence/fresh_context_v1"
CONTROL = ROOT / "results/task_evidence/fresh_context_learned_v1"
OUT = ROOT / "results/task_evidence/hyperprior_risk_development_v1"
CAPS = (4096,8192,16384,32768)
METHODS = ("mse","uniform_quality","change_quality","current_distance_quality","temporal_quality")


def pool(values):
    return np.asarray(values).reshape(8,2,8,2).mean((1,3)).reshape(-1)


def main():
    lock = json.loads((MAIN / "method_lock.json").read_text("utf-8"))
    manifest = json.loads((ROOT / "data/second_task_context_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]:p for p in manifest["pairs"]}
    ids = sorted(lock["pair_ids"], key=lambda p:hashlib.sha256(("codec-selection-development:"+p).encode()).hexdigest())[:64]
    names = ("scripts/run_hyperprior_risk_selection.py","safesync/learned_codec.py","safesync/task_evidence_wire.py",
        "safesync/entropy_bridge.py","scripts/run_learned_codec.py","resources/learned_codec/entropy_server.py")
    spec = dict(pair_ids=ids, caps=CAPS, heldouts=(5,3,4), methods=METHODS,
        source_sha256={p:sha(ROOT/p) for p in names}, source_lock_sha256=sha(MAIN/"method_lock.json"),
        source_control_sha256=sha(CONTROL/"complete.json"), gallery=lock["galleries"],
        objective="max sum rank_weight_i * max(1 - decoded native 64x64 RGB MSE_i/0.001,0); stable MSE/grid tie-break",
        weights="uniform; change rank; .75 change+.25 current known distance rank; .75 change+.25 positive distance increment rank",
        change="cosine feature distance from current to acknowledged previous observation", novelty_k=10,
        scope="Exploratory, after the complete fresh-context results. Reuses64 preinspected source pairs by metadata-only hash. Cannot replace that failed confirmation.",
        hypothesis="A weak candidate objective may leave unknown-evidence value unrealized even with a strong fixed codec. A quality-only control distinguishes the objective change from the unknown signal.",
        success="Increment over both MSE and uniform-quality/change-quality controls, with fixed task states and a measured later-use check. Development evidence only.",
        new_training=False, new_network=False, candidates="Exactly the existing8 qualities x5 scales, actual rANS streams, no ROI/context/codec change",
        expected_records=3840)
    spec = json.loads(json.dumps(spec))
    OUT.mkdir(parents=True,exist_ok=True)
    spec_path = OUT/"run_spec.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text("utf-8"))
        if {k:v for k,v in stored.items() if k != "locked_utc"} != spec:
            raise ValueError("Changed exploratory risk-selection intervention")
    else:
        spec["locked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(spec_path,spec)
    codec = learned.HyperpriorCodec(ROOT/"resources/learned_codec",device="cuda",threads=2,entropy_bridge=True)
    write_json(OUT/"assets.json",codec.prepare())
    galleries = {}
    for h in (5,3,4):
        path = ROOT/lock["galleries"][str(h)]["path"]
        if sha(path) != lock["galleries"][str(h)]["sha256"]:
            raise ValueError("Changed known-only gallery")
        with np.load(path,allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"],device=codec.device)
    all_rows = []
    for pair_id in ids:
        folder = OUT/"pairs"/pair_id
        receipt_path = folder/"complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            if receipt["spec_sha256"] != sha(spec_path) or sha(folder/"records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed risk-selection pair")
            all_rows.extend(json.loads((folder/"records.json").read_text("utf-8")))
            continue
        folder.mkdir(parents=True,exist_ok=True)
        source = MAIN/"pairs"/pair_id
        ready = json.loads((source/"sender_inputs_ready.json").read_text("utf-8"))
        if sha(source/"sender_features.npz") != ready["features_sha256"]:
            raise ValueError("Changed committed features")
        item = pairs[pair_id]["files"]["after"]
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed sender RGB")
        current = np.asarray(Image.open(ROOT/item["path"]).convert("RGB"))
        with np.load(source/"sender_features.npz",allow_pickle=False) as stored:
            f0,f1 = stored["before"].astype(np.float32),stored["current"].astype(np.float32)
        change = rankdata(pool(1-np.sum(f0*f1,axis=1)))/64
        query = torch.nn.functional.normalize(torch.tensor(np.concatenate([f0,f1]),device=codec.device),dim=1)
        candidate_folder = CONTROL/"pairs"/pair_id/"candidates"
        saved = json.loads((candidate_folder/"candidates.json").read_text("utf-8"))
        if saved["identity"]["source_rgb_sha256"] != hashlib.sha256(current.tobytes()).hexdigest():
            raise ValueError("Candidate cache has different RGB")
        candidates,_,_ = cached_candidates(codec,current,candidate_folder,saved["identity"])
        # All candidate errors are computed before any semantic GT is read.
        values = {}
        for index,candidate in enumerate(candidates):
            if candidate.packet_bytes > max(CAPS):
                continue
            packet = learned.pack(candidate,region=bytes(16),base_version=0,base_digest=bytes(16),observation=1)
            rgb,_ = codec.decode_packet(packet)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != candidate.decoded_rgb_sha256:
                raise AssertionError("Stored entropy candidate decoded differently")
            error = ((current.astype(np.float64)-rgb)/255)**2
            cell = error.reshape(8,64,8,64,3).mean((1,3,4)).reshape(-1)
            values[index] = np.maximum(1-cell/.001,0)
        rows = []
        for h in (5,3,4):
            common = ready["common"][str(h)]
            boot_path,task_path = ROOT/common["bootstrap_path"], ROOT/common["labels_only_path"]
            if sha(boot_path) != common["bootstrap_sha256"] or sha(task_path) != common["labels_only_sha256"]:
                raise ValueError("Changed common task-state packets")
            boot = boot_path.read_bytes()
            head = wire.HEADER.unpack_from(boot)
            blank = wire.TaskEvidenceState(np.zeros((512,512,3),np.uint8),np.zeros((16,16),np.uint8),head[1],head[2])
            prior,_ = wire.receive(blank,boot)
            service,_ = wire.receive(prior,task_path.read_bytes())
            pixel_prior = PixelState(prior.rgb,prior.region,prior.block,prior.version,prior.observed_at,prior.statuses)
            nearest = torch.topk(query@galleries[h].T,k=10,sorted=True).values[:,-1]
            distance = torch.clamp(2-2*nearest,min=0).cpu().numpy().reshape(2,256)
            rankings = dict(uniform_quality=np.ones(64), change_quality=change,
                current_distance_quality=.75*change+.25*rankdata(pool(distance[1]))/64,
                temporal_quality=.75*change+.25*rankdata(pool(np.maximum(distance[1]-distance[0],0)))/64)

            def decode_image(base,payload):
                parsed = learned.parse(payload)
                if (parsed["region"] != base.region or parsed["base_version"] != base.version
                        or parsed["base_digest"] != pixel_prior.digest() or parsed["observation"] != 1):
                    raise ValueError("Nested learned selection reference mismatch")
                return codec.decode_packet(payload)[0]

            empty,_ = wire.encode(prior,service.labels,observation=1)
            for cap in CAPS:
                available = cap-len(empty)-wire.ACK_BYTES
                fitting = [i for i,c in enumerate(candidates) if c.packet_bytes <= available]
                if not fitting:
                    raise ValueError("No candidate fits the real task-state envelope")
                for method in METHODS:
                    if method == "mse":
                        index = min(fitting,key=lambda i:candidates[i].sender_pixel_mse)
                    else:
                        ranking = rankings[method]
                        index = min(fitting,key=lambda i:(-float(np.dot(ranking,values[i])/ranking.sum()),candidates[i].sender_pixel_mse,i))
                    candidate = candidates[index]
                    inner = learned.pack(candidate,region=prior.region,base_version=prior.version,base_digest=pixel_prior.digest(),observation=1)
                    packet,costs = wire.encode(prior,service.labels,observation=1,payload=inner,image_mode=wire.LEARNED)
                    state,ack = wire.receive(prior,packet,learned_decoder=decode_image)
                    if len(packet)+len(ack) > cap or not np.array_equal(state.labels,service.labels):
                        raise AssertionError("Risk-selected task state or actual cap")
                    if method == "mse" and packet != (CONTROL/"pairs"/pair_id/f"h{h}_b{cap}.bin").read_bytes():
                        raise AssertionError("Original MSE codec control was not reproduced exactly")
                    path = folder/f"h{h}_{method}_b{cap}.bin"
                    path.write_bytes(packet)
                    path.with_suffix(".ack").write_bytes(ack)
                    png = path.with_suffix(".received.png")
                    Image.fromarray(state.rgb).save(png)
                    rows.append(dict(pair_id=pair_id,heldout=h,method=method,seed=0,budget=cap,
                        down_bytes=len(packet),ack_bytes=len(ack),total_bytes=len(packet)+len(ack),bootstrap_bytes=common["bootstrap_bytes"],
                        bootstrap_path=common["bootstrap_path"],packet_path=path.relative_to(ROOT).as_posix(),packet_sha256=sha(path),
                        receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
                        received_png=png.relative_to(ROOT).as_posix(),received_png_sha256=sha(png),candidate_index=index,
                        candidate_quality=candidate.quality,coded_shape=candidate.coded_shape,components=costs))
        write_json(folder/"sender_commit.json",rows)
        write_json(folder/"records.json",rows)
        write_json(receipt_path,dict(status="complete",records=60,spec_sha256=sha(spec_path),records_sha256=sha(folder/"records.json"),
            sender_commit_sha256=sha(folder/"sender_commit.json"),labels_read=False,candidate_cache_sha256=sha(candidate_folder/"candidates.json")))
        all_rows.extend(rows)
        write_json(OUT/"progress.json",dict(completed_pairs=len(all_rows)//60,expected_pairs=64,records=len(all_rows)))
        print(f"score-only hyperprior intervention {len(all_rows)//60}/64",flush=True)
    if len(all_rows) != 3840:
        raise AssertionError("Incomplete exploratory candidate selection")
    write_json(OUT/"records.json",all_rows)
    write_json(OUT/"complete.json",dict(status="complete",pairs=64,records=3840,spec_sha256=sha(spec_path),records_sha256=sha(OUT/"records.json")))


if __name__ == "__main__":
    main()
