"""Actual new-cohort learned-base evidence transactions in disjoint workers."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync import task_evidence_wire as wire
from safesync import residual_evidence as enhancement
from safesync import learned_codec as learned
from safesync import full_image_codecs as whole
from safesync.pixel_wire import PixelState
from scripts.run_learned_codec import cached_candidates
from scripts.run_residual_evidence_development import fit,pool
from scripts.run_task_evidence_development import fit_full
from scripts.run_task_evidence_selector import quality
from scripts.run_locked_second_test import sha,write_json

OUT = ROOT/"results/task_evidence/residual_confirmation_v1"
DATA = ROOT/"data/second_residual_test_v1"
LOCK = OUT/"method_lock.json"


def encode_pair(pair,lock,codec,assets,galleries):
    folder = OUT/"pairs"/pair["id"]
    ready = json.loads((folder/"sender_inputs_ready.json").read_text("utf-8"))
    if ready["lock_sha256"] != sha(LOCK) or sha(folder/"sender_features.npz") != ready["features_sha256"]:
        raise ValueError("Changed new sender state")
    after = pair["files"]["after"]
    if sha(ROOT/after["path"]) != after["sha256"]:
        raise ValueError("New sender current image differs")
    current = np.asarray(Image.open(ROOT/after["path"]).convert("RGB"))
    with np.load(folder/"sender_features.npz",allow_pickle=False) as stored:
        f0,f1 = stored["before"].astype(np.float32),stored["current"].astype(np.float32)
    change = rankdata(pool(1-np.sum(f0*f1,axis=1)))/64
    query = torch.nn.functional.normalize(torch.tensor(f1,device=codec.device),dim=1)
    identity = dict(method_lock_sha256=sha(LOCK),source_rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
        runtime=codec.runtime,weights=assets["weights"])
    candidates,search_ms,_ = cached_candidates(codec,current,folder/"candidates",identity)
    decoded,atlas_cache,full_cache = {},{},{}
    original_atlas,original_full = wire.encode_atlas,whole.encode_payload

    def cached_atlas(image,indices,*,codec="webp",quality=70,block=64):
        key = hashlib.sha256(image.tobytes()).digest(),image.shape,tuple(map(int,indices)),codec,quality,block
        if key not in atlas_cache:
            atlas_cache[key] = original_atlas(image,indices,codec=codec,quality=quality,block=block)
        return atlas_cache[key]

    def cached_full(image,codec,quality):
        key = image.size,codec,quality
        if key not in full_cache:
            full_cache[key] = original_full(image,codec,quality)
        return full_cache[key]

    rows = []

    def commit(h,cap,method,seed,packet,state,ack,*,refined=0,enhanced=False):
        if cap and len(packet)+len(ack) > cap:
            raise AssertionError("Locked residual packet+feedback exceeds cap")
        if method != "old_only" and hashlib.sha256(state.labels.tobytes()).hexdigest() != ready["common"][str(h)]["current_labels_sha256"]:
            raise AssertionError("Locked current-task state differs")
        path = folder/f"h{h}_{method}_s{seed}_b{cap}.bin"
        path.write_bytes(packet)
        path.with_suffix(".ack").write_bytes(ack)
        png = path.with_suffix(".received.png")
        Image.fromarray(state.rgb).save(png)
        common = ready["common"][str(h)]
        rows.append(dict(pair_id=pair["id"],heldout=h,method=method,seed=seed,budget=cap,
            down_bytes=len(packet),ack_bytes=len(ack),total_bytes=len(packet)+len(ack),bootstrap_bytes=common["bootstrap_bytes"],
            bootstrap_path=common["bootstrap_path"],packet_path=path.relative_to(ROOT).as_posix(),packet_sha256=sha(path),
            receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
            received_png=png.relative_to(ROOT).as_posix(),received_png_sha256=sha(png),refined_rois=refined,
            enhancement_selected=enhanced,complete_candidate_search_ms=search_ms))

    wire.encode_atlas,whole.encode_payload = cached_atlas,cached_full
    try:
        for h in lock["heldouts"]:
            common = ready["common"][str(h)]
            boot_path,task_path = ROOT/common["bootstrap_path"],ROOT/common["labels_only_path"]
            if sha(boot_path) != common["bootstrap_sha256"] or sha(task_path) != common["labels_only_sha256"]:
                raise ValueError("Changed common locked task packets")
            boot = boot_path.read_bytes()
            head = wire.HEADER.unpack_from(boot)
            blank = wire.TaskEvidenceState(np.zeros((512,512,3),np.uint8),np.zeros((16,16),np.uint8),head[1],head[2])
            prior,_ = wire.receive(blank,boot)
            service,_ = wire.receive(prior,task_path.read_bytes())
            pixel_prior = PixelState(prior.rgb,prior.region,prior.block,prior.version,prior.observed_at,prior.statuses)
            nearest = torch.topk(query@galleries[h].T,k=10,sorted=True).values[:,-1]
            distance = torch.clamp(2-2*nearest,min=0).cpu().numpy()
            mixed = .75*change+.25*rankdata(pool(distance))/64

            def decode_base(payload):
                parsed = learned.parse(payload)
                if (parsed["region"] != prior.region or parsed["base_version"] != prior.version
                        or parsed["base_digest"] != pixel_prior.digest() or parsed["observation"] != 1):
                    raise ValueError("Locked residual learned-base reference mismatch")
                key = hashlib.sha256(payload).hexdigest()
                if key not in decoded:
                    decoded[key] = codec.decode_packet(payload)[0]
                return decoded[key]

            def decode_image(base,payload):
                return enhancement.decode(payload,base.rgb.shape[:2],decode_base) if payload[:4] == b"RE01" else decode_base(payload)

            empty,_ = wire.encode(prior,service.labels,observation=1)
            for cap in lock["caps"]:
                full_inner,_ = learned.select(candidates,budget=cap-len(empty)-wire.ACK_BYTES,region=prior.region,
                    base_version=prior.version,base_digest=pixel_prior.digest(),observation=1)
                full,_ = wire.encode(prior,service.labels,observation=1,payload=full_inner,image_mode=wire.LEARNED)
                full_state,full_ack = wire.receive(prior,full,learned_decoder=decode_image)
                base_inner,_ = learned.select(candidates,budget=cap//2-len(empty)-wire.ACK_BYTES,region=prior.region,
                    base_version=prior.version,base_digest=pixel_prior.digest(),observation=1)
                base_rgb = decode_base(base_inner)
                commit(h,cap,"hyperprior",0,full,full_state,full_ack)
                for name,mode in (("absolute",enhancement.ABSOLUTE),("residual",enhancement.RESIDUAL)):
                    rankings = [("change",0,change),("current",0,mixed)]
                    if name == "residual":
                        rankings += [("random",seed,np.random.default_rng(seed+int(pair["id"])*1009).random(64)) for seed in lock["seeds"]]
                    for route,seed,ranking in rankings:
                        candidate,refined = fit(prior,service.labels,current,base_inner,base_rgb,ranking,mode,cap)
                        state,ack = wire.receive(prior,candidate,learned_decoder=decode_image)
                        enhanced = quality(current,state.rgb,ranking) > quality(current,full_state.rgb,ranking)
                        packet,selected,selected_ack = (candidate,state,ack) if enhanced else (full,full_state,full_ack)
                        commit(h,cap,f"{name}_{route}",seed,packet,selected,selected_ack,refined=refined,enhanced=enhanced)
                webp,_,_ = fit_full(prior,service.labels,current,"webp",cap)
                webp_state,webp_ack = wire.receive(prior,webp)
                commit(h,cap,"full_webp",0,webp,webp_state,webp_ack)
            task_packet = task_path.read_bytes()
            task_state,task_ack = wire.receive(prior,task_packet)
            commit(h,0,"labels_only",0,task_packet,task_state,task_ack)
            commit(h,0,"old_only",0,b"",prior,b"")
            stream = io.BytesIO()
            Image.fromarray(current).save(stream,format="PNG")
            packet,_ = wire.encode(prior,service.labels,observation=1,payload=stream.getvalue(),image_mode=wire.FULL)
            state,ack = wire.receive(prior,packet)
            commit(h,0,"full_current",0,packet,state,ack)
    finally:
        wire.encode_atlas,whole.encode_payload = original_atlas,original_full
    if len(rows) != 117:
        raise AssertionError("Locked residual source-pair matrix is incomplete")
    write_json(folder/"sender_commit.json",rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--worker-index",type=int,default=0)
    parser.add_argument("--finalize-only",action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text("utf-8"))
    if args.finalize_only:
        rows = []
        for pair_id in lock["pair_ids"]:
            folder = OUT/"pairs"/pair_id
            receipt = json.loads((folder/"complete.json").read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or sha(folder/"records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed residual pair")
            rows.extend(json.loads((folder/"records.json").read_text("utf-8")))
        if len(rows) != lock["expected_records"]:
            raise AssertionError("Incomplete locked residual cohort")
        write_json(OUT/"records.json",rows)
        write_json(OUT/"complete.json",dict(status="complete",pairs=256,records=len(rows),records_sha256=sha(OUT/"records.json"),lock_sha256=sha(LOCK)))
        return
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Invalid residual worker assignment")
    codec = learned.HyperpriorCodec(ROOT/"resources/learned_codec",device="cuda",threads=2,entropy_bridge=True)
    assets = codec.prepare()
    write_json(OUT/f"worker_{args.worker_index}_assets.json",assets)
    galleries = {}
    for h in lock["heldouts"]:
        item = lock["galleries"][str(h)]
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed locked known gallery")
        with np.load(ROOT/item["path"],allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"],device=codec.device)
    assigned = lock["pair_ids"][args.worker_index::args.workers]
    for index,pair_id in enumerate(assigned):
        folder = OUT/"pairs"/pair_id
        while not (folder/"sender_inputs_ready.json").exists():
            time.sleep(10)
        for relative,expected in lock["source_sha256"].items():
            if sha(ROOT/relative) != expected:
                raise ValueError("Locked residual implementation changed")
        if (folder/"complete.json").exists():
            receipt = json.loads((folder/"complete.json").read_text("utf-8"))
            if receipt["lock_sha256"] != sha(LOCK) or receipt["records_sha256"] != sha(folder/"records.json"):
                raise ValueError("Changed completed locked pair")
        else:
            manifest = json.loads((DATA/"manifest.json").read_text("utf-8"))
            pair = next(p for p in manifest["pairs"] if p["id"] == pair_id)
            rows = encode_pair(pair,lock,codec,assets,galleries)
            write_json(folder/"records.json",rows)
            write_json(folder/"complete.json",dict(status="complete",records=117,lock_sha256=sha(LOCK),
                records_sha256=sha(folder/"records.json"),sender_commit_sha256=sha(folder/"sender_commit.json"),labels_read=False))
        write_json(OUT/f"worker_{args.worker_index}_progress.json",dict(completed_pairs=index+1,expected_pairs=len(assigned)))
        print(f"locked residual worker{args.worker_index}: {index+1}/{len(assigned)}",flush=True)
    write_json(OUT/f"worker_{args.worker_index}_complete.json",dict(status="complete",pairs=len(assigned)))


if __name__ == "__main__":
    main()
