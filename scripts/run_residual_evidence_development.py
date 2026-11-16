"""Controlled learned-base enhancement experiment on32 inspected source pairs."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"results/layered_tasks/.deps"))
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from safesync import learned_codec as learned
from safesync import residual_evidence as enhancement
from safesync import task_evidence_wire as wire
from safesync.pixel_wire import PixelState
from scripts.run_learned_codec import cached_candidates
from scripts.run_locked_second_test import sha,write_json
from scripts.run_task_evidence_selector import quality

MAIN = ROOT/"results/task_evidence/fresh_context_v1"
CONTROL = ROOT/"results/task_evidence/fresh_context_learned_v1"
OUT = ROOT/"results/task_evidence/residual_development_v1"
CAPS = (4096,8192,16384,32768)
SEEDS = (1729,2718,31415)


def pool(values):
    return np.asarray(values).reshape(8,2,8,2).mean((1,3)).reshape(-1)


def fit(prior,labels,current,base_packet,base_rgb,ranking,mode,cap):
    order = np.argsort(-ranking,kind="stable")

    def make(count):
        image = enhancement.encode(base_packet,current,base_rgb,order[:count],mode=mode)
        return wire.encode(prior,labels,observation=1,payload=image,image_mode=wire.LEARNED)[0]

    best = make(0)
    if len(best)+wire.ACK_BYTES > cap:
        raise ValueError("Learned base and task state exceed the total cap")
    selected,low,high = 0,1,64
    while low <= high:
        middle = (low+high)//2
        packet = make(middle)
        if len(packet)+wire.ACK_BYTES <= cap:
            best,selected,low = packet,middle,middle+1
        else:
            high = middle-1
    for count in range(selected+1,min(selected+3,64)+1):
        packet = make(count)
        if len(packet)+wire.ACK_BYTES <= cap:
            best,selected = packet,count
    return best,selected


def main():
    lock = json.loads((MAIN/"method_lock.json").read_text("utf-8"))
    previous = ROOT/"results/task_evidence/hyperprior_risk_development_v1/run_spec.json"
    ids = json.loads(previous.read_text("utf-8"))["pair_ids"][:32]
    manifest = json.loads((ROOT/"data/second_task_context_test_v1/manifest.json").read_text("utf-8"))
    pairs = {p["id"]:p for p in manifest["pairs"]}
    names = ("scripts/run_residual_evidence_development.py","safesync/residual_evidence.py",
        "safesync/task_evidence_wire.py","safesync/learned_codec.py","scripts/run_task_evidence_selector.py")
    proposed = dict(pair_ids=ids,caps=CAPS,heldouts=(5,3,4),seeds=SEEDS,expected_records=3072,
        source_sha256={p:sha(ROOT/p) for p in names},prior_score_spec_sha256=sha(previous),
        source_lock_sha256=sha(MAIN/"method_lock.json"),base_fraction=.5,quality=70,
        factorial="Absolute/residual atlas x change/current-distance mixture. Each compares with the same full-cap hyperprior incumbent using the same bounded quality proxy. Three residual-random controls.",
        residual="round((current-decoded_learned_base)/2+128) uint8; decode clipped(base+2*(decoded-128)); native WebP70 atlas",
        task="The same exact current task array and actual TE01 envelope/60-byte ACK for every method",
        scope="Exploratory32 previously inspected image pairs, after failed fresh joint confirmation and score-only diagnosis",
        hypothesis="The global hyperprior quality/scale grid offers too little spatial allocation freedom. Residual-coded native regions may improve the evidence/later-use frontier relative to absolute region coding.",
        success="Residual benefit over matched absolute coding, plus an unknown-score increment over matched change/random, actual task identity and later-use evaluation. Retain allclass/cap outcomes.")
    proposed = json.loads(json.dumps(proposed))
    OUT.mkdir(parents=True,exist_ok=True)
    spec_path = OUT/"run_spec.json"
    if spec_path.exists():
        stored = json.loads(spec_path.read_text("utf-8"))
        if {k:v for k,v in stored.items() if k != "locked_utc"} != proposed:
            raise ValueError("Changed residual-evidence development experiment")
    else:
        proposed["locked_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(spec_path,proposed)
        for name in names:
            target = OUT/"source_snapshot"/name.replace("/","__")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    codec = learned.HyperpriorCodec(ROOT/"resources/learned_codec",device="cuda",threads=2,entropy_bridge=True)
    write_json(OUT/"assets.json",codec.prepare())
    galleries = {}
    for h in (5,3,4):
        item = lock["galleries"][str(h)]
        if sha(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Changed known-only gallery")
        with np.load(ROOT/item["path"],allow_pickle=False) as stored:
            galleries[h] = torch.tensor(stored["features"],device=codec.device)
    all_rows = []
    for pair_id in ids:
        folder = OUT/"pairs"/pair_id
        if (folder/"complete.json").exists():
            receipt = json.loads((folder/"complete.json").read_text("utf-8"))
            if receipt["spec_sha256"] != sha(spec_path) or sha(folder/"records.json") != receipt["records_sha256"]:
                raise ValueError("Changed completed residual source pair")
            all_rows.extend(json.loads((folder/"records.json").read_text("utf-8")))
            continue
        folder.mkdir(parents=True,exist_ok=True)
        source = MAIN/"pairs"/pair_id
        ready = json.loads((source/"sender_inputs_ready.json").read_text("utf-8"))
        if sha(source/"sender_features.npz") != ready["features_sha256"]:
            raise ValueError("Changed acknowledged feature history")
        after = pairs[pair_id]["files"]["after"]
        if sha(ROOT/after["path"]) != after["sha256"]:
            raise ValueError("Changed current image")
        current = np.asarray(Image.open(ROOT/after["path"]).convert("RGB"))
        with np.load(source/"sender_features.npz",allow_pickle=False) as stored:
            f0,f1 = stored["before"].astype(np.float32),stored["current"].astype(np.float32)
        change = rankdata(pool(1-np.sum(f0*f1,axis=1)))/64
        query = torch.nn.functional.normalize(torch.tensor(f1,device=codec.device),dim=1)
        saved_folder = CONTROL/"pairs"/pair_id/"candidates"
        saved = json.loads((saved_folder/"candidates.json").read_text("utf-8"))
        if saved["identity"]["source_rgb_sha256"] != hashlib.sha256(current.tobytes()).hexdigest():
            raise ValueError("Different learned candidate input")
        candidates,_,_ = cached_candidates(codec,current,saved_folder,saved["identity"])
        decoded = {}
        rows = []
        for h in (5,3,4):
            common = ready["common"][str(h)]
            boot_path,task_path = ROOT/common["bootstrap_path"],ROOT/common["labels_only_path"]
            if sha(boot_path) != common["bootstrap_sha256"] or sha(task_path) != common["labels_only_sha256"]:
                raise ValueError("Changed common service packets")
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
                    raise ValueError("Residual learned-base reference mismatch")
                identity = hashlib.sha256(payload).hexdigest()
                if identity not in decoded:
                    decoded[identity] = codec.decode_packet(payload)[0]
                return decoded[identity]

            def decode_image(base,payload):
                return (enhancement.decode(payload,base.rgb.shape[:2],decode_base)
                    if payload[:4] == b"RE01" else decode_base(payload))

            empty,_ = wire.encode(prior,service.labels,observation=1)
            for cap in CAPS:
                full_inner,_ = learned.select(candidates,budget=cap-len(empty)-wire.ACK_BYTES,
                    region=prior.region,base_version=prior.version,base_digest=pixel_prior.digest(),observation=1)
                full,_ = wire.encode(prior,service.labels,observation=1,payload=full_inner,image_mode=wire.LEARNED)
                full_state,full_ack = wire.receive(prior,full,learned_decoder=decode_image)
                if full != (CONTROL/"pairs"/pair_id/f"h{h}_b{cap}.bin").read_bytes():
                    raise AssertionError("Strong incumbent changed")
                base_inner,_ = learned.select(candidates,budget=cap//2-len(empty)-wire.ACK_BYTES,
                    region=prior.region,base_version=prior.version,base_digest=pixel_prior.digest(),observation=1)
                base_rgb = decode_base(base_inner)
                variants = [("hyperprior",0,None,None)]
                variants += [(f"{name}_{route}",0,mode,ranking) for name,mode in (("absolute",enhancement.ABSOLUTE),("residual",enhancement.RESIDUAL)) for route,ranking in (("change",change),("current",mixed))]
                for seed in SEEDS:
                    random = np.random.default_rng(seed+int(pair_id)*1009).random(64)
                    variants.append(("residual_random",seed,enhancement.RESIDUAL,random))
                for method,seed,mode,ranking in variants:
                    packet,state,ack = full,full_state,full_ack
                    refined,used_enhancement = 0,False
                    if ranking is not None:
                        candidate,refined = fit(prior,service.labels,current,base_inner,base_rgb,ranking,mode,cap)
                        candidate_state,candidate_ack = wire.receive(prior,candidate,learned_decoder=decode_image)
                        if quality(current,candidate_state.rgb,ranking) > quality(current,full_state.rgb,ranking):
                            packet,state,ack = candidate,candidate_state,candidate_ack
                            used_enhancement = True
                    if len(packet)+len(ack) > cap or not np.array_equal(state.labels,service.labels):
                        raise AssertionError("Residual task state or real-byte cap")
                    path = folder/f"h{h}_{method}_s{seed}_b{cap}.bin"
                    path.write_bytes(packet)
                    path.with_suffix(".ack").write_bytes(ack)
                    png = path.with_suffix(".received.png")
                    Image.fromarray(state.rgb).save(png)
                    rows.append(dict(pair_id=pair_id,heldout=h,method=method,seed=seed,budget=cap,
                        down_bytes=len(packet),ack_bytes=len(ack),total_bytes=len(packet)+len(ack),bootstrap_bytes=common["bootstrap_bytes"],
                        bootstrap_path=common["bootstrap_path"],packet_path=path.relative_to(ROOT).as_posix(),packet_sha256=sha(path),
                        receiver_rgb_sha256=hashlib.sha256(state.rgb.tobytes()).hexdigest(),receiver_task_sha256=hashlib.sha256(state.labels.tobytes()).hexdigest(),
                        received_png=png.relative_to(ROOT).as_posix(),received_png_sha256=sha(png),
                        refined_rois=refined,enhancement_selected=used_enhancement))
        if len(rows) != 96:
            raise AssertionError("Incomplete residual factorial source pair")
        write_json(folder/"sender_commit.json",rows)
        write_json(folder/"records.json",rows)
        write_json(folder/"complete.json",dict(status="complete",records=96,spec_sha256=sha(spec_path),records_sha256=sha(folder/"records.json"),
            sender_commit_sha256=sha(folder/"sender_commit.json"),labels_read=False))
        all_rows.extend(rows)
        write_json(OUT/"progress.json",dict(completed_pairs=len(all_rows)//96,expected_pairs=32,records=len(all_rows)))
        print(f"absolute/residual learned-base factorial {len(all_rows)//96}/32",flush=True)
    write_json(OUT/"records.json",all_rows)
    write_json(OUT/"complete.json",dict(status="complete",pairs=32,records=3072,spec_sha256=sha(spec_path),records_sha256=sha(OUT/"records.json")))


if __name__ == "__main__":
    main()
