"""Verify pixel-free encoding/ACK commitment against retained actual transactions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from safesync import task_evidence_wire as wire
from safesync.sender_reference import SenderReference

SOURCE = ROOT/"results/task_evidence/residual_development_v1"
COMMON = ROOT/"results/task_evidence/fresh_context_v1"
OUT = ROOT/"results/task_evidence/sender_reference_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_receipt = json.loads((SOURCE/"complete.json").read_text("utf-8"))
    if source_receipt["records_sha256"] != sha(SOURCE/"records.json"):
        raise ValueError("Actual transaction source changed")
    rows = json.loads((SOURCE/"records.json").read_text("utf-8"))
    priors,labels,memories = {},{},{}
    verified = []
    for row in rows:
        pair_id,h = row["pair_id"],row["heldout"]
        key = pair_id,h
        if key not in priors:
            boot_path = ROOT/row["bootstrap_path"]
            boot = boot_path.read_bytes()
            head = wire.HEADER.unpack_from(boot)
            blank = wire.TaskEvidenceState(np.zeros((512,512,3),np.uint8),np.zeros((16,16),np.uint8),head[1],head[2])
            prior,_ = wire.receive(blank,boot)
            priors[key] = SenderReference.from_decoded_state(prior)
            task_packet = COMMON/"pairs"/pair_id/f"h{h}_labels_only.bin"
            labels[key] = wire.receive(prior,task_packet.read_bytes())[0].labels
            if pair_id not in memories:
                with np.load(COMMON/"pairs"/pair_id/"sender_features.npz",allow_pickle=False) as stored:
                    feature_bytes = stored["before"].astype(np.float16).nbytes
                size = len(priors[key].serialize())
                memories[pair_id] = dict(pair_id=pair_id,raw_rgb_bytes=prior.rgb.nbytes,actual_png_bytes=head[16],
                    serialized_reference_bytes=size,common_fp16_observation_feature_bytes=feature_bytes,
                    compact_plus_features_bytes=size+feature_bytes,png_plus_features_bytes=head[16]+feature_bytes)
        reference = SenderReference.deserialize(priors[key].serialize())
        path = ROOT/row["packet_path"]
        png_path = ROOT/row["received_png"]
        if sha(path) != row["packet_sha256"] or sha(png_path) != row["received_png_sha256"]:
            raise ValueError("Retained actual packet or registered-decoder PNG changed")
        packet = path.read_bytes()
        head = wire.HEADER.unpack_from(packet)
        offset = wire.HEADER.size+head[14]+head[11]*2+head[15]
        indices = np.frombuffer(packet,dtype="<u2",count=head[11],offset=wire.HEADER.size+head[14])
        compact,_ = reference.encode(labels[key],observation=head[5],payload=packet[offset:-16],image_mode=head[13],indices=indices)
        if compact != packet:
            raise AssertionError("Pixel-free sender changed actual transmitted bytes")
        rgb = np.asarray(Image.open(png_path).convert("RGB"))
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row["receiver_rgb_sha256"]:
            raise ValueError("Actual registered-decoder RGB changed")
        ack = path.with_suffix(".ack").read_bytes()
        advanced = reference.acknowledged(compact,ack,rgb,labels[key])
        if advanced.version != reference.version+1 or reference.version != priors[key].version:
            raise AssertionError("Reference committed before valid acknowledgment")
        verified.append(dict(pair_id=pair_id,heldout=h,budget=row["budget"],method=row["method"],seed=row["seed"],
            packet_sha256=row["packet_sha256"],serialized_before_bytes=len(reference.serialize()),
            serialized_after_bytes=len(advanced.serialize()),target_digest=advanced.digest().hex()))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,group in (("verified",verified),("memory",list(memories.values()))):
        with (OUT/(name+".csv")).open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(group[0]))
            writer.writeheader()
            writer.writerows(group)
    values = list(memories.values())
    result = dict(status="passed",pairs=len(values),actual_transactions=len(verified),
        serialized_reference_bytes=sorted({r["serialized_reference_bytes"] for r in values}),
        mean_actual_png_bytes=float(np.mean([r["actual_png_bytes"] for r in values])),
        mean_combined_history_reduction_vs_png=float(np.mean([1-r["compact_plus_features_bytes"]/r["png_plus_features_bytes"] for r in values])),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),ROOT/"safesync/sender_reference.py",SOURCE/"complete.json",SOURCE/"records.json")},
        verified_sha256=sha(OUT/"verified.csv"),memory_sha256=sha(OUT/"memory.csv"),
        scope="Current full-frame payload modes, identical real bytes and valid ACKs. Observed-source FP16 features remain a separate common sender cache. PNG comparator omits its task/manifest metadata, giving a conservative storage comparison. Model/gallery memory, active current images, decoder workspace and pending packets are separate.")
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n","utf-8")
    print(json.dumps({k:result[k] for k in ("status","pairs","actual_transactions","serialized_reference_bytes","mean_actual_png_bytes","mean_combined_history_reduction_vs_png")},indent=2),flush=True)


if __name__ == "__main__":
    main()
