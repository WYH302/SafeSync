"""Deterministic numerical/protocol checks; these are not EO field results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from safesync.scoring import normalize
from safesync.wire import ReferenceMismatch, State, encode, quantize, receive, signature

ROOT = Path(__file__).resolve().parents[1]


def delta_numeric():
    rng = np.random.default_rng(20261028)
    sequence = rng.normal(0,.03,(1000,16)).cumsum(axis=0)
    state, wrong_receiver, original_reference = np.zeros(16), np.zeros(16), np.zeros(16)
    rows = []
    for index, feature in enumerate(sequence):
        state += np.round((feature-state)/.01)*.01
        wrong_receiver += np.round((feature-original_reference)/.01)*.01
        original_reference = feature
        rows.append({"step":index+1,"closed_loop_max_abs_error":float(np.max(np.abs(state-feature))),"inconsistent_reference_max_abs_error":float(np.max(np.abs(wrong_receiver-feature)))})
    return {"quantization_step":.01,"theoretical_half_step_bound":.005,"observed_closed_loop_max_error":max(x["closed_loop_max_abs_error"] for x in rows),"wrong_reference_is_a_diagnostic_not_a_valid_baseline":True,"series":rows}


def protocol_simulation(policy, loss=.1, seed=41, period=8):
    rng = np.random.default_rng(seed)
    sequence = normalize(rng.normal(size=(48,16,8)).cumsum(axis=0))
    identity = signature("synthetic-protocol-v1")
    sender = State(quantize(sequence[0]),4,identity)
    receiver = State(sender.codes,4,identity)
    nack, forward_bytes, reverse_bytes, anchors, rejected = False, 0, 0, 0, 0
    rows = []
    for step, feature in enumerate(sequence[1:],1):
        anchor = (policy=="periodic" and step%period==0) or (policy=="risk" and nack)
        packet,_ = encode(sender,quantize(feature),list(range(16)),{},np.zeros(16,bool),anchor=anchor)
        forward_bytes += len(packet)
        anchors += int(anchor)
        if rng.random() < loss:
            receiver = receive(receiver,None)
            nack = True  # reliable timeout/feedback is an explicit simulation assumption
        else:
            try:
                receiver = receive(receiver,packet)
                nack = False
            except ReferenceMismatch:
                rejected += 1
                receiver = receive(receiver,None)
                nack = True
        reverse_bytes += 40  # sequence/version + digest + status, every opportunity
        # Sender progression is provisional; subsequent base mismatch is detected.
        sender = receive(sender,packet)
        rows.append({"step":step,"sync_mse":float(np.mean((sender.features-receiver.features)**2)),"world_mse":float(np.mean((feature-receiver.features)**2)),"stale_units":int(np.sum(receiver.statuses==0)),"anchor":anchor,"forward_bytes_cumulative":forward_bytes,"reverse_bytes_cumulative":reverse_bytes})
    return {"policy":policy,"loss_probability":loss,"loss_seed":seed,"period":period if policy=="periodic" else None,"anchors":anchors,"rejected_mismatches":rejected,"forward_bytes":forward_bytes,"reverse_bytes":reverse_bytes,"mean_sync_mse":float(np.mean([x['sync_mse'] for x in rows])),"mean_world_mse":float(np.mean([x['world_mse'] for x in rows])),"series":rows,"scope":"synthetic protocol mechanics; reliable immediate NACK/timeout feedback; no EO long-horizon or anchor superiority claim"}


def main():
    destination=ROOT/"results/protocol"
    destination.mkdir(parents=True,exist_ok=True)
    result={"delta":delta_numeric(),"simulations":[protocol_simulation(p,loss,seed,period) for seed in (41,42,43) for loss in (0.,.1) for p,period in (("no_anchor",8),("periodic",4),("periodic",8),("periodic",16),("risk",8))]}
    (destination/"diagnostics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"closed_loop_max_error":result['delta']['observed_closed_loop_max_error'],"simulations":len(result['simulations']),"output":str(destination/'diagnostics.json')},indent=2))


if __name__ == "__main__":
    main()
