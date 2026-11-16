"""Reconstruct one tile using only initialization frame, update and registry.

This program never reads current sender features or original current images.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from safesync.wire import HEADER, State, receive


def replay(initial_packet,update_packet,grid=16,dimension=384,expected_identity=None):
    transmitted_identity=HEADER.unpack_from(initial_packet)[-1]
    if expected_identity is not None and transmitted_identity!=expected_identity:
        raise ValueError('initial frame does not match registered region/model identity')
    initial=State(np.zeros((grid**2,dimension),np.int8),grid,transmitted_identity if expected_identity is None else expected_identity)
    previous=receive(initial,initial_packet)
    return receive(previous,update_packet)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initial',required=True)
    parser.add_argument('--packet',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--identity-hex',required=True,help='registered identity, including original pair ID and model/codec')
    args=parser.parse_args()
    state=replay(Path(args.initial).read_bytes(),Path(args.packet).read_bytes(),expected_identity=bytes.fromhex(args.identity_hex))
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,codes=state.codes,features=state.features,statuses=state.statuses,age=state.age)
    print(json.dumps({'version':state.version,'digest':state.digest.hex(),'shape':list(state.codes.shape),'roi_count':len(state.rois),'output':str(target)},indent=2))


if __name__=='__main__':
    main()
