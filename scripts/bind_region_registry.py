"""Correct frame registry binding without changing scientific content or sizes.

Only identity/digest/checksum bytes change. All decoded codes, statuses and image
payload bytes are required to remain identical. Original v1 runs stay archived.
"""
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from safesync.image_wire import IMAGE_HEADER
from safesync.wire import ANCHOR_BASE,CRC,HEADER,State,receive,signature

ROOT=Path(__file__).resolve().parents[1]


def main():
    changes=[]
    for relative in ('results/mvp','results/repair1_jpeg80','results/repair2_reserve10'):
        folder=ROOT/relative
        env=json.loads((folder/'environment.json').read_text(encoding='utf-8'))
        model=json.dumps(env['identity'],sort_keys=True)
        revised=0
        for initial_path in (folder/'bitstreams/initial').glob('*.bin'):
            pair_id=initial_path.stem
            old=initial_path.read_bytes()
            old_header=list(HEADER.unpack_from(old))
            expected=signature(model+f";pair={pair_id};native={(512,512)}")
            old_state=receive(State(np.zeros((256,384),np.int8),16,old_header[-1]),old)
            new_digest=hashlib.sha256(expected+old_state.codes.tobytes()).digest()
            if old_header[-1]!=expected:
                old_header[-1]=expected
                frame=HEADER.pack(*old_header)+old[HEADER.size:-CRC.size]
                initial_path.write_bytes(frame+CRC.pack(zlib.crc32(frame)))
                revised+=1
            new_state=receive(State(np.zeros((256,384),np.int8),16,expected),initial_path.read_bytes())
            assert np.array_equal(old_state.codes,new_state.codes)
            for subdir in ('h5','h3','h4'):
                for path in (folder/'bitstreams'/subdir).glob(f'{pair_id}_*.bin'):
                    packet=path.read_bytes()
                    if packet.startswith(b'SS01'):
                        fields=list(HEADER.unpack_from(packet))
                        if fields[-1]==expected:
                            continue
                        before=receive(old_state,packet)
                        fields[-1]=expected
                        fields[-2]=bytes(32) if fields[5]==ANCHOR_BASE else new_digest
                        framed=HEADER.pack(*fields)+packet[HEADER.size:-CRC.size]
                        updated=framed+CRC.pack(zlib.crc32(framed))
                        after=receive(new_state,updated)
                        assert np.array_equal(before.codes,after.codes)
                        assert np.array_equal(before.statuses,after.statuses)
                        assert before.rois==after.rois
                    elif packet.startswith(b'SI01'):
                        fields=list(IMAGE_HEADER.unpack_from(packet))
                        if fields[-1]==expected:
                            continue
                        fields[-1],fields[-2]=expected,new_digest
                        framed=IMAGE_HEADER.pack(*fields)+packet[IMAGE_HEADER.size:-CRC.size]
                        updated=framed+CRC.pack(zlib.crc32(framed))
                        assert packet[IMAGE_HEADER.size:-CRC.size]==updated[IMAGE_HEADER.size:-CRC.size]
                    else:
                        raise ValueError('unexpected frame')
                    assert len(updated)==len(packet)
                    path.write_bytes(updated)
                    revised+=1
        changes.append({'variant':relative,'registry_corrected_frames':revised,'decoded_tokens_statuses_rois_unchanged':True,'wire_lengths_unchanged':True})
    (ROOT/'results/verification/registry_correction.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
    print(json.dumps(changes,indent=2))


if __name__=='__main__':
    main()
