"""Compact references preserve actual transactions and ACK-gated commitment."""
import io
import unittest

import numpy as np
from PIL import Image

from safesync import task_evidence_wire as wire
from safesync.sender_reference import SenderReference


def png(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream,format="PNG")
    return stream.getvalue()


class SenderReferenceTests(unittest.TestCase):
    def test_multiple_updates_match_full_state_bytes(self):
        state = wire.TaskEvidenceState(np.zeros((128,128,3),np.uint8),np.zeros((4,4),np.uint8),b"r"*16,b"m"*16)
        reference = SenderReference.from_decoded_state(state)
        rng = np.random.default_rng(19)
        for observation in range(5):
            current = rng.integers(0,256,(128,128,3),dtype=np.uint8)
            labels = rng.integers(1,6,(4,4),dtype=np.uint8)
            arguments = dict(observation=observation,payload=png(current),image_mode=wire.FULL)
            full,_ = wire.encode(state,labels,**arguments)
            compact,_ = reference.encode(labels,**arguments)
            self.assertEqual(full,compact)
            received,ack = wire.receive(state,full)
            next_reference = reference.acknowledged(compact,ack,current,labels)
            self.assertEqual(reference.version,state.version)
            self.assertEqual(next_reference.digest(),received.digest())
            reference = SenderReference.deserialize(next_reference.serialize())
            state = received
        self.assertFalse(isinstance(reference.rgb,np.ndarray))

    def test_missing_or_incorrect_ack_never_commits(self):
        state = wire.TaskEvidenceState(np.zeros((128,128,3),np.uint8),np.ones((4,4),np.uint8),b"r"*16,b"m"*16)
        reference = SenderReference.from_decoded_state(state)
        current = np.full_like(state.rgb,90)
        packet,_ = reference.encode(state.labels,observation=0,payload=png(current),image_mode=wire.FULL)
        _,ack = wire.receive(state,packet)
        for wrong in (b"",ack[:-1],ack[:-1]+bytes([ack[-1]^1])):
            with self.assertRaises(ValueError):
                reference.acknowledged(packet,wrong,current,state.labels)
        with self.assertRaises(ValueError):
            reference.acknowledged(packet,ack,np.zeros_like(current),state.labels)
        self.assertEqual(reference.digest(),state.digest())

    def test_current_frame_contract_and_storage_integrity(self):
        state = wire.TaskEvidenceState(np.zeros((512,512,3),np.uint8),np.ones((16,16),np.uint8),b"r"*16,b"m"*16)
        reference = SenderReference.from_decoded_state(state)
        self.assertEqual(len(reference.serialize()),424)
        state.labels[:] = 2
        self.assertTrue(np.all(reference.labels == 1))
        with self.assertRaises(ValueError):
            reference.labels[:] = 3
        with self.assertRaises(ValueError):
            reference.encode(state.labels,observation=0,payload=b"",image_mode=wire.ROI)
        with self.assertRaises(ValueError):
            SenderReference.deserialize(reference.serialize()[:-1])


if __name__ == "__main__":
    unittest.main()
