"""Use the unchanged author-fold ground interface on actual TC10 decoder PNGs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_layered_receipts as common
from scripts import evaluate_learned_receipts as lic
from resources.pretrained_ground_crossfit.run_receipts import probability, measure_frame
from safesync.crossfit_ground import CrossfitGroundEvaluator
from safesync.pixel_wire import ACK, checksum
import numpy as np
from PIL import Image

SOURCE = ROOT / "results/tcm_sn7/transport_v1"
OUT = ROOT / "results/tcm_sn7/ground_v1"
MANIFEST = ROOT / "data/spacenet7_pilot/confirmation_manifest_v1.json"


def main():
    lic.require_no_codec_imports()
    lock = common.read(SOURCE / "method_lock.json")
    if common.sha(Path(__file__)) != lock["source_sha256"]["scripts/evaluate_sn7_tcm_control.py"]:
        raise ValueError("Changed monthly TCM ground evaluator")
    calibration = ROOT / "resources/pretrained_ground_crossfit/calibration.json"
    spec = dict(control_lock_sha256=common.sha(SOURCE / "method_lock.json"), source_sha256=common.sha(Path(__file__)),
        calibration_sha256=common.sha(calibration), fold_selection_sha256=common.sha(ROOT / "resources/pretrained_ground_crossfit/fold_selection.json"),
        thresholds=dict(building=.5, event=common.read(calibration)["change_threshold"]), expected_records=534,
        scope="Same author validation-fold-best evaluator and original two-AOI threshold. No new model fitting. Existing validation-best selection limitation remains.")
    path = OUT / "evaluation_lock.json"
    if path.exists() and common.read(path) != spec:
        raise ValueError("Changed monthly TCM ground specification")
    common.save_json(path, spec)
    manifest = common.read(MANIFEST)
    all_rows = []
    for index, aoi in enumerate(manifest["aois"]):
        folder = OUT / f"area_{index}"
        evaluator = CrossfitGroundEvaluator(ROOT, aoi["aoi_id"])
        lic.assert_ground_flags(lic.numerical_flags(evaluator.torch))
        model_hash = common.canonical_hash(json.loads(evaluator.identity))
        threshold_hash = common.canonical_hash(spec["thresholds"])
        first = aoi["months"][0]
        initial_path = SOURCE / "initial" / aoi["aoi_id"] / "decoded.png"
        while not initial_path.exists():
            time.sleep(15)
        initial_rgb = np.asarray(Image.open(initial_path).convert("RGB"))
        reference_key = common.canonical_hash(dict(native_initial_RGB=common.sha_bytes(initial_rgb.tobytes()), model=model_hash, threshold=threshold_hash))
        reference, _ = probability(evaluator, initial_rgb, folder, model_hash, threshold_hash, dict(bootstrap=common.relative(initial_path)))
        rows, cumulative = [], {cap:None for cap in lock["caps"]}
        try:
            for step, month in enumerate(aoi["months"][1:], 1):
                source = SOURCE / f"area_{index}/step_{step:03d}"
                while not (source / "complete.json").exists():
                    time.sleep(15)
                receipt = common.read(source / "complete.json")
                if receipt["control_lock_sha256"] != spec["control_lock_sha256"] or receipt["records_sha256"] != common.sha(source / "records.json"):
                    raise ValueError("Changed actual monthly transaction source")
                event = ROOT / f"results/layered/confirmation_v1/area_{index}/gt" / aoi["aoi_id"] / month["month"] / "new_track.png"
                gt = common.offline_masks(month, first, aoi["aoi_id"], event, True)
                for row in common.read(source / "records.json"):
                    cap = row["cap"]
                    png, packet = ROOT / row["decoded_path"], ROOT / row["packet_path"]
                    if common.sha(png) != row["decoded_png_sha256"] or common.sha(packet) != row["packet_sha256"]:
                        raise ValueError("Changed actual monthly decoded PNG or packet")
                    native = np.asarray(Image.open(png).convert("RGB"))
                    if common.sha_bytes(native.tobytes()) != row["decoded_native_rgb_sha256"]:
                        raise ValueError("Changed monthly decoder pixels")
                    ack = packet.with_suffix(".ack").read_bytes()
                    if (len(ack) != 60 or common.sha_bytes(ack) != row["ack_sha256"] or checksum(ack[:-16]) != ack[-16:]
                            or ACK.unpack_from(ack)[3] != checksum(packet.read_bytes())):
                        raise ValueError("Changed monthly feedback")
                    initial = dict(bootstrap_path=common.relative(initial_path), bootstrap_down_bytes=row["initial_down_bytes"], bootstrap_ack_bytes=row["initial_ack_bytes"])
                    if cumulative[cap] is None:
                        cumulative[cap] = [row["initial_down_bytes"], row["initial_ack_bytes"]]
                    cumulative[cap][0] += row["down_bytes"]
                    cumulative[cap][1] += row["up_bytes"]
                    wire = dict(step=step, month=month["month"], packet_path=row["packet_path"], packet_sha256=row["packet_sha256"],
                        down_bytes=row["down_bytes"], ack_bytes=row["up_bytes"], cumulative_down_bytes=cumulative[cap][0], cumulative_up_bytes=cumulative[cap][1],
                        receiver_RGB_sha256_verified=row["receiver_rgb_sha256"], protocol="SP03/TC10", source_encoding_ms=row["sender_search_ms_shared"])
                    rows.append(measure_frame(evaluator, native, wire, initial, reference, reference_key, folder, model_hash,
                        threshold_hash, "tcm", cap, aoi, gt, common.sha(path)))
                common.write_csv(folder / "records.csv", rows)
                print(f"SN7 TCM ground area{index} update{step}/{len(aoi['months'])-1}", flush=True)
            common.save_json(folder / "complete.json", dict(status="complete", records=len(rows), records_sha256=common.sha(folder / "records.csv"), evaluation_lock_sha256=common.sha(path)))
            all_rows.extend(rows)
        finally:
            evaluator.close()
    if len(all_rows) != 534:
        raise ValueError("Incomplete monthly TCM task matrix")
    common.write_csv(OUT / "records.csv", all_rows)
    common.save_json(OUT / "complete.json", dict(status="complete", records=534, aois=8, records_sha256=common.sha(OUT / "records.csv"), evaluation_lock_sha256=common.sha(path)))


if __name__ == "__main__":
    main()
