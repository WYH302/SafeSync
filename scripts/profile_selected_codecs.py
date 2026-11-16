"""Measure one codec call at the operating point chosen by the prior search.

This distinguishes intrinsic codec-call cost from the multiscale rate search;
it does not replace the measured complete input-to-packet pipeline.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from safesync.full_image_codecs import encode_payload,HEAD
from scripts.run_layered_sequences import pad


def main():
    folder=ROOT/"results/sender_profile/layered_v1"
    manifest=json.loads((ROOT/"data/spacenet7_pilot/development_manifest_v1.json").read_text())
    aoi=manifest["aois"][0]
    results=[]
    for method in ("full_jpeg420","full_jpeg2000","full_webp"):
        source=folder/(method+".json")
        profile=json.loads(source.read_text())
        observations=[]
        for row in profile["records"]:
            month=aoi["months"][row["step"]]
            current=pad(np.array(Image.open(ROOT/month["rgb_path"])))
            meta=row["encoder_meta"]
            ch,cw,q=meta["coded_height"],meta["coded_width"],meta["quality"]
            started=time.perf_counter()
            image=Image.fromarray(current).resize((cw,ch),Image.Resampling.LANCZOS)
            resize_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            payload=encode_payload(image,method.removeprefix("full_"),q)
            encode_ms=(time.perf_counter()-started)*1000
            if len(payload)+HEAD.size+16!=row["down_bytes"]:
                raise ValueError("Single-call payload differs from the recorded search operating point")
            observations.append(dict(step=row["step"],month=month["month"],quality=q,
                coded_shape=[ch,cw],payload_bytes=len(payload),resize_ms=resize_ms,
                encode_call_ms=encode_ms,search_encoder_calls=meta["codec_trials"]))
        results.append(dict(method=method,source_profile_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            records=observations,
            encode_mean_ms=float(np.mean([r["encode_call_ms"] for r in observations])),
            encode_median_ms=float(np.median([r["encode_call_ms"] for r in observations])),
            encode_p95_ms=float(np.percentile([r["encode_call_ms"] for r in observations],95)),
            resize_mean_ms=float(np.mean([r["resize_ms"] for r in observations])),
            mean_search_calls=float(np.mean([r["search_encoder_calls"] for r in observations]))))
    value=dict(status="complete",scope="same twelve original calibration updates; chosen settings only",
        distinction="single encode call excludes rate/resolution search, source read, MSE evaluation, protocol and decode",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=results)
    (folder/"selected_codec_calls.json").write_text(json.dumps(value,indent=2),encoding="utf-8")
    for row in results:print(row["method"],row["encode_mean_ms"],row["mean_search_calls"],flush=True)


if __name__=="__main__":main()
