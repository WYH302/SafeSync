"""Isolated CPU stage measurements on original calibration imagery.

Run only when transmission-generation jobs are idle. Timings distinguish sender
work from the simulated receiver; full-codec search is not one encoder call.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ["gaussian_20261029", "dct_bound", "full_cache_mse",
           "full_jpeg420", "full_jpeg2000", "full_webp"]

def memory_info():
    import ctypes
    from ctypes import wintypes
    size = ctypes.c_size_t
    class Counters(ctypes.Structure):
        _fields_ = [("cb",wintypes.DWORD),("page_faults",wintypes.DWORD),
                    ("peak_rss",size),("rss",size),("peak_paged_pool",size),
                    ("paged_pool",size),("peak_nonpaged_pool",size),
                    ("nonpaged_pool",size),("pagefile",size),
                    ("peak_pagefile",size),("private_bytes",size)]
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb):
        raise ctypes.WinError()
    return {name:int(getattr(info,name)) for name,_ in info._fields_ if name!="cb"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def child(method, budget, updates, out):
    import platform
    import numpy as np
    from PIL import Image
    from safesync.layered_cache import DetailCache
    import safesync.layered_wire as lw
    import scripts.run_pixel_sequences as router
    from safesync.full_image_codecs import fit, receive
    from safesync.pixel_wire import PixelState, encode_update, decode_update, UPDATED
    from scripts.run_layered_sequences import config_for, pad
    manifest = json.loads((ROOT / "data/spacenet7_pilot/development_manifest_v1.json").read_text())
    aoi = manifest["aois"][0]
    months = aoi["months"][:updates + 1]
    if len(months) != updates + 1:
        raise ValueError("insufficient fixed calibration observations")
    memory_before = memory_info()
    calls = {}

    def wrapper(name, original):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            value = original(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            calls.setdefault(name, []).append(elapsed)
            return value
        return measured

    # This process-local instrumentation never changes frozen source files.
    DetailCache.scores = wrapper("score_ms", DetailCache.scores)
    DetailCache.updated = wrapper("summary_update_ms", DetailCache.updated)
    lw.parse_frame = wrapper("payload_parse_decode_ms", lw.parse_frame)
    router.choose_packet = wrapper("packet_prefix_search_ms", router.choose_packet)
    start = time.perf_counter()
    first = pad(np.array(Image.open(ROOT / months[0]["rgb_path"])))
    region = hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
    full = method.startswith("full_") and method != "full_cache_mse"
    if full:
        h, w = first.shape[:2]
        n = h * w // 4096
        blank = PixelState(np.zeros_like(first), region)
        packet, _ = encode_update(shape=(h,w),block=64,region=region,base_version=0,
            base_digest=blank.digest(),observed_at=0,current_rgb=first,
            indices=np.arange(n),statuses=np.full(n,UPDATED,np.uint8),codec="png")
        state, ack, _ = decode_update(blank, packet)
        sender = receiver = None
    else:
        sender = lw.LayeredSender(keep_full_archive=method == "full_cache_mse")
        receiver = lw.LayeredReceiver()
        packet, _ = sender.bootstrap(first,region,config_for(method))
        ack = receiver.receive(packet)
        sender.acknowledge(ack)
    bootstrap_ms = (time.perf_counter() - start) * 1000
    records = []
    memory_after_bootstrap = memory_info()
    for step, month in enumerate(months[1:], 1):
        calls.clear()
        start = time.perf_counter()
        current = pad(np.array(Image.open(ROOT / month["rgb_path"])))
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if full:
            packet, metadata = fit(current,codec=method.removeprefix("full_"),
                budget=budget,region=region,base_version=state.version,
                base_digest=state.digest(),observation=step)
        else:
            ranking = "exact" if method == "full_cache_mse" else "bound"
            packet, metadata = sender.propose(current,step,budget,ranking=ranking)
        sender_ms = (time.perf_counter() - start) * 1000
        sender_calls = {k:sum(v) for k,v in calls.items()}
        calls.clear()
        start = time.perf_counter()
        if full:
            state, ack = receive(state, packet)
        else:
            receiver.expect(step)
            ack = receiver.receive(packet)
            receiver.state.render()
        receiver_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if not full:
            sender.acknowledge(ack)
        ack_commit_ms = (time.perf_counter() - start) * 1000
        record = dict(step=step,month=month["month"],preprocess_ms=preprocess_ms,
            sender_ms=sender_ms,receiver_ms=receiver_ms,ack_commit_ms=ack_commit_ms,
            input_to_packet_ms=preprocess_ms+sender_ms,down_bytes=len(packet),
            ack_bytes=len(ack),stages=sender_calls)
        if not full:
            record.update(prefix_trials=metadata["encoding_trials"],
                serialized_summary_bytes=len(sender.cache.wire),
                historical_RGB_bytes=0 if sender.full_archive is None else sender.full_archive.nbytes)
        else:
            record["encoder_meta"] = metadata
        records.append(record)
    fields = ("preprocess_ms","sender_ms","receiver_ms","ack_commit_ms","input_to_packet_ms")
    result = dict(method=method,budget=budget,updates=updates,calibration_aoi=aoi["aoi_id"],
        threads=4,bootstrap_both_endpoints_ms=bootstrap_ms,
        software=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
        memory_before=memory_before,memory_after_bootstrap=memory_after_bootstrap,
        memory_final=memory_info(),records=records,
        statistics={f:dict(mean=float(np.mean([r[f] for r in records])),
                          median=float(np.median([r[f] for r in records])),
                          p95=float(np.percentile([r[f] for r in records],95)))
                    for f in fields},
        limitations="desktop CPU implementation; 12 fixed calibration months; fresh process per method; receiver measured separately; prefix/multiscale search work included; not flight or energy measurement")
    save(out / (method + ".json"), result)
    print(method, json.dumps(result["statistics"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--out", default="results/sender_profile/layered_v1")
    args = parser.parse_args()
    out = ROOT / args.out
    if args.method:
        child(args.method,args.budget,args.updates,out)
    else:
        for method in METHODS:
            subprocess.run([sys.executable,"-X","utf8",str(Path(__file__)),
                "--method",method,"--budget",str(args.budget),"--updates",str(args.updates),
                "--out",args.out],cwd=ROOT,check=True)
        snapshots = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                     for p in ["scripts/profile_layered_sender.py","safesync/layered_cache.py",
                               "safesync/layered_wire.py","safesync/full_image_codecs.py"]}
        save(out/"provenance.json",dict(status="complete",source_sha256=snapshots,
             methods=METHODS,budget=args.budget,updates=args.updates))


if __name__ == "__main__":
    main()
