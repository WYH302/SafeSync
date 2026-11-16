"""Two fixed calibration frames: CPU4 real LIC search and selected-call costs.

No frozen codec/generation source is changed. All model work is CPU-only;
actual official rANS is the existing local WSL bridge. No GT/task is opened.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/sender_profile/lic_cpu_v1"
RESOURCES = ROOT / "resources/learned_codec"
FRAME_LIMIT_SECONDS = 180
RSS_LIMIT_BYTES = 8 * 1024**3


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def memory_info():
    size = ctypes.c_size_t
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    ("peak_rss", size), ("rss", size), ("peak_paged_pool", size),
                    ("paged_pool", size), ("peak_nonpaged_pool", size),
                    ("nonpaged_pool", size), ("pagefile", size),
                    ("peak_pagefile", size), ("private_bytes", size)]
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return {name: int(getattr(info, name)) for name, _ in info._fields_ if name != "cb"}


def wsl_entropy_memory(allowed_pid=None):
    # Exact command matching returns only this public local entropy server's
    # PIDs, never any other process's command line or memory contents.
    windows = PureWindowsPath(RESOURCES / "entropy_server.py")
    path = "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
    code = """import json,pathlib,re,subprocess,sys
pattern='^python3 '+re.escape(sys.argv[1])+'$'
found=subprocess.run(['pgrep','-f',pattern],capture_output=True,text=True)
pids=[int(x) for x in found.stdout.split()]
out=[]
for pid in pids:
 if sys.argv[2]!='none' and pid!=int(sys.argv[2]): continue
 data=pathlib.Path('/proc')/str(pid)/'status'
 if not data.exists(): continue
 fields={}
 for line in data.read_text().splitlines():
  if line.startswith(('VmRSS:','VmHWM:')):
   key,value,*_=line.split(); fields[key[:-1]+'_bytes']=int(value)*1024
 out.append({'pid':pid,**fields})
print(json.dumps(out))
"""
    try:
        result = subprocess.run(["wsl.exe", "-d", "Astra-Ubuntu-22.04", "--", "python3", "-c", code,
            path, str(allowed_pid) if allowed_pid is not None else "none"], capture_output=True,
            text=True, encoding="utf-8", timeout=10, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "complete.json").exists():
        raise RuntimeError("completed fixed profile exists; do not silently repeat or overwrite")
    from safesync.learned_codec import Candidate, HyperpriorCodec, QUALITIES, SCALES, pack, receive, select
    from safesync.pixel_wire import PixelState, decode_update
    import numpy as np
    from PIL import Image

    baseline = json.loads((ROOT / "results/sender_profile/layered_v1/gaussian_20261029.json").read_text())
    manifest_path = ROOT / "data/spacenet7_pilot/development_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    aoi = manifest["aois"][0]
    months = aoi["months"][1:3]
    assert aoi["aoi_id"] == baseline["calibration_aoi"]
    assert [m["month"] for m in months] == [r["month"] for r in baseline["records"][:2]]
    frozen_path = ROOT / "results/learned_codec/confirmation_q1_8_v1/run_spec.json"
    frozen = json.loads(frozen_path.read_text())
    source_hashes = {name: sha((ROOT / name).read_bytes()) for name in frozen["source_sha256"]}
    if source_hashes != frozen["source_sha256"]:
        raise ValueError("frozen codec/generation sources differ before CPU profiling")

    stop_monitor = threading.Event()
    memory_exceeded = threading.Event()
    monitored_peak = [0]
    def monitor():
        while not stop_monitor.is_set():
            info = memory_info()
            monitored_peak[0] = max(monitored_peak[0], info["rss"], info["peak_rss"])
            if info["rss"] > RSS_LIMIT_BYTES:
                memory_exceeded.set()
            stop_monitor.wait(.05)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    memory_before = memory_info()
    previous_servers = wsl_entropy_memory() or []
    started = time.perf_counter()
    codec = HyperpriorCodec(RESOURCES, device="cpu", threads=4, entropy_bridge=True)
    constructor_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    assets = codec.prepare()
    weights_and_CDF_ms = (time.perf_counter() - started) * 1000
    assert not codec.torch.cuda.is_initialized()
    assert all(parameter.device.type == "cpu" for model in codec.models.values() for parameter in model.parameters())
    after_servers = wsl_entropy_memory() or []
    new_servers = [s for s in after_servers if s["pid"] not in {p["pid"] for p in previous_servers}]
    service_pid = new_servers[0]["pid"] if len(new_servers) == 1 else None
    service_samples = [{"phase": "after_model_predeployment", "memory": new_servers}]

    def bridge_snapshot():
        return json.loads(json.dumps(codec.bridge.stats))
    def bridge_delta(before):
        now = codec.bridge.stats
        return {"wall_ms": now["wall_ms"] - before["wall_ms"],
            "request_bytes": now["request_bytes"] - before["request_bytes"],
            "response_bytes": now["response_bytes"] - before["response_bytes"],
            "calls": {key: now["calls"].get(key,0)-before["calls"].get(key,0) for key in now["calls"]}}

    def prepare_tensor(current, coded_shape, padded_shape):
        ch,cw=coded_shape;ph,pw=padded_shape
        small=np.asarray(Image.fromarray(current).resize((cw,ch),Image.Resampling.LANCZOS))
        tensor=codec.torch.from_numpy(small.copy()).permute(2,0,1).unsqueeze(0)
        tensor=tensor.to("cpu",codec.torch.float32).div_(255)
        return codec.torch.nn.functional.pad(tensor,(0,pw-cw,0,ph-ch),value=0)

    first=aoi["months"][0]
    with Image.open(ROOT/first["rgb_path"]) as image:
        initial_rgb=np.asarray(image.convert("RGB"))
    initial_rgb=np.pad(initial_rgb,((0,-initial_rgb.shape[0]%64),(0,-initial_rgb.shape[1]%64),(0,0)))
    region=hashlib.sha256(aoi["aoi_id"].encode()).digest()[:16]
    initial_path=ROOT/"results/pixel_sequences/development_side4"/aoi["aoi_id"]/"initial.bin"
    started=time.perf_counter()
    initial_wire=initial_path.read_bytes()
    initial_state,initial_ack,_=decode_update(PixelState(np.zeros_like(initial_rgb),region),initial_wire)
    bootstrap_load_decode_ms=(time.perf_counter()-started)*1000
    assert np.array_equal(initial_state.rgb,initial_rgb)
    states={cap:initial_state for cap in (32768,65536,131072)}
    records=[];budget_stop=None
    provenance=dict(status="running",method="official bmshj2018_hyperprior MSE",qualities=list(QUALITIES),scales=list(SCALES),threads=4,
        calibration_aoi=aoi["aoi_id"],months=[m["month"] for m in months],caps=list(states),
        input_scope="same calibration AOI index0 first two updates as layered_v1; RGB only",
        source_sha256=source_hashes,profile_source_sha256=sha(Path(__file__).read_bytes()),
        manifest_sha256=sha(manifest_path.read_bytes()),frozen_confirmation_spec_sha256=sha(frozen_path.read_bytes()),
        runtime=codec.runtime,cold_setup=dict(codec_constructor_including_imports_bridge_ms=constructor_ms,
            all_eight_weight_loading_CDF_generation_ms=weights_and_CDF_ms,bootstrap_load_decode_ms=bootstrap_load_decode_ms),
        model_predeployment_bytes=sum(w["bytes"] for w in assets["weights"]),memory_before=memory_before,
        memory_after_model_predeployment=memory_info(),WSL_service_samples=service_samples,
        GPU_inference_calls=0,CUDA_visible_devices="",CUDA_initialized=False,
        search_deadline_seconds=FRAME_LIMIT_SECONDS,Windows_process_RSS_limit_bytes=RSS_LIMIT_BYTES)
    save(OUT/"provenance.json",provenance)

    try:
        for step,month in enumerate(months,1):
            if memory_exceeded.is_set():
                budget_stop="Windows process RSS exceeded 8 GiB before next frame"
                break
            image_path=ROOT/month["rgb_path"]
            source_sha=sha(image_path.read_bytes())
            assert source_sha==month["derived_sha256"]["rgb_path"]
            started=time.perf_counter()
            with Image.open(image_path) as image:
                current=np.asarray(image.convert("RGB"))
            current=np.pad(current,((0,-current.shape[0]%64),(0,-current.shape[1]%64),(0,0)))
            preprocess_ms=(time.perf_counter()-started)*1000
            h,w=current.shape[:2]
            candidates=[];candidate_meta=[];artifact_ms=0.0
            bridge_before=bridge_snapshot();search_started=time.perf_counter()
            folder=OUT/"frames"/month["month"];folder.mkdir(parents=True,exist_ok=True)
            for quality in QUALITIES:
                net=codec.model(quality)
                for scale in SCALES:
                    ch,cw=max(1,round(h*scale)),max(1,round(w*scale))
                    ph,pw=math.ceil(ch/64)*64,math.ceil(cw/64)*64
                    tensor=prepare_tensor(current,(ch,cw),(ph,pw))
                    started=time.perf_counter()
                    with codec.torch.inference_mode(): encoded=net.compress(tensor)
                    compress_ms=(time.perf_counter()-started)*1000
                    streams=tuple(bytes(group[0]) for group in encoded["strings"])
                    shape=tuple(map(int,encoded["shape"]))
                    started=time.perf_counter()
                    rgb=codec._decode(quality=quality,streams=streams,latent_shape=shape,padded_shape=(ph,pw),coded_shape=(ch,cw),native_shape=(h,w),weight_sha256=codec.weights[quality]["sha256"])
                    decode_ms=(time.perf_counter()-started)*1000
                    mse=float(np.mean(((current.astype(np.float64)-rgb)/255)**2))
                    candidate=Candidate(quality,(h,w),(ch,cw),(ph,pw),shape,codec.weights[quality]["sha256"],streams,mse,compress_ms,decode_ms,sha(rgb.tobytes()))
                    candidates.append(candidate)
                    checkpoint=time.perf_counter()
                    number=len(candidates)-1
                    (folder/f"{number:02d}.y.bin").write_bytes(streams[0]);(folder/f"{number:02d}.z.bin").write_bytes(streams[1])
                    candidate_meta.append({**candidate.metadata(),"scale":scale,"stream_sha256":[sha(s) for s in streams]})
                    save(folder/"candidate_progress.json",dict(status="in_progress",candidates=candidate_meta,input_png_sha256=source_sha,input_RGB_sha256=sha(current.tobytes())))
                    artifact_ms+=(time.perf_counter()-checkpoint)*1000
                    wall=time.perf_counter()-search_started
                    if wall>FRAME_LIMIT_SECONDS or memory_exceeded.is_set():
                        budget_stop="full search wall time exceeded 180s" if wall>FRAME_LIMIT_SECONDS else "Windows process RSS exceeded 8 GiB"
                        break
                if budget_stop: break
            search_wall_ms=(time.perf_counter()-search_started)*1000
            record=dict(step=step,month=month["month"],input_png_sha256=source_sha,input_RGB_sha256=sha(current.tobytes()),input_shape=list(current.shape),preprocess_ms=preprocess_ms,
                completed_candidates=len(candidates),search_complete=len(candidates)==40 and not budget_stop,
                full_search_wall_including_checkpoint_IO_ms=search_wall_ms,artifact_checkpoint_IO_ms=artifact_ms,
                full_search_excluding_checkpoint_IO_ms=search_wall_ms-artifact_ms,full_search_bridge_RPC=bridge_delta(bridge_before),
                memory_after_search=memory_info(),selections=[],selected_single_calls=[])
            if budget_stop:
                record["budget_stop"]=budget_stop;records.append(record);save(OUT/"records.json",records)
                break
            for cap in states:
                prior=states[cap];started=time.perf_counter()
                wire,meta=select(candidates,budget=cap,region=region,base_version=prior.version,base_digest=prior.digest(),observation=step)
                select_ms=(time.perf_counter()-started)*1000
                selected=next(c for c in candidates if c.quality==meta["quality"] and c.coded_shape==(meta["coded_height"],meta["coded_width"]))
                packet_path=folder/f"selected_{cap}.bin";packet_path.write_bytes(wire)
                started=time.perf_counter();state,ack=receive(prior,packet_path.read_bytes(),codec)
                receiver_ms=(time.perf_counter()-started)*1000
                assert len(wire)<=cap and len(ack)==60 and sha(state.rgb.tobytes())==selected.decoded_rgb_sha256
                packet_path.with_suffix(".ack").write_bytes(ack)
                Image.fromarray(state.rgb).save(folder/f"selected_{cap}.decoded.png")
                states[cap]=state
                record["selections"].append(dict(cap=cap,**meta,packet_sha256=sha(wire),ack_bytes=len(ack),ack_sha256=sha(ack),receiver_RGB_sha256=sha(state.rgb.tobytes()),select_serialize_ms=select_ms,selected_packet_receive_ms=receiver_ms,input_to_packet_search_ms=preprocess_ms+search_wall_ms-artifact_ms+select_ms))
                started=time.perf_counter();tensor=prepare_tensor(current,selected.coded_shape,selected.padded_shape)
                selected_preprocess_ms=(time.perf_counter()-started)*1000
                before=bridge_snapshot();started=time.perf_counter()
                with codec.torch.inference_mode(): repeated=codec.model(selected.quality).compress(tensor)
                single_compress_ms=(time.perf_counter()-started)*1000;encode_rpc=bridge_delta(before)
                repeated_streams=tuple(bytes(group[0]) for group in repeated["strings"])
                before=bridge_snapshot();started=time.perf_counter()
                repeated_rgb=codec._decode(quality=selected.quality,streams=repeated_streams,latent_shape=tuple(repeated["shape"]),padded_shape=selected.padded_shape,coded_shape=selected.coded_shape,native_shape=selected.native_shape,weight_sha256=selected.weight_sha256)
                single_decompress_ms=(time.perf_counter()-started)*1000;decode_rpc=bridge_delta(before)
                assert repeated_streams==selected.streams and sha(repeated_rgb.tobytes())==selected.decoded_rgb_sha256
                record["selected_single_calls"].append(dict(cap=cap,quality=selected.quality,coded_height=selected.coded_shape[0],coded_width=selected.coded_shape[1],
                    selected_preprocess_ms=selected_preprocess_ms,single_compress_ms_including_local_RPC=single_compress_ms,
                    single_decompress_ms_including_local_RPC_and_native_restoration=single_decompress_ms,
                    array_to_stream_ms=selected_preprocess_ms+single_compress_ms,encode_RPC_subtotal=encode_rpc,decode_RPC_subtotal=decode_rpc,
                    actual_stream_bytes=sum(map(len,repeated_streams)),actual_packet_bytes=len(wire),decoded_RGB_sha256=sha(repeated_rgb.tobytes()),repeat_streams_and_RGB_equal=True,
                    scope="one warm call after full search; weights/CDF/kernel cache already resident"))
            sample=wsl_entropy_memory(service_pid) if service_pid is not None else None
            service_samples.append(dict(phase=f"after_frame_{step}",memory=sample))
            record["memory_after_selected_calls"]=memory_info();records.append(record)
            save(OUT/"records.json",records);save(folder/"candidate_progress.json",dict(status="complete",candidates=candidate_meta,input_png_sha256=source_sha,input_RGB_sha256=sha(current.tobytes())))
            print(f"CPU4 frame{step} {month['month']}40 candidates: {record['full_search_excluding_checkpoint_IO_ms']/1000:.3f}s, RSSpeak {record['memory_after_selected_calls']['peak_rss']}B",flush=True)
            if memory_exceeded.is_set(): budget_stop="Windows process RSS exceeded 8 GiB after frame";break
        assert not codec.torch.cuda.is_initialized()
    finally:
        final_service=wsl_entropy_memory(service_pid) if service_pid is not None else None
        service_samples.append(dict(phase="final",memory=final_service))
        codec.bridge.close();stop_monitor.set();thread.join(timeout=1)
    assert {name:sha((ROOT/name).read_bytes()) for name in source_hashes}==source_hashes
    summary=dict(status="complete" if len(records)==2 and all(r["search_complete"] for r in records) and not budget_stop else "partial_budget_stop",frames_completed=sum(r["search_complete"] for r in records),budget_stop=budget_stop,
        threads=4,GPU_inference_calls=0,CUDA_initialized=codec.torch.cuda.is_initialized(),runtime=codec.runtime,cold_setup=provenance["cold_setup"],
        records=records,Windows_process_memory_final=memory_info(),Windows_process_RSS_monitored_peak_bytes=monitored_peak[0],WSL_entropy_service_PID=service_pid,WSL_entropy_service_memory_samples=service_samples,
        frozen_534_source_hashes_unchanged=True,full_search_measurement="40 actual compress/decompress candidates plus sender current-RGB MSE; checkpoint IO reported separately",
        selected_call_measurement="one new warm compress and one decompress per cap after selection, with resize/tensor setup separately timed; real stream/RGB repeat checked",
        IPC_scope="all call/search totals include local bridge conversion, pipes and original C++ ANS; RPC subtotal contains pipe+server C++ work and is not pure transport overhead or a separable CPU-native speed estimate",
        RAM_scope="Windows process working set includes Python/Torch/DLLs/all8 model weights/temporary CPU workspace; WSL service VmRSS/VmHWM reported separately. Neither is flight total RAM.",
        comparison_limitations="Same original input months as layered_v1, but existing Torch runtime uses Python3.11/NumPy2.4.6 versus layered profile Python3.12/NumPy2.4.3. Full40-candidate search must not be compared as if it were one codec encode. No GPU-to-CPU stream compatibility is assumed.")
    save(OUT/"summary.json",summary)
    save(OUT/"complete.json",dict(status=summary["status"],frames_completed=summary["frames_completed"],summary_sha256=sha((OUT/"summary.json").read_bytes()),profile_source_sha256=sha(Path(__file__).read_bytes()),GPU_inference_calls=0))
    print(json.dumps(dict(status=summary["status"],frames=summary["frames_completed"],search_seconds=[r['full_search_excluding_checkpoint_IO_ms']/1000 for r in records],Windows_peak_RSS=summary['Windows_process_memory_final']['peak_rss'],budget_stop=budget_stop),indent=2),flush=True)


if __name__=="__main__":
    main()
