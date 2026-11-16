"""Bounded official-S3 range resume for the four higher public qualities."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
NAMES = {5: "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
         6: "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
         7: "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
         8: "bmshj2018-hyperprior-8-a583f0cf.pth.tar"}
DEADLINE = 600
CHUNK = 1024 * 1024


def main():
    started = time.monotonic()
    end_time = started + DEADLINE
    weights, scratch = HERE / "weights", HERE / "weight_ranges"
    scratch.mkdir(exist_ok=True)
    records = []

    def describe(item):
        quality, name = item
        url = "https://compressai.s3.us-west-1.amazonaws.com/models/v1/" + name
        final, prefix = weights / name, weights / (name + ".download")
        if final.exists():
            return {"quality": quality, "name": name, "url": url, "complete": True}
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=20) as response:
                    if response.status != 206:
                        raise RuntimeError("official bucket response does not support Range")
                    total = int(response.headers["Content-Range"].split("/")[-1])
                offset = prefix.stat().st_size if prefix.exists() else 0
                if offset > total:
                    raise ValueError("retained prefix exceeds official file size")
                return {"quality": quality, "name": name, "url": url, "complete": False,
                        "total_bytes": total, "prefix_bytes": offset}
            except OSError as error:
                last = error
        raise last

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(describe, NAMES.items()))
    (HERE / "regional_weight_download.json").write_text(json.dumps({"status": "in_progress",
        "deadline_seconds": DEADLINE, "workers": 8, "chunk_bytes": CHUNK,
        "canonical_source": "official CompressAI zoo URLs in compressai.s3.amazonaws.com",
        "bucket_region_from_official_HEAD": "us-west-1", "files": records}, indent=2) + "\n")
    jobs = []
    for item in records:
        if not item["complete"]:
            part_dir = scratch / str(item["quality"])
            part_dir.mkdir(exist_ok=True)
            for k, start in enumerate(range(item["prefix_bytes"], item["total_bytes"], CHUNK)):
                jobs.append((item, k, start, min(start + CHUNK - 1, item["total_bytes"] - 1)))
    progress, lock = {"done": 0, "retry_errors": 0}, threading.Lock()

    def transfer(job):
        item, k, start, end = job
        path = scratch / str(item["quality"]) / f"{k:04d}.part"
        size = end - start + 1
        done = path.stat().st_size if path.exists() else 0
        if done > size:
            raise ValueError("retained range exceeds its declared extent")
        attempts = 0
        while done < size:
            if time.monotonic() >= end_time:
                raise TimeoutError("bounded public weight range deadline exceeded")
            request = urllib.request.Request(item["url"], headers={"Range": f"bytes={start + done}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start + done}-{end}/{item['total_bytes']}":
                        raise RuntimeError("official bucket returned a different byte range")
                    with path.open("ab") as output:
                        while done < size:
                            block = response.read(min(65536, size - done))
                            if not block:
                                break
                            output.write(block)
                            done += len(block)
                            if time.monotonic() >= end_time:
                                raise TimeoutError("bounded public weight range deadline exceeded")
            except OSError:
                attempts += 1
                with lock:
                    progress["retry_errors"] += 1
                if attempts >= 6 or time.monotonic() >= end_time:
                    raise
        with lock:
            progress["done"] += 1
            if progress["done"] % 16 == 0 or progress["done"] == len(jobs):
                print(f"official weight ranges {progress['done']}/{len(jobs)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(transfer, jobs))
    for item in records:
        final = weights / item["name"]
        if not item["complete"]:
            prefix = weights / (item["name"] + ".download")
            assembled = weights / (item["name"] + ".assembled")
            parts = sorted((scratch / str(item["quality"])).glob("*.part"))
            with assembled.open("wb") as output:
                for source in ([prefix] if prefix.exists() else []) + parts:
                    with source.open("rb") as stream:
                        while block := stream.read(1024 * 1024):
                            output.write(block)
            digest = hashlib.sha256(assembled.read_bytes()).hexdigest()
            expected = item["name"].split("-")[-1].split(".")[0]
            if assembled.stat().st_size != item["total_bytes"] or not digest.startswith(expected):
                raise ValueError("assembled official weight size/SHA256 prefix differs")
            assembled.replace(final)
            for source in ([prefix] if prefix.exists() else []) + parts:
                if not source.resolve().is_relative_to(HERE.resolve()):
                    raise ValueError("scratch cleanup escaped dedicated resource directory")
                source.unlink()
        item.update(sha256=hashlib.sha256(final.read_bytes()).hexdigest(), bytes=final.stat().st_size,
                    canonical_url="https://compressai.s3.amazonaws.com/models/v1/" + item["name"],
                    complete=True)
    result = {"status": "complete", "elapsed_seconds": time.monotonic() - started,
              "deadline_seconds": DEADLINE, "progress": progress, "files": records}
    (HERE / "regional_weight_download.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "seconds": result["elapsed_seconds"],
                      "weights": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
