"""Resume existing public weights through one measured requests network path.

This uses only binary public HTTP ranges and existing exact-offset part files.
It never logs proxy values, netrc credentials or redirect query strings.
"""
import argparse
from collections import deque
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import requests
from requests.auth import AuthBase

HERE = Path(__file__).resolve().parent


class NoNetrcAuth(AuthBase):
    def __call__(self, request):
        return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trust-env", action="store_true")
    parser.add_argument("--base-url", default="https://compressai.s3-us-west-1.amazonaws.com/models/v1/")
    parser.add_argument("--route-config", type=Path)
    args = parser.parse_args()
    info = json.loads((HERE / "regional_weight_download.json").read_text())
    routes = json.loads(args.route_config.read_text()) if args.route_config else {}
    previous = HERE / "requests_weight_transfer_progress.json"
    prior = json.loads(previous.read_text()) if previous.exists() else {}
    prior_objects = prior.get("downloaded_objects", []) + prior.get("files", [])
    official_etags = {item["quality"]: item.get("official_etag") or item["etag"]
                      for item in prior_objects if item.get("official_etag") or item.get("etag")}
    started = time.monotonic()
    deadline = started + args.duration_seconds
    local, lock = threading.local(), threading.Lock()
    progress = {"status": "running", "started_epoch": time.time(), "elapsed_seconds": 0,
                "duration_limit_seconds": args.duration_seconds, "workers": args.workers,
                "host": "WSL" if str(HERE).startswith("/") else "Windows",
                "trust_env": args.trust_env, "new_bytes": 0, "errors": 0,
                "last_error_type": None, "quality_complete": [], "files": []}
    last_saved = [0.0]

    def session():
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.trust_env = args.trust_env
            local.session.auth = NoNetrcAuth()
            local.session.headers.update({"Accept-Encoding": "identity",
                                          "User-Agent": "SafeSync-public-codec-resume/1.0"})
        return local.session

    def record(force=False):
        now = time.monotonic()
        if force or now - last_saved[0] > 10:
            progress["elapsed_seconds"] = now - started
            path = HERE / "requests_weight_transfer_progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(path)
            last_saved[0] = now

    def header(item):
        item = dict(item)
        item["transfer_url"] = routes.get("urls", {}).get(str(item["quality"]), args.base_url + item["name"])
        url = item["transfer_url"]
        if not url.startswith("https://"):
            raise ValueError("public model transport must retain verified HTTPS")
        item["official_etag"] = official_etags.get(item["quality"])
        if not args.route_config and item["official_etag"]:
            item["etag"] = item["official_etag"]
            item["etag_source"] = "previous validated official object response; rechecked on every range"
            return item
        while time.monotonic() < deadline:
            try:
                with session().get(url, headers={"Range": "bytes=0-0"}, stream=True,
                                   timeout=(10, 15)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes 0-0/{item['total_bytes']}":
                        raise ValueError("wrong public object size/range; error body rejected")
                    item["etag"] = response.headers.get("ETag")
                    if not item["etag"]:
                        raise ValueError("public object has no ETag for consistency checks")
                    if len(response.content) != 1:
                        raise ValueError("public one-byte response differs")
                return item
            except Exception as error:
                with lock:
                    progress["errors"] += 1
                    progress["last_error_type"] = type(error).__name__
                    record()
                time.sleep(1)
        raise TimeoutError("public model header deadline")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(header, info["files"]))
    jobs = []
    for item in items:
        final = HERE / "weights" / item["name"]
        if final.exists():
            progress["quality_complete"].append(item["quality"])
            continue
        count = 0
        for k, start in enumerate(range(item["prefix_bytes"], item["total_bytes"], info["chunk_bytes"])):
            path = HERE / "weight_ranges" / str(item["quality"]) / f"{k:04d}.part"
            path.parent.mkdir(parents=True, exist_ok=True)
            end = min(start + info["chunk_bytes"] - 1, item["total_bytes"] - 1)
            size = end - start + 1
            if path.exists() and path.stat().st_size > size:
                raise ValueError("saved public model range is too large")
            if not path.exists() or path.stat().st_size < size:
                jobs.append((item, path, start, end))
                count += 1
        progress["files"].append({"quality": item["quality"], "etag": item["etag"],
                                  "pending_parts": count, "total_bytes": item["total_bytes"]})
    record(True)
    completed_parts = [0]
    # Keep distinct public model objects active, rather than queueing all of
    # one large object ahead of the others on a fragile per-object path.
    grouped = {q: deque(job for job in jobs if job[0]["quality"] == q) for q in (5, 6, 7, 8)}
    jobs = []
    while any(grouped.values()):
        for q in grouped:
            if grouped[q]:
                jobs.append(grouped[q].popleft())

    def transfer(job):
        item, path, start, end = job
        size = end - start + 1
        done = path.stat().st_size if path.exists() else 0
        while done < size and time.monotonic() < deadline:
            request_end = min(start + done + 256 * 1024 - 1, end)
            expected_bytes = request_end - (start + done) + 1
            headers = {"Range": f"bytes={start + done}-{request_end}", "If-Match": item["etag"]}
            try:
                with session().get(item["transfer_url"], headers=headers, stream=True,
                                   timeout=(10, 20)) as response:
                    if (response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start + done}-{request_end}/{item['total_bytes']}"
                            or response.headers.get("ETag") != item["etag"]
                            or response.headers.get("Content-Encoding", "identity") != "identity"):
                        raise ValueError("public range/object identity changed; body rejected")
                    received = 0
                    with path.open("ab") as output:
                        for block in response.iter_content(16384):
                            if received + len(block) > expected_bytes:
                                raise ValueError("public range body exceeds declared extent")
                            output.write(block)
                            received += len(block)
                            done += len(block)
                            with lock:
                                progress["new_bytes"] += len(block)
                                record()
                            if time.monotonic() >= deadline:
                                break
            except Exception as error:
                with lock:
                    progress["errors"] += 1
                    progress["last_error_type"] = type(error).__name__
                    record()
                time.sleep(.25)
        with lock:
            completed_parts[0] += int(done == size)
            if completed_parts[0] and completed_parts[0] % 8 == 0:
                print(json.dumps({"completed_parts": completed_parts[0], "new_bytes": progress["new_bytes"],
                                  "seconds": time.monotonic() - started}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(transfer, jobs))
    for item in items:
        parts = sorted((HERE / "weight_ranges" / str(item["quality"])).glob("*.part"))
        expected_parts = (item["total_bytes"] - item["prefix_bytes"] + info["chunk_bytes"] - 1) // info["chunk_bytes"]
        complete = len(parts) == expected_parts and all(p.stat().st_size == min(info["chunk_bytes"], item["total_bytes"] - item["prefix_bytes"] - int(p.stem) * info["chunk_bytes"]) for p in parts)
        if not complete:
            continue
        final = HERE / "weights" / item["name"]
        prefix = HERE / "weights" / (item["name"] + ".download")
        temporary = HERE / "weights" / (item["name"] + ".assembled")
        with temporary.open("wb") as output:
            for source in [prefix] + parts:
                with source.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        output.write(block)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if temporary.stat().st_size != item["total_bytes"] or not digest.startswith(item["name"].split("-")[-1].split(".")[0]):
            raise ValueError("assembled public model size/SHA256 prefix differs")
        item["sha256"] = digest
        item["assembled_md5"] = hashlib.md5(temporary.read_bytes()).hexdigest()
        expected_official_etag = item.get("official_etag") or item["etag"]
        item["official_etag_matches_md5"] = expected_official_etag.strip('"') == item["assembled_md5"]
        if re.fullmatch(r'"?[0-9a-fA-F]{32}"?', expected_official_etag) and not item["official_etag_matches_md5"]:
            raise ValueError("assembled public model differs from pinned official object ETag/MD5")
        temporary.replace(final)
        if item["quality"] not in progress["quality_complete"]:
            progress["quality_complete"].append(item["quality"])
    progress["status"] = "complete" if set(progress["quality_complete"]) == {5,6,7,8} else "deadline_partial_retained"
    progress["downloaded_objects"] = [{k:v for k,v in item.items() if k != "transfer_url"} for item in items]
    record(True)
    print(json.dumps({"status": progress["status"], "quality_complete": progress["quality_complete"],
                      "new_bytes": progress["new_bytes"], "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
