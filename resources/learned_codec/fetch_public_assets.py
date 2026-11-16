"""Fetch only official public source/weights; usable before torch is installed."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
WEIGHTS = {
    1: "bmshj2018-hyperprior-1-7eb97409.pth.tar",
    2: "bmshj2018-hyperprior-2-93677231.pth.tar",
    3: "bmshj2018-hyperprior-3-6d87be32.pth.tar",
    4: "bmshj2018-hyperprior-4-de1b779c.pth.tar",
    5: "bmshj2018-hyperprior-5-f8b614e1.pth.tar",
    6: "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
    7: "bmshj2018-hyperprior-7-3804dcbd.pth.tar",
    8: "bmshj2018-hyperprior-8-a583f0cf.pth.tar",
}


def fetch(url, path, expected_hash=None, hash_prefix=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        deadline = time.monotonic() + 600
        temporary = path.with_suffix(path.suffix + ".download")
        for attempt in range(3):
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded public weight download deadline exceeded")
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    mode = "ab" if response.status == 206 and offset else "wb"
                    with temporary.open(mode) as output:
                        while block := response.read(64 * 1024):
                            output.write(block)
                            if time.monotonic() >= deadline:
                                raise TimeoutError("bounded public weight download deadline exceeded")
                break
            except (OSError, TimeoutError) as error:
                print(f"download attempt {attempt + 1}/3 {path.name}: {error}", flush=True)
                if attempt == 2 or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if expected_hash and digest != expected_hash or hash_prefix and not digest.startswith(hash_prefix):
            raise ValueError(f"official downloaded asset checksum differs: {path.name}")
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash and digest != expected_hash or hash_prefix and not digest.startswith(hash_prefix):
        raise ValueError(f"official asset checksum differs: {path.name}")
    result = {"url": url, "file": path.name, "sha256": digest, "bytes": path.stat().st_size}
    print(json.dumps(result), flush=True)
    return result


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {quality: pool.submit(fetch,
            "https://compressai.s3.amazonaws.com/models/v1/" + filename,
            HERE / "weights" / filename, hash_prefix=filename.split("-")[-1].split(".")[0])
            for quality, filename in WEIGHTS.items()}
        weights = [{"quality": quality, **job.result()} for quality, job in jobs.items()]
    path = HERE / "source" / "compressai-1.2.8.tar.gz"
    provenance = path.with_suffix(".download.json")
    if path.exists() and provenance.exists():
        item = json.loads(provenance.read_text())
        source = fetch(item["url"], path, expected_hash=item["sha256"])
    else:
        metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/compressai/1.2.8/json", timeout=30))
        item = next(f for f in metadata["urls"] if f["filename"] == path.name)
        source = fetch(item["url"], path, expected_hash=item["digests"]["sha256"])
    path.with_suffix(".download.json").write_text(json.dumps(source, indent=2) + "\n")
    (HERE / "public_downloads.json").write_text(json.dumps({"weights": weights,
        "source_archive": source}, indent=2) + "\n")


if __name__ == "__main__":
    main()
