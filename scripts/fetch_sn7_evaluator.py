"""Read/fetch one small, official SpaceNet7 winning-model component anonymously."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resources/pretrained_ground"
REPO = "SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions"
WEIGHT_PREFIX = "spacenet-model-weights/spacenet-7/"
WEIGHT_KEY = WEIGHT_PREFIX + "4-motokimura/weights/exp_0200/model_best.pth"
HOSTS = ["https://spacenet-dataset.s3.amazonaws.com/", "https://s3.amazonaws.com/spacenet-dataset/"]
FORK_COMMIT = "e590675a2c23e2fb68a37541e88fccd57d8075e5"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def get(url, **kwargs):
    for attempt in range(8):
        try:
            response = requests.get(url, timeout=(15, 60), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(8, attempt + 1))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path, kind="sha256"):
    value = hashlib.new(kind)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def official_etag_matches(path, etag):
    if re.fullmatch(r"[a-f0-9]{32}", etag):
        return digest(path, "md5") == etag
    # This fixed official checkpoint's seven-part ETag was verified against
    # the author's 8-MiB S3 multipart boundaries; it is not a content MD5.
    if not re.fullmatch(r"[a-f0-9]{32}-7", etag):
        raise ValueError("unexpected multipart ETag for the selected fixed checkpoint")
    hashes = []
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hashes.append(hashlib.md5(chunk).digest())
    actual = hashlib.md5(b"".join(hashes)).hexdigest() + "-" + str(len(hashes))
    return actual == etag


def inspect_sources():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = OUT / "source_identity.json"
    if identity.exists():
        return json.loads(identity.read_text(encoding="utf-8"))
    commit = get(f"https://api.github.com/repos/{REPO}/commits/master").json()["sha"]
    tree = get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}", params={"recursive": "1"}).json()
    write_json(OUT / "official_source_tree.json", tree)
    paths = [entry["path"] for entry in tree["tree"] if entry["type"] == "blob" and
             (entry["path"] in ["LICENSE", "README.md"] or entry["path"].startswith("4-motokimura/"))
             and not entry["path"].endswith(".png")]
    def fetch(path):
        target = OUT / "official_source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = get(f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}").content
        target.write_bytes(content)
        return {"path": path, "bytes": len(content), "sha256": digest(target)}
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        files = list(pool.map(fetch, paths))
    listing = get(HOSTS[0], params={"list-type": "2", "prefix": WEIGHT_PREFIX})
    result = ET.fromstring(listing.content)
    if result.findtext("s:IsTruncated", namespaces=NS) == "true":
        raise ValueError("weight listing unexpectedly paginated; selection must use complete listing")
    objects = [{"key": n.findtext("s:Key", namespaces=NS),
                "bytes": int(n.findtext("s:Size", namespaces=NS)),
                "etag": n.findtext("s:ETag", namespaces=NS).strip('"')}
               for n in result.findall("s:Contents", NS)]
    write_json(OUT / "official_weight_listing.json", objects)
    chosen = next(item for item in objects if item["key"] == WEIGHT_KEY)
    value = {"repository": "https://github.com/" + REPO, "commit": commit,
             "source_files": files, "source_bytes": sum(f["bytes"] for f in files),
             "selection": "Smallest individual official winning checkpoint family; lexicographically first experiment exp_0200, chosen before task results. Single component, not the winning ensemble.",
             "weight": chosen, "weight_url": HOSTS[0] + WEIGHT_KEY,
             "source_review_required_before_inference": True}
    write_json(identity, value)
    print(json.dumps({k: v for k, v in value.items() if k != "source_files"}, ensure_ascii=False), flush=True)
    return value


def fetch_weight(identity, maximum):
    expected = identity["weight"]
    if expected["bytes"] + identity["source_bytes"] > maximum:
        raise ValueError("explicit source/weight download cap exceeded")
    path = OUT / "model_best_exp_0200.pth"
    etag_is_md5 = bool(re.fullmatch(r"[a-f0-9]{32}", expected["etag"]))
    if path.exists() and path.stat().st_size == expected["bytes"] and official_etag_matches(path, expected["etag"]):
        write_json(OUT / "weight_receipt.json", {**expected, "url": identity["weight_url"],
                   "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                   "etag_is_content_md5": etag_is_md5, "official_etag_verified": True,
                   "multipart_chunk_bytes": None if etag_is_md5 else 8 * 1024 * 1024})
        print("verified existing weight", path, flush=True)
        return path
    # Fixed HTTP byte ranges persist separately; a failed long transfer loses one range only.
    chunks = [(start, min(start + 1024 * 1024 - 1, expected["bytes"] - 1))
              for start in range(0, expected["bytes"], 1024 * 1024)]
    cache = OUT / "weight_ranges"
    cache.mkdir(exist_ok=True)
    def fetch(pair):
        start, end = pair
        target = cache / f"{start}-{end}.bin"
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(10):
            try:
                response = get(HOSTS[attempt % 2] + WEIGHT_KEY, stream=True,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    params={"sn7-evaluator-range": f"{start}-{end}"})
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected['bytes']}":
                    response.close()
                    raise ValueError("unexpected HTTP byte range")
                if len(response.content) != end - start + 1:
                    raise ValueError("truncated byte range")
                target.write_bytes(response.content)
                return target
            except (requests.RequestException, ValueError):
                if attempt == 9:
                    raise
                time.sleep(min(8, attempt + 1))
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        pieces = list(pool.map(fetch, chunks))
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        for piece in pieces:
            stream.write(piece.read_bytes())
    if temporary.stat().st_size != expected["bytes"] or not official_etag_matches(temporary, expected["etag"]):
        raise ValueError("official checkpoint size/MD5 mismatch")
    temporary.replace(path)
    write_json(OUT / "weight_receipt.json", {**expected, "url": identity["weight_url"],
               "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
               "etag_is_content_md5": etag_is_md5, "official_etag_verified": True,
               "multipart_chunk_bytes": None if etag_is_md5 else 8 * 1024 * 1024})
    print("weight ready", path, path.stat().st_size, flush=True)
    return path


def fetch_model_source():
    url = f"https://codeload.github.com/motokimura/segmentation_models.pytorch.sn7/zip/{FORK_COMMIT}"
    content = get(url).content
    if len(content) > 64_000_000:
        raise ValueError("unexpectedly large model-source archive")
    archive_path = OUT / "smp_author_fork.zip"
    archive_path.write_bytes(content)
    files = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts[1:]
            if not parts or info.is_dir():
                continue
            if ".." in parts:
                raise ValueError("unsafe source archive member")
            if parts[0] != "segmentation_models_pytorch" and parts[0] not in ["LICENSE", "requirements.txt", "setup.py"]:
                continue
            if parts[0] == "segmentation_models_pytorch" and not parts[-1].endswith(".py"):
                continue
            target = OUT / "model_source" / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            files.append({"path": target.relative_to(ROOT).as_posix(), "sha256": digest(target)})
    write_json(OUT / "model_source_identity.json", {"repository": "https://github.com/motokimura/segmentation_models.pytorch.sn7",
               "commit": FORK_COMMIT, "archive_url": url, "archive_bytes": len(content),
               "archive_sha256": digest(archive_path), "files": files})
    print("author model source ready", len(files), "files; archive bytes", len(content), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-weight", action="store_true")
    parser.add_argument("--fetch-model-source", action="store_true")
    parser.add_argument("--max-download-bytes", type=int, default=300_000_000)
    args = parser.parse_args()
    identity = inspect_sources()
    if args.fetch_weight:
        fetch_weight(identity, args.max_download_bytes)
    if args.fetch_model_source:
        fetch_model_source()


if __name__ == "__main__":
    main()
