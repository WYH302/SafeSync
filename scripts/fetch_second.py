"""Retrieve a label-blind, frozen pilot sample from the public SECOND mirror.

Only ZIP byte ranges are fetched. Original official test images are excluded.
Every extracted file has a SHA256 entry; no credentials or browser state are used.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/14037770/files/SECOND.zip/content"


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.transferred = 0
        self.cache: list[tuple[int, bytes]] = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def get(self, start: int, end: int) -> bytes:
        for a, b in self.cache:
            if a <= start and end < a + len(b):
                return b[start-a:end-a+1]
        # Small persisted chunks survive unstable long-lived transfers.
        if end - start + 1 > 196608:
            jobs = [(a, min(end, a + 196607)) for a in range(start, end + 1, 196608)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                return b"".join(pool.map(lambda pair: self.get(*pair), jobs))
        block_cache = ROOT / "data/second_pilot/ranges"
        block_cache.mkdir(parents=True, exist_ok=True)
        cached = block_cache / f"{start}-{end}.bin"
        if cached.exists() and cached.stat().st_size == end - start + 1:
            return cached.read_bytes()
        for attempt in range(5):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 90), stream=True)
                if response.status_code in (429, 500, 502, 503, 504):
                    delay = min(60, int(response.headers.get("Retry-After", 5 * (attempt + 1))))
                    response.close()
                    print(f"retry {attempt + 1}, wait {delay}s", flush=True)
                    time.sleep(delay)
                    continue
                if response.status_code != 206:
                    response.close()
                    raise RuntimeError(f"Range unsupported: HTTP {response.status_code}; refusing full archive")
                expected = f"bytes {start}-{end}/{self.size}"
                if response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"unexpected range {response.headers.get('Content-Range')}")
                body = response.content
                response.close()
                if len(body) != end - start + 1:
                    raise RuntimeError("truncated range")
                self.transferred += len(body)
                cached.write_bytes(body)
                return body
            except requests.RequestException as error:
                print(f"range {start}-{end} retry {attempt+1}: {type(error).__name__}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("range retry budget exhausted")

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.pos + size)
        if end <= self.pos:
            return b""
        out = self.get(self.pos, end - 1)
        self.pos = end
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "data" / "second_pilot"
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/mvp.json").read_text(encoding="utf-8"))
    metadata_path = target / "source_record.json"
    if metadata_path.exists():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        response = requests.get("https://zenodo.org/api/records/14037770", timeout=45)
        response.raise_for_status()
        record = response.json()
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    item = next(f for f in record["files"] if f["key"] == "SECOND.zip")
    remote = RangeFile(URL, item["size"])
    # Prefetch central directory once; zipfile's subsequent small seeks hit cache.
    tail_start = item["size"] - 65536
    tail_file = target / "zip_tail.bin"
    tail = tail_file.read_bytes() if tail_file.exists() else remote.get(tail_start, item["size"] - 1)
    tail_file.write_bytes(tail)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP EOCD not found")
    fields = struct.unpack_from("<4s4H2IH", tail, eocd)
    directory_size, directory_start = fields[5], fields[6]
    directory_file = target / "zip_directory.bin"
    print(f"central directory {directory_size/1e6:.2f} MB", flush=True)
    directory = directory_file.read_bytes() if directory_file.exists() else remote.get(directory_start, item["size"] - 1)
    directory_file.write_bytes(directory)
    remote.cache = [(directory_start, directory)]
    with zipfile.ZipFile(remote) as archive:
        infos = {x.filename: x for x in archive.infolist() if not x.is_dir()}
        (target / "archive_inventory.json").write_text(json.dumps({k: v.file_size for k, v in infos.items()}, indent=2), encoding="utf-8")
        print("files", len(infos), "examples", list(infos)[:18], flush=True)
        print("text files", [x for x in infos if x.endswith('.txt')][:20], flush=True)
        if args.inspect_only:
            return
        image_folder = "T1"
        candidates = [x for x in infos if "/train/" in f"/{x.lower()}" and "/t1/" in x.lower() and x.lower().endswith((".png", ".jpg", ".tif"))]
        if not candidates:
            raise RuntimeError("Unrecognized archive layout; inspect inventory before changing the adapter")
        candidates.sort(key=lambda x: hashlib.sha256(f"{config['seed']}:{PurePosixPath(x).name}".encode()).hexdigest())
        selected = candidates[:config["sample_pairs"]]
        if len(selected) < config["sample_pairs"]:
            raise RuntimeError("insufficient train pairs")
        (target / "selection_before_label_access.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        def extract(member, local):
            info = infos[member]
            if local.exists():
                content = local.read_bytes()
                if len(content) == info.file_size and zlib.crc32(content) == info.CRC:
                    return content
            block = remote.get(info.header_offset, min(item["size"] - 1, info.header_offset + 30 + len(member.encode()) + len(info.extra) + info.compress_size + 1023))
            if block[:4] != b"PK\x03\x04":
                raise RuntimeError("bad local ZIP header")
            filename_len, extra_len = struct.unpack_from("<HH", block, 26)
            start = 30 + filename_len + extra_len
            compressed = block[start:start + info.compress_size]
            content = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed if info.compress_type == zipfile.ZIP_STORED else None
            if content is None or len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                raise RuntimeError("ZIP entry size/CRC mismatch")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            return content
        extract("SECOND/README.txt", target / "SOURCE_README.txt")
        def fetch_pair(pair):
            index, im1 = pair
            name = PurePosixPath(im1).name
            split = "calibration" if index < config["calibration_pairs"] else "development_a" if index < config["calibration_pairs"] + config["development_a_pairs"] else "development_b"
            entry = {"id": PurePosixPath(name).stem, "group_id": PurePosixPath(name).stem, "split": split, "source_split": "train", "files": {}}
            for folder, key in [("T1", "before"), ("T2", "after"), ("GT_T1", "label_before"), ("GT_T2", "label_after"), ("GT_CD", "binary_change")]:
                member = im1.replace(f"/{image_folder}/", f"/{folder}/")
                if member not in infos:
                    raise RuntimeError(f"Missing corresponding member: {member}")
                local = target / folder / name
                content = extract(member, local)
                entry["files"][key] = {"path": local.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "archive_member": member}
            print(f"{index+1}/{len(selected)} {entry['id']} {split}, fetched={remote.transferred/1e6:.1f} MB", flush=True)
            return entry
        manifest = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for entry in pool.map(fetch_pair, enumerate(selected)):
                manifest.append(entry)
                (target / "manifest.partial.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output = {"protocol": config["protocol_version"], "source": URL, "archive_bytes": item["size"], "archive_checksum": item.get("checksum"), "downloaded_bytes_this_run": remote.transferred, "selection": "SHA256(seed:original_filename), no label access before selection", "geographic_independence": "original image pairs disjoint; geographic independence unknown (mirror has no geographic grouping)", "pairs": manifest}
        (target / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
