"""Bounded Range download of the exact official CPU wheel, preserving chunks."""
from __future__ import annotations

import concurrent.futures
import hashlib
import html.parser
import json
import pathlib
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
FILENAME = "torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl"
DEADLINE_SECONDS = 300


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def main():
    deadline = time.monotonic() + DEADLINE_SECONDS
    folder = HERE / "wheelhouse"
    folder.mkdir(exist_ok=True)
    index = "https://download.pytorch.org/whl/cpu/torch/"
    parser = Links()
    parser.feed(urllib.request.urlopen(index, timeout=30).read().decode())
    href = next(link for link in parser.links
                if urllib.parse.unquote(urllib.parse.urlparse(link).path).endswith("/" + FILENAME))
    full = urllib.parse.urljoin(index, href)
    parsed = urllib.parse.urlparse(full)
    expected = urllib.parse.parse_qs(parsed.fragment)["sha256"][0]
    url = urllib.parse.urlunparse(parsed._replace(fragment=""))
    print(json.dumps({"official_url": url, "sha256": expected}), flush=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=30) as response:
                if response.status != 206:
                    raise RuntimeError("official wheel URL does not support bounded HTTP Range")
                total = int(response.headers["Content-Range"].split("/")[-1])
            break
        except OSError as error:
            print(f"official Range probe attempt {attempt + 1}/3: {error}", flush=True)
            if attempt == 2 or time.monotonic() >= deadline:
                raise
    prefix = folder / "cpu_torch.prefix"
    offset = prefix.stat().st_size if prefix.exists() else 0
    if offset > total:
        raise ValueError("saved prefix exceeds official wheel size")
    info = {"url": url, "sha256": expected, "bytes": total, "prefix_bytes": offset,
            "deadline_seconds": DEADLINE_SECONDS, "workers": 8}
    (folder / "official_cpu_wheel.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
    chunk_size = 4 * 1024 * 1024

    def chunk(item):
        k, start, end = item
        path = folder / f"cpu_torch.chunk{k:03d}"
        done = path.stat().st_size if path.exists() else 0
        size = end - start + 1
        if done > size:
            raise ValueError("saved Range chunk is too large")
        while done < size:
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded CPU wheel download deadline exceeded")
            request = urllib.request.Request(url, headers={"Range": f"bytes={start + done}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected_range = f"bytes {start + done}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("official wheel response Range differs")
                    with path.open("ab") as output:
                        while block := response.read(min(65536, size - done)):
                            output.write(block)
                            done += len(block)
                            if time.monotonic() >= deadline:
                                raise TimeoutError("bounded CPU wheel download deadline exceeded")
                            if done == size:
                                break
            except (OSError, TimeoutError) as error:
                print(f"chunk {k} retained {done}/{size}: {error}", flush=True)
                if time.monotonic() >= deadline:
                    raise
        print(f"chunk {k} complete {size} bytes", flush=True)
        return path

    items = [(k, start, min(start + chunk_size - 1, total - 1))
             for k, start in enumerate(range(offset, total, chunk_size))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(chunk, items))
    temporary = folder / (FILENAME + ".download")
    with temporary.open("wb") as output:
        for path in ([prefix] if offset else []) + paths:
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if temporary.stat().st_size != total or actual != expected:
        raise ValueError("assembled official CPU wheel size/SHA256 differs")
    temporary.replace(folder / FILENAME)
    for path in ([prefix] if offset else []) + paths:
        if path.resolve().parent != folder.resolve():
            raise ValueError("scratch download path escaped its dedicated folder")
        path.unlink()
    print("official CPU wheel assembled and SHA256 verified", flush=True)


if __name__ == "__main__":
    main()
