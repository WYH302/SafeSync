"""Fetch a small official DINOv2 inference source subset, pinned to one SHA."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "resources/dinov2"


def get(url):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(10, 35))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(attempt+1)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    meta = DEST / "SOURCE_COMMIT.txt"
    if meta.exists():
        sha = meta.read_text().strip()
    else:
        sha = get("https://api.github.com/repos/facebookresearch/dinov2/commits/main").json()["sha"]
        meta.write_text(sha)
    tree_file = DEST / "source_tree.json"
    tree = json.loads(tree_file.read_text()) if tree_file.exists() else get(f"https://api.github.com/repos/facebookresearch/dinov2/git/trees/{sha}?recursive=1").json()
    tree_file.write_text(json.dumps(tree))
    paths = [x["path"] for x in tree["tree"] if x["type"] == "blob" and (x["path"] in ("hubconf.py", "LICENSE", "dinov2/__init__.py", "dinov2/models/__init__.py", "dinov2/models/vision_transformer.py") or x["path"].startswith("dinov2/layers/") and x["path"].endswith(".py") or x["path"].startswith("dinov2/hub/") and x["path"].endswith(".py"))]
    def fetch(path):
        local = DEST / path
        if not local.exists():
            content = get(f"https://raw.githubusercontent.com/facebookresearch/dinov2/{sha}/{path}").content
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
        print(path, flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, paths))
    print("source ready", sha, flush=True)


if __name__ == "__main__":
    main()
