"""Obtain the complete authors' public MSE menu without an account session."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"resources/modern_codec/weights"
PLAN = ROOT/"results/task_evidence/tcm_control_v1/plan.json"


def download(slot,item):
    name = f"tcm_N{item['channels_N']}_lambda{item['lambda_mse']:g}.pth.tar"
    path,receipt = BASE/name,BASE/(name+".download.json")
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.exists() and json.loads(receipt.read_text("utf-8"))["sha256"] != digest:
            raise ValueError("Cached public TCM checkpoint changed")
        result = dict(status="complete",slot=slot,**item,path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest)
        receipt.write_text(json.dumps(result,indent=2)+"\n","utf-8")
        return result
    file_id = re.search(r"/d/([^/]+)",item["public_url"])[1]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    for attempt in range(3):
        try:
            response = requests.get(url,stream=True,timeout=(25,120))
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type",""):
                raise RuntimeError("Public checkpoint endpoint returned an HTML access page")
            digest,count = hashlib.sha256(),0
            partial = path.with_suffix(path.suffix+".part")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count > 2*1024**3:
                        raise RuntimeError("Unexpected model-sized payload above2 GiB")
            if count < 1024*1024:
                raise RuntimeError("Public checkpoint response is too small")
            partial.replace(path)
            result = dict(status="complete",slot=slot,**item,path=path.relative_to(ROOT).as_posix(),bytes=count,sha256=digest.hexdigest())
            receipt.write_text(json.dumps(result,indent=2)+"\n","utf-8")
            print(f"public TCM model{slot}/7 complete, {count//(1024*1024)} MiB",flush=True)
            return result
        except (requests.RequestException,RuntimeError):
            if attempt == 2:
                raise
            time.sleep(5*(attempt+1))


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    plan = json.loads(PLAN.read_text("utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download,slot,item) for slot,item in enumerate(plan["models"],1)]
        models = [future.result() for future in futures]
    result = dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(),models=models,
        source_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),authenticated_account_used=False,
        source="Authors' pinned README links, all7 advertised MSE checkpoints")
    (BASE/"registry.json").write_text(json.dumps(result,indent=2)+"\n","utf-8")
    print(json.dumps(dict(status="complete",models=len(models),bytes=sum(r["bytes"] for r in models))),flush=True)


if __name__ == "__main__":
    main()
