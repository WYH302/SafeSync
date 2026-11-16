"""Bounded public-object route probes; no proxy values/netrc/credentials logged."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import urllib.parse

import requests
from requests.auth import AuthBase

HERE = Path(__file__).resolve().parent
NAME = "bmshj2018-hyperprior-8-a583f0cf.pth.tar"
SIZE = 1024 * 1024
TOTAL = 48212373
ROUTES = {
    "path_dot": "https://s3.us-west-1.amazonaws.com/compressai/models/v1/" + NAME,
    "path_hyphen": "https://s3-us-west-1.amazonaws.com/compressai/models/v1/" + NAME,
    "virtual_hyphen": "https://compressai.s3-us-west-1.amazonaws.com/models/v1/" + NAME,
    "virtual_dot": "https://compressai.s3.us-west-1.amazonaws.com/models/v1/" + NAME,
}


class NoNetrcAuth(AuthBase):
    def __call__(self, request):
        return request


def probe(job):
    label, url, trust_env = job
    session = requests.Session()
    session.trust_env = trust_env
    session.auth = NoNetrcAuth()  # Never look up .netrc for a public URL.
    started = time.monotonic()
    data = bytearray()
    result = {"route": label, "url": url, "trust_env": trust_env,
              "requested_bytes": SIZE, "connect_read_timeout_s": [6, 6]}
    try:
        with session.get(url, headers={"Range": f"bytes=0-{SIZE - 1}",
                "User-Agent": "SafeSync-public-codec-route-probe/1.0"},
                timeout=(6, 6), stream=True) as response:
            result.update(http_status=response.status_code,
                content_range=response.headers.get("Content-Range"),
                etag=response.headers.get("ETag"))
            parsed = urllib.parse.urlsplit(response.url)
            result["final_public_url"] = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes 0-{SIZE - 1}/{TOTAL}":
                result["error"] = "wrong public object status/range; body rejected"
                return result
            for block in response.iter_content(65536):
                data.extend(block)
                if time.monotonic() - started > 15:
                    result["error"] = "15-second transfer deadline"
                    break
            result["complete"] = len(data) == SIZE
    except Exception as error:
        # Error messages can embed configured proxy URLs, so emit the type only.
        result["error_type"] = type(error).__name__
        result["complete"] = False
    finally:
        session.close()
        result["seconds"] = time.monotonic() - started
        result["received_bytes"] = len(data)
        result["bytes_per_second"] = len(data) / max(result["seconds"], .001)
        result["received_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(result), flush=True)
    return result


def main():
    if "--extended" in sys.argv:
        ROUTES.clear()
        ROUTES.update({
            "dualstack": "https://compressai.s3.dualstack.us-west-1.amazonaws.com/models/v1/" + NAME,
            "accelerate": "https://compressai.s3-accelerate.amazonaws.com/models/v1/" + NAME,
        })
    jobs = [(label, url, trust) for label, url in ROUTES.items() for trust in (False, True)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(probe, jobs))
    good = [r for r in results if r.get("complete")]
    result = {"host_platform": platform.system(), "probes": results,
              "selected": max(good, key=lambda x: x["bytes_per_second"]) if good else None,
              "no_global_network_changes": True, "no_credentials_or_proxy_values_logged": True}
    name = "s3_routes_" + platform.system().lower() + ".json"
    if "--extended" in sys.argv:
        name = name.replace(".json", "_extended.json")
    (HERE / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete_routes": len(good), "selected": result["selected"]}), flush=True)


if __name__ == "__main__":
    main()
