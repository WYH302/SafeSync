"""Small public S3 transfer probe; no model or package dependencies."""
import json
import time
import urllib.request

url = "https://compressai.s3.us-west-1.amazonaws.com/models/v1/bmshj2018-hyperprior-8-a583f0cf.pth.tar"
started = time.perf_counter()
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-1048575"}), timeout=20) as response:
        data = response.read()
        print(json.dumps({"status": response.status, "bytes": len(data),
            "seconds": time.perf_counter() - started, "host": "WSL stdlib urllib"}))
except Exception as error:
    print(json.dumps({"error": str(error), "seconds": time.perf_counter() - started}))
