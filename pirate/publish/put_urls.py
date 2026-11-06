"""Runs on the GPU machine. Reads signed upload URLs from stdin, one JSON object per line ({path, url, headers}), and uploads each file.
The URLs are the only access this machine ever gets: each one covers one object and expires. Standard library only.
Usage: put_urls.py OUT_DIR"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]
def put(job):
    with open(os.path.join(OUT, job["path"]), "rb") as f: body = f.read()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(job["url"], data=body, method="PUT", headers=job["headers"]), timeout=300) as r:
                if r.status == 200: return None
        except Exception as e: err = type(e).__name__   # never print the exception's text: it can quote the signed URL
        time.sleep(5 * (attempt + 1))
    return f'{job["path"]}: {err}'
with ThreadPoolExecutor(8) as ex: failed = [x for x in ex.map(put, map(json.loads, sys.stdin)) if x]
for x in failed[:20]: print("FAILED", x, flush=True)
sys.exit(1 if failed else 0)
