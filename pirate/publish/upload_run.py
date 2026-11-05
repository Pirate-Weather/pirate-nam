"""Publish one built run (publish_product.py output) to the public bucket and update what the viewer lists.
Credentials are fetched in memory and never printed. Files go first, runs.json last, so the viewer never lists a partial run.
Layout: <CYCLE>/woof/grib/*.grib2, <CYCLE>/woof/zarr/... ; KEEP newest WOOF cycles stay, older ones are deleted.
Usage: upload_run.py OUT_DIR CYCLE [KEEP]
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import boto3, ovh
OUT, CYCLE = sys.argv[1:3]; KEEP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
BUCKET, DOMAIN = "pirate-nam-tor", "woof"
# OVH High Performance class, in Toronto: BHS object storage answers each request in 440-710 ms (measured 2026-11-03).
STORAGE_CLASS = "EXPRESS_ONEZONE"
c = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60)); pid = os.environ["OVH_PROJECT"]; uid = int(os.environ["OVH_USER"])
ak = c.get(f"/cloud/project/{pid}/user/{uid}/s3Credentials")[0]["access"]
sk = c.post(f"/cloud/project/{pid}/user/{uid}/s3Credentials/{ak}/secret")["secret"]
s3 = boto3.client("s3", endpoint_url="https://s3.ca-east-tor.io.cloud.ovh.net", region_name="ca-east-tor", aws_access_key_id=ak, aws_secret_access_key=sk)
def put(path, key, ctype="application/octet-stream", cache="public, max-age=86400"):
    s3.upload_file(path, BUCKET, key, ExtraArgs={"ACL": "public-read", "ContentType": ctype, "CacheControl": cache, "StorageClass": STORAGE_CLASS})
jobs = []
for root, _, files in os.walk(OUT):
    for name in files:
        if name.endswith(".part"): continue
        p = os.path.join(root, name); jobs.append((p, f"{CYCLE}/{DOMAIN}/{os.path.relpath(p, OUT)}"))
print(len(jobs), "objects", flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(lambda j: put(*j), jobs))
# Check before listing: every object must be readable with its local size.
for p, key in jobs:
    if s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] != os.path.getsize(p): raise SystemExit(f"size mismatch {key}")
try: runs = json.load(s3.get_object(Bucket=BUCKET, Key="runs.json")["Body"])
except s3.exceptions.NoSuchKey: runs = {"runs": []}
entry = next((r for r in runs["runs"] if r["cycle"] == CYCLE), None)
if entry is None: entry = {"cycle": CYCLE, "domains": []}; runs["runs"].append(entry)
if DOMAIN not in entry.get("domains", []): entry.setdefault("domains", []).append(DOMAIN)
woof = sorted(r["cycle"] for r in runs["runs"] if DOMAIN in r.get("domains", []))
for old in woof[:-KEEP]:
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{old}/{DOMAIN}/")
    keys = [o["Key"] for pg in pages for o in pg.get("Contents", [])]
    for i in range(0, len(keys), 500): s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in keys[i:i + 500]]})
    r = next(x for x in runs["runs"] if x["cycle"] == old); r["domains"].remove(DOMAIN)
    if not r["domains"]: runs["runs"].remove(r)
    print("removed", old, len(keys), flush=True)
s3.put_object(Bucket=BUCKET, Key="runs.json", Body=json.dumps(runs).encode(), ContentType="application/json", ACL="public-read", StorageClass=STORAGE_CLASS, CacheControl="public, max-age=60")
print("published", CYCLE, flush=True)
