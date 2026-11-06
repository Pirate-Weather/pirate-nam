"""Publish one built run from the GPU machine to the public bucket, and update what the viewer lists. Runs on the control host.
The build stays on the GPU machine and so do the credentials here: it gets one signed, expiring upload URL per file, and uploads directly.
Credentials are fetched in memory and never printed. Files go first, runs.json last, so the viewer never lists a partial run.
Layout: <CYCLE>/woof/grib/*.grib2, <CYCLE>/woof/zarr/... ; KEEP newest WOOF cycles stay, older ones are deleted.
Usage: upload_run.py REMOTE_OUT_DIR CYCLE [KEEP] [--no-list]    Environment: GPU_IP, NAM_SSH_KEY, OVH_CONFIG, OVH_PROJECT, OVH_USER
--no-list uploads and checks the files but leaves runs.json alone.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import boto3, ovh
from botocore.config import Config
LIST = "--no-list" not in sys.argv; args = [a for a in sys.argv[1:] if a != "--no-list"]
OUT, CYCLE = args[:2]; KEEP = int(args[2]) if len(args) > 2 else 8
BUCKET, DOMAIN = "pirate-nam-tor", "woof"
# OVH High Performance class, in Toronto: BHS object storage answers each request in 440-710 ms (measured 2026-11-03).
STORAGE_CLASS = "EXPRESS_ONEZONE"
CTYPE, CACHE, URL_LIFETIME_S = "application/octet-stream", "public, max-age=86400", 7200
c = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60)); pid = os.environ["OVH_PROJECT"]; uid = int(os.environ["OVH_USER"])
ak = c.get(f"/cloud/project/{pid}/user/{uid}/s3Credentials")[0]["access"]
sk = c.post(f"/cloud/project/{pid}/user/{uid}/s3Credentials/{ak}/secret")["secret"]
s3 = boto3.client("s3", endpoint_url="https://s3.ca-east-tor.io.cloud.ovh.net", region_name="ca-east-tor", aws_access_key_id=ak, aws_secret_access_key=sk,
                  config=Config(signature_version="s3v4", max_pool_connections=16))
SSH = ["ssh", "-i", os.environ["NAM_SSH_KEY"], "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=60", "-o", "LogLevel=ERROR", "ubuntu@" + os.environ["GPU_IP"]]
listing = subprocess.run(SSH + [f"find {OUT} -type f ! -name '*.part' -printf '%s\\t%P\\n'"], check=True, capture_output=True, text=True).stdout
jobs = [(int(size), path, f"{CYCLE}/{DOMAIN}/{path}") for size, path in (line.split("\t", 1) for line in listing.splitlines())]
if not jobs: raise SystemExit(f"nothing to upload in {OUT}")
print(len(jobs), "objects", round(sum(j[0] for j in jobs) / 1e9, 2), "GB", flush=True)
HEADERS = {"Content-Type": CTYPE, "Cache-Control": CACHE, "x-amz-acl": "public-read", "x-amz-storage-class": STORAGE_CLASS}
def signed(job):
    url = s3.generate_presigned_url("put_object", ExpiresIn=URL_LIFETIME_S, Params={"Bucket": BUCKET, "Key": job[2], "ACL": "public-read", "ContentType": CTYPE, "CacheControl": CACHE, "StorageClass": STORAGE_CLASS})
    return json.dumps({"path": job[1], "url": url, "headers": HEADERS})
payload = "\n".join(signed(j) for j in jobs) + "\n"
subprocess.run(SSH + [f"python3 /woof/work/gpuda/put_urls.py {OUT}"], input=payload, text=True, check=True)
# Check before listing: every object must be readable with the size it has on the GPU machine.
def size_matches(job): return s3.head_object(Bucket=BUCKET, Key=job[2])["ContentLength"] == job[0]
with ThreadPoolExecutor(16) as ex:
    if not all(ex.map(size_matches, jobs)): raise SystemExit("size mismatch after upload")
if not LIST: print("uploaded, not listed", CYCLE, flush=True); raise SystemExit
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
