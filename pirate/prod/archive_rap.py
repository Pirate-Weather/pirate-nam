"""Copy RAP observation files to the private archive bucket before NOMADS ages them out (about 2 days). One file at a time per
worker through temporary disk, each checked by size. Safe to rerun: objects already there with the right size are skipped.
Usage: archive_rap.py YYYYMMDD [YYYYMMDD ...]"""
import os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import boto3, botocore, ovh
BUCKET = "pirate-nam-archive"; N = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"
KINDS = ("prepbufr.tm00.nr", "satwnd.tm00.bufr_d", "lgycld.tm00.bufr_d", "nexrad.tm00.bufr_d")
c = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60)); pid = os.environ["OVH_PROJECT"]; uid = int(os.environ["OVH_USER"])
ak = c.get(f"/cloud/project/{pid}/user/{uid}/s3Credentials")[0]["access"]; sk = c.post(f"/cloud/project/{pid}/user/{uid}/s3Credentials/{ak}/secret")["secret"]
s3 = boto3.client("s3", endpoint_url="https://s3.bhs.io.cloud.ovh.net", region_name="bhs", aws_access_key_id=ak, aws_secret_access_key=sk)
TMP = os.environ["PIRATE_DATA"] + "/archive/.tmp_rap"; os.makedirs(TMP, exist_ok=True)
LOCAL = os.environ["PIRATE_DATA"] + "/archive/obsproc"

def one(job):
    day, name = job; key = f"archive/obsproc/rap.{day}/{name}"; local = f"{LOCAL}/rap.{day}/{name}"
    try:
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        if size > 0: return "already", size
    except botocore.exceptions.ClientError: pass
    if os.path.getsize(local) > 0 if os.path.exists(local) else False:
        s3.upload_file(local, BUCKET, key); got = os.path.getsize(local)
    else:
        fd, tmp = tempfile.mkstemp(dir=TMP); os.close(fd)
        try:
            r = subprocess.run(["curl", "-sf", "--http1.1", "--retry", "3", "--retry-delay", "20", "-m", "1500", "-o", tmp, f"{N}/rap.{day}/{name}"])
            if r.returncode == 22 or os.path.getsize(tmp) == 0: return "missing", 0
            if r.returncode: return f"failed curl {r.returncode}", 0
            s3.upload_file(tmp, BUCKET, key); got = os.path.getsize(tmp)
        finally:
            os.remove(tmp)
    if s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] != got: return "size mismatch", 0
    return "copied", got

for day in sys.argv[1:]:
    jobs = [(day, f"rap.t{h:02d}z.{k}") for h in range(24) for k in KINDS]; tally = {}; bytes_ = 0; t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        for job, (state, size) in zip(jobs, ex.map(one, jobs)):
            tally[state] = tally.get(state, 0) + 1; bytes_ += size
            if state not in ("already", "copied"): print(state, job[1], flush=True)
    print(f"rap.{day}: {tally}, {bytes_ / 1e9:.1f} GB, {time.time() - t0:.0f} s", flush=True)
print("ARCHIVE_RAP_DONE", flush=True)
