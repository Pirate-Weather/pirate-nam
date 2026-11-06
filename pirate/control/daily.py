"""Run one Pirate NAM cycle a day on a throwaway GPU machine: create it from a snapshot, run the cycle, delete it.

This process does almost no work itself, so it fits a very small host. It wakes every CHECK_MINUTES and only acts when the day's
cycle isn't published and its inputs are out. A failed cycle publishes nothing, so the last good forecast stays live.

Every machine this creates is named PREFIX + cycle, and nothing without that prefix is ever deleted. Any such machine found while
no run is active is left over from a crash, and gets deleted, because it bills by the hour whether it's working or not.

Usage: daily.py                run forever
       daily.py --once         check once and exit, for a CronJob
       daily.py --cycle CYCLE  run this cycle now, whatever's published, then exit
Secrets, from the environment: OVH_CONFIG (file), NAM_SSH_KEY (file), OVH_PROJECT, OVH_USER, and optionally GH_TOKEN for failure issues.
"""
import fcntl, json, os, signal, subprocess, sys, time, urllib.error, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import ovh

E = os.environ.get
CYCLE_HOUR = int(E("CYCLE_HOUR", "0")); FLAVOR = E("GPU_FLAVOR", "t2-90"); REGION = E("GPU_REGION", "DE1"); IMAGE = E("GPU_IMAGE", "nam-auto-image")
PREFIX = E("GPU_PREFIX", "nam-auto-"); CARDS = int(E("GPU_CARDS", "2")); FCST_HOURS = int(E("FCST_HOURS", "84"))
CHECK_S = int(E("CHECK_MINUTES", "15")) * 60; MAX_RUN_S = int(float(E("MAX_RUN_HOURS", "6")) * 3600)
MAX_ATTEMPTS = int(E("MAX_ATTEMPTS", "2")); DEADLINE_H = int(E("DEADLINE_HOURS", "14"))
STATE = Path(E("STATE_DIR", "/state")); LOGS = STATE / "logs"; KEY = os.environ["NAM_SSH_KEY"]; PID = os.environ["OVH_PROJECT"]
REPO = E("GITHUB_REPO", "Pirate-Weather/pirate-nam"); HERE = Path(__file__).resolve().parent.parent
BUCKET_URL = "https://pirate-nam-tor.s3.ca-east-tor.io.cloud.ovh.net"
api = ovh.Client(config_file=os.environ["OVH_CONFIG"], timeout=(10, 60))
SSH_OPTS = ["-i", KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", "-o", "LogLevel=ERROR", "-o", "BatchMode=yes"]
HOUR = 3600
# OVH's API drops about 1 call in 12 at a 15 s spacing (measured 2026-11-06), and also answers with passing 500s.
API_ATTEMPTS = 5; API_RETRY_S = 10
# Booting the 67 GB snapshot (800 GB minimum disk) outlasted 24 min on 2026-11-06, and deleting a machine mid-build took over an hour.
BUILD_POLLS = int(float(E("BUILD_WAIT_MINUTES", "90")) * 60 / 15)


def log(*a): print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)

def cycle_time(c): return datetime.strptime(c, "%Y%m%d%H").replace(tzinfo=timezone.utc)

def newest_cycle(now):
    t = now.replace(hour=CYCLE_HOUR, minute=0, second=0, microsecond=0)
    return (t if t <= now else t - timedelta(days=1)).strftime("%Y%m%d%H")

def url_exists(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as r: return r.status == 200
    except (urllib.error.URLError, TimeoutError): return False

def published():
    with urllib.request.urlopen(f"{BUCKET_URL}/runs.json", timeout=30) as r: return {x["cycle"] for x in json.load(r)["runs"]}

def inputs_ready(c):
    """RAP's analysis-hour observations and the older GFS run's last boundary file: the same test run_cycle.sh makes."""
    g = (cycle_time(c) - timedelta(hours=6)).strftime("%Y%m%d%H")
    return (url_exists(f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{g[:8]}/{g[8:]}/atmos/gfs.t{g[8:]}z.pgrb2.0p25.f{FCST_HOURS + 6:03d}.idx")
            and url_exists(f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/rap.{c[:8]}/rap.t{c[8:]}z.prepbufr.tm00.nr"))

def load_state():
    try: return json.loads((STATE / "state.json").read_text())
    except (OSError, ValueError): return {}

def save_state(s):
    # Only the newest few cycles matter, and a half-written file must never be read back.
    s = dict(sorted(s.items())[-14:]); tmp = STATE / "state.json.tmp"; tmp.write_text(json.dumps(s, indent=1)); tmp.replace(STATE / "state.json")

def alert(title, body):
    """Failure notice. Public, so it carries only the cycle and step, never logs. Always also leaves a local record."""
    log("ALERT", title, "-", body)
    with open(STATE / "alerts.log", "a") as f: f.write(f"{datetime.now(timezone.utc).isoformat()} {title}: {body}\n")
    token = E("GH_TOKEN")
    if not token: return
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues", data=json.dumps({"title": title, "body": body}).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "pirate-nam-control"})
    try: urllib.request.urlopen(req, timeout=30).close()
    except (urllib.error.URLError, TimeoutError) as e: log("could not open the issue:", getattr(e, "code", type(e).__name__))

def api_transient(e):
    """Dropped connections, unreadable answers and 5xx. Not refusals like quota or stock, which the API answers with a 4xx."""
    if isinstance(e, (ovh.exceptions.HTTPError, ovh.exceptions.NetworkError, ovh.exceptions.InvalidResponse)): return True
    return type(e) is ovh.exceptions.APIError and getattr(e.response, "status_code", 0) >= 500

def api_retry(call, path, **params):
    """For GET and DELETE, which are safe to repeat. A lost answer mustn't end a run: that deletes a machine mid-build, which can stick."""
    for attempt in range(API_ATTEMPTS):
        try: return call(path, **params)
        except ovh.exceptions.APIError as e:
            if attempt == API_ATTEMPTS - 1 or not api_transient(e): raise
            log("OVH API:", type(e).__name__, "retrying"); time.sleep(API_RETRY_S)

def ours(): return [i for i in api_retry(api.get, f"/cloud/project/{PID}/instance") if i["name"].startswith(PREFIX)]

def delete_ours():
    """Delete every machine with our prefix, and confirm they've gone. Returns the names still there."""
    for i in ours():
        assert i["name"].startswith(PREFIX)
        if i["status"] == "DELETING": log(i["name"], "is already being deleted"); continue
        log("deleting", i["name"])
        try: api_retry(api.delete, f"/cloud/project/{PID}/instance/{i['id']}")
        except ovh.exceptions.APIError as e: log("delete refused:", str(e)[:120])
    for _ in range(20):
        left = ours()
        if not left: return []
        time.sleep(15)
    return [i["name"] for i in left]

def create(cycle):
    """Returns (id, ip), or None if the machine can't be had right now, which isn't the cycle's fault."""
    images = [s for s in api_retry(api.get, f"/cloud/project/{PID}/snapshot") if s["name"] == IMAGE and s["region"] == REGION and s["status"] == "active"]
    if not images: raise RuntimeError(f"no active snapshot {IMAGE} in {REGION}")
    image = max(images, key=lambda s: s["creationDate"])
    flavor = [f["id"] for f in api_retry(api.get, f"/cloud/project/{PID}/flavor", region=REGION) if f["name"] == FLAVOR and f["osType"] == "linux"]
    if not flavor: raise RuntimeError(f"no flavour {FLAVOR} in {REGION}")
    pub = subprocess.run(["ssh-keygen", "-y", "-f", KEY], check=True, capture_output=True, text=True).stdout.strip()
    try:
        r = api.post(f"/cloud/project/{PID}/instance", flavorId=flavor[0], imageId=image["id"], name=PREFIX + cycle, region=REGION, monthlyBilling=False,
                     userData=f"#cloud-config\nssh_authorized_keys:\n  - {pub}\n")
    except ovh.exceptions.APIError as e:
        # POST isn't safe to repeat: a lost answer may still have made the machine, so look for it instead of risking a second one.
        r = next((i for i in ours() if i["name"] == PREFIX + cycle), None) if api_transient(e) else None
        if r is None: log("machine refused:", str(e)[:160]); return None
        log("machine request lost its answer, but", r["name"], "exists")
    s = r; build_start = time.time()
    for _ in range(BUILD_POLLS):
        time.sleep(15); s = api_retry(api.get, f"/cloud/project/{PID}/instance/{r['id']}")
        if s["status"] != "BUILD": break
    if s["status"] != "ACTIVE":
        log("machine ended up", s["status"]); return None
    log(f"machine built in {(time.time() - build_start) / 60:.0f} min")
    return r["id"], next(a["ip"] for a in s["ipAddresses"] if a["version"] == 4 and a["type"] == "public")

def wait_for_ssh(ip):
    for _ in range(60):
        r = subprocess.run(["ssh", *SSH_OPTS, f"ubuntu@{ip}", "nvidia-smi -L | wc -l; docker images -q pirate-nam-publish | wc -l"], capture_output=True, text=True)
        if r.returncode == 0:
            lines = r.stdout.split()
            if int(lines[0]) >= CARDS and int(lines[1]) >= 1: return True
            log("machine isn't right:", r.stdout.split()); return False
        time.sleep(15)
    return False

def run_cycle(cycle, attempt):
    """One attempt, machine created and deleted inside. Returns "done", "no_machine", or the name of the step that failed."""
    LOGS.mkdir(parents=True, exist_ok=True); path = LOGS / f"{cycle}.{attempt}.log"
    Path.home().joinpath(".ssh/known_hosts").unlink(missing_ok=True)    # cloud addresses get reused, with new host keys
    try:
        made = create(cycle)
        if made is None: return "no_machine"
        _, ip = made; log("machine up at", ip)
        if not wait_for_ssh(ip): return "machine setup"
        env = {**os.environ, "FCST_HOURS": str(FCST_HOURS)}
        with open(path, "w") as out:
            p = subprocess.Popen(["bash", str(HERE / "prod/run_cycle.sh"), ip, cycle], stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try: p.wait(timeout=MAX_RUN_S)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait(); return "time limit"
            finally:
                if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
        text = path.read_text()
        for name in ("boco", "catchup", "fc"):    # the machine and its logs are about to go
            subprocess.run(["scp", "-q", *SSH_OPTS, f"ubuntu@{ip}:/woof/work/runs/{cycle}/{name}.log", str(LOGS / f"{cycle}.{attempt}.{name}.log")], capture_output=True)
        if "CYCLE_DONE" in text and cycle in published(): return "done"
        failed = [l for l in text.splitlines() if l.startswith("CYCLE_FAILED")]
        return failed[-1].split()[1].rstrip(",") if failed else "unknown"
    finally:
        # An exception here would hide the run's real result, and the next check retries the delete anyway.
        try: left = delete_ours()
        except ovh.exceptions.APIError as e: left = [f"{PREFIX}{cycle} ({type(e).__name__})"]
        if left: alert(f"Pirate NAM GPU still running: {left[0]}", f"Deleting {left} failed after cycle {cycle}. It bills until it's gone; the next check retries.")
        for old in sorted(LOGS.glob("*"))[:-120]: old.unlink()

def check():
    """One pass. Returns only when there's nothing left to do for now."""
    left = delete_ours()    # no run is active here, so anything of ours is a leftover
    if left: alert(f"Pirate NAM GPU still running: {left[0]}", f"Deleting leftover machines {left} failed.")
    now = datetime.now(timezone.utc); c = newest_cycle(now); state = load_state(); st = state.setdefault(c, {"attempts": 0, "no_machine": 0})
    (STATE / "heartbeat").write_text(now.isoformat())
    if c in published() or st.get("given_up"): return
    late = now > cycle_time(c) + timedelta(hours=DEADLINE_H)
    if st["attempts"] >= MAX_ATTEMPTS or late:
        reason = f"last step: {st.get('step')}, attempts: {st['attempts']}" if st["attempts"] else f"never started: inputs or a GPU weren't available, {st['no_machine']} refusals"
        alert(f"Pirate NAM {c} not published", reason); st["given_up"] = True; save_state(state); return
    if not inputs_ready(c): log(c, "inputs aren't out yet"); return
    st["attempts"] += 1; save_state(state)
    log(c, "attempt", st["attempts"])
    try: result = run_cycle(c, st["attempts"])
    except ovh.exceptions.APIError as e:
        # Only making the machine calls OVH inside a run, so no forecast work was lost. Like no stock, this isn't the cycle's failure.
        log(c, "OVH API failed:", type(e).__name__, str(e)[:160]); result = "no_machine"
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as e:
        result = "control: " + type(e).__name__; log(c, "error:", str(e)[:200])
    log(c, "result:", result)
    if result == "no_machine": st["attempts"] -= 1; st["no_machine"] += 1    # capacity isn't a cycle failure, so it doesn't use up an attempt
    else: st["step"] = result
    save_state(state)

def main():
    STATE.mkdir(parents=True, exist_ok=True)
    lock = open(STATE / "lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("another copy is running")
    def stop(*_): raise SystemExit(0)    # raised here so run_cycle's finally deletes the machine on docker stop
    signal.signal(signal.SIGTERM, stop)
    if "--cycle" in sys.argv:
        c = sys.argv[sys.argv.index("--cycle") + 1]; delete_ours(); r = run_cycle(c, "manual"); log(c, "result:", r); sys.exit(0 if r == "done" else 1)
    if "--once" in sys.argv: check(); return
    log("started: cycle", f"{CYCLE_HOUR:02d}Z,", FLAVOR, "in", REGION, "from", IMAGE)
    while True:
        try: check()
        except (OSError, ValueError, RuntimeError, urllib.error.URLError, ovh.exceptions.APIError) as e:
            # A passing network or API problem shouldn't end the loop. The message is logged, and the next check tries again.
            log("check failed:", type(e).__name__, str(e)[:200])
        time.sleep(CHECK_S)

if __name__ == "__main__": main()
