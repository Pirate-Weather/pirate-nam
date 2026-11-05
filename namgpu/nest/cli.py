"""woof-nam nest: a WOOF-NAM nest anywhere on Earth from one command.

    tools/woof-nam nest --center 41.0S,174.0E --size-km 1000 --res-km 3 --source gfs \
        --start 2026110312 --hours 48 --out runs/nest-nz [--cards N] [--spec-adv] [--no-maps]

--center takes LAT,LON as signed degrees or with N/S/E/W. --source auto (default) picks from
domain.SOURCE_BY_REGION by centre and start date (gfs over North America and the US territories; elsewhere ifs
from 2026-10-20 12Z, ifs-0p25 from 2026-02-11 12Z, gfs before). Boundaries: a past start (the start cycle's own
forecast already published, sources row published_after_h) takes them from that same cycle (age 0); a live start
takes the row's operational age (gfs 6 h as NAM). --boco-age-h overrides both.

Stages (each timed, with its CPU cores sampled every 2 s and the process tree's VRAM from nvidia-smi):
  domain     the configfile <out>/nam_configfile_<name> from domain.RES_TABLE (no CUDA)
  preflight  free VRAM (nvidia-smi, no context) against domain.vram_estimate: picks NAMGPU_RRTM_ROWS or refuses
  warm       kernels.warm: every NVRTC module compiled into the cache on all cores (a no-op read when warm)
  static     init-static-build on the GPU, cached in <static cache>/<name>-<grid key>-<YYYYMMDD>.npz
  fetch      the source row's fetch (FETCH table) into <cases root>/<YYYY-MM-DD_HHz>/raw/ (shared by every domain
             of that cycle; existing files are kept)
  forecast   tools/run_forecast.py: the init chain (GFS start, bocos into boco-<name>/), TDFI (filter_method 3,
             drv_dfi_tdfi_driver through drv-time-loop; tdfi_s in the receipt), the forecast, history, with the CuPy
             pool capped at the pre-flight's pool_limit for the memory path the forecast takes on the card (fit
             below 64 GiB, full at and above; CUPY_GPU_MEMORY_LIMIT); --cards N > 1 saves the init once
             (slab_init) and loads it per rank
  maps       run_forecast's post-run maps tail (rw_wrfbatch, no CUDA context)
Receipt: <out>/nest-receipt.json (grid, keys, sources and licences, the static build record, ms/step, peak VRAM,
wall and cores per stage, single-core windows over 30 s, wall per 48 h).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from namgpu.nest import domain

ROOT = domain.ROOT
CASES = Path(os.environ.get("NAMGPU_CASES", str(ROOT / "cases")))
STATIC_CACHE = Path(os.environ.get("NAMGPU_STATIC_CACHE", str(ROOT / "static" / "nest-cache")))

# Source rows the door can run: name -> (fetch callable "module:function", licence line). The fetch itself is
# table work in namgpu/init/fetch_case.py FETCH_ROWS (lane init-gfs-anywhere) and sources.SOURCES; a source is
# runnable when sources.SOURCES has its row. fetch(cycle, case_dir, hours, cfg, args) -> {"failed": [...], ...}.
FETCH = {
    "gfs": ("namgpu.nest.cli:fetch_rows", "NOAA NCEP GFS 0.25 deg, US Government work, public domain"),
    "ifs": ("namgpu.nest.cli:fetch_rows", "ECMWF open data (IFS HRES 0.1 deg), CC BY 4.0; sea ice from NOAA GFS "
            "f000 (public domain)"),
    "ifs-0p25": ("namgpu.nest.cli:fetch_rows", "ECMWF open data (IFS HRES 0.25 deg), CC BY 4.0; sea ice from "
                 "NOAA GFS f000 (public domain)"),
    "parent": ("namgpu.nest.cli:fetch_parent", "WOOF-NAM 12 km history (ours)"),
}
SINGLE_CORE_S = 30.0


# ------------------------------------------------------------------------------------------ args

def parse_center(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--center {text!r}: want LAT,LON (e.g. 38.6,-98.9 or 41S,174E)")
    out = []
    for p, (pos, neg) in zip(parts, (("N", "S"), ("E", "W"))):
        m = re.fullmatch(r"([+-]?\d+(?:\.\d*)?)\s*([NSEWnsew]?)", p)
        if not m:
            raise SystemExit(f"--center {text!r}: cannot read {p!r}")
        v = float(m.group(1))
        h = m.group(2).upper()
        if h and h not in (pos, neg):
            raise SystemExit(f"--center {text!r}: {p!r} has hemisphere {h}, expected {pos}/{neg}")
        out.append(-v if h == neg else v)
    return out[0], out[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="woof-nam", description="WOOF-NAM on the GPU")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nest", help="a nest anywhere: centre, size, resolution -> forecast, maps, receipt")
    n.add_argument("--center", required=True, help="LAT,LON (signed degrees or with N/S/E/W)")
    n.add_argument("--size-km", type=float, default=1000.0)
    n.add_argument("--size-y-km", type=float, default=None, help="north-south size (default = --size-km)")
    n.add_argument("--res-km", type=float, default=3.0, help=f"one of {sorted(domain.RES_TABLE)} (RES_TABLE)")
    n.add_argument("--source", default="auto", help="auto | " + " | ".join(FETCH))
    n.add_argument("--parent-run", type=Path, default=None, help="--source parent: the 12 km run folder")
    n.add_argument("--parent-credit", default=None, help="--source parent with a parent history written before its "
                   "start-source attribution was recorded: what the parent started from, e.g. 'ECMWF open data, CC "
                   "BY 4.0' (the maps must carry it)")
    n.add_argument("--no-handoff", action="store_true", help="a 12 km (parent row) run: do not write the parent "
                   "hand-off set (history_handoff) into its history")
    n.add_argument("--start", required=True, help="YYYYMMDDHH (00/06/12/18 for gfs/ifs)")
    n.add_argument("--hours", type=float, default=48.0)
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--cards", type=int, default=1)
    n.add_argument("--card-list", default=os.environ.get("NAM_CARDS", ""))
    n.add_argument("--spec-adv", action="store_true", help="NAM's nest value spec_adv true (not yet run in the port)")
    n.add_argument("--rrtm-rows", type=int, default=None, help="force NAMGPU_RRTM_ROWS (default: pre-flight picks)")
    n.add_argument("--no-maps", action="store_true")
    n.add_argument("--no-warm", action="store_true")
    n.add_argument("--dry-run", action="store_true", help="domain, pre-flight and static only")
    n.add_argument("--wait-min", type=float, default=0.0, help="shared card: if the nest does not fit now, fetch "
                   "anyway and wait up to this many minutes for free memory before the forecast (never kills anyone)")
    n.add_argument("--cases-root", type=Path, default=CASES)
    n.add_argument("--boco-age-h", type=int, default=None, help="boundary cycle age in hours (default: 0 when the "
                   "start cycle's forecast is already published, else the source row's, 6 for gfs as NAM)")
    n.add_argument("--no-sheets", action="store_true")
    from namgpu.nest import twoway_door           # --two-way / --concurrent (a parent and this nest in one process)
    twoway_door.add_args(n)
    return ap


# ------------------------------------------------------------------------------------------ sampler

def _proc_tree(root: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            ppid = int(s[s.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _cpu_s(pid: int) -> float | None:
    """utime + stime + cutime + cstime of one live process (its own CPU plus its reaped children's), s."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        v = s[s.rindex(")") + 2:].split()
        return (int(v[11]) + int(v[12]) + int(v[13]) + int(v[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _gpu_mib(pids) -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    want = set(pids)
    tot = 0.0
    for line in out.splitlines():
        try:
            p, m = (x.strip() for x in line.split(","))
            if int(p) in want:
                tot += float(m)
        except ValueError:
            pass
    return tot


def _gpu_util(pids=None) -> float:
    """SM % of the visible card used by THIS run's processes (nvidia-smi pmon, one 1 s sample, no CUDA context);
    -1 when unknown. Per process, not the whole card: on a shared card another lane keeps the card busy, and the
    whole-card number hid a one-core stage of this run while the card waited on it (fix round 1, 2026-11-05)."""
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    try:
        out = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u", "-i", idx],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return -1.0
    if not out.strip():
        return -1.0
    want, tot = set(pids or ()), 0.0
    for line in out.splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 4:
            continue
        try:
            if int(f[1]) in want and f[3] != "-":
                tot += float(f[3])
        except ValueError:
            pass
    return tot


def _cpu_idle_ticks() -> tuple[float, float]:
    """(idle + iowait, total) jiffies of the whole host (/proc/stat cpu line)."""
    try:
        with open("/proc/stat") as f:
            v = [float(x) for x in f.readline().split()[1:]]
        return v[3] + v[4], sum(v[:8])
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0


class Sampler:
    """Every 2 s: CPU seconds of this process tree (cores = delta CPU / delta wall) and its VRAM (nvidia-smi, every
    4 s). Each process counts its own plus its reaped children's time, so a process that exits moves its total
    into its parent: the sampler subtracts the last value it saw of every process that disappeared. Stage labels
    come from mark()."""

    def __init__(self, period=2.0):
        self.period, self.samples, self.stage = period, [], "start"
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()
        return self

    def mark(self, stage):
        self.stage = stage

    def _snap(self):
        out = {}
        for p in _proc_tree(os.getpid()):
            v = _cpu_s(p)
            if v is not None:
                out[p] = v
        return out

    def _loop(self):
        last_t, last = time.perf_counter(), self._snap()
        last_idle, ncpu = _cpu_idle_ticks(), os.cpu_count() or 1
        k = 0
        while not self._stop.wait(self.period):
            t, now = time.perf_counter(), self._snap()
            d = sum(v - last.get(p, 0.0) for p, v in now.items()) - sum(v for p, v in last.items() if p not in now)
            gpu = _gpu_mib(list(now)) if k % 2 == 0 else (self.samples[-1]["vram_mib"] if self.samples else 0.0)
            idle = _cpu_idle_ticks()
            dt_tot = idle[1] - last_idle[1]
            idle_cores = round(ncpu * (idle[0] - last_idle[0]) / dt_tot, 1) if dt_tot > 0 else -1.0
            self.samples.append({"t": round(t, 2), "stage": self.stage, "cores": round(max(0.0, d) / (t - last_t), 2),
                                 "vram_mib": gpu, "gpu_util": _gpu_util(now), "idle_cores": idle_cores,
                                 "nproc": len(now)})
            last_t, last, last_idle, k = t, now, idle, k + 1

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)

    def summary(self) -> dict:
        by: dict[str, dict] = {}
        for s in self.samples:
            b = by.setdefault(s["stage"], {"samples": 0, "cores_sum": 0.0, "cores_max": 0.0, "vram_peak_mib": 0.0})
            b["samples"] += 1
            b["cores_sum"] += s["cores"]
            b["cores_max"] = max(b["cores_max"], s["cores"])
            b["vram_peak_mib"] = max(b["vram_peak_mib"], s["vram_mib"])
        for b in by.values():
            b["cores_mean"] = round(b.pop("cores_sum") / max(1, b["samples"]), 2)
        # single-core windows: consecutive samples at 0.7..1.3 cores of this run for more than 30 s while the card
        # waits on this run (this run's own SM share under 20 %, pmon per pid: another tenant's load does not hide
        # it) or while other cores sit idle (2 or more idle host cores, own share under 50 %). One host core
        # driving the card (the forecast's launch loop at a busy own share) is not one.
        wins, cur = [], None
        for s in self.samples:
            own = s.get("gpu_util", -1.0)
            waits = own < 20.0
            idle = s.get("idle_cores", -1.0) >= 2.0 and own < 50.0
            if 0.7 <= s["cores"] <= 1.3 and (waits or idle):
                cur = cur or {"stage": s["stage"], "t0": s["t"], "why": "card waits" if waits else "cores idle"}
                cur["t1"] = s["t"]
            else:
                if cur and cur["t1"] - cur["t0"] + self.period > SINGLE_CORE_S:
                    wins.append(cur)
                cur = None
        if cur and cur["t1"] - cur["t0"] + self.period > SINGLE_CORE_S:
            wins.append(cur)
        t0 = self.samples[0]["t"] if self.samples else 0.0
        return {"by_stage": by, "single_core_windows": [
            {"stage": w["stage"], "from_s": round(w["t0"] - t0, 1), "length_s": round(w["t1"] - w["t0"] + self.period, 1),
             "why": w["why"]} for w in wins], "vram_peak_mib": max((s["vram_mib"] for s in self.samples), default=0.0)}


# ------------------------------------------------------------------------------------------ fetch rows

def fetch_rows(cycle: datetime, case_dir: Path, hours: int, cfg, args) -> dict:
    """The source's rows of fetch_case.FETCH_ROWS (init, boundaries, and the referee/truth rows whose region
    holds the domain), byte-range subsets in a 16-thread pool."""
    from namgpu.init import fetch_case
    lines = []
    bad = fetch_case.fetch(cycle, case_dir, hours, 16, cfg, args.source_name, args.boco_age_h,
                           log=lambda m: (lines.append(m), print(m, flush=True)))
    raw = Path(case_dir) / "raw"
    nbytes = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file()) if raw.is_dir() else 0
    return {"failed": [l for l in lines if l.startswith("FAILED") and "(optional)" not in l] if bad else [],
            "optional_failed": [l for l in lines if l.startswith("FAILED (optional)")],
            "skipped_rows": [l for l in lines if l.startswith("fetch-case: skip")],
            "case_raw_bytes": nbytes}


def fetch_parent(cycle: datetime, case_dir: Path, hours: int, cfg, args) -> dict:
    """The parent is our own run on this disk: check its hand-off history covers start .. start + hours."""
    run = args.parent_run
    if run is None or not Path(run).is_dir():
        raise SystemExit(f"--source parent needs --parent-run DIR (a WOOF-NAM run written with history_handoff: "
                         f"true); got {run}")
    need = [cycle + timedelta(hours=h) for h in range(0, hours + 1, 3)]
    miss = [f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}" for t in need
            if not (Path(run) / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}").is_file()]
    return {"failed": [f"parent history missing: {miss[:4]}{' ...' if len(miss) > 4 else ''}"] if miss else [],
            "parent_run": str(Path(run).resolve())}


def resolve_fetch(name: str):
    if name not in FETCH:
        raise SystemExit(f"--source {name}: no row in namgpu/nest/cli.py FETCH (rows: {sorted(FETCH)})")
    from namgpu.init import sources
    if name not in sources.SOURCES:
        raise SystemExit(f"--source {name}: namgpu/init/sources.py has no '{name}' row in this tree (rows: "
                         f"{sorted(sources.SOURCES)})")
    target, lic = FETCH[name]
    mod, fn = target.split(":")
    return getattr(importlib.import_module(mod), fn), lic


# ------------------------------------------------------------------------------------------ stages

def run_logged(cmd, env, log: Path, sampler: Sampler, marks=()):
    """Run cmd with output to log (and stdout), switching the sampler's stage label when a line matches."""
    with open(log, "a") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             cwd=str(ROOT))
        stamps = {}
        for line in p.stdout:
            lf.write(line)
            lf.flush()                       # the stage logs are watched live (box owners, status files)
            sys.stdout.write(line)
            sys.stdout.flush()
            for pat, stage in marks:
                if stage not in stamps and re.search(pat, line):
                    stamps[stage] = time.perf_counter()
                    sampler.mark(stage)
        return p.wait(), stamps


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd != "nest":
        return 2
    # A live parent (--two-way, --concurrent, or any --parent-* option) runs two-way unless --concurrent asks for the
    # feedback off. The --source parent hand-off and the one-domain door below stay one-way.
    if a.two_way or a.concurrent or a.parent_center or a.parent_size_km or a.parent_size_y_km:
        a.two_way = a.two_way or not a.concurrent
        from namgpu.nest import twoway_door
        return twoway_door.main(a, argv)
    t_all = time.perf_counter()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = parse_center(a.center)
    cycle = datetime.strptime(a.start, "%Y%m%d%H")
    hours = float(a.hours)
    source, region = (domain.default_source(lat, lon, cycle) if a.source == "auto" else (a.source, "given"))
    if a.source != "auto" and source in FETCH and source != "parent" and not domain.serves(source, cycle):
        from namgpu.init import sources as _src
        r_ = _src.SOURCES.get(source, {})
        raise SystemExit(f"--source {source}: no files for {cycle:%Y-%m-%d %H}Z (row from {r_.get('valid_from')})"
                         + (f"; use --source {r_['before_use']} or auto" if r_.get("before_use") else ""))
    if source == "parent":
        a.parent_run = resolve_parent_run(a.parent_run)
    boco_age, boco_why = boco_age_for(source, cycle, a.boco_age_h)
    credit, credit_lines = data_credits(source, parent_run=a.parent_run, given=a.parent_credit)
    sampler = Sampler().start()
    stages, rec = {}, {"door": "woof-nam nest", "argv": sys.argv[1:] if argv is None else list(argv),
                       "start": cycle.isoformat(), "hours": hours, "source": source, "source_region": region,
                       "boco_age_h": boco_age, "boco_age_why": boco_why, "data_credit": credit,
                       "data_licences": credit_lines, "host": os.uname().nodename}

    def stage(name):
        sampler.mark(name)
        stages[name] = {"t0": time.perf_counter()}

    def done(name, **kw):
        s = stages[name]
        s["wall_s"] = round(time.perf_counter() - s.pop("t0"), 2)
        s.update(kw)
        print(f"woof-nam: {name} {s['wall_s']} s", flush=True)

    # domain
    stage("domain")
    g = domain.nest_grid(lat, lon, a.size_km, a.res_km, a.size_y_km)
    name = domain.domain_name(g, a.size_km, a.size_y_km)
    STATIC_CACHE.mkdir(parents=True, exist_ok=True)
    # the key names the dataset table that builds the file (static_build.datasets_key): a static built from other
    # data (the FAO soil / ISLSCP deep-soil T builds from before static-world) is never reused under a new table
    from namgpu.init import static_build as _sb
    static = STATIC_CACHE / f"{name}-{domain.grid_key(g)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    cfgfile = out / f"nam_configfile_{name}"
    # a parent-row run (12 km) writes the hand-off set a --source parent nest reads (parent_start.parent_grid), so the
    # door can make a parent; without it every 12 km door run was refused as a parent ("carries no parent hand-off
    # set") and the only parent was a hand-edited configfile (fix round 2). Several cards: the history merge has no
    # soil-layer dimension (io_history_fields.handoff_fields refuses), so the run goes ahead without it and says so.
    handoff = bool(domain.RES_TABLE[g["res_km"]].get("handoff")) and not a.no_handoff
    if handoff and a.cards > 1:
        handoff = False
        rec["handoff"] = ("off: --cards > 1 (tools/history_merge.py writes no soil_layers_stag dimension; "
                          "io_history_fields.handoff_fields is one-card only)")
        print(f"woof-nam: {rec['handoff']}", flush=True)
    keys = domain.write_configfile(cfgfile, g, name, static, a.spec_adv, handoff=handoff)
    rec.update(domain=name, grid=g, configfile=str(cfgfile), keys=keys)
    rec.setdefault("handoff", "on" if handoff else "off")
    done("domain")
    print(f"woof-nam: {name}: {g['im']} x {g['jm']} x {g['lm']}, {g['extent_km']} km, dt {keys['dt']} s, "
          f"{keys['steps_per_hour']:.0f} steps/h, source {source} ({region})", flush=True)

    # preflight: every card the run will use (the visible card for one, --card-list or 0..N-1 for several)
    stage("preflight")
    if a.cards > 1:
        cards = [x.strip() for x in a.card_list.split(",") if x.strip()] or [str(i) for i in range(a.cards)]
        if len(cards) < a.cards:
            raise SystemExit(f"woof-nam: --cards {a.cards} but --card-list has {len(cards)} cards: {cards}")
        cards = cards[:a.cards]
    else:
        cards = [domain._visible_index()]
    pf = domain.preflight_cards(g, cards, a.rrtm_rows)
    rec["preflight"] = pf
    done("preflight")
    if not pf["ok"]:
        msg = (f"woof-nam: {'does not fit now' if a.wait_min > 0 else 'refused'}: the nest needs about {pf['tried'][-1]['total_gib']} GiB at NAMGPU_RRTM_ROWS "
               f"{pf['tried'][-1]['rrtm_rows']} (+{domain.VRAM['margin_gib']} margin) and the card has "
               f"{pf['free_gib']} GiB free ({pf['card']}); estimates tried: {pf['tried']}")
        print(msg, flush=True)
        rec["refused_at_start"] = msg
        if not a.dry_run and a.wait_min <= 0:
            rec["refused"] = msg
            _write(out, rec, stages, sampler, t_all)
            return 3
    rows = pf["pick"]["rrtm_rows"] if pf["ok"] else (a.rrtm_rows or 8)
    if pf["ok"]:
        print(f"woof-nam: pre-flight {pf.get('free_gib')} GiB free (by card {pf['free_by_card']}), estimate "
              f"{pf['pick']['total_gib']} GiB per card at NAMGPU_RRTM_ROWS {rows}, {pf['pick']['memory_path']} memory "
              f"path, pool cap {pf['pick']['pool_limit_gib']} GiB", flush=True)

    # renderer: the maps tail and the sheets both draw with namgpu.io.maps.renderer() (tools/renderer/bin is
    # gitignored). Checked before the forecast: on a fresh clone the door ran the whole forecast (10-46 min) and then
    # died at maps with SystemExit, rc 1, no failed_stage and an unfinished receipt (review round 3, #3).
    if not (a.no_maps and a.no_sheets):
        stage("renderer")
        from namgpu.io import maps as _maps
        try:
            rw = _maps.renderer()
        except SystemExit as e:
            done("renderer", ok=False)
            rec["failed_stage"] = f"renderer: {e}"
            print(f"woof-nam: refused before the forecast: {rec['failed_stage']}", flush=True)
            _write(out, rec, stages, sampler, t_all)
            return 6
        done("renderer", ok=True, path=str(rw))

    # warm
    if not a.no_warm:
        stage("warm")
        from namgpu import kernels
        w = kernels.warm()
        done("warm", **{k: w[k] for k in ("workers", "modules", "compile_s_sum", "failed", "slowest")})
        if w["failed"]:
            print(f"woof-nam: kernel compile failures {w['failed']}", flush=True)

    env = dict(os.environ, NAMGPU_CONFIGFILE=str(cfgfile), NAMGPU_RRTM_ROWS=str(rows), PYTHONUNBUFFERED="1",
               NAMGPU_DATA_CREDIT=credit, NAMGPU_DATA_LICENCES=json.dumps(credit_lines), NAMGPU_START_SOURCE=source,
               PYTHONPATH=str(ROOT) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    env["NAMGPU_BOCO_AGE_H"] = str(boco_age)
    a.boco_age_h = boco_age
    if pf.get("pick"):
        _pool_cap(env, rec, pf["pick"])

    # static
    stage("static")
    stale = _static_stale(static) if static.is_file() else None
    if stale:
        print(f"woof-nam: cached static {static.name} not reused: {stale}", flush=True)
        rec["static_rebuilt_because"] = stale
    if static.is_file() and not stale:
        done("static", cached=True, path=str(static))
    else:
        code, _ = run_logged([sys.executable, "-m", "namgpu.init.static_build", "--config", str(cfgfile),
                              "--date", f"{cycle:%Y-%m-%d}", "--out", str(static)], env, out / "static.log", sampler)
        done("static", cached=False, path=str(static), rc=code)
        if code:
            _write(out, rec, stages, sampler, t_all)
            return code
    rec["static_meta"] = _static_meta(static)
    if a.dry_run:
        _write(out, rec, stages, sampler, t_all)
        return 0

    # fetch
    stage("fetch")
    case = f"{cycle:%Y-%m-%d_%H}z"
    case_dir = (a.cases_root / case).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    fetch, lic = resolve_fetch(source)
    from namgpu import config as configmod
    a.source_name = source
    fr = fetch(cycle, case_dir, max(6, int(hours + 0.999)), configmod.load(cfgfile), a)
    rec["source_licence"] = lic
    done("fetch", **{k: v for k, v in fr.items()})
    if fr.get("failed"):
        print(f"woof-nam: fetch failed: {fr['failed']}", flush=True)
        _write(out, rec, stages, sampler, t_all)
        return 4

    # the card must have room now (a shared card: wait for it, never take memory from another process).
    # Checked again here even when the pre-flight passed: the fetch takes minutes, and on a shared card another run took
    # 29.8 GiB of the card during a 192 s fetch, so the init died with OutOfMemoryError in init-grib2-decode
    # (nest/all smoke 2026-11-04 23:31Z) instead of waiting or refusing by name.
    # Every card of a --cards N run is checked the same way (it was skipped for N > 1, review round 3, #2).
    if True:
        stage("wait-vram")
        t_lim = time.perf_counter() + a.wait_min * 60
        while True:
            pf = domain.preflight_cards(g, cards, a.rrtm_rows)
            if pf["ok"] or time.perf_counter() > t_lim:
                break
            time.sleep(30)
        done("wait-vram", free_gib=pf["free_gib"], free_by_card=pf["free_by_card"], ok=pf["ok"])
        rec["preflight_at_forecast"] = pf
        if not pf["ok"]:
            rec["refused"] = (f"waited {a.wait_min} min: the fullest card still has {pf['free_gib']} GiB free (by card "
                              f"{pf['free_by_card']}), the nest needs about {pf['tried'][-1]['total_gib']} GiB per card")
            print("woof-nam: refused:", rec["refused"], flush=True)
            _write(out, rec, stages, sampler, t_all)
            return 3
        rows = pf["pick"]["rrtm_rows"]
        env["NAMGPU_RRTM_ROWS"] = str(rows)
        _pool_cap(env, rec, pf["pick"])
        print(f"woof-nam: card has room: {pf['free_gib']} GiB free, NAMGPU_RRTM_ROWS {rows}", flush=True)

    # forecast (init + bocos + TDFI + run) and maps
    run_dir = out / "run"
    cmd = [sys.executable, "-m", "namgpu.run_forecast", "--case", case, "--case-dir", str(case_dir),
           "--source", source, "--hours", str(hours), "--out", str(run_dir)]
    if a.no_maps:
        cmd.append("--no-maps")
    if source == "parent":
        cmd += ["--parent-run", str(Path(a.parent_run).resolve())]
    init_tmp = None
    if a.cards > 1:
        stage("init-save")
        init_dir = Path(os.environ.get("NAMGPU_INIT_DIR", f"/dev/shm/{name}-init"))
        if "NAMGPU_INIT_DIR" not in os.environ:
            init_tmp = init_dir      # the door-made copy in /dev/shm (host RAM): removed when the run ends
        # the init runs on the first card of the list, which the pre-flight checked
        code, _ = run_logged([sys.executable, "-m", "namgpu.init.slab_init", "save", "--case", case, "--case-dir",
                              str(case_dir), "--source", source, "--hours", str(hours), "--out", str(init_dir),
                              "--managed"], dict(env, CUDA_VISIBLE_DEVICES=cards[0]), out / "init-save.log", sampler)
        done("init-save", rc=code)
        if code:
            _rm_init(init_tmp, rec)
            _write(out, rec, stages, sampler, t_all)
            return code
        cmd += ["--cards", str(a.cards), "--init-from", str(init_dir)]
        if a.card_list:
            cmd += ["--card-list", a.card_list]
    stage("init")
    marks = [(r"^init done ", "forecast"), (r"^card released", "maps"), (r"^launched \d+ ranks", "forecast")]
    code, stamps = run_logged(cmd, env, out / "run.log", sampler, marks)
    _rm_init(init_tmp, rec)
    t_end = time.perf_counter()
    t_init0 = stages["init"]["t0"]
    t_fc = stamps.get("forecast", t_end)
    t_maps = stamps.get("maps", t_end)
    stages["init"] = {"wall_s": round(t_fc - t_init0, 2)}
    stages["forecast"] = {"wall_s": round(t_maps - t_fc, 2)}
    if "maps" in stamps:
        stages["maps"] = {"wall_s": round(t_end - t_maps, 2)}
    rec["forecast_rc"] = code
    rp = run_dir / "forecast-receipt.json"
    if rp.is_file():
        fr = json.loads(rp.read_text())
        rec["forecast"] = {k: fr.get(k) for k in ("init_s", "tdfi_s", "filter_method", "steps_done", "run_s",
                                                  "ms_per_step", "nonfinite_end",
                                                  "after_init_used_gib", "pool_end_gib", "card_used_gib", "error",
                                                  "maps_s", "maps_error", "ms_per_step_by_rank",
                                                  "nonfinite_end_by_rank", "wall_s")}
        rec["forecast"]["init_stages"] = (fr.get("init") or {}).get("stages") if isinstance(fr.get("init"), dict) else None
        rec["forecast"]["hour_lines"] = fr.get("hour_lines", [])[-3:]
        if code and fr.get("maps_error") and fr.get("exit_code", 0) == 0:
            rec["failed_stage"] = f"maps: {fr['maps_error']}"
        steps, run_s = fr.get("steps_done"), fr.get("run_s")
        if steps and run_s:
            rec["forecast"]["wall_per_48h_min"] = round(run_s / steps * 48 * keys["steps_per_hour"] / 60, 1)

    # sheets: referee | WOOF-NAM | source (or parent), every output hour, off the card on all cores
    # (several cards too: history_merge writes the same wrfout_d01_* set with the same writer; the stage was skipped
    # for --cards N > 1 with nothing in the receipt, review round 3, #2)
    if code != 0 or a.no_sheets:
        rec["sheets"] = "skipped: " + ("--no-sheets" if a.no_sheets else f"forecast rc {code}")
    if code == 0 and not a.no_sheets:
        stage("sheets")
        hrs = int(hours)
        # a checkout draws through tools/nam_domain_sheets.py; an installed wheel calls the same nest_sheets directly
        sheets_tool = ROOT / "tools" / "nam_domain_sheets.py"
        if sheets_tool.is_file():
            sheets_cmd = [sys.executable, str(sheets_tool), "--run", str(run_dir), "--nest", source,
                          "--out", str(out / "sheets"), "--hours", f"1-{hrs}", "--start", f"{cycle:%Y-%m-%dT%H}",
                          "--cache", str(case_dir / "referee-cache")]
        else:
            sheets_cmd = [sys.executable, "-m", "namgpu.io.io_referee_sheets", "nest", "--history",
                          *sorted(str(p) for p in run_dir.glob("wrfout_d01_*")), "--source", source,
                          "--out", str(out / "sheets"), "--leads", f"1-{hrs}", "--cache", str(case_dir / "referee-cache")]
        sc, _ = run_logged(sheets_cmd + (["--parent-run", str(Path(a.parent_run).resolve())] if source == "parent" else []),
                           env, out / "sheets.log", sampler)
        done("sheets", rc=sc, dir=str(out / "sheets"))
        if sc:
            # the door exited 0 while its sheets stage had died (Gabrielle fix round 1: sheets rc 1 in the receipt,
            # "Door rc 0" in the report); a failed stage is the door's exit code now
            rec["failed_stage"] = f"sheets rc {sc} (see {out / 'sheets.log'})"
            print(f"woof-nam: {rec['failed_stage']}", flush=True)
            code = 5
    _write(out, rec, stages, sampler, t_all)
    return code


def _pool_cap(env: dict, rec: dict, pick: dict) -> None:
    """Cap the CuPy pool at the pre-flight's pool_limit for the memory path the forecast takes on the card (CuPy frees
    its cached blocks and retries at the cap, so the run stays inside what was checked). No cap when the path is
    unknown (domain.vram_estimate gives pool_limit_bytes None): a cap from the wrong path's model killed the Milton
    12 km parent on a 96 GB card (2026-11-05)."""
    rec["memory_path"] = pick.get("memory_path")
    if pick.get("pool_limit_bytes"):
        env["CUPY_GPU_MEMORY_LIMIT"] = str(pick["pool_limit_bytes"])
        rec["pool_limit_gib"] = pick["pool_limit_gib"]
    else:
        env.pop("CUPY_GPU_MEMORY_LIMIT", None)
        rec["pool_limit_gib"] = None
        rec["pool_limit_why_none"] = "memory path unknown (no card size from nvidia-smi, no NAMGPU_MEMORY_PATH)"


def _rm_init(init_dir, rec: dict) -> None:
    """Remove the door-made /dev/shm init copy (a whole-domain init in host RAM; it was never removed)."""
    if init_dir is None:
        return
    import shutil
    shutil.rmtree(init_dir, ignore_errors=True)
    rec["init_dir_removed"] = str(init_dir)


def _static_meta(static: Path) -> dict:
    try:
        import numpy as np
        z = np.load(static)
        meta = json.loads(str(z["META"])) if "META" in z.files else {}
        prov = static.with_name(static.name + ".PROVENANCE.json")      # static_build.write's name
        if prov.is_file():
            meta["provenance"] = json.loads(prov.read_text())
        return meta
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _static_stale(static: Path) -> str | None:
    """Why a cached static must be rebuilt, or None: its META must carry the dataset table this tree builds with
    (static_build.DATASETS), none of it held. The breakage: a cache file from before static-world (FAO DSMW soil,
    ISLSCP deep-soil T, no META datasets) reused silently."""
    from namgpu.init import static_build as sb
    meta = _static_meta(static)
    if "error" in meta:
        return f"unreadable ({meta['error']})"
    ds = meta.get("datasets")
    if not ds:
        return "META has no datasets table (built before static-world)"
    names = {r.get("role"): r.get("dataset") for r in ds}
    held = [n for n in names.values() if n in sb.HELD]
    if held:
        return f"built from held datasets {held}"
    want = {k: v["dataset"] for k, v in sb.DATASETS.items()}
    if names != want:
        return f"datasets {names} != this tree's {want}"
    return None


# ------------------------------------------------------------------------------------------ boundaries, credits

def boco_age_for(source: str, cycle: datetime, given: int | None) -> tuple[int, str]:
    """Boundary cycle age (h) and why. A past start whose own cycle forecast is published (now >= cycle + the
    row's published_after_h) takes its boundaries from that same cycle: the 6 h older GFS cycle cost the Japan
    nest 71 vs 23 km track error at f24 and 171 vs 76 km at f36 (2026-11-04). A live start
    keeps the row's operational age (NAM's 6 h)."""
    if given is not None:
        return int(given), "given (--boco-age-h)"
    from namgpu.init import sources
    row = sources.SOURCES.get(source) or {}
    b = row.get("boundary") or {}
    age = int(b.get("age_h", row.get("boco_age_h", 0)) or 0)
    if not age:
        return 0, f"the {source} row's own (start and boundaries are one forecast)"
    pub = float(b.get("published_after_h", row.get("published_after_h", 6)))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if now >= cycle + timedelta(hours=pub):
        return 0, (f"past start: the {cycle:%Y-%m-%d %H}Z {source} forecast is published ({pub:g} h after its "
                   f"cycle), so the boundaries are that cycle's own leads")
    return age, f"live start: the {source} row's operational age {age} h (the start cycle's leads are not out yet)"


def resolve_parent_run(parent_run) -> Path:
    """--parent-run as given -> the folder holding the parent's wrfout_d01_* history (the folder itself or its run/
    subfolder, parent_start.run_dir); refuses naming the folders it looked in."""
    if parent_run is None:
        raise SystemExit("--source parent needs --parent-run DIR (the parent run's folder: the nest door's --out "
                         "folder or its run/ folder)")
    from namgpu.init import parent_start
    try:
        return parent_start.run_dir(parent_run).resolve()
    except ValueError as e:
        raise SystemExit(f"--parent-run {parent_run}: {e}") from None


def parent_credit(parent_run, given: str | None = None) -> dict:
    """What a parent run started from, from its hand-off history's global attributes (io_history_fields.handoff_attrs:
    START_SOURCE, DATA_CREDIT, DATA_LICENCES). A parent from before those were written needs --parent-credit: a nest
    under an IFS-started parent otherwise rendered maps with no ECMWF credit (CC BY 4.0) (fix round 2)."""
    run = resolve_parent_run(parent_run)
    hist = sorted(run.glob("wrfout_d01_*"))
    import netCDF4
    with netCDF4.Dataset(hist[0]) as nc:
        at = {k: nc.getncattr(k) for k in nc.ncattrs()}
    run = run.resolve()
    if run.name == "run":           # the door writes history to <out>/run: name the door output folder
        run = run.parent
    if at.get("DATA_LICENCES"):
        return {"run": run.name, "source": str(at.get("START_SOURCE") or "unrecorded"),
                "credit": str(at.get("DATA_CREDIT") or ""), "lines": list(json.loads(str(at["DATA_LICENCES"])))}
    if given:
        return {"run": run.name, "source": f"given: {given}", "credit": given,
                "lines": [f"start and boundaries: {given} (given with --parent-credit; the parent history does not "
                          f"record it)"]}
    raise SystemExit(f"--source parent: {hist[0].name} records no start-source attribution (written before "
                     f"io_history_fields.handoff_attrs carried DATA_LICENCES), so a CC BY credit the parent's start "
                     f"needs could be missing from every map: pass --parent-credit 'TEXT' naming what the parent "
                     f"started from (e.g. 'NOAA/NCEP GFS (public domain)' or 'ECMWF open data, CC BY 4.0')")


def data_credits(source: str, static_ds=None, parent_run=None, given: str | None = None) -> tuple[str, list[str]]:
    """(short credit for every map and sheet, licence lines for INDEX.md and the receipt): the start/boundary
    source row's attribution (for a parent nest: the parent run's own start source, parent_credit) and the static
    datasets that require attribution (static_build.DATASETS)."""
    from namgpu.init import sources, static_build as sb
    row = sources.SOURCES.get(source) or {}
    short, lines = [], []
    if source == "parent":
        pc = parent_credit(parent_run, given)
        lines.append(f"start and boundaries: our own WOOF-NAM run {pc['run']} (history hand-off), itself started "
                     f"from {pc['source']}")
        lines += [f"parent run, {ln}" for ln in pc["lines"]]
        short += [c.strip() for c in pc["credit"].split(",") if c.strip()]
    elif row.get("attribution"):
        lines.append(f"start and boundaries: {row['attribution']}")
        if str(row.get("licence", "")).startswith("CC-BY"):
            short.append(row["attribution"].replace(", CC BY", " CC BY"))
        if (row.get("fetch") or {}).get("donors"):
            lines.append("sea ice: NOAA NCEP GFS f000 (public domain)")
    for role, d in (static_ds or sb.DATASETS).items():
        lines.append(f"static {role}: {d['dataset']}: {d['source_and_licence']}")
        if d.get("attribution"):
            short.append({"soil_top": "SoilGrids CC BY 4.0", "soiltemp": "ERA5 C3S"}.get(role, d["dataset"]))
    return ", ".join(dict.fromkeys(short)), lines


def _write(out: Path, rec: dict, stages: dict, sampler: Sampler, t_all: float):
    sampler.stop()
    summ = sampler.summary()
    for nm, s in stages.items():
        s.pop("t0", None)
        b = summ["by_stage"].get(nm)
        if b:
            s.update(cores_mean=b["cores_mean"], cores_max=b["cores_max"], vram_peak_gib=round(b["vram_peak_mib"] / 1024, 2))
    rec["stages"] = stages
    rec["single_core_windows_over_30s"] = summ["single_core_windows"]
    rec["vram_peak_gib"] = round(summ["vram_peak_mib"] / 1024, 2)
    rec["wall_s"] = round(time.perf_counter() - t_all, 1)
    rec["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "nest-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    (out / "nest-samples.json").write_text(json.dumps(sampler.samples))
    print(f"woof-nam: receipt {out / 'nest-receipt.json'}: wall {rec['wall_s']} s, peak VRAM {rec['vram_peak_gib']} "
          f"GiB, single-core windows {summ['single_core_windows'] or 'none'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
