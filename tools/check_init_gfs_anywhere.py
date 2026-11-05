"""The GFS start and boundaries for a 3 km nest anywhere: the real run of lane init-gfs-anywhere.

    python tools/check_init_gfs_anywhere.py --cycle 2026110412 --work ~/scratch/nest-init-gfs \
        --centre dateline:-41:174 --centre arctic:70:-45 --centre tropics:1:102 [--forecast dateline --fc-hours 1]

Per centre: a 3 km domain configfile (NAM's nest keys, size --size-km, the domain's own pole at the centre),
its static (init-static-build), the case fetch (fetch_case rows for that domain), the GFS start (0 h history,
tools/run_init_case.py), the boundaries for --hours (init-gfs-bocos, which refuses non-finite strips), a
NaN census of the 0 h history, and f00 maps of 2 m T, MSLP and 10 m wind beside the same GFS f000 fields
(Rust renderer, CF route). Optionally a short forecast on one centre (tools/run_forecast.py).

Every stage runs as its own process; a sampler reads /proc twice a second for that process tree and records
wall, CPU seconds, mean and peak cores, the longest stretch at about one core (0.6-1.4 cores; the "single core
on the critical path" signature) and peak device memory of the tree (nvidia-smi). receipt.json in --work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
CLK = os.sysconf("SC_CLK_TCK")


# ------------------------------------------------------------------------------------------------ sampler
def _tree(root: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def _ticks(pids: list[int], root: int) -> int:
    t = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as f:
                v = f.read().rsplit(")", 1)[1].split()
            t += int(v[11]) + int(v[12])                 # utime + stime
            if p == root:
                t += int(v[13]) + int(v[14])             # reaped children of the stage process
        except (OSError, IndexError, ValueError):
            pass
    return t


def _gpu_mib(pids: set[int]) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    tot = 0
    for line in out.splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) == 2 and a[0].isdigit() and int(a[0]) in pids:
            tot += int(a[1]) if a[1].isdigit() else 0
    return tot


def run_stage(name: str, cmd: list[str], log: Path, env: dict, receipt: dict) -> int:
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        p = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        series, peak_gpu, stop = [], 0, threading.Event()
        known: set[int] = set()

        def sample():
            nonlocal peak_gpu
            last_t, last_c, k = time.perf_counter(), 0, 0
            while not stop.is_set():
                time.sleep(0.5)
                pids = _tree(p.pid)
                known.update(pids)
                c = _ticks(pids, p.pid)
                now = time.perf_counter()
                series.append((now - t0, max(0, c - last_c) / CLK / (now - last_t)))
                last_t, last_c = now, max(c, last_c)
                k += 1
                if k % 4 == 0:
                    peak_gpu = max(peak_gpu, _gpu_mib(set(pids)))
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        rc = p.wait()
        stop.set()
        th.join()
    wall = time.perf_counter() - t0
    cores = np.array([c for _, c in series]) if series else np.zeros(1)
    one, run_s, longest = 0.0, 0.0, 0.0
    for k, c in enumerate(cores):
        dt = 0.5
        run_s = run_s + dt if 0.6 <= c <= 1.4 else 0.0
        longest = max(longest, run_s)
        one += dt if 0.6 <= c <= 1.4 else 0.0
    rec = {"rc": rc, "wall_s": round(wall, 1), "cpu_s": round(float(cores.sum() * 0.5), 1),
           "mean_cores": round(float(cores.mean()), 1), "peak_cores": round(float(cores.max()), 1),
           "longest_one_core_s": round(longest, 1), "peak_gpu_mib": peak_gpu, "log": str(log)}
    receipt.setdefault("stages", {})[name] = rec
    print(f"[{name}] {json.dumps(rec)}", flush=True)
    return rc


# ------------------------------------------------------------------------------------------------ domain
def write_config(work: Path, name: str, lat: float, lon: float, size_km: float, res_km: float) -> Path:
    """A res_km nest of size_km x size_km centred on (lat, lon), NAM's nest keys (make_domain_configs
    NEST_PHYSICS), the domain's own pole at the centre. Grid: dphd = 0.009 res, dlmd = 0.0105 res (NAM's 12 km
    0.108/0.126 scaled), odd point counts so the centre is a point. dt = 25 s x res/12 (6.25 s at 3 km, as
    NAM's nest), filt_dt = dt / 3.125 (NAM's nest 2 s), nrads = nradl = 20 min."""
    import make_domain_configs as mdc
    from fractions import Fraction
    dphd, dlmd = 0.009 * res_km, 0.0105 * res_km
    km = 6371.2 * np.pi / 180.0
    im = 2 * int(round(size_km / (dlmd * km) / 2.0)) + 1
    jm = 2 * int(round(size_km / (dphd * km) / 2.0)) + 1
    dt = Fraction(25) * Fraction(res_km).limit_denominator(100) / 12
    fdt = dt / Fraction(25, 8)
    nrad = int(round(1200 / float(dt)))
    repl = dict(mdc.NEST_PHYSICS, tph0d=f"{lat:.4f}", tlm0d=f"{lon:.4f}", im=str(im), jm=str(jm),
                wbd=f"{-(im - 1) / 2 * dlmd:.6f}", sbd=f"{-(jm - 1) / 2 * dphd:.6f}",
                dt_int=str(int(dt)), dt_num=str((dt - int(dt)).numerator), dt_den=str((dt - int(dt)).denominator),
                filt_dt_int=str(int(fdt)), filt_dt_num=str((fdt - int(fdt)).numerator),
                filt_dt_den=str((fdt - int(fdt)).denominator), nrads=str(nrad), nradl=str(nrad),
                static_npz=str(work / "static" / f"nest{res_km:g}km_{name}.npz"), domain_name=f"nest{res_km:g}km_{name}")
    out = work / "configs" / f"nam_configfile_nest_{name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mdc.render((ROOT / "parm" / "nam_configfile").read_text(), repl, f"nest {name}"))
    return out


# ------------------------------------------------------------------------------------------------ look
LOOK = ("t2m", "mslp", "wind10")


def _frame(lon):
    """Renderer bounds for these longitudes: -180..180 unless the domain crosses the dateline, then 0..360."""
    l1 = (lon + 180.0) % 360.0 - 180.0
    l2 = lon % 360.0
    lon_use = l2 if (np.ptp(l2) < np.ptp(l1)) else l1
    return lon_use


def look(work: Path, name: str, hist: Path, gfs: Path, cfg_path: Path, out: Path, label: str) -> dict:
    """f00 2 m T, MSLP, 10 m wind of the nest (0 h history) beside the same GFS f000 fields cut to the same frame."""
    import tempfile
    import cupy as cp
    from namgpu.io import maps
    from namgpu.init import init_grib2_decode as g2
    valid, start, f = maps._read(hist, {"XLAT", "XLONG", "T2", "MSLP", "U10", "V10"})
    lat, lon = f["XLAT"], _frame(f["XLONG"])
    flds, _ = maps.fields_for(None, (valid, start, f, hist), list(LOOK))
    pad = 1.0
    # rw_wrfbatch takes bounds inside -180..180 only: a frame past the dateline is clipped at 180 here (the slice
    # east of it is not drawn; the dateline frame is lane plots-referee's)
    e = lon.max() + pad
    clipped = e > 180.0
    bounds = f"{lon.min() - pad:.2f},{min(e, 180.0):.2f},{lat.min() - pad:.2f},{lat.max() + pad:.2f}"
    if np.ptp(lon) > 300.0:                 # a domain around a pole: every longitude, the cap up to the pole
        bounds = f"-180,180,{max(lat.min() - pad, -90):.2f},{min(lat.max() + pad, 90):.2f}"
    # GFS f000 2 m T, PRMSL, 10 m wind on the same frame (GFS rows cut to the frame, same longitude convention)
    sel = {"T2": ((0, 0, 0), "103", 2.0), "PRMSL": ((0, 3, 1), "101", 0.0),
           "U10": ((0, 2, 2), "103", 10.0), "V10": ((0, 2, 3), "103", 10.0)}

    def s_of(spec):
        dcp, lt, lv = spec
        return lambda r: r["dcp"] == dcp and str(r["level_type"]) == lt and abs(float(r["level_value"]) - lv) < 0.01
    inv = g2.inventory(str(gfs))
    nx, ny = int(inv[0]["nx"]), int(inv[0]["ny"])
    from types import SimpleNamespace
    got = g2.run(SimpleNamespace(stream=None, tables={}), str(gfs), {k: s_of(v) for k, v in sel.items()},
                 missing_value=-9999.0, tmpdir=str(work / "scratch"), expect_shape=(nx, ny))
    glat = -90.0 + np.arange(ny) * (180.0 / (ny - 1))
    glon = np.arange(nx) * (360.0 / nx)
    glon = glon if lon.max() > 180.0 else (glon + 180.0) % 360.0 - 180.0
    jj = np.flatnonzero((glat >= lat.min() - pad) & (glat <= lat.max() + pad))
    ii = np.flatnonzero((glon >= lon.min() - pad) & (glon <= lon.max() + pad) & (glon < 180.0))   # 180.0 itself wraps to -180 in the renderer
    ii = ii[np.argsort(glon[ii])]
    G = {k: cp.asnumpy(v)[np.ix_(jj, ii)].astype(np.float64) for k, v in got.items()}
    LA, LO = np.meshgrid(glat[jj], glon[ii], indexing="ij")
    gf = {"t2m": (maps._k2f(G["T2"]), "degF"), "mslp": (G["PRMSL"] / 100.0, "hPa"),
          "ws10m": (np.hypot(G["U10"], G["V10"]), "m s-1")}
    w = Path(tempfile.mkdtemp(prefix=".look-", dir=work))
    table = w / "cf-table.json"
    table.write_text(json.dumps([maps.TABLE[p][1] for p in LOOK]))
    res = {}
    for tag, la, lo, fl in (("WOOF-NAM 3 km", lat, lon, flds), ("GFS 0.25 f000", LA, LO, gf)):
        cf = w / f"{tag.split()[0]}.nc"
        maps._write_cf(cf, la, lo, valid, fl, f"{tag} {name}, init {start:%Y-%m-%d %HZ} valid {valid:%Y-%m-%d %HZ} (+0h)")
        for prod in LOOK:
            var = maps.TABLE[prod][0]
            od = w / f"r_{tag.split()[0]}_{var}"
            cmd = [maps._bridges() / "rw_wrfbatch", "--cf", cf, "--variable", var, "--out-dir", od, "--valid",
                   valid.strftime("%Y-%m-%dT%H:%M:%SZ"), "--bounds", bounds, "--cf-table", table,
                   "--width", "1000", "--height", "900"]
            p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            pngs = sorted(od.rglob("*.png")) if od.is_dir() else []
            key = f"{tag.split()[0]}_{prod}"
            if p.returncode or not pngs:
                res[key] = f"FAILED exit {p.returncode}: {(p.stderr or p.stdout).strip()[-300:]}"
                continue
            dst = out / f"{name}_f00_{prod}_{tag.split()[0].lower()}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pngs[0].replace(dst)
            res[key] = str(dst)
    import shutil
    shutil.rmtree(w, ignore_errors=True)
    stats = {k: [round(float(np.nanmin(v[0])), 1), round(float(np.nanmean(v[0])), 1), round(float(np.nanmax(v[0])), 1)]
             for k, v in flds.items()}
    gstats = {k: [round(float(np.nanmin(v[0])), 1), round(float(np.nanmean(v[0])), 1), round(float(np.nanmax(v[0])), 1)]
              for k, v in gf.items()}
    return {"pngs": res, "bounds": bounds, "clipped_at_180": bool(clipped), "nest_min_mean_max": stats, "gfs_min_mean_max": gstats}


def nan_census(path: Path) -> dict:
    import netCDF4
    bad = {}
    with netCDF4.Dataset(path) as nc:
        for n, v in nc.variables.items():
            if v.dtype.kind == "f":
                a = np.asarray(v[:], dtype=np.float64)
                k = int((~np.isfinite(a)).sum())
                if k:
                    bad[n] = k
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--centre", action="append", required=True, help="name:lat:lon")
    ap.add_argument("--size-km", type=float, default=1000.0)
    ap.add_argument("--res-km", type=float, default=3.0)
    ap.add_argument("--forecast", default=None, help="centre name for the short forecast")
    ap.add_argument("--fc-hours", type=float, default=1.0)
    ap.add_argument("--init-steps", type=int, default=0, help="model steps after the init (all physics, census each)")
    ap.add_argument("--stages", default="config,static,fetch,init,bocos,look,forecast")
    ap.add_argument("--deliver", type=Path, default=None, help="folder for the look PNGs")
    a = ap.parse_args(argv)
    stages = set(a.stages.split(","))
    work = a.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch").mkdir(exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H")
    case = cycle.strftime("%Y-%m-%d_%Hz")
    case_dir = work / "cases" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    link = ROOT / "cases" / case                 # the tools read ROOT/cases/<case>
    if not link.exists():
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(case_dir)
    rpath = work / "receipt.json"
    receipt = json.loads(rpath.read_text()) if rpath.is_file() else {}
    receipt.update({"cycle": a.cycle, "hours": a.hours, "case_dir": str(case_dir), "tree": str(ROOT)})
    py = sys.executable
    for spec in a.centre:
        name, lat, lon = spec.split(":")
        lat, lon = float(lat), float(lon)
        cfg_path = write_config(work, name, lat, lon, a.size_km, a.res_km)
        env = dict(os.environ, NAMGPU_CONFIGFILE=str(cfg_path), PYTHONPATH=str(ROOT), TMPDIR=str(work / "scratch"))
        from namgpu import config
        cfg = config.load(cfg_path)
        r = receipt.setdefault("centres", {}).setdefault(name, {})
        r.update({"centre": [lat, lon], "grid": [cfg.im, cfg.jm, cfg.lm], "dlmd": cfg.dlmd, "dphd": cfg.dphd,
                  "dt": cfg.dt, "configfile": str(cfg_path)})
        print(f"== {name} {lat},{lon}: {cfg.im} x {cfg.jm} x {cfg.lm}, dt {cfg.dt} s", flush=True)
        L = lambda s: work / "logs" / f"{name}-{s}.log"  # noqa: E731
        sub = {}
        if "static" in stages and not Path(cfg.raw["static_npz"]).is_file():
            run_stage("static", [py, "-m", "namgpu.init.static_build", "--config", cfg_path, "--date",
                                 cycle.strftime("%Y-%m-%d"), "--out", cfg.raw["static_npz"]], L("static"), env, sub)
        if "fetch" in stages:
            run_stage("fetch", [py, "-m", "namgpu.init.fetch_case", "--cycle", a.cycle, "--hours", a.hours,
                                "--config", cfg_path, "--case-dir", case_dir], L("fetch"), env, sub)
        init_out = work / "init" / name
        if "init" in stages:
            run_stage("init", [py, "tools/run_init_case.py", "--case", case, "--source", "gfs", "--out", init_out,
                               "--no-maps", "--steps", a.init_steps], L("init"), env, sub)
            hist = sorted(init_out.glob("wrfout_d01_*"))
            if hist:
                r["history_f00"] = str(hist[0])
                r["nan_census_f00"] = nan_census(hist[0])
                print(f"[{name}] f00 non-finite: {r['nan_census_f00'] or 'none'}", flush=True)
        if "bocos" in stages:
            run_stage("bocos", [py, "-m", "namgpu.init.gfs_bocos", "--case", case, "--hours", a.hours],
                      L("bocos"), env, sub)
        if "look" in stages and r.get("history_f00"):
            try:
                r["look"] = look(work, name, Path(r["history_f00"]), case_dir / "raw" / "gfs" /
                                 f"gfs.t{cycle:%H}z.pgrb2.0p25.f000", cfg_path,
                                 a.deliver or work / "look", "WOOF-NAM")
            except Exception as e:  # noqa: BLE001
                r["look"] = f"FAILED {type(e).__name__}: {e}"
            print(f"[{name}] look {json.dumps(r['look'])[:600]}", flush=True)
        if "forecast" in stages and a.forecast == name:
            fc = work / "fc" / name
            run_stage("forecast", [py, "tools/run_forecast.py", "--case", case, "--source", "gfs", "--hours",
                                   a.fc_hours, "--out", fc, "--no-maps"], L("forecast"), env, sub)
            rec = fc / "forecast-receipt.json"
            if rec.is_file():
                r["forecast_receipt"] = json.loads(rec.read_text())
        r.setdefault("stages", {}).update(sub.get("stages", {}))
        rpath.write_text(json.dumps(receipt, indent=1, default=str))
    rpath.write_text(json.dumps(receipt, indent=1, default=str))
    print("receipt", rpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
