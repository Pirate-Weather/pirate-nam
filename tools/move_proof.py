"""The L2 proof runs of the moving-nest window shift and hand-off (namgpu/nest/move.py), one command each.

    python tools/move_proof.py base      the unbroken 12 h nest (restart sets at the landing hours 3 and 9)
    python tools/move_proof.py optionoff the same nest from the base tree (no window code): hourly history cmp
    python tools/move_proof.py null      a null move at 3 h (same window, fresh start at 15Z + the 3 h set): cmp
    python tools/move_proof.py onecell   a one-cell move (di = dj = 1) at 3 h: carried / geometry / static checks
    python tools/move_proof.py ampil     the 12 h storm-following nest: moves at 3 h and 9 h onto the storm

Work folder $W (default move-proof): nam_configfile_base (the reference window, 31.5N 140.8E, 1000 km, 3 km),
cases/2024-08-15_12z (GFS 2024-08-15 12Z, boundaries from the same cycle), starts/<YYYY-MM-DD_HHz>/raw/gfs (the
12Z cycle's f003/f006/f009 with the start record set, used as the fresh start of a window at that hour). This
is test orchestration; the door's controller (namgpu/nest/follow.py, the tracker) and preparer (the preparer) replace the
simple fixer and the inline preparation here.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

W = Path(os.environ.get("W", "move-proof"))
ROOT = Path(__file__).resolve().parents[1]
CASE = "2024-08-15_12z"
CYCLE = datetime(2024, 8, 15, 12)
CASE_DIR = W / "cases" / CASE
BASE = W / "nam_configfile_base"
HOURS = 12
TDFI = os.environ.get("L2_TDFI", "1") == "1"      # the fresh start's own TDFI (shock numbers)
sys.path.insert(0, str(ROOT))


def log(msg):
    print(f"{datetime.utcnow():%H:%M:%S} move_proof: {msg}", flush=True)


def sh(cmd, logf: Path, env=None) -> int:
    logf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(logf, "a") as f:
        f.write(f"\n# {' '.join(map(str, cmd))}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            env=dict(os.environ, **(env or {}))).returncode
    log(f"rc {rc} in {time.perf_counter() - t0:.1f} s: {' '.join(map(str, cmd[:4]))} ... (log {logf})")
    return rc


def py(*args):
    return [sys.executable, "-m", *args]


# ------------------------------------------------------------------------------------------ preparation

def window(i0: int, j0: int, name: str) -> dict:
    """The window's configfile, static (GPU, ~10 s) and boundaries for the whole run (named from the run start)."""
    from namgpu.nest import move
    d = W / "windows" / name
    d.mkdir(parents=True, exist_ok=True)
    static = W / "static" / f"{name}.npz"
    cfg = d / "nam_configfile"
    t0 = time.perf_counter()
    info = move.window_configfile(BASE, cfg, i0, j0, name, static)
    walls = {}
    if not static.is_file():
        t = time.perf_counter()
        if sh(py("namgpu.init.static_build", "--config", cfg, "--date", f"{CYCLE:%Y-%m-%d}", "--out", static),
              d / "static.log", {"NAMGPU_CONFIGFILE": str(cfg)}):
            raise SystemExit(f"static build failed for {name}")
        walls["static_s"] = round(time.perf_counter() - t, 1)
    bdir = CASE_DIR / f"boco-{name}"
    if not (bdir / f"boco.{HOURS - 3:04d}").is_file():
        t = time.perf_counter()
        if sh(py("namgpu.init.gfs_bocos", "--case", CASE_DIR, "--hours", HOURS, "--boco-age", 0),
              d / "bocos.log", {"NAMGPU_CONFIGFILE": str(cfg)}):
            raise SystemExit(f"bocos failed for {name}")
        walls["bocos_s"] = round(time.perf_counter() - t, 1)
    info.update(walls, prepare_s=round(time.perf_counter() - t0, 1), boco_dir=str(bdir), static=str(static))
    (d / "window.json").write_text(json.dumps(info, indent=1))
    log(f"window {name}: i0 {i0} j0 {j0} centre {info['centre_geo']} {walls}")
    return info


def fresh(name: str, cfg: Path, hour: int) -> Path:
    """The window's fresh start at the landing hour (the source's state then), saved for --init-from."""
    out = W / "fresh" / f"{name}-h{hour:02d}{'-tdfi' if TDFI else ''}"
    if (out / "skeleton.pkl").is_file():
        return out
    from namgpu import config
    bdir = CASE_DIR / config.boco_dirname(config.load(cfg))
    t = CYCLE + timedelta(hours=hour)
    sdir = CASE_DIR if hour == 0 else W / "starts" / f"{t:%Y-%m-%d_%H}z"
    if sh(py("namgpu.init.slab_init", "save", "--case", f"{t:%Y-%m-%d_%H}z", "--case-dir", sdir, "--source", "gfs",
             "--hours", HOURS, "--out", out, "--boco-dir", bdir, "--from-hour", hour)
          + (["--tdfi"] if TDFI else []),
          Path(str(out) + ".log"), {"NAMGPU_CONFIGFILE": str(cfg)}):
        raise SystemExit(f"fresh start failed for {name} at {hour} h")
    return out


def segment(cfg: Path, out: Path, hours: int, restart_at=(), init_from=None, move_from=None, check=False,
            tree: Path | None = None) -> dict:
    cmd = py("namgpu.run_forecast", "--case", CASE, "--case-dir", CASE_DIR, "--source", "gfs", "--hours", hours,
             "--out", out, "--no-maps")
    if restart_at:
        cmd += ["--restart-at", ",".join(str(h) for h in restart_at), "--restart-keep", "4",
                "--restart-dir", out / "restart"]
    if init_from:
        cmd += ["--init-from", init_from]
    if move_from:
        cmd += ["--move-from", move_from]
    if check:
        cmd += ["--move-check"]
    env = {"NAMGPU_CONFIGFILE": str(cfg)}
    if tree is not None:
        env.update(NAMGPU_ROOT=str(tree), PYTHONPATH=str(tree))
    t0 = time.perf_counter()
    rc = subprocess.run([str(c) for c in cmd], cwd=str(tree or ROOT), env=dict(os.environ, **env),
                        stdout=open(Path(str(out) + ".log"), "w"), stderr=subprocess.STDOUT).returncode
    log(f"segment {out.name}: rc {rc}, {time.perf_counter() - t0:.0f} s")
    rp = out / "forecast-receipt.json"
    rec = json.loads(rp.read_text()) if rp.is_file() else {}
    if rc:
        raise SystemExit(f"segment {out} failed rc {rc}: {rec.get('error')} (log {out}.log)")
    return rec


def set_at(run: Path, hour: int) -> Path:
    t = CYCLE + timedelta(hours=hour)
    p = run / "restart" / f"wrfrst_d01_{t:%Y-%m-%d_%H:%M:%S}"
    if not (p / "COMPLETE").is_file():
        raise SystemExit(f"no restart set {p}")
    return p


def cmp_hours(a: Path, b: Path, hours) -> dict:
    out = {}
    for h in hours:
        t = CYCLE + timedelta(hours=h)
        fa, fb = a / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}", b / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}"
        out[h] = ("identical" if filecmp.cmp(fa, fb, shallow=False) else "DIFFER") if fa.is_file() and fb.is_file() \
            else f"missing {fa.is_file()} {fb.is_file()}"
    return out


def field_diff(a: Path, b: Path, hour: int) -> dict:
    """Per-variable max |a - b| of one hour's history (what differs when cmp fails)."""
    import netCDF4
    t = CYCLE + timedelta(hours=hour)
    name = f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}"
    da, db = netCDF4.Dataset(a / name), netCDF4.Dataset(b / name)
    out = {}
    for v in da.variables:
        if v not in db.variables or da[v].dtype.kind not in "fi":
            continue
        x, y = np.asarray(da[v][:], np.float64), np.asarray(db[v][:], np.float64)
        if x.shape == y.shape and not np.array_equal(x, y, equal_nan=True):
            out[v] = float(np.nanmax(np.abs(x - y)))
    return out


# ------------------------------------------------------------------------------------------ storm fix

def storm_fix(run: Path, hour: int) -> dict:
    """The storm centre in one hourly frame: minimum of the 25-cell-smoothed MSLP away from the boundary
    (the hex TCFixer's first guess; the door uses namgpu/nest/track.py)."""
    import netCDF4
    from scipy.ndimage import uniform_filter
    t = CYCLE + timedelta(hours=hour)
    d = netCDF4.Dataset(run / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}")
    p = np.asarray(d["MSLP"][0], np.float64)
    lat, lon = np.asarray(d["XLAT"][0]), np.asarray(d["XLONG"][0])
    s = uniform_filter(p, 25, mode="nearest")
    b = 30
    s[:b] = s[-b:] = np.inf
    s[:, :b] = s[:, -b:] = np.inf
    j, i = np.unravel_index(np.argmin(s), s.shape)
    return {"hour": hour, "lat": float(lat[j, i]), "lon": float(lon[j, i]), "j": int(j), "i": int(i),
            "min_mslp_hpa": round(float(p.min()) / 100.0, 2), "smoothed_min_hpa": round(float(s[j, i]) / 100.0, 2)}


def predict(fixes: list, hour: float) -> tuple:
    """Least-squares straight-line motion through the fixes (lat/lon in degrees), at `hour`."""
    h = np.array([f["hour"] for f in fixes], float)
    la = np.array([f["lat"] for f in fixes])
    lo = np.array([f["lon"] for f in fixes])
    if len(h) == 1:
        return float(la[0]), float(lo[0])
    a_la, a_lo = np.polyfit(h, la, 1), np.polyfit(h, lo, 1)
    return float(np.polyval(a_la, hour)), float(np.polyval(a_lo, hour))


# ------------------------------------------------------------------------------------------ commands

def cmd_base(a):
    out = W / "runs" / "base12"
    rec = segment(BASE, out, HOURS, restart_at=(3, 9))
    log(f"base: {rec.get('steps_done')} steps, {rec.get('ms_per_step')} ms/step, nonfinite {rec.get('nonfinite_end')}")


def cmd_optionoff(a):
    tree = W / "base-tree"
    out = W / "runs" / "optionoff3"
    segment(BASE, out, 3, tree=tree)
    res = cmp_hours(out, W / "runs" / "base12", range(0, 4))
    log(f"option-off (base tree e738a9a vs this branch, no window keys), hourly history cmp: {res}")
    (W / "runs" / "optionoff3.json").write_text(json.dumps(res, indent=1))


def cmd_null(a):
    fr = fresh("base", BASE, 3)
    out = W / "runs" / "null3"
    rec = segment(BASE, out, 6, init_from=fr, move_from=set_at(W / "runs" / "base12", 3), check=True)
    res = cmp_hours(out, W / "runs" / "base12", range(4, 7))
    diff = {h: field_diff(out, W / "runs" / "base12", h) for h, v in res.items() if v != "identical"}
    summary = {"cmp": res, "field_max_abs_diff": diff, "move": rec.get("move")}
    (W / "runs" / "null3.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"null move at 3 h: hourly history cmp vs the unbroken run: {res}")
    if diff:
        log(f"null move differences: { {h: dict(list(d.items())[:12]) for h, d in diff.items()} }")


def cmd_onecell(a):
    info = window(1, 1, "w+1+1")
    cfg = Path(info["configfile"])
    fr = fresh("w+1+1", cfg, 3)
    out = W / "runs" / "onecell3"
    rec = segment(cfg, out, 4, init_from=fr, move_from=set_at(W / "runs" / "base12", 3), check=True)
    mv = rec.get("move", {})
    ov = mv.get("overlay", {})
    summary = {"window": info, "check": mv.get("check"),
               "geometry": ov.get("geometry"), "static_in_state": ov.get("static"),
               "rows": ov.get("rows"), "window_constants": ov.get("window_constants"),
               "globals_from_set": ov.get("globals_from_set"), "monitors": mv.get("monitors"),
               "nonfinite_end": rec.get("nonfinite_end")}
    (W / "runs" / "onecell3.json").write_text(json.dumps(summary, indent=1, default=str))
    chk = mv.get("check") or {}
    log(f"one-cell move: {chk.get('arrays')} carried arrays, {chk.get('mismatched_arrays')} with mismatches; "
        f"geometry mismatches {[g['overlap_mismatch'] for g in ov.get('geometry', [])]}; "
        f"static {chk.get('static')}")


def cmd_ampil(a):
    """12 h storm-following nest from the storm's 12Z position: land at 3 h and 9 h, each window centred on the
    storm's predicted position 3 h after its landing (the middle of the next dwell)."""
    runs = W / "runs" / "ampil"
    runs.mkdir(parents=True, exist_ok=True)
    base = W / "runs" / "base12"
    plan = []
    # segment A = the unbroken run's first 3 h (the same bits: the restart set at 3 h is in base12)
    fixes = [storm_fix(base, h) for h in (1, 2, 3)]
    from namgpu import config
    ref = config.load(BASE)
    prev_set, prev_win = set_at(base, 3), (0, 0)
    landings = [(3, 9), (9, HOURS)]
    seg_runs = {0: base}
    for k, (land, nxt) in enumerate(landings, start=1):
        tgt = predict(fixes, land + 3)
        i0, j0 = __import__("namgpu.nest.move", fromlist=["snap"]).snap(ref, *tgt)
        name = f"ampil-g{k}"
        info = window(i0, j0, name)
        cfg = Path(info["configfile"])
        t0 = time.perf_counter()
        fr = fresh(name, cfg, land)
        prep_s = round(time.perf_counter() - t0, 1)
        out = runs / f"seg{k}"
        restart = (nxt,) if nxt < HOURS else ()
        rec = segment(cfg, out, nxt, restart_at=restart, init_from=fr, move_from=prev_set, check=True)
        mv = rec.get("move", {})
        shift = mv.get("shift_dj_di")
        plan.append({"landing_h": land, "fixes": fixes[-3:], "target": tgt, "window": info, "shift_dj_di": shift,
                     "fresh_s": prep_s, "monitors": mv.get("monitors"), "check": mv.get("check"),
                     "ms_per_step": rec.get("ms_per_step"), "nonfinite_end": rec.get("nonfinite_end"),
                     "stats_end": rec.get("stats_end")})
        log(f"move {k} at {land} h: storm fixes {[(f['hour'], round(f['lat'], 2), round(f['lon'], 2)) for f in fixes[-3:]]}"
            f" -> window centre {info['centre_geo']}, shift {shift} cells; segment to {nxt} h nonfinite "
            f"{rec.get('nonfinite_end')}")
        seg_runs[k] = out
        fixes = fixes + [storm_fix(out, h) for h in range(land + 1, nxt + 1)]
        if nxt < HOURS:
            prev_set = set_at(out, nxt)
    (runs / "plan.json").write_text(json.dumps({"segments": {k: str(v) for k, v in seg_runs.items()},
                                                "moves": plan, "fixes": fixes}, indent=1, default=str))
    log(f"ampil: done; fixes {[(f['hour'], round(f['lat'], 2), round(f['lon'], 2), f['min_mslp_hpa']) for f in fixes]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "optionoff", "null", "onecell", "ampil", "window"])
    ap.add_argument("--i0", type=int, default=0)
    ap.add_argument("--j0", type=int, default=0)
    ap.add_argument("--name", default="w")
    a = ap.parse_args(argv)
    if a.cmd == "window":
        return print(window(a.i0, a.j0, a.name))
    return {"base": cmd_base, "optionoff": cmd_optionoff, "null": cmd_null, "onecell": cmd_onecell,
            "ampil": cmd_ampil}[a.cmd](a)


if __name__ == "__main__":
    main()
