"""The storm-following preparer: everything the next window needs, built while the card integrates.

    python -m namgpu.nest.prepare --out <run> --gen K --i0 I --j0 J --land-hour H [--wait-min M]
    python -m namgpu.nest.prepare --window <prep>/window.json --landing-s L --out <prep>     (the controller's form,
        namgpu.nest.follow ProcessLauncher: the preparation folder <run>/gen-<key>-at<L>/, READY.json when done)

Reads the run record <run>/follow/ref-grid.json (generation-0 grid, start, hours, source, case folder;
window.write_ref) and writes one preparation folder (window.gen_dir(run, K), or --out with --window):
  static     init-static-build of the window on the GPU (cached by grid key like the door's statics), then stamped
             from the run's static mosaic (<run>/follow/static-mosaic.npz, under a file lock): every cell some
             earlier window of this run already had keeps that window's static, and the cells new to the run are
             added to the mosaic. So any two windows agree bitwise on their overlap whatever order they were
             prepared in (candidates are prepared ahead), and every carried column keeps the terrain, land use,
             soil and albedo it was integrated with. The cells where the fresh build differed from the mosaic are
             counted per field (the alignment record: the tile fills sample each window's own coordinates, so a few
             cells at ties and along the old window's smoothing halo can differ).
  bocos      the boundary files from the landing hour H to the end of the run, named in hours from the run start
             (gfs_bocos.build / parent_bocos.build with from_hour), in <prep>/boco with BOCO-SOURCE.json / SOURCE.json
             stamped with the window's grid and the generation. <prep>/boco-view holds the same files named from H
             (symlinks): the fresh start's init reads tboco from boco.0000 there.
  fresh      the whole window's initial state at the landing time T = start + H, from the source's full state at T:
             gfs: the boundary cycle's lead H + age fetched with the start record set (fetch_case follow mode);
             ifs / ifs-0p25: the lead itself (it carries the start selection), sea-ice and snow donors from the
             cycle's GFS f000; parent: the parent history at T. It runs the ordinary init chain (no fetch, no bocos) in
             a case folder <prep>/case/<T> whose raw/ names are links to those files, and is saved with
             slab_init.save to --init-dir (default /dev/shm/<run>-<prep>-init). The next segment loads it and lays the
             restart set over it (the move step, namgpu/nest/move.py).
  GEN.json   the window (offset, landing hour; generation -1 until window.commit_generation at the hand-off); the
             window configfile names it as follow_meta, so the generation's history carries FOLLOW_* attributes.
  PREPARED.json (= READY.json)  walls per stage (static build and stamp, bocos, init, save), cores per stage and
             single-core windows (the door's sampler), memory waited for, the alignment counts and every path.
Memory: the preparer runs beside the forecast on the same card, or on another card (CUDA_VISIBLE_DEVICES, launch(card=)).
It starts only when the card has room for the window's estimate (domain.vram_estimate + margin, nvidia-smi, no CUDA
context), waiting up to --wait-min, and caps its CuPy pool there: it never takes memory from another process.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from namgpu.nest import domain, window

UNIT = "follow-prepare"
SOURCES_OK = ("gfs", "ifs", "ifs-0p25", "parent")
# Where the source's full state at a boundary lead is (relative to the case folder): the start record set of the
# lead. gfs: fetch_case FETCH_ROWS gfs-boco follow_dest; ifs rows: the boundary file itself.
LEAD_START = {"gfs": "raw/gfs/gfs.t{bcyc:02d}z.f{lead:03d}.start.grib2",
              "ifs": "raw/ifs/ifs.t{bcyc:02d}z.0p1.f{lead:03d}.grib2",
              "ifs-0p25": "raw/ifs/ifs.t{bcyc:02d}z.0p25.f{lead:03d}.grib2"}
MOSAIC = "static-mosaic.npz"


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def find_ref(run_or_prep: Path) -> tuple[Path, dict]:
    """(run folder, ref-grid.json) for a run folder or a preparation folder inside it."""
    for d in (Path(run_or_prep), Path(run_or_prep).parent, Path(run_or_prep).parent.parent):
        p = d / window.FOLLOW_DIR / "ref-grid.json"
        if p.is_file():
            return d, _load(p)
    raise SystemExit(f"{UNIT}: no {window.FOLLOW_DIR}/ref-grid.json at or above {run_or_prep} (the moving run's door "
                     f"writes it with window.write_ref before the first preparation)")


# ------------------------------------------------------------------------------------------ static mosaic

def stamp_static(run: Path, ref: dict, fresh: dict, i0: int, j0: int) -> tuple[dict, dict]:
    """The window's static with every cell the run already has taken from the mosaic, and its new cells added.
    fresh: {field: (jm, im) float32} of the fresh build. Returns (merged fields, alignment record)."""
    import numpy as np
    jm, im = int(ref["grid"]["jm"]), int(ref["grid"]["im"])
    keys = [k for k, a in fresh.items() if k != "META" and getattr(a, "shape", None) == (jm, im)]
    d = Path(run) / window.FOLLOW_DIR
    lock = open(d / (MOSAIC + ".lock"), "a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        mp = d / MOSAIC
        if mp.is_file():
            z = np.load(mp)
            m = {k: z[k] for k in z.files}
        else:
            z0 = np.load(ref["static"])               # generation 0 seeds the mosaic
            m = {k: z0[k].astype(np.float32) for k in keys}
            m["DEFINED"] = np.ones((jm, im), np.uint8)
            m["ORIGIN"] = np.array([0, 0], np.int64)
        oi, oj = (int(v) for v in m["ORIGIN"])
        mj, mi = m["DEFINED"].shape
        # grow the mosaic to hold the window (index space of the generation-0 grid)
        ni0, nj0 = min(oi, i0), min(oj, j0)
        ni1, nj1 = max(oi + mi, i0 + im), max(oj + mj, j0 + jm)
        if (ni0, nj0, ni1, nj1) != (oi, oj, oi + mi, oj + mj):
            for k in keys + ["DEFINED"]:
                big = np.zeros((nj1 - nj0, ni1 - ni0), m[k].dtype)
                big[oj - nj0:oj - nj0 + mj, oi - ni0:oi - ni0 + mi] = m[k]
                m[k] = big
            m["ORIGIN"] = np.array([ni0, nj0], np.int64)
            oi, oj = ni0, nj0
        sl = (slice(j0 - oj, j0 - oj + jm), slice(i0 - oi, i0 - oi + im))
        have = m["DEFINED"][sl].astype(bool)
        merged, diff = dict(fresh), {}
        for k in keys:
            a = np.asarray(fresh[k], np.float32).copy()
            old = m[k][sl]
            diff[k] = int(np.count_nonzero(have & (a != old)))
            a[have] = old[have]
            merged[k] = a
            m[k][sl] = np.where(have, old, a)
        m["DEFINED"][sl] = 1
        tmp = mp.with_name(MOSAIC + ".part.npz")
        np.savez(tmp, **m)
        tmp.replace(mp)
        rec = {"overlap_mismatch_before_stamp": diff, "cells_from_mosaic": int(have.sum()),
               "cells_new_to_run": int((~have).sum()), "mosaic_shape": list(m["DEFINED"].shape),
               "mosaic_origin": [oi, oj]}
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return merged, rec


def write_static(out: Path, merged: dict, fresh_path: Path, rec: dict) -> None:
    import numpy as np
    meta = json.loads(str(merged["META"])) if "META" in merged else {}
    meta["follow"] = dict(rec, fresh_build=str(fresh_path))
    res = {k: v for k, v in merged.items() if k != "META"}
    res["META"] = np.array(json.dumps(meta))
    tmp = out.with_name(out.name + ".part.npz")
    np.savez(tmp, **res)
    tmp.replace(out)
    prov = fresh_path.with_name(fresh_path.name + ".PROVENANCE.json")
    if prov.is_file():
        shutil.copy2(prov, out.with_name(out.name + ".PROVENANCE.json"))


# ------------------------------------------------------------------------------------------ fresh start case

def virtual_case(ref: dict, pdir: Path, t_land: datetime, lead_h: int) -> tuple[Path, list]:
    """<prep>/case/<T>/ whose raw/ names are links to the source's full state at T: the start file the init reads
    at T (sources.SOURCES[src]["file"] at hour T) -> the lead's start record set, plus the row's donor files."""
    from namgpu.init import sources
    src = ref["source"]
    row = sources.SOURCES[src]
    real = Path(ref["case_dir"])
    start = datetime.strptime(ref["start"], "%Y%m%d%H")
    bcyc = start - timedelta(hours=int(ref["boco_age_h"]))
    case = pdir / "case" / f"{t_land:%Y-%m-%d_%H}z"
    links = []

    def link(name: str, target: Path):
        dst = case / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if not target.is_file():
            raise FileNotFoundError(f"{UNIT}: {target} missing: a move landing at {t_land:%Y-%m-%d %H}Z needs the "
                                    f"{src} state there (woof-nam nest --follow fetches every landable lead with the "
                                    f"start record set; fetch_case --follow)")
        dst.symlink_to(target.resolve())
        links.append([name, str(target)])

    link(row["file"].format(cyc=t_land.hour), real / LEAD_START[src].format(bcyc=bcyc.hour, lead=lead_h))
    for d in (row.get("fetch") or {}).get("donors", []):
        link(d["file"].format(hh=f"{t_land:%H}"), real / d["file"].format(hh=f"{bcyc:%H}"))
    return case, links


def cotenants(index: str | None) -> list:
    """Other compute processes on the card (nvidia-smi, no CUDA context): pid, MiB, name. Taken at the start and
    end of a preparation, it shows whether it ran beside the integrating forecast segment."""
    try:
        smi = ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"]
        if index:
            smi += ["-i", index]
        txt = subprocess.run(smi, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    rows = []
    for ln in txt.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) >= 3 and f[0].isdigit() and int(f[0]) != me:
            rows.append({"pid": int(f[0]), "mib": float(f[1]) if f[1].replace(".", "").isdigit() else None,
                         "name": f[2][-60:]})
    return rows


def wait_room(g: dict, rows: int, wait_min: float, index: str | None) -> dict:
    t_lim = time.perf_counter() + wait_min * 60
    t0 = time.perf_counter()
    while True:
        pf = domain.preflight(g, 1, rows, index=index)
        if pf["ok"] or time.perf_counter() > t_lim:
            pf["waited_s"] = round(time.perf_counter() - t0, 1)
            return pf
        time.sleep(15)


# ------------------------------------------------------------------------------------------ main

def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m namgpu.nest.prepare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="the moving run's folder (with --gen), or the preparation folder (with --window)")
    ap.add_argument("--gen", type=int, default=None, help="generation (1 = the first move): folder <out>/follow/gKK")
    ap.add_argument("--i0", type=int, default=None, help="window offset in cells from generation 0 (west-east)")
    ap.add_argument("--j0", type=int, default=None, help="window offset in cells from generation 0 (south-north)")
    ap.add_argument("--land-hour", type=float, default=None, help="forecast hour the move lands at")
    ap.add_argument("--window", type=Path, default=None, help="the controller's window.json (i0, j0, grid keys)")
    ap.add_argument("--landing-s", type=float, default=None, help="landing time, seconds from the run start")
    ap.add_argument("--start", default=None, help="check: the run start YYYYMMDDHH")
    ap.add_argument("--hours", type=float, default=None, help="check: the run length")
    ap.add_argument("--source", default=None, help="check: the run's source")
    ap.add_argument("--decided-hour", type=float, default=None)
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--init-dir", type=Path, default=None, help="where the fresh start is saved (default /dev/shm)")
    ap.add_argument("--wait-min", type=float, default=30.0, help="wait this long for card memory, then refuse")
    ap.add_argument("--rrtm-rows", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    if a.window is not None:
        w = _load(a.window)
        i0, j0 = int(w["i0"]), int(w["j0"])
        land_s = a.landing_s if a.landing_s is not None else float(w["landing_s"])
        H = land_s / 3600.0
        pdir = a.out.resolve()
        run, ref = find_ref(pdir)
        gen = -1
    else:
        if None in (a.gen, a.i0, a.j0, a.land_hour):
            raise SystemExit(f"{UNIT}: give --gen --i0 --j0 --land-hour, or --window --landing-s")
        run, ref = find_ref(a.out.resolve())
        i0, j0, H, gen = a.i0, a.j0, float(a.land_hour), a.gen
        pdir = window.gen_dir(run, gen)
    src = ref["source"]
    for k, want in (("start", a.start), ("source", a.source)):
        if want is not None and str(want) != str(ref[k]):
            raise SystemExit(f"{UNIT}: --{k} {want} is not this run's ({ref[k]}, {run}/follow/ref-grid.json)")
    if a.hours is not None and float(a.hours) != float(ref["hours"]):
        raise SystemExit(f"{UNIT}: --hours {a.hours:g} is not this run's ({ref['hours']:g})")
    if src not in SOURCES_OK:
        raise SystemExit(f"{UNIT}: source {src!r}: a window move needs a full state at the landing time, which the "
                         f"rows {SOURCES_OK} have (the native NAM analysis exists only at the start)")
    start = datetime.strptime(ref["start"], "%Y%m%d%H")
    hours = float(ref["hours"])
    if H != int(H):
        raise SystemExit(f"{UNIT}: landing at +{H:g} h: moves land on whole boundary hours")
    H = int(H)
    every = int(ref.get("landable_every_h") or 3)
    if H % every or not 0 < H < hours:
        raise SystemExit(f"{UNIT}: landing at +{H} h is not a landable time of this run (every {every} h, inside "
                         f"0 < H < {hours:g}: the {src} state exists only there)")
    t_land = start + timedelta(hours=H)
    pdir.mkdir(parents=True, exist_ok=True)
    for f in ("READY", "READY.json"):
        (pdir / f).unlink(missing_ok=True)
    rg = ref["grid"]
    g = window.window_grid(rg, i0, j0)
    name = window.window_name(ref["name"], i0, j0)
    move = {"gen": gen, "decided_hour": a.decided_hour, "land_hour": H, "land_time": f"{t_land:%Y-%m-%dT%H:%M}Z",
            "to": [i0, j0], "center_geo": g["center_geo"], "forced": bool(a.forced)}
    meta = window.gen_meta(rg, ref["name"], gen, i0, j0, H, [move] if gen > 0 else [], ref["start"])
    gen_json = window.write_gen_meta(run, meta, pdir)
    merged = pdir / f"static-{name}.npz"
    cfgfile = pdir / f"nam_configfile_{name}"
    keys = window.write_configfile(cfgfile, g, name, merged, bool(ref.get("spec_adv")), follow_meta=gen_json,
                                   base=Path(ref["configfile"]))
    # every default config.load() in this process (and the init chain) reads the window's configfile
    os.environ["NAMGPU_CONFIGFILE"] = str(cfgfile)
    os.environ["NAMGPU_BOCO_AGE_H"] = str(ref["boco_age_h"])
    rows = a.rrtm_rows or int(ref.get("rrtm_rows") or 8)
    os.environ["NAMGPU_RRTM_ROWS"] = str(rows)

    from namgpu.nest.cli import Sampler
    sampler = Sampler().start()
    stages, rec = {}, {"unit": UNIT, "gen": gen, "window": name, "offset": [i0, j0], "land_hour": H,
                       "land_time": move["land_time"], "source": src, "move": move, "host": os.uname().nodename,
                       "card": domain.card_name(), "prep_dir": str(pdir)}

    def stage(nm):
        sampler.mark(nm)
        stages[nm] = {"t0": time.perf_counter()}

    def done(nm, **kw):
        s = stages[nm]
        s["wall_s"] = round(time.perf_counter() - s.pop("t0"), 2)
        s.update(kw)
        print(f"{UNIT}: {pdir.name} {nm} {s['wall_s']} s", flush=True)

    t_all = time.perf_counter()
    rec["started"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    card_ix = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or None
    rec["card_cotenants_start"] = cotenants(card_ix)
    stage("wait-vram")
    pf = wait_room(g, rows, a.wait_min, card_ix)
    done("wait-vram", free_gib=pf["free_gib"], ok=pf["ok"], waited_s=pf["waited_s"])
    if not pf["ok"]:
        rec["refused"] = (f"waited {a.wait_min} min: the card has {pf['free_gib']} GiB free, the window needs about "
                          f"{pf['tried'][-1]['total_gib']} GiB")
        print(f"{UNIT}: refused: {rec['refused']}", flush=True)
        _finish(pdir, rec, stages, sampler, t_all, ok=False)
        return 3
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=int(pf["pick"]["pool_limit_bytes"]))
    rec["pool_limit_gib"] = pf["pick"]["pool_limit_gib"]

    import numpy as np
    from namgpu import config as configmod
    from namgpu.init import static_build as sb
    cfg = configmod.load(cfgfile)

    # 1. static: fresh build (cached by grid key, as the door caches), then stamped from the run's mosaic
    stage("static-build")
    cache = Path(os.environ.get("NAMGPU_STATIC_CACHE", str(domain.ROOT / "static" / "nest-cache")))
    # "-win": built with the window geometry (reference origin + whole-cell offset, the move step); a cache file of the same
    # grid built before that computes its coordinates from its own wbd/sbd and is not reused
    fresh = cache / f"{name}-{domain.grid_key(g)}-ds{sb.datasets_key()}-{start:%Y%m%d}-win.npz"
    cached = fresh.is_file()
    if not cached:
        res, brec = sb.build(cfg, datetime(start.year, start.month, start.day))
        sb.write(res, brec, fresh, cfg)
        del res
        cp.get_default_memory_pool().free_all_blocks()
    done("static-build", cached=cached, path=str(fresh))
    stage("static-stamp")
    z = np.load(fresh)
    merged_f, align = stamp_static(run, ref, {k: z[k] for k in z.files}, i0, j0)
    write_static(merged, merged_f, fresh, align)
    done("static-stamp", **align)
    rec["alignment"] = align

    # 2. boundaries from H to the end, named from the run start
    stage("bocos")
    from types import SimpleNamespace
    from namgpu import grid as gridmod
    gr = gridmod.build(cfg)
    bdir = pdir / "boco"
    bctx = SimpleNamespace(cfg=cfg, grid=gr, stream=cp.cuda.Stream(non_blocking=True), case_dir=Path(ref["case_dir"]),
                           start=start, init_source=src, parent_run=ref.get("parent_run"),
                           boco_interval_h=ref.get("boco_interval_h"), tables={})
    with bctx.stream:
        if src == "parent":
            from namgpu.init import parent_bocos
            parent_bocos.build(bctx, int(hours + 0.999), bdir, from_hour=H, generation=gen)
        else:
            from namgpu.init import gfs_bocos
            gfs_bocos.build(bctx, int(hours + 0.999), bdir, from_hour=H, generation=gen)
    bctx.stream.synchronize()
    del bctx
    cp.get_default_memory_pool().free_all_blocks()
    view = pdir / "boco-view"
    shutil.rmtree(view, ignore_errors=True)
    view.mkdir()
    files = sorted(bdir.glob("boco.[0-9]*"))
    for f in files:
        (view / f"boco.{int(f.name.split('.')[1]) - H:04d}").symlink_to(f.resolve())
    done("bocos", files=len(files), first=files[0].name if files else None, last=files[-1].name if files else None,
         dir=str(bdir))

    # 3. the fresh start at T from the source's state at T
    stage("init")
    if src == "parent":
        case_dir, links = Path(ref["case_dir"]), []
    else:
        case_dir, links = virtual_case(ref, pdir, t_land, H + int(ref["boco_age_h"]))
    rec["start_files"] = links
    from namgpu import state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver, slab_init
    st = statemod.State(gr)
    ctx = Ctx(cfg=cfg, grid=gr, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = case_dir
    ctx.start = ctx.start_time = t_land
    ctx.init_source = src
    ctx.parent_run = ref.get("parent_run")
    ctx.forecast_hours = max(6, int(hours - H + 0.999))
    ctx.out_dir = pdir / "init-run"
    ctx.out_dir.mkdir(exist_ok=True)
    ctx.boco_dir = str(view)
    ctx.driver = Driver(ctx)
    ctx.dt = cfg.dt
    chain = [u for u in init_case_driver.CHAIN if u not in ("init-fetch-case", "init-gfs-bocos")]
    with ctx.stream:
        irec = init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    # The fresh start's own TDFI (filter_method 3, as the door's generation 0 runs it), on boundaries named from the
    # landing hour (boco-view): the strip the next segment takes from this start is then a filtered state, not the raw
    # source interpolation. Breakage without it: the unfiltered 364 km strip of the first door smoke (S4, 2026-11-05)
    # gave a window-mean |dPD/dt| of 0.54 Pa/s one minute after the move against 0.023 before, still 1.9x after an
    # hour ; the move step measured the same transient (slab_init save --tdfi).
    tdfi_s = None
    if int(cfg.filter_method) > 0 and os.environ.get("NAMGPU_FOLLOW_FRESH_TDFI", "1") != "0":
        from namgpu.steps import drv_dfi_tdfi_driver
        t_ = time.perf_counter()
        ctx.filter_active = True
        try:
            with ctx.stream:
                drv_dfi_tdfi_driver.run(ctx)
        finally:
            ctx.filter_active = False
        ctx.ntimestep = 0
        ctx.first_step = True
        ctx.stream.synchronize()
        tdfi_s = round(time.perf_counter() - t_, 2)
    rec["fresh_tdfi_s"] = tdfi_s
    # The segment steps on from the run's step count, so its boundary files are the ones named from the run start
    # (boco.<H>, ...), and the interval that starts at H is read into the boundary arrays here, as READ_BC reads it at
    # that step: the new window's strips carry the boundary state and tendency from the landing on. (The init itself
    # read tboco from boco-view/boco.0000.) Breakage without it: the first moving run's segments took boco-view
    # names counted from H, so a segment that landed at +3 h read the +6 h file at +6 h (3 h late) and one that
    # landed at +9 h died looking for boco-view/boco.0009 (2026-11-05 05:12Z).
    ctx.boco_dir = str(bdir)
    from namgpu.boundary import bnd_read_bc
    ctx.ntimestep = int(round(H * 3600.0 / float(cfg.dt)))
    with ctx.stream:
        hdr = bnd_read_bc.run(ctx)
    ctx.stream.synchronize()
    ctx.ntimestep = 0
    rec["bc_read_at_landing"] = hdr
    done("init", stages=irec.get("stages"), nonfinite=irec.get("nonfinite"), bc_read=hdr)
    stage("save")
    init_dir = a.init_dir or Path("/dev/shm") / f"{run.name}-{pdir.name}-init"
    shutil.rmtree(init_dir, ignore_errors=True)
    srec = slab_init.save(ctx, init_dir)
    done("save", gib=srec.get("gib"), arrays=srec.get("arrays"), dir=str(init_dir))
    del ctx, st
    cp.get_default_memory_pool().free_all_blocks()
    rec["paths"] = {"gen_json": str(gen_json), "configfile": str(cfgfile), "static": str(merged),
                    "static_fresh": str(fresh), "boco_dir": str(bdir), "boco_view": str(view), "case_dir": str(case_dir),
                    "init_dir": str(init_dir)}
    rec["keys"] = keys
    rec["card_cotenants_end"] = cotenants(card_ix)
    rec["segment_contract"] = ("load paths.init_dir (slab_init.load: its ctx.boco_dir is paths.boco_dir, files named "
                               "from the run start, and its boundary arrays hold the interval read at the landing); "
                               "lay the restart set over it; NAMGPU_CONFIGFILE = paths.configfile")
    _finish(pdir, rec, stages, sampler, t_all, ok=True)
    return 0


def _finish(pdir: Path, rec: dict, stages: dict, sampler, t_all: float, ok: bool) -> None:
    sampler.stop()
    summ = sampler.summary()
    for nm, s in stages.items():
        s.pop("t0", None)
        b = summ["by_stage"].get(nm)
        if b:
            s.update(cores_mean=b["cores_mean"], cores_max=b["cores_max"], vram_peak_gib=round(b["vram_peak_mib"] / 1024, 2))
    rec["stages"] = stages
    rec["walls_s"] = {k: v.get("wall_s") for k, v in stages.items()}
    rec["single_core_windows_over_30s"] = summ["single_core_windows"]
    rec["single_core_windows"] = summ["single_core_windows"]          # the names the door's follow block reads
    rec["walls"] = rec["walls_s"]
    rec["waited_s"] = (stages.get("wait-vram") or {}).get("waited_s")
    rec["vram_peak_gib"] = round(summ["vram_peak_mib"] / 1024, 2)
    rec["wall_s"] = round(time.perf_counter() - t_all, 2)
    rec["ok"] = ok
    rec["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    txt = json.dumps(rec, indent=1, default=str)
    (pdir / "PREPARED.json").write_text(txt)
    if ok:
        (pdir / "READY.json").write_text(txt)
        (pdir / "READY").write_text(rec["finished"] + "\n")
    print(f"{UNIT}: {pdir.name} {'ready' if ok else 'FAILED'} in {rec['wall_s']} s: {rec['walls_s']}, "
          f"single-core windows {summ['single_core_windows'] or 'none'}", flush=True)


# ------------------------------------------------------------------------------------------ controller side

def launch(run: Path, gen: int, i0: int, j0: int, land_hour: float, card: str | None = None,
           decided_hour: float | None = None, forced: bool = False, wait_min: float = 30.0,
           env: dict | None = None) -> subprocess.Popen:
    """Start the preparer for generation gen in the background (own process group, log <gen dir>/prepare.log).
    card: the card index to prepare on (default: the caller's visible card). Kill by the returned Popen's pid."""
    gdir = window.gen_dir(run, gen)
    gdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "namgpu.nest.prepare", "--out", str(run), "--gen", str(gen), "--i0", str(i0),
           "--j0", str(j0), "--land-hour", str(land_hour), "--wait-min", str(wait_min)]
    if decided_hour is not None:
        cmd += ["--decided-hour", str(decided_hour)]
    if forced:
        cmd.append("--forced")
    e = dict(env or os.environ)
    e.pop("NAMGPU_CONFIGFILE", None)              # the preparer sets the window's own
    e.pop("CUPY_GPU_MEMORY_LIMIT", None)          # it caps its pool from its own estimate
    if card is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(card)
    with open(gdir / "prepare.log", "a") as log:
        return subprocess.Popen(cmd, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                cwd=str(domain.ROOT))


def wait_ready(pdir: Path, proc: subprocess.Popen | None = None, timeout_s: float = 1800.0) -> dict:
    """Wait for a preparation folder's READY (or the preparer's exit); returns PREPARED.json or raises by name."""
    pdir = Path(pdir)
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if (pdir / "READY").is_file():
            return _load(pdir / "PREPARED.json")
        if proc is not None and proc.poll() is not None and not (pdir / "READY").is_file():
            why = _load(pdir / "PREPARED.json").get("refused") if (pdir / "PREPARED.json").is_file() else None
            raise RuntimeError(f"{UNIT}: {pdir.name}: preparer exited rc {proc.returncode} without READY"
                               + (f": {why}" if why else f" (see {pdir / 'prepare.log'})"))
        time.sleep(2)
    raise TimeoutError(f"{UNIT}: {pdir.name} not ready after {timeout_s:.0f} s (see {pdir / 'prepare.log'})")


if __name__ == "__main__":
    sys.exit(main())
