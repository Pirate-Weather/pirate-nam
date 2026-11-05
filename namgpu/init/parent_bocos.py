"""init-parent-bocos: the nest's lateral boundary files boco.NNNN from the parent's own history (one-way, ndown-like).

The 'parent' row's half of the chain stage init-gfs-bocos (gfs_bocos.run dispatches here for kind nam_history). For
each boundary time the parent's hand-off history at start + fh goes through the same columns + init-gfs-vinterp +
wind rotation as the start (parent_start.model_state) into a small private state, and the strips are cut and written
by gfs_bocos.strips / gfs_bocos.write_boco, so the files are byte-for-byte the format READ_BC reads.

Interval: 3 h by default (configfile key boco_interval_h, or ctx.boco_interval_h). 1 h is the knob: READ_BC
names the file of every interval from the forecast hour (bnd_read_bc.boco_file_name) and takes its cadence from
tboco in boco.0000 (drv-consts), so hourly files need nothing else; tboco is written per interval.
Condensate: the boundary CW is the vinterp CWM sum of the parent's species (do_clouds on this private config),
not 0 as from GFS.
Reads: every boundary time's parent window at once, one process per (time, field), all cores, before the GPU loop.
"""
from __future__ import annotations

import dataclasses
import time
from datetime import timedelta
from pathlib import Path

UNIT = "init-parent-bocos"
NAMES = ("PRES", "PH", "T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "U", "V",
         "PSFC", "HGT", "T2", "Q2", "U10", "V10")


def interval_h(ctx) -> int:
    v = getattr(ctx, "boco_interval_h", None) or ctx.cfg.raw.get("boco_interval_h") or 3
    v = int(v)
    if v not in (1, 3):
        raise ValueError(f"{UNIT}: boco_interval_h {v}: the parent history is hourly and READ_BC's names are whole "
                         "forecast hours; 1 or 3")
    return v


def parent_files(run: Path, start, hours: int, step_h: int):
    from namgpu.init import parent_start as ps
    return [(fh, ps.history_path(run, start + timedelta(hours=fh))) for fh in range(0, int(hours) + 1, step_h)]


def build(ctx, hours: int, out_dir: Path, log=print):
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import state as statemod
    from namgpu.init import gfs_bocos, gfs_start, init_grid_geo, parent_start as ps

    g, cfg = ctx.grid, ctx.cfg
    lm, lnsh, lnsv = int(g.lm), int(cfg.lnsh), int(cfg.lnsv)
    step = interval_h(ctx)
    run = ps.parent_run(ctx)
    files = parent_files(run, ctx.start, hours, step)
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{UNIT}: parent history missing for {len(missing)} boundary times: {missing[:3]} "
                                f"(the parent run must cover start + {hours} h)")
    pg = ps.parent_grid(files[0][1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STAMP).unlink(missing_ok=True)       # files from another parent are not this run's
    bcfg = dataclasses.replace(cfg, do_clouds=True)
    st = statemod.State(g, names=gfs_bocos.BC_FIELDS)
    bctx = SimpleNamespace(cfg=bcfg, grid=g, st=st, stream=ctx.stream, tables={}, case_dir=ctx.case_dir,
                           start=ctx.start, first_step=True, init_source="parent", analysis=None)
    init_grid_geo.run(bctx)
    static, _ = gfs_start.load_static(bctx)
    st.add("HGT_M", static["HGT_SFC"].shape)[...] = static["HGT_SFC"]
    pl = ps.placement(st, pg)
    ps.check_inside(pl)
    t0 = time.perf_counter()
    got = ps.read_windows([(p, n, pl["window"]) for _, p in files for n in NAMES])
    t_read = time.perf_counter() - t0
    log(f"{UNIT}: read {len(files)} parent times x {len(NAMES)} fields (window "
        f"{pl['window'][3] - pl['window'][2]}x{pl['window'][1] - pl['window'][0]}) in {t_read:.1f} s on "
        f"{min(len(files) * len(NAMES), 24)} processes")
    tboco = float(step * 3600)
    prev = None
    t1 = time.perf_counter()
    for k, (fh, path) in enumerate(files):
        f = {n: cp.asarray(got.pop((str(path), n))) for n in NAMES}
        ps.model_state(bctx, f, pg, pl)
        del f
        cur = gfs_bocos.strips(st, lm, lnsh, lnsv)
        bad = {n: int(sum(int((~cp.isfinite(x)).sum()) for x in v)) for n, v in cur.items()}
        if any(bad.values()):
            raise RuntimeError(f"{UNIT}: non-finite boundary values at +{fh} h: {bad}")
        if prev is not None:
            t0b = ctx.start + timedelta(hours=files[k - 1][0])
            name = out_dir / f"boco.{files[k - 1][0]:04d}"
            gfs_bocos.write_boco(name, t0b, prev, cur, tboco=tboco)
            log(f"{UNIT}: {name.name} (+{files[k - 1][0]} h -> +{fh} h, tboco {tboco:.0f} s)")
        prev = cur
    log(f"{UNIT}: {len(files) - 1} boco files from {run} in {time.perf_counter() - t1:.1f} s on the GPU "
        f"(+{t_read:.1f} s reads); smallest margin {pl['min_cells']} parent cells ({pl['min_km']} km)")
    write_stamp(out_dir, ctx)
    del st, bctx
    cp.get_default_memory_pool().free_all_blocks()
    return out_dir


def wanted(out_dir: Path, hours: int, step: int):
    return [Path(out_dir) / f"boco.{fh:04d}" for fh in range(0, int(hours), step)]


# The boco folder is keyed by the domain (config.boco_dirname), not by the parent: two parents for one nest domain
# would share it, and a run would silently take the other parent's boundaries (found 2026-11-04 when the box's native
# parent was about to reuse the mini parent's files). Each build writes SOURCE.json; a mismatch rebuilds.
STAMP = "SOURCE.json"


def stamp(ctx) -> dict:
    from namgpu.init import parent_start as ps
    from namgpu import config as configmod
    return {"parent_run": str(ps.parent_run(ctx).resolve()), "boco_interval_h": interval_h(ctx),
            "start": f"{ctx.start:%Y-%m-%dT%H}", "grid": configmod.grid_stamp(ctx.cfg)}


def current(out_dir: Path, hours: int, ctx) -> bool:
    import json
    p = Path(out_dir) / STAMP
    try:
        same = json.loads(p.read_text()) == stamp(ctx)
    except (OSError, ValueError):
        same = False
    return same and all(f.is_file() for f in wanted(out_dir, hours, interval_h(ctx)))


def write_stamp(out_dir: Path, ctx) -> None:
    import json
    (Path(out_dir) / STAMP).write_text(json.dumps(stamp(ctx)))
