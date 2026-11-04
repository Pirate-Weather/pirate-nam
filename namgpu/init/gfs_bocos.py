"""init-gfs-bocos: lateral boundary files boco.NNNN from GFS, in the format NPS writes and READ_BC reads.

The NPS boundary step (nam_nps.fd nemsinterp module_boundary.F90:630-1366), on the GPU. For every
boundary time (GFS every 3 h, the 6-h-old cycle as operational NAM: CONTRACT section 5) the GFS file
goes through the same path as the GFS init (gfs_start.model_state: metgrid-style columns,
init-gfs-hinterp, init-gfs-vinterp, wind rotation to grid-relative), into a small private state,
and the boundary strips are cut from the full model-level fields. boco.NNNN (NNNN = forecast hour of
the interval start) then holds, per variable, slot 1 = the value at the interval start and slot 2 =
(value at the end - value at the start) / tboco, tboco = 10800 s (module_boundary.F90:639-823).

File layout (Fortran sequential unformatted, little-endian 4-byte record markers, float32):
  record 1   runbc (logical, 4 bytes), idatbc(3) = day, month, year, ihrstbc (int32), tboco (real)
             of the interval start (module_boundary.F90:985-1000, 1360)
  record 2   PD    south, north, west, east                         (2-D H, write 1361)
  record 3-5 T, Q, CWM: south, north, west, east                     (3-D H, writes 1362-1364)
  record 6-7 U, V:      south, north, west, east                     (3-D V, writes 1365-1366)
  Each array is written whole in Fortran order: SOUTH(ids:ide, 1:lnsh[, 1:lm], 1:2) and
  WEST(1:lnsh, jds:jde[, 1:lm], 1:2), i.e. C order [2][LM][lnsh][IM] and [2][LM][JM][lnsh].
  Rows (FLTBNDS bocoh/bocov): H south j = 1..lnsh, north j = jde-lnsh+1..jde, west i = 1..lnsh, east
  i = ide-lnsh+1..ide; V south j = 1..lnsv, north j = jde-lnsv..jde-1, west i = 1..lnsv, east
  i = ide-lnsv..ide-1 (V(ide) and V(jde) are unused). l = 1 is the model top (KFLIP = L, 645).

Run: as the chain stage (run(ctx), skipped when every file for the run already exists), or before a
forecast on its own:  python -m namgpu.init.gfs_bocos --case 2025-03-14_12z --hours 48
Output: <case>/boco/boco.0000 ... boco.(hours-3), or <case>/boco-<domain_name>/ for a domain configfile
that names one (config.boco_dirname); ctx.boco_dir is set to that folder.
GPU: one private state of about 3 GB plus the decoded GFS set (~1 GB) per boundary time.
"""
from __future__ import annotations

import struct
from datetime import timedelta
from pathlib import Path

import numpy as np

UNIT = "init-gfs-bocos"
TBOCO = 10800.0
BC_FIELDS = ("T", "Q", "CW", "U", "V", "PD", "PINT", "FIS", "HGT_M", "SOILHGT",
             "GLAT", "GLON", "VLAT", "VLON")


def gfs_files(case_dir: Path, start, hours: int, interval_h: int = 3, age_h: int = 6):
    """[(forecast hour, path)] of the GFS boundary files: the cycle age_h hours older than start."""
    cyc = start - timedelta(hours=age_h)
    out = []
    for fh in range(0, int(hours) + 1, interval_h):
        p = Path(case_dir) / "raw" / "gfs" / f"gfs.t{cyc.hour:02d}z.f{fh + age_h:03d}.grib2"
        out.append((fh, p))
    return out


def strips(st, lm, lnsh, lnsv):
    """Device strips of one boundary time: dict name -> (south, north, west, east)."""
    import cupy as cp
    out = {}
    pd = st["PD"]
    out["PD"] = (pd[:lnsh, :], pd[-lnsh:, :], pd[:, :lnsh], pd[:, -lnsh:])
    for n in ("T", "Q", "CW"):
        a = st[n]
        out[n] = (a[:, :lnsh, :], a[:, -lnsh:, :], a[:, :, :lnsh], a[:, :, -lnsh:])
    for n in ("U", "V"):
        a = st[n]
        out[n] = (a[:, :lnsv, :], a[:, -lnsv - 1:-1, :], a[:, :, :lnsv], a[:, :, -lnsv - 1:-1])
    # Always COPY. ascontiguousarray returns the same view when the slice is already contiguous (PD
    # south/north rows), so the next GFS time overwrote prev in place: PD S/N got tend = 0 and slot 1
    # = the value 3 h ahead in every boco file (run lead 1120Z, scratch/runlead2/bcchk2.py).
    return {k: tuple(cp.array(x, copy=True, order="C") for x in v) for k, v in out.items()}


def _record(f, payload: bytes):
    f.write(struct.pack("<i", len(payload)))
    f.write(payload)
    f.write(struct.pack("<i", len(payload)))


def write_boco(path: Path, t0, old, new):
    """One boco file: slot 1 = old, slot 2 = (new - old) / TBOCO, in NPS record order."""
    import cupy as cp
    hdr = struct.pack("<i3iif", 1, t0.day, t0.month, t0.year, t0.hour, np.float32(TBOCO))
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        _record(f, hdr)
        for name in ("PD", "T", "Q", "CW", "U", "V"):
            parts = []
            for a0, a1 in zip(old[name], new[name]):
                both = cp.stack([a0, (a1 - a0) / cp.float32(TBOCO)])   # slot axis outermost (Fortran last)
                parts.append(cp.asnumpy(both).astype("<f4").tobytes())
            _record(f, b"".join(parts))
    tmp.replace(path)


def build(ctx, hours: int, out_dir: Path | None = None, log=print):
    """Write every boco file for a run of `hours` hours. Returns the folder."""
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import state as statemod
    from namgpu.init import gfs_start, init_grid_geo

    g, cfg = ctx.grid, ctx.cfg
    lm, lnsh, lnsv = int(g.lm), int(cfg.lnsh), int(cfg.lnsv)
    from namgpu import config as configmod
    out_dir = Path(out_dir or Path(ctx.case_dir) / configmod.boco_dirname(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = gfs_files(ctx.case_dir, ctx.start, hours)
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{UNIT}: GFS boundary files missing: {missing[:3]} (run init-fetch-case)")
    st = statemod.State(g, names=BC_FIELDS)
    bctx = SimpleNamespace(cfg=cfg, grid=g, st=st, stream=ctx.stream, tables={}, case_dir=ctx.case_dir,
                           start=ctx.start, first_step=True, init_source="gfs", analysis=None)
    init_grid_geo.run(bctx)
    static, _ = gfs_start.load_static(bctx)
    st.add("HGT_M", static["HGT_SFC"].shape)[...] = static["HGT_SFC"]
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    prev = None
    for k, (fh, path) in enumerate(files):
        gfs_start.model_state(bctx, path, tmpdir, sfc_keys=gfs_start.SFC_COLUMNS)
        cur = strips(st, lm, lnsh, lnsv)
        bad = {n: int(sum(int((~cp.isfinite(x)).sum()) for x in v)) for n, v in cur.items()}
        if any(bad.values()):
            raise RuntimeError(f"{UNIT}: non-finite boundary values at +{fh} h: {bad}")
        if prev is not None:
            t0 = ctx.start + timedelta(hours=files[k - 1][0])
            name = out_dir / f"boco.{files[k - 1][0]:04d}"
            write_boco(name, t0, prev, cur)
            log(f"{UNIT}: {name.name} (+{files[k - 1][0]} h -> +{fh} h)")
        prev = cur
    del st, bctx
    cp.get_default_memory_pool().free_all_blocks()
    return out_dir


def run(ctx, hours: int | None = None):
    """Chain stage: make sure every boco file of the run exists; sets ctx.boco_dir."""
    hours = hours or getattr(ctx, "forecast_hours", None)
    if hours is None:
        # parm/nam_configfile carries nhours_fcst as the job-script placeholder LENGTH; NAM runs 84 h,
        # our case runs 48 h. A numeric configfile value wins; the placeholder means the case default.
        raw = str(ctx.cfg.raw.get("nhours_fcst", "48")).strip()
        hours = int(raw) if raw.lstrip("-").isdigit() else 48
    hours = int(hours)
    from namgpu import config as configmod
    out_dir = Path(ctx.case_dir) / configmod.boco_dirname(ctx.cfg)
    want = [out_dir / f"boco.{fh:04d}" for fh in range(0, hours, 3)]
    if not all(p.is_file() for p in want):
        build(ctx, hours, out_dir)
    ctx.boco_dir = str(out_dir)
    return out_dir


def main(argv=None):
    import argparse
    from datetime import datetime
    import cupy as cp
    from namgpu import config, grid as gridmod
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--hours", type=int, default=48)
    a = ap.parse_args(argv)
    from namgpu import paths
    cfg = config.load()
    g = gridmod.build(cfg)
    from types import SimpleNamespace
    ctx = SimpleNamespace(cfg=cfg, grid=g, stream=cp.cuda.Stream(non_blocking=True),
                          case_dir=paths.case_dir(a.case),
                          start=datetime.strptime(Path(a.case).name[:14], "%Y-%m-%d_%Hz"))
    with ctx.stream:
        out = build(ctx, a.hours)
    ctx.stream.synchronize()
    print(f"boundaries in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
