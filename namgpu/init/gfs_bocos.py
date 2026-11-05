"""init-gfs-bocos: lateral boundary files boco.NNNN from GFS (or any isobaric SOURCES row with a "boundary"
entry: IFS open data, ERA5), in the format NPS writes and READ_BC reads.

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

import contextlib
import struct
from datetime import timedelta
from pathlib import Path

import numpy as np

UNIT = "init-gfs-bocos"
TBOCO = 10800.0
PREFETCH = 2          # boundary times decoded ahead of the interpolation (host decode threads, same stream)
BC_FIELDS = ("T", "Q", "CW", "U", "V", "PD", "PINT", "FIS", "HGT_M", "SOILHGT",
             "GLAT", "GLON", "VLAT", "VLON")


def gfs_files(case_dir: Path, start, hours: int, interval_h: int | None = None, age_h: int | None = None,
              row: dict | None = None, cfg=None):
    """[(forecast hour, path)] of the boundary files: leads of the cycle age_h hours older than start, every
    interval_h. Both default to the source row (sources.SOURCES["gfs"]: boco_age_h 6 as operational NAM,
    boco_interval_h 3); age_h also follows NAMGPU_BOCO_AGE_H and the configfile key boco_age_h (0 = the start
    cycle itself, for past cases). File names are the row's boco_file (the names fetch_case writes)."""
    from namgpu.init import sources as srcmod
    row = row or srcmod.SOURCES["gfs"]
    age_h = srcmod.boco_age_h(row, cfg, age_h)
    interval_h = int(interval_h or row.get("boco_interval_h", 3))
    cyc = start - timedelta(hours=age_h)
    out = []
    for fh in range(0, int(hours) + 1, interval_h):
        p = Path(case_dir) / row["boco_file"].format(bcyc=cyc.hour, lead=fh + age_h)
        out.append((fh, p))
    return out


def _marker(start, hours, files, age_h, interval_h, source="gfs", grid=None):
    """What a boco folder was built from (BOCO-SOURCE.json): a folder built from another boundary cycle,
    interval, file set or grid (config.grid_stamp) is rebuilt, not reused."""
    return {"source": source, "start": f"{start:%Y-%m-%dT%H}", "hours": int(hours), "age_h": int(age_h),
            "interval_h": int(interval_h), "files": [Path(p).name for _, p in files], "grid": grid}


def boundary_row(ctx):
    """(name, row) of the source the boundaries come from: the start's own row when it has a "boundary" entry
    (gfs, ifs, ...), else GFS (the native NAM start takes GFS boundaries, as operational NAM)."""
    from namgpu.init import sources as srcmod
    name, row = srcmod.source_row(ctx)
    if "boundary" not in row:
        name, row = "gfs", srcmod.SOURCES["gfs"]
    return name, row


def source_files(case_dir: Path, start, hours: int, row: dict, cfg=None):
    """[(forecast hour, path)] of a row's boundary files: its cycle age_h older than start (sources.boco_age_h:
    NAMGPU_BOCO_AGE_H, configfile boco_age_h, else the row), every interval_h."""
    from namgpu.init import sources as srcmod
    b = dict(row["boundary"], age_h=srcmod.boco_age_h(row, cfg))
    if int(b["interval_h"]) * 3600 != int(TBOCO):
        raise ValueError(f"{UNIT}: boundary interval {b['interval_h']} h; boco files carry tboco = {TBOCO} s")
    cyc = start - timedelta(hours=int(b["age_h"]))
    return [(fh, Path(case_dir) / b["file"].format(bcyc=cyc.hour, lead=fh + int(b["age_h"])))
            for fh in range(0, int(hours) + 1, int(b["interval_h"]))]


def boco_dir(ctx) -> Path:
    """<case>/boco[-<domain>] for GFS boundaries (and the native start, which takes GFS's), with -<source> added
    for any other boundary source: a folder of GFS boundaries must never be reused by an IFS run of the same
    domain (run() skips the build when every file exists)."""
    from namgpu import config as configmod
    name, _ = boundary_row(ctx)
    d = configmod.boco_dirname(ctx.cfg)
    return Path(ctx.case_dir) / (d if name == "gfs" else f"{d}-{name}")


class ReadAhead:
    """Host reads of a regular-netcdf source's boundary files (gfs_start.read_regular_netcdf), `depth` files at
    once ahead of the GPU work. Each read is one single-core Rust process (~20 s for an ERA5 hour of the NAM band),
    so one at a time left the card idle for 9 x 20 s of a 24 h ERA5 build (189 s, 2026-11-05; 49 s with 6 ahead);
    the values are the same file reads. Host memory: up to depth + 1 decoded files (~1 GB each for the NAM 12 km band)."""

    def __init__(self, paths, depth):
        from concurrent.futures import ThreadPoolExecutor
        from namgpu.init import gfs_start
        self._read = gfs_start.read_regular_netcdf
        self.paths, self.depth = list(paths), max(1, int(depth))
        self.ex = ThreadPoolExecutor(self.depth)
        self.fut = {k: self.ex.submit(self._read, self.paths[k]) for k in range(min(self.depth, len(self.paths)))}

    def get(self, k):
        res = self.fut.pop(k).result()
        nxt = k + self.depth
        if nxt < len(self.paths) and nxt not in self.fut:
            self.fut[nxt] = self.ex.submit(self._read, self.paths[nxt])
        return res

    def close(self):
        self.ex.shutdown(wait=False, cancel_futures=True)


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


def write_boco(path: Path, t0, old, new, tboco: float = TBOCO):
    """One boco file: slot 1 = old, slot 2 = (new - old) / tboco, in NPS record order (tboco 10800 s for GFS;
    the parent source can write 3600 s intervals, parent_bocos.py)."""
    import cupy as cp
    hdr = struct.pack("<i3iif", 1, t0.day, t0.month, t0.year, t0.hour, np.float32(tboco))
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        _record(f, hdr)
        for name in ("PD", "T", "Q", "CW", "U", "V"):
            parts = []
            for a0, a1 in zip(old[name], new[name]):
                both = cp.stack([a0, (a1 - a0) / cp.float32(tboco)])   # slot axis outermost (Fortran last)
                parts.append(cp.asnumpy(both).astype("<f4").tobytes())
            _record(f, b"".join(parts))
    tmp.replace(path)


def build(ctx, hours: int, out_dir: Path | None = None, log=print, from_hour: int = 0, generation: int | None = None):
    """Write every boco file for a run of `hours` hours. Returns the folder.
    from_hour (a storm-following window that lands at that hour, namgpu/nest/prepare.py): only the intervals from
    from_hour on, still named in hours from the run start (READ_BC names a file from the step count, which a
    moving run keeps from its start); generation is recorded in the folder's BOCO-SOURCE.json with the grid."""
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import state as statemod
    from namgpu.init import gfs_start, init_grid_geo, source_records as srec

    g, cfg = ctx.grid, ctx.cfg
    lm, lnsh, lnsv = int(g.lm), int(cfg.lnsh), int(cfg.lnsv)
    out_dir = Path(out_dir or boco_dir(ctx))
    out_dir.mkdir(parents=True, exist_ok=True)
    from namgpu.init import sources as srcmod
    bname, brow = boundary_row(ctx)
    rec = srec.table(brow)
    age_h = srcmod.boco_age_h(brow, cfg)
    interval_h = int(brow["boundary"]["interval_h"])
    files = source_files(ctx.case_dir, ctx.start, hours, brow, cfg)
    if from_hour:
        if int(from_hour) % interval_h:
            raise ValueError(f"{UNIT}: from_hour {from_hour}: a window can land only on a {bname} boundary time "
                             f"(every {interval_h} h from the start)")
        files = [(fh, p) for fh, p in files if fh >= int(from_hour)]
        if len(files) < 2:
            raise ValueError(f"{UNIT}: from_hour {from_hour} leaves no boundary interval before +{hours} h")
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{UNIT}: {bname} boundary files missing: {missing[:3]} (run init-fetch-case)")
    st = statemod.State(g, names=BC_FIELDS)
    bctx = SimpleNamespace(cfg=cfg, grid=g, st=st, stream=ctx.stream, tables={}, case_dir=ctx.case_dir,
                           start=ctx.start, first_step=True, init_source=bname, analysis=None)
    init_grid_geo.run(bctx)
    static, _ = gfs_start.load_static(bctx)
    st.add("HGT_M", static["HGT_SFC"].shape)[...] = static["HGT_SFC"]
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    prev = None
    # Decode PREFETCH boundary times ahead on worker threads (each decode is ~11 grib2_dump processes plus the
    # device placement, on bctx.stream) while this thread interpolates the current one: the decodes of one time
    # used ~5 of 24 cores and the boundary stage was a series of them (58 s for 17 times, nests 2026-11-04).
    from concurrent.futures import ThreadPoolExecutor
    # A regular-netcdf source (era5): its host read is the slow part (one single-core process per file, ~20 s per
    # ERA5 hour of the NAM band), so those reads go further ahead than the PREFETCH device decodes (ReadAhead).
    reads = None
    if rec.get("container") == "regular-netcdf":
        from namgpu import cores
        reads = ReadAhead([p for _, p in files], min(len(files), 6, max(1, cores.usable() // 2)))

    def _decode(k, path):
        scope = bctx.stream if bctx.stream is not None else contextlib.nullcontext()
        with scope:
            return gfs_start.decode_source(bctx, path, tmpdir, sfc_keys=rec["sfc_columns"], rec=rec,
                                           host=reads.get(k) if reads is not None else None)
    pool = ThreadPoolExecutor(max_workers=PREFETCH)
    ahead = [pool.submit(_decode, k, p) for k, (_, p) in enumerate(files[:PREFETCH])]
    for k, (fh, path) in enumerate(files):
        dec = ahead[k].result()
        if k + PREFETCH < len(files):
            ahead.append(pool.submit(_decode, k + PREFETCH, files[k + PREFETCH][1]))
        gfs_start.model_state(bctx, path, tmpdir, sfc_keys=rec["sfc_columns"], rec=rec, decoded=dec)
        del dec
        ahead[k] = None
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
    pool.shutdown()
    if reads is not None:
        reads.close()
    import json
    from namgpu import config as configmod
    mk = _marker(ctx.start, hours, files, age_h, interval_h, bname, configmod.grid_stamp(ctx.cfg))
    if from_hour or generation is not None:
        mk.update(from_hour=int(from_hour), generation=generation)
    # The case folder boco-<domain> is shared by every source of the cycle with that domain name. Breakage this
    # prevents: a parent nest (hourly files, SOURCE.json) overwrote a GFS nest's boco.0000 while the GFS marker
    # BOCO-SOURCE.json still matched, so the next GFS run of that domain reused the parent's hourly boco.0000
    # (tboco 3600 instead of 10800) and its moving segment asked for boco.0016 (P1 Ampil, 2026-11-05 06:29Z).
    (out_dir / "SOURCE.json").unlink(missing_ok=True)          # parent_bocos' marker: its files are gone
    (out_dir / "BOCO-SOURCE.json").write_text(json.dumps(mk))
    log(f"{UNIT}: {len(files) - 1} boco files from the {age_h} h old cycle "
        f"({ctx.start - timedelta(hours=age_h):%Y-%m-%d %HZ}) in {out_dir}")
    del st, bctx
    from namgpu import devmem  # noqa: PLC0415
    devmem.release()
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
    from namgpu.init import sources as srcmod
    out_dir = Path(ctx.case_dir) / configmod.boco_dirname(ctx.cfg)    # parent nests
    _, row = srcmod.source_row(ctx)
    if row["kind"] == "nam_history":
        # one-way nest: boundaries from the parent's own history (parent_bocos.py), every 1 or 3 h
        from namgpu.init import parent_bocos
        if not parent_bocos.current(out_dir, hours, ctx):     # same parent, interval and start, every file
            parent_bocos.build(ctx, hours, out_dir)
        ctx.boco_dir = str(out_dir)
        return out_dir
    from namgpu.domains import bc_hours
    hours = bc_hours(hours)      # 3-hourly GFS steps: a 7 h run reads boco.0006, so it needs GFS to +9 h
    out_dir = boco_dir(ctx)
    want = [out_dir / f"boco.{fh:04d}" for fh in range(0, hours, 3)]
    import json
    bname, row = boundary_row(ctx)
    age_h, interval_h = srcmod.boco_age_h(row, ctx.cfg), int(row["boundary"]["interval_h"])
    grid = configmod.grid_stamp(ctx.cfg)
    mk = out_dir / "BOCO-SOURCE.json"
    try:
        have = json.loads(mk.read_text())
    except (OSError, ValueError):
        have = None
    if have is not None and (out_dir / "SOURCE.json").is_file():
        have = None                    # a parent build wrote this folder after the GFS marker (see build)
    if have is None:
        # a folder from before the marker: kept if complete only for the fixed NAM grid (folder "boco"); a
        # boco-<domain> folder is shared by every grid of that name, so without a grid record it is rebuilt
        same = not ctx.cfg.raw.get("domain_name")
        why = "no source marker (grid unknown)"
    else:
        want_m = {"source": bname, "age_h": age_h, "interval_h": interval_h, "start": f"{ctx.start:%Y-%m-%dT%H}",
                  "grid": grid}
        diff = [k for k, v in want_m.items() if have.get(k, "gfs" if k == "source" else None) != v]
        same = not diff
        why = "; ".join(f"{k} {have.get(k)} -> {want_m[k]}" for k in diff)
    if not same:
        print(f"{UNIT}: {out_dir.name} does not match this run ({why}): rebuilding", flush=True)
    if not same or not all(p.is_file() for p in want):
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
    ap.add_argument("--boco-age", type=int, default=None, help="boundary cycle age, h (default: the source row)")
    a = ap.parse_args(argv)
    from namgpu import paths
    if a.boco_age is not None:
        import os
        os.environ["NAMGPU_BOCO_AGE_H"] = str(a.boco_age)
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
