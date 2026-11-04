"""init-gwd-orog: gravity-wave-drag orographic statistics on the 12 km B grid from 30-arcsec terrain.

Unit: init-gwd-orog (group init, mode new; NPS geogrid do_gwd path, lsmask_orog.f calc_orog).
Kernels: namgpu/kernels/init_gwd_orog.cu, seven launches. Passes 1-3 run over the find_bounds_ll
window of the 43200 x 21600 source grid (one thread per source cell); the finalisers run one thread
per model cell. The window is computed once per run from the model H-point geometry. Host syncs
happen only at init (the origin check, the zero-count refusal and the zero-denominator count).

In (ctx):
    ctx.tables["gwd_dem"]   cupy int16 [JSRC=21600, ISRC=43200], the 30-arcsec terrain (m), water
                            = -9999, row 0 = northernmost row, column 0 = lon -180 + dlon/2. Build it
                            with load_gwd_dem(path) from the raw NPS orography file.
    ctx.tables["grid_sw"]   cupy float32 [2] radians, geographic (lat, lon) of the SW H point,
                            from init-grid-geo.
    ctx.st["GLAT"], ctx.st["GLON"]  float32 [JM, IM] radians, H-point geographic coordinates.
    ctx.cfg.tph0d, tlm0d, dphd, dlmd, wbd, sbd  rotated grid (configfile; NPS centlat/centlon_parent,
                            dy, dx, and the SW origin wbd/sbd that the origin check compares with).
Out (ctx.st, [JM, IM] float32, NPS GWD field names, output_nemsinit.F90:262-278):
    HSTDV (orog_stnd_dev), HCNVX (convexity), HASYW/HASYS/HASYSW/HASYNW (oa 1..4),
    HLENW/HLENS/HLENSW/HLENNW (ol 1..4), HANGL (theta), HANIS (gamma), HSLOP (sigma), HZMAX (max_height)
Scratch (ctx.st, zeroed each run): GWD_COUNT, GWD_SUMTOPO, GWD_SDX2, GWD_SDY2, GWD_SDXY, GWD_TOPO,
    GWD_SUM2, GWD_SUM4 (float32); GWD_NUL, GWD_NLL, GWD_NUR, GWD_NLR, GWD_NUL2, GWD_NLL2, GWD_NUR2,
    GWD_NLR2, GWD_CNUL, GWD_CNLL, GWD_CNUR, GWD_CNLR (int32).
Origin scalars (wbd, sbd of ll2xy_bgrid_pt_loc) are computed on device into ctx.tables["gwd_origin"]
and refused on the host if they differ from cfg.wbd/sbd or the asin argument leaves [-1, 1].
Refusals (named breakage): zero source count in any model cell (Fortran aborts at 1552-1555; here
the topo mean and derivative means would be NaN); origin mismatch (wrong SW corner, every source cell
maps to the wrong model cell).

Self-test (full-size DEM and model grid, geometry generated from the configured origin, NaN/Inf check):
    source ~/nam-gpu/env.sh && python -m namgpu.init.init_gwd_orog
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np

from namgpu import kernels

UNIT = "init_gwd_orog"
ISRC = 43200
JSRC = 21600
DTR = math.pi / 180.0
OUTPUTS = ("HSTDV", "HCNVX", "HASYW", "HASYS", "HASYSW", "HASYNW",
           "HLENW", "HLENS", "HLENSW", "HLENNW", "HANGL", "HANIS", "HSLOP", "HZMAX")
FLOAT_SCRATCH = ("GWD_SUMTOPO", "GWD_SDX2", "GWD_SDY2", "GWD_SDXY", "GWD_TOPO", "GWD_SUM2", "GWD_SUM4")
INT_SCRATCH = ("GWD_COUNT", "GWD_NUL", "GWD_NLL", "GWD_NUR", "GWD_NLR", "GWD_NUL2", "GWD_NLL2",
               "GWD_NUR2", "GWD_NLR2", "GWD_CNUL", "GWD_CNLL", "GWD_CNUR", "GWD_CNLR")
# Fortran find_bounds_ll constants (lsmask_orog.f 1240-1243, find_bounds_ll.f cushion)
F32 = np.float32
DLON_SRC = F32(1.0) / F32(120.0)
DLAT_SRC = -(F32(1.0) / F32(120.0))
LON11_SRC = F32(-180.0) + DLON_SRC * F32(0.5)
LAT11_SRC = F32(90.0) + DLAT_SRC * F32(0.5)
CUSHION = F32(2.0)


def _scalars(cfg):
    """Rotated-grid scalars of ll2xy_bgrid_pt_loc, in radians (NPS first-call block)."""
    tph0 = cfg.tph0d * DTR
    tlm0 = cfg.tlm0d * DTR
    return dict(
        tlm0=np.float32(tlm0),
        stph0=np.float32(math.sin(tph0)),
        ctph0=np.float32(math.cos(tph0)),
        dph=np.float32(cfg.dphd * DTR),
        dlm=np.float32(cfg.dlmd * DTR),
    )


def _nint(x):
    """Fortran NINT: nearest integer, halves away from zero."""
    x = float(x)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def find_bounds(glat, glon, im, jm):
    """Fortran find_bounds_ll (find_bounds_ll.f, one task holding the full model domain).

    glat, glon: cupy float32 [JM, IM] radians. Returns 1-based (istart, iend, jstart, jend) on the
    30-arcsec source grid. i may run below 1 or above ISRC near the dateline; the kernels wrap it.
    """
    import cupy as cp

    lat_deg = glat * F32(180.0 / math.pi)
    lon_deg = glon * F32(180.0 / math.pi)
    lat_max = float(cp.max(lat_deg))
    lat_min = float(cp.min(lat_deg))

    maxlat = min(F32(lat_max) + CUSHION, F32(90.0))
    minlat = max(F32(lat_min) - CUSHION, F32(-90.0))
    jstart = _nint((maxlat - LAT11_SRC) / DLAT_SRC + F32(1.0))
    jend = _nint((minlat - LAT11_SRC) / DLAT_SRC + F32(1.0))
    jstart = max(jstart, 1)
    jend = min(jend, JSRC)

    if lat_max > 85.0 or lat_min < -85.0:
        istart, iend = 1, ISRC           # near a pole all i points are checked
    else:
        imid0 = (im - 1) // 2            # Fortran imid/jmid, 1 task: (n-1)/2 + 1, 0-based here
        jmid0 = (jm - 1) // 2
        lon_c = F32(float(lon_deg[jmid0, imid0]))
        diff = cp.mod(lon_deg - lon_c + F32(3600.0), F32(360.0))
        west = diff > F32(180.0)
        big = F32(99999.0)
        minlon = float(cp.min(cp.where(west, diff - F32(360.0), big)))
        maxlon = float(cp.max(cp.where(west, -big, diff)))
        istart = _nint((F32(minlon) + lon_c - CUSHION - LON11_SRC) / DLON_SRC + F32(1.0))
        iend = _nint((F32(maxlon) + lon_c + CUSHION - LON11_SRC) / DLON_SRC + F32(1.0))
    print(f"{UNIT}: SOURCE GRID BOUNDS IS/IE/JS/JE: {istart} {iend} {jstart} {jend}")
    return istart, iend, jstart, jend


def load_gwd_dem(path):
    """Read the NPS 30-arcsec orography file (raw big-endian int16, row 0 = northernmost).

    Fortran lsmask_orog.f 1336-1360 reads it with MPI-IO and to_native_endianness; here numpy's
    '>i2' does the byte swap. Returns a cupy int16 [JSRC, ISRC] array.
    """
    import cupy as cp

    need = JSRC * ISRC * 2
    size = os.path.getsize(path)
    if size != need:
        raise ValueError(f"{UNIT}: orography file {path} is {size} bytes, expected {need} "
                         f"(big-endian int16 {JSRC} x {ISRC})")
    raw = np.fromfile(path, dtype=">i2")
    return cp.asarray(raw.reshape(JSRC, ISRC).astype(np.int16))


def dem_from_wps_tiles(tile_dir, workers=32):
    """Assemble the global 30-arcsec terrain from a WPS geogrid tile set (topo_gmted2010_30s: USGS
    GMTED2010, public domain) into the load_gwd_dem layout: int16 [JSRC, ISRC], row 0 = northernmost,
    column 0 = lon -180 + dlon/2. Tiles are big-endian int16, tile_x = tile_y = 1200 with a tile_bdr
    border, x from lon 0 eastwards, y from the south pole northwards (WPS index file). Tiles are read
    on `workers` threads (host file I/O at init, not on the step path). Returns a numpy array.

    Declared difference from NAM: NPS reads its own raw 30-arcsec orography file with water = -9999;
    GMTED2010 carries 0 m over the ocean, so water cells enter the statistics at sea level.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    tile_dir = Path(tile_dir)
    idx = {}
    for line in (tile_dir / "index").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            idx[k.strip()] = v.strip().strip('"')
    tx, ty = int(idx["tile_x"]), int(idx["tile_y"])
    bdr = int(idx.get("tile_bdr", 0))
    if int(idx["wordsize"]) != 2 or abs(float(idx["dx"]) * 120.0 - 1.0) > 1e-4:
        raise ValueError(f"{UNIT}: {tile_dir} is not a 2-byte 30-arcsec tile set")
    if abs(float(idx["known_lon"]) - 1.0 / 240.0) > 1e-4 or abs(float(idx["known_lat"]) + 90.0 - 1.0 / 240.0) > 1e-4:
        raise ValueError(f"{UNIT}: {tile_dir} origin is not (lat -90, lon 0) + half a cell")
    south_first = np.empty((JSRC, ISRC), dtype=np.int16)

    def one(ij):
        i0, j0 = ij
        name = f"{i0 + 1:05d}-{i0 + tx:05d}.{j0 + 1:05d}-{j0 + ty:05d}"
        a = np.fromfile(tile_dir / name, dtype=">i2")
        a = a.reshape(ty + 2 * bdr, tx + 2 * bdr)[bdr:bdr + ty, bdr:bdr + tx]
        south_first[j0:j0 + ty, i0:i0 + tx] = a
    jobs = [(i0, j0) for j0 in range(0, JSRC, ty) for i0 in range(0, ISRC, tx)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs))
    return np.ascontiguousarray(np.roll(south_first[::-1], ISRC // 2, axis=1))


def default_dem(ctx):
    """Find the 30-arcsec terrain for the run: NAMGPU_GWD_DEM (raw NPS big-endian file) if set, else the
    WPS GMTED2010 tile set under NAMGPU_GEOG (default /work/geog, then ~/WPS_GEOG), assembled once and
    cached next to it as gwd_dem_30s.npy. Returns a cupy int16 [JSRC, ISRC]."""
    import cupy as cp
    from pathlib import Path

    raw = os.environ.get("NAMGPU_GWD_DEM")
    if raw:
        return load_gwd_dem(raw)
    roots = [Path(os.environ["NAMGPU_GEOG"])] if os.environ.get("NAMGPU_GEOG") else []
    roots += [Path("/work/geog"), Path.home() / "WPS_GEOG"]
    for r in roots:
        tiles = r / "topo_gmted2010_30s"
        if (tiles / "index").is_file():
            cache = r / "gwd_dem_30s.npy"
            if cache.is_file():
                host = np.load(cache)
            else:
                host = dem_from_wps_tiles(tiles)
                try:
                    np.save(cache, host)
                except OSError:
                    pass
            print(f"{UNIT}: 30-arcsec terrain from {tiles} (GMTED2010)", flush=True)
            return cp.asarray(host)
    raise RuntimeError(f"{UNIT}: no 30-arcsec terrain: set NAMGPU_GWD_DEM (NPS raw file) or put the WPS "
                       f"topo_gmted2010_30s tiles under NAMGPU_GEOG (looked in {[str(r) for r in roots]})")


def run(ctx):
    import cupy as cp

    cfg = ctx.cfg
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    st = ctx.st

    dem = ctx.tables.get("gwd_dem")
    if dem is None:
        dem = ctx.tables["gwd_dem"] = default_dem(ctx)
    if tuple(dem.shape) != (JSRC, ISRC) or dem.dtype != cp.int16:
        raise ValueError(f"gwd_dem must be int16 {(JSRC, ISRC)}, got {dem.dtype} {dem.shape}")
    sw = ctx.tables.get("grid_sw")
    if sw is None:
        raise RuntimeError("init-gwd-orog needs ctx.tables['grid_sw'] from init-grid-geo")
    for name in ("GLAT", "GLON"):
        if tuple(st[name].shape) != (jm, im):
            raise ValueError(f"{name} shape {st[name].shape}, expected {(jm, im)}")

    shape = (jm, im)
    out = {n: st.add(n, shape) for n in OUTPUTS}
    scr_f = {n: st.add(n, shape) for n in FLOAT_SCRATCH}
    scr_i = {n: st.add(n, shape, dtype="int32") for n in INT_SCRATCH}

    for n in FLOAT_SCRATCH:
        scr_f[n].fill(0.0)
    for n in INT_SCRATCH:
        scr_i[n].fill(0)
    out["HZMAX"].fill(-9999.9)          # max_height init (1397)

    sc = _scalars(cfg)
    blk = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    k_origin = kernels.get(UNIT, "gwd_bgrid_origin")
    k_accum = kernels.get(UNIT, "gwd_src_accum")
    k_topo = kernels.get(UNIT, "gwd_model_topo")
    k_var = kernels.get(UNIT, "gwd_src_topo_var")
    k_std = kernels.get(UNIT, "gwd_model_std_conv")
    k_crit = kernels.get(UNIT, "gwd_src_crit")
    k_oaol = kernels.get(UNIT, "gwd_model_oaol")

    with scope:
        origin = cp.zeros(3, dtype=cp.float32)
        ctx.tables["gwd_origin"] = origin
        k_origin((1, 1, 1), (1, 1, 1), (sw, sc["tlm0"], sc["stph0"], sc["ctph0"], origin))
        o = cp.asnumpy(origin)                      # init-time sync: origin check
        if not (-1.0 <= float(o[2]) <= 1.0):
            raise RuntimeError(f"{UNIT}: ll2xy sbd asin argument {float(o[2]):.6f} is outside [-1, 1]; "
                               "sbd is NaN and every source cell would be skipped. Refused.")
        wbd_cfg, sbd_cfg = float(cfg.wbd) * DTR, float(cfg.sbd) * DTR    # cfg origin is in degrees
        if abs(float(o[0]) - wbd_cfg) > 1e-5 or abs(float(o[1]) - sbd_cfg) > 1e-5:
            raise RuntimeError(f"{UNIT}: origin wbd/sbd from the SW H point ({float(o[0]):.7f}, "
                               f"{float(o[1]):.7f}) rad disagree with the configured origin "
                               f"({wbd_cfg:.7f}, {sbd_cfg:.7f}) rad. Refused: the source cells "
                               "would map to the wrong model cells.")

        istart, iend, jstart, jend = find_bounds(st["GLAT"], st["GLON"], im, jm)
        ni, nj = iend - istart + 1, jend - jstart + 1
        if ni <= 0 or nj <= 0:
            raise RuntimeError(f"{UNIT}: empty source window IS/IE/JS/JE {istart} {iend} {jstart} {jend}")
        g_src = kernels.grid_2d(ni, nj, 1)
        g_mdl = kernels.grid_2d(im, jm, 1)

        # pass 1: count, sum_topo, max_height, derivative sums
        k_accum(g_src, blk, (
            dem, origin, np.int32(ISRC), np.int32(JSRC), np.int32(istart), np.int32(jstart),
            np.int32(ni), np.int32(nj), np.int32(im), np.int32(jm),
            sc["stph0"], sc["ctph0"], sc["tlm0"], sc["dph"], sc["dlm"],
            scr_i["GWD_COUNT"], scr_f["GWD_SUMTOPO"], out["HZMAX"],
            scr_f["GWD_SDX2"], scr_f["GWD_SDY2"], scr_f["GWD_SDXY"]))

        n_zero = int(cp.count_nonzero(scr_i["GWD_COUNT"] == 0))   # init-time sync
        if n_zero:
            raise RuntimeError(f"{UNIT}: {n_zero} model cells got no source terrain inside the window "
                               "(Fortran 1552-1555 aborts here). The topo and derivative means would "
                               "divide by zero and write NaN into HSTDV/HZMAX/HANGL. Refused: check the "
                               "GLAT/GLON geometry and the DEM coverage.")

        # model topo, max_height, theta, sigma, gamma
        k_topo(g_mdl, blk, (
            np.int32(im), np.int32(jm), scr_i["GWD_COUNT"], scr_f["GWD_SUMTOPO"],
            scr_f["GWD_SDX2"], scr_f["GWD_SDY2"], scr_f["GWD_SDXY"],
            scr_f["GWD_TOPO"], out["HZMAX"], out["HANGL"], out["HANIS"], out["HSLOP"]))

        # pass 2: squared and fourth-power topo deviations
        k_var(g_src, blk, (
            dem, origin, np.int32(ISRC), np.int32(JSRC), np.int32(istart), np.int32(jstart),
            np.int32(ni), np.int32(nj), np.int32(im), np.int32(jm),
            sc["stph0"], sc["ctph0"], sc["tlm0"], sc["dph"], sc["dlm"],
            scr_f["GWD_TOPO"], scr_f["GWD_SUM2"], scr_f["GWD_SUM4"]))

        # orog_stnd_dev and convexity
        k_std(g_mdl, blk, (
            np.int32(im), np.int32(jm), scr_i["GWD_COUNT"],
            scr_f["GWD_SUM2"], scr_f["GWD_SUM4"], out["HSTDV"], out["HCNVX"]))

        # pass 3: quadrant counts above topo and above crit
        k_crit(g_src, blk, (
            dem, origin, np.int32(ISRC), np.int32(JSRC), np.int32(istart), np.int32(jstart),
            np.int32(ni), np.int32(nj), np.int32(im), np.int32(jm),
            sc["stph0"], sc["ctph0"], sc["tlm0"], sc["dph"], sc["dlm"],
            st["GLAT"], st["GLON"], scr_f["GWD_TOPO"], out["HSTDV"],
            scr_i["GWD_NUL"], scr_i["GWD_NLL"], scr_i["GWD_NUR"], scr_i["GWD_NLR"],
            scr_i["GWD_NUL2"], scr_i["GWD_NLL2"], scr_i["GWD_NUR2"], scr_i["GWD_NLR2"],
            scr_i["GWD_CNUL"], scr_i["GWD_CNLL"], scr_i["GWD_CNUR"], scr_i["GWD_CNLR"]))

        # oa(1..4), ol(1..4)
        k_oaol(g_mdl, blk, (
            np.int32(im), np.int32(jm),
            scr_i["GWD_NUL"], scr_i["GWD_NLL"], scr_i["GWD_NUR"], scr_i["GWD_NLR"],
            scr_i["GWD_NUL2"], scr_i["GWD_NLL2"], scr_i["GWD_NUR2"], scr_i["GWD_NLR2"],
            scr_i["GWD_CNUL"], scr_i["GWD_CNLL"], scr_i["GWD_CNUR"], scr_i["GWD_CNLR"],
            out["HASYW"], out["HASYS"], out["HASYSW"], out["HASYNW"],
            out["HLENW"], out["HLENS"], out["HLENSW"], out["HLENNW"]))

        # declared divergence, counted: ol with a zero denominator is written 0 (Fortran 0/0 = NaN)
        cul, cll = scr_i["GWD_CNUL"], scr_i["GWD_CNLL"]
        cur, clr = scr_i["GWD_CNUR"], scr_i["GWD_CNLR"]
        zero_ol = {
            "HLENW": int(cp.count_nonzero((cul + cll) == 0)),
            "HLENS": int(cp.count_nonzero((cll + clr) == 0)),
            "HLENSW": int(cp.count_nonzero((cul + clr) == 0)),
            "HLENNW": int(cp.count_nonzero((cur + cll) == 0)),
        }
    ctx.tables["gwd_report"] = {"bounds_is_ie_js_je": (istart, iend, jstart, jend),
                                "zero_ol_denominator_cells": zero_ol}
    print(f"{UNIT}: ol cells with zero denominator (written 0, Fortran writes NaN): {zero_ol}")
    return out


def _synthetic_geometry(cfg, im, jm):
    """H-point geographic lat/lon (radians, [JM, IM]) from the configured rotated grid.

    Inverse of ll2xy_bgrid_pt_loc: rotated H point tlat = sbd + j*dphd, tlon = wbd + i*dlmd, rotated
    back with the pole (tph0d, tlm0d). Used only by the self-test, so the geometry is consistent
    with the configured origin.
    """
    tph0 = cfg.tph0d * DTR
    tlm0 = cfg.tlm0d * DTR
    sph0, cph0 = math.sin(tph0), math.cos(tph0)
    j = np.arange(jm, dtype=np.float64)[:, None]
    i = np.arange(im, dtype=np.float64)[None, :]
    tlat = cfg.sbd * DTR + j * cfg.dphd * DTR          # cfg origin and steps are in degrees
    tlon = cfg.wbd * DTR + i * cfg.dlmd * DTR
    xp = np.cos(tlat) * np.cos(tlon)
    yp = -np.cos(tlat) * np.sin(tlon)
    zp = np.sin(tlat)
    a = cph0 * xp - sph0 * zp
    b = yp
    s = sph0 * xp + cph0 * zp
    lat = np.arctan2(s, np.hypot(a, b))
    lon = tlm0 - np.arctan2(b, a)
    return lat, lon


def _selftest():
    """Full-size DEM and model grid; one run; NaN/Inf check and zero-count check."""
    import cupy as cp
    import time

    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=())
    im, jm = int(g.im), int(g.jm)
    lat, lon = _synthetic_geometry(cfg, im, jm)
    shape = (jm, im)
    st.add("GLAT", shape)[...] = cp.asarray(lat, dtype=cp.float32)
    st.add("GLON", shape)[...] = cp.asarray(lon, dtype=cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    rng = cp.random.RandomState(7)
    dem = cp.empty((JSRC, ISRC), dtype=cp.int16)
    for j0 in range(0, JSRC, 2000):                 # chunked: keeps the int64 intermediate small
        blk = rng.randint(-50, 3000, size=(min(2000, JSRC - j0), ISRC))
        dem[j0:j0 + blk.shape[0]] = blk.astype(cp.int16)
    dem[dem < 0] = -9999
    ctx.tables["gwd_dem"] = dem
    ctx.tables["grid_sw"] = cp.asarray([lat[0, 0], lon[0, 0]], dtype=cp.float32)
    t0 = time.perf_counter()
    with ctx.stream:
        out = run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    bad = [n for n, a in out.items() if not bool(cp.isfinite(a).all())]
    print(f"{UNIT}: {dt:.2f} s, NaN/Inf in {bad or 'none'}")
    for n in ("HSTDV", "HCNVX", "HZMAX", "HASYW", "HLENW", "HANIS"):
        a = out[n]
        print(f"  {n}: min {float(a.min()):.4g} mean {float(a.mean()):.4g} max {float(a.max()):.4g}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
