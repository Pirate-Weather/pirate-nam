"""Start the NAM from an isobaric analysis (GFS 0.25 deg, IFS open data, ERA5; HRRR/RRFS rows once their grids are placed).

The NPS replacement for any date (NAM retired 2026-11-03 12Z, CONTRACT "Init after NAM retirement").
Same order as NPS: metgrid-style columns (a surface level, then the isobaric levels ground up) on the
rotated B grid by init-gfs-hinterp (four_pt + average_4pt), then nemsinterp's vertical interpolation
(init-gfs-vinterp: surface pressure on the model terrain, PD/PINT, T, Q, U, V), then the earth-relative
winds are turned to grid-relative at the V points with the same rotation coefficients as
init-gfs-wind-rotate. Everything after decode is on the GPU; the decode is host I/O (gpuwm bridge).

Model-grid result (model_state): PD, PINT, T, Q, CW, U, V, FIS and P3D_OUT (mid-layer pressure) in
the target state. For the init start, stage() then binds the names the rest of the chain reads (TMP,
SPFH, UGRD, VGRD alias T, Q, U, V; PRES = P3D_OUT; PSFC = PSFC_OUT), so init-native-atm,
init-vcoord-recover and init-derive-state run unchanged whatever the source.

Static surface (terrain HGT_M, land mask, vegetation and soil type, roughness, greenness, albedo) is
the NAM grid's own, from the case's native file when it has one, else from static/nam12_static.npz
(build it once from any archived bgrd3d00: python -m namgpu.init.gfs_start --build-static FILE).
Time-varying surface (skin/SST, soil temperature and moisture in the same four Noah layers, snow, sea
ice, canopy water) comes from the source. Declared fills: soil fields where the source is water but
the NAM grid is land are filled from the nearest source land (ring averaging until every point is filled); the
3 m soil temperature TG is the source's 1-2 m layer (GFS has no 3 m level).
"""
from __future__ import annotations

from namgpu import cores as _cores

import contextlib
import os
from pathlib import Path

import numpy as np

from namgpu.init import source_records as srec

UNIT = "init-gfs-start"
from namgpu import paths  # noqa: E402

ROOT = paths.work_root()          # cases/, static/, scratch/ live here (namgpu/paths.py)
STATIC_NPZ = ROOT / "static" / "nam12_static.npz"
MISSING = -9999.0
PL_ORDER = ("HGT", "TMP", "SPFH", "RH", "UGRD", "VGRD", "CLMR", "ICMR", "RWMR", "SNMR")
# The record keys are per source row (source_records.py TABLES); these module names stay as the GFS table.
PL_DCP = srec.GFS["pl"]
# Isobaric source surface records: name -> (d, c, p, level_type, level_value, second_level_value)
SFC = srec.GFS["sfc"]
SOILW_NO_LAND = 0.25   # m3/m3, soil moisture where the source window has no land at all (declared fill)
SOIL_KEYS = ("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4", "SOILW1", "SOILW2", "SOILW3", "SOILW4", "CNWAT")
STATIC_KEYS = ("HGT_SFC", "LAND", "VGTYP", "SOTYP", "SFCR", "VEG", "ALBDO")
STATIC_EXTRA = ("ALBASE", "MXSNAL")   # init-static-build files only (fractions); NAM's bgrd3d00 has neither
# Source window kept around the domain (decode_source). 5 deg of latitude (~550 km) on every side, and the
# same distance in longitude (5 deg / cos(lat) at the domain's most poleward row), so the soil ring fill
# (_ring_fill) reaches the nearest source land for island and coastal domains inside the window. A domain
# within POLE_FULL_DEG of a pole keeps every longitude (its polar cap touches every meridian).
SRC_MARGIN_DEG = 5.0
POLE_FULL_DEG = 85.0
BIG_GRID_POINTS = 2_000_000   # records above this (IFS 0.1 deg: 6.5 M points) are split evenly over the cores
LAT_MARGIN = 2.0      # (init-ifs) deg of source rows south of the domain; the window now uses SRC_MARGIN_DEG


def _scope(ctx):
    return ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()


def _sel_pl(dcp, pa):
    def sel(r):
        return (r["dcp"] == dcp and str(r["level_type"]) == "100"
                and abs(float(r["level_value"]) - pa) < 0.5)
    return sel


# Surface records the vertical interpolation itself reads (columns()); the rest of SFC is the start's
# land-surface state. Boundary times (gfs_bocos) need only these: the GFS forecast files (f006 ...)
# carry no CNWAT, so asking for the full SFC set refused every boundary file (2026-11-04).
SFC_COLUMNS = srec.GFS["sfc_columns"]


def source_window(lat_deg, lon_deg, nx, ny, dlat, dlon, margin=SRC_MARGIN_DEG):
    """Rows and columns of a global source (rows south to north from -90, columns east from 0) that cover
    the points (lat_deg, lon_deg: device or host arrays, degrees) plus margin.

    Returns (j0, j1, cols, lon_periodic): rows j0..j1-1; cols an int array of column indices in window order
    (wrapping modulo nx, so a domain across 0/360 or the dateline is one window), or None for every column
    (lon_periodic True). The window is the complement of the widest gap of empty columns around the
    circle, so it is the shortest arc that holds every point."""
    import cupy as cp
    lat_deg = cp.asarray(lat_deg, dtype=cp.float64).ravel()
    lon_deg = cp.asarray(lon_deg, dtype=cp.float64).ravel()
    la0, la1 = float(lat_deg.min()), float(lat_deg.max())
    j0 = max(0, int(np.floor((la0 - margin + 90.0) / dlat)))
    j1 = min(ny, int(np.ceil((la1 + margin + 90.0) / dlat)) + 1)
    pole = max(abs(la0), abs(la1)) + margin
    if pole >= POLE_FULL_DEG:
        return j0, j1, None, True
    occ = cp.zeros(nx, dtype=cp.int32)
    ic = cp.floor(cp.mod(lon_deg, 360.0) / dlon).astype(cp.int64) % nx
    occ[ic] = 1
    occ = cp.asnumpy(occ).astype(bool)
    pad = int(np.ceil(margin / max(np.cos(np.radians(pole)), 1e-3) / dlon)) + 1
    k = np.flatnonzero(occ)
    keep = np.zeros(nx, bool)
    for d in range(-pad, pad + 2):          # +2: the east neighbour column of the bilinear stencil
        keep[(k + d) % nx] = True
    if keep.all():
        return j0, j1, None, True
    # widest circular run of columns not kept; the window starts right after it
    empty = ~keep
    best_len, best_end, run = 0, 0, 0
    for i in range(2 * nx):
        if empty[i % nx]:
            run += 1
            if run > best_len:
                best_len, best_end = min(run, nx), i % nx
        else:
            run = 0
    start = (best_end + 1) % nx
    cols = (start + np.arange(nx - best_len)) % nx
    return j0, j1, cols, False


def decode_source(ctx, path, tmpdir, pts=None, sfc_keys=None, rec=None, host=None):
    """Decode one isobaric source file (GDT 0, regular lat-lon, global) into init-gfs-decode's GfsDecoded, cut to
    the window around the domain (source_window of pts = [(lat, lon) radians device arrays], default the state's
    H and V points; columns counted from the file's own first longitude), plus the surface fields on the same
    window, through the source row's record table rec (source_records; default GFS). Returns (dec, sfc, grid).
    Was every row north of the domain's south edge and every column (2026-11-04, nests anywhere): a 3 km nest
    decoded ~1.7 GB of pressure-level fields for a 10-degree domain."""
    import time
    import cupy as cp
    from namgpu.init import init_grib2_decode as g2
    from namgpu.init import init_gfs_decode as gd
    from namgpu.init import init_gfs_hinterp as hin
    from namgpu.init.sources import _selsfc

    rec = srec.GFS if rec is None else rec
    pl_dcp, sfc_tab = rec["pl"], rec["sfc"]
    t_dec = time.perf_counter()
    if rec.get("container", "grib2") == "regular-netcdf":
        # the latitude band of the points (default the state's H and V points), every longitude (the fetch
        # writes full-longitude bands: fetch_case.domain_band)
        if pts is None:
            pts = [(ctx.st["GLAT"], ctx.st["GLON"]), (ctx.st["VLAT"], ctx.st["VLON"])]
        la = cp.concatenate([cp.degrees(a.ravel()) for a, _ in pts])
        pl, plev, sfc, grid, what = decode_regular_netcdf(path, rec, float(la.min()), float(la.max()), sfc_keys,
                                                          host=host)
        del la
        dec = _derive_and_decode(ctx, rec, pl, plev, sfc, sfc_keys)
        print(f"{UNIT}: decoded {Path(path).name} ({Path(path).stat().st_size / 1e6:.0f} MB): {what}, "
              f"{len(plev)} levels, {time.perf_counter() - t_dec:.1f} s", flush=True)
        return dec, sfc, grid
    inv = g2.inventory(path)
    r0 = inv[0]
    if int(r0["gdt"]) != 0:
        raise ValueError(f"{UNIT}: {path} is GDT {r0['gdt']}, the isobaric path places only GDT 0")
    nx, ny = int(r0["nx"]), int(r0["ny"])
    # The gather writes rows south to north (row j = -90 + j dlat) for a north-first global file (lat1 = 90):
    # GFS and ECMWF open data both are. Any other first latitude would be misregistered by this row arithmetic.
    if abs(abs(float(r0["lat1"])) - 90.0) > 1e-6:
        raise ValueError(f"{UNIT}: {path} first latitude {r0['lat1']}; the isobaric path places global grids only")
    dlat = 180.0 / (ny - 1)
    dlon = 360.0 / nx
    lev = {}
    for r in inv:
        if str(r["level_type"]) == "100":
            lev.setdefault(r["dcp"], set()).add(float(r["level_value"]))
    need = [pl_dcp[n] for n in rec["pl_need"]]
    plev = sorted(set.intersection(*[lev[d] for d in need]), reverse=True)      # ground up (high p first)
    plev = [p for p in plev if p >= 100.0]                                     # 1 hPa and below
    if pts is None:
        st = ctx.st
        pts = [(st["GLAT"], st["GLON"]), (st["VLAT"], st["VLON"])]
    lon1 = float(r0["lon1"])          # the file's first longitude (GFS 0, ECMWF open data 180 = -180)
    la = cp.concatenate([cp.degrees(a.ravel()) for a, _ in pts])
    lo = cp.concatenate([cp.degrees(b.ravel()) for _, b in pts]) - lon1
    j0, j1, cols, periodic = source_window(la, lo, nx, ny, dlat, dlon)
    del la, lo
    nrow = j1 - j0
    ncol = nx if cols is None else len(cols)
    cols_d = None if cols is None else cp.asarray(cols)

    def cut(a):
        a = a[j0:j1]
        return a if cols_d is None else a[:, cols_d]
    kw = dict(missing_value=MISSING, tmpdir=tmpdir, expect_shape=(nx, ny))
    pl = cp.zeros((len(PL_ORDER), len(plev), nrow, ncol), dtype=cp.float32)
    keys = tuple(sfc_tab) if sfc_keys is None else tuple(sfc_keys)
    optional = set(rec.get("sfc_optional", ()))
    if optional:
        keys = tuple(n for n in keys if n not in optional or any(_selsfc(sfc_tab[n])(r) for r in inv))
    # grib2_dump processes at once, one core each. Big grids: the records split evenly over the cores (one process
    # per variable left 6 cores busy and 13-18 s per IFS 0.1 deg time: 14 CCSDS records of 6.5 M points in a row,
    # 2026-11-04 lane init-ifs). Each record is decoded on its own, so the split does not change a value.
    sels = []
    for v, name in enumerate(PL_ORDER):
        if name not in pl_dcp:
            continue                     # a slot this source does not publish stays 0
        have = lev.get(pl_dcp[name], set())
        sels += [((v, k), _sel_pl(pl_dcp[name], p)) for k, p in enumerate(plev) if p in have]
    sels += [(("sfc", n), _selsfc(sfc_tab[n])) for n in keys]
    if nx * ny <= BIG_GRID_POINTS:
        # small grids (GFS 0.25 deg, 1.04 M points): one process per variable, as before; splitting further was
        # slower (each grib2_dump re-reads the whole file: 22 s against 7 s for the 502 MB GFS f000)
        jobs = {}
        for key, sel in sels:
            jobs.setdefault(key[0], {})[key] = sel
    else:
        ncore = max(1, min(len(sels), _cores.usable()))
        per = -(-len(sels) // ncore)
        jobs = {c: dict(sels[c * per:(c + 1) * per]) for c in range(-(-len(sels) // per))}
    stream = getattr(ctx, "stream", None)

    def one(item):
        # keep only the domain rows of each record as soon as it is placed: a whole 0.1 deg record is 26 MB on the
        # card, and 88 of them held at once OOMed the IFS start beside other work on a shared 32 GB card (2026-11-04)
        key, sel = item
        with (stream if stream is not None else contextlib.nullcontext()):
            got = g2.run(ctx, path, sel, **kw)
            got = {k: cp.ascontiguousarray(cut(a)) for k, a in got.items()}
        return key, got
    from concurrent.futures import ThreadPoolExecutor
    sfc = {}
    # Each grib2_dump is a Rust process whose rayon pool defaults to every core it can see: 76 dumps x 96 threads
    # on a rented 96-core container (pids.max 5888, 2026-11-04) failed to start their pools and killed the IFS boundary
    # build at +33 h. The dumps already run side by side, so each gets its share of the usable cores.
    rayon_prev = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = str(max(1, _cores.usable() // max(1, len(jobs))))
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            for _, got in ex.map(one, jobs.items()):
                for (v, k), a in got.items():
                    if v == "sfc":
                        sfc[k] = cp.ascontiguousarray(a)
                    else:
                        pl[v, k] = a
                del got
    finally:
        if rayon_prev is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = rayon_prev
    # Missing (bitmap) points: the sanitiser fills from present neighbours; MISSING marks them.
    pl[pl == MISSING] = cp.float32(np.nan)                          # in place, no second copy of the set
    if rec["pl_derive"]:
        srec.derive_pl(rec, pl, plev, PL_ORDER)
    if rec["sfc_derive"]:
        srec.derive_sfc(rec, sfc, only=None if sfc_keys is None else rec["derive_columns"])
    s2 = cp.stack([sfc["PRES"], sfc["HGT"]])
    dec = gd.decode(pl, np.asarray(plev, np.float64), s2, stream=getattr(ctx, "stream", None))
    if periodic:
        grid = hin.Grid(lat0=-90.0 + j0 * dlat, lon0=lon1, dlat=dlat, dlon=dlon, lon_periodic=True)
    else:
        grid = hin.Grid(lat0=-90.0 + j0 * dlat, lon0=lon1 + float(cols[0]) * dlon, dlat=dlat, dlon=dlon,
                        lon_periodic=False, lon_window=True)
    if getattr(ctx, "tables", None) is not None:
        ctx.tables["gfs_window"] = {"rows": [j0, j1], "lat": [-90.0 + j0 * dlat, -90.0 + (j1 - 1) * dlat],
                                    "lon0": grid.lon0, "ncol": ncol, "periodic": bool(periodic)}
    nrec = sum(len(x) for x in jobs.values())
    print(f"{UNIT}: decoded {Path(path).name} ({Path(path).stat().st_size / 1e6:.0f} MB): {nrec} records, "
          f"{len(jobs)} grib2_dump processes at once (one core each), rows {j0}..{j1 - 1} of {ny}, "
          f"{ncol} of {nx} columns, {len(plev)} levels, {time.perf_counter() - t_dec:.1f} s", flush=True)
    return dec, sfc, grid


def _derive_and_decode(ctx, rec, pl, plev, sfc, sfc_keys):
    """The record table's derived slots and surface fields, then init-gfs-decode (every container alike)."""
    import cupy as cp
    from namgpu.init import init_gfs_decode as gd
    # Missing (bitmap) points: the sanitiser fills from present neighbours; MISSING marks them.
    pl = cp.where(pl == MISSING, cp.float32(np.nan), pl)
    if rec["pl_derive"]:
        srec.derive_pl(rec, pl, plev, PL_ORDER)
    if rec["sfc_derive"]:
        srec.derive_sfc(rec, sfc, only=None if sfc_keys is None else rec["derive_columns"])
    s2 = cp.stack([sfc["PRES"], sfc["HGT"]])
    return gd.decode(pl, np.asarray(plev, np.float64), s2, stream=getattr(ctx, "stream", None))


def read_regular_netcdf(path):
    """Host half of decode_regular_netcdf: gpuwm's reader on one file (one Rust bridge process, one core; about
    20 s for an ERA5 hour of the NAM band). gfs_bocos runs several at once ahead of the GPU work."""
    try:
        from gpuwm.ingest.regular_netcdf import decode_regular_netcdf_files
    except ImportError as e:            # gpuwm is a package dependency; a bare checkout may not have it
        raise ImportError(f"{UNIT}: a regular-netcdf source (era5) is read by gpuwm's reader: pip install "
                          "'gpuwm>=2.9.11'") from e
    return decode_regular_netcdf_files((Path(path),))


def decode_regular_netcdf(path, rec, lat_min_deg, lat_max_deg=None, sfc_keys=None, host=None):
    """One regular lat-lon NetCDF in gpuwm's regular-forcing schema (container "regular-netcdf": the files
    gpuwm.era5_arco writes), read by gpuwm's own reader (gpuwm.ingest.regular_netcdf, Rust NetCDF bridge), into the
    arrays decode_source makes from GRIB2: pl [len(PL_ORDER), nlev, nrow, nx] float32 on the device, rows south to
    north, levels ground up (Pa); sfc {name: [nrow, nx]} with missing points = MISSING; the init-gfs-hinterp Grid.
    Rows are the domain band lat_min-LAT_MARGIN .. lat_max+LAT_MARGIN (lat_max None: to the file's north edge).

    Refusals (each names the breakage it prevents):
      - more than one valid time in the file: the start and every boundary time are one file each; reading the
        first of several would put the wrong hour into the run silently.
      - an irregular grid, or a longitude span other than 360 deg: init-gfs-hinterp places rows and columns by
        lat0 + j dlat, lon0 + i dlon and wraps longitude mod 360, so either would misplace points.
      - a latitude band that does not cover the domain's band: the interpolation would read rows the file does not
        have (a file fetched for another domain)."""
    import cupy as cp
    from namgpu.init import init_gfs_hinterp as hin
    res = read_regular_netcdf(path) if host is None else host      # host: read_regular_netcdf(path), read ahead
    if len(res.snapshots) != 1:
        raise ValueError(f"{UNIT}: {path} holds {len(res.snapshots)} valid times; one file per start or boundary "
                         "time is read")
    snap = res.snapshots[0]
    lat = np.asarray(snap.latitude, np.float64)
    lon = np.asarray(snap.longitude, np.float64)
    ny, nx = lat.size, lon.size
    dlat, dlon = abs(float(lat[1] - lat[0])), float(lon[1] - lon[0])
    if (dlon <= 0.0 or not np.allclose(np.abs(np.diff(lat)), dlat, atol=1e-6)
            or not np.allclose(np.diff(lon), dlon, atol=1e-6)):
        raise ValueError(f"{UNIT}: {path} is not a regular lat-lon grid with increasing longitudes")
    if abs(nx * dlon - 360.0) > 1e-6:
        raise ValueError(f"{UNIT}: {path} spans {nx * dlon:g} deg of longitude; the isobaric path wraps longitude "
                         "mod 360 and places full-longitude bands only")
    north_first = lat[0] > lat[-1]
    lat_s = lat[::-1] if north_first else lat
    lo = max(-90.0, lat_min_deg - LAT_MARGIN)
    hi = 90.0 if lat_max_deg is None else min(90.0, lat_max_deg + LAT_MARGIN)
    if lat_s[0] > lo + 1e-6 or lat_s[-1] < hi - 1e-6:
        raise ValueError(f"{UNIT}: {path} covers latitudes {lat_s[0]:g}..{lat_s[-1]:g}; this domain needs "
                         f"{lo:g}..{hi:g} (fetch the source again for this domain)")
    j0 = max(0, int(np.floor((lo - lat_s[0]) / dlat)))
    j1 = min(ny, int(np.ceil((hi - lat_s[0]) / dlat)) + 1)
    missing = getattr(res, "bitmap_missing", {}) or {}

    def rows(name):
        """The field south to north, band rows only, float32, NaN where the file marks it missing."""
        a = np.asarray(snap.fields[name], np.float64)
        m = missing.get((snap.valid_time, name))
        if m is not None and np.any(m):
            a = np.where(m, np.nan, a)
        if north_first:
            a = a[..., ::-1, :]
        return np.ascontiguousarray(a[..., j0:j1, :], dtype=np.float32)

    lev = np.asarray(snap.levels_hpa, np.float64) * 100.0           # hPa -> Pa
    order = [int(k) for k in np.argsort(-lev, kind="stable") if lev[k] >= 100.0]   # ground up, 1 hPa and below
    plev = [float(lev[k]) for k in order]
    pl = cp.zeros((len(PL_ORDER), len(plev), j1 - j0, nx), dtype=cp.float32)
    for v, slot in enumerate(PL_ORDER):
        if slot in rec["pl"]:            # a slot this source does not publish stays 0
            pl[v] = cp.asarray(rows(rec["pl"][slot])[order])
    sfc = {}
    for n in (tuple(rec["sfc"]) if sfc_keys is None else tuple(sfc_keys)):
        a = rows(rec["sfc"][n])
        a[~np.isfinite(a)] = MISSING
        sfc[n] = cp.asarray(a)
    grid = hin.Grid(lat0=float(lat_s[j0]), lon0=float(lon[0]), dlat=dlat, dlon=dlon, lon_periodic=True)
    what = (f"{snap.valid_time:%Y-%m-%d %HZ}, gpuwm regular-netcdf reader, {len(snap.fields)} fields, "
            f"rows {j0}..{j1 - 1} of {ny} (south first)")
    return pl, plev, sfc, grid, what


def _shift(a, dj, di, lon_periodic, fill):
    """a moved by (dj, di) rows/columns: wraps in longitude only for a periodic (global) source; rows (and the
    columns of a window) do not wrap, the vacated edge is fill."""
    import cupy as cp
    if lon_periodic:
        out = cp.roll(a, di, 1)
    elif di == 0:
        out = a
    else:
        out = cp.full_like(a, fill)
        if di > 0:
            out[:, di:] = a[:, :-di]
        else:
            out[:, :di] = a[:, -di:]
    if dj == 0:
        return out
    res = cp.full_like(a, fill)
    if dj > 0:
        res[dj:] = out[:-dj]
    else:
        res[:dj] = out[-dj:]
    return res


def _ring_fill(a, missing, rings=2000, lon_periodic=True):
    """Fill missing points from the mean of present 8-neighbours, ring by ring (device), until none is left.
    Was 12 rings (~3 deg): a NAM land cell whose nearest source land is further (Jan Mayen, 71.07N 8.15W, ~18 rings
    from Iceland/Greenland on GFS 0.25 deg) kept STC = TG = -9999, Noah drove TSKIN to -639 K and QGH went inf at
    step 10 of the GFS start (run lead 1140Z, scratch/runlead2/probe_qgh.py). The loop stops once nothing is missing
    or a ring fills nothing (nothing present is connected to what is left). Neighbours wrap in longitude only for a
    global source, never across the pole rows or the edges of a longitude window (2026-11-04, nests anywhere: a
    window's east edge is not next to its west edge). Returns (filled, still-missing mask)."""
    import cupy as cp
    a = a.copy()
    m = missing.copy()
    for _ in range(rings):
        if not bool(m.any()):
            break
        v = cp.where(m, 0.0, a)
        w = (~m).astype(cp.float32)
        s = cp.zeros_like(a)
        n = cp.zeros_like(a)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                s += _shift(v, dj, di, lon_periodic, 0.0)
                n += _shift(w, dj, di, lon_periodic, 0.0)
        newv = m & (n > 0)
        if not bool(newv.any()):
            break
        a = cp.where(newv, s / cp.maximum(n, 1.0), a)
        m = m & ~newv
    return a, m


def columns(ctx, dec, sfc, grid):
    """Metgrid-style model-grid columns for init-gfs-vinterp: level 0 = the source surface (PRES =
    surface pressure, GHT = source terrain, TT = 2 m T, RH = 2 m RH, UU/VV = 10 m wind), then the
    isobaric levels ground up. H fields at H points, winds at V points (earth-relative)."""
    import cupy as cp
    from namgpu.init import init_gfs_hinterp as hin

    st = ctx.st
    H = (st["GLAT"], st["GLON"])
    V = (st["VLAT"], st["VLON"])
    s = getattr(ctx, "stream", None)
    nl = int(dec.pl.shape[1])
    jm, im = (int(x) for x in st["GLAT"].shape)

    def f3(name, pts):
        k = PL_ORDER.index(name)
        out, _ = hin.interp_full(dec.pl[k], dec.pl_mask[k], pts[0], pts[1], grid=grid, stream=s)
        return out

    def f2(a, pts, miss=None):
        a = cp.ascontiguousarray(a, dtype=cp.float32)[None]
        m = cp.ones(a.shape, dtype=cp.uint8) if miss is None else (~miss).astype(cp.uint8)[None]
        out, _ = hin.interp_full(a, m, pts[0], pts[1], grid=grid, stream=s)
        return out[0]

    def stack(sfc2d, iso):
        out = cp.empty((nl + 1, jm, im), dtype=cp.float32)
        out[0] = sfc2d
        out[1:] = iso
        return out

    zero = cp.zeros((jm, im), dtype=cp.float32)
    psfc = f2(sfc["PRES"], H)
    pres = cp.empty((nl + 1, jm, im), dtype=cp.float32)
    pres[0] = psfc
    pres[1:] = dec.plev.astype(cp.float32)[:, None, None]
    gi = {
        "PRES_IN": pres,
        "GHT_IN": stack(f2(sfc["HGT"], H), f3("HGT", H)),
        "TT_IN": stack(f2(sfc["T2"], H), f3("TMP", H)),
        "RH_IN": stack(f2(sfc["RH2"], H), f3("RH", H)),
        "UU_IN": stack(f2(sfc["U10"], V), f3("UGRD", V)),
        "VV_IN": stack(f2(sfc["V10"], V), f3("VGRD", V)),
        "SNMR_IN": stack(zero, f3("SNMR", H)),
        "CLWMR_IN": stack(zero, f3("CLMR", H)),
        "RWMR_IN": stack(zero, f3("RWMR", H)),
        "CICE_IN": stack(zero, f3("ICMR", H)),
        "FRIMEF_IN": cp.zeros((nl + 1, jm, im), dtype=cp.float32),
    }
    return gi, f2(sfc["HGT"], H)


def rotate_winds(ctx, u, v):
    """Earth-relative -> grid-relative at V points, in place (init-gfs-wind-rotate's coefficients)."""
    from namgpu.init import init_gfs_wind_rotate as wr
    ce_e, ce_n, cn_e, cn_n = wr._coefficients(ctx)
    with _scope(ctx):
        ug = u * ce_e[None] + v * ce_n[None]
        vg = u * cn_e[None] + v * cn_n[None]
        u[...] = ug
        v[...] = vg


def ensure_vcoord(ctx):
    """Put the NAM hybrid coordinate on the grid BEFORE the vertical interpolation, as NPS does (vcgenerator
    runs first, then nemsinterp interpolates onto it). The chain's init-vcoord-recover stage runs after
    init-grib2-decode, so without this the GFS start interpolated onto whatever SG the grid held at that
    point: the lowest ~20 layers came out as one constant column (2026-11-04, runs/init_gfs: theta 315 K
    and q 16.6 g/kg from 1007 to 864 hPa over the Gulf). Same header init-vcoord-recover uses."""
    from namgpu.init import vcoord_nps
    hdr = vcoord_nps.header_from_config(ctx.cfg)
    ctx.grid.set_vcoord(hdr["PT"], hdr["PDTOP"], hdr["SG1"], hdr["SG2"])
    for k in ("vi_sgml1", "vi_sgml2", "vi_sg1", "vi_sg2"):
        ctx.tables.pop(k, None)


def model_state(ctx, path, tmpdir, sfc_keys=None, rec=None, decoded=None, host=None):
    """Decode, interpolate and vertically interpolate one isobaric file onto the model grid; the
    result is in ctx.st (PD, PINT, T, Q, CW, U, V grid-relative, FIS, PSFC_OUT, P3D_OUT).
    ctx.st must carry HGT_M (model terrain) and GLAT/GLON/VLAT/VLON. decoded: decode_source's result for
    this file when the caller decoded it ahead (init-gfs-bocos prefetch); else it is decoded here."""
    import cupy as cp
    from namgpu.init import init_gfs_vinterp as vin

    st = ctx.st
    ensure_vcoord(ctx)
    dec, sfc, grid = (decoded if decoded is not None
                      else decode_source(ctx, path, tmpdir, sfc_keys=sfc_keys, rec=rec, host=host))
    gi, soilhgt = columns(ctx, dec, sfc, grid)
    del dec
    st.add("SOILHGT", soilhgt.shape)[...] = soilhgt
    ctx.tables["gfs_hinterp"] = gi
    first = ctx.first_step
    ctx.first_step = True                 # full-domain interpolation (no interior skip)
    try:
        vin.run(ctx)
    finally:
        ctx.first_step = first
        ctx.tables.pop("gfs_hinterp", None)
    rotate_winds(ctx, st["U"], st["V"])
    return sfc, grid


def check_valid(ctx, name, row):
    """Refuse a cycle before the row's first published cycle (valid_from): the files do not exist, or (IFS 0.1 deg
    2026-10-20 06Z) exist without the soil layers the start needs."""
    if ctx.start is None:
        return
    t = ctx.start.strftime("%Y-%m-%dT%H")
    if row.get("valid_from") and t < row["valid_from"]:
        alt = f"; use init_source {row['before_use']}" if row.get("before_use") else ""
        raise ValueError(f"{UNIT}: source {name!r} starts at {row['valid_from']}Z, not {t}Z{alt}")


def decode_donor(ctx, spec, tmpdir):
    """One donor field (source_records donors) from another source's file, on that file's own global grid."""
    from namgpu.init import init_grib2_decode as g2
    from namgpu.init import init_gfs_hinterp as hin
    from namgpu.init.sources import _selsfc
    cyc = int(ctx.start.hour) if ctx.start is not None else 0
    path = Path(ctx.case_dir) / spec["file"].format(cyc=cyc)
    if not path.is_file():
        raise FileNotFoundError(f"{UNIT}: donor file {path} missing (run fetch_case --source)")
    sel = _selsfc(spec["spec"])
    inv = [r for r in g2.inventory(path) if sel(r)]
    if len(inv) != 1 or int(inv[0]["gdt"]) != 0 or abs(abs(float(inv[0]["lat1"])) - 90.0) > 1e-6:
        raise ValueError(f"{UNIT}: donor {path}: need exactly one global GDT 0 record for {spec['spec']}")
    nx, ny = int(inv[0]["nx"]), int(inv[0]["ny"])
    got = g2.run(ctx, path, {"X": sel}, missing_value=MISSING, tmpdir=tmpdir, expect_shape=(nx, ny))
    return got["X"], hin.Grid(lat0=-90.0, lon0=float(inv[0]["lon1"]), dlat=180.0 / (ny - 1), dlon=360.0 / nx,
                              lon_periodic=True)


def load_static(ctx):
    """NAM-grid static surface: the case's native file if present, else static/nam12_static.npz."""
    import cupy as cp
    from namgpu import config as configmod
    from namgpu.init import sources as srcmod

    own = configmod.static_npz(ctx.cfg) if getattr(ctx, "cfg", None) is not None else None
    if own is not None:
        # A domain other than the NAM grid: its init-static-build file (the case's native file is on the
        # NAM 12 km grid and does not apply).
        if not own.is_file():
            raise FileNotFoundError(f"{UNIT}: configfile static_npz {own} missing (python -m namgpu.init.static_build)")
        z = np.load(own)
        bad = [k for k in STATIC_KEYS if z[k].shape != (int(ctx.grid.jm), int(ctx.grid.im))]
        if bad:
            raise ValueError(f"{UNIT}: {own} fields {bad} are not on the {ctx.grid.im}x{ctx.grid.jm} grid")
        # The builder also carries the snow-free base albedo and the maximum snow albedo from data
        # (albedo_modis, maxsnowalb_modis); init-static-missing puts them in the state before its table fill.
        if getattr(ctx, "tables", None) is not None:
            ctx.tables["static_extra"] = {k: cp.asarray(z[k]) for k in STATIC_EXTRA if k in z.files}
        return {k: cp.asarray(z[k]) for k in STATIC_KEYS}, str(own)
    path = Path(ctx.case_dir) / srcmod.SOURCES["native"]["file"]
    if path.is_file():
        tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
        got, _ = srcmod.decode_native(ctx, path, [], list(STATIC_KEYS), int(ctx.grid.lm), tmpdir,
                                      lambda n, a: None, top_first=True)
        return {k: got[k] for k in STATIC_KEYS}, str(path)
    if not STATIC_NPZ.is_file():
        raise FileNotFoundError(f"{UNIT}: no NAM static surface: neither {path} nor {STATIC_NPZ}. Build it once "
                                "with: python -m namgpu.init.gfs_start --build-static <any archived bgrd3d00>")
    z = np.load(STATIC_NPZ)
    return {k: cp.asarray(z[k]) for k in STATIC_KEYS}, str(STATIC_NPZ)


def stage(ctx, name, row):
    """Chain stage for an isobaric init source (sources.stage -> here)."""
    import cupy as cp
    from namgpu.init import init_grid_geo, sources as srcmod

    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    rec = srec.table(row)
    check_valid(ctx, name, row)
    path = srcmod.source_path(ctx, row)
    if not path.is_file():
        url = row.get("url", "").format(ymd=f"{ctx.start:%Y%m%d}", cyc=ctx.start.hour)
        raise FileNotFoundError(f"{UNIT}: {path} missing" + (f" (fetch {url})" if url else ""))
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    Path(tmpdir).mkdir(exist_ok=True)
    init_grid_geo.run(ctx)                 # the interpolation needs GLAT/GLON/VLAT/VLON first
    static, static_from = load_static(ctx)
    st.add("HGT_M", (jm, im))[...] = static["HGT_SFC"]
    sfc, grid = model_state(ctx, path, tmpdir, rec=rec)
    donor_grid = {}
    for k, spec in rec["donors"].items():        # fields the source does not publish (source_records donors)
        sfc[k], donor_grid[k] = decode_donor(ctx, spec, tmpdir)
    fallback_used = {}
    for k, spec in rec.get("donor_fallback", {}).items():   # optional records this file does not carry
        if k not in sfc:
            sfc[k], donor_grid[k] = decode_donor(ctx, spec, tmpdir)
            fallback_used[k] = spec["file"].format(cyc=int(ctx.start.hour) if ctx.start is not None else 0)
    if fallback_used:
        print(f"{UNIT}: {name} file carries no {sorted(fallback_used)} source records; taken from "
              f"{sorted(set(fallback_used.values()))}", flush=True)

    # Bind the names the chain reads after this stage (see module docstring).
    for a, b in (("TMP", "T"), ("SPFH", "Q"), ("UGRD", "U"), ("VGRD", "V"), ("PRES", "P3D_OUT")):
        st.fields[a] = st[b]
    zero = st.add("ZERO3", (lm, jm, im))
    for n in srcmod.SPECIES:
        st.fields[n] = zero
    st.add("PSFC", (jm, im))[...] = st["PSFC_OUT"]
    ctx.tables.pop("native_warm", None)

    # Surface: source values on the NAM grid, static fields from the NAM grid.
    land_src = sfc["LANDSRC"] > 0.5
    soil = {}
    left = {}
    for k in SOIL_KEYS:
        a = sfc[k]
        miss = (a == MISSING) | ~land_src
        a, m = _ring_fill(a, miss, lon_periodic=grid.lon_periodic)
        soil[k] = a
        left[k] = m
    # Declared fill: a source window with no land at all (a small island the source grid does not resolve,
    # far from any other land) leaves the soil unfilled. Soil temperature then takes the source skin
    # temperature (the sea surface there), moisture SOILW_NO_LAND, canopy water 0, and the count is printed.
    nleft = int(left["TSOIL1"].sum())
    if nleft:
        for k in SOIL_KEYS:
            fill = sfc["TSK"] if k.startswith("TSOIL") else (SOILW_NO_LAND if k.startswith("SOILW") else 0.0)
            soil[k] = cp.where(left[k], fill, soil[k])
        print(f"{UNIT}: no source land in the window for {nleft} source points: soil T = skin T, "
              f"SOILW = {SOILW_NO_LAND}, CNWAT = 0 (declared fill)", flush=True)
    from namgpu.init import init_gfs_hinterp as hin

    def h2(a, src_grid=None):
        a = cp.ascontiguousarray(a, dtype=cp.float32)[None]
        out, _ = hin.interp_full(a, cp.ones(a.shape, dtype=cp.uint8), st["GLAT"], st["GLON"],
                                 grid=grid if src_grid is None else src_grid, stream=getattr(ctx, "stream", None))
        return out[0]

    def nonneg(a):
        return cp.maximum(cp.where(a == MISSING, 0.0, a), 0.0)

    raw = {
        "TSOIL": cp.ascontiguousarray(cp.stack([h2(soil[f"TSOIL{k}"]) for k in range(1, 5)])),
        "SOILW": cp.ascontiguousarray(cp.stack([cp.clip(h2(soil[f"SOILW{k}"]), 0.02, 0.6)
                                                for k in range(1, 5)])),
        "TMP_SFC": h2(sfc["TSK"]),
        "ICEC": cp.clip(h2(nonneg(sfc["ICEC"]), donor_grid.get("ICEC")), 0.0, 1.0),
        "TSOIL_3M": h2(soil["TSOIL4"]),
        "SNOD": h2(nonneg(sfc["SNOD"]), donor_grid.get("SNOD")),
        "WEASD": h2(nonneg(sfc["WEASD"]), donor_grid.get("WEASD")),
        "CNWAT": h2(nonneg(soil["CNWAT"])),
        "LAND": static["LAND"], "ALBDO": static["ALBDO"], "VEG": static["VEG"], "SFCR": static["SFCR"],
        "VGTYP": cp.rint(static["VGTYP"]).astype(cp.int32),
        "SOTYP": cp.rint(static["SOTYP"]).astype(cp.int32),
    }
    raw["WTMP"] = raw["TMP_SFC"]

    # Shelter / 10 m state (NMMB reads it from its input file; sources.shelter_state). Source 2 m T moved
    # from the source terrain to the NAM terrain with the standard lapse rate 6.5 K/km; source 2 m q kept;
    # 10 m winds interpolated to H points and turned from earth-relative to grid-relative with the H-point
    # vectors of io-wind-rotate (inverse of its rotation: the vectors are orthonormal).
    from namgpu.io import io_wind_rotate as iwr
    t2 = h2(sfc["T2"]) - cp.float32(0.0065) * (st["HGT_M"] - st["SOILHGT"])
    q2 = h2(nonneg(sfc["Q2"]))
    ue, vn = h2(sfc["U10"]), h2(sfc["V10"])
    ce_e, ce_n, cn_e, cn_n = iwr.h_coefficients(ctx)
    srcmod.shelter_state(ctx, t2, q2, ue * ce_e + vn * ce_n, ue * cn_e + vn * cn_n)
    ctx.tables["raw_sfc"] = raw
    ctx.tables["soil_hydraulics"] = {k: cp.asarray(v) for k, v in srcmod.soil_hydraulics().items()}
    ctx.tables["grid_source"] = {"source": name, "file": str(path), "static": static_from,
                                 "levels": int(ctx.st["P3D_OUT"].shape[0]),
                                 "attribution": row.get("attribution"), "licence": row.get("licence"),
                                 "donors": {k: v["file"] for k, v in rec["donors"].items()},
                                 "donors_used_for_missing_records": fallback_used}
    print(f"{UNIT}: {name} start from {path.name}: isobaric -> NMMB hybrid on the GPU; static surface "
          f"from {Path(static_from).name}", flush=True)
    return ctx.tables["grid_source"]


def build_static(bgrd, out=STATIC_NPZ):
    """Write the NAM-grid static surface (STATIC_KEYS) from an archived bgrd3d00 file."""
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import config, grid as gridmod
    from namgpu.init import sources as srcmod

    cfg = config.load()
    g = gridmod.build(cfg)
    ctx = SimpleNamespace(grid=g, stream=None)
    got, _ = srcmod.decode_native(ctx, Path(bgrd), [], list(STATIC_KEYS), int(g.lm), str(ROOT / "scratch"),
                                  lambda n, a: None, top_first=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: cp.asnumpy(got[k]) for k in STATIC_KEYS})
    print(f"wrote {out}")


def main(argv=None) -> int:
    import argparse
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Build the NAM 12 km grid's static surface fields (terrain, land mask, soil and "
                    "vegetation type, roughness, greenness, albedo) from one archived NAM analysis. A GFS "
                    "start needs them: build them once and every later GFS start reuses the file. Uses "
                    "the GPU for a few seconds.",
        epilog="example: woof-nam static cases/2025-03-14_12z/raw/nam/bgrd3d00.grib2\n"
               "(that file comes from: woof-nam fetch --cycle 2025031412 --hours 24 "
               "--case-dir cases/2025-03-14_12z)",
        formatter_class=HelpFormatter)
    ap.add_argument("bgrd", nargs="?", metavar="BGRD3D00",
                    help="an archived NAM native analysis, <case-dir>/raw/nam/bgrd3d00.grib2")
    ap.add_argument("--build-static", dest="bgrd_opt", metavar="BGRD3D00",
                    help="same as BGRD3D00 (older form)")
    ap.add_argument("--out", type=Path, default=STATIC_NPZ,
                    help=f"where to write it (default static/nam12_static.npz in the work root: {STATIC_NPZ})")
    a = ap.parse_args(argv)
    bgrd = a.bgrd or a.bgrd_opt
    if not bgrd:
        ap.error("give the archived NAM analysis file (BGRD3D00)")
    build_static(bgrd, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
