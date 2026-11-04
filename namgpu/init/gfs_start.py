"""Start the NAM from an isobaric analysis (GFS 0.25 deg; HRRR/RRFS rows once their grids are placed).

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

import contextlib
from pathlib import Path

import numpy as np

UNIT = "init-gfs-start"
ROOT = Path(__file__).resolve().parents[2]
STATIC_NPZ = ROOT / "static" / "nam12_static.npz"
MISSING = -9999.0
PL_ORDER = ("HGT", "TMP", "SPFH", "RH", "UGRD", "VGRD", "CLMR", "ICMR", "RWMR", "SNMR")
PL_DCP = {"HGT": (0, 3, 5), "TMP": (0, 0, 0), "SPFH": (0, 1, 0), "RH": (0, 1, 1), "UGRD": (0, 2, 2),
          "VGRD": (0, 2, 3), "CLMR": (0, 1, 22), "ICMR": (0, 1, 23), "RWMR": (0, 1, 24), "SNMR": (0, 1, 25)}
# Isobaric source surface records: name -> (d, c, p, level_type, level_value, second_level_value)
SFC = {
    "PRES": (0, 3, 0, "1", "0", None), "HGT": (0, 3, 5, "1", "0", None),
    "TSK": (0, 0, 0, "1", "0", None),
    "T2": (0, 0, 0, "103", "2", None), "RH2": (0, 1, 1, "103", "2", None),
    "Q2": (0, 1, 0, "103", "2", None),
    "U10": (0, 2, 2, "103", "10", None), "V10": (0, 2, 3, "103", "10", None),
    "TSOIL1": (2, 0, 2, "106", "0", "0.1"), "TSOIL2": (2, 0, 2, "106", "0.1", "0.4"),
    "TSOIL3": (2, 0, 2, "106", "0.4", "1"), "TSOIL4": (2, 0, 2, "106", "1", "2"),
    "SOILW1": (2, 0, 192, "106", "0", "0.1"), "SOILW2": (2, 0, 192, "106", "0.1", "0.4"),
    "SOILW3": (2, 0, 192, "106", "0.4", "1"), "SOILW4": (2, 0, 192, "106", "1", "2"),
    "CNWAT": (2, 0, 196, "1", "0", None), "WEASD": (0, 1, 13, "1", "0", None),
    "SNOD": (0, 1, 11, "1", "0", None), "ICEC": (10, 2, 0, "1", "0", None),
    "LANDSRC": (2, 0, 0, "1", "0", None),
}
SOIL_KEYS = ("TSOIL1", "TSOIL2", "TSOIL3", "TSOIL4", "SOILW1", "SOILW2", "SOILW3", "SOILW4", "CNWAT")
STATIC_KEYS = ("HGT_SFC", "LAND", "VGTYP", "SOTYP", "SFCR", "VEG", "ALBDO")
LAT_MARGIN = 2.0      # deg of source rows kept south of the domain's southernmost point


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
SFC_COLUMNS = ("PRES", "HGT", "T2", "RH2", "U10", "V10")


def decode_source(ctx, path, tmpdir, lat_min_deg, sfc_keys=None):
    """Decode one isobaric source file (GDT 0, regular lat-lon) into init-gfs-decode's GfsDecoded
    (cropped to rows north of lat_min_deg) plus the surface fields. Returns (dec, sfc, grid)."""
    import cupy as cp
    from namgpu.init import init_grib2_decode as g2
    from namgpu.init import init_gfs_decode as gd
    from namgpu.init import init_gfs_hinterp as hin
    from namgpu.init.sources import _selsfc

    inv = g2.inventory(path)
    r0 = inv[0]
    if int(r0["gdt"]) != 0:
        raise ValueError(f"{UNIT}: {path} is GDT {r0['gdt']}, the isobaric path places only GDT 0")
    nx, ny = int(r0["nx"]), int(r0["ny"])
    dlat = 180.0 / (ny - 1)
    dlon = 360.0 / nx
    lev = {}
    for r in inv:
        if str(r["level_type"]) == "100":
            lev.setdefault(r["dcp"], set()).add(float(r["level_value"]))
    need = [PL_DCP[n] for n in ("HGT", "TMP", "RH", "UGRD", "VGRD")]
    plev = sorted(set.intersection(*[lev[d] for d in need]), reverse=True)      # ground up (high p first)
    plev = [p for p in plev if p >= 100.0]                                     # 1 hPa and below
    j0 = max(0, int(np.floor((lat_min_deg - LAT_MARGIN + 90.0) / dlat)))
    nrow = ny - j0
    kw = dict(missing_value=MISSING, tmpdir=tmpdir, expect_shape=(nx, ny))
    pl = cp.zeros((len(PL_ORDER), len(plev), nrow, nx), dtype=cp.float32)
    keys = tuple(SFC) if sfc_keys is None else tuple(sfc_keys)
    # One grib2_dump process per variable, all at once (each dump is one host process on one core; run
    # in series they were most of the 16 s per boundary time). The device placement stays per variable.
    jobs = {}
    for v, name in enumerate(PL_ORDER):
        have = lev.get(PL_DCP[name], set())
        sel = {f"{k}": _sel_pl(PL_DCP[name], p) for k, p in enumerate(plev) if p in have}
        if sel:
            jobs[v] = sel
    jobs["sfc"] = {n: _selsfc(SFC[n]) for n in keys}
    stream = getattr(ctx, "stream", None)

    def one(item):
        key, sel = item
        if stream is not None:
            with stream:
                return key, g2.run(ctx, path, sel, **kw)
        return key, g2.run(ctx, path, sel, **kw)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = dict(ex.map(one, jobs.items()))
    for v, got in results.items():
        if v == "sfc":
            continue
        for k, a in got.items():
            pl[v, int(k)] = a[j0:]
    sfc_raw = results.pop("sfc")
    del results
    sfc = {n: cp.ascontiguousarray(a[j0:]) for n, a in sfc_raw.items()}
    del sfc_raw
    # Missing (bitmap) points: the sanitiser fills from present neighbours; MISSING marks them.
    pl = cp.where(pl == MISSING, cp.float32(np.nan), pl)
    s2 = cp.stack([sfc["PRES"], sfc["HGT"]])
    dec = gd.decode(pl, np.asarray(plev, np.float64), s2, stream=getattr(ctx, "stream", None))
    grid = hin.Grid(lat0=-90.0 + j0 * dlat, lon0=0.0, dlat=dlat, dlon=dlon, lon_periodic=True)
    return dec, sfc, grid


def _ring_fill(a, missing, rings=2000):
    """Fill missing points from the mean of present 8-neighbours, ring by ring (device), until none is left.
    Was 12 rings (~3 deg): a NAM land cell whose nearest source land is further (Jan Mayen, 71.07N 8.15W, ~18 rings
    from Iceland/Greenland on GFS 0.25 deg) kept STC = TG = -9999, Noah drove TSKIN to -639 K and QGH went inf at
    step 10 of the GFS start (run lead 1140Z, scratch/runlead2/probe_qgh.py). The loop stops once nothing is missing."""
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
                s += cp.roll(cp.roll(v, dj, 0), di, 1)
                n += cp.roll(cp.roll(w, dj, 0), di, 1)
        newv = m & (n > 0)
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


def model_state(ctx, path, tmpdir, sfc_keys=None):
    """Decode, interpolate and vertically interpolate one isobaric file onto the model grid; the
    result is in ctx.st (PD, PINT, T, Q, CW, U, V grid-relative, FIS, PSFC_OUT, P3D_OUT).
    ctx.st must carry HGT_M (model terrain) and GLAT/GLON/VLAT/VLON."""
    import cupy as cp
    from namgpu.init import init_gfs_vinterp as vin

    st = ctx.st
    ensure_vcoord(ctx)
    lat_min = float(cp.degrees(cp.minimum(st["GLAT"].min(), st["VLAT"].min())))
    dec, sfc, grid = decode_source(ctx, path, tmpdir, lat_min, sfc_keys=sfc_keys)
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


def load_static(ctx):
    """NAM-grid static surface: the case's native file if present, else static/nam12_static.npz."""
    import cupy as cp
    from namgpu.init import sources as srcmod

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
    path = srcmod.source_path(ctx, row)
    if not path.is_file():
        url = row.get("url", "").format(ymd=f"{ctx.start:%Y%m%d}", cyc=ctx.start.hour)
        raise FileNotFoundError(f"{UNIT}: {path} missing" + (f" (fetch {url})" if url else ""))
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    Path(tmpdir).mkdir(exist_ok=True)
    init_grid_geo.run(ctx)                 # the interpolation needs GLAT/GLON/VLAT/VLON first
    static, static_from = load_static(ctx)
    st.add("HGT_M", (jm, im))[...] = static["HGT_SFC"]
    sfc, grid = model_state(ctx, path, tmpdir)

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
    for k in SOIL_KEYS:
        a = sfc[k]
        miss = (a == MISSING) | ~land_src
        a, _ = _ring_fill(a, miss)
        soil[k] = a
    from namgpu.init import init_gfs_hinterp as hin

    def h2(a):
        a = cp.ascontiguousarray(a, dtype=cp.float32)[None]
        out, _ = hin.interp_full(a, cp.ones(a.shape, dtype=cp.uint8), st["GLAT"], st["GLON"],
                                 grid=grid, stream=getattr(ctx, "stream", None))
        return out[0]

    def nonneg(a):
        return cp.maximum(cp.where(a == MISSING, 0.0, a), 0.0)

    raw = {
        "TSOIL": cp.ascontiguousarray(cp.stack([h2(soil[f"TSOIL{k}"]) for k in range(1, 5)])),
        "SOILW": cp.ascontiguousarray(cp.stack([cp.clip(h2(soil[f"SOILW{k}"]), 0.02, 0.6)
                                                for k in range(1, 5)])),
        "TMP_SFC": h2(sfc["TSK"]),
        "ICEC": cp.clip(h2(nonneg(sfc["ICEC"])), 0.0, 1.0),
        "TSOIL_3M": h2(soil["TSOIL4"]),
        "SNOD": h2(nonneg(sfc["SNOD"])),
        "WEASD": h2(nonneg(sfc["WEASD"])),
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
                                 "levels": int(ctx.st["P3D_OUT"].shape[0])}
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-static", required=True)
    build_static(ap.parse_args().build_static)
