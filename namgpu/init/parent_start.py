"""init-parent-start: start a one-way nest from our own WOOF-NAM history (offline, like WRF ndown).

The 'parent' SOURCES row (kind nam_history). The parent is any nam-gpu run whose configfile turned on
history_handoff (io_history_fields.HANDOFF_FIELDS: skin/SST, sea ice, soil, snow, canopy water, land mask, plus the
parent grid keys as global attributes). The nest is its own rotated lat-lon B grid (its own pole, static and levels).

Path, every stage on the GPU except the NetCDF reads (host, one process per field, all cores):
  1. placement   nest H and V points (GLAT/GLON) -> the parent's rotated (tlm, tph) in degrees
                 (inverse of grid.rotated_to_geographic on the parent pole). Refused, with the measured margin,
                 unless every nest point is inside the parent minus its lnsh/lnsv boundary rows (check_inside).
  2. read        only the parent window that covers the nest (plus WINDOW_PAD cells), one process per field.
  3. columns     metgrid-style columns for init-gfs-vinterp, exactly the shape gfs_start.columns builds:
                 level 0 = the parent surface (PSFC, HGT, T2, RH from Q2, U10/V10), then the parent's own model
                 levels ground up (history is bottom-up), then a top cap at p = PT: PRES (3-D, varying),
                 height = midlayer of PH/g,
                 T = (T+300) (PRES/1e5)^(R/cp), RH from QVAPOR with the rhq kernel's own saturation formula
                 (so init-gfs-vinterp turns it back into the same q), earth-relative U/V at the nest V points.
                 Horizontal: the existing init-gfs-hinterp kernel (four_pt + average_4pt) on a Grid in the
                 parent's rotated degrees (lat0 = sbd + j0 dphd, lon0 = wbd + i0 dlmd, not periodic), the
                 nest points given as rotated (tph, tlm).
  4. vinterp     init-gfs-vinterp unchanged (PD, PINT, T, Q, U, V on the nest's hybrid levels), winds turned to
                 grid-relative with init-gfs-wind-rotate's coefficients (gfs_start.rotate_winds).
  5. condensate  QCLOUD, QRAIN, QICE, QSNOW go through the vinterp log kernel one species at a time into CLMR,
                 RWMR, CICE, SNMR on the nest levels, with the 1e-11 floor of the source (cwm_acc), and the
                 start sets cfg.do_clouds so init-native-atm builds CW, F_ICE, F_RAIN from them (the native-start
                 rule). The parent history writes QICE = 0 (Ferrier carries ice in QSNOW).
  6. surface     soil (TSLB/SMOIS/SH2O, TMN), snow, canopy water: the parent's LAND values, ring-filled over
                 parent water (no wrap), then interpolated. Skin temperature: the parent's land TSK on nest land,
                 the parent's water SST on nest water, each filled across the other mask, so a 3 km coastline
                 never takes a 12 km land/water mix. Sea ice from the parent water points. Static fields (terrain,
                 land mask, vegetation and soil type, roughness, greenness, albedo) are the nest's own
                 (configfile static_npz). The parent's liquid soil water is kept (ctx.tables['sfc_override'],
                 applied by init-native-sfc after its own frozen-fraction split), capped at SMC.
  7. shelter     T2 moved from the parent terrain to the nest terrain at 6.5 K/km, Q2 kept, U10/V10 turned to
                 grid-relative at the nest H points (as gfs_start does).

Run directory of the parent: ctx.parent_run, else configfile key parent_run, else env NAMGPU_PARENT_RUN.
The boundaries come from the same history (parent_bocos.py).
"""
from __future__ import annotations

from namgpu import cores as _cores

import os
from pathlib import Path

import numpy as np

UNIT = "init-parent-start"
G = 9.8060226                 # NMMB_G (history PH = g * zint, io_history_fields.cu hist_ph)
CAPPA = 287.04 / 1004.6       # NMMB_CAPPA (history T = theta - 300 with P00 = 1e5)
P00 = 1.0e5
WINDOW_PAD = 3                # parent cells read beyond the nest's extent (four_pt needs i+1; fill reach)
ATM3 = ("PRES", "PH", "T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "U", "V")
ATM2 = ("PSFC", "HGT", "T2", "Q2", "U10", "V10")
SFC2 = ("TSK", "SST", "SEAICE", "SNOW", "SNOWH", "CANWAT", "TMN", "LANDMASK", "SNOWC")
SOIL3 = ("TSLB", "SMOIS", "SH2O")
GRID_ATTRS = ("PARENT_TPH0D", "PARENT_TLM0D", "PARENT_WBD", "PARENT_SBD", "PARENT_DLMD", "PARENT_DPHD",
              "PARENT_IM", "PARENT_JM", "PARENT_LM", "PARENT_LNSH", "PARENT_LNSV", "PARENT_PT")
KM_PER_DEG = 6371.2 * np.pi / 180.0


# ---- where the parent is ------------------------------------------------------------------------------------------
def parent_run(ctx) -> Path:
    v = getattr(ctx, "parent_run", None) or ctx.cfg.raw.get("parent_run") or os.environ.get("NAMGPU_PARENT_RUN")
    if not v:
        raise ValueError(f"{UNIT}: source 'parent' needs the parent run directory (--parent-run, configfile key "
                         "parent_run, or NAMGPU_PARENT_RUN)")
    return Path(str(v)).expanduser()


def history_path(run: Path, valid) -> Path:
    return Path(run) / f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"


def parent_grid(path: Path) -> dict:
    """The parent grid keys from a hand-off history's global attributes (io_history_fields.handoff_attrs)."""
    import netCDF4
    if not Path(path).is_file():
        raise FileNotFoundError(f"{UNIT}: parent history {path} missing")
    with netCDF4.Dataset(path) as nc:
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
        have = set(nc.variables)
    miss = [k for k in GRID_ATTRS if k not in attrs]
    if miss or not attrs.get("HANDOFF"):
        raise ValueError(f"{UNIT}: {path.name} carries no parent hand-off set (attributes {miss} missing): run the "
                         "parent with the configfile key history_handoff: true")
    pg = {k[len("PARENT_"):].lower(): (int(attrs[k]) if k.endswith(("IM", "JM", "LM", "LNSH", "LNSV"))
                                       else float(attrs[k])) for k in GRID_ATTRS}
    pg["fields"] = have
    return pg


# ---- placement (geographic -> parent rotated) ----------------------------------------------------------------------
def to_rotated(lat, lon, tph0d, tlm0d):
    """Geographic (radians, cupy) -> rotated (tph, tlm) in DEGREES on the pole (tph0d, tlm0d): the inverse of
    grid.rotated_to_geographic (rotated equator through the centre)."""
    import cupy as cp
    lat = lat.astype(cp.float64)
    lon = lon.astype(cp.float64)
    p0, l0 = np.radians(tph0d), np.radians(tlm0d)
    x = cp.cos(lat) * cp.cos(lon - l0)
    y = cp.cos(lat) * cp.sin(lon - l0)
    z = cp.sin(lat)
    tph = cp.arcsin(cp.clip(np.cos(p0) * z - np.sin(p0) * x, -1.0, 1.0))
    tlm = cp.arctan2(y, np.cos(p0) * x + np.sin(p0) * z)
    return cp.degrees(tph), cp.degrees(tlm)


def placement(st, pg) -> dict:
    """Nest H and V points in the parent's rotated degrees, the parent window that covers them, and the margin."""
    import cupy as cp
    h = to_rotated(st["GLAT"], st["GLON"], pg["tph0d"], pg["tlm0d"])
    v = to_rotated(st["VLAT"], st["VLON"], pg["tph0d"], pg["tlm0d"])
    tph = cp.concatenate([h[0].ravel(), v[0].ravel()])
    tlm = cp.concatenate([h[1].ravel(), v[1].ravel()])
    fi = (tlm - pg["wbd"]) / pg["dlmd"]          # fractional parent H column (0-based)
    fj = (tph - pg["sbd"]) / pg["dphd"]
    lo_i, hi_i = float(fi.min()), float(fi.max())
    lo_j, hi_j = float(fj.min()), float(fj.max())
    im, jm = pg["im"], pg["jm"]
    # V-point rows/columns use lnsv; H uses lnsh. The nest must sit inside both: use the larger.
    nb = max(pg["lnsh"], pg["lnsv"])
    edge = {"west": lo_i - nb, "east": (im - 1 - nb) - hi_i, "south": lo_j - nb, "north": (jm - 1 - nb) - hi_j}
    tph_c = float(cp.abs(tph).max())
    km = {"west": edge["west"] * pg["dlmd"] * KM_PER_DEG * np.cos(np.radians(tph_c)),
          "east": edge["east"] * pg["dlmd"] * KM_PER_DEG * np.cos(np.radians(tph_c)),
          "south": edge["south"] * pg["dphd"] * KM_PER_DEG, "north": edge["north"] * pg["dphd"] * KM_PER_DEG}
    i0 = max(0, int(np.floor(lo_i)) - WINDOW_PAD)
    i1 = min(im, int(np.ceil(hi_i)) + 1 + WINDOW_PAD)
    j0 = max(0, int(np.floor(lo_j)) - WINDOW_PAD)
    j1 = min(jm, int(np.ceil(hi_j)) + 1 + WINDOW_PAD)
    side = min(edge, key=edge.get)
    return {"h": h, "v": v, "window": (j0, j1, i0, i1),
            "margin_cells": {k: round(x, 2) for k, x in edge.items()},
            "margin_km": {k: round(float(x), 1) for k, x in km.items()},
            "min_side": side, "min_cells": round(edge[side], 2), "min_km": round(float(km[side]), 1),
            "inside": bool(edge[side] >= 0.0)}


def check_inside(pl: dict, what: str = "nest") -> None:
    if not pl["inside"]:
        raise ValueError(
            f"{UNIT}: the {what} is not inside the parent minus its boundary rows: on the {pl['min_side']} side it "
            f"reaches {-pl['min_cells']:.2f} parent cells ({-pl['min_km']:.1f} km) into the boundary zone or past "
            f"the edge (margins in cells {pl['margin_cells']}, km {pl['margin_km']}). Move or shrink the nest, or "
            "run a bigger parent: the boundary rows are relaxed toward the parent's own boundaries, not model state.")


# ---- reading the parent window (host, all cores) -------------------------------------------------------------------
def _read_one(args):
    path, name, window = args
    import netCDF4
    j0, j1, i0, i1 = window
    with netCDF4.Dataset(path) as nc:
        v = nc.variables[name]
        a = v[0, ..., j0:j1, i0:i1]
    return name, np.ascontiguousarray(np.ma.filled(a, np.nan), dtype=np.float32)


def read_windows(jobs, workers=None) -> dict:
    """[(path, name, window)] -> {(path, name): float32 array}. One process per field (netCDF-C is not
    thread-safe and its zlib decode is one core per call), so the reads run on every core."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or max(1, min(len(jobs), _cores.usable(), 24))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        return {(str(j[0]), n): a for j, (n, a) in zip(jobs, ex.map(_read_one, jobs, chunksize=1))}


# ---- columns for init-gfs-vinterp -----------------------------------------------------------------------------------
def _rh(tk, p, w):
    """RH (%) such that init_gfs_vinterp_rhq gives back mixing ratio w (same svp constants, es in hPa)."""
    import cupy as cp
    es = cp.float32(6.112) * cp.exp(cp.float32(17.67) * (tk - cp.float32(273.15)) / (tk - cp.float32(29.65)))
    qs = cp.float32(0.622) * es / (p / cp.float32(100.0) - es)
    return cp.float32(100.0) * cp.maximum(w, 0.0) / qs


def _grid(pg, window):
    from namgpu.init import init_gfs_hinterp as hin
    j0, _, i0, _ = window
    return hin.Grid(lat0=pg["sbd"] + j0 * pg["dphd"], lon0=pg["wbd"] + i0 * pg["dlmd"],
                    dlat=pg["dphd"], dlon=pg["dlmd"], lon_periodic=False)


def _pts(rot):
    import cupy as cp
    tph, tlm = rot
    return (cp.ascontiguousarray(cp.radians(tph), dtype=cp.float32),
            cp.ascontiguousarray(cp.radians(tlm), dtype=cp.float32))


def interp3(a, pts, grid, stream=None):
    import cupy as cp
    from namgpu.init import init_gfs_hinterp as hin
    a = cp.ascontiguousarray(a, dtype=cp.float32)
    if a.ndim == 2:
        a = a[None]
    out, _ = hin.interp_full(a, cp.ones(a.shape, dtype=cp.uint8), pts[0], pts[1], grid=grid, stream=stream)
    return out


def columns(ctx, f, pg, pl):
    """Metgrid-style columns on the nest grid from the parent window f (device arrays, history order)."""
    import cupy as cp
    s = getattr(ctx, "stream", None)
    grid = _grid(pg, pl["window"])
    H, V = _pts(pl["h"]), _pts(pl["v"])
    pres = f["PRES"]                                              # [LMp, jw, iw], bottom-up
    tk = (f["T"] + cp.float32(300.0)) * (pres / cp.float32(P00)) ** cp.float32(CAPPA)
    zmid = cp.float32(0.5 / G) * (f["PH"][:-1] + f["PH"][1:])
    rh = _rh(tk, pres, f["QVAPOR"])
    q2 = f["Q2"]
    rh2 = _rh(f["T2"], f["PSFC"], q2 / (cp.float32(1.0) - q2))

    # Top cap: the parent's top mid-level sits half a layer below its model top PT, and the nest's top mid-level can
    # lie above it (its PD differs), where the vinterp kernels have nothing to bracket and leave their -99999.9 fill
    # (2026-11-04 first nest start: level 1 of T, Q, U, V at 84-89 k of 96 k points, the run blew up in step 0). So
    # every column ends with one entry AT the model top: p = PT, height = the parent's top interface (PH[LM]/g), and
    # the top layer's temperature, mixing ratio and wind (constant above the last level), as an isobaric source
    # supplies levels above the model top.
    ptop = cp.full(f["PSFC"].shape, cp.float32(pg["pt"]), dtype=cp.float32)
    rh_top = _rh(tk[-1], ptop, f["QVAPOR"][-1])

    def stack(s2, s3, top=None):
        out = cp.empty((s3.shape[0] + 2,) + s3.shape[1:], dtype=cp.float32)
        out[0] = s2
        out[1:-1] = s3
        out[-1] = s3[-1] if top is None else top
        return out

    zero = cp.zeros(f["PSFC"].shape, dtype=cp.float32)
    on_h = lambda a: interp3(a, H, grid, s)                       # noqa: E731
    on_v = lambda a: interp3(a, V, grid, s)                       # noqa: E731
    gi = {
        "PRES_IN": on_h(stack(f["PSFC"], pres, ptop)),
        "GHT_IN": on_h(stack(f["HGT"], zmid, f["PH"][-1] * cp.float32(1.0 / G))),
        "TT_IN": on_h(stack(f["T2"], tk)),
        "RH_IN": on_h(stack(rh2, rh, rh_top)),
        "UU_IN": on_v(stack(f["U10"], f["U"])),
        "VV_IN": on_v(stack(f["V10"], f["V"])),
        "SNMR_IN": on_h(stack(zero, f["QSNOW"])),
        "CLWMR_IN": on_h(stack(zero, f["QCLOUD"])),
        "RWMR_IN": on_h(stack(zero, f["QRAIN"])),
        "CICE_IN": on_h(stack(zero, f["QICE"])),
    }
    gi["FRIMEF_IN"] = cp.zeros_like(gi["PRES_IN"])
    return gi, on_h(f["HGT"])[0]


def model_state(ctx, f, pg, pl):
    """Parent window -> nest model state (PD, PINT, T, Q, CW, U, V grid-relative, FIS, PSFC_OUT, P3D_OUT), the
    same calls as gfs_start.model_state after its decode. Returns the vinterp input columns (gi)."""
    from namgpu.init import gfs_start
    from namgpu.init import init_gfs_vinterp as vin
    st = ctx.st
    gfs_start.ensure_vcoord(ctx)
    gi, soilhgt = columns(ctx, f, pg, pl)
    st.add("SOILHGT", soilhgt.shape)[...] = soilhgt
    ctx.tables["gfs_hinterp"] = gi
    first = ctx.first_step
    ctx.first_step = True
    try:
        vin.run(ctx)
    finally:
        ctx.first_step = first
        ctx.tables.pop("gfs_hinterp", None)
    gfs_start.rotate_winds(ctx, st["U"], st["V"])
    return gi


def species(ctx, gi):
    """QCLOUD/QRAIN/QICE/QSNOW columns -> CLMR/RWMR/CICE/SNMR on the nest levels (vinterp log kernel, as its
    condensate branch, with the source's 1e-11 floor)."""
    import cupy as cp
    from namgpu import kernels
    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    pres, p3d = gi["PRES_IN"], st["P3D_OUT"]
    nlev = int(pres.shape[0])
    k = kernels.get("init_gfs_vinterp", "init_gfs_vinterp_log")
    i32 = np.int32
    for dst, src in (("CLMR", "CLWMR_IN"), ("RWMR", "RWMR_IN"), ("CICE", "CICE_IN"), ("SNMR", "SNMR_IN")):
        out = st.add(dst, (lm, jm, im))
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            pres, p3d, gi[src], out, i32(im), i32(jm), i32(nlev), i32(lm),
            i32(0), i32(1), i32(0), i32(int(cfg.lnsh)), i32(im + 1), i32(jm + 1)))
        out[...] = cp.where(out < cp.float32(1.0e-11), cp.float32(0.0), out)
    # Warm fractions. The NPS cold-start rule (init-native-atm) sums the species into CW and leaves F_ICE = F_RAIN = 0,
    # so every parent hydrometeor would start as cloud water: the first nest from the real 12 km parent had QSNOW and
    # QRAIN 0 and no echo at f00 where the parent had 48.8 dBZ (2026-11-04). The parent history is the Ferrier state
    # written by the spec_adv=F rule (io_history_fields.cu hist_moist_theta: QSNOW = CW F_ICE, QRAIN = CW (1-F_ICE)
    # F_RAIN, QCLOUD = CW (1-F_ICE)(1-F_RAIN)), so the fractions come back exactly as the history's spec_adv=T
    # rebuild does (io_history_fields._spec_adv_condensate): F_ICE = (QS+QI)/CWM, F_RAIN = QR/(QC+QR). F_RIMEF has no
    # history field: 1 (unrimed), the floor the native warm start uses. init-native-atm.warm_start applies them.
    qc, qr = st["CLMR"], st["RWMR"]
    qs = st["SNMR"] + st["CICE"]
    liq = qc + qr
    cwm = liq + qs
    ctx.tables["parent_warm"] = {
        "F_ICE": cp.where(cwm > 0, qs / cp.where(cwm > 0, cwm, 1.0), 0.0).astype(cp.float32),
        "F_RAIN": cp.where(liq > 0, qr / cp.where(liq > 0, liq, 1.0), 0.0).astype(cp.float32),
    }


# ---- surface -------------------------------------------------------------------------------------------------------
def fill_masked(a, valid, rings=4000):
    """Fill invalid points from the mean of valid 8-neighbours, ring by ring, without wrapping (a parent window
    is not periodic). a [..., jw, iw]; valid bool [jw, iw]. Raises when the window has no valid point."""
    import cupy as cp
    a = cp.where(valid, a, cp.float32(0.0)).astype(cp.float32)
    m = ~valid
    if not bool(valid.any()):
        raise ValueError(f"{UNIT}: no parent point of the needed kind in the window")
    for _ in range(rings):
        if not bool(m.any()):
            break
        v = cp.where(m, cp.float32(0.0), a)
        w = (~m).astype(cp.float32)
        pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
        vp, wp = cp.pad(v, pad), cp.pad(w, ((1, 1), (1, 1)))
        jw, iw = m.shape
        sm = cp.zeros_like(a)
        n = cp.zeros(m.shape, dtype=cp.float32)
        for dj in (0, 1, 2):
            for di in (0, 1, 2):
                if dj == 1 and di == 1:
                    continue
                sm += vp[..., dj:dj + jw, di:di + iw]
                n += wp[dj:dj + jw, di:di + iw]
        new = m & (n > 0)
        if not bool(new.any()):
            break
        a = cp.where(new, sm / cp.maximum(n, 1.0), a)
        m = m & ~new
    return a


def surface(ctx, f, pg, pl, static):
    """raw_sfc for init-native-sfc (the keys gfs_start.stage fills) from the parent hand-off fields."""
    import cupy as cp
    s = getattr(ctx, "stream", None)
    grid = _grid(pg, pl["window"])
    H = _pts(pl["h"])
    land_p = f["LANDMASK"] > 0.5
    water_p = ~land_p
    nest_land = static["LAND"] > 0.5

    def on_h(a):
        return interp3(a, H, grid, s)

    def land(a):
        return on_h(fill_masked(a, land_p))

    def water(a):
        return on_h(fill_masked(a, water_p))[0] if bool(water_p.any()) else None

    tsk_l = land(f["TSK"])[0]
    tsk_w = water(f["SST"])
    tsk = tsk_l if tsk_w is None else cp.where(nest_land, tsk_l, tsk_w)
    ice = water(cp.clip(f["SEAICE"], 0.0, 1.0))
    tslb = land(f["TSLB"])
    smois = cp.clip(land(f["SMOIS"]), 0.02, 0.6)
    sh2o = cp.minimum(cp.maximum(land(f["SH2O"]), 0.0), smois)
    raw = {
        "TSOIL": cp.ascontiguousarray(tslb),
        "SOILW": cp.ascontiguousarray(smois),
        "TMP_SFC": tsk,
        "ICEC": cp.zeros_like(tsk) if ice is None else cp.clip(ice, 0.0, 1.0),
        "TSOIL_3M": land(f["TMN"])[0],
        "SNOD": cp.maximum(land(f["SNOWH"])[0], 0.0),
        "WEASD": cp.maximum(land(f["SNOW"])[0], 0.0),
        "CNWAT": cp.maximum(land(f["CANWAT"])[0], 0.0),
        "SNOWC": cp.clip(land(f["SNOWC"])[0], 0.0, 1.0),
        "LAND": static["LAND"], "ALBDO": static["ALBDO"], "VEG": static["VEG"], "SFCR": static["SFCR"],
        "VGTYP": cp.rint(static["VGTYP"]).astype(cp.int32),
        "SOTYP": cp.rint(static["SOTYP"]).astype(cp.int32),
    }
    raw["WTMP"] = raw["TMP_SFC"]
    return raw, sh2o


# ---- the chain stage -----------------------------------------------------------------------------------------------
def read_time(path: Path, pl, names) -> dict:
    import cupy as cp
    got = read_windows([(path, n, pl["window"]) for n in names])
    return {n: cp.asarray(got[(str(path), n)]) for n in names}


def stage(ctx, name, row):
    """Chain stage for the 'parent' row (sources.stage -> here)."""
    import time
    import cupy as cp
    from namgpu.init import gfs_start, init_grid_geo, sources as srcmod

    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    run = parent_run(ctx)
    path = history_path(run, ctx.start)
    pg = parent_grid(path)
    init_grid_geo.run(ctx)
    static, static_from = gfs_start.load_static(ctx)
    st.add("HGT_M", (jm, im))[...] = static["HGT_SFC"]
    pl = placement(st, pg)
    check_inside(pl)
    t0 = time.perf_counter()
    f = read_time(path, pl, ATM3 + ATM2 + SFC2 + SOIL3)
    t_read = time.perf_counter() - t0
    gi = model_state(ctx, f, pg, pl)
    species(ctx, gi)
    del gi
    ctx.cfg.do_clouds = True        # the parent carries condensate: init-native-atm builds CW/F_ICE/F_RAIN from it

    for a, b in (("TMP", "T"), ("SPFH", "Q"), ("UGRD", "U"), ("VGRD", "V"), ("PRES", "P3D_OUT")):
        st.fields[a] = st[b]
    st.add("PSFC", (jm, im))[...] = st["PSFC_OUT"]
    ctx.tables.pop("native_warm", None)

    raw, sh2o = surface(ctx, f, pg, pl, static)
    ctx.tables["sfc_override"] = {"SH2O": sh2o}

    from namgpu.init import init_gfs_hinterp as hin  # noqa: F401
    from namgpu.io import io_wind_rotate as iwr
    grid = _grid(pg, pl["window"])
    H = _pts(pl["h"])
    s = getattr(ctx, "stream", None)
    t2 = interp3(f["T2"], H, grid, s)[0] - cp.float32(0.0065) * (st["HGT_M"] - st["SOILHGT"])
    q2 = cp.maximum(interp3(f["Q2"], H, grid, s)[0], 0.0)
    ue, vn = interp3(f["U10"], H, grid, s)[0], interp3(f["V10"], H, grid, s)[0]
    ce_e, ce_n, cn_e, cn_n = iwr.h_coefficients(ctx)
    srcmod.shelter_state(ctx, t2, q2, ue * ce_e + vn * ce_n, ue * cn_e + vn * cn_n)
    ctx.tables["raw_sfc"] = raw
    ctx.tables["soil_hydraulics"] = {k: cp.asarray(v) for k, v in srcmod.soil_hydraulics().items()}
    j0, j1, i0, i1 = pl["window"]
    ctx.tables["grid_source"] = {
        "source": name, "file": str(path), "static": static_from, "parent_levels": int(f["PRES"].shape[0]),
        "parent_grid": {k: v for k, v in pg.items() if k != "fields"},
        "window_ji": [j0, j1, i0, i1], "margin_cells": pl["margin_cells"], "margin_km": pl["margin_km"],
        "read_s": round(t_read, 2), "read_procs": min(len(ATM3 + ATM2 + SFC2 + SOIL3), _cores.usable())}
    print(f"{UNIT}: nest start from {path.name} ({pg['im']}x{pg['jm']}x{pg['lm']} parent, window "
          f"{i1 - i0}x{j1 - j0} cells read in {t_read:.1f} s); smallest margin {pl['min_cells']} cells "
          f"({pl['min_km']} km, {pl['min_side']}); static {Path(static_from).name}", flush=True)
    return ctx.tables["grid_source"]
