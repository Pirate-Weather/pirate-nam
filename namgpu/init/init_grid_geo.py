"""init-grid-geo: rotated lat-lon geometry of the H and V points and the SW corner of the domain.

Ported from nmm/module_CONTROL.F90:351-1139 (GRID_CONSTS, nam.v4.2.11, 83df4e13c1), the
geographic-coordinate statements only; the metric constants are unit init-grid-consts.
Kernel:  namgpu/kernels/init_grid_geo.cu (init_grid_geo), one launch, one thread per (i, j).
In:      ctx.cfg (tph0d, tlm0d, sbd, wbd, dlmd, dphd, global_), ctx.grid (im, jm), ctx.case_dir.
Out:     ctx.st['GLAT'], ctx.st['GLON'], ctx.st['VLAT'], ctx.st['VLON'] float32 [jm, im] radians
         ctx.tables['grid_sw']   float32[2] device array: (glat_sw, glon_sw) radians
         ctx.tables['geo_check'] dict with the measured analysis errors (see below)

The analysis check (added after the 2026-11-04 review):
  The NAM bgrd3d00 analysis (case_dir/raw/nam/bgrd3d00.grib2) carries the geographic latitude and
  longitude of every H point as NCEP records NLAT and ELON (GRIB2 discipline 0, category 191,
  parameters 192 and 193, GDT 3.32769, degrees). The kernel's H geometry is compared with them at
  every point, longitude wrapped across +/-pi (the domain contains the north pole, so the dateline
  is inside it). The analysis has no V-point record, so each V point is compared with the centroid
  of the four analysis H points that surround it (the V point is the rotated-grid midpoint of those
  four), an independent reference built from the analysis data.
  Both errors are read on the host once, in init (not on the step path). A failure raises: the
  geometry comes from the configuration, so a disagreement with the analysis means the
  configuration or the analysis decode is wrong, and rewriting the geometry from the analysis
  would hide that.
  ctx.analysis = {'NLAT': deg [jm, im], 'ELON': deg [jm, im]} overrides the file (tests).

Stream:  the kernel launches on ctx.stream; the check is device arithmetic plus one host read.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "init_grid_geo"
KERNEL = "init_grid_geo"
ANALYSIS = Path("raw/nam/bgrd3d00.grib2")
# GRIB2 selectors for the analysis geographic records (NCEP local, discipline 0, category 191).
NLAT_SEL = (0, 191, 192)
ELON_SEL = (0, 191, 193)
GDT_NAM = 32769
# Tolerances in radians, set from the packing of the records (measured 2026-11-04 on the 2025-03-14 12Z
# analysis with float64 geometry, scratch/geo_check.py): NLAT is packed to 0.01 deg and ELON to 0.1 deg,
# so the analysis itself is only good to 0.005 deg in latitude and 0.05 deg in longitude (cos-weighted);
# measured max 0.0061 deg and 0.050 deg. The error is the great-circle angle (no longitude blow-up at
# the pole, which is inside the domain). 1e-3 rad = 0.057 deg sits above the packing and below half a
# grid step (0.108 deg = 1.9e-3 rad), so a one-row or one-column shift of the decode is still refused.
GEO_TOL_RAD = 1.0e-3
VGEO_TOL_RAD = 1.0e-3


def _selector(disc, cat, par):
    def sel(row):
        return (int(row["discipline"]) == disc and int(row["category"]) == cat
                and int(row["parameter"]) == par and int(row["gdt"]) == GDT_NAM)
    return sel


def load_analysis(case_dir):
    """NLAT, ELON (degrees, device float32 [jm, im], NMMB layout) from the analysis GRIB2."""
    from namgpu.init import init_grib2_decode

    if case_dir is None:
        raise ValueError(f"{UNIT}: ctx.case_dir is not set, so the analysis cannot be read; the "
                         "geometry would go unchecked against the native analysis")
    path = Path(case_dir) / ANALYSIS
    if not path.is_file():
        raise FileNotFoundError(f"{UNIT}: {path} is missing; the geometry cannot be checked "
                                "against NLAT/ELON (run init-fetch-case)")
    got = init_grib2_decode.run(None, path, {"NLAT": _selector(*NLAT_SEL),
                                             "ELON": _selector(*ELON_SEL)})
    return got["NLAT"], got["ELON"]


def _check_units(nlat, elon):
    """Refuse anything that is not degrees on the NAM domain (radians would be off by 57x)."""
    import cupy as cp

    lat_max = float(cp.abs(nlat).max())
    lon_min, lon_max = float(elon.min()), float(elon.max())
    if lat_max > 90.0 + 1e-3:
        raise ValueError(f"{UNIT}: analysis NLAT reaches {lat_max:.3f}; not degrees")
    if lat_max < 2.0:
        raise ValueError(f"{UNIT}: analysis NLAT peaks at {lat_max:.3f}; this looks like radians, "
                         "and the NAM 12 km domain reaches the pole (NLAT ~ 90 deg)")
    if lon_min < -360.0 - 1e-3 or lon_max > 360.0 + 1e-3:
        raise ValueError(f"{UNIT}: analysis ELON spans [{lon_min:.3f}, {lon_max:.3f}]; not degrees")


def _wrap(x):
    import cupy as cp

    return cp.arctan2(cp.sin(x), cp.cos(x))


def _unit_vectors(lat, lon):
    import cupy as cp

    cl = cp.cos(lat)
    return cp.stack([cl * cp.cos(lon), cl * cp.sin(lon), cp.sin(lat)], axis=-1)


def check(st, nlat_deg, elon_deg):
    """Errors against the analysis in radians: (H max, H index, V max, V index), device values.

    H: max over points of the great-circle angle between (GLAT, GLON) and (NLAT, ELON).
    V: angle between (VLAT, VLON) and the normalised centroid of the four analysis H points that
       surround that V point (V(i0, j0) lies between H(i0, j0), H(i0+1, j0), H(i0, j0+1), H(i0+1, j0+1)).
    """
    import cupy as cp

    _check_units(nlat_deg, elon_deg)
    nlat = cp.radians(nlat_deg.astype(cp.float64))
    elon = cp.radians(elon_deg.astype(cp.float64))
    glat = st["GLAT"].astype(cp.float64)
    glon = st["GLON"].astype(cp.float64)
    ua = _unit_vectors(glat, glon)
    ub = _unit_vectors(nlat, elon)
    herr = cp.arctan2(cp.linalg.norm(cp.cross(ua, ub), axis=-1), cp.sum(ua * ub, axis=-1))
    h_max = herr.max()
    h_idx = cp.unravel_index(herr.argmax(), herr.shape)

    uh = _unit_vectors(nlat, elon)                       # [jm, im, 3]
    c = uh[:-1, :-1] + uh[:-1, 1:] + uh[1:, :-1] + uh[1:, 1:]
    c = c / cp.linalg.norm(c, axis=-1, keepdims=True)    # centroid, [jm-1, im-1, 3]
    uv = _unit_vectors(st["VLAT"].astype(cp.float64)[:-1, :-1],
                       st["VLON"].astype(cp.float64)[:-1, :-1])
    verr = cp.arctan2(cp.linalg.norm(cp.cross(uv, c), axis=-1), cp.sum(uv * c, axis=-1))
    v_max = verr.max()
    v_idx = cp.unravel_index(verr.argmax(), verr.shape)
    return h_max, h_idx, v_max, v_idx


def run(ctx):
    import cupy as cp

    cfg = ctx.cfg
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    st = ctx.st
    for name in ("GLAT", "GLON", "VLAT", "VLON"):
        if tuple(st[name].shape) != (jm, im):
            raise ValueError(f"{name} shape {st[name].shape}, expected {(jm, im)} ([j, i])")
        if st[name].dtype != cp.float32:
            raise ValueError(f"{name} must be float32")

    sw = ctx.tables.get("grid_sw")
    if sw is None:
        sw = cp.zeros(2, dtype=cp.float32)
        ctx.tables["grid_sw"] = sw

    is_global = bool(cfg.global_)
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(1 if is_global else 0),
            np.float32(cfg.tph0d), np.float32(cfg.tlm0d),
            np.float32(cfg.ref_sbd), np.float32(cfg.ref_wbd),     # reference origin (= sbd/wbd without a window)
            np.float32(cfg.dlmd), np.float32(cfg.dphd),
            np.int32(cfg.win_i0), np.int32(cfg.win_j0),           # moving-nest window offset (0, 0 otherwise)
            st["GLAT"], st["GLON"], st["VLAT"], st["VLON"], sw,
        ))
        analysis = getattr(ctx, "analysis", None)
        if analysis is None and not (Path(ctx.case_dir) / ANALYSIS).is_file() and \
                str(getattr(ctx, "init_source", "native")) != "native":
            # GFS/HRRR/RRFS start after NAM retirement: no native analysis to compare with. The
            # geometry is the configuration's; it was checked on every archived case that has one.
            ctx.tables["geo_check"] = {"checked": False, "why": "no NAM native analysis in the case"}
            return sw
        from namgpu import config as configmod
        if analysis is None and configmod.static_npz(cfg) is not None:
            # A domain configfile with its own grid (static_npz): the case's NAM analysis is on the NAM
            # 12 km grid, so it is not this grid's reference. The rotation is the same code the 12 km check
            # passes; the domain's GLAT/GLON also sit in its init-static-build file.
            ctx.tables["geo_check"] = {"checked": False, "why": "domain grid is not the NAM analysis grid"}
            return sw
        if analysis is None:
            nlat, elon = load_analysis(ctx.case_dir)
            win = cfg.raw.get("row_window")
            if win:   # NAMGPU_ROW_WINDOW band: the analysis is on the parent grid; check the band rows against it
                nlat, elon = nlat[win[0] - 1:win[1]], elon[win[0] - 1:win[1]]
        else:
            nlat = cp.asarray(analysis["NLAT"], dtype=cp.float32)
            elon = cp.asarray(analysis["ELON"], dtype=cp.float32)
        if tuple(nlat.shape) != (jm, im) or tuple(elon.shape) != (jm, im):
            raise ValueError(f"{UNIT}: analysis NLAT {nlat.shape} / ELON {elon.shape}, expected {(jm, im)}")
        h_max, h_idx, v_max, v_idx = check(st, nlat, elon)
        h_max, v_max = float(h_max), float(v_max)
        h_at = tuple(int(x) for x in h_idx)
        v_at = tuple(int(x) for x in v_idx)

    report = {"h_max_rad": h_max, "h_at_ji": h_at, "v_max_rad": v_max, "v_at_ji": v_at,
              "h_tol_rad": GEO_TOL_RAD, "v_tol_rad": VGEO_TOL_RAD,
              "dlmd": float(cfg.dlmd), "dphd": float(cfg.dphd), "wbd": float(cfg.wbd),
              "sbd": float(cfg.sbd), "tph0d": float(cfg.tph0d), "tlm0d": float(cfg.tlm0d)}
    ctx.tables["geo_check"] = report
    if h_max > GEO_TOL_RAD or v_max > VGEO_TOL_RAD:
        raise RuntimeError(
            f"{UNIT}: geometry disagrees with the analysis: H max {h_max:.3e} rad at (j,i)={h_at} "
            f"(tol {GEO_TOL_RAD:.0e}), V max {v_max:.3e} rad at (j,i)={v_at} (tol {VGEO_TOL_RAD:.0e}); "
            f"config dlmd={cfg.dlmd} dphd={cfg.dphd} wbd={cfg.wbd} sbd={cfg.sbd} "
            f"tph0d={cfg.tph0d} tlm0d={cfg.tlm0d}. Refused: the configuration or the analysis "
            "decode is wrong, so the geometry is not rewritten from the analysis.")
    return sw
