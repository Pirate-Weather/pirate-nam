"""init-gfs-wind-rotate: rotate GFS earth-relative winds to rotated-grid-relative winds at V points,
on the boundary strips.

Mode "new" (no NAM Fortran routine; the NMMB solver tree and the NPS sources have none; see
kernels/init_gfs_wind_rotate.cu). Kernel: init_gfs_wind_rotate_strips, one thread per strip point.

Geometry: the map is namgpu.grid.rotated_to_geographic (the same function that builds GLON/GLAT and
VLON/VLAT). At each V point the unit vectors of increasing rotated longitude (grid i) and rotated
latitude (grid j) in the local (east, north) basis are derived by central differences of that map and
normalised (the rotation is conformal, so the two vectors are orthogonal). Built once per grid, cached
on ctx.grid.

In:      ctx.gfs.u, ctx.gfs.v   GFS earth-relative eastward / northward winds, float32 [NL,NJS,NIS] on the
                                 GFS grid (ctx.gfs.grid), staged by the GFS path. NL must equal LM (the
                                 model levels). The strips UE/VE on the V-point grid are interpolated here with
                                 the init-gfs-hinterp rule (interp_strips, width LNSV). Missing staging is a
                                 refusal, never a zero fill.
Out:     ctx.st.U, ctx.st.V     grid-relative winds, written on the V boundary strips only (LNSV rows and
                                 columns, last row and column excluded). Interior U/V are left untouched.
Scratch: four [JM,IM] float32 coefficient arrays (ctx.grid._wind_rotate_coef) and the strip buffers of
         init-gfs-hinterp (slots VU and VV).
Launch:  one kernel on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.grid import rotated_to_geographic
from namgpu.init import init_gfs_hinterp as hin

UNIT = "init_gfs_wind_rotate"
KERNEL = "init_gfs_wind_rotate_strips"
BLOCK = 256
_H = 1.0e-6  # rad, central-difference step in rotated coordinates


def _unit_tangents(g):
    """Unit (east, north) components of the grid-i and grid-j directions at each V point.

    Returns four float32 [JM,IM] arrays: (ce_e, ce_n) for grid i, (cn_e, cn_n) for grid j.
    """
    jm, im = g.jm, g.im
    jj = np.arange(jm)[:, None]
    ii = np.arange(im)[None, :]
    tlm_v = np.broadcast_to(g.wb + ii * g.dlm + 0.5 * g.dlm, (jm, im)).astype(np.float64)
    tph_v = np.broadcast_to(g.sb + jj * g.dph + 0.5 * g.dph, (jm, im)).astype(np.float64)
    _, lat0 = rotated_to_geographic(tlm_v, tph_v, g.tlm0, g.tph0)
    cos0 = np.cos(lat0)

    def dgeo(dtlm, dtph):
        lon_p, lat_p = rotated_to_geographic(tlm_v + dtlm, tph_v + dtph, g.tlm0, g.tph0)
        lon_m, lat_m = rotated_to_geographic(tlm_v - dtlm, tph_v - dtph, g.tlm0, g.tph0)
        dlon = np.arctan2(np.sin(lon_p - lon_m), np.cos(lon_p - lon_m)) / (2.0 * _H)
        dlat = (lat_p - lat_m) / (2.0 * _H)
        return cos0 * dlon, dlat          # (east, north) before normalisation

    def unit(e, n):
        m = np.hypot(e, n)
        return (e / m).astype(np.float32), (n / m).astype(np.float32)

    ce_e, ce_n = unit(*dgeo(_H, 0.0))
    cn_e, cn_n = unit(*dgeo(0.0, _H))
    return ce_e, ce_n, cn_e, cn_n


def _coefficients(ctx):
    g = ctx.grid
    cache = getattr(g, "_wind_rotate_coef", None)
    if cache is None:
        import cupy as cp
        cache = tuple(cp.asarray(a) for a in _unit_tangents(g))
        g._wind_rotate_coef = cache
    return cache


def _staged_winds(ctx):
    gfs = getattr(ctx, "gfs", None)
    if gfs is None or getattr(gfs, "u", None) is None or getattr(gfs, "v", None) is None:
        raise RuntimeError(
            "init-gfs-wind-rotate: no staged GFS winds (ctx.gfs.u and ctx.gfs.v are required). "
            "Refusing to rotate zeros: the unit would overwrite the V boundary strips with zero wind.")
    return gfs


def run(ctx, time_level: int = 0):
    """Rotate the staged GFS winds at boundary time `time_level` onto the V strips of st.U / st.V.

    One time per call: st.U / st.V hold one time. The caller binds the second boundary time with
    time_level=1 into its own state, as read_bc does.
    """
    gfs = _staged_winds(ctx)
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    w = int(hin.LNSV)
    t = int(time_level)
    if not (0 <= t < hin.TIME_LEVELS):
        raise RuntimeError(f"init-gfs-wind-rotate: time_level {time_level} outside 0..{hin.TIME_LEVELS - 1}")
    if gfs.grid.lat_order != hin.LAT_ORDER:
        raise RuntimeError("init-gfs-wind-rotate: staged winds are not south to north")

    # Interpolation order: the earth-relative components are interpolated first (init-gfs-hinterp,
    # four_pt+average_4pt), then rotated here at V points.
    ue = hin.interp_strips(gfs.u.src, gfs.u.mask, st["VLAT"], st["VLON"], width=w, grid=gfs.grid,
                           slot="VU", stream=ctx.stream)
    ve = hin.interp_strips(gfs.v.src, gfs.v.mask, st["VLAT"], st["VLON"], width=w, grid=gfs.grid,
                           slot="VV", stream=ctx.stream)
    hin.require_read_bc_levels(ue, lm)
    hin.require_read_bc_levels(ve, lm)
    ce_e, ce_n, cn_e, cn_n = _coefficients(ctx)

    def slab(a):
        return a[t * lm:(t + 1) * lm]

    kern = kernels.get(UNIT, KERNEL)
    npts = max(w * im, jm * w)
    launch = ((npts + BLOCK - 1) // BLOCK, 4, lm)
    args = (
        np.int32(im), np.int32(jm), np.int32(lm), np.int32(w),
        ce_e, ce_n, cn_e, cn_n,
        slab(ue["s"]), slab(ue["n"]), slab(ue["w"]), slab(ue["e"]),
        slab(ve["s"]), slab(ve["n"]), slab(ve["w"]), slab(ve["e"]),
        st.U, st.V,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(launch, (BLOCK, 1, 1), args)
