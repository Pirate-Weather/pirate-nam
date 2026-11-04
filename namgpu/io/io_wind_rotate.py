"""io-wind-rotate: grid-relative winds at V points to earth-relative (east, north) winds at H points.

Port of the NMMB post rotation, sorc/nam_post0.fd/PROF_NMMB.f lines 2115-2150 and 2232-2237 (notice in
kernels/io_wind_rotate.cu). The rotation is done per V neighbour with that V point's exact inverse map,
then the earth-relative vectors of the in-domain neighbours are averaged. This is a decision, not the
post's average-then-rotate order: the geographic pole lies inside this domain, and the measured reason is
in the kernel header.

Coefficients. The grid-i and grid-j unit vectors (east, north) at each V point come from the grid's own
rotated-pole map (central differences of rotated_to_geographic, normalised): the same map that
init-gfs-wind-rotate inverts. This unit computes them itself, so it has no dependency on that unit.

Measured on the full 954 x 835 grid (tools/check-io-wind-rotate.py):
  - the post's closed-form sine SINALP = SINPH0*sin(TLON)/cos(STNLAT), with TLON = atan2(YY, XX) and
    DLM = STNLON - TLM0, agrees with the grid's sine to ~1e-7 away from the pole. The literal
    DLM = STNLON + TLM0D gives XX < 0 at ~237k of 797k points and does not reproduce the grid, so the sign
    of TLM0 used here is the one that does;
  - the post's COSALP = COS(ASIN(...)) is >= 0 by construction and cannot represent cos(theta) < 0, so the
    cosine comes from the map, not from the asin form;
  - the post's SINALP has the opposite sign to the one it is applied with on this grid. The rotation used
    here is the exact inverse of init-gfs-wind-rotate (round trip checked), not the post's sign. Which NAM
    convention is right is an open question for the project lead.

Reads:   ctx.st.U, ctx.st.V   grid-relative winds, [LM,JM,IM] float32 (V points)
         ctx.st.U10, ctx.st.V10   grid-relative 10 m winds, [JM,IM] float32 (H points, NMMB phy state)
Writes:  ctx.st.UEARTH, ctx.st.VEARTH   earth-relative eastward / northward winds at H points,
         [LM,JM,IM] float32, added with st.add() on first call. Every level, every H point is written.
         ctx.st.UE10, ctx.st.VE10   the same rotation of the 10 m winds, [JM,IM]; io-history-fields
         reads these as the surface winds for the history U10/V10.
Launch:  one kernel launch on ctx.stream for the 3-D winds, two elementwise ops for the 10 m winds
         (H points, rotated in place with the H-point vectors), no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.grid import rotated_to_geographic

UNIT = "io_wind_rotate"
KERNEL = "io_wind_rotate"
_H = 1.0e-6  # rad, central-difference step in rotated coordinates


def _v_rotation(g, offset=0.5):
    """Unit (east, north) components of the grid-i and grid-j directions at each V point, float32 [JM,IM].

    V(i,j) sits at rotated (wb + (i+1/2)*dlm, sb + (j+1/2)*dph), the same points as namgpu.grid.build.
    offset=0 gives the same vectors at the H points (wb + i*dlm, sb + j*dph).
    Returns (ce_e, ce_n, cn_e, cn_n).
    """
    jm, im = int(g.jm), int(g.im)
    jj = np.arange(jm)[:, None]
    ii = np.arange(im)[None, :]
    tlm_v = np.broadcast_to(g.wb + ii * g.dlm + offset * g.dlm, (jm, im)).astype(np.float64)
    tph_v = np.broadcast_to(g.sb + jj * g.dph + offset * g.dph, (jm, im)).astype(np.float64)
    _, lat0 = rotated_to_geographic(tlm_v, tph_v, g.tlm0, g.tph0)
    cos0 = np.cos(lat0)

    def dgeo(dtlm, dtph):
        lon_p, lat_p = rotated_to_geographic(tlm_v + dtlm, tph_v + dtph, g.tlm0, g.tph0)
        lon_m, lat_m = rotated_to_geographic(tlm_v - dtlm, tph_v - dtph, g.tlm0, g.tph0)
        dlon = np.arctan2(np.sin(lon_p - lon_m), np.cos(lon_p - lon_m)) / (2.0 * _H)
        dlat = (lat_p - lat_m) / (2.0 * _H)
        return cos0 * dlon, dlat  # (east, north) before normalisation

    def unit(e, n):
        m = np.hypot(e, n)
        return (e / m).astype(np.float32), (n / m).astype(np.float32)

    ce_e, ce_n = unit(*dgeo(_H, 0.0))
    cn_e, cn_n = unit(*dgeo(0.0, _H))
    return ce_e, ce_n, cn_e, cn_n


def _coefficients(ctx):
    g = ctx.grid
    cache = getattr(g, "_io_wind_rotate_coef", None)
    if cache is None:
        import cupy as cp
        cache = tuple(cp.asarray(a) for a in _v_rotation(g))
        g._io_wind_rotate_coef = cache
    return cache


def _rotate(ctx, u, v, ue, ve, lm: int) -> None:
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    ce_e, ce_n, cn_e, cn_n = _coefficients(ctx)
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        ce_e, ce_n, cn_e, cn_n,
        u, v, ue, ve,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    ue = st.add("UEARTH", (lm, jm, im))
    ve = st.add("VEARTH", (lm, jm, im))
    _rotate(ctx, st.U, st.V, ue, ve, lm)

    # 10 m winds. NMMB U10/V10 are H-point fields (parm/nam_phy_state.txt column H; TURBL hands JSFC the
    # H-averaged lowest-layer winds and SFCDIAGS writes U10/V10 there, module_TURBULENCE.F90:798-827),
    # grid-relative. They are rotated with the H-point vectors, no V-neighbour averaging.
    ue10 = st.add("UE10", (jm, im))
    ve10 = st.add("VE10", (jm, im))
    he_e, he_n, hn_e, hn_n = h_coefficients(ctx)
    u10, v10 = st.U10, st.V10
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        ue10[...] = u10 * he_e + v10 * hn_e
        ve10[...] = u10 * he_n + v10 * hn_n


def h_coefficients(ctx):
    """(ce_e, ce_n, cn_e, cn_n) at the H points, cupy float32 [JM,IM], built once per grid."""
    g = ctx.grid
    cache = getattr(g, "_io_wind_rotate_coef_h", None)
    if cache is None:
        import cupy as cp
        cache = tuple(cp.asarray(a) for a in _v_rotation(g, offset=0.0))
        g._io_wind_rotate_coef_h = cache
    return cache
