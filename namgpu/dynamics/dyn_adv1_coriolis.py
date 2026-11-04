"""dyn-adv1-coriolis: Adams-Bashforth extrapolation of U and V (Fortran 1829-1833), the Coriolis and
curvature tendency (TCU, TCV) and the diagonal V-point flux sums (PFXX1..PFNWY1) of ADV1, for all
levels in one launch.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1829-1866 (NAM nam.v4.2.11, 83df4e13c1), body of adv1
(module_DYNAMICS_ROUTINES.F90:1233). Lines 1829-1833 sit in the Fortran range of dyn-adv1-vert-uv
but no unit ported them, so they run here as kernel stage [0], before the Coriolis stage uses U1D/V1D.
Kernel: namgpu/kernels/dyn_adv1_coriolis.cu (adv1_coriolis, one thread per (i, j, l), grid_2d).

Reads : U, V (state), UP, VP (state; previous-step values, overwritten with U, V by stage [0]),
        CURV [JM] (ctx.tables, init-grid-consts), F_V [JM,IM] (ctx.tables, init-grid-consts: Fortran
        f(i,j) of module_CONTROL), PFX, PFY, PFNE, PFNW (produced by dyn-dht, Fortran pfx at
        module_DYNAMICS_ROUTINES.F90:685, which runs earlier in the step), TCU, TCV (inout).
Writes: U1D, V1D (level-indexed scratch, written by stage [0]); UP, VP; TCU, TCV;
        PFXX1, PFYX1, PFNEX1, PFNWX1, PFXY1, PFYY1, PFNEY1, PFNWY1 in ctx.st.
Scratch: U1D, V1D, PF*1 are level-indexed [LM,JM,IM] arrays added with st.add(). They are 3-D
         because the whole vertical loop is one launch (CONTRACT section 3): 10 arrays x 191 MB,
         about 1.9 GB of VRAM at the 12 km shape. The ADV1 units that consume PF*1 must use these names.
Missing inputs raise: nothing is zero-filled, so a producer that has not run stops the step.
Regional only: global=True raises (the kernel carries the regional loop limits, not the global ones).
Launch : one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv1_coriolis"
KERNEL = "adv1_coriolis"
# Adams-Bashforth weights of module_DYNAMICS_ROUTINES adv1 (kfpt=single): cfc=1.533, bfc=1.-cfc
CFC = np.float32(1.533)
BFC = np.float32(1.0) - CFC


def _need(st, name: str, producer: str):
    """State field that must already exist: raises if its producer has not run (no zero fill)."""
    try:
        return st[name]
    except KeyError:
        raise RuntimeError(
            f"dyn-adv1-coriolis: {name} missing; produced by {producer}, which must run first") from None


def _curv(ctx, jm: int):
    """Device copy of curv(j) [JM], built by init-grid-consts (ctx.tables['CURV'])."""
    if "CURV" not in ctx.tables:
        raise RuntimeError(
            "dyn-adv1-coriolis: CURV missing from ctx.tables; produced by init-grid-consts, "
            "which must run first")
    curv = ctx.tables["CURV"]
    if tuple(curv.shape) != (jm,):
        raise RuntimeError(f"dyn-adv1-coriolis: CURV shape {tuple(curv.shape)}, expected {(jm,)}")
    return curv


def _f_v(ctx, jm: int, im: int):
    """Device copy of the Coriolis parameter f(i,j) [JM,IM], built by init-grid-consts (ctx.tables['F_V'])."""
    if "F_V" not in ctx.tables:
        raise RuntimeError(
            "dyn-adv1-coriolis: F_V missing from ctx.tables; produced by init-grid-consts, "
            "which must run first")
    f2d = ctx.tables["F_V"]
    if tuple(f2d.shape) != (jm, im):
        raise RuntimeError(f"dyn-adv1-coriolis: F_V shape {tuple(f2d.shape)}, expected {(jm, im)}")
    return f2d


def run(ctx):
    if bool(ctx.cfg.global_):
        raise RuntimeError(
            "dyn-adv1-coriolis: global=True is not ported (the kernel carries the regional loop "
            "limits; the Fortran global J=3..JM-2 branch is missing)")
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    curv = _curv(ctx, jm)
    f2d = _f_v(ctx, jm, im)
    u = _need(st, "U", "the state load")
    v = _need(st, "V", "the state load")
    up = _need(st, "UP", "the state load")
    vp = _need(st, "VP", "the state load")
    pfx = _need(st, "PFX", "dyn-dht")
    pfy = _need(st, "PFY", "dyn-dht")
    pfne = _need(st, "PFNE", "dyn-dht")
    pfnw = _need(st, "PFNW", "dyn-dht")
    u1d = st.add("U1D", shape3)
    v1d = st.add("V1D", shape3)
    pfxx1 = st.add("PFXX1", shape3)
    pfyx1 = st.add("PFYX1", shape3)
    pfnex1 = st.add("PFNEX1", shape3)
    pfnwx1 = st.add("PFNWX1", shape3)
    pfxy1 = st.add("PFXY1", shape3)
    pfyy1 = st.add("PFYY1", shape3)
    pfney1 = st.add("PFNEY1", shape3)
    pfnwy1 = st.add("PFNWY1", shape3)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(ctx.dt), CFC, BFC,
        u, v, up, vp, curv, f2d, u1d, v1d, st.TCU, st.TCV,
        pfx, pfy, pfne, pfnw,
        pfxx1, pfyx1, pfnex1, pfnwx1, pfxy1, pfyy1, pfney1, pfnwy1,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
