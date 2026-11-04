"""dyn-adv1-vert-uv: Crank-Nicolson vertical advection of U and V (ADV1, vertical section).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1706-1837 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_adv1_vert_uv.cu (dyn_adv1_vert_uv, one thread per column, grid_cols).

Reads:   DSG2 [LM], PDSG1 [LM] (device copy cached in ctx.tables), PD, PDO [JM,IM],
         PSGDT [LM+1,JM,IM] (levels 0..LM-2), U, V [LM,JM,IM].
Writes:  TCU, TCV [LM,JM,IM] (vertical U/V tendencies; interior columns i in [2,IM-2], j in
         [2,JM-2] 1-based, as the Fortran b2 limits).
         PDOP_UV [JM,IM] = pdop of the UV section, the 8-point *0.125 value (Fortran 1731-1732);
         dyn-adv1-horiz-uv reads it. Not named PDOP: dyn-adv1-horiz-t uses PDOP for the T-section
         (pd+pdo)*0.5 formula, and sharing the name would hand it the UV value.
Scratch: RCMW, CRW, RSTU, RSTV [LM,JM,IM] added with st.add() on the first call (the backward sweep
         needs them at every level; the Fortran keeps the same temporaries).
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv1_vert_uv"
KERNEL = "dyn_adv1_vert_uv"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pdsg1 is None:
        raise RuntimeError(
            "dyn-adv1-vert-uv: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    rcmw = st.add("RCMW", (lm, jm, im))
    crw = st.add("CRW", (lm, jm, im))
    rstu = st.add("RSTU", (lm, jm, im))
    rstv = st.add("RSTV", (lm, jm, im))
    pdop_uv = st.add("PDOP_UV", (jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(ctx.dt),
        st["DSG2"], pdsg1, st["PD"], st["PDO"], st["PSGDT"], st["U"], st["V"],
        st["TCU"], st["TCV"], rcmw, crw, rstu, rstv, pdop_uv,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, args)
