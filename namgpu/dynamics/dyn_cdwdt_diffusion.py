"""dyn-cdwdt-diffusion: lateral diffusion of w added to DWDT (NMMB CDWDT, diffw branch).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3553-3592 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdwdt_diffusion.cu (cdwdt_diffusion, one thread per (i, j, l), grid_2d).

Reads:   W [LM,JM,IM], HDACX/HDACY [JM,IM] from ctx.st.
         DEF3D [LM,JM,IM] is the deformation (def3d) written by dyn-hdiff-coef earlier in the step
         (UNITS.json dependency). It is read, never created here: a missing DEF3D raises, so the
         term can never silently run on a zero deformation.
Writes:  DWDT [LM,JM,IM] in place (adds the diffusion increment to the local derivative from
         dyn-cdwdt-local-deriv).
Scalars: dt = ctx.dt, the DYNAMICS dt argument. diffw is dt > 0, so the backward digital-filter
         step (dt < 0) launches nothing.
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_diffusion"
KERNEL = "cdwdt_diffusion"


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    dt = ctx.dt if getattr(ctx, "dt", None) is not None else ctx.cfg.dt
    if not float(dt) > 0.0:
        return  # Fortran if(diffw): no lateral diffusion of w in the backward filter step

    if "DEF3D" not in st.fields:
        raise RuntimeError(
            "dyn-cdwdt-diffusion: DEF3D is not in ctx.st; dyn-hdiff-coef must run earlier in the "
            "step (it writes DEF3D, Fortran def3d)")
    deff = st["DEF3D"]

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(dt),
        st.W, deff, st.HDACX, st.HDACY, st.DWDT,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
