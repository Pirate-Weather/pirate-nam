"""dyn-updateuv: apply the U, V tendencies TCU, TCV, then the CFL wind-speed limit.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2461-2526, subroutine UPDATEUV (NAM nam.v4.2.11,
83df4e13c1). Kernel: namgpu/kernels/dyn_updateuv.cu (dyn_updateuv, one thread per (i, j),
level in grid.z).

Reads : U, V, TCU, TCV [LM,JM,IM] from ctx.st (TCU/TCV from dyn-pgforce-uvupdate).
Writes: U, V (updated in place); TCU, TCV zeroed on the _b1/_b2 range (as the Fortran).
No scratch arrays. One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_updateuv"
KERNEL = "dyn_updateuv"


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(lm), np.int32(im), np.int32(jm),
        st.U, st.V, st.TCU, st.TCV,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
