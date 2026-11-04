"""dyn-updatet: apply the temperature tendency, T = T + TCT, then reset TCT to zero.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2410-2456, subroutine UPDATET (NAM nam.v4.2.11,
83df4e13c1). Called from SOLVER_RUN after VTOA (module_SOLVER_GRID_COMP.F90:4648).
Kernel: namgpu/kernels/dyn_updatet.cu (kernel dyn_updatet, one thread per (i, j, l) point,
level in grid.z, interior limits 1-based i, j in [2, IM-1], [2, JM-1]).

Reads : T [LM,JM,IM], TCT [LM,JM,IM] (TCT is accumulated by dyn-vtoa and the advection units).
Writes: T (in place, T += TCT), TCT (set to 0 on the same points).
Scratch: none. One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_updatet"
KERNEL = "dyn_updatet"


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    args = (np.int32(lm), np.int32(im), np.int32(jm), st.T, st.TCT)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_2D, args)
