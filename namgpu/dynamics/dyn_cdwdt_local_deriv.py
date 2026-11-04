"""dyn-cdwdt-local-deriv: local time derivative of w on H points, DWDT = (W - DWDT) / dt.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3533-3552, the local-derivative block of CDWDT
(NAM nam.v4.2.11, 83df4e13c1). Called from CDWDT after dyn-cdwdt-boundary-smooth.
Kernel: namgpu/kernels/dyn_cdwdt_local_deriv.cu (kernel dyn_cdwdt_local_deriv, one thread per
(i, j, l) point, level in grid.z, full H grid).

Reads : W [LM,JM,IM] (current w), DWDT [LM,JM,IM] (holds the old w on entry).
Writes: DWDT [LM,JM,IM] (in place, local dw/dt).
rdt = float32(1.0) / float32(dt), the single-precision value of the Fortran rdt = 1./dt.
Scratch: none. One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_local_deriv"
KERNEL = "dyn_cdwdt_local_deriv"


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    rdt = np.float32(1.0) / np.float32(ctx.dt)
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    args = (np.int32(lm), np.int32(im), np.int32(jm), rdt, st.W, st.DWDT)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_2D, args)
