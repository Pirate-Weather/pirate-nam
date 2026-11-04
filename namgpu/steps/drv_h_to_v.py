"""drv-h-to-v: H_TO_V_2D / H_TO_V_3D, average a field from H (mass) points to V (wind) points.

Ported from nmm/module_H_TO_V.F90:42-122 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/drv_h_to_v.cu, kernels h_to_v_2d (one thread per (i, j)) and h_to_v_3d
(one thread per (i, j, level), grid_2d). Full single-tile domain: the stencil runs over
i = 1..IM-1, j = 1..JM-1 (ITE_B1/JTE_B1), so the last V column and row are not written.

run(ctx, array_h, array_v)    V(i,j) = 0.25 * (H(i,j) + H(i+1,j) + H(i,j+1) + H(i+1,j+1)).
                              array_h and array_v are both [JM, IM] (2-D) or both [LM, JM, IM]
                              (3-D). Called from SOLVER_RUN wherever the Fortran calls H_TO_V.
run(ctx) with no arrays       done-rule check (tools/check_unit.py): H = st.T (3-D),
                              V = scratch st.add('H_TO_V_CHECK_V', shape of T), so no tabled
                              state field is overwritten.

One launch on ctx.stream, no host sync, no per-call allocation (the scratch is allocated once).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.steps import drv_loop_limits as dll

UNIT = "drv_h_to_v"
KERNEL_2D = "h_to_v_2d"
KERNEL_3D = "h_to_v_3d"
CHECK_V = "H_TO_V_CHECK_V"


def run(ctx, array_h=None, array_v=None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if array_h is None:
        array_h = st.T
        array_v = st.add(CHECK_V, array_h.shape)
    if array_v is None:
        raise ValueError("drv-h-to-v: array_v is required when array_h is given")
    if array_h.shape != array_v.shape:
        raise ValueError(f"drv-h-to-v: shape mismatch {array_h.shape} vs {array_v.shape}")
    lim = dll.table(ctx)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        if array_h.ndim == 2:
            if array_h.shape != (jm, im):
                raise ValueError(f"drv-h-to-v: 2-D shape {array_h.shape} != (JM, IM)=({jm}, {im})")
            kern = kernels.get(UNIT, KERNEL_2D)
            grid = kernels.grid_2d(im, jm, 1)
            kern(grid, kernels.BLOCK_2D, (np.int32(im), np.int32(jm), lim, array_h, array_v,
                                          np.int32(slab.rows(ctx)[0])))
        elif array_h.ndim == 3:
            if array_h.shape != (lm, jm, im):
                raise ValueError(f"drv-h-to-v: 3-D shape {array_h.shape} != (LM, JM, IM)")
            kern = kernels.get(UNIT, KERNEL_3D)
            grid = kernels.grid_2d(im, jm, lm)
            kern(grid, kernels.BLOCK_2D,
                 (np.int32(lm), np.int32(im), np.int32(jm), lim, array_h, array_v,
                  np.int32(slab.rows(ctx)[0])))
        else:
            raise ValueError(f"drv-h-to-v: array rank must be 2 or 3, got {array_h.ndim}")
