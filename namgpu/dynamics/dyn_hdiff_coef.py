"""dyn-hdiff-coef: HDIFF setup and the grand vertical loop (deformation DEF3D, slope masks FMLX_3D,
FMLY_3D).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2531-2833 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_hdiff_coef.cu (dyn_hdiff_coef, one thread per (i, j, l), one launch).

Reads:  U, V, W, Q2, Z, SM, SICE from ctx.st.
        DXV, RARE, RDXH [JM] from ctx.tables['grid'] (init-grid-consts GRID_CONSTS rows, the same
        rows the rest of the step uses); DYH, RDYH from ctx.tables['grid']['scalars'] (read to the
        host once, on the first call).
        LPT2 (int) and EPSQ2 [LM] from ctx.tables: published by init-vcoord-recover
        (namgpu.init.init_vcoord_recover.publish_vcoord_tables) from the vertical coordinate.
        T, Q and CW are not read here: the apply unit (dyn-hdiff-apply) uses them.
Writes: DEF3D, FMLX_3D, FMLY_3D (added to ctx.st on the first call, reused every later step).
        hkx/hky/hkfx/hkfy and the gradient terms are dyn-hdiff-apply's, not written here.
        FACDIF (Fortran 2659-2662, 1.0 in both branches) is not used here; dyn-hdiff-apply takes it.
Flags:  hydro and global come from ctx.cfg (both false in the 12 km CONUS configuration).
Launch: one kernel on ctx.stream, no per-step allocation. The only host read is the one-time DYH/RDYH
        scalar copy on the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_hdiff_coef"
KERNEL = "dyn_hdiff_coef"
SC_DYH, SC_RDYH = 0, 2          # GRID_CONSTS scalars layout (init_grid_consts.cu scal[8])


def _grid(ctx):
    grid = ctx.tables.get("grid")
    if grid is None:
        raise RuntimeError(
            "dyn-hdiff-coef: ctx.tables['grid'] is not set; init-grid-consts must run first")
    return grid


def _init_value(ctx, key: str):
    if key not in ctx.tables:
        raise RuntimeError(
            f"dyn-hdiff-coef: ctx.tables[{key!r}] is not set; init-vcoord-recover must publish it "
            "(publish_vcoord_tables) before the first step")
    return ctx.tables[key]


def _grid_scalar(ctx, idx: int, key: str) -> np.float32:
    """Host copy of one GRID_CONSTS scalar, read once and cached in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = np.float32(cp.asnumpy(_grid(ctx)["scalars"][idx]))
    return ctx.tables[key]


def run(ctx):
    st = ctx.st
    cfg = ctx.cfg
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    def3d = st.add("DEF3D", (lm, jm, im))
    fmlx = st.add("FMLX_3D", (lm, jm, im))
    fmly = st.add("FMLY_3D", (lm, jm, im))

    grid = _grid(ctx)
    dxv = grid["dxv"]
    rare = grid["rare"]
    rdxh = grid["rdxh"]
    dyh = _grid_scalar(ctx, SC_DYH, "hdiff_coef_dyh")
    rdyh = _grid_scalar(ctx, SC_RDYH, "hdiff_coef_rdyh")
    epsq2 = _init_value(ctx, "EPSQ2")
    lpt2 = int(_init_value(ctx, "LPT2"))

    kern = kernels.get(UNIT, KERNEL)
    grid_dims = kernels.grid_2d(im, jm, lm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid_dims, kernels.BLOCK_2D, (
            st["U"], st["V"], st["W"], st["Q2"], st["Z"], st["SM"], st["SICE"],
            dxv, rare, rdxh, epsq2,
            def3d, fmlx, fmly,
            dyh, rdyh,
            np.int32(im), np.int32(jm), np.int32(lm),
            np.int32(lpt2), np.int32(bool(cfg.hydro)), np.int32(bool(cfg.global_)),
        ))
