"""io-uphlcy: updraft helicity, layer 2-5 km AGL and 0-3 km AGL, hourly max and min.

Ported from nmm/module_DIAGNOSE.F90:2114-2186 (CALC_UPHLCY, NAM nam.v4.2.11, 83df4e13c1), called
from MAX_FIELDS_HR at lines 2472 and 2477 (fer_hires path; the same calls sit in MAX_FIELDS at 2066
and 2071). Kernel: namgpu/kernels/io_uphlcy.cu (io_uphlcy).

Reads : U, V, W_TOT (the Fortran W dummy: the caller passes int_state%W_TOT, module_SOLVER_GRID_COMP.F90:5049),
        Z [LM,JM,IM] float32 (state); FIS [JM,IM] (state); DXH [JM] (grid.dxh, uploaded
        once into ctx.tables as "DXH_ROW"); DYH scalar (grid.dyh).
        W_TOT must hold W + BARO. VSOUND writes it when it runs; it is zero at init (Fortran line 956).
        On the OPER restart path Fortran sets W_TOT = W (module_SOLVER_GRID_COMP.F90:1498); restart read
        is not ported yet (driver.py refuses restarts), so that line must be ported with the restart reader.
Writes: UPHLMAX25, UPHLMIN25, UPHLMAX03, UPHLMIN03 [JM,IM] (state, hourly max/min; in place,
        MAX/MIN with the existing value, as the Fortran INOUT arguments).
Launch: one kernel on ctx.stream (one thread per column, the level loop inside), both layers in
        the same launch. No host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "io_uphlcy"
KERNEL = "io_uphlcy"

# HLOWER/HUPPER of the two Fortran calls (MAX_FIELDS_HR lines 2064-2071).
LAYER_25 = (2000.0, 5000.0)
LAYER_03 = (0.0, 3000.0)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        if host is None:
            raise RuntimeError(f"io-uphlcy: host vector {key!r} is not set on the grid")
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    dxh = _device_vector(ctx, "DXH_ROW", g.dxh)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm), np.int32(im), np.int32(jm),
        st.U, st.V, st.W_TOT, st.Z, st.FIS, dxh, np.float32(g.dyh),
        np.float32(LAYER_25[0]), np.float32(LAYER_25[1]),
        np.float32(LAYER_03[0]), np.float32(LAYER_03[1]),
        st.UPHLMAX25, st.UPHLMIN25, st.UPHLMAX03, st.UPHLMIN03,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
