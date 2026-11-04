"""dyn-cdzdt-extmode: take the external (barotropic) mode out of W.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3236-3273 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdzdt_extmode.cu (cdzdt_extmode_col, one thread per column).

Reads : PDO [JM,IM], DSG2 [LM], PDSG1 [LM] (grid, cached in ctx.tables), W [LM,JM,IM].
Writes: W (inout, external mode removed), BARO [JM,IM] (interior columns; read by cdwdt).
Launch: one kernel on ctx.stream, no host sync, no per-step allocation (BARO is a state-table array).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdzdt_extmode"
KERNEL = "cdzdt_extmode_col"


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
    if getattr(g, "pdsg1", None) is None:
        raise RuntimeError(
            "dyn-cdzdt-extmode: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    baro = st.add("BARO", (jm, im))
    dsg2 = st["DSG2"]
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        dsg2, pdsg1, st["PDO"], st["W"], baro,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
