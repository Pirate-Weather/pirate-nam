"""dyn-cdwdt-vert-adv: vertical advection of W (column), accumulated into DWDT.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3593-3652 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdwdt_vert_adv.cu (cdwdt_vert_adv_col, one thread per column).

Reads : DSG2 [LM], PDSG1 [LM] (grid, cached in ctx.tables), PDO [JM,IM], W [LM,JM,IM],
        PSGDT [LM+1,JM,IM] (levels 0..LM-2).
Writes: DWDT [LM,JM,IM] (inout: adds the vertical-advection term to the existing DWDT).
Scratch: none (ttb/tta are per-column registers).
One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_vert_adv"
KERNEL = "cdwdt_vert_adv_col"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if getattr(g, "pdsg1", None) is None:
        raise RuntimeError(
            "dyn-cdwdt-vert-adv: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    dsg2 = st["DSG2"]
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        dsg2, pdsg1, st["PDO"], st["W"], st["PSGDT"], st["DWDT"],
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
