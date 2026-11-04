"""dyn-pgforce-geopot: PGFORCE setup and the geopotential / APEL / DFI column recursion.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:51-248 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_pgforce_geopot.cu (one thread per column, upward l-loop).

Reads : FIS, PD, T, Q, CW, PINT, DSG2 from ctx.st; PDSG1 from ctx.grid (device copy cached in
        ctx.tables).
Writes: RTOP, APEL_3D, FIM_3D, DFI_3D in ctx.st (the last four are not NMM state-table names
        and are added with st.add on the first call; they are reused every later step).
One launch on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "dyn_pgforce_geopot"
KERNEL = "dyn_pgforce_geopot"


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
            "dyn-pgforce-geopot: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    rtop = st.add("RTOP", (lm, jm, im))
    apel_3d = st.add("APEL_3D", (lm, jm, im))
    fim_3d = st.add("FIM_3D", (lm, jm, im))
    dfi_3d = st.add("DFI_3D", (lm, jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    with ctx.stream:
        kern(grid, kernels.BLOCK_COL, (
            st["FIS"], st["PD"], st["T"], st["Q"], st["CW"], st["PINT"], st["DSG2"], pdsg1,
            rtop, apel_3d, fim_3d, dfi_3d,
            np.int32(im), np.int32(jm), np.int32(lm),
        ))
