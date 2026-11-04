"""dyn-cdzdt-nonhydro: CDZDT setup and the nonhydrostatic equation (height Z and local dw/dt, w).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2920-3139 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdzdt_nonhydro.cu (dyn_cdzdt_nonhydro, one thread per column, l-loops inside).

Reads : T, Q, CW, PINT, PD, PDO, FIS, PSGDT, RTOP, DWDT, W, DSG2 from ctx.st;
        PDSG1 from ctx.grid (device copy cached in ctx.tables).
Writes: W, DWDT (inout), PDWDT and Z in ctx.st. PDWDT and Z are not NMM state-table names and are
        added with st.add on the first call (zeros), then reused every step. Z is [LM+1,JM,IM]
        (the shape PHINT and turbl use); only levels 1..LM are written here.
        BARO is declared by the Fortran routine but never assigned here; not written.
hydro is read from ctx.cfg (config hydro = false for the 12 km NAM run).
Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdzdt_nonhydro"
KERNEL = "dyn_cdzdt_nonhydro"


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
            "dyn-cdzdt-nonhydro: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    pdwdt = st.add("PDWDT", (lm, jm, im))
    z = st.add("Z", (lm + 1, jm, im))   # interface array [LM+1,JM,IM]; levels 1..LM written here
    rtop = st.add("RTOP", (lm, jm, im))   # written by dyn-pgforce-geopot; same array, not re-zeroed
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm),
            np.int32(1 if ctx.cfg.hydro else 0),
            np.float32(ctx.dt),
            st["DSG2"], pdsg1, st["FIS"], st["PD"], st["PDO"], st["PSGDT"],
            st["CW"], st["Q"], rtop, st["T"], st["PINT"],
            st["DWDT"], pdwdt, st["W"], z,
        ))
