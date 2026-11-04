"""dyn-adv1-vert-t: ADV1 setup and Crank-Nicolson vertical advection of T (TCT, vertical part).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1233-1538 (NAM nam.v4.2.11, 83df4e13c1), the vertical
Crank-Nicolson block only. Kernel: namgpu/kernels/dyn_adv1_vert_t.cu (adv1_vert_t, one thread per
column, l-loop inside).

Reads:   T [LM,JM,IM], PSGDT [LM+1,JM,IM] (levels 1..LM-1), PD, PDO [JM,IM], DSG2 [LM], PDSG1 [LM]
         (device copy of pdsg1 cached in ctx.tables, as in dyn_pgforce_geopot).
Writes:  TCT [LM,JM,IM] (vertical T tendency; the horizontal ADV1 units accumulate onto it later in
         the same step). TCT is written at interior columns only (its_b1..ite_b1, jts_b1..jte_b1).
Scratch: ADV1_RCMT and ADV1_CRT [LM,JM,IM], added once with st.add() and reused every step.
         The Fortran rstt is held in TCT during the sweep (see the kernel header).
Launch:  one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv1_vert_t"
KERNEL = "adv1_vert_t"


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
            "dyn-adv1-vert-t: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    rcmt = st.add("ADV1_RCMT", (lm, jm, im))
    crt = st.add("ADV1_CRT", (lm, jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm),
            np.float32(ctx.dt),
            st["T"], st["PSGDT"], st["PD"], st["PDO"], st["DSG2"], pdsg1,
            rcmt, crt, st["TCT"],
        ))
