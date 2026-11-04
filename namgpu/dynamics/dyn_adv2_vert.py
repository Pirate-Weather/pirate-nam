"""dyn-adv2-vert: Crank-Nicolson vertical advection of the tracer group (ADV2, vertical block).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:4463-4569 (subroutine adv2, nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_adv2_vert.cu (adv2_vert, one thread per column, species and level loops inside).

Reads : PD, PDO [JM,IM], PSGDT [LM+1,JM,IM] (levels 0..LM-2 used), DSG2 [LM] from ctx.st;
        PDSG1 [LM] from ctx.grid (device copy cached in ctx.tables, as in dyn_adv1_vert_t);
        TRACERS_SQRT [NTR,LM,JM,IM] (Fortran s1, produced by the horizontal ADV2 setup before this block).
Writes: TRACERS_TEND [NTR,LM,JM,IM] (Fortran tcs) for species kss..kse, interior columns only.
Scratch: ADV2_RCMS, ADV2_CRS [LM,JM,IM] added once with st.add() and reused every step.
Ranges: kss/kse are Fortran 1-based slots passed per call by the driver step plan (driver.py):
        the Q call is INDX_Q..INDX_Q with idtadt = IDTADTQ; the other call is INDX_CW..INDX_Q2 with
        idtadt = IDTADT (module_SOLVER_GRID_COMP.F90:4321-4327 and 4459-4461). The defaults (all tracers,
        cfg.idtadt) are for standalone use only.
Producer: dyn-adv2-q2-mid records its range in ctx.tables["ADV2_SQRT_RANGE"]; run() raises unless it
        matches, so TRACERS_SQRT outside the range is never read silently.
Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv2_vert"
KERNEL = "adv2_vert"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx, kss: int | None = None, kse: int | None = None, idtadt: int | None = None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pdsg1 is None:
        raise RuntimeError(
            "dyn-adv2-vert: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    ntr = int(st["TRACERS"].shape[0])
    # TRACERS_SQRT and TRACERS_TEND are the int_state temporaries of SOLVER_RUN; not in the NAM tables.
    s1 = st.add("TRACERS_SQRT", (ntr, lm, jm, im))
    tcs = st.add("TRACERS_TEND", (ntr, lm, jm, im))

    kss = 1 if kss is None else int(kss)
    kse = ntr if kse is None else int(kse)
    if not (1 <= kss <= kse <= ntr):
        raise ValueError(f"dyn-adv2-vert: tracer range {kss}..{kse} outside 1..{ntr}")
    idt = int(ctx.cfg.idtadt) if idtadt is None else int(idtadt)

    rec = ctx.tables.get("ADV2_SQRT_RANGE")
    if rec != (kss, kse):
        raise RuntimeError(
            f"dyn-adv2-vert: TRACERS_SQRT was last written for range {rec}, need {kss}..{kse}; "
            "dyn-adv2-q2-mid must run on the same range first")

    rcms = st.add("ADV2_RCMS", (lm, jm, im))
    crs = st.add("ADV2_CRS", (lm, jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm), np.int32(ntr),
            np.int32(kss - 1), np.int32(kse - 1),
            np.float32(ctx.dt), np.int32(idt),
            st["PD"], st["PDO"], st["PSGDT"], st["DSG2"], pdsg1,
            s1, rcms, crs, tcs,
        ))
    ctx.tables["ADV2_TCS_RANGE"] = (kss, kse)   # read by dyn-adv2-horiz: vertical TCS written first
