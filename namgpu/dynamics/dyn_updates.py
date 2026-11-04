"""dyn-updates: apply the tracer tendency to the tracers, TRACERS += TRACERS_TEND, then zero it.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2353-2405 (NAM nam.v4.2.11, 83df4e13c1), UPDATES.
Kernel: namgpu/kernels/dyn_updates.cu (dyn_updates, one thread per (i, j) with the level in grid.z).

Reads and writes: TRACERS [NTR, LM, JM, IM] (updated in place for tracers kss..kse, 1-based) and
    TRACERS_TEND [NTR, LM, JM, IM] (added to TRACERS, then reset to 0 in the interior).
Scratch: TRACERS_TEND is not in the NAM state tables (the Fortran keeps it as the tracer tendency
    in int_state); it is added once with st.add() and shared with the advection unit (adv2).
Ranges: the driver always passes them (UNITS plan entries in driver.py). SOLVER_RUN calls UPDATES
    with KSS = INDX_Q, KSE1 = INDX_Q in the q_tracer block (4321-4392), and with KSS = INDX_CW,
    KSE1 = INDX_Q2 in the SPEC_ADV=.false. branch (4459-4461, call at 4543). The defaults 1..NTR are
    only for standalone calls. NTR is TRACERS.shape[0], which State allocates as Fortran
    NUM_TRACERS_TOTAL; a mismatch with ctx.tables["indx"] raises.
Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_updates"
KERNEL = "dyn_updates"


def run(ctx, kss: int | None = None, kse: int | None = None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntr = int(st.TRACERS.shape[0])
    tab = ctx.tables.get("indx") if hasattr(ctx, "tables") else None
    if tab is not None and int(tab["NUM_TRACERS_TOTAL"]) != ntr:
        raise ValueError(f"dyn-updates: TRACERS has {ntr} slots, Fortran NUM_TRACERS_TOTAL is "
                         f"{int(tab[NUM_TRACERS_TOTAL])}")
    kss = 1 if kss is None else int(kss)
    kse = ntr if kse is None else int(kse)
    if not (1 <= kss <= kse <= ntr):
        raise ValueError(f"dyn-updates: tracer range {kss}..{kse} outside 1..{ntr}")

    tend = st.add("TRACERS_TEND", (ntr, lm, jm, im))

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(lm), np.int32(im), np.int32(jm), np.int32(ntr),
        np.int32(kss), np.int32(kse),
        st.TRACERS, tend,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
