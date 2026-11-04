"""dyn-vtoa: omega-alpha term of the thermodynamic equation and the interface-pressure update.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2191-2347, subroutine VTOA (NAM nam.v4.2.11,
83df4e13c1). Called once per step from SOLVER_RUN (module_SOLVER_GRID_COMP.F90:4611).
Kernel: namgpu/kernels/dyn_vtoa.cu (kernel dyn_vtoa, one thread per column, l-sweep inside).

Reads : SG2 [LM+1], PSDT [JM,IM] (from dyn-pdtsdt), DWDT [LM,JM,IM] (state carried from init
        and the cdwdt units), RTOP [LM,JM,IM] (from dyn-pgforce-geopot), TDIV [LM,JM,IM] (from
        dyn-dht), PINT [LM+1,JM,IM].
Writes: OMGALF and TCT (accumulated, += toa), PINT (levels 2..LM+1 recomputed, level 1 = PT).
EF4T  : int_state%EF4T, held as the host scalar st.scalars["EF4T"]. It starts at the CONSTS value
        0.5*dt/cp (cp = 1004.6, module_CONTROL.F90:299, single precision, dt = cfg.dt). Only the
        DFI filter rescales it, through ef4t_filter_start, ef4t_direction_change and
        ef4t_filter_end, which mirror SOLVER_RUN (module_SOLVER_GRID_COMP.F90:3151-3155,
        3228-3237, 3330-3336). Outside a DFI filter the value is the CONSTS value.
PT    : grid.pt.
Scratch: PSDT, TDIV and RTOP are not NMM state-table names; st.add() returns the arrays the
         earlier units created (same shapes), so the call is idempotent.
One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_vtoa"
KERNEL = "dyn_vtoa"
NMMB_CP = np.float32(1004.6)
EF4T_KEY = "EF4T"


def consts_ef4t(dt):
    """CONSTS ef4t = 0.5*dt/cp (module_CONTROL.F90:299), single precision."""
    return np.float32(0.5) * np.float32(dt) / NMMB_CP


def ef4t(ctx):
    """int_state%EF4T: the CONSTS value until a DFI filter step rescales it."""
    v = ctx.st.scalars.get(EF4T_KEY)
    if v is None:
        v = consts_ef4t(ctx.cfg.dt)
        ctx.st.scalars[EF4T_KEY] = v
    return np.float32(v)


def ef4t_filter_start(ctx, dt_step, fund_dt=None):
    """SOLVER_RUN FIRST_PASS block (module_SOLVER_GRID_COMP.F90:3151-3155): save EF4T_SV, then
    EF4T = IFACT*DTRATIO*EF4T with IFACT = +1 for dt_step >= 0 else -1 and DTRATIO = ABS(DT/FUND_DT)
    (line 2958). FUND_DT is the fundamental step, cfg.dt by default. Returns EF4T_SV."""
    if fund_dt is None:
        fund_dt = ctx.cfg.dt
    saved = ef4t(ctx)
    ifact = np.float32(1.0) if dt_step >= 0 else np.float32(-1.0)
    dtratio = np.abs(np.float32(dt_step) / np.float32(fund_dt))
    ctx.st.scalars[EF4T_KEY] = ifact * dtratio * saved
    return saved


def ef4t_direction_change(ctx):
    """changedir block (module_SOLVER_GRID_COMP.F90:3228-3237): IFACT = -1, EF4T = IFACT*EF4T."""
    ctx.st.scalars[EF4T_KEY] = np.float32(-1.0) * ef4t(ctx)


def ef4t_filter_end(ctx, saved):
    """end_filt block (module_SOLVER_GRID_COMP.F90:3330-3336): EF4T = EF4T_SV."""
    ctx.st.scalars[EF4T_KEY] = np.float32(saved)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    psdt = st.add("PSDT", (jm, im))
    tdiv = st.add("TDIV", (lm, jm, im))
    rtop = st.add("RTOP", (lm, jm, im))
    ef = ef4t(ctx)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    args = (
        np.int32(lm), np.int32(im), np.int32(jm),
        np.float32(ctx.dt), ef, np.float32(g.pt),
        st.SG2, psdt, st.DWDT, rtop, st.OMGALF, st.PINT, tdiv, st.TCT,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_COL, args)
