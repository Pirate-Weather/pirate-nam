"""dyn-adv2-q2-mid: ADV2 setup for a tracer group (floor, Q2 interface-to-midlayer, sqrt transform).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:4238-4461 (ADV2, NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_adv2_q2_mid.cu (adv2_q2_mid_col, one thread per column).

Reads:   TRACERS [NT,LM,JM,IM] (group kss..kse, inout), EPSQ2 [LM] (ctx.tables, init-owned:
         init-derive-state uploads it from the Fortran init rule; absent raises), PD and PDO [JM,IM].
Writes:  TRACERS (floored in the group; Q2 interpolated to midlayers when Q2 is in the group),
         TRACERS_SQRT [NT,LM,JM,IM] (sqrt of TRACERS over the group, scratch added here; the name
         dyn-adv2-vert, dyn-adv2-horiz, dyn-adv2-bndry and dyn-mono-fix read),
         PDOP [JM,IM] ((pd+pdo)*0.5, scratch added here, read by the CN section).

Group:   run(ctx, kss, kse) takes Fortran 1-based slots. The two SOLVER_RUN calls are
         Q only (kss = kse = INDX_Q, module_SOLVER_GRID_COMP.F90:4321-4327): no Q2, no interpolation;
         CW..Q2 (kss = INDX_CW, kse = INDX_Q2, lines 4460-4470): interpolation runs.
         The default is the CW..Q2 call.
Slots:   ctx.tables['indx'] (drv-tracer-index) gives Q, CW, Q2. If it is not built yet, the slots come
         from cfg through the same host mirror drv-tracer-index uses (microphysics, num_tracers_chem);
         there is no hardcoded default (module_SOLVER_INTERNAL_STATE.F90:1382-1400, module_VARS_STATE.F90:797).
Launch:  one kernel on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv2_q2_mid"
KERNEL = "adv2_q2_mid_col"
from namgpu.steps.drv_tracer_index import host_index


def _init_value(ctx, key: str):
    if key not in ctx.tables:
        raise RuntimeError(
            f"dyn-adv2-q2-mid: ctx.tables[{key!r}] is not set; init-derive-state uploads EPSQ2 "
            "from the NAM init rule (module_INIT_READ_BIN.F90:237-240) before the first step")
    return ctx.tables[key]


def run(ctx, kss=None, kse=None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    indx = ctx.tables.get("indx")
    if indx is None:
        indx = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
    if kss is None:
        kss = indx["INDX_CW"]
    if kse is None:
        kse = indx["INDX_Q2"]
    epsq2 = _init_value(ctx, "EPSQ2")

    tracers = st.TRACERS
    nt = int(tracers.shape[0])
    if not (1 <= kss <= kse <= nt):
        raise ValueError(f"dyn-adv2-q2-mid: tracer group {kss}..{kse} outside 1..{nt}")

    s1 = st.add("TRACERS_SQRT", (nt, lm, jm, im))
    pdop = st.add("PDOP", (jm, im))

    # host-side record of the range TRACERS_SQRT was written for (no device sync): the consumer
    # dyn-adv2-vert refuses to run on another range.
    ctx.tables["ADV2_SQRT_RANGE"] = (int(kss), int(kse))
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.int32(kss), np.int32(kse), np.int32(indx["INDX_Q2"]),
        epsq2, tracers, s1, st.PD, st.PDO, pdop,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
