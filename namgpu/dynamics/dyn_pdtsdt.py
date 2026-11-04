"""dyn-pdtsdt: surface-pressure tendency PSDT, vertical mass flux PSGDT, PD update.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1045-1228 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_pdtsdt.cu (pdtsdt_col, one thread per column).

Reads:   SG2 [LM+1], PD, DIV [LM,JM,IM], TDIV [LM,JM,IM] (TDIV is produced by dyn-dht).
Writes:  PD (updated in place), PDO, PSDT [JM,IM] (added to ctx.st), PSGDT [LM+1,JM,IM] (levels
         1..LM-1), DIV (cumulative sum over levels, in place: the later routines read it that way).
Scratch: PSDT and TDIV are not in the NAM state tables; they are added once with st.add().
Launch:  one kernel on ctx.stream, no host sync.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_pdtsdt"
KERNEL = "pdtsdt_col"


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    psdt = st.add("PSDT", (jm, im))
    tdiv = st.add("TDIV", (lm, jm, im))

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(ctx.dt),
        st.SG2, st.PD, st.PDO, psdt, st.PSGDT, st.DIV, tdiv,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
