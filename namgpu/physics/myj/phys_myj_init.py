"""phys-myj-init: MYJPBL_INIT (module_BL_MYJPBL.F90:1612-1657, nam.v4.2.11), the whole subroutine.

Fortran: MYJPBL_INIT(EXCH_H, RESTART, IDS..LM, IMS..JME, ITS..JTE). Body:
  1. if not RESTART: EXCH_H = 0 over I < IM, J < JM, K = 1..LM (kernel phys_myj_init_exch_h);
  2. CALL HERFTBL, TABLEPT, TABLETT, unconditionally (restart or not), via phys_myj_tables.run(ctx),
     which builds HERFF, PTBL, TTBL and the table scalars into ctx.tables.
Both steps launch on ctx.stream. No host sync, no host math on the tables.

Writes: ctx.st["EXCH_H"] (CuPy float32 [LM+1, JM, IM], the state.py LEVELS_LMP1 allocation). The Fortran
declares EXCH_H(IMS:IME, JMS:JME, 1:LM); levels 0..LM-1 here are Fortran K = 1..LM and are the only
levels zeroed. Level LM is outside the Fortran declaration and is not touched. ctx.tables gets the MYJ tables.
Restart: restart=None binds ctx.restart (Fortran int_state%RESTART, resolved in namgpu/driver.py). A context
with no restart flag and no explicit argument raises, so a restart can never zero EXCH_H by omission.
"""
from __future__ import annotations

import contextlib

from namgpu import kernels
from namgpu.physics.myj import phys_myj_tables

UNIT = "phys_myj_init"
BLOCK = kernels.BLOCK_2D


def run(ctx, restart: bool | None = None):
    """MYJPBL_INIT. Returns ctx.st["EXCH_H"]."""
    if restart is None:
        if not hasattr(ctx, "restart"):
            raise ValueError("phys_myj_init.run: restart not given and ctx has no restart flag "
                             "(Fortran int_state%RESTART)")
        restart = bool(ctx.restart)
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    exch_h = ctx.st["EXCH_H"]
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        if not restart:
            k_zero = kernels.get(UNIT, "phys_myj_init_exch_h")
            k_zero(kernels.grid_2d(im, jm, lm), BLOCK, (exch_h, im, jm, lm))
        if getattr(ctx, "tables", None) is None:
            ctx.tables = {}
        phys_myj_tables.run(ctx)
    return exch_h
