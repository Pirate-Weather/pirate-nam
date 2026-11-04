"""drv-rime-factor-update: RIME_FACTOR_UPDATE, rime factor before and after species advection.

Ported from nmm/module_SOLVER_GRID_COMP.F90:11688-11771 (NAM nam.v4.2.11, 83df4e13c1), called from
SOLVER_RUN at 4448 (RIME_FACTOR_INPUT = .TRUE., before advection) and 4590 (RIME_FACTOR_INPUT =
.FALSE., after advection). Kernel: namgpu/kernels/drv_rime_factor_update.cu, kernels
rime_factor_before and rime_factor_after (one thread per (i, j), level in grid.z, domain only).

Gate (RIME_FACTOR_ADVECT, module_SOLVER_GRID_COMP.F90:2885-2890): both calls run only when
MICROPHYSICS == 'fer_hires' AND F_QG AND SPEC_ADV. F_QG is .TRUE. for fer_hires
(module_SOLVER_INTERNAL_STATE.F90:890-903), so the port tests microphysics and spec_adv.
parm/nam_configfile sets spec_adv: .false., so the unit is dormant in the 12 km config.

run(ctx, branch="after")      after advection (default, plan entry at 4590):
                              F_RIMEF = MIN(50, MAX(1, QG/QS)) where QG, QS > EPSQ, else 1.
                              Reads QS, QG [LM,JM,IM]; writes F_RIMEF [LM,JM,IM].
run(ctx, branch="before")     before advection (plan entry at 4448): QG = QS * F_RIMEF.
                              Reads QS, F_RIMEF; writes QG.
The halo exchange in the before branch is dropped (single card, CONTRACT.md section 1). Domain
only: the port stores no halo ring, so the Fortran after-branch ring cells (IMS..IME outside
IDS..IDE) are not written. Declared residue, see the kernel header.

One launch on ctx.stream, no host sync, no per-step allocation. Arrays QS, QG and F_RIMEF are
the NMMB names from parm/nam_solver_state.txt (3D real, H points), allocated by namgpu.state.State
as float32 [LM, JM, IM] (state.shape_of, rank 3).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_rime_factor_update"
KERNEL_BEFORE = "rime_factor_before"
KERNEL_AFTER = "rime_factor_after"


def advect_enabled(cfg) -> bool:
    """RIME_FACTOR_ADVECT: MICROPHYSICS == 'fer_hires' .AND. F_QG .AND. SPEC_ADV (2885-2890)."""
    return str(cfg.microphysics).strip() == "fer_hires" and bool(cfg.raw.get("spec_adv", False))


def _launch(ctx, kernel, args):
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    kern = kernels.get(UNIT, kernel)
    grid = kernels.grid_2d(im, jm, lm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_2D, (np.int32(lm), np.int32(im), np.int32(jm)) + args)


def run(ctx, branch: str = "after"):
    """Plan entry dispatch. branch='after' is the RIME_FACTOR_INPUT=.FALSE. call (4590);
    branch='before' is the RIME_FACTOR_INPUT=.TRUE. call (4448). Both are gated the same way."""
    if branch == "before":
        return run_before(ctx)
    if branch != "after":
        raise ValueError(f"drv-rime-factor-update: branch must be 'before' or 'after', got {branch!r}")
    if not advect_enabled(ctx.cfg):
        return
    st = ctx.st
    _launch(ctx, KERNEL_AFTER, (st.QS, st.QG, st.F_RIMEF))


def run_before(ctx):
    """RIME_FACTOR_INPUT = .TRUE.: QG = QS * F_RIMEF (before species advection)."""
    if not advect_enabled(ctx.cfg):
        return
    st = ctx.st
    _launch(ctx, KERNEL_BEFORE, (st.QS, st.F_RIMEF, st.QG))
