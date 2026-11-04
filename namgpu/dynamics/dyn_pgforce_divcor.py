"""dyn-pgforce-divcor: divergence correction (Janjic 1974/1979) of the pressure-gradient force.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:290-451 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_pgforce_divcor.cu (one thread per (i, j) point, level in blockIdx.z).

Reads : PGX_3D, PGY_3D, PGNE_3D, PGNW_3D (pgf scratch of dyn-pgforce-pgf, [LM, JM, IM]) from
        ctx.st; wpdar [JM] from ctx.tables["grid"]["wpdar"] (init-grid-consts, device array);
        tile limits and boundary flags from ctx.tables["NMMB_LIMITS"] (drv-loop-limits, device
        table read inside the kernel, so no host sync).
Writes: DIV [LM, JM, IM] in ctx.st. Cells the routine does not write keep their previous value.
Branch: ctx.cfg.global_ selects the global or the regional row set (the NAM 12 km parent is
        regional).
One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.steps import drv_loop_limits as dll

UNIT = "dyn_pgforce_divcor"
KERNEL = "dyn_pgforce_divcor"
LIMITS_KEY = "NMMB_LIMITS"


def _check_limits(ctx, limits, im, jm):
    """Once, at the first call (one host read of 36 ints, not on the step path): the kernel's
    tile must be the state arrays. Global needs the ghost-line array shape of
    module_DM_PARALLEL.F90:662-664 (ide=IM+2, jde=JM+2), which this port does not allocate, so
    the branch is refused rather than run against the wrong bounds."""
    host = limits.get()
    v = {n: int(host[i]) for i, n in enumerate(dll.NAMES)}
    if ctx.cfg.global_:
        raise RuntimeError(
            "dyn-pgforce-divcor: global = .true. is refused. drv-loop-limits gives ide=%d, jde=%d "
            "(module_DM_PARALLEL.F90:662-664 ghost-line convention) but the state arrays are %d x %d "
            "and this port does not allocate ghost lines (NAM runs global = .false., CONTRACT.md)"
            % (v["ide"], v["jde"], im, jm))
    _, jmg = slab.rows(ctx)
    if (v["ids"], v["jds"], v["ide"], v["jde"]) != (1, 1, im, jmg):
        raise RuntimeError(
            "dyn-pgforce-divcor: tile bounds ids=%d jds=%d ide=%d jde=%d do not match the state "
            "arrays %d x %d" % (v["ids"], v["jds"], v["ide"], v["jde"], im, jm))
    ctx.tables["DIVCOR_LIMITS_CHECKED"] = True


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    grid_tab = ctx.tables.get("grid")
    if grid_tab is None or grid_tab.get("wpdar") is None:
        raise RuntimeError(
            "dyn-pgforce-divcor: wpdar is not in ctx.tables['grid']; init-grid-consts must run "
            "first (module_CONTROL.F90 grid_consts writes the row weights there)")
    wpdar = grid_tab["wpdar"]

    limits = ctx.tables.get(LIMITS_KEY)
    if limits is None:
        raise RuntimeError(
            "dyn-pgforce-divcor: tile limits not built; drv-loop-limits must run first "
            "(ctx.tables['NMMB_LIMITS'], module_DM_PARALLEL looplimits)")
    if "DIVCOR_LIMITS_CHECKED" not in ctx.tables:
        _check_limits(ctx, limits, im, jm)

    global_flag = 1 if ctx.cfg.global_ else 0
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        limits,
        st["PGX_3D"], st["PGY_3D"], st["PGNE_3D"], st["PGNW_3D"],
        wpdar, st["DIV"],
        np.int32(im), np.int32(jm), np.int32(lm), np.int32(global_flag), *slab.kargs(ctx),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
