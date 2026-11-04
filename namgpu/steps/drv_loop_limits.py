"""drv-loop-limits: looplimits and the tile loop limits (its_b1 ... jte_b1_h2) as a device table.

Ported from nmm/module_DM_PARALLEL.F90:1941-1969 (looplimits), :777-822 (halo-inclusive bounds
and tile limits) and :23-24 (ihalo = jhalo = 3), NAM nam.v4.2.11, 83df4e13c1.
Kernel:  namgpu/kernels/drv_loop_limits.cu  (drv_loop_limits, one thread, one launch).
Writes:  ctx.tables["NMMB_LIMITS"] int32 [NLIM], 1-based Fortran values, indices DL_* in
         namgpu/kernels/nmmb_common.cuh (every tile-limit consumer reads the table, not macros).
Reads:   ctx.grid.im, ctx.grid.jm (namelist IM, JM: config.py sets grid.im = raw "im"; the
         global +2 is applied only in the kernel, so no double count); ctx.cfg.raw["global"];
         ctx.decomp = (mype, inpes, jnpes) when set, else one task (0, 1, 1).
Gap:     ntasks > 1 is refused. The Fortran tile of each task is the ESMF computationalCount
         taken by DECOMP in module_DOMAIN_GRID_COMP.F90 (CALL DECOMP ~line 4815), not the
         looplimits block rule; that ESMF rule is not ported yet. One card is exact.
table(ctx) is the entry point for consumers: it builds the table once (no host sync) and reuses it.
host_limits() is the same arithmetic on the host.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_loop_limits"
KERNEL = "drv_loop_limits"
NLIM = 42
TABLE_KEY = "NMMB_LIMITS"
IHALO = 3   # module_DM_PARALLEL.F90:23
JHALO = 3   # module_DM_PARALLEL.F90:24

# Index order must match the DL_* defines in kernels/nmmb_common.cuh.
NAMES = (
    "ids", "ide", "jds", "jde", "its", "ite", "jts", "jte",
    "its_b1", "ite_b1", "its_b2", "ite_b2", "jts_b1", "jte_b1", "jts_b2", "jte_b2",
    "its_h1", "ite_h1", "its_h2", "ite_h2", "jts_h1", "jte_h1", "jts_h2", "jte_h2",
    "its_b1_h1", "ite_b1_h1", "ite_b1_h2", "jts_b1_h1", "jte_b1_h1", "jte_b1_h2",
    "ids_p1", "ide_m1", "ide_m2", "jds_p1", "jde_m1", "jde_m2",
    "ihalo", "jhalo", "ims", "ime", "jms", "jme",
)
assert len(NAMES) == NLIM


def _looplimits(taskid: int, ntasks: int, lb: int, ub: int) -> tuple[int, int]:
    """Fortran looplimits(taskid, ntasks, lb, ub, i1, i2), taskid 0-based."""
    nwork = ub - lb + 1
    chunk = nwork // ntasks
    nt1 = nwork - ntasks * chunk
    netdisp = lb
    disp, cnt = lb, chunk
    for itask in range(1, ntasks + 1):
        c = chunk + 1 if itask <= nt1 else chunk
        if itask == taskid + 1:
            disp, cnt = netdisp, c
        netdisp = min(ub, netdisp + c)
    return disp, min(ub, disp + cnt - 1)


def host_limits(im: int, jm: int, global_flag: bool = False,
                mype: int = 0, inpes: int = 1, jnpes: int = 1) -> dict[str, int]:
    """Host mirror of the kernel: {name: value} for every DL_* index."""
    ids, jds = 1, 1
    ide = im + 2 if global_flag else im
    jde = jm + 2 if global_flag else jm
    its, ite = _looplimits(mype % inpes, inpes, ids, ide)
    jts, jte = _looplimits(mype // inpes, jnpes, jds, jde)
    v = {
        "ids": ids, "ide": ide, "jds": jds, "jde": jde,
        "its": its, "ite": ite, "jts": jts, "jte": jte,
        "ids_p1": max(its, ids + 1), "ide_m1": min(ite, ide - 1), "ide_m2": min(ite, ide - 2),
        "jds_p1": max(jts, jds + 1), "jde_m1": min(jte, jde - 1), "jde_m2": min(jte, jde - 2),
        "its_b1": max(its, ids + 1), "ite_b1": min(ite, ide - 1),
        "its_b2": max(its, ids + 2), "ite_b2": min(ite, ide - 2),
        "jts_b1": max(jts, jds + 1), "jte_b1": min(jte, jde - 1),
        "jts_b2": max(jts, jds + 2), "jte_b2": min(jte, jde - 2),
        "its_h1": max(its - 1, ids), "ite_h1": min(ite + 1, ide),
        "its_h2": max(its - 2, ids), "ite_h2": min(ite + 2, ide),
        "jts_h1": max(jts - 1, jds), "jte_h1": min(jte + 1, jde),
        "jts_h2": max(jts - 2, jds), "jte_h2": min(jte + 2, jde),
        "its_b1_h1": max(its - 1, ids + 1), "ite_b1_h1": min(ite + 1, ide - 1),
        "ite_b1_h2": min(ite + 2, ide - 1),
        "jts_b1_h1": max(jts - 1, jds + 1), "jte_b1_h1": min(jte + 1, jde - 1),
        "jte_b1_h2": min(jte + 2, jde - 1),
        "ihalo": IHALO, "jhalo": JHALO,
        "ims": its - IHALO, "ime": ite + IHALO,
        "jms": jts - JHALO, "jme": jte + JHALO,
    }
    return {k: int(v[k]) for k in NAMES}


def _decomp(ctx) -> tuple[int, int, int]:
    d = getattr(ctx, "decomp", None)
    return (0, 1, 1) if d is None else (int(d[0]), int(d[1]), int(d[2]))


def run(ctx):
    """Build the limit table on the device (one launch, no host sync)."""
    import cupy as cp

    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    global_flag = 1 if bool(ctx.cfg.raw.get("global", False)) else 0
    mype, inpes, jnpes = _decomp(ctx)
    if inpes * jnpes != 1:
        raise RuntimeError(
            f"drv-loop-limits: ntasks={inpes * jnpes} refused. The multi-task tile is the ESMF "
            "computationalCount from DECOMP (module_DOMAIN_GRID_COMP.F90 ~4815), which is not "
            "ported; looplimits() is not that rule. Single card only until it is.")
    if TABLE_KEY not in ctx.tables:
        ctx.tables[TABLE_KEY] = cp.zeros((NLIM,), dtype=cp.int32)
    table = ctx.tables[TABLE_KEY]
    fn = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        fn((1, 1, 1), (1, 1, 1),
           (np.int32(im), np.int32(jm), np.int32(global_flag),
            np.int32(mype), np.int32(inpes), np.int32(jnpes), table))
    return table


def table(ctx):
    """The limit table for this run: built once, then reused by every consumer."""
    if TABLE_KEY in ctx.tables:
        return ctx.tables[TABLE_KEY]
    return run(ctx)
