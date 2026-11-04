"""dyn-adv1-bndry-t: regional upstream advection of T along the four lateral boundary strips.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1571-1705 (NAM nam.v4.2.11, 83df4e13c1), the
`if(.not.global.and.adv_upstream)` branch inside ADV1's vertical_loop1.
Kernel: namgpu/kernels/dyn_adv1_bndry_t.cu (one thread per (i, j), level in grid.z).

Reads:   U, V, T, OMGALF [LM,JM,IM] from ctx.st; RDXH [JM] and RDYH from ctx.grid
         (RDXH = 1/dxh(j) on the regional branch, module_CONTROL.F90:926; cached in ctx.tables).
Writes:  TCT [LM,JM,IM] in ctx.st, accumulated in place on the boundary strips only.
Scalars: lnsad = lnsh + 2 (module_CONTROL.F90:260), dt from ctx.dt, RDYH = 1/dyh.
One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels, slab

UNIT = "dyn_adv1_bndry_t"
KERNEL = "dyn_adv1_bndry_t"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _adv_upstream(im: int, jm: int, lnsad: int) -> bool:
    """module_CONTROL.F90:264-270 with single-process its=ids=1, ite=ide=IM, jts=jds=1, jte=jde=JM."""
    its, ite, ids, ide = 1, im, 1, im
    jts, jte, jds, jde = 1, jm, 1, jm
    return (jts < jds + 1 + lnsad or jte >= jde - 1 - lnsad
            or its < ids + 1 + lnsad or ite >= ide - 1 - lnsad)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsad = int(ctx.cfg.lnsh) + 2
    if ctx.cfg.global_:
        return  # Fortran: `if(.not.global.and.adv_upstream)`
    if not _adv_upstream(im, jm, lnsad):
        return
    jmg = slab.rows(ctx)[1]
    if im <= 2 * lnsad + 1 or jmg <= 2 * lnsad + 1:
        raise RuntimeError(
            f"dyn-adv1-bndry-t: domain {im}x{jm} too small for lnsad={lnsad}; the four "
            "boundary strips would overlap and the single-pass kernel would double-count")

    rdxh = _device_vector(ctx, "rdxh", 1.0 / np.asarray(g.dxh, dtype=np.float64))
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    with ctx.stream:
        kern(grid, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm),
            np.int32(lnsad),
            np.float32(ctx.dt),
            np.float32(1.0 / g.dyh),
            rdxh,
            st["U"], st["V"], st["T"], st["OMGALF"], st["TCT"],
            *slab.kargs(ctx),
        ))
