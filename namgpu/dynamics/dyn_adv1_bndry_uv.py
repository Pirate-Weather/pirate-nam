"""dyn-adv1-bndry-uv: upstream advection of momentum along the four lateral boundaries.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:2022-2186 (NAM nam.v4.2.11, 83df4e13c1), the regional
branch of ADV1 (`if(.not.global.and.adv_upstream)`).  Accumulates into TCU and TCV.
Kernel: namgpu/kernels/dyn_adv1_bndry_uv.cu (dyn_adv1_bndry_uv, one thread per (i, j, l)).

Reads : U, V [LM,JM,IM] from ctx.st; RDXV [JM] (1/dxv, device copy cached in ctx.tables);
        RDYV scalar (1/dyv); dt from ctx.dt; lnsad = cfg.lnsh + 2 (module_CONTROL.F90 regional).
Writes: TCU, TCV [LM,JM,IM] in ctx.st (inout: the upstream terms are added to the existing values).
One launch on ctx.stream, no host sync, no per-step allocation (RDXV uploaded once).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab

UNIT = "dyn_adv1_bndry_uv"
KERNEL = "dyn_adv1_bndry_uv"


def _adv_upstream(im: int, jm: int, lnsad: int) -> bool:
    """module_CONTROL.F90:264-270 with single-process its=ids=1, ite=ide=IM, jts=jds=1, jte=jde=JM."""
    its, ite, ids, ide = 1, im, 1, im
    jts, jte, jds, jde = 1, jm, 1, jm
    return (jts < jds + 1 + lnsad or jte >= jde - 1 - lnsad
            or its < ids + 1 + lnsad or ite >= ide - 1 - lnsad)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


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
            f"dyn-adv1-bndry-uv: domain {im}x{jm} too small for lnsad={lnsad}")

    # RDXV(j) = 1./DXV(j) (module_CONTROL.F90 regional rows); RDYV = 1./DYV (single precision).
    dxv = np.asarray(g.dxv, dtype=np.float32)
    rdxv = _device_vector(ctx, "rdxv", np.float32(1.0) / dxv)
    rdyv = np.float32(np.float32(1.0) / np.float32(g.dyv))

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.int32(lnsad),
        np.float32(ctx.dt),
        rdyv,
        rdxv,
        st.U, st.V, st.TCU, st.TCV,
        *slab.kargs(ctx),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, kernels.BLOCK_2D, args)
