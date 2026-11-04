"""dyn-mono-global-sum: global mass sums per species and the conservation rescale of tcs.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:5076-5203 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/dyn_mono_global_sum.cu
         mono_global_sum_reduce (one block per species) and mono_global_sum_rescale (grid_2d).

Reads:   MONO_XSUMS [2*NTR] float64 in ctx.tables (written by dyn-mono-fix, the use_allreduce branch;
         the same buffer, not a copy), TRACERS_SQRT [NTR,LM,JM,IM] (S1 of the Fortran MONO, from dyn-mono-fix),
         PD, DSG2 (state), PDSG1 [LM] (ctx.grid.pdsg1), DARE [JM] (ctx.tables["grid"]["dare"], built by
         init-grid-consts from module_CONTROL grid_consts, dare(j) = dxh(j)*dyh).
Writes:  TRACERS_TEND [NTR,LM,JM,IM] (inout; the tracer tendency the Fortran MONO rescales as its tcs argument,
         and the one dyn-updates applies).
         GSUMS [2*NTR] and SFAC [2*NTR] double scratch in ctx.tables (device).
Branch:  both branches are ported, chosen by cfg.raw["use_allreduce"] (the same flag dyn-mono-fix reads).
         Allreduce: gsums = MONO_XSUMS (identity on one card). Send/recv: gsums from the per-column sums of
         TRACERS_SQRT, which dyn-mono-fix wrote as dsp on that branch.
Species: kss..kse (1-based, 1..NTR by default for standalone calls). The driver passes the same range as the
         MONO call that owns it (driver.py plan rows; SOLVER_RUN 4370 and 4519).
Two launches, no host sync.

Missing input is an error, not a default: DARE, MONO_XSUMS and TRACERS_SQRT must already exist (their producers run
first in the plan). Nothing is zero-filled here.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_mono_global_sum"
RED_KERNEL = "mono_global_sum_reduce"
RESCALE_KERNEL = "mono_global_sum_rescale"
RED_BLOCK = (256, 1, 1)


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables (float32)."""
    if key not in ctx.tables:
        import cupy as cp
        if host is None:
            raise RuntimeError(f"dyn-mono-global-sum: {key} missing on the grid")
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _dare(ctx):
    """DARE row from init-grid-consts (ctx.tables["grid"]["dare"], a device float32 [JM] array)."""
    tab = ctx.tables.get("grid")
    if tab is None or tab.get("dare") is None:
        raise RuntimeError(
            "dyn-mono-global-sum: DARE row missing; init-grid-consts must run first "
            "(module_CONTROL grid_consts, dare(j)=dxh(j)*dyh)")
    return tab["dare"]


def run(ctx, kss: int = 1, kse: int | None = None):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntr = int(st["TRACERS"].shape[0])
    if kse is None:
        kse = ntr
    if not (1 <= kss <= kse <= ntr):
        raise ValueError(f"dyn-mono-global-sum: tracer range {kss}..{kse} outside 1..{ntr}")
    nspec = kse - kss + 1

    allreduce = bool(ctx.cfg.raw["use_allreduce"])
    from namgpu import slab as slabmod
    joff, jmg = slabmod.rows(ctx)

    dare = _dare(ctx)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    dsg2 = st["DSG2"]

    s1 = st.add("TRACERS_SQRT", (ntr, lm, jm, im))
    tcs = st.add("TRACERS_TEND", (ntr, lm, jm, im))
    xsums = ctx.tables.get("MONO_XSUMS")
    if xsums is None or xsums.shape != (2 * ntr,):
        raise RuntimeError(
            "dyn-mono-global-sum: MONO_XSUMS missing or wrong shape; dyn-mono-fix must run first "
            f"(expected [2*NTR]={2 * ntr})")  # dyn-mono-fix allocates it on both branches
    gsums = _double_scratch(ctx, "GSUMS", 2 * ntr)
    sfac = _double_scratch(ctx, "SFAC", 2 * ntr)

    red = kernels.get(UNIT, RED_KERNEL)
    resc = kernels.get(UNIT, RESCALE_KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        red((nspec, 1, 1), RED_BLOCK,
            (np.int32(kss), np.int32(kse), np.int32(im), np.int32(jm), np.int32(lm),
             np.int32(1 if allreduce else 0), xsums, s1, gsums, sfac))
        resc(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
             (np.int32(kss), np.int32(kse), np.int32(im), np.int32(jm), np.int32(lm),
              st["PD"], dsg2, pdsg1, dare, s1, sfac, tcs, np.int32(joff), np.int32(jmg)))


def _double_scratch(ctx, key: str, n: int):
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.zeros((n,), dtype=cp.float64)
    return ctx.tables[key]
