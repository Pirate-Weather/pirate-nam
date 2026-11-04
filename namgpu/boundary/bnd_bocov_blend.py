"""bnd-bocov-blend: write the V-point boundary values into the domain V fields, blended over the LNSV
outer rows and columns, then the boundary smoothing passes (nsmud >= 1).

Ported from nmm/module_FLTBNDS.F90:6590-7010, the blend and smoothing block of subroutine BOCOV
(NAM nam.v4.2.11, 83df4e13c1). Kernel: namgpu/kernels/bnd_bocov_blend.cu (bocov_blend_side,
bocov_smooth_gather, bocov_smooth_scatter). The wv weights are BOCOV lines 6461-6464, rebuilt here
in single precision (the update unit bnd-bocov-update does not bind them).

Reads : bnd_bocov_view.v_view(ctx).var_2d[n] and .var_3d[n] (view of ctx.bc and ctx.st), each with
          .full                       domain V-point field: [JM, IM] (2-D) or [LM, JM, IM] (3-D),
                                      float32 CuPy, aliasing the ctx.st array (U, V, ...)
          .south, .north              [2, LNSV, IM] (2-D) / [2, LM, LNSV, IM] (3-D)
          .west, .east                [2, JM, LNSV] (2-D) / [2, LM, JM, LNSV] (3-D)
        slot 0 of each strip is the value (Fortran slot 1, the time level used here); slot 1 is the
        tendency and is not read. ctx.cfg.lnsv (5), ctx.grid.im/jm/lm.
Writes: .full of every variable, in place (the blended values; the smoothing passes when nsmud >= 1).

Order (as the Fortran): South, North, West, East; within each side the 2-D variables, then the 3-D.
The sides overlap only at corners, so this order fixes the corner values.
Loop limits: single full domain (ids = jds = 1, ide = IM, jde = JM, its = ts = 1); see the .cu header.
Launches: one per (side, variable) for the blend, four regions times two kernels per variable per
smoothing pass when nsmud >= 1. No host sync, no per-step allocation: the wv vector is cached in
ctx.tables and the smoothing scratch is allocated on the first smoothing call only.
The Fortran sets nsmud = 0 (BOCOV line 6459), so NAM never runs the smoothing branch; run(ctx) with
the default nsmud = 0 is the NAM path.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.boundary import bnd_bocov_view

UNIT = "bnd_bocov_blend"
FACES = ("south", "north", "west", "east")  # Fortran order: S, N, W, E (side index 0..3)


def _wv_host(lnsv: int) -> np.ndarray:
    """BOCOV lines 6461-6464: wv(1) = 1; wv(k) = 1 - (0.9/real(lnsv-1))*(k-1), single precision."""
    wv = np.empty(lnsv, dtype=np.float32)
    wv[0] = np.float32(1.0)
    c = np.float32(0.9) / np.float32(lnsv - 1)
    for k in range(2, lnsv + 1):
        wv[k - 1] = np.float32(1.0) - c * np.float32(k - 1)
    return wv


def _regions(im: int, jm: int, lines: int):
    """Smoothing regions (1-based inclusive i1, i2, j1, j2) from the BOCOV smooth block, single process.

    The East entry repeats the West one: the Fortran East block uses i = ite-lines..ite-2 as well.
    """
    west = (im - lines, im - 2, max(1, 1 + lines), min(jm, jm - lines - 1))
    return {
        "S": (2, im - 2, 2, lines),
        "N": (2, im - 2, jm - lines, jm - 2),
        "W": west,
        "E": west,
    }


def _grid(nx: int, ny: int, nz: int):
    bx, by, _ = kernels.BLOCK_2D
    return ((nx + bx - 1) // bx, (ny + by - 1) // by, nz)


def _check(arr, shape, name):
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"bnd-bocov-blend: {name} has shape {tuple(arr.shape)}, expected {tuple(shape)}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"bnd-bocov-blend: {name} is not C-contiguous")


def _blend_side_vars(group, face: str, three_d: bool, lm: int, im: int, jm: int, lnsv: int):
    """Shape checks for one side and one variable group; yields (full, strip) pairs."""
    lev = (lm,) if three_d else ()
    full_shape = lev + (jm, im)
    for v in group:
        full = v.full
        strip = getattr(v, face)
        _check(full, full_shape, f"{face} full_var")
        if face in ("south", "north"):
            _check(strip, (2,) + lev + (lnsv, im), f"{face} strip")
        else:
            _check(strip, (2,) + lev + (jm, lnsv), f"{face} strip")
        yield full, strip


def run(ctx, nsmud: int = 0):
    import cupy as cp

    bv = bnd_bocov_view.v_view(ctx)
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsv = int(ctx.cfg.lnsv)
    block = kernels.BLOCK_2D

    wv_key = f"bocov_wv_{lnsv}"
    if wv_key not in ctx.tables:
        ctx.tables[wv_key] = cp.asarray(_wv_host(lnsv))
    wv = ctx.tables[wv_key]

    blend = kernels.get(UNIT, "bocov_blend_side")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    s_bdy, n_bdy = slab.edges(ctx)         # row slab: S strips on the first rank, N on the last
    if nsmud >= 1 and slab.of(ctx) is not None:
        raise RuntimeError("bnd-bocov-blend: nsmud >= 1 smooths by global row regions (_regions) and needs a "
                           "halo exchange between iterations; not built for row slabs (NAM runs nsmud = 0)")
    with scope:
        # ---- blend: S, N, W, E; each side does the 2-D variables, then the 3-D ----
        for side, face in enumerate(FACES):
            if (side == 0 and not s_bdy) or (side == 1 and not n_bdy):
                continue
            if side < 2:                       # S, N: thread (i, jb) over IM x LNSV
                P, Q = im, lnsv
            else:                              # W, E: thread (ib, j) over LNSV x JM
                P, Q = lnsv, jm
            for three_d, group, nl in ((False, bv.var_2d, 1), (True, bv.var_3d, lm)):
                for full, strip in _blend_side_vars(group, face, three_d, lm, im, jm, lnsv):
                    blend(_grid(P, Q, nl), block,
                          (full, strip, wv, np.int32(side), np.int32(im), np.int32(jm),
                           np.int32(nl), np.int32(lnsv), *slab.kargs(ctx)))

        # ---- smoothing (BOCOV `if(nsmud>=1)`): dead in NAM (nsmud = 0), ported in full ----
        if nsmud >= 1:
            if "bocov_smooth_tmp" not in ctx.tables:
                ctx.tables["bocov_smooth_tmp"] = cp.empty((lm, jm, im), dtype=cp.float32)
            tmp = ctx.tables["bocov_smooth_tmp"]
            gather = kernels.get(UNIT, "bocov_smooth_gather")
            scatter = kernels.get(UNIT, "bocov_smooth_scatter")
            regs = _regions(im, jm, lnsv)
            for _ks in range(int(nsmud)):
                for name in ("S", "N", "W", "E"):
                    i1, i2, j1, j2 = regs[name]
                    for three_d, group, nl in ((False, bv.var_2d, 1), (True, bv.var_3d, lm)):
                        grid = _grid(i2 - i1 + 1, j2 - j1 + 1, nl)
                        for v in group:
                            gather(grid, block, (tmp, v.full, i1, i2, j1, j2, im, jm, nl))
                            scatter(grid, block, (v.full, tmp, i1, i2, j1, j2, im, jm, nl))
