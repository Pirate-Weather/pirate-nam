"""dyn-hdiff-apply: second-order horizontal diffusion of T, Q, CW, Q2, U, V behind the mass points.

NOTICE: NOAA/NCEP NMMB source (US government work); see NOTICE at the repo root.
Ported from nmm/module_DYNAMICS_ROUTINES.F90:2834-2915 (HDIFF apply section, nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/dyn_hdiff_apply.cu (dyn_hdiff_apply_diff, dyn_hdiff_apply_update).

Reads : T, Q, CW, Q2, U, V, HDACX, HDACY, HDACVX, HDACVY from ctx.st;
        DEF3D (deformation) and FMLX_3D / FMLY_3D (slope masks) from ctx.st, both produced by
        dyn-hdiff-coef earlier in the same step (the masks are not recomputed here).
        Nothing else: the stencil needs only def3d and the masks, so no grid, LPT2 or hydro input.
Writes: T, Q, CW, Q2, U, V in ctx.st (in place, as the Fortran does).
Scratch: six [LM, JM, IM] diff arrays in ctx.tables, allocated once (no per-step allocation).

Two launches on ctx.stream, no host sync, one wrapper call. This is the split CONTRACT.md section 3
allows for a routine that is split: the diff pass must finish reading the old fields before the
update pass writes them (tdif(i,j) reads t(i+1,j)). The FACDIF = 1.0 constant of the regional
branch is passed to the update launch.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_hdiff_apply"
DIFF_KERNEL = "dyn_hdiff_apply_diff"
UPDATE_KERNEL = "dyn_hdiff_apply_update"
FACDIF = np.float32(1.0)   # Fortran 2659-2662: facdif = 1.0 in both the global and regional branch


def _diff_scratch(ctx, lm, jm, im):
    import cupy as cp
    out = {}
    for name in ("tdif", "qdif", "cdif", "q2dif", "udif", "vdif"):
        key = f"hdiff_{name}"
        if key not in ctx.tables:
            ctx.tables[key] = cp.zeros((lm, jm, im), dtype=cp.float32)
        out[name] = ctx.tables[key]
    return out


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    def3d = st["DEF3D"]
    fmlx = st["FMLX_3D"]
    fmly = st["FMLY_3D"]
    s = _diff_scratch(ctx, lm, jm, im)

    diff = kernels.get(UNIT, DIFF_KERNEL)
    update = kernels.get(UNIT, UPDATE_KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        diff(grid, block, (
            np.int32(lm), np.int32(im), np.int32(jm), fmlx, fmly,
            st["T"], st["Q"], st["CW"], st["Q2"], st["U"], st["V"], def3d,
            st["HDACX"], st["HDACY"], st["HDACVX"], st["HDACVY"],
            s["tdif"], s["qdif"], s["cdif"], s["q2dif"], s["udif"], s["vdif"],
        ))
        update(grid, block, (
            np.int32(lm), np.int32(im), np.int32(jm), FACDIF,
            st["T"], st["Q"], st["CW"], st["Q2"], st["U"], st["V"],
            s["tdif"], s["qdif"], s["cdif"], s["q2dif"], s["udif"], s["vdif"],
        ))
