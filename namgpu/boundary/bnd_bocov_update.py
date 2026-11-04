"""bnd-bocov-update: advance the V-point boundary working objects by their tendencies.

Ported from nmm/module_FLTBNDS.F90:6392-6589, subroutine BOCOV (NAM nam.v4.2.11, 83df4e13c1), the
update block only: for every 2-D and 3-D V-point boundary variable, value(slot 1) += tendency
(slot 2) * dt on the south, north, west and east strips. Kernels: namgpu/kernels/bnd_bocov_update.cu
(bocov_sn_2d, bocov_sn_3d, bocov_we_2d, bocov_we_3d).

Reads : bnd_bocov_view.v_view(ctx).var_2d[n] and .var_3d[n] (the view of ctx.bc and ctx.st, see that module),
        float32 CuPy arrays with the Fortran indices reversed (CONTRACT section 1):
          south/north 2-D [2, LNSV, IM]       3-D [2, LM, LNSV, IM]
          west/east   2-D [2, JM, LNSV]       3-D [2, LM, JM, LNSV]
        slot 0 is the Fortran slot 1 (value), slot 1 the Fortran slot 2 (tendency).
        dt from ctx.dt; LNSV from ctx.cfg.lnsv; IM, JM, LM from ctx.grid.
Writes: the value slot of every strip array, in place.
Loop limits: single full domain (ids = jds = 1, ide = IM, jde = JM): S/N strips use ib = 1..IM-1,
W/E strips use jb = 1..JM-1, all LNSV boundary rows/columns. The wv weights and smoothing passes of
the routine (lines 6392-6589) are not used in this range (nsmud = 0), so they are not ported here.
Launches: 8 at most (S, N, W, E times 2-D and 3-D), each skipped when the variable list is empty.
No host sync, no per-step allocation (the pointer tables are built once per array set and cached).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels, slab
from namgpu.boundary import bnd_bocov_view

UNIT = "bnd_bocov_update"

# Full-domain edge flags: the regional 12 km domain is one tile that owns all four edges.
S_BDY = N_BDY = W_BDY = E_BDY = True

_PTR_CACHE: dict = {}


def _ptr_table(arrs):
    """Device table of the array base addresses (uint64), built once per array set."""
    import cupy as cp

    key = tuple(int(a.data.ptr) for a in arrs)
    tbl = _PTR_CACHE.get(key)
    if tbl is None:
        tbl = cp.asarray(np.array(key, dtype=np.uint64))
        _PTR_CACHE[key] = tbl
    return tbl


def _check_shape(arrs, shape, name):
    for a in arrs:
        if tuple(a.shape) != shape:
            raise ValueError(f"bnd-bocov-update: {name} has shape {tuple(a.shape)}, expected {shape}")


def run(ctx):
    bv = bnd_bocov_view.v_view(ctx)
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsv = int(ctx.cfg.lnsv)
    dt = np.float32(ctx.dt)
    block = kernels.BLOCK_2D

    def grid_for(n_along, nb, nz):
        bx, by, _ = block
        return ((n_along + bx - 1) // bx, (nb + by - 1) // by, nz)

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        # ---- S and N strips: ib = 1..IM-1, jb = 1..LNSV ----
        sn_faces = []
        if S_BDY:
            sn_faces.append("south")
        if N_BDY:
            sn_faces.append("north")
        for face in sn_faces:
            if bv.var_2d:
                arrs = [getattr(v, face) for v in bv.var_2d]
                _check_shape(arrs, (2, lnsv, im), f"{face} 2-D")
                tbl = _ptr_table(arrs)
                kernels.get(UNIT, "bocov_sn_2d")(
                    grid_for(im - 1, lnsv, len(arrs)), block,
                    (np.int32(len(arrs)), np.int32(lnsv), np.int32(im), np.int32(im - 1), dt, tbl),
                )
            if bv.var_3d:
                arrs = [getattr(v, face) for v in bv.var_3d]
                _check_shape(arrs, (2, lm, lnsv, im), f"{face} 3-D")
                tbl = _ptr_table(arrs)
                kernels.get(UNIT, "bocov_sn_3d")(
                    grid_for(im - 1, lnsv, len(arrs) * lm), block,
                    (np.int32(len(arrs)), np.int32(lm), np.int32(lnsv), np.int32(im),
                     np.int32(im - 1), dt, tbl),
                )

        # ---- W and E strips: jb = 1..JM-1, ib = 1..LNSV ----
        we_faces = []
        if W_BDY:
            we_faces.append("west")
        if E_BDY:
            we_faces.append("east")
        # row slab: jb = 1..JM-1 is the GLOBAL range; a rank below the top holds no row JMG
        n_we = jm - 1 if slab.edges(ctx)[1] else jm
        for face in we_faces:
            if bv.var_2d:
                arrs = [getattr(v, face) for v in bv.var_2d]
                _check_shape(arrs, (2, jm, lnsv), f"{face} 2-D")
                tbl = _ptr_table(arrs)
                kernels.get(UNIT, "bocov_we_2d")(
                    grid_for(n_we, lnsv, len(arrs)), block,
                    (np.int32(len(arrs)), np.int32(lnsv), np.int32(jm), np.int32(n_we), dt, tbl),
                )
            if bv.var_3d:
                arrs = [getattr(v, face) for v in bv.var_3d]
                _check_shape(arrs, (2, lm, jm, lnsv), f"{face} 3-D")
                tbl = _ptr_table(arrs)
                kernels.get(UNIT, "bocov_we_3d")(
                    grid_for(n_we, lnsv, len(arrs) * lm), block,
                    (np.int32(len(arrs)), np.int32(lm), np.int32(lnsv), np.int32(jm),
                     np.int32(n_we), dt, tbl),
                )
