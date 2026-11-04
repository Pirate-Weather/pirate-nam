"""dyn-cdzdt-advect-z: grand horizontal loop of CDZDT (height advection into W, w = 0 on the boundaries).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3140-3235, subroutine CDZDT (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdzdt_advect_z.cu (dyn_cdzdt_advect_z, one thread per (i, j), level in grid.z).

Reads : Z [LM+1,JM,IM] (dyn-cdzdt-nonhydro), PFX, PFY, PFNE, PFNW [LM,JM,IM] (dyn-pgforce-pgf),
        PDO [JM,IM], DSG2 [LM] from ctx.st; FAH [JM] from ctx.tables[grid]; PDSG1 [LM] from ctx.grid (device copies cached in
        st.fields, uploaded once).
Writes: W [LM,JM,IM] in ctx.st, in place (W += advection term; then w = 0 on the lateral boundary strips).
Flags : s_bdy, n_bdy, w_bdy, e_bdy are all 1 and global is 0 (one process owns the full regional domain;
        module_CONTROL.F90:542-545). One launch on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdzdt_advect_z"
KERNEL = "dyn_cdzdt_advect_z"


def _grid_const(st, grid, name: str, attr: str):
    """Upload a host grid constant once and cache it in the state (never per step)."""
    if name in st.fields:
        return st.fields[name]
    host = getattr(grid, attr, None)
    if host is None:
        raise RuntimeError(f"dyn-cdzdt-advect-z: grid constant {attr!r} is not set on the grid object")
    import cupy as cp
    st.fields[name] = cp.asarray(np.asarray(host, dtype=np.float32))
    return st.fields[name]


def _fah_row(ctx):
    """FAH [JM] device row from init-grid-consts (ctx.tables["grid"]["fah"], module_CONTROL.F90:614-741).

    fah is a grid row, not a BGrid attribute, so it is read from the init-grid-consts output table.
    """
    grid = ctx.tables.get("grid")
    if grid is None or "fah" not in grid:
        raise RuntimeError("dyn-cdzdt-advect-z: ctx.tables[grid][fah] missing; produced by unit init-grid-consts")
    return grid["fah"]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    pfx = st.add("PFX", (lm, jm, im))
    pfy = st.add("PFY", (lm, jm, im))
    pfne = st.add("PFNE", (lm, jm, im))
    pfnw = st.add("PFNW", (lm, jm, im))
    fah = _fah_row(ctx)
    pdsg1 = _grid_const(st, g, "PDSG1", "pdsg1")

    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
             (st.Z, pfx, pfy, pfne, pfnw, fah, st.DSG2, pdsg1, st.PDO, st.W,
              np.float32(ctx.dt), np.int32(im), np.int32(jm), np.int32(lm),
              np.int32(0),                       # global = false (regional)
              np.int32(1), np.int32(1), np.int32(1), np.int32(1)))  # s/n/w/e boundary, full domain
