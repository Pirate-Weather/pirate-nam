"""dyn-vsound: implicit vertically propagating sound waves (NMMB VSOUND).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3904-4234 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_vsound.cu (vsound_col, one thread per column, l-loops inside).

Reads : PD [JM,IM], CW, Q, RTOP, DWDT, T, W, BARO [JM,IM], PINT [LM+1,JM,IM] from ctx.st;
        DSG2 [LM] from ctx.st; PDSG1 [LM] from ctx.grid (device copy cached in ctx.tables).
Writes: PINT, W, T, DWDT (inout) and W_TOT (out) in ctx.st. RTOP and BARO are state-owned
        scratch added with st.add (written by dyn-pgforce-geopot / dyn-cdzdt-extmode).
Gates : the Fortran returns at once when hydro is on or ntsd < 2, so the wrapper launches nothing then.
        ntsd is abs(ctx.ntimestep) (the driver counts from 0: the first step is ntimestep 0, as Fortran NTSD is 0 on the first step).
Flags : single process, regional (global = false): all four BDY flags are true, so the boundary
        override [H] applies on every strip. Global runs pass global_flag = 1 and skip [H].
Launch: one kernel on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_vsound"
KERNEL = "vsound_col"
VS_LMAX = 64   # must match VS_LMAX in kernels/dyn_vsound.cu


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx):
    cfg = ctx.cfg
    if cfg.hydro:
        return
    if abs(int(ctx.ntimestep)) < 2:
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > VS_LMAX:
        raise RuntimeError(
            f"dyn-vsound: LM={lm} exceeds VS_LMAX={VS_LMAX} (per-column work arrays in the kernel)")
    if getattr(g, "pdsg1", None) is None:
        raise RuntimeError(
            "dyn-vsound: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    rtop = st.add("RTOP", (lm, jm, im))
    baro = st.add("BARO", (jm, im))
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    global_flag = 1 if cfg.global_ else 0

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.float32(ctx.dt), np.float32(g.pt),
        np.int32(1), np.int32(1), np.int32(1), np.int32(1),   # s/n/w/e boundary flags (regional, 1 task)
        np.int32(global_flag),
        st["DSG2"], pdsg1, st["PD"], st["CW"], st["Q"], rtop, baro,
        st["DWDT"], st["T"], st["W"], st["W_TOT"], st["PINT"],
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
