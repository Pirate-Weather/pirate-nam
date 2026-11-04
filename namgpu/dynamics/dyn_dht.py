"""dyn-dht: horizontal mass divergence, pressure-advection term of the T equation and omega-alpha.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:498-796 (NAM nam.v4.2.11, 83df4e13c1), regional case.
Kernels: namgpu/kernels/dyn_dht.cu  (dht_flux, dht_div, dht_cum).

Reads : U, V [LM,JM,IM]; PD, PDO [JM,IM]; PCX, PCY, PCNE, PCNW [LM,JM,IM] (dyn-pgforce-pgf);
        DIV [LM,JM,IM] (inout, the Fortran intent(inout) div); DSG2, PDSG1 [LM] (grid);
        DXV [JM] (grid); DYH = dyv (grid); FCP, FDIV [JM] (ctx.tables['grid'], owned by init-grid-consts).
Refuses global=.true. (global path not ported).
Writes: DIV (in place), OMGALF [LM,JM,IM] (state, H), PFX, PFY, PFNE, PFNW [LM,JM,IM] and TDIV
        [LM,JM,IM] (scratch, added once with st.add; TDIV is the name dyn-pdtsdt reads).
Note  : the Fortran 2-D pre-pass (pdx, pdy, pdne, pdnw) is evaluated inside dht_flux; the Fortran
        2-D level temporaries tx, ty, tne, tnw are recomputed where dht_div reads them. Neither is
        stored, so no extra VRAM beyond the outputs above.
Launch: three kernels on ctx.stream, no host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_dht"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        if host is None:
            raise RuntimeError(f"dyn-dht: host vector {key!r} is not set on the grid")
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _row_table(ctx, key: str):
    """FCP and FDIV are GRID_CONSTS row arrays owned by init-grid-consts. init-grid-consts publishes
    them only inside ctx.tables["grid"] (its out dict), so they are read from there."""
    grid_tab = ctx.tables.get("grid")
    if grid_tab is None or key not in grid_tab:
        raise RuntimeError(
            f"dyn-dht: row constant {key!r} missing from ctx.tables['grid'] (produced by init-grid-consts)")
    return grid_tab[key]


def _require_field(st, name: str):
    """PCX, PCY, PCNE, PCNW are written by dyn-pgforce-pgf earlier in the step. A missing field is a
    step-order error, not a zero field, so it fails instead of silently allocating zeros."""
    if name not in st.fields:
        raise RuntimeError(
            f"dyn-dht: {name} is not set; dyn-pgforce-pgf must run before dyn-dht in the step order")
    return st.fields[name]


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)
    if bool(ctx.cfg.global_):
        # The CUDA kernels implement only the regional zero-divergence ring (Fortran dht, global=.false.).
        # The global path needs periodic/halo values for i=1 and i=IM in the pfx/pfne reads, not ported.
        raise RuntimeError("dyn-dht: global=.true. is not ported; the regional ring is hard-coded in "
                           "dht_div, so a global config would silently give regional results")

    pfx = st.add("PFX", shape3)
    pfy = st.add("PFY", shape3)
    pfne = st.add("PFNE", shape3)
    pfnw = st.add("PFNW", shape3)
    tdiv = st.add("TDIV", shape3)

    dsg2 = _device_vector(ctx, "dsg2", g.dsg2)
    pdsg1 = _device_vector(ctx, "pdsg1", g.pdsg1)
    dxv = _device_vector(ctx, "dxv", g.dxv)
    pcx = _require_field(st, "PCX")
    pcy = _require_field(st, "PCY")
    pcne = _require_field(st, "PCNE")
    pcnw = _require_field(st, "PCNW")
    fcp = _row_table(ctx, "fcp")
    fdiv = _row_table(ctx, "fdiv")

    dyv = np.float32(g.dyh)
    k_flux = kernels.get(UNIT, "dht_flux")
    k_div = kernels.get(UNIT, "dht_div")
    k_cum = kernels.get(UNIT, "dht_cum")
    g2 = kernels.grid_2d(im, jm, lm)
    gc = kernels.grid_cols(im, jm)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_flux(g2, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm), dyv,
            dsg2, pdsg1, dxv, st.PD, st.PDO, st.U, st.V,
            pfx, pfy, pfne, pfnw,
        ))
        k_div(g2, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm), dyv,
            dsg2, pdsg1, dxv, fcp, fdiv, st.PD, st.U, st.V,
            pcx, pcy, pcne, pcnw,
            pfx, pfy, pfne, pfnw,
            st.DIV, tdiv, st.OMGALF,
        ))
        k_cum(gc, kernels.BLOCK_COL, (np.int32(im), np.int32(jm), np.int32(lm), tdiv))
