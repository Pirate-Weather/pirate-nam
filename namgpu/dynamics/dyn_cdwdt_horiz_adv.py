"""dyn-cdwdt-horiz-adv: horizontal advection of w into DWDT (NMMB cdwdt, advection of w).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:3653-3727 (NAM nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/dyn_cdwdt_horiz_adv.cu (cdwdt_horiz_adv, one thread per (i, j, l), grid_2d).

Reads : W [LM,JM,IM], PFX, PFY, PFNE, PFNW [LM,JM,IM] (mass fluxes, written by dyn-dht; must already
        exist, no zero fill), PDO [JM,IM], DSG2 [LM] from ctx.st; PDSG1 [LM] from ctx.grid (cached in
        ctx.st.fields); FAH [JM] is the shared int_state%FAH array in ctx.st.fields["FAH"].
Writes: DWDT [LM,JM,IM] in ctx.st (inout: the horizontal advection increment is added to DWDT).
Scalars: dt = ctx.dt, the DYNAMICS dt argument of CDWDT (Fortran DT). It is signed in the TDFI backward
         leg, as in Fortran; outside a filter ctx.dt is cfg.dt. epsvw = 9999 (cdwdt parameter).
FAH: Fortran cdwdt reads the shared int_state%FAH, which the TDFI filter rescales in place
         (module_SOLVER_GRID_COMP.F90:3137-3170, 3208-3250, 3330-3344; see dyn-adv1-horiz-t). The same
         array is read here (seeded once from the init-grid-consts row ctx.tables["grid"]["fah"], the same
         seeding as dyn-adv2-horiz), so the kernel computes fahp = -FAH/DT with the phase's FAH and DT.
Launch: one kernel on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_cdwdt_horiz_adv"
KERNEL = "cdwdt_horiz_adv"
FAH_KEY = "FAH"          # shared with dyn-adv1-horiz-t and dyn-adv2-horiz (the TDFI-scaled array)
PF_PRODUCER = "dyn-dht"  # writes PFX, PFY, PFNE, PFNW (Fortran pfx/pfy/pfne/pfnw, DHT)

EPSVW = np.float32(9999.0)  # module_DYNAMICS_ROUTINES cdwdt parameter epsvw


def _grid_const(st, grid, name: str, attr: str):
    """Upload a host grid constant once and cache it in the state (never per step)."""
    if name in st.fields:
        return st.fields[name]
    host = getattr(grid, attr, None)
    if host is None:
        raise RuntimeError(f"dyn-cdwdt-horiz-adv: grid constant {attr!r} is not set on the grid object")
    import cupy as cp
    st.fields[name] = cp.asarray(np.asarray(host, dtype=np.float32))
    return st.fields[name]


def _need(st, name: str):
    """State field that must already exist (written by PF_PRODUCER); raises instead of zero filling."""
    try:
        return st.fields[name]
    except KeyError:
        raise RuntimeError(
            f"dyn-cdwdt-horiz-adv: {name} missing; produced by {PF_PRODUCER}, which must run first") from None


def _shared_fah(ctx):
    """The shared FAH [JM] device array. Seeded once from the init-grid-consts row
    (ctx.tables["grid"]["fah"], module_CONTROL.F90:614-741, FAH = -DT/(3*DXH*DY) per row), then owned
    by the TDFI state machine; returned by reference so every filter rescale is seen here."""
    st = ctx.st
    if FAH_KEY in st.fields:
        return st.fields[FAH_KEY]
    grid = ctx.tables.get("grid")
    if grid is None or "fah" not in grid:
        raise RuntimeError("dyn-cdwdt-horiz-adv: ctx.tables[grid][fah] missing; produced by unit init-grid-consts")
    st.fields[FAH_KEY] = grid["fah"]          # the same array (drv-filter-consts scales it)
    return st.fields[FAH_KEY]


def run(ctx):
    if ctx.cdwdt_skip:              # CDWDT returned at Fortran 3427 this step
        return
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pdsg1 is None:
        raise RuntimeError(
            "dyn-cdwdt-horiz-adv: vertical coordinate not set (grid.set_vcoord); pdsg1 missing")

    shape3 = (lm, jm, im)
    w = _need(st, "W")
    pfx = _need(st, "PFX")
    pfy = _need(st, "PFY")
    pfne = _need(st, "PFNE")
    pfnw = _need(st, "PFNW")
    dsg2 = _need(st, "DSG2")
    pdo = _need(st, "PDO")
    dwdt = _need(st, "DWDT")
    for name, arr, shp in (("W", w, shape3), ("PFX", pfx, shape3), ("PFY", pfy, shape3),
                           ("PFNE", pfne, shape3), ("PFNW", pfnw, shape3), ("DWDT", dwdt, shape3),
                           ("DSG2", dsg2, (lm,)), ("PDO", pdo, (jm, im))):
        if tuple(arr.shape) != tuple(shp):
            raise RuntimeError(f"dyn-cdwdt-horiz-adv: {name} has shape {tuple(arr.shape)}, want {tuple(shp)}")
    fah = _shared_fah(ctx)
    if tuple(fah.shape) != (jm,):
        raise RuntimeError(f"dyn-cdwdt-horiz-adv: FAH has shape {tuple(fah.shape)}, want {(jm,)}")
    pdsg1 = _grid_const(st, g, "PDSG1", "pdsg1")

    dt_phase = ctx.dt if getattr(ctx, "dt", None) is not None else ctx.cfg.dt
    dt = np.float32(dt_phase)
    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        dt, EPSVW,
        fah, dsg2, pdsg1, pdo, w,
        pfx, pfy, pfne, pfnw, dwdt,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
