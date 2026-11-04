"""dyn-adv1-horiz-t: temperature horizontal advection, secadv 2nd order, Adams-Bashforth in time.

Ported from nmm/module_DYNAMICS_ROUTINES.F90:1539-1570 (ADV1, standard-region branch; NAM nam.v4.2.11,
83df4e13c1). Kernels: namgpu/kernels/dyn_adv1_horiz_t.cu (adv1_pdop_h, adv1_t_ab_col, adv1_t_horiz_std).

Reads:   T [LM,JM,IM], PD, PDO [JM,IM] (PDOP source), PFX/PFY/PFNE/PFNW [LM,JM,IM] (mass fluxes, written by
         dyn-dht, module_DYNAMICS_ROUTINES.F90:685), OMGALF [LM,JM,IM], DSG2 [LM], PDSG1 [LM] and FAH [JM]
         (grid constants).
Writes:  PDOP [JM,IM] = (PD+PDO)*0.5 (state, rewritten every step before the vertical loop, Fortran 1427-1431),
         TP [LM,JM,IM] (set to T, Adams-Bashforth history), TCT [LM,JM,IM] (added to, as Fortran).
Scratch: T1 [LM,JM,IM] (Adams-Bashforth-positioned T, filled by adv1_t_ab_col, read by the advection).
Launch:  three kernels on ctx.stream, no host sync. Grid constants are uploaded once and cached.

Gates (Fortran, module_CONTROL.F90:247-275, single process):
  lnsad = 0 (global) or lnsh + 2 (regional);  adv_standard = (IM >= 2+lnsad and JM >= 2+lnsad).
  The standard-region flux block (Fortran 1552) is skipped when adv_standard is false; the PDOP and
  Adams-Bashforth items are not gated.

FAH during the TDFI filter: the solver rescales the shared int_state%FAH array (module_SOLVER_GRID_COMP.F90:
  3137-3170 at the first filter pass: FAH = IFACT*DTRATIO*FAH, FAH_SV saved; 3208-3237 on direction change:
  IFACT = -1, FAH = IFACT*FAH; 3330-3344 at the end: FAH = FAH_SV). IFACT = +1 for dt_step >= 0 else -1,
  DTRATIO = ABS(DT/FUND_DT) (line 2958), FUND_DT = cfg.dt. The same state machine as dyn-vtoa's EF4T.
  The DFI driver calls fah_filter_start, fah_direction_change and fah_filter_end at those points. Outside
  a filter the cached FAH is the grid constant (init-grid-consts, FAH = -dt/(3*DXH*DY)).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "dyn_adv1_horiz_t"
FAH_KEY = "FAH"
FAH_SAVE_KEY = "FAH_SV"


def _grid_const(st, grid, name: str, attr: str, tables=None):
    """Upload a grid constant once and cache it in the state (never per step). Source: the grid
    object attribute, else the init-grid-consts row ctx.tables["grid"][attr] (FAH lives there).
    FAH is the shared int_state%FAH row (the TDFI rescales it through drv-filter-consts).
    fah_filter_start / fah_direction_change / fah_filter_end below are superseded by drv-filter-consts
    and must not be called with it (they would scale FAH twice)."""
    if name in st.fields:
        return st.fields[name]
    import cupy as cp
    host = getattr(grid, attr, None)
    if host is None and tables is not None and attr in (tables.get("grid") or {}):
        host = tables["grid"][attr]
    if host is None:
        raise RuntimeError(f"dyn-adv1-horiz-t: grid constant {attr!r} is on neither the grid object nor "
                           f"ctx.tables['grid'] (init-grid-consts)")
    # FAH: alias the init-grid-consts row (drv-filter-consts scales that array for the TDFI); others copy.
    if attr == "fah" and isinstance(host, cp.ndarray):
        st.fields[name] = host
    else:
        st.fields[name] = cp.array(host, dtype=cp.float32)
    return st.fields[name]


def adv_gates(cfg) -> tuple[int, bool]:
    """(lnsad, adv_standard) for a single-process run, from module_CONTROL.F90:247-275."""
    lnsad = 0 if bool(cfg.global_) else int(cfg.lnsh) + 2
    im, jm = int(cfg.im), int(cfg.jm)
    adv_standard = im >= 2 + lnsad and jm >= 2 + lnsad
    return lnsad, adv_standard


def fah_filter_start(ctx, dt_step, fund_dt=None):
    """SOLVER_RUN FIRST_PASS block (module_SOLVER_GRID_COMP.F90:3137-3170): save FAH_SV, then
    FAH = IFACT*DTRATIO*FAH. Call once when the TDFI filter starts, with the first filter step dt."""
    st, g = ctx.st, ctx.grid
    fah = _grid_const(st, g, FAH_KEY, "fah", ctx.tables)
    if fund_dt is None:
        fund_dt = ctx.cfg.dt
    ifact = np.float32(1.0) if dt_step >= 0 else np.float32(-1.0)
    dtratio = np.abs(np.float32(dt_step) / np.float32(fund_dt))
    st.fields[FAH_SAVE_KEY] = fah.copy()
    fah *= ifact * dtratio               # in place: the shared FAH the kernels read


def fah_direction_change(ctx):
    """changedir block (module_SOLVER_GRID_COMP.F90:3208-3250): IFACT = -1, FAH = IFACT*FAH."""
    fah = ctx.st.fields[FAH_KEY]
    fah *= np.float32(-1.0)


def fah_filter_end(ctx):
    """end_filt block (module_SOLVER_GRID_COMP.F90:3344): FAH = FAH_SV."""
    st = ctx.st
    fah = st.fields[FAH_KEY]
    fah[...] = st.fields.pop(FAH_SAVE_KEY)


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsad, adv_standard = adv_gates(ctx.cfg)

    t1 = st.add("T1", (lm, jm, im))
    pfx = st.add("PFX", (lm, jm, im))
    pfy = st.add("PFY", (lm, jm, im))
    pfne = st.add("PFNE", (lm, jm, im))
    pfnw = st.add("PFNW", (lm, jm, im))
    pdop = st.add("PDOP", (jm, im))
    fah = _grid_const(st, g, FAH_KEY, "fah", ctx.tables)
    pdsg1 = _grid_const(st, g, "PDSG1", "pdsg1", ctx.tables)

    pdop_k = kernels.get(UNIT, "adv1_pdop_h")
    ab = kernels.get(UNIT, "adv1_t_ab_col")
    horiz = kernels.get(UNIT, "adv1_t_horiz_std")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        pdop_k(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D,
               (np.int32(im), np.int32(jm), st.PD, st.PDO, pdop))
        ab(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
           (np.int32(im), np.int32(jm), np.int32(lm), st.T, st.TP, t1))
        if adv_standard:
            horiz(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                  (np.int32(im), np.int32(jm), np.int32(lm), np.int32(lnsad),
                   t1, pfx, pfy, pfne, pfnw, fah, st.DSG2, pdsg1, pdop, st.OMGALF, st.TCT))
