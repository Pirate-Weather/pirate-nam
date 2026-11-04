"""dyn-adv2-horiz: tracer fluxes on H points and horizontal advection of species (secadv, Adams-Bashforth).

Ported from nmm/module_DYNAMICS_ROUTINES.F90:4560-4599 (ADV2 horizontal section, standard-region branch;
NAM nam.v4.2.11, 83df4e13c1). Kernels: namgpu/kernels/dyn_adv2_horiz.cu (adv2_h_ss1, adv2_h_flux_std).

Reads:   TRACERS_SQRT [NTR,LM,JM,IM] (s1 = sqrt(s), dyn-adv2-q2-mid), TRACERS_PREV [NTR,LM,JM,IM] (sp),
         PFX/PFY/PFNE/PFNW [LM,JM,IM] (mass fluxes from PGFORCE, dyn-pgforce-pgf),
         PDOP [JM,IM] ((pd+pdo)*0.5, written by the ADV2 section before the vertical loop),
         DSG2 [LM], PDSG1 [LM], FAH [JM] (shared state FAH, see below), idtadt (configfile idtadt).
Writes:  TRACERS_PREV [NTR,LM,JM,IM] (set to s1, Adams-Bashforth history, as Fortran),
         TRACERS_TEND [NTR,LM,JM,IM] (added to, as Fortran; read by dyn-updates).
Scratch: SS1 [LM,JM,IM] (Adams-Bashforth-positioned sqrt(s) of the species being advected).
Species: the Fortran loops ks = kss..kse outside the level loop. The host loops over that short tracer
    range (2 launches per species, no host sync); no loop over cells on the host.

FAH: Fortran fah is the shared int_state%FAH array. The TDFI filter rescales it (dyn-adv1-horiz-t
    fah_filter_start / fah_direction_change / fah_filter_end), so adv2 must read that same array. It is
    seeded once from the init-grid-consts row (ctx.tables["grid"]["fah"]) into st.fields["FAH"].
Gates (module_CONTROL.F90:247-275): lnsad and adv_standard come from dyn_adv1_horiz_t.adv_gates. The
    flux kernel is not launched when adv_standard is false (Fortran :4581). SS1/SP (item [1]) always run.
Ordering (Fortran call order: q2-mid, then the vertical TCS of adv2, then this horizontal section):
    ADV2_SQRT_RANGE (dyn-adv2-q2-mid) and ADV2_TCS_RANGE (dyn-adv2-vert) must equal (kss, kse). The
    regional branch (dyn-adv2-bndry) refuses to run unless ADV2_HORIZ_RANGE (written here) equals it.
Regional branch (adv_upstream, 4600+) is unit dyn-adv2-bndry, not here.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.dynamics.dyn_adv1_horiz_t import adv_gates

UNIT = "dyn_adv2_horiz"
FAH_KEY = "FAH"          # shared with dyn-adv1-horiz-t (the TDFI-scaled array)


def _grid_const(st, grid, name: str, attr: str):
    """Upload a host grid constant once and cache it in the state (never per step)."""
    if name in st.fields:
        return st.fields[name]
    host = getattr(grid, attr, None)
    if host is None:
        raise RuntimeError(f"dyn-adv2-horiz: grid constant {attr!r} is not set on the grid object")
    import cupy as cp
    st.fields[name] = cp.asarray(np.asarray(host, dtype=np.float32))
    return st.fields[name]


def _shared_fah(ctx):
    """The shared FAH [JM] device array. Seeded once from init-grid-consts; afterwards the TDFI filter
    may have rescaled it in place, and that scaled array is what the kernel must read."""
    st = ctx.st
    if FAH_KEY in st.fields:
        return st.fields[FAH_KEY]
    grid = ctx.tables.get("grid")
    if grid is None or "fah" not in grid:
        raise RuntimeError("dyn-adv2-horiz: ctx.tables[grid][fah] missing; produced by unit init-grid-consts")
    st.fields[FAH_KEY] = grid["fah"]          # the same array (drv-filter-consts scales it)
    return st.fields[FAH_KEY]


def _tile(ctx, im: int, jm: int):
    """1-based tile limits (its, ite, jts, jte) and domain limits (ids, ide, jds, jde). One process:
    the tile is the full domain. A ctx.tile dict overrides it when a decomposition exists."""
    dom = (1, im, 1, jm)
    tile = getattr(ctx, "tile", None)
    if tile is None:
        return dom, dom
    return dom, (tile["its"], tile["ite"], tile["jts"], tile["jte"])


def _standard_bounds(lnsad: int, dom, tile):
    """0-based inclusive standard-region bounds: max(its,ids+1+lnsad)..min(ite,ide-1-lnsad), same in j
    (module_DYNAMICS_ROUTINES.F90:4583-4586), minus 1 for 0-based indexing."""
    ids, ide, jds, jde = dom
    its, ite, jts, jte = tile
    ibeg = max(its, ids + 1 + lnsad)
    iend = min(ite, ide - 1 - lnsad)
    jbeg = max(jts, jds + 1 + lnsad)
    jend = min(jte, jde - 1 - lnsad)
    return ibeg - 1, iend - 1, jbeg - 1, jend - 1


def _require_order(ctx, kss: int, kse: int):
    if ctx.tables.get("ADV2_SQRT_RANGE") != (kss, kse):
        raise RuntimeError(
            f"dyn-adv2-horiz: TRACERS_SQRT was last written for {ctx.tables.get("ADV2_SQRT_RANGE")}, "
            f"need {kss}..{kse}; dyn-adv2-q2-mid must run on the same range first")
    if ctx.tables.get("ADV2_TCS_RANGE") != (kss, kse):
        raise RuntimeError(
            f"dyn-adv2-horiz: vertical TCS was last written for {ctx.tables.get("ADV2_TCS_RANGE")}, "
            f"need {kss}..{kse}; dyn-adv2-vert must run on the same range first (Fortran :4508-4539)")


def run(ctx, kss: int | None = None, kse: int | None = None, idtadt: int | None = None):
    """idtadt: the ADV2 call's IDTADT argument, IDTADTQ (1 regional) for the Q call (SOLVER_RUN 4321-4325)
    and IDTADT for the tracer call (4459-4468); the driver plan passes it. Default: configfile idtadt."""
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntr = int(st.TRACERS.shape[0])
    kss = 1 if kss is None else int(kss)
    kse = ntr if kse is None else int(kse)
    if not (1 <= kss <= kse <= ntr):
        raise ValueError(f"dyn-adv2-horiz: tracer range {kss}..{kse} outside 1..{ntr}")
    _require_order(ctx, kss, kse)

    lnsad, adv_standard = adv_gates(ctx.cfg)
    idtadt = int(ctx.cfg.idtadt) if idtadt is None else int(idtadt)
    dom, tile = _tile(ctx, im, jm)
    ibeg, iend, jbeg, jend = _standard_bounds(lnsad, dom, tile)

    s1 = st.add("TRACERS_SQRT", (ntr, lm, jm, im))
    sp = st.add("TRACERS_PREV", (ntr, lm, jm, im))
    tend = st.add("TRACERS_TEND", (ntr, lm, jm, im))
    ss1 = st.add("SS1", (lm, jm, im))
    pfx = st.add("PFX", (lm, jm, im))
    pfy = st.add("PFY", (lm, jm, im))
    pfne = st.add("PFNE", (lm, jm, im))
    pfnw = st.add("PFNW", (lm, jm, im))
    pdop = st.add("PDOP", (jm, im))
    fah = _shared_fah(ctx)
    pdsg1 = _grid_const(st, g, "PDSG1", "pdsg1")

    ss_k = kernels.get(UNIT, "adv2_h_ss1")
    flux_k = kernels.get(UNIT, "adv2_h_flux_std")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        for ks in range(kss, kse + 1):
            n0 = np.int32(ks - 1)
            ss_k(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                 (np.int32(im), np.int32(jm), np.int32(lm), n0, s1, sp, ss1))
            if adv_standard:
                flux_k(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
                       (np.int32(im), np.int32(jm), np.int32(lm),
                        np.int32(ibeg), np.int32(iend), np.int32(jbeg), np.int32(jend),
                        np.int32(idtadt), n0,
                        ss1, pfx, pfy, pfne, pfnw, fah, st.DSG2, pdsg1, pdop, tend))
    ctx.tables["ADV2_HORIZ_RANGE"] = (kss, kse)   # read by dyn-adv2-bndry
