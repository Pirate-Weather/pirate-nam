"""drv-update-water: UPDATE_WATER, the coupling of CW with QC, QR, QS, QI, QG and the fractions.

Fortran: nmm/module_SOLVER_GRID_COMP.F90:11069-11530 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE
UPDATE_WATER, called from PHY_RUN (SOLVER_RUN physics step 12, line 5202).
Kernel: namgpu/kernels/drv_update_water.cu (drv_update_water, one thread per (i, j, l), one launch,
        no host sync, no per-step allocation after the first call).

Reads:   CW [LM, JM, IM] (total condensate, Fortran CWM), F_ICE, F_RAIN, F_RIMEF, T (state tables).
         QC, QR, QS, QI, QG [LM, JM, IM]: Fortran POINTER views into TRACERS at INDX_QC..INDX_QG
         (module_SOLVER_INTERNAL_STATE.F90:1413-1428). Bound here as views of TRACERS[slot-1], so the
         water slots that RRTM, history, restart and the ADV2 tracer loops read are the updated values.
         A species the scheme does not carry (slot 0 in ctx.tables[indx]) stays a zero scratch array.
Writes:  CW, F_ICE, F_RAIN, F_RIMEF and QC, QR, QS, QI, QG (in place, the TRACERS water slots).
Case:    microphysics string as NMMB's TRIM(MICROPHYSICS): fer, fer_hires, gfs, wsm6, thompson;
         any other value is the CASE DEFAULT (nothing written). cld_init = (ntimestep <= 1).
         spec_adv: the configured value (parm/nam_configfile spec_adv = .false.), passed in or read
         from cfg.raw['spec_adv'] (the parm key; default False, as parm/nam_configfile).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_tracer_index import host_index

UNIT = "drv_update_water"
KERNEL = "drv_update_water"

MODES = {"fer": 1, "fer_hires": 1, "gfs": 2, "wsm6": 3, "thompson": 4}
_SPECIES = ("QC", "QR", "QS", "QI", "QG")


def _bind_water(st, tab):
    """Bind QC..QG to their TRACERS water slots (Fortran pointers, SOLVER_INTERNAL_STATE 1413-1428).
    The slot numbers are 1-based in ctx.tables["indx"]; TRACERS is zero-based in its slot axis."""
    trac = st["TRACERS"]
    for name in _SPECIES:
        slot = int(tab[f"INDX_{name}"])
        if slot <= 0:              # this scheme has no such species (Fortran leaves it unassociated)
            continue
        st.fields[name] = trac[slot - 1]


def run(ctx, ntimestep: int | None = None, spec_adv: bool | None = None,
        microphysics: str | None = None):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    mp = str(cfg.microphysics if microphysics is None else microphysics).strip()
    mode = MODES.get(mp, 0)

    nts = int(ctx.ntimestep if ntimestep is None else ntimestep)
    cld_init = 1 if nts <= 1 else 0
    if spec_adv is None:
        spec_adv = str(cfg.raw.get("spec_adv", False)).strip().lower() in ("true", ".true.", "t", "yes", "1")

    shape = (lm, jm, im)
    cw = st["CW"]
    f_ice = st["F_ICE"]
    f_rain = st["F_RAIN"]
    f_rimef = st["F_RIMEF"]
    t = st["T"]
    tab = ctx.tables.get("indx") if hasattr(ctx, "tables") else None
    if tab is None:
        tab = host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))
    _bind_water(st, tab)
    qc, qr, qs, qi, qg = (st.add(name, shape) for name in _SPECIES)

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.int32(mode), np.int32(cld_init), np.int32(1 if spec_adv else 0),
        cw, f_ice, f_rain, f_rimef, t, qc, qr, qs, qi, qg,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
    return {"QC": qc, "QR": qr, "QS": qs, "QI": qi, "QG": qg}
