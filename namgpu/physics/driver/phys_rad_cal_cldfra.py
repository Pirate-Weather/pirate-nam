"""phys-rad-cal-cldfra: CAL_CLDFRA, cloud fraction from condensate.

Fortran: nmm/module_RADIATION.F90:1045-1134 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE CAL_CLDFRA.
Called from RADIATION (module_RADIATION.F90:470) in the lwrad CASE DEFAULT branch only, and only
when RADIATION's own gate passes (module_RADIATION.F90:412-414): not both LW and SW off, and
ITIMESTEP == 1 or MOD(ITIMESTEP, NRAD) == 0. The 12 km NAM configuration uses longwave rrtm, which
takes CASE (RRTMLWSCHEME) and does not call this routine (RRTM computes its own cloud fraction),
so this unit is not reached on the proof path.
Kernel: namgpu/kernels/phys_rad_cal_cldfra.cu (phys_rad_cal_cldfra, one thread per (i, j, l),
        one launch, no host sync, no per-step allocation after the first call).

Reads:   QC, QI [LM, JM, IM] (state table fields, see parm/nam_solver_state.txt).
Writes:  CLDFRA [LM, JM, IM] (state table, H point), interior cells only:
         Fortran i = 2..IM-1, j = 2..JM-1 (ITS_B1..ITE_B1, JTS_B1..JTE_B1 on a single card);
         the boundary rows and columns are left as they were.
Flags:   F_QC and F_QI are the microphysics switches of module_SOLVER_INTERNAL_STATE.F90
         (fer, fer_hires: F_QC on, F_QI off; wsm6, thompson and gfs: both on). Any other
         microphysics name is refused, as the Fortran STOPs on it.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rad_cal_cldfra"
KERNEL = "phys_rad_cal_cldfra"

# Fortran SOLVER_INTERNAL_STATE microphysics branches (module_SOLVER_INTERNAL_STATE.F90:868-940).
_FLAGS = {
    "fer": (True, False),
    "fer_hires": (True, False),
    "wsm6": (True, True),
    "thompson": (True, True),
    "gfs": (True, True),
}


def cldfra_flags(microphysics: str) -> tuple[bool, bool]:
    """(F_QC, F_QI) for the configured microphysics, as set in SOLVER_INTERNAL_STATE.

    Unknown names raise: the Fortran writes 'Unknown microphysics' and STOPs, so this port
    refuses the same way rather than guessing flags.
    """
    mp = str(microphysics).strip()
    if mp not in _FLAGS:
        raise ValueError(
            f"phys-rad-cal-cldfra: unknown microphysics {mp!r}; module_SOLVER_INTERNAL_STATE.F90 "
            "stops on it (supported: " + ", ".join(sorted(_FLAGS)) + ")")
    return _FLAGS[mp]


def cldfra_gate(itimestep: int, nrad: int, sw_physics: int, lw_physics: int) -> bool:
    """RADIATION's gate before CAL_CLDFRA (module_RADIATION.F90:412-414).

    Returns False when Fortran would return early (LW and SW both off) or would not reach
    CAL_CLDFRA on this step (cadence). Codes are the Fortran SW_PHYSICS / LW_PHYSICS values.
    """
    if lw_physics == 0 and sw_physics == 0:
        return False
    return itimestep == 1 or itimestep % nrad == 0


def run(ctx, *, itimestep: int, sw_physics: int, lw_physics: int,
        f_qc: bool | None = None, f_qi: bool | None = None,
        microphysics: str | None = None):
    """Write CLDFRA for this step when RADIATION would call CAL_CLDFRA; otherwise do nothing.

    itimestep is the Fortran ITIMESTEP of the RADIATION call (the driver's ntimestep_rad).
    Returns {"CLDFRA": ...} when it ran, None when the gate is closed.
    """
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    nrad = int(cfg.nrads)   # NRAD = NRADS (module_RADIATION.F90:247)
    if not cldfra_gate(int(itimestep), nrad, int(sw_physics), int(lw_physics)):
        return None

    if f_qc is None or f_qi is None:
        mp = cfg.microphysics if microphysics is None else microphysics
        dqc, dqi = cldfra_flags(mp)
        f_qc = dqc if f_qc is None else f_qc
        f_qi = dqi if f_qi is None else f_qi

    # The kernel's interior bounds below equal Fortran ITS_B1..ITE_B1 only when the domain is
    # the whole grid on one card (IDS=1, IDE=IM, JDS=1, JDE=JM). Refuse a split domain loudly.
    ids, ide = getattr(g, "ids", 1), getattr(g, "ide", im)
    jds, jde = getattr(g, "jds", 1), getattr(g, "jde", jm)
    if (ids, ide, jds, jde) != (1, im, 1, jm):
        raise ValueError(
            "phys-rad-cal-cldfra: the interior bounds assume the full domain on one card "
            f"(IDS,IDE,JDS,JDE = {ids},{ide},{jds},{jde}; IM,JM = {im},{jm})")

    shape = (lm, jm, im)
    cldfra = st["CLDFRA"]
    # QC and QI are state-table fields: a missing one fails loudly here (KeyError), it is not
    # silently zero-filled.
    qc, qi = st["QC"], st["QI"]
    if qc.shape != shape or qi.shape != shape:
        raise ValueError(f"phys-rad-cal-cldfra: QC/QI shape {qc.shape}/{qi.shape}, expected {shape}")

    # Fortran IQS..IQE = ITS_B1..ITE_B1 = max(its, ids+1)..min(ite, ide-1) with ids = 1,
    # ide = IM (single card): Fortran 2..IM-1, zero-based 1..IM-2. JQS..JQE likewise.
    i_lo, i_hi = 1, im - 2
    j_lo, j_hi = 1, jm - 2

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    args = (
        np.int32(im), np.int32(jm), np.int32(lm),
        np.int32(i_lo), np.int32(i_hi), np.int32(j_lo), np.int32(j_hi),
        np.int32(1 if f_qc else 0), np.int32(1 if f_qi else 0),
        cldfra, qc, qi,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)
    return {"CLDFRA": cldfra}
