"""phys-fer-fer-hires: FER_HIRES column driver (Ferrier-Aligo microphysics, one column per thread).

Ported from phys/module_MP_FER_HIRES.F90:208-604 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE FER_HIRES.
Kernel: namgpu/kernels/phys_fer_fer_hires.cu (fer_hires), one launch, grid_cols(IM, JM).
The per-column microphysics EGCP01COLUMN_hr is the column units' kernel-side device function
(phys-fer-col-*, phys-fer-condense, phys-fer-deposit, phys-fer-get-indexr); this wrapper cannot
run until those land (the EGCP01COLUMN_hr driver body, fer_egcp01column_hr, is not yet defined).

Arrays are passed explicitly. The NMMB caller (phys-fer-driver, module_MICROPHYSICS.F90:60-558)
owns the mapping from NMMB state names to FER_HIRES arguments. Shapes (CONTRACT section 1):
    dz8w, rho_phy, p_phy, pi_phy, th_phy, q, qt, qc, qr, qs, f_ice, f_rain, f_rimef, refl_10cm
                  float32 [LM, JM, IM]   (level 0 = model top)
    lowlyr        int32   [JM, IM]       (Fortran LOWLYR, 1-based level of the surface)
    rainnc, rainncv, sr  float32 [JM, IM]  (updated in place; sr is an output)
    mprates       float32 [D_SS, LM, JM, IM]  (updated in place)
    mp_state      float32 [43]            (MP_RESTART_STATE, read only: MY_GROWTH_NMM 1..35, then
                                           C1XPVS0, C2XPVS0, C1XPVS, C2XPVS, CIACW, CIACR, CRACW, BRAUT)
    tbpvs, tbpvs0 float32 [NX = 7501]     (TBPVS, TBPVS0, read only)
Tile: its, ite, jts, jte are the Fortran ITS/ITE/JTS/JTE the caller passes (1-based, inclusive). The
GSMDRIVE caller passes ITSLOC..ITELOC and JTSLOC..JTELOC (module_MICROPHYSICS.F90:335-338, 363-364);
FER_HIRES loops over that tile only, so boundary rows and columns are never written.
Not bound (FER_HIRES never reads them): ITIMESTEP, DX, DY (dummy arguments of the Fortran body only).
No host sync, no device allocation in this wrapper.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_fer_hires"
KERNEL = "fer_hires"
FER_MAXLEV = 64


def _tbl_args(ctx):
    """FERRIER_INIT_hr tables for the column (FerTbl in phys_fer_column.cuh), from ctx.tables["fer"]
    (phys-fer-init), plus the DEPOSIT non-convergence counters (int32 [1] and [8], allocated once)."""
    fer = ctx.tables["fer"]
    if "deposit_counters" not in fer:
        import cupy as cp
        fer["deposit_counters"] = (cp.zeros(1, dtype=cp.int32), cp.zeros(8, dtype=cp.int32))
    nonconv, warn2 = fer["deposit_counters"]
    return (fer["massi"], fer["vsnowi"], fer["accri"], fer["venti1"], fer["venti2"], fer["massr"],
            fer["vrain"], fer["accrr"], fer["ventr1"], fer["ventr2"], fer["vel_rf"], fer["growth"],
            fer["sc"], nonconv, warn2)


def run(
    ctx,
    *,
    dz8w,
    rho_phy,
    p_phy,
    pi_phy,
    th_phy,
    q,
    qt,
    qc,
    qr,
    qs,
    f_ice,
    f_rain,
    f_rimef,
    refl_10cm,
    lowlyr,
    rainnc,
    rainncv,
    sr,
    mprates,
    mp_state,
    tbpvs,
    tbpvs0,
    dt: float,
    rhgrd: float,
    d_ss: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
):
    lm, jm, im = (int(s) for s in th_phy.shape)
    if lm > FER_MAXLEV:
        raise ValueError(f"FER_HIRES kernel holds {FER_MAXLEV} levels; LM={lm}")
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(
            kernels.grid_cols(im, jm),
            (kernels.BLOCK_COL[0], 1, 1),
            (
                dz8w, rho_phy, p_phy, pi_phy, th_phy, q, qt, qc, qr, qs,
                f_ice, f_rain, f_rimef, refl_10cm, rainnc, rainncv, sr,
                lowlyr, mprates, np.int32(d_ss), mp_state, tbpvs, tbpvs0,
                np.float32(dt), np.float32(rhgrd),
                np.int32(im), np.int32(jm), np.int32(lm),
                np.int32(its), np.int32(ite), np.int32(jts), np.int32(jte),
                *_tbl_args(ctx),
            ),
        )
    return sr
