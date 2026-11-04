"""phys-fer-col-satcheck: grid-scale saturation checks and per-level initialisation, Ferrier-Aligo
microphysics (module_MP_FER_HIRES.F90:911-1028, NAM nam.v4.2.11, 83df4e13c1).

Kernel: namgpu/kernels/phys_fer_col_satcheck.cu (phys_fer_col_satcheck). One launch, one thread per
column (grid_cols), DO L loop inside, no host sync, no per-step allocation after the first call.
The device function fer_satcheck_level is the unit's real product: the fused column loop of
phys-fer-fer-hires includes the .cu and calls it per level. This wrapper is the standalone run.

Reads (ctx.st, Fortran column arrays, all [LM, JM, IM] float32 unless noted):
    T           state table, T_col
    FER_P       P_col (Pa)            FER_Q      Q_col (specific humidity)
    FER_WC      WC_col (condensate)   FER_QI     QI_col     FER_QR   QR_col     FER_QW  QW_col
    FER_THICK   THICK_col             CLEAR_col  float32 (1.0 clear, 0.0 not), written by
                                      phys-fer-col-setup (the setup kernel's CLEAR_col output)
    FER_LSFC    int32 [JM, IM]        LSFC per column, written by the column prep of phys-fer-fer-hires.
                                      The loop runs L=1..LSFC per column, as Fortran big_loop does.
    The FER_* column arrays are built by the column prep of phys-fer-fer-hires. Not yet built, so
    the done-rule check builds them itself (tools/check_phys_fer_col_satcheck.py).
    ctx.tables with the phys-fer-gpvs keys (fer_gpvs_tbpvs, fer_gpvs_tbpvs0, fer_gpvs_coef), built by
    phys_fer_gpvs.run(ctx) here if absent.
Writes (scratch, st.add):
    FER_SC_CLEAR      int32   [LM, JM, IM]  CLEAR after the satcheck section; 0 above LSFC
    FER_SC_RHO        float32 [LM, JM, IM]  RHO of the non-clear levels (0 on clear levels and above LSFC)
    FER_SC_INDEXRMIN  int32   [JM, IM]      INDEXRmin after the satcheck section only: MDRmin at entry,
                                            reset to MDRmin on CLEAR levels. This is NOT the Fortran
                                            end-of-column value: the Fortran also assigns INDEXRmin at
                                            lines 1288-1303 and reads it at 1305, 1571 and 1913, outside
                                            this unit. The entry value is an un-SAVEd local that the source
                                            does not define; MDRmin is the standalone start, and the fused
                                            loop must replace it.
Scalars: RHgrd (default 0.98, CONTRACT section 5), DTPH = NPRECIP * dt from ctx.cfg (nprecip, dt), the
same source as phys-fer-init and phys-fer-col-rain-sedim (NPHS*DT = 25 s x 2 at the production step).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_satcheck"
KERNEL = "phys_fer_col_satcheck"
BLOCK = 128

IN_T = "T"
LSFC_NAME = "FER_LSFC"
IN_FER = {
    "P": "FER_P",
    "Q": "FER_Q",
    "WC": "FER_WC",
    "QI": "FER_QI",
    "QR": "FER_QR",
    "QW": "FER_QW",
    "THICK": "FER_THICK",
    "CLEAR": "CLEAR_col",
}
OUT_CLEAR = "FER_SC_CLEAR"
OUT_RHO = "FER_SC_RHO"
OUT_INDEXRMIN = "FER_SC_INDEXRMIN"

RHGRD_DEFAULT = 0.98


def run(ctx, rhgrd: float | None = None, dtph: float | None = None):
    from namgpu.physics.fer_hires import phys_fer_gpvs

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    tables = ctx.tables
    if phys_fer_gpvs.KEY_COEF not in tables:
        phys_fer_gpvs.run(ctx)
    tbpvs = tables[phys_fer_gpvs.KEY_TBPVS]
    tbpvs0 = tables[phys_fer_gpvs.KEY_TBPVS0]
    coef = tables[phys_fer_gpvs.KEY_COEF]

    if rhgrd is None:
        rhgrd = RHGRD_DEFAULT
    if dtph is None:
        dtph = float(ctx.cfg.nprecip) * float(ctx.cfg.dt)

    t = st[IN_T]
    arrs = {}
    for key, name in IN_FER.items():
        if name not in st.fields:
            raise RuntimeError(f"phys-fer-col-satcheck: column input {name!r} is not in the state")
        arrs[key] = st.fields[name]
    if LSFC_NAME not in st.fields:
        raise RuntimeError(
            f"phys-fer-col-satcheck: {LSFC_NAME} is not set; the column prep of phys-fer-fer-hires "
            "must run before phys-fer-col-satcheck (it sets the loop bound LSFC).")
    lsfc = st.fields[LSFC_NAME]

    clear_out = st.add(OUT_CLEAR, shape3, "int32")
    rho_out = st.add(OUT_RHO, shape3)
    indexrmin_out = st.add(OUT_INDEXRMIN, (jm, im), "int32")

    k = kernels.get(UNIT, KERNEL)
    grid = ((im * jm + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (t, arrs["P"], arrs["Q"], arrs["WC"], arrs["QI"], arrs["QR"], arrs["QW"], arrs["THICK"],
           arrs["CLEAR"], lsfc, tbpvs, tbpvs0, coef,
           np.float32(rhgrd), np.float32(dtph),
           clear_out, rho_out, indexrmin_out,
           np.int32(im), np.int32(jm), np.int32(lm)))

    return {OUT_CLEAR: clear_out, OUT_RHO: rho_out, OUT_INDEXRMIN: indexrmin_out}
