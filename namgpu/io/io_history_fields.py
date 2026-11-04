"""io-history-fields: wrfout-shaped history fields on device, from the NMMB state.

Unit io-history-fields (mode "new"); kernel namgpu/kernels/io_history_fields.cu:
  hist_moist_theta (grid_2d, LM levels), hist_ph (grid_cols, LM+1 interfaces), hist_surface (2-D).
Reads  : T, Q, CW, F_ICE, F_RAIN, PD [JM,IM] (or [LM,JM,IM]); FIS, PINT (LM+1), ACPREC_TOT, CUPREC_TOT [m];
         UE10, VE10 = earth-relative 10 m winds on H points, written by io-wind-rotate (dep, must run
         before this unit; it raises if they are absent); Z (state, midlayer). Grid: psgml1, sgml2 (ctx.tables).
         UEARTH, VEARTH = earth-relative winds on H points, LM levels (io-wind-rotate). W, TSHLTR, QSHLTR
         (state). REFL_10CM (LM levels) only when the Ferrier microphysics (phys-fer-calmict) has
         written it; COMPOSITE is its column maximum.
Writes (scratch via st.add, names H_*): H_P, H_T, H_QVAPOR, H_QCLOUD, H_QRAIN, H_QICE, H_QSNOW, H_PH,
         H_PSFC, H_HGT, H_RAINNC, H_RAINC, H_COMPOSITE, and H_CW_SPEC, H_FICE_SPEC, H_FRAIN_SPEC when
         spec_adv is on (see "Condensate" below).
Returns: dict  name -> (device array, dims) for io-history (history.write_history). Names:
         PSFC, HGT, T (theta-300), PRES (full mid-layer pressure, Pa; NOT WRF's perturbation P, and
         there is no PB, so no reader may add a base state), PH (g*Z, m2 s-2, interfaces), QVAPOR,
         QCLOUD, QRAIN, QICE, QSNOW (mixing ratios), U10, V10 (earth-relative), RAINNC, RAINC (mm),
         U, V (earth-relative at mass points, LM levels, not WRF's staggered U/V), W (LM levels),
         T2 (2 m temperature from TSHLTR theta and PSHLTR), Q2 (QSHLTR), MSLP (MEMBRANE_MSLP, Pa, when
         io-quasipost-driver ran for this output; it sets ctx.quasipost_done), and REFL_10CM and
         COMPOSITE when the Ferrier field exists.
Launch: kernels and the small array ops on ctx.stream, no host sync, no per-step allocation after
the first call.

Owners and conventions (review of io-history-fields, 2026-11-04):
  Z.      Midlayer height. dyn-cdzdt-nonhydro writes it every step (Fortran z(i,j,l) = zlo + dz/2,
          module_DYNAMICS_ROUTINES.F90:3075); init-derive-state writes the same convention at init, so the
          0 h history is consistent. Interfaces are rebuilt here from FIS: zint(l) = 2 Z(l) - zint(l+1).
          The state table (parm/nam_solver_state.txt) says "Height at midlayers". Z[LM] is not used.
  Accumulators. ACPREC_TOT is the total precip since start, convective plus grid-scale (Fortran
          module_MICROPHYSICS.F90:548 and module_CONVECTION.F90:619); reset only at restart output
          (module_SOLVER_GRID_COMP.F90:10317), which is not ported yet. CUPREC_TOT is a port addition,
          the convective total since start, because the Fortran keeps only the NPREC bucket CUPREC,
          which drv-bucket-reset empties every NPREC steps. RAINC = CUPREC_TOT, RAINNC = ACPREC_TOT - CUPREC_TOT.
          Neither is the bucket.
  Condensate. Only the fer and fer_hires microphysics are ported (UPDATE_WATER, module_SOLVER_GRID_COMP.F90
          11144-11200). Other microphysics refuse. With spec_adv=F the state CW, F_ICE, F_RAIN are the
          output. With spec_adv=T (ELSE branch 11169-11200) the output condensate is rebuilt from QC, QR, QS
          here, read-only. At step 0 (CLD_INIT) the CW path is used, as the Fortran F branch does.
          Declared gap: the step-path write-back of that ELSE branch (CWM, F_ICE, F_RAIN in the state) is not
          ported yet, so physics under spec_adv=T still reads the state values.
  EPSQ. 1e-12 (nmm/module_CONSTANTS.F90 "epsq=1.e-12", same as nmmb_common.cuh NMMB_EPSQ). Checked.
"""
from __future__ import annotations

import contextlib

import cupy as cp
import numpy as np

from namgpu import kernels

UNIT = "io_history_fields"
EPSQ = 1.0e-12  # module_CONSTANTS epsq (nmm/module_CONSTANTS.F90)

DIMS2 = ("south_north", "west_east")
DIMS3 = ("bottom_top", "south_north", "west_east")
DIMS3S = ("bottom_top_stag", "south_north", "west_east")

MICROPHYSICS_PORTED = ("fer", "fer_hires")


def _vec(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        if host is None:
            raise RuntimeError(f"io-history-fields: host vector {key!r} is not set on the grid")
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _spec_adv_condensate(ctx, s3):
    """spec_adv=T output condensate: UPDATE_WATER ELSE branch (module_SOLVER_GRID_COMP.F90:11169-11200).

    CWM = QC+QR+QS; F_ICE = QS/CWM if QS > EPSQ else 0; F_RAIN = QR/(QC+QR) if QR > EPSQ else 0.
    Whole-array operations on the device; the state is read, never written.
    """
    import cupy as cp
    st = ctx.st
    qc, qr, qs = st["QC"], st["QR"], st["QS"]
    cw = st.add("H_CW_SPEC", s3)
    f_ice = st.add("H_FICE_SPEC", s3)
    f_rain = st.add("H_FRAIN_SPEC", s3)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        liq = qc + qr
        cwm = liq + qs
        cw[...] = cwm
        f_ice[...] = cp.where(qs > EPSQ, qs / cp.where(cwm > 0, cwm, 1.0), 0.0)
        f_rain[...] = cp.where(qr > EPSQ, qr / cp.where(liq > 0, liq, 1.0), 0.0)
    return cw, f_ice, f_rain


def run(ctx):
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    mp = str(ctx.cfg.microphysics).strip()
    if mp not in MICROPHYSICS_PORTED:
        raise RuntimeError(
            f"io-history-fields: microphysics {mp!r} has no CWM/F_ICE/F_RAIN couple ported; "
            f"only {MICROPHYSICS_PORTED} are. The history would read stale condensate.")
    for need in ("UE10", "VE10", "UEARTH", "VEARTH"):
        if need not in st.fields:
            raise RuntimeError(
                f"io-history-fields: {need} missing; io-wind-rotate must run first")
    if "ACPREC_TOT" not in st.fields:
        raise RuntimeError("io-history-fields: ACPREC_TOT missing (state table ACPREC_TOT)")
    # CUPREC_TOT is zero until a convection scheme writes it; zero is the correct value with no convection.
    cup_tot = st.add("CUPREC_TOT", (jm, im))

    s3 = (lm, jm, im)
    s3s = (lm + 1, jm, im)
    s2 = (jm, im)

    spec_adv = bool(ctx.cfg.raw.get("spec_adv", False))
    if spec_adv and int(ctx.ntimestep) > 0:
        cw, f_ice, f_rain = _spec_adv_condensate(ctx, s3)
    else:
        cw, f_ice, f_rain = st["CW"], st["F_ICE"], st["F_RAIN"]

    psgml1 = _vec(ctx, "hist_psgml1", g.psgml1)
    sgml2 = _vec(ctx, "hist_sgml2", g.sgml2)

    p_out = st.add("H_P", s3)
    t_out = st.add("H_T", s3)
    qv = st.add("H_QVAPOR", s3)
    qc = st.add("H_QCLOUD", s3)
    qr = st.add("H_QRAIN", s3)
    qi = st.add("H_QICE", s3)
    qs = st.add("H_QSNOW", s3)
    ph = st.add("H_PH", s3s)
    psfc = st.add("H_PSFC", s2)
    hgt = st.add("H_HGT", s2)
    rainnc = st.add("H_RAINNC", s2)
    rainc = st.add("H_RAINC", s2)

    k_moist = kernels.get(UNIT, "hist_moist_theta")
    k_ph = kernels.get(UNIT, "hist_ph")
    k_sfc = kernels.get(UNIT, "hist_surface")
    g3 = kernels.grid_2d(im, jm, lm)
    gcol = kernels.grid_cols(im, jm)
    g2 = kernels.grid_2d(im, jm, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_moist(g3, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm),
            st.T, st.Q, cw, f_ice, f_rain, st.PD, psgml1, sgml2,
            p_out, t_out, qv, qc, qr, qi, qs,
        ))
        k_ph(gcol, kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm), st.Z, st.FIS, ph,
        ))
        k_sfc(g2, kernels.BLOCK_2D, (
            np.int32(im), np.int32(jm), np.int32(lm),
            st.PINT, st.FIS, st.ACPREC_TOT, cup_tot, psfc, hgt, rainnc, rainc,
        ))

    # T2. NMMB TSHLTR is the 2 m POTENTIAL temperature (parm/nam_phy_state.txt "Theta at 2-m", JSFC/SFCDIAGS
    # write theta); the post turns it into temperature with T2 = TSHLTR * (PSHLTR * 1e-5) ** CAPA
    # (nam_post0.fd SURFCE.f, CAPA = R/CP = 0.28589641). History T2 is that temperature, not theta.
    t2 = st.add("H_T2", s2)
    with scope:
        t2[...] = st["TSHLTR"] * (st["PSHLTR"] * cp.float32(1.0e-5)) ** cp.float32(0.28589641)
    out = {
        "PSFC": (psfc, DIMS2),
        "HGT": (hgt, DIMS2),
        "T": (t_out, DIMS3),
        "PRES": (p_out, DIMS3),
        "PH": (ph, DIMS3S),
        "QVAPOR": (qv, DIMS3),
        "QCLOUD": (qc, DIMS3),
        "QRAIN": (qr, DIMS3),
        "QICE": (qi, DIMS3),
        "QSNOW": (qs, DIMS3),
        "U10": (st.UE10, DIMS2),
        "V10": (st.VE10, DIMS2),
        "RAINNC": (rainnc, DIMS2),
        "RAINC": (rainc, DIMS2),
        "U": (st.UEARTH, DIMS3),
        "V": (st.VEARTH, DIMS3),
        "W": (st["W"], DIMS3),
        "T2": (t2, DIMS2),
        "Q2": (st["QSHLTR"], DIMS2),
    }
    if "MEMBRANE_MSLP" in st.fields and getattr(ctx, "quasipost_done", False):
        out["MSLP"] = (st["MEMBRANE_MSLP"], DIMS2)
    if "REFL_10CM" in st.fields:
        refl = st["REFL_10CM"]
        comp = st.add("H_COMPOSITE", s2)
        with scope:
            comp[...] = refl.max(axis=0)
        out["REFL_10CM"] = (refl, DIMS3)
        out["COMPOSITE"] = (comp, DIMS2)
    return out
