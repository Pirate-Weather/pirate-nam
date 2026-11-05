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
         COMPOSITE when the Ferrier field exists. With the configfile key history_handoff: true also the
         parent hand-off set (HANDOFF_FIELDS below: skin/SST, sea ice, soil, snow, canopy water, land mask) and
         the parent grid keys as global attributes (handoff_attrs); off, the field list is unchanged.
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

# The physics fields NAM's awphys GRIB2 reads (woof-nam products, namgpu/products/awphys.tsv), written under their
# NMMB state names (parm/nam_phy_state.txt) when the state holds them, H points, [JM,IM]: fluxes, surface layer,
# radiation at the ground, land/sea/snow state, cloud cover, and the hourly extremes io-max-fields keeps (reset
# every hour, so each history hour holds the maximum or minimum of the hour that ends there). Views of the state:
# no device copy. U10MAX/V10MAX (the u, v of the hour's strongest 10 m wind) are written next to U10/V10, rotated
# to earth-relative the same way (H points: io_max_fields reads U10/V10 at the H cell).
PRODUCT_CARRIERS_2D = (
    "TWBS", "QWBS", "USTAR", "Z0", "PBLH", "AKMS", "ALBEDO", "RSWIN", "RLWIN", "RSWOUT", "RADOT", "THS", "SST",
    "SICE", "SM", "SNO", "SI", "SNOWC", "CMC", "TG", "VEGFRC", "ISLTYP", "IVGTYP", "CFRACL", "CFRACM", "CFRACH",
    "PSHLTR", "POTEVP", "SFCEVP",
    "T02MAX", "T02MIN", "RH02MAX", "RH02MIN", "SPD10MAX", "REFDMAX", "UPVVELMAX", "DNVVELMAX", "PRATEMAX",
    "UPHLMAX25", "UPHLMIN25", "UPHLMAX03", "UPHLMIN03",
    # soil moisture availability and total (Noah), the radiation time factors the post applies to RSWIN/RSWOUT/
    # RLWIN (CZEN, CZMEAN, SIGT4), and the BMJ convective cloud bottom/top between outputs (cloud base)
    "SMSTAV", "SMSTOT", "CZEN", "CZMEAN", "SIGT4", "CNVBOT", "CNVTOP",
)
# [LM,JM,IM] carriers: cloud fraction, and the MYJ turbulent kinetic energy (Q2 holds 2 TKE).
PRODUCT_CARRIERS_3D = ("CLDFRA",)
# Noah soil layers, wrfout names, on soil_layers_stag with layer 1 at the top (0-10/10-40/40-100/100-200 cm):
# the history writer flips only bottom_top fields. One card only: the several-card merge writer has no soil
# dimension and flips every 3-D field, so on several cards these records are left out (named in the run index).
DIMS_SOIL = ("soil_layers_stag", "south_north", "west_east")
SOIL_CARRIERS = {"TSLB": "STC", "SMOIS": "SMC", "SH2O": "SH2O"}


def _dragco(ctx, scope):
    """The post's surface drag coefficient on the NMMB B grid (UPP CALDRG.f, gridtype B, MODELNAME NMM): the mean
    of the four V-point winds of the lowest layer around each H point, averaged with the z0 wind (UZ0, VZ0, H
    points), CD = USTAR^2 / that speed squared where it is over 1e-6, 0 elsewhere and on the outermost rows and
    columns (the post loops JSTA_M..JEND_M, ISTA_M..IEND_M). None when the state lacks an input or on several cards
    (the slab edge rows have no neighbour row here)."""
    st = ctx.st
    if getattr(ctx, "slab", None) is not None or any(n not in st.fields for n in ("U", "V", "UZ0", "VZ0", "USTAR")):
        return None
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    d = st.add("H_DRAGCO", (jm, im))
    with scope:
        u, v = st["U"][-1], st["V"][-1]                 # lowest layer (NMMB l = LM is the last index)
        ul = cp.float32(0.25) * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[:-2, 1:-1] + u[:-2, :-2])
        vl = cp.float32(0.25) * (v[1:-1, 1:-1] + v[1:-1, :-2] + v[:-2, 1:-1] + v[:-2, :-2])
        ub = cp.float32(0.5) * (ul + st["UZ0"][1:-1, 1:-1])
        vb = cp.float32(0.5) * (vl + st["VZ0"][1:-1, 1:-1])
        w2 = ub * ub + vb * vb
        ust = st["USTAR"][1:-1, 1:-1]
        d[...] = cp.float32(0.0)
        d[1:-1, 1:-1] = cp.where(w2 > cp.float32(1.0e-6), ust * ust / cp.where(w2 > cp.float32(1.0e-6), w2,
                                                                                cp.float32(1.0)), cp.float32(0.0))
    return d


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
    for name in PRODUCT_CARRIERS_2D:
        if name in st.fields and tuple(st[name].shape) == s2:
            out.setdefault(name, (st[name], DIMS2))
    for name in PRODUCT_CARRIERS_3D:
        if name in st.fields and tuple(st[name].shape) == s3:
            out.setdefault(name, (st[name], DIMS3))
    if getattr(ctx, "slab", None) is None:
        for name, src in SOIL_CARRIERS.items():
            if src in st.fields and st[src].ndim == 3 and tuple(st[src].shape[1:]) == s2:
                out.setdefault(name, (st[src], DIMS_SOIL))
    drag = _dragco(ctx, scope)
    if drag is not None:
        out["DRAGCO"] = (drag, DIMS2)
    # SPD10MAX: io-max-fields keeps the hour's largest U10**2 + V10**2, as MAX_FIELDS_HR does (MAGW2, stored as is,
    # module_DIAGNOSE.F90:2460-2465); NCEP's post never reads it (it reads U10MAX/V10MAX), so the native history
    # carries m2 s-2 under a speed name, and the 3 km nests wrote values over 3,000 (Helene f57: 2670 = 51.7 m/s).
    # The history writes the speed in m s-1, the square root. A cell with no sample since the reset still holds the
    # -999 start value; the start frame (NTIMESTEP 0, written before the first step's hourly reset, so the state there
    # still holds its allocation zeros) has no sample either. Both are written as the 10 m speed of that instant (the
    # maximum over a window holding one time), and U10MAX/V10MAX below as U10/V10.
    nosample = None
    if "SPD10MAX" in st.fields and tuple(st["SPD10MAX"].shape) == s2 and "U10" in st.fields and "V10" in st.fields:
        spd = st.add("H_SPD10MAX", s2)
        with scope:
            nosample = (st["SPD10MAX"] < cp.float32(0.0)) | (int(ctx.ntimestep) == 0)
            u10g, v10g = st["U10"], st["V10"]
            spd[...] = cp.sqrt(cp.where(nosample, u10g * u10g + v10g * v10g, st["SPD10MAX"]))
        out["SPD10MAX"] = (spd, DIMS2)
    if "U10MAX" in st.fields and "V10MAX" in st.fields and tuple(st["U10MAX"].shape) == s2:
        # NMMB U10/V10 are H-point, grid-relative (io_wind_rotate); io_max_fields keeps the pair of the hour's
        # strongest 10 m wind at the same cell, so the extreme is rotated with the same H-point coefficients
        from namgpu.io.io_wind_rotate import h_coefficients
        he_e, he_n, hn_e, hn_n = h_coefficients(ctx)
        ue = st.add("H_UE10MAX", s2)
        ve = st.add("H_VE10MAX", s2)
        u10x, v10x = st["U10MAX"], st["V10MAX"]
        with scope:
            if nosample is not None:
                u10x = cp.where(nosample, st["U10"], u10x)
                v10x = cp.where(nosample, st["V10"], v10x)
            ue[...] = u10x * he_e + v10x * hn_e
            ve[...] = u10x * he_n + v10x * hn_n
        out["U10MAX"] = (ue, DIMS2)
        out["V10MAX"] = (ve, DIMS2)
    # The total vertical velocity W + BARO (VSOUND), which NMMB's history writes as 'vvel' (module_WRITE_ROUTINES
    # RUNHISTORY: W_TOT -> vvel) and NAM's post turns into DZDT and omega; the state holds LM+1 levels, VSOUND
    # writes the first LM (the mid layers).
    if "W_TOT" in st.fields and st["W_TOT"].shape[0] >= lm:
        out["W_TOT"] = (st["W_TOT"][:lm], DIMS3)
    # The model's own (nonhydrostatic) interface pressures: NAM's post integrates its heights over them
    out["PINT"] = (st.PINT, DIMS3S)
    if "Q2" in st.fields and tuple(st["Q2"].shape) == s3:
        tke = st.add("H_TKE_PBL", s3)
        with scope:
            tke[...] = st["Q2"] * cp.float32(0.5)       # MYJ Q2 = 2 TKE (parm/nam_phy_state.txt)
        out["TKE_PBL"] = (tke, DIMS3)
    if "REFL_10CM" in st.fields:
        refl = st["REFL_10CM"]
        comp = st.add("H_COMPOSITE", s2)
        with scope:
            comp[...] = refl.max(axis=0)
        out["REFL_10CM"] = (refl, DIMS3)
        out["COMPOSITE"] = (comp, DIMS2)
    if handoff_on(ctx.cfg):
        out.update(handoff_fields(ctx, scope))
    return out


# ---- parent hand-off field set (configfile key history_handoff: true) ---------------------------------------------
# What a one-way nest started from this history needs beyond the atmosphere (namgpu/init/parent_start.py, the
# 'parent' SOURCES row): the land and water surface state. WRF wrfout names and units, so the file still reads as a
# wrfout. Off (the default) the field list, and so the file, is unchanged.
#   TSK     TSKIN (K)                     SST     SST (K)                      SEAICE  SICE (fraction)
#   TSLB    STC [4 soil layers] (K)       SMOIS   SMC (m3 m-3)                 SH2O    SH2O liquid (m3 m-3)
#   SNOW    SNO (kg m-2, the state's mm)  SNOWH   SI * 0.001 (m; SI is mm)     CANWAT  CMC * 1000 (kg m-2; CMC is m)
#   TMN     TG deep soil (K)              LANDMASK 1 - SM (1 = land)           SNOWC   SNOWC (fraction)
# Soil fields are on soil_layers_stag, layer 1 at the top (NMMB order, Noah 0-10/10-40/40-100/100-200 cm): the
# writer flips only bottom_top / bottom_top_stag fields (history.LEVEL_DIMS).
HANDOFF_KEY = "history_handoff"
DIMS_SOIL = ("soil_layers_stag", "south_north", "west_east")
HANDOFF_FIELDS = ("TSK", "SST", "SEAICE", "TSLB", "SMOIS", "SH2O", "SNOW", "SNOWH", "CANWAT", "TMN",
                  "LANDMASK", "SNOWC")


def handoff_on(cfg) -> bool:
    raw = getattr(cfg, "raw", None)
    v = raw.get(HANDOFF_KEY, False) if raw is not None else False
    if isinstance(v, str):
        return v.strip().strip(".").lower() in ("true", "t", "yes", "y", "1")
    return bool(v)


def handoff_fields(ctx, scope) -> dict:
    st = ctx.st
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    if getattr(ctx, "slab", None) is not None:
        # tools/history_merge.py (several cards) writes no soil_layers_stag dimension and io_history._write_slab
        # flips every 3-D field, so the soil layers would come out reversed or the merge would fail.
        raise RuntimeError("io-history-fields: history_handoff is one-card only for now (the several-card merge "
                           "writer has no soil_layers_stag dimension and flips every 3-D field)")
    need = ("TSKIN", "SST", "SICE", "STC", "SMC", "SH2O", "SNO", "SI", "CMC", "TG", "SM", "SNOWC")
    miss = [n for n in need if n not in st.fields]
    if miss:
        raise RuntimeError(f"io-history-fields: history_handoff needs {miss} in the state (init-native-sfc)")
    s2 = (jm, im)
    snowh = st.add("H_SNOWH", s2)
    canwat = st.add("H_CANWAT", s2)
    land = st.add("H_LANDMASK", s2)
    with scope:
        snowh[...] = st["SI"] * cp.float32(0.001)
        canwat[...] = st["CMC"] * cp.float32(1000.0)
        land[...] = cp.float32(1.0) - st["SM"]
    return {
        "TSK": (st["TSKIN"], DIMS2), "SST": (st["SST"], DIMS2), "SEAICE": (st["SICE"], DIMS2),
        "TSLB": (st["STC"], DIMS_SOIL), "SMOIS": (st["SMC"], DIMS_SOIL), "SH2O": (st["SH2O"], DIMS_SOIL),
        "SNOW": (st["SNO"], DIMS2), "SNOWH": (snowh, DIMS2), "CANWAT": (canwat, DIMS2),
        "TMN": (st["TG"], DIMS2), "LANDMASK": (land, DIMS2), "SNOWC": (st["SNOWC"], DIMS2),
    }


def handoff_attrs(ctx) -> dict:
    """Global attributes a child reads to place itself in this grid (parent_start.parent_grid)."""
    cfg, g = ctx.cfg, ctx.grid
    return {"HANDOFF": np.int32(1), "PARENT_TPH0D": float(cfg.tph0d), "PARENT_TLM0D": float(cfg.tlm0d),
            "PARENT_WBD": float(cfg.wbd), "PARENT_SBD": float(cfg.sbd),
            "PARENT_DLMD": float(cfg.dlmd), "PARENT_DPHD": float(cfg.dphd),
            "PARENT_IM": np.int32(g.im), "PARENT_JM": np.int32(g.jm), "PARENT_LM": np.int32(g.lm),
            "PARENT_LNSH": np.int32(cfg.lnsh), "PARENT_LNSV": np.int32(cfg.lnsv),
            "PARENT_PT": float(g.pt), "PARENT_PDTOP": float(g.pdtop), "PARENT_DT": float(cfg.dt),
            **_start_credit(ctx)}


def _start_credit(ctx) -> dict:
    """START_SOURCE, DATA_CREDIT, DATA_LICENCES of this run (the door's NAMGPU_START_SOURCE / NAMGPU_DATA_* env, else
    nest.cli.data_credits of ctx.init_source): a child nest's maps carry its parent's start-source credit from these
    (a nest under an IFS-started parent showed no ECMWF CC BY credit; fix round 2)."""
    import json
    import os
    src = os.environ.get("NAMGPU_START_SOURCE") or str(getattr(ctx, "init_source", "") or "")
    lic = os.environ.get("NAMGPU_DATA_LICENCES")
    credit = os.environ.get("NAMGPU_DATA_CREDIT", "")
    if not lic and src and src != "parent":
        from namgpu.nest.cli import data_credits
        credit, lines = data_credits(src)
        lic = json.dumps(lines)
    if not lic:
        return {}
    return {"START_SOURCE": src, "DATA_CREDIT": credit, "DATA_LICENCES": lic}
