"""init-gfs-vinterp: pressure-level source (GFS) -> NMMB hybrid levels, as nemsinterp does.

Ported from NOAA NAM nemsinterp (nam.v4.2.11, 83df4e13c1): caller lines 423-1010 of
sorc/nam_nps.fd/sorc/NMMB_init/NPS/nemsinterp/src/module_vinterp_routines.F90 (surface pressure,
PD/PINT/P3D, V-point pressures, T, Q, condensate CWM, U/V with wind_adjust), kernel
namgpu/kernels/init_gfs_vinterp.cu (one thread per column, one launch per routine).

Reads (ctx.tables["gfs_hinterp"], built by init_gfs_hinterp.full_domain_inputs from the decoded
init-gfs-decode set, metgrid level order, index 0 = lowest level, shape [nlev, JM, IM]):
  PRES_IN, GHT_IN, TT_IN, RH_IN, UU_IN, VV_IN, SNMR_IN, CLWMR_IN, RWMR_IN, CICE_IN, FRIMEF_IN.
  HGT_M [JM, IM] (model terrain, m) is read from the state (st["HGT_M"]); it is clamped in place
  when the input is in reverse order, as the source does.
  The log interpolation clamps these inputs in place to 1e-13, as the source does.
  Missing inputs are refused (RuntimeError naming the owner). No random or fallback data is built.
Writes ctx.st: PD [JM,IM], PINT [LM+1,JM,IM], T, Q, CW, U, V [LM,JM,IM], FIS [JM,IM], and scratch
  PSFC_OUT, PSFC_OUTV, PDVP [JM,IM]; P3D_OUT, P3DV_OUT, CWM_TMP,
  FRIMEF_LEV [LM,JM,IM]; P3DV_IN, QTMP2 [nlev,JM,IM].

Gates, each as in the source:
  spectral     .false. only (the spectral branches are not ported; refused otherwise).
  hyb_coor     reverse-order input (GHT(1,1,10) >= GHT(1,1,11)) reverses GHT, PRES, TT, UU, VV, RH and
               clamps HGT_M to GHT(2) +/- 150 m; wind_adjust runs only then (caller line 767).
  do_clouds    CWM and the condensate/FRIMEF interpolations run only for do_clouds AND maxval(CLWMR)>0;
               otherwise CW = 0 (source "CWM definition").
  GHT(1)       GHT(1) = SOILHGT where PRES(1) != 200100, HGT_M == GHT(1) and HGT_M != 0 (source 397-405).
  surface      compute_nmm_surfacep STOPs when PSFC_IN == 0; the same condition is checked here.
  B-grid east  U, V at IDE-1 are zeroed when gtype != 'A' and |IDE-1-ITE| <= 1 (GTYPE below).
Launches only on the step path: the few host reads are scalar gates at init, not per step.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.init import init_gfs_hinterp as hin

UNIT = "init_gfs_vinterp"
G_FORTRAN = np.float32(9.81)    # module_vinterp_routines.F90: REAL,PARAMETER :: G=9.81; FIS = HGT_M*G
VCOORD = 1                      # parm/nam_namelist.nps vcoord = 1 (PD = PSFC - PT)
DEPTH_REPLACE = np.float32(5000.0)
GTYPE = "E"                     # NAM NMMB parent grid is an E grid; not read from the domain setup yet
PSFC_SENTINEL = 200100.0        # PRES(1) value that marks the sea-level / surface entry in the source
REVERSED_KEYS = ("GHT_IN", "PRES_IN", "TT_IN", "UU_IN", "VV_IN", "RH_IN")   # reverse_vert_coord calls


def _dev_vec(ctx, key: str, host):
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _fortran_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().strip(".").lower() in ("true", "t", "yes", "y")
    return bool(v)


def run(ctx):
    import cupy as cp  # noqa: F401  (device arrays are cupy)

    g = ctx.grid
    cfg = ctx.cfg
    st = ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    if _fortran_bool(cfg.raw.get("spectral", False)):
        raise RuntimeError(
            "init-gfs-vinterp: spectral=.true. is not ported. The spectral branches (module_vinterp_routines.F90 "
            "422-433 surfp on PINT, 534-540 T, 701-730 winds, 842-847 Q with SPECHUMD) do not exist here. "
            "The parent config is spectral=.false.; refusing the spectral run.")
    if "HGT_M" not in st.fields:
        raise RuntimeError(
            "init-gfs-vinterp: model terrain HGT_M is not in the state. No unit in the tree stages it yet "
            "(owner: init-native-sfc / static terrain from the NAM bgrd file). Refusing to run on a missing terrain.")
    terrain = st["HGT_M"]

    gi = ctx.tables.get("gfs_hinterp")
    if gi is None:
        gi = hin.full_domain_inputs(ctx)
        ctx.tables["gfs_hinterp"] = gi
    nlev = int(gi["PRES_IN"].shape[0])
    if nlev < 11:
        raise RuntimeError(f"init-gfs-vinterp: {nlev} input levels; the hyb_coor test reads level 11.")

    # hyb_coor (source 185-189): ground-up file order unless GHT(1,1,10) >= GHT(1,1,11)
    hyb_coor = not bool(gi["GHT_IN"][9, 0, 0] < gi["GHT_IN"][10, 0, 0])
    if hyb_coor:
        for key in REVERSED_KEYS:
            gi[key] = cp.ascontiguousarray(gi[key][::-1])
        # source 224-245: HGT_M limited to GHT(2) +/- 150 m (reversed order, so level 2 is index 1)
        ref = gi["GHT_IN"][1]
        terrain[...] = cp.minimum(cp.maximum(terrain, ref - 150.0), ref + 150.0)

    pres = gi["PRES_IN"]
    ght = gi["GHT_IN"]
    tt = gi["TT_IN"]

    # GHT(1) = SOILHGT swap (source 397-405), before surfp; in place on GHT as the source does
    if "SOILHGT" not in st.fields:
        raise RuntimeError(
            "init-gfs-vinterp: SOILHGT is not in the state. The GHT(1) = SOILHGT swap (source 397-405) "
            "changes Z3D_IN(1) for every cell with HGT_M == GHT(1). No unit in the tree stages SOILHGT yet "
            "(owner: init-native-sfc, from the NAM bgrd file). Refusing to run without it.")
    swap = (pres[0] != PSFC_SENTINEL) & (terrain == ght[0]) & (terrain != 0.0)
    ght[0] = cp.where(swap, st["SOILHGT"], ght[0])

    first = bool(ctx.first_step)
    skip_interior = 0 if first else 1
    lnsh = int(cfg.lnsh)
    lnsv = int(cfg.lnsv)
    ide, jde = im + 1, jm + 1
    pdtop = np.float32(g.pdtop)
    pt = np.float32(g.pt)
    # B-grid eastern boundary zeroing (source 790-800): gtype != 'A' and |IDE-1-ITE| <= 1, ITE = IM here
    zero_east = 1 if (GTYPE != "A" and abs(ide - 1 - im) <= 1) else 0

    # scratch, allocated once (st.add returns the existing array on later calls)
    psfc = st.add("PSFC_OUT", (jm, im))
    psfc_v = st.add("PSFC_OUTV", (jm, im))
    pdvp = st.add("PDVP", (jm, im))
    p3d = st.add("P3D_OUT", (lm, jm, im))
    p3dv_in = st.add("P3DV_IN", (nlev, jm, im))
    p3dv_out = st.add("P3DV_OUT", (lm, jm, im))
    qtmp2 = st.add("QTMP2", (nlev, jm, im))
    cwm_tmp = st.add("CWM_TMP", (lm, jm, im))
    rimef_lev = st.add("FRIMEF_LEV", (lm, jm, im))

    sgml1 = _dev_vec(ctx, "vi_sgml1", g.sgml1)
    sgml2 = _dev_vec(ctx, "vi_sgml2", g.sgml2)
    sg1 = _dev_vec(ctx, "vi_sg1", g.sg1)
    sg2 = _dev_vec(ctx, "vi_sg2", g.sg2)

    # surfp STOP (source 1498-1501): PSFC_IN == 0 aborts; same selection of PSFC_IN as the kernel
    psfc_in_chk = cp.where(pres[0] != PSFC_SENTINEL, pres[0], pres[1])
    if bool(cp.any(psfc_in_chk == 0.0)):
        raise RuntimeError("init-gfs-vinterp: PSFC_IN == 0 at a point; compute_nmm_surfacep STOPs there "
                           "(module_vinterp_routines.F90 1498-1501). Refusing to continue.")

    grid_c = kernels.grid_cols(im, jm)
    blk_c = kernels.BLOCK_COL
    grid_2d_lev = kernels.grid_2d(im, jm, nlev)
    grid_2d_lm = kernels.grid_2d(im, jm, lm)
    blk_2d = kernels.BLOCK_2D
    i32 = np.int32
    f32 = np.float32

    def K(name):
        return kernels.get(UNIT, name)

    fis = st["FIS"] if "FIS" in st.fields else st.add("FIS", (jm, im))
    fis[...] = terrain * G_FORTRAN

    # CWM gate (source "CWM definition"): do_clouds AND maxval(CLWMR) > 0
    cloud_on = bool(cfg.do_clouds) and float(cp.max(gi["CLWMR_IN"])) > 0.0

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        # 1. surface pressure at the model terrain (compute_nmm_surfacep, non-spectral)
        K("init_gfs_vinterp_surfp")(grid_c, blk_c, (
            terrain, pres, ght, tt, psfc, i32(im), i32(jm), i32(nlev)))
        # 2. PD, PINT, P3D on H points (compute_3d_pressure)
        K("init_gfs_vinterp_pd")(grid_c, blk_c, (
            psfc, sgml1, sgml2, sg1, sg2, st["PD"], p3d, st["PINT"],
            i32(im), i32(jm), i32(lm), f32(pdtop), f32(pt), i32(VCOORD), i32(1)))
        # 3. V-point surface pressure, PD and input pressures (AGRID block)
        K("init_gfs_vinterp_vpoints")(grid_c, blk_c, (
            psfc, st["PD"], pres, psfc_v, pdvp, p3dv_in, i32(im), i32(jm), i32(nlev)))
        # 4. V-point mid-level pressure; compute_3d_pressure recomputes PDVP from PSFC_OUTV
        K("init_gfs_vinterp_pd")(grid_c, blk_c, (
            psfc_v, sgml1, sgml2, sg1, sg2, pdvp, p3dv_out, st["PINT"],
            i32(im), i32(jm), i32(lm), f32(pdtop), f32(pt), i32(VCOORD), i32(0)))
        # 5. temperature: linear in log p, extrapolated, direct T (direct_temp = .TRUE.)
        K("init_gfs_vinterp_lin")(grid_c, blk_c, (
            pres, p3d, tt, st["T"], i32(im), i32(jm), i32(nlev), i32(lm),
            i32(1), i32(1), i32(1), i32(skip_interior), i32(lnsh), i32(ide), i32(jde)))
        # 6. humidity: RH -> mixing ratio -> specific humidity, log-p interpolation
        K("init_gfs_vinterp_rhq")(grid_2d_lev, blk_2d, (
            gi["RH_IN"], tt, pres, qtmp2, i32(im), i32(jm), i32(nlev)))
        K("init_gfs_vinterp_log")(grid_c, blk_c, (
            pres, p3d, qtmp2, st["Q"], i32(im), i32(jm), i32(nlev), i32(lm),
            i32(0), i32(1), i32(skip_interior), i32(lnsh), i32(ide), i32(jde)))
        # 7. condensates (only under the CWM gate); otherwise CWM = 0 (source else branch)
        if cloud_on:
            for n_sp, src in enumerate(("SNMR_IN", "CLWMR_IN", "RWMR_IN", "CICE_IN")):
                K("init_gfs_vinterp_log")(grid_c, blk_c, (
                    pres, p3d, gi[src], cwm_tmp, i32(im), i32(jm), i32(nlev), i32(lm),
                    i32(0), i32(1), i32(skip_interior), i32(lnsh), i32(ide), i32(jde)))
                K("init_gfs_vinterp_cwm_acc")(grid_2d_lm, blk_2d, (
                    cwm_tmp, st["CW"], i32(1 if n_sp == 0 else 0),
                    i32(im), i32(jm), i32(lm)))
            K("init_gfs_vinterp_lin")(grid_c, blk_c, (
                pres, p3d, gi["FRIMEF_IN"], rimef_lev, i32(im), i32(jm), i32(nlev), i32(lm),
                i32(0), i32(1), i32(0), i32(skip_interior), i32(lnsh), i32(ide), i32(jde)))
        else:
            st["CW"][...] = 0.0
        # 8. winds on V points: linear in log p with no extrapolation; wind_adjust only when hyb_coor
        #    (source 767); the B-grid eastern-boundary zeroing runs regardless
        K("init_gfs_vinterp_lin")(grid_c, blk_c, (
            p3dv_in, p3dv_out, gi["UU_IN"], st["U"], i32(im), i32(jm), i32(nlev), i32(lm),
            i32(0), i32(1), i32(0), i32(skip_interior), i32(lnsv), i32(ide), i32(jde)))
        K("init_gfs_vinterp_lin")(grid_c, blk_c, (
            p3dv_in, p3dv_out, gi["VV_IN"], st["V"], i32(im), i32(jm), i32(nlev), i32(lm),
            i32(0), i32(1), i32(0), i32(skip_interior), i32(lnsv), i32(ide), i32(jde)))
        K("init_gfs_vinterp_wind")(grid_c, blk_c, (
            p3dv_in, p3dv_out, gi["UU_IN"], gi["VV_IN"], st["U"], st["V"],
            i32(im), i32(jm), i32(nlev), i32(lm), DEPTH_REPLACE,
            i32(1 if hyb_coor else 0), i32(zero_east)))
    return st
