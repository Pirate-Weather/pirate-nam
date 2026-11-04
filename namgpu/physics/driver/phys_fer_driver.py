"""phys-fer-driver: GSMDRIVE glue for NPRECIP microphysics on the fer_hires route.

Fortran: nmm/module_MICROPHYSICS.F90:60-558 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE GSMDRIVE,
called from SOLVER_RUN every NPRECIP steps (module_SOLVER_GRID_COMP.F90:5880, NPHS argument =
NPRECIP, DT = int_state%DT). Only the `fer_hires` case is wired here; the other MICROPHYSICS
cases are other units.
Kernels: namgpu/kernels/phys_fer_driver.cu (phys_fer_driver_prep, phys_fer_driver_post,
         phys_fer_driver_precip). One launch each, no host sync, no per-step allocation after the
         first call.

Sequence (one call per microphysics step), all on ctx.stream:
  1. prep kernel: DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY from T, Q, PD; LOWLYR, RAINNC, RAINNCV
     zeroed; AVRAIN += 1.
  2. FER_HIRES: the column microphysics unit phys-fer-fer-hires, called with every argument the
     Fortran call binds (module_MICROPHYSICS.F90:363-382):
        DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY (in-place), Q, CW (the Fortran QT argument, in-place),
        QC, QR, QS, F_ICE, F_RAIN, F_RIMEF (in-place), LOWLYR, SR (out), RAINNC and RAINNCV (in-place),
        REFL_10CM (out), MPRATES (4-D, in-place, D_SS = 1), MP_RESTART_STATE, TBPVS, TBPVS0 (from the
        Ferrier init tables), RHGRD (cfg.rhgrd), DT = DTPHS = NPRECIP*DT, and the tile ITSLOC..ITELOC,
        JTSLOC..JTELOC from the GSMDRIVE limits (below).
  3. post kernel: TRAIN and T update from TH_PHY and PI_PHY (USE_RADAR branch optional).
  4. precip kernel: PREC, ACPREC, ACPREC_TOT from RAINNCV.

Reads:   T, Q, PD, DSG2, SGML2 (state tables, 1-D DSG2/SGML2 are [LM]); PDSG1, PSGML1 (grid
         vertical constants, cached in ctx.tables under "pdsg1"/"psgml1" so the vertical-coordinate
         recovery drops them and they are re-uploaded); AVRAIN, PREC, ACPREC, ACPREC_TOT, TRAIN
         (state); DFI_TTEN (state, used only with use_radar); the FER_HIRES state from ctx.tables["fer"]
         (built once by phys-fer-init, cold-start F_ICE/F_RAIN/F_RIMEF fill included).
Writes:  T, TRAIN, PREC, ACPREC, ACPREC_TOT, AVRAIN (state); QC, QR, QS, F_*, CW, SR, REFL_10CM,
         MPRATES (state, written by FER_HIRES); scratch DZ8W, RHO_PHY, P_PHY, PI_PHY, TH_PHY, LOWLYR,
         RAINNC, RAINNCV, MPRATES (st.add, allocated on first call).
Constants (Fortran REAL, single precision on the host):
         DTPHS = NPRECIP*DT (GSMDRIVE line 199), RDTPHS = 1/DTPHS (line 200),
         D_SS = 1 (int_state%D_SS, module_SOLVER_INTERNAL_STATE.F90:859),
         RHGRD = cfg.rhgrd (0.98, parm/nam_configfile).
Tile limits (GSMDRIVE lines 335-338, the FER_HIRES call passes ITS=ITSLOC, ITE=ITELOC, ...):
         ITS_B1 = MAX(ITS, IDS+1), ITE_B1 = MIN(ITE, IDE-1); ITSLOC = MAX(ITS_B1, IDS),
         ITELOC = MIN(ITE_B1, IDE-1); same in J. Single card, full domain: IDS = ITS = 1, IDE = ITE = IM,
         giving 2..IM-1 (and 2..JM-1). ITIMESTEP, DX, DY are not read by FER_HIRES and are not bound.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.fer_hires import phys_fer_fer_hires, phys_fer_init

UNIT = "phys_fer_driver"
K_PREP = "phys_fer_driver_prep"
K_POST = "phys_fer_driver_post"
K_PRECIP = "phys_fer_driver_precip"

D_SS = 1  # int_state%D_SS (module_SOLVER_INTERNAL_STATE.F90:859); the MPRATES fourth dimension

_SCRATCH3 = ("DZ8W", "RHO_PHY", "P_PHY", "PI_PHY", "TH_PHY")


def _vcoord(ctx, attr: str):
    """Upload a host grid vertical constant once and cache it in ctx.tables under its lowercase name.

    Cached in ctx.tables (not st.fields) on purpose: init_vcoord_recover.py drops the lowercase
    pdsg1/psgml1 keys when the vertical coordinate is recovered, so the next call re-uploads them."""
    key = attr.lower()
    if key in ctx.tables:
        return ctx.tables[key]
    host = getattr(ctx.grid, attr, None)
    if host is None:
        raise RuntimeError(f"phys-fer-driver: grid constant {attr!r} is not set on the grid")
    import cupy as cp
    ctx.tables[key] = cp.asarray(np.asarray(host, dtype=np.float32))
    return ctx.tables[key]


def run(ctx, use_radar: bool = False, nprecip: int | None = None):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    nphs = int(cfg.nprecip if nprecip is None else nprecip)
    dtphs = np.float32(nphs) * np.float32(ctx.dt)
    rdtphs = np.float32(1.0) / dtphs
    rhgrd = np.float32(cfg.rhgrd)

    # Ferrier tables and restart state: built once (FERRIER_INIT_hr); a per-call run() would refill F_ICE.
    fer = ctx.tables.get("fer")
    if fer is None:
        phys_fer_init.run(ctx)
        fer = ctx.tables["fer"]

    # GSMDRIVE tile limits (module_MICROPHYSICS.F90:335-338), single card, IDS = ITS = 1, IDE = ITE = IM/JM.
    ids, ide, jds, jde = 1, im, 1, jm
    its, ite, jts, jte = 1, im, 1, jm
    its_b1, ite_b1 = max(its, ids + 1), min(ite, ide - 1)
    jts_b1, jte_b1 = max(jts, jds + 1), min(jte, jde - 1)
    itsloc, iteloc = max(its_b1, ids), min(ite_b1, ide - 1)
    jtsloc, jteloc = max(jts_b1, jds), min(jte_b1, jde - 1)

    s3 = (lm, jm, im)
    s2 = (jm, im)
    dz8w, rho_phy, p_phy, pi_phy, th_phy = (st.add(n, s3) for n in _SCRATCH3)
    lowlyr = st.add("LOWLYR", s2, dtype="int32")
    rainnc = st.add("RAINNC", s2)
    rainncv = st.add("RAINNCV", s2)
    # The state table's MPRATES row is rank 4 with the tracer count in place of D_SS; FER_HIRES needs D_SS.
    mp_shape = (D_SS, lm, jm, im)
    if "MPRATES" not in st.fields or tuple(st.fields["MPRATES"].shape) != mp_shape:
        st.fields["MPRATES"] = st.xp.zeros(mp_shape, dtype=np.float32)
    mprates = st["MPRATES"]

    dsg2 = st["DSG2"]
    sgml2 = st["SGML2"]
    pdsg1 = _vcoord(ctx, "pdsg1")
    psgml1 = _vcoord(ctx, "psgml1")

    block = kernels.BLOCK_2D
    grid3 = kernels.grid_2d(im, jm, lm)
    grid2 = kernels.grid_2d(im, jm, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        # 1. column preparation (GSMDRIVE lines 216-249 and the zeroing/AVRAIN statements)
        kernels.get(UNIT, K_PREP)(grid3, block, (
            np.int32(im), np.int32(jm), np.int32(lm),
            st["PD"], st["T"], st["Q"], dsg2, sgml2, pdsg1, psgml1,
            dz8w, rho_phy, p_phy, pi_phy, th_phy,
            st["AVRAIN"], rainnc, rainncv, lowlyr,
        ))

        # 2. FER_HIRES (its own unit; in-place on Q, CW, TH_PHY, QC, QR, QS, F_*, RAINNC(V); SR, REFL_10CM out)
        phys_fer_fer_hires.run(
            ctx,
            dz8w=dz8w, rho_phy=rho_phy, p_phy=p_phy, pi_phy=pi_phy, th_phy=th_phy,
            q=st["Q"], qt=st["CW"], qc=st["QC"], qr=st["QR"], qs=st["QS"],
            f_ice=st["F_ICE"], f_rain=st["F_RAIN"], f_rimef=st["F_RIMEF"],
            refl_10cm=st["REFL_10CM"],
            lowlyr=lowlyr, rainnc=rainnc, rainncv=rainncv, sr=st["SR"],
            mprates=mprates, mp_state=fer["mp_restart"],
            tbpvs=fer["tbpvs"], tbpvs0=fer["tbpvs0"],
            dt=float(dtphs), rhgrd=float(rhgrd), d_ss=D_SS,
            its=itsloc, ite=iteloc, jts=jtsloc, jte=jteloc,
        )

        # 3. TRAIN and T update (lines 503-531), B1 interior
        kernels.get(UNIT, K_POST)(grid3, block, (
            np.int32(im), np.int32(jm), np.int32(lm),
            th_phy, pi_phy, st["DFI_TTEN"], st["T"], st["TRAIN"],
            rdtphs, np.int32(1 if use_radar else 0),
        ))

        # 4. precipitation accumulation (lines 547-562), B1 interior
        kernels.get(UNIT, K_PRECIP)(grid2, block, (
            np.int32(im), np.int32(jm),
            rainncv, st["PREC"], st["ACPREC"], st["ACPREC_TOT"],
        ))

    return {"DZ8W": dz8w, "RHO_PHY": rho_phy, "P_PHY": p_phy, "PI_PHY": pi_phy,
            "TH_PHY": th_phy, "LOWLYR": lowlyr, "RAINNC": rainnc, "RAINNCV": rainncv,
            "MPRATES": mprates}
