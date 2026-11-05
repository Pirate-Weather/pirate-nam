"""phys-bmj-driver: CUCNVC glue, run every NPRECIP steps (BMJ convection branch).

Ported from nmm/module_CONVECTION.F90:48-658 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE CUCNVC,
the CONVECTION == 'bmj' path. Kernels: namgpu/kernels/phys_bmj_driver.cu (three launches:
rad_reset, prep, post). The BMJ call in the middle is unit phys-bmj-bmjdrv, run through its own
wrapper ``namgpu.physics.bmj.phys_bmj_bmjdrv.run(ctx)``.

Order of one call (CONTRACT section 2, item 18):
  1. rad_reset  on NRADS/NRADL multiples (Fortran runs this before the NCNVC early return)
  2. return     unless NTSD % NCNVC == 0
  3. prep       counters, PHINT/PHMID/RR/EXNER/TH, U_PHY/V_PHY, DZ, zeroed tendencies, CUTOP/CUBOT
  4. bmjdrv     the BMJ column scheme: reads the ctx.st names below, writes RAINCV, CUTOP, CUBOT,
                RTHCUTEN, RQCUTEN, DUDT_PHY, DVDT_PHY (and CLDEFI, CU_ACT_FLAG)
  5. post       output-window reset (MNTO condition), T/Q/TCUCN update, precipitation, cloud top/base

Inputs read from ctx.st (names are the NMMB table names; LPBL is written by the PBL unit):
  PD, T, Q, U, V, FIS, SM, LPBL (int32), TCUCN, AVCNVC, ACUTIM, HTOP, HBOT, CUPPT, CNVBOT,
  CNVTOP, HBOTD, HTOPD, HBOTS, HTOPS, PREC, ACPREC, ACPREC_TOT, CUPREC, CPRATE.
  Vertical constants SG2, PSG1 [LM+1] and SGML2, PSGML1 [LM] from ctx.grid, uploaded once to
  ctx.tables["bmj_vert"].
Scratch added with st.add (the bmjdrv unit reads them by these names):
  PHINT [LM+1], PHMID, EXNER, TH, RR, DZ, U_PHY, V_PHY, RTHCUTEN, RQCUTEN [LM]; RAINCV, RAINC,
  NCA, XLAND, SFCZ, CUTOP, CUBOT [JM, IM] float32; KPBL [JM, IM] int32.
  DUDT_PHY, DVDT_PHY [LM] are not new arrays: they are ctx.st "DUDT", "DVDT" (aliased, see
  _scratch), the H-point tendencies that drv-h-to-v-tend reads after this call.

Scope and refusals:
  - CU_PHYSICS gate (Fortran 445-447): the bmjdrv call runs only when CU_PHYSICS /= 0, and only
    BMJSCHEME (2) is ported. CU_PHYSICS is ctx.tables["cu_physics"], set by BMJ_INIT (Fortran
    10941-10946, drv-physics-initialize). With CU_PHYSICS 0, prep and post still run (both sit
    outside the Fortran IF), so RAINCV stays 0 and CUTOP/CUBOT keep the prep values.
  - convection must be 'bmj' (nam_configfile `convection: bmj`). kf, sas and sashur would run
    with no convective tendency from this glue; their branches are not ported, so the wrapper
    raises instead of running silently.
  - LPBL must be in ctx.st. Without it the KPBL copy would be zeros and bmjdrv would search the
    wrong layers, so the wrapper raises.
  - The radiation reset is done only when this wrapper is called. The driver calls it on
    NPRECIP multiples; NRADS and NRADL (48) are multiples of NPRECIP (2), so every reset step is
    a call step.

Launch: one kernel per routine, on ctx.stream, no host sync, no per-step allocation (scratch is
added once). Scalars are formed on the host only from the step counter and the config.
"""
from __future__ import annotations

import contextlib
import importlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_driver"
BMJ_UNIT = "namgpu.physics.bmj.phys_bmj_bmjdrv"

# Scratch arrays this glue writes and bmjdrv reads (added once with st.add).
BMJSCHEME = 2   # CU_PHYSICS selector, Fortran CASE (BMJSCHEME) at module_CONVECTION.F90:449

SCRATCH_LMP1 = ("PHINT",)
SCRATCH_LM = ("PHMID", "EXNER", "TH", "RR", "DZ", "U_PHY", "V_PHY",
              "RTHCUTEN", "RQCUTEN")
SCRATCH_2D = ("RAINCV", "RAINC", "NCA", "XLAND", "SFCZ", "CUTOP", "CUBOT")


def _vertical(ctx):
    """SG2, PSG1 [LM+1] and SGML2, PSGML1 [LM] as float32 device vectors, uploaded once."""
    if "bmj_vert" not in ctx.tables:
        import cupy as cp
        g = ctx.grid
        ctx.tables["bmj_vert"] = {
            "SG2": cp.asarray(np.ascontiguousarray(g.sg2, dtype=np.float32)),
            "PSG1": cp.asarray(np.ascontiguousarray(g.psg1, dtype=np.float32)),
            "SGML2": cp.asarray(np.ascontiguousarray(g.sgml2, dtype=np.float32)),
            "PSGML1": cp.asarray(np.ascontiguousarray(g.psgml1, dtype=np.float32)),
        }
    return ctx.tables["bmj_vert"]


def _scratch(ctx):
    """Allocate the glue's scratch once (st.add returns the existing array on later calls)."""
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    for n in SCRATCH_LMP1:
        st.add(n, (lm + 1, jm, im))
    for n in SCRATCH_LM:
        st.add(n, (lm, jm, im))
    for n in SCRATCH_2D:
        st.add(n, (jm, im))
    # Fortran passes int_state%DUDT and int_state%DVDT as CUCNVC's DUDT_PHY and DVDT_PHY
    # (module_SOLVER_GRID_COMP.F90:5768). The H_TO_V_TEND bmj branch (5827-5830) reads them
    # back, and the NPHS TURBL call rewrites them, so one array per name, shared.
    st.add("DUDT", (lm, jm, im))
    st.add("DVDT", (lm, jm, im))
    st.fields["DUDT_PHY"] = st.fields["DUDT"]
    st.fields["DVDT_PHY"] = st.fields["DVDT"]
    st.add("KPBL", (jm, im), dtype="int32")


def _launch(ctx, name, args, n_cols):
    kern = kernels.get(UNIT, name)
    grid = kernels.grid_cols(n_cols[0], n_cols[1])
    with (ctx.stream or contextlib.nullcontext()):
        kern(grid, kernels.BLOCK_COL, args)


def _cu_physics(ctx) -> int:
    """CU_PHYSICS as set by BMJ_INIT (Fortran 10941-10946). Refuses when BMJ_INIT did not run."""
    value = ctx.tables.get("cu_physics")
    if value is None:
        raise RuntimeError(
            "phys-bmj-driver: CU_PHYSICS is not in ctx.tables; BMJ_INIT (drv-physics-initialize) "
            "must set it before the first CUCNVC call.")
    return int(value)


def _nint(x: np.float32) -> int:
    """Fortran NINT: nearest integer, halves away from zero (x > 0 here)."""
    return int(math.floor(float(x) + 0.5))


def _setup(ctx):
    """Whole-domain work before the row bands (namgpu/physics/band.py): DUDT/DVDT and their DUDT_PHY/DVDT_PHY
    aliases live in the whole State (drv-h-to-v-tend reads them after this call), and BMJ_DXH [JM] is uploaded
    whole so each band reads its own rows of it."""
    st, g = ctx.st, ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    st.add("DUDT", (lm, jm, im))
    st.add("DVDT", (lm, jm, im))
    st.fields["DUDT_PHY"] = st.fields["DUDT"]
    st.fields["DVDT_PHY"] = st.fields["DVDT"]
    from namgpu.physics.bmj import phys_bmj_bmjdrv
    phys_bmj_bmjdrv._dxh_device(st, g)


def run(ctx):
    """CUCNVC over row bands (band.py): BMJ columns are independent, so PHINT, PHMID, EXNER, TH, RR, DZ, U_PHY,
    V_PHY, RTHCUTEN and RQCUTEN exist for one band at a time. Same result bit for bit."""
    from namgpu.physics import band
    from namgpu.state import PHYS_SCRATCH, UNIT_SCRATCH
    uid = "phys-bmj-driver"
    return band.run(ctx, uid, _run, setup=_setup, scratch=PHYS_SCRATCH | UNIT_SCRATCH.get(uid, frozenset()),
                    keep_whole=("DUDT", "DVDT"), drop=UNIT_SCRATCH.get(uid, ()))


def _run(ctx):
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    if cfg.convection != "bmj":
        raise RuntimeError(
            f"phys-bmj-driver: convection={cfg.convection!r}; only 'bmj' is ported. kf, sas and "
            "sashur would run with no convective tendency from this glue (their branches are "
            "not ported).")
    if "LPBL" not in st.fields:
        raise RuntimeError(
            "phys-bmj-driver: LPBL is not in ctx.st. KPBL is copied from it; without the PBL "
            "top layer bmjdrv would search the wrong levels. The PBL unit (phys-turbl-driver) "
            "must write LPBL first.")

    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntsd = int(ctx.ntimestep)
    ncnvc = int(cfg.nprecip)
    nrads, nradl = int(cfg.nrads), int(cfg.nradl)
    ns = (im, jm)

    _scratch(ctx)
    vert = _vertical(ctx)
    i32 = np.int32
    f32 = np.float32

    # 1. Fortran 281-290: radiation-step reset, before the NCNVC early return.
    if ntsd % nrads == 0 or ntsd % nradl == 0:
        _launch(ctx, "phys_bmj_driver_rad_reset",
                (i32(im), i32(jm), i32(lm), st["HTOP"], st["HBOT"], st["CUPPT"]), ns)

    # 2. Fortran 292-293: only every NCNVC steps.
    if ntsd % ncnvc != 0:
        return

    # 3. prep (Fortran 335-436).
    _launch(ctx, "phys_bmj_driver_prep", (
        i32(im), i32(jm), i32(lm),
        st["PD"], st["T"], st["Q"], st["U"], st["V"], st["FIS"], st["SM"], st["LPBL"],
        vert["SG2"], vert["PSG1"], vert["SGML2"], vert["PSGML1"],
        st["PHINT"], st["PHMID"], st["EXNER"], st["TH"], st["RR"], st["DZ"],
        st["U_PHY"], st["V_PHY"], st["DUDT_PHY"], st["DVDT_PHY"],
        st["RTHCUTEN"], st["RQCUTEN"],
        st["RAINCV"], st["RAINC"], st["NCA"], st["XLAND"], st["SFCZ"],
        st["CUTOP"], st["CUBOT"], st["KPBL"],
        st["AVCNVC"], st["ACUTIM"],
    ), ns)

    # 4. Fortran 445-447: BMJ column scheme only when CU_PHYSICS /= 0 (unit phys-bmj-bmjdrv).
    # It reads the prep arrays above by name. Any other CU_PHYSICS value is not ported: refuse.
    cu_physics = _cu_physics(ctx)
    if cu_physics != 0:
        if cu_physics != BMJSCHEME:
            raise RuntimeError(
                f"phys-bmj-driver: CU_PHYSICS={cu_physics}; only BMJSCHEME (2) is ported. The "
                "Fortran CASE DEFAULT only prints a message and runs no scheme.")
        importlib.import_module(BMJ_UNIT).run(ctx)

    # 5. Output-window flags (Fortran 504-512), all scalar and in single precision.
    cf_hi = np.float32(cfg.raw.get("minutes_history", 60)) / np.float32(60.0)
    dt = np.float32(cfg.dt)
    n_out = _nint(np.float32(3600.0) * cf_hi / dt)
    if n_out <= 0:
        raise RuntimeError(f"phys-bmj-driver: N_TIMSTPS_OUTPUT={n_out} (minutes_history, dt)")
    mnto = ntsd % n_out
    out_reset = 1 if (mnto > 0 and mnto <= ncnvc) else 0

    # Convective running total since start (port addition; Fortran keeps only the NPREC bucket CUPREC).
    st.add("CUPREC_TOT", (jm, im))

    # post (Fortran 487-589).
    _launch(ctx, "phys_bmj_driver_post", (
        i32(im), i32(jm), i32(lm), i32(ncnvc), dt, i32(out_reset),
        st["EXNER"], st["RTHCUTEN"], st["RQCUTEN"],
        st["T"], st["Q"], st["TCUCN"],
        st["RAINCV"], st["CUTOP"], st["CUBOT"],
        st["PREC"], st["ACPREC"], st["ACPREC_TOT"], st["CUPREC"], st["CUPREC_TOT"], st["CUPPT"], st["CPRATE"],
        st["HTOP"], st["HBOT"], st["CNVBOT"], st["CNVTOP"],
        st["HBOTD"], st["HTOPD"], st["HBOTS"], st["HTOPS"],
    ), ns)
