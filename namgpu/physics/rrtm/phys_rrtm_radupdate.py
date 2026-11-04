"""phys-rrtm-radupdate: calendar-driven radiation update control (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:456-602, radupdate_nmmb. NOAA US
government work; the NOAA header of grrad_nmmb.f is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_radupdate.cu (phys_rrtm_radupdate, one thread, one launch).

Caller (module_RADIATION.F90:386-397, every step while sw or lw is rrtm):
    call radupdate_nmmb(JDAT, JDAT, DTSW, DTX, LSSWR, MYPE, slag, sdec, cdec, solcon)
    with DTSW = NRADS*DT, DTX = DT, LSSWR = MOD(ITIMESTEP, NRADS) == 0 (IDAT = JDAT for a forecast).

The routine has no grid loops. The kernel decides, from the calendar, which time-varying radiation
data are refreshed and writes the decisions into a flags array. This wrapper then runs the sibling
units those flags select:
  F_CALL_SOL (= LSSWR)  sol_update_nmmb   (phys-rrtm-sol-update), which also produces SOLCON.
  F_CALL_AER (= month change)  aer_update (phys-rrtm-aer-update), with iyear/imon from JDAT.
  F_CALL_GAS            gas_update (phys-rrtm-gas-update): NOT wired here yet (see STATUS.md).

Scratch (owned by the caller, allocated once in ctx.tables):
  rst    int32[4]   month0, iyear0, monthd, loz1st   (radinit sets loz1st = ioznflg == 0)
  flags  int32[14]  see F_* below, written every call
The 14-int flags are read to the host once per call (the sibling branches are host control flow);
that copy is the only host transfer on this path.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_aer_update as aer_update
from namgpu.physics.rrtm import phys_rrtm_sol_update as sol_update

UNIT = "phys_rrtm_radupdate"
KERNEL = "phys_rrtm_radupdate"

RST_LEN = 4
FLAGS_LEN = 14
F_LSOL, F_LMON, F_LCO2 = 0, 1, 2
F_CALL_SOL, F_CALL_AER, F_CALL_GAS = 3, 4, 5
F_KYEAR, F_KMON, F_KDAY, F_KHOUR = 6, 7, 8, 9
F_LOZ_ARG, F_LOZ_AFTER = 10, 11
F_IYEAR, F_IMON = 12, 13          # jdate(1), jdate(2): the aer_update arguments (not the idate month)

KEY_RST = "rrtm_rst"
KEY_FLAGS = "rrtm_flags"
KEY_SOLCON = "rrtm_solcon"        # host tuple (slag, sdec, cdec, solcon) of the last sol_update call
KEY_SOL_ISOLFLG = "sol_isolflg"   # written by phys-rrtm-sol-init (radinit)


def new_state(loz1st: bool):
    """rst scratch as radinit leaves it: month0 = iyear0 = monthd = 0, loz1st from ioznflg."""
    import cupy as cp

    rst = cp.zeros(RST_LEN, dtype=cp.int32)
    rst[3] = 1 if loz1st else 0
    return rst


def new_flags():
    import cupy as cp

    return cp.zeros(FLAGS_LEN, dtype=cp.int32)


def radupdate(rst, flags, jdate, idate, isolar: int, ictmflg: int, lsswr: bool, stream=None):
    """Launch the kernel. jdate, idate: 8-element sequences (yr, mon, day, tzone, hr, min, sec, msec)."""
    k = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1),
          (rst, flags,
           np.int32(jdate[0]), np.int32(jdate[1]), np.int32(jdate[2]), np.int32(jdate[4]),
           np.int32(idate[0]), np.int32(idate[1]), np.int32(idate[2]), np.int32(idate[4]),
           np.int32(isolar), np.int32(ictmflg), np.int32(1 if lsswr else 0)))
    return flags


def run(ctx, *, jdat, dtsw: float, dtx: float, lssw: bool, mype: int = 0) -> dict:
    """radupdate_nmmb for one step. Returns {'SOLCON', 'SLAG', 'SDEC', 'CDEC'} (SI, W m-2 for SOLCON).

    jdat: the NCEP JDAT array (8 integers), ctx.tables['JDAT'] from the clock unit. The Fortran passes
    JDAT as both IDAT and JDAT in the forecast, so idate = jdate here. dtsw, dtx: DTSW and DTX in
    seconds. lssw: LSSWR. mype: print control only (the Fortran prints nothing on this path).
    """
    cfg = ctx.cfg
    tabs = ctx.tables
    jd = [int(v) for v in jdat]
    if len(jd) != 8:
        raise ValueError(f"phys-rrtm-radupdate: JDAT must have 8 integers, got {len(jd)}")
    stream = ctx.stream

    if KEY_RST not in tabs:
        tabs[KEY_RST] = new_state(loz1st=(int(cfg.ioznflg) == 0))
    if KEY_FLAGS not in tabs:
        tabs[KEY_FLAGS] = new_flags()
    rst = tabs[KEY_RST]
    flags = tabs[KEY_FLAGS]

    # 1. Calendar decisions (device). isolar, ictmflg from the physparam defaults in namgpu/config.py.
    radupdate(rst, flags, jd, jd, int(cfg.isolar), int(cfg.ictmflg), bool(lssw), stream)
    # The kernel ran on ctx.stream (non-blocking); a plain .get() reads on the legacy stream and can
    # see the flags before the kernel wrote them (run lead 0940Z: F_CALL_SOL read as 0 at step 0, so
    # SOLCON was never set). Read on the same stream.
    f = flags.get(stream=stream) if stream is not None else flags.get()   # the one 14-int host read per call

    # 2. sol_update_nmmb (Fortran: call only when lsswr). It writes SLAG, SDEC, CDEC, SOLCON.
    if int(f[F_CALL_SOL]):
        if KEY_SOL_ISOLFLG not in tabs:
            raise RuntimeError("phys-rrtm-radupdate: sol_isolflg is missing; phys-rrtm-sol-init "
                               "(phys-rrtm-radinit) must run before the first radiation update")
        isolflg = int((tabs[KEY_SOL_ISOLFLG].get(stream=stream) if stream is not None else tabs[KEY_SOL_ISOLFLG].get())[0])
        out, _state, _nstp = sol_update.run(
            ctx, jdate=jd, kyear=int(f[F_KYEAR]), deltsw=float(dtsw), deltim=float(dtx),
            lsol_chg=bool(f[F_LSOL]), isolflg=isolflg, isolar=int(cfg.isolar))
        slag, sdec, cdec, solcon = (float(x) for x in (out.get(stream=stream) if stream is not None else out.get()))
        tabs[KEY_SOLCON] = (slag, sdec, cdec, solcon)

    # 3. aer_update (Fortran: call only when the month changed), iyear and imon from JDAT.
    if int(f[F_CALL_AER]):
        aer_update.run(ctx, iyear=int(f[F_IYEAR]), imon=int(f[F_IMON]))

    # 4. gas_update, every call (grrad_nmmb.f:588-596): kyear/kmon/kday/khour, loz1st as passed
    #    (F_LOZ_ARG, before the kernel cleared it), lco2_chg (F_LCO2). The host work inside
    #    (ozone time factors, the CO2 file read on a year change) is the Fortran host work.
    if int(f[F_CALL_GAS]):
        from namgpu.physics.rrtm import phys_rrtm_gas_update as gas_update
        gas_update.run(ctx, iyear=int(f[F_KYEAR]), imon=int(f[F_KMON]), iday=int(f[F_KDAY]),
                       ihour=int(f[F_KHOUR]), loz1st=bool(f[F_LOZ_ARG]), ldoco2=bool(f[F_LCO2]))

    if KEY_SOLCON not in tabs:
        raise RuntimeError("phys-rrtm-radupdate: SOLCON has not been set; the first sol_update_nmmb "
                           "call (LSSWR true) has not run")
    slag, sdec, cdec, solcon = tabs[KEY_SOLCON]
    return {"SOLCON": solcon, "SLAG": slag, "SDEC": sdec, "CDEC": cdec}


def selftest():
    """One launch on one plausible calendar with a fresh state; checks the flags and the state."""
    rst = new_state(loz1st=True)
    flags = new_flags()
    jdate = [2023, 3, 31, 0, 12, 0, 0, 0]
    idate = [2023, 3, 31, 0, 12, 0, 0, 0]
    radupdate(rst, flags, jdate, idate, isolar=2, ictmflg=0, lsswr=True)
    f = flags.get()
    s = rst.get()
    assert f.shape == (FLAGS_LEN,) and s.shape == (RST_LEN,)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(s))
    # first call: month changed (0 -> 3), year changed (0 -> 2023), gas refresh, loz1st consumed
    assert f[F_LMON] == 1 and f[F_LSOL] == 1 and f[F_LCO2] == 1
    assert f[F_CALL_SOL] == 1 and f[F_CALL_AER] == 1 and f[F_CALL_GAS] == 1
    assert f[F_KYEAR] == 2023 and f[F_KMON] == 3 and f[F_KDAY] == 31 and f[F_KHOUR] == 12
    assert f[F_IYEAR] == 2023 and f[F_IMON] == 3
    assert f[F_LOZ_ARG] == 1 and f[F_LOZ_AFTER] == 0 and s[3] == 0
    assert s[0] == 3 and s[1] == 2023 and s[2] == 3
    return {"flags": f.tolist(), "rst": s.tolist()}
