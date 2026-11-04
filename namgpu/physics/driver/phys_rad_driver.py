"""phys-rad-driver: RADIATION, the outer driver of the NMMB radiation package (shortwave rrtm,
longwave rrtm, cldfraction default: the NAM 12 km parent configuration).

Ported from nmm/module_RADIATION.F90:66-995 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE RADIATION.
Kernel: namgpu/kernels/phys_rad_driver.cu (rad_zero_outputs, one thread per column, one launch).
Fortran-to-CUDA review and the list of dead branches: the kernel header.

Per call (every step, as the Fortran):
  1. radupdate_nmmb (module_RADIATION.F90:391-397, every step while a scheme is rrtm): the solar
     declination and constant, through the phys-rrtm-radupdate unit (scalar, no cell loop).
  2. rad_zero_outputs: CLDFRA, CFRACH, CFRACL, CFRACM, CZMEAN, SIGT4 zeroed over the whole domain
     (Fortran lines 308-327).
  3. On radiation steps only (ITIMESTEP == 1 or MOD(ITIMESTEP, NRAD) == 0 with NRAD = NRADS,
     line 414): the default cloud-input branch (lines 545-562) and the RRTM call over rows 2..JM-1
     and all columns (lines 586-625), through the phys-rrtm-driver-call unit.

Reads : ctx.st: CW, F_ICE, F_RAIN, QC, QI, QS, QR, QG, NI (cloud inputs), T, Q, O3, PD, SM, ALBEDO,
        CUPPT, SNOWC, SI, HTOP, HBOT, TSKIN, Z0, SICE, F_RIMEF, MXSNAL, STDH, OMGALF, GLAT, GLON,
        the sigma vectors DSG2, SGML2, PDSG1, PSGML1, and the RRTM output fields.
        ctx.tables['JDAT'] (the NMMB JDAT date array, 8 integers) from the clock unit.
Writes: ctx.st CLDFRA, CFRACH, CFRACL, CFRACM, CZMEAN, SIGT4 (zeroed here); the RRTM outputs
        RLWTT, RSWTT, RLWIN, RSWIN, RSWINC, RSWOUT, RLWTOA, RSWTOA, ACFRST, NCFRST, ACFRCV, NCFRCV
        are written by the RRTM unit (CZMEAN and SIGT4 included: RRTM is their writer on rad steps).

Hooks: the RRTM unit and radupdate are separate units. ``run(ctx, radupdate=, rrtm_call=)`` takes
        them as arguments; with None it imports namgpu.physics.rrtm.phys_rrtm_radupdate.run and
        namgpu.physics.rrtm.phys_rrtm_driver_call.run and raises a named error if either is not
        written yet. Signatures:
          radupdate(ctx, jdat=..., dtsw=..., dtx=..., lssw=...) -> dict with 'SOLCON' (W m-2)
          rrtm_call(ctx, args)  args: dict keyed by the Fortran RRTM argument names (_rrtm_args).

Refused (the Fortran would STOP or run a different scheme; this port does not run them):
  shortwave/longwave other than rrtm (gfdl, dudh, gsfc); cldfraction other than default (thompson
  calls cal_cldfra3 of module_mp_thompson, phys-alt-thompson, not selected here).
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rad_driver"
KERNEL = "rad_zero_outputs"

# Flags the Fortran sets per microphysics string (module_SOLVER_INTERNAL_STATE.F90:868-940).
# F_NR is not an argument of RRTM and is omitted.
MP_FLAGS = {
    "fer":       dict(num_water=4, F_QC=1, F_QI=0, F_QS=1, F_QR=1, F_QG=1, F_NI=0),
    "fer_hires": dict(num_water=4, F_QC=1, F_QI=0, F_QS=1, F_QR=1, F_QG=1, F_NI=0),
    "wsm6":      dict(num_water=5, F_QC=1, F_QI=1, F_QS=1, F_QR=1, F_QG=1, F_NI=0),
    "thompson":  dict(num_water=7, F_QC=1, F_QI=1, F_QS=1, F_QR=1, F_QG=1, F_NI=1),
    "gfs":       dict(num_water=2, F_QC=1, F_QI=1, F_QS=0, F_QR=0, F_QG=0, F_NI=0),
}

_RRTM_UNIT = "namgpu.physics.rrtm.rrtm_call"   # the whole RRTM chain (prep .. grrad .. post)
_RADUPDATE_UNIT = "namgpu.physics.rrtm.phys_rrtm_radupdate"


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables (same key as drv-tqadjust)."""
    if key not in ctx.tables:
        import cupy as cp
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def _need(st, name, why):
    """A field the NMMB state tables do not carry; the init unit that owns it must add it."""
    if name not in st.fields:
        raise KeyError(f"phys-rad-driver: {name} is missing ({why})")
    return st.fields[name]


def clock_jdat(ctx):
    """NMMB JDAT (year, month, day, tz, hour, minute, second, ms) of the current step.

    SOLVER_RUN refreshes JDAT from the clock every step: the time at the start of the step,
    start + NTIMESTEP*DT. Computed here from ctx.start and ctx.ntimestep and stored in
    ctx.tables["JDAT"]. Without ctx.start an existing ctx.tables["JDAT"] is used as given.
    """
    import datetime as _dt
    start = ctx.start if ctx.start is not None else ctx.start_time
    if start is None:
        jd = ctx.tables.get("JDAT")
        if jd is None:
            raise RuntimeError("phys-rad-driver: no clock (ctx.start) and no ctx.tables['JDAT']")
        return jd
    t = start + _dt.timedelta(seconds=float(ctx.ntimestep) * float(ctx.dt or ctx.cfg.dt))
    jd = [t.year, t.month, t.day, 0, t.hour, t.minute, t.second, 0]
    ctx.tables["JDAT"] = jd
    return jd


def _hook(explicit, module_name, what):
    if explicit is not None:
        return explicit
    try:
        return importlib.import_module(module_name).run
    except ImportError as exc:
        raise NotImplementedError(
            f"phys-rad-driver needs {what} ({module_name}), which is not written yet") from exc


def _rrtm_args(ctx, st, jdat, nts, dt, flags, solcon, mype):
    """The RRTM call's arguments (module_RADIATION.F90:601-625), keyed by the Fortran names.

    Default cloud branch (lines 545-562): the RRTM inputs QTdum, FIdum, FRdum, QCW are copies in
    Fortran. Here they are read-only aliases of the state. module_RA_RRTM.F90 declares QC, QS, QI,
    QR, QG and NI INTENT(INOUT); this port passes the live state arrays for all six, so the RRTM
    unit must never write any of them (contract for phys-rrtm-driver-call; not yet verified
    against the full RRTM call chain, which is not wired here). QCI, QSNOW and NCI are copied only when their flag
    is set; when it is not, Fortran leaves them undefined, so they are zero fields here.
    """
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    shape3 = (lm, jm, im)

    def zero(name):
        return st.add(name, shape3)   # zero-filled once on first use, never written

    qci = st["QI"] if flags["F_QI"] else zero("RAD_ZERO_QCI")
    qsnow = st["QS"] if flags["F_QS"] else zero("RAD_ZERO_QSNOW")
    nci = st["NI"] if flags["F_NI"] else zero("RAD_ZERO_NCI")
    for name in ("NCFRST", "NCFRCV"):      # Fortran INTEGER INTENT(INOUT); a float array would corrupt them
        if np.dtype(st[name].dtype) != np.dtype(np.int32):
            raise TypeError(f"phys-rad-driver: {name} must be int32, got {st[name].dtype}")
    return {
        "NTIMESTEP": int(nts), "DT": float(dt), "JDAT": jdat, "NPHS": int(ctx.cfg.nphs),
        "GLAT": st["GLAT"], "GLON": st["GLON"],
        "NRADS": int(ctx.cfg.nrads), "NRADL": int(ctx.cfg.nradl),
        "DSG2": st["DSG2"], "SGML2": st["SGML2"],
        "PDSG1": _device_vector(ctx, "pdsg1", g.pdsg1), "PSGML1": _device_vector(ctx, "psgml1", g.psgml1),
        "PT": float(g.pt), "PD": st["PD"],
        "T": st["T"], "Q": st["Q"],
        "QTdum": st["CW"], "O3": st["O3"], "ALBEDO": st["ALBEDO"],
        "FIdum": st["F_ICE"], "FRdum": st["F_RAIN"],
        "QCW": st["QC"], "QCI": qci, "QSNOW": qsnow, "QR": st["QR"], "QG": st["QG"], "NCI": nci,
        "F_QC": bool(flags["F_QC"]), "F_QI": bool(flags["F_QI"]), "F_QS": bool(flags["F_QS"]),
        "F_QR": bool(flags["F_QR"]), "F_QG": bool(flags["F_QG"]), "F_NI": bool(flags["F_NI"]),
        "NUM_WATER": int(flags["num_water"]), "CLD_FRACTION": 0,
        "SM": st["SM"], "CLDFRA": st["CLDFRA"],
        "RLWTT": st["RLWTT"], "RSWTT": st["RSWTT"], "RLWIN": st["RLWIN"], "RSWIN": st["RSWIN"],
        "RSWINC": st["RSWINC"], "RSWOUT": st["RSWOUT"], "RLWTOA": st["RLWTOA"], "RSWTOA": st["RSWTOA"],
        "CZMEAN": st["CZMEAN"], "SIGT4": st["SIGT4"],
        "CFRACL": st["CFRACL"], "CFRACM": st["CFRACM"], "CFRACH": st["CFRACH"],
        "ACFRST": st["ACFRST"], "NCFRST": st["NCFRST"], "ACFRCV": st["ACFRCV"], "NCFRCV": st["NCFRCV"],
        "CUPPT": st["CUPPT"], "SNOWC": st["SNOWC"], "SI": st["SI"],
        "HTOP": st["HTOP"], "HBOT": st["HBOT"],
        "TSKIN": st["TSKIN"], "Z0": st["Z0"], "SICE": st["SICE"],
        "F_RIMEF": st["F_RIMEF"], "MXSNAL": st["MXSNAL"], "SGM": st["SGM"],
        "STDH": _need(st, "STDH", "terrain standard deviation for RRTM's HPRIME; no state table "
                                  "carries it, the init unit must load it"),
        "OMGALF": st["OMGALF"],
        # Domain of the RRTM calls: all columns; rows JDS+1..JDE-1 (Fortran line 586-600 test).
        "IMS": 1, "IME": im, "JMS": 1, "JME": jm,
        "I_S": 1, "I_E": im, "J_S": 2, "J_E": jm - 1,
        "LM": lm, "SOLCON": float(solcon), "MYPE": int(mype),
    }


def run(ctx, *, radupdate=None, rrtm_call=None, mype: int = 0) -> None:
    """RADIATION runs in the run directory, as NMMB does: radupdate/gas_update/sol_update open
    co2historicaldata_YYYY.txt, solarconstant_*.txt and friends by bare name in the forecast
    directory that exnam_fcst.sh links them into. Our run directory for them is the fix directory
    drv-physics-initialize uses (ctx.fix_dir, $NAMGPU_FIX_DIR, else <repo>/fix); the caller's working
    directory is restored afterwards. Without this, gas_update at step 0 stopped with "Can not find
    co2 data source file" whenever the process did not start inside fix/."""
    import os
    from namgpu.steps.drv_physics_initialize import FIX_DIR
    fix = str(getattr(ctx, "fix_dir", None) or os.environ.get("NAMGPU_FIX_DIR") or FIX_DIR)
    cwd = os.getcwd()
    os.chdir(fix)
    try:
        return _run(ctx, radupdate=radupdate, rrtm_call=rrtm_call, mype=mype)
    finally:
        os.chdir(cwd)


def _run(ctx, *, radupdate=None, rrtm_call=None, mype: int = 0) -> None:
    cfg = ctx.cfg
    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    sw = str(cfg.shortwave).strip()
    lw = str(cfg.longwave).strip()
    if sw != "rrtm" or lw != "rrtm":
        raise NotImplementedError(
            "phys-rad-driver runs shortwave=rrtm longwave=rrtm (parm/nam_configfile); "
            f"got shortwave={sw!r} longwave={lw!r}; the gfdl/dudh/gsfc branches are not ported")
    cldf = str(getattr(cfg, "cldfraction", "default")).strip()
    if cldf != "default":
        raise NotImplementedError(
            f"phys-rad-driver: cldfraction={cldf!r} needs cal_cldfra3 (module_mp_thompson), "
            "not ported; parm/nam_configfile selects 'default'")
    mp = str(cfg.microphysics).strip()
    if mp not in MP_FLAGS:
        raise NotImplementedError(f"phys-rad-driver: no flag table for microphysics {mp!r}")
    flags = MP_FLAGS[mp]

    # Fortran RADIATION(NTIMESTEP_RAD, ...) (SOLVER_RUN 5312): 0 at step 0, n+1 after (4938-4942).
    nts = int(ctx.ntimestep_rad)
    dt = float(ctx.dt)
    nrads = int(cfg.nrads)
    nrad = nrads   # NRAD = NRADS (module_RADIATION.F90:247)

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    jdat = clock_jdat(ctx)

    # 1. Zeroing of the radiation outputs, every step (Fortran 308-327, before radupdate at 391-397).
    kern = kernels.get(UNIT, KERNEL)
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
             (np.int32(im), np.int32(jm), np.int32(lm),
              st["CLDFRA"], st["CFRACH"], st["CFRACL"], st["CFRACM"], st["CZMEAN"], st["SIGT4"]))

    # 2. radupdate_nmmb, every step (SW or LW is rrtm; Fortran 391-397).
    upd = _hook(radupdate, _RADUPDATE_UNIT, "the radupdate_nmmb unit")
    out = upd(ctx, jdat=jdat, dtsw=nrads * dt, dtx=dt, lssw=(nts % nrads == 0))
    solcon = float(out["SOLCON"])

    # 3. Radiation step (ITIMESTEP == 1 or MOD(ITIMESTEP, NRAD) == 0): default cloud branch, RRTM.
    if nts == 1 or nts % nrad == 0:
        call = _hook(rrtm_call, _RRTM_UNIT, "the RRTM driver call unit")
        call(ctx, _rrtm_args(ctx, st, jdat, nts, dt, flags, solcon, mype))
