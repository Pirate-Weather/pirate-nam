"""phys-rrtm-sol-update: sol_update_nmmb + solar (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_astronomy_nmmb.f:265-570 (sol_update_nmmb)
and :575-739 (solar). NOAA US government work; the NOAA headers are kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sol_update.cu (sol_update_nmmb, one thread, one launch).

What it computes, per radiation call (every NRADS = 48 steps):
  slag (equation of time, rad), sdec / cdec (sin / cos of solar declination), solcon (W/m2, solar constant
  adjusted by the sun-earth distance), plus the module state sollag, sindec, cosdec, anginc, nstp that
  coszmn (phys-rrtm-coszmn) reads.

Host / device split (CONTRACT section 3, "no host sync"):
  - device, every call: the one-thread kernel. It reads solc0 from ctx.tables["sol_solc0"] (written by
    phys-rrtm-sol-init) and writes four outputs, the module state and nstp into device scratch.
  - host, only when lsol_chg is true (flags[F_CALL_SOL] of phys-rrtm-radupdate) and the year changed, or
    for isolflg 4 on a new month: the solar-constant TABLE read. That is file I/O (Fortran inquire / open /
    list-directed read), it changes solc0 once a year, and it is written back to the device with a single
    small host-to-device copy. It is ported literally from the Fortran (cycle approximation, record search).
    Host state iyr_sav and smon_sav lives in ctx.tables as the Fortran module variables do.

Scratch (ctx.st, allocated once, reused): SOL_OUT float32[4] = slag, sdec, cdec, solcon;
  SOL_STATE float32[4] = sollag, sindec, cosdec, anginc; SOL_NSTP int32[1] = nstp.
Table in: ctx.tables["sol_solc0"] float64[1] (sol-init storage).
THE solar contract (one place): ctx.st[SOL_STATE] float32[4] = sollag, sindec, cosdec, anginc, device.
phys-rrtm-coszmn and phys-rrtm-dayparts read it on the device; solhr is ctx.clock.solhr, not part of it.
"""
from __future__ import annotations

import contextlib
import os
import re

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_sol_init import solar_fname

UNIT = "phys_rrtm_sol_update"
KERNEL = "sol_update_nmmb"
KEY_SOLC0 = "sol_solc0"            # shared with phys-rrtm-sol-init
KEY_IYR_SAV = "sol_iyr_sav"        # Fortran module variable iyr_sav (host int)
KEY_SMON_SAV = "sol_smon_sav"      # Fortran module variable smon_sav(12) (host float32)
ST_OUT = "SOL_OUT"
ST_STATE = "SOL_STATE"
ST_NSTP = "SOL_NSTP"


def _tokens(line: str) -> list[str]:
    """List-directed read tokenisation: blanks and commas separate items."""
    return [t for t in re.split(r"[,\s]+", line.strip()) if t]


def _set_solc0(ctx, value: np.float32) -> None:
    """Write the new solar constant (kind_phys = single) into the device table."""
    import cupy as cp

    ctx.tables[KEY_SOLC0][0:1] = cp.asarray([float(value)], dtype=cp.float64)


def _read_table(ctx, iyear: int, imon: int, isolflg: int, isolar: int, solar_dir: str | None) -> None:
    """Fortran sol_update lines 388-516 for lsol_chg: pick the table year and set solc0 (host I/O)."""
    smon_sav = ctx.tables[KEY_SMON_SAV]
    if ctx.tables[KEY_IYR_SAV] == iyear:          # same year, no new reading necessary
        if isolflg == 4:
            _set_solc0(ctx, smon_sav[imon - 1])
        return

    ctx.tables[KEY_IYR_SAV] = iyear
    name = solar_fname(isolar).rstrip()
    base = solar_dir if solar_dir is not None else os.getcwd()
    path = os.path.join(base, name)
    if not os.path.isfile(path):
        raise RuntimeError(" !!! ERROR! Can not find solar constant file!!!")

    with open(path) as fh:
        # list-directed reads skip blank records; each READ consumes the next record
        records = [ln for ln in fh.read().splitlines() if ln.strip()]
    pos = 0

    def next_record():
        nonlocal pos
        if pos >= len(records):
            raise RuntimeError("end of file reading solar constant table " + name)
        rec = _tokens(records[pos])
        pos += 1
        return rec

    # read (NIRADSF,*) iyr1,iyr2,icy1,icy2,smean,cline
    hdr = next_record()
    iyr1, iyr2, icy1, icy2 = int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3])
    smean = np.float32(hdr[4])

    # check if there is a cycle approximation for the year
    iyr = iyear
    if iyr < iyr1:
        icy = icy1 - iyr1 + 1
        while iyr < iyr1:
            iyr = iyr + icy
    elif iyr > iyr2:
        icy = iyr2 - icy2 + 1
        while iyr > iyr2:
            iyr = iyr - icy

    solc0 = None
    i = iyr2
    if isolflg < 4:                               # annual mean tables
        while i >= iyr1:
            rec = next_record()
            jyr, solc1 = int(rec[0]), np.float32(rec[1])
            if i == iyr and iyr == jyr:
                solc0 = np.float32(smean + solc1)
                break
            i -= 1
    else:                                         # monthly mean tables
        while i >= iyr1:
            rec = next_record()
            jyr = int(rec[0])
            smon = np.array([np.float32(x) for x in rec[1:13]], dtype=np.float32)
            if i == iyr and iyr == jyr:
                for nn in range(12):
                    smon_sav[nn] = np.float32(smean + smon[nn])
                solc0 = np.float32(smean + smon[imon - 1])
                break
            i -= 1

    if solc0 is not None:
        _set_solc0(ctx, solc0)


def run(ctx, jdate, kyear: int, deltsw: float, deltim: float, lsol_chg: bool,
        isolflg: int, isolar: int, solar_dir: str | None = None):
    """sol_update_nmmb for one radiation call.

    jdate: 8 integers (yr, mon, day, tzone, hr, min, sec, msec), as NCEP writes them.
    kyear: upper year limit (unused by the active Fortran; kept for the call signature).
    isolflg: the current solar-constant scheme flag (sol-init output, host int from the config).
    isolar: physparam isolar, required. It names the table exactly as sol_init does (solar_fname(15:26)).
    isolflg: sol_init's post-fallback flag (ctx.tables sol_isolflg); the Fortran isolflg==4 branch reads it.
    lsol_chg: computed by phys-rrtm-radupdate (flags F_LSOL) with the grrad_nmmb.f:557-564 rule and the
    iyear0 state, the same caller that passes isolar. Verified against the Fortran, not recomputed here.
    Returns (out, state, nstp): device float32[4] = slag, sdec, cdec, solcon; device float32[4] =
    sollag, sindec, cosdec, anginc; device int32[1] = nstp.
    """
    import cupy as cp

    if KEY_SOLC0 not in ctx.tables:
        raise RuntimeError("phys-rrtm-sol-update needs phys-rrtm-sol-init to run first (sol_solc0)")
    if KEY_IYR_SAV not in ctx.tables:
        ctx.tables[KEY_IYR_SAV] = 0
        ctx.tables[KEY_SMON_SAV] = np.full(12, np.float32(1.3608e3), dtype=np.float32)  # data smon_sav = 12*con_solr (physcons.f:51)

    if lsol_chg:
        _read_table(ctx, int(jdate[0]), int(jdate[1]), int(isolflg),
                    int(isolar), solar_dir)

    st = ctx.st
    out = st.add(ST_OUT, (4,), dtype="float32")
    state = st.add(ST_STATE, (4,), dtype="float32")
    nstp = st.add(ST_NSTP, (1,), dtype="int32")

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1),
          (np.int32(jdate[0]), np.int32(jdate[1]), np.int32(jdate[2]),
           np.int32(jdate[4]), np.int32(jdate[5]), np.int32(jdate[6]),
           np.float32(deltsw), np.float32(deltim),
           ctx.tables[KEY_SOLC0], out, state, nstp))
    # rrtm_sol: the solar module variables the RRTM consumers read (coszmn, dayparts). One copy of four
    # floats per sol_update call (once per radiation call), not per step.
    sol_host = cp.asnumpy(state)
    ctx.tables["rrtm_sol"] = {"sollag": float(sol_host[0]), "sindec": float(sol_host[1]),
                              "cosdec": float(sol_host[2]), "anginc": float(sol_host[3])}
    return out, state, nstp
