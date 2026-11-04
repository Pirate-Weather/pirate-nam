"""phys-rrtm-gas-update: monthly CO2 and climatology-ozone time factors (NAM gas_update).

Fortran: phys/radiation_gases_nmmb.f:488-838 (NAM nam.v4.2.11, 83df4e13c1), subroutine gas_update.
Kernel: namgpu/kernels/phys_rrtm_gas_update.cu (gas_co2_monthly, one thread per (i, j) of the
24 x 12 CO2 grid, the 12 months as a per-thread loop, one launch, no host sync).

Host work, kept on the host as the Fortran routine is (none of it is per step):
  - the climatology-ozone time factors facoz, k1oz, k2oz (Fortran 555-581): twelve scalars,
    computed from the date;
  - the CO2 file reads (Fortran 610-833): fixed-width records, a few KB per call;
  - the scalar global mean co2_glb and the gco2cyc reset.
The routine runs only when the year changes (kyrsav), so it is not on the step path.

Physparam flags (ictmflg, ico2flg, ioznflg, co2dat_file, co2gbl_file) come from ctx.cfg, the same
NamConfig that phys-rrtm-gas-init reads, as the Fortran reads them from module physparam. Explicit
keyword arguments override them (the done-rule check passes them).

State shared with the other gas units (one table, one set of names):
  ctx.tables["rrtm_gases"]      written here: co2_glb [1], co2vmr_sav [12, JMXCO2, IMXCO2] (Fortran
                                co2vmr_sav(IMXCO2,JMXCO2,12)), gco2cyc [12], kmonsav. phys-rrtm-gas-init
                                writes co2cyc_sav there (read by the annual mode) and allocates the rest.
                                phys-rrtm-getgases reads this table.
  ctx.tables["rrtm_gas_update"] the GasState (k1oz, k2oz, facoz, ...) read by phys-rrtm-getozn.

Refusals (the Fortran STOPs, kept as RuntimeError with the same breakage): the historical file missing,
the monthly file missing for a requested year (ictmflg > 10), no earlier monthly file found, the
climatology ozone index used before any change day set it (Fortran leaves it undefined), a truncated
data file (Fortran end of file).
Prints that the Fortran writes under me == 0 are not reproduced (diagnostics only).
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_gas_init import _f, _dev_array, gases_table, physparam

UNIT = "phys_rrtm_gas_update"
KERNEL = "gas_co2_monthly"
KEY = "rrtm_gas_update"
IMXCO2 = 24          # Fortran IMXCO2: input co2 data lon points
JMXCO2 = 12          # Fortran JMXCO2: input co2 data lat points
MINYEAR = 1957       # Fortran MINYEAR: earliest year of 2-d co2 data
MODE_MONTHLY, MODE_ANNUAL, MODE_HIST = 0, 1, 2
CO2VMR_DEF = np.float32(350.0e-6)  # Fortran co2vmr_def
MDAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # Fortran mdays(1:12); mdays(13) unused


def _md(k: int) -> int:
    return MDAYS[k - 1]


@dataclass
class GasState:
    """Fortran module/SAVE variables of gas_update."""
    kyrsav: int = 0
    kmonsav: int = 0
    k1oz: int = 0        # 0 = not yet set by a change day (Fortran: no initialiser, undefined)
    k2oz: int = 0
    midm: int = 15       # Fortran initial value (data statement, save)
    midp: int = 45
    facoz: np.float32 = np.float32(0.0)
    co2_glb: np.float32 = CO2VMR_DEF


def _state(ctx) -> GasState:
    st = ctx.tables.get(KEY)
    if st is None:
        st = GasState()
        ctx.tables[KEY] = st
    return st


def ozone_factor(st: GasState, imon: int, iday: int, ihour: int, loz1st: bool) -> None:
    """Climatology-ozone time factors, Fortran 555-581 (ioznflg == 0)."""
    midmon = _md(imon) // 2 + 1
    change = bool(loz1st) or (iday == midmon and ihour == 0)
    if change:
        if iday < midmon:
            st.k1oz = (imon + 10) % 12 + 1          # mod(imon+10, 12) + 1
            st.midm = _md(st.k1oz) // 2 + 1
            st.k2oz = imon
            st.midp = _md(st.k1oz) + midmon
        else:
            st.k1oz = imon
            st.midm = midmon
            st.k2oz = imon % 12 + 1                 # mod(imon, 12) + 1
            st.midp = _md(st.k2oz) // 2 + 1 + _md(st.k1oz)
    if iday < midmon:
        if st.k1oz == 0:
            raise RuntimeError("phys-rrtm-gas-update: k1oz is used before any change day set it "
                               "(Fortran leaves it undefined); the first call must have loz1st true")
        ident = iday + _md(st.k1oz)
    else:
        ident = iday
    st.facoz = np.float32(ident - st.midm) / np.float32(st.midp - st.midm)


def co2_file_name(template: str, year: int) -> str:
    """write(cfile1(19:22), '(i4.4)') year on a 26-character copy of the template (Fortran 680)."""
    s = template.ljust(26)
    return (s[:18] + f"{year:04d}" + s[22:]).rstrip()


def _record(lines: list[str], pos: int) -> str:
    if pos >= len(lines):
        raise EOFError("gas_update: end of co2 data file (Fortran end of file)")
    return lines[pos]


def _list_record(lines: list[str], pos: int) -> tuple[list[str], int]:
    """read (unit,*): list-directed, blank records skipped. Returns the tokens and the next record."""
    while pos < len(lines) and not lines[pos].strip():
        pos += 1
    return _record(lines, pos).split(), pos + 1


def _i(s: str) -> int:
    """Fortran Iw read (i4): blank reads as zero, no implied decimal point."""
    s = s.strip()
    return int(s) if s else 0


def _fields(line: str, width: int, n: int) -> np.ndarray:
    """One Fortran record of n fixed-width fw.2 fields; blank fields read as zero (BN)."""
    line = line.ljust(width * n)
    return np.array([_f(line[k * width:(k + 1) * width]) for k in range(n)], np.float32)


def _read_grid(lines: list[str], pos: int) -> tuple[np.ndarray, int]:
    """read(NICO2CN, cform) co2dat with cform '(24f7.2)': the format is reused per record, so one
    read consumes JMXCO2 records; record j holds co2dat(1:IMXCO2, j). Returns [JMXCO2, IMXCO2]."""
    grid = np.empty((JMXCO2, IMXCO2), np.float32)
    for j in range(JMXCO2):
        grid[j, :] = _fields(_record(lines, pos), 7, IMXCO2)
        pos += 1
    return grid, pos


def _launch(ctx, mode: int, *, rate=0.0, g1=0.0, g2=0.0, dat=None) -> None:
    """One launch of gas_co2_monthly on ctx.stream. dat is the host [12, JMXCO2, IMXCO2] ppm block."""
    import cupy as cp
    tab = gases_table(ctx)
    vmr = _dev_array(tab, "co2vmr_sav", (12, JMXCO2, IMXCO2))
    cyc = tab.get("co2cyc_sav")
    if mode == MODE_ANNUAL and cyc is None:
        raise RuntimeError("phys-rrtm-gas-update annual mode (ictmflg -2) needs co2cyc_sav from "
                           "phys-rrtm-gas-init")
    dat_d = cp.asarray(dat, dtype=cp.float32) if dat is not None else vmr
    cyc_d = cyc if cyc is not None else vmr
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(IMXCO2, JMXCO2), kernels.BLOCK_COL,
          (np.int32(IMXCO2), np.int32(JMXCO2), np.int32(mode), np.float32(rate),
           np.float32(g1), np.float32(g2), dat_d, cyc_d, vmr))


def run(ctx, *, iyear: int, imon: int, iday: int, ihour: int, loz1st: bool, ldoco2: bool,
        co2dat_file: str | None = None, co2gbl_file: str | None = None, me: int = 0,
        ioznflg: int | None = None, ico2flg: int | None = None, ictmflg: int | None = None) -> GasState:
    """gas_update(iyear, imon, iday, ihour, loz1st, ldoco2, me) with the physparam flags from ctx.cfg.

    Returns the GasState; the device CO2 field is ctx.tables["rrtm_gases"]["co2vmr_sav"].
    """
    pp = physparam(ctx, dict(co2dat_file=co2dat_file, co2gbl_file=co2gbl_file, ioznflg=ioznflg,
                             ico2flg=ico2flg, ictmflg=ictmflg))
    co2dat_file, co2gbl_file = pp["co2dat_file"], pp["co2gbl_file"]
    ioznflg, ico2flg, ictmflg = int(pp["ioznflg"]), int(pp["ico2flg"]), int(pp["ictmflg"])

    st = _state(ctx)
    if ioznflg == 0:
        ozone_factor(st, imon, iday, ihour, loz1st)
    if ico2flg == 0:              # prescribed global mean co2
        return st
    if ictmflg == -1:             # user provided co2 data
        # Fortran leaves kmonsav at its initial 0 on this path and then reads gco2cyc(0) out of
        # bounds (above prsco2, ico2flg 2). gco2cyc is zero on this path, so the month index is
        # inert; setting it keeps the getgases index in range and gives the same values.
        gases_table(ctx)["kmonsav"] = imon
        st.kmonsav = imon
        return st
    if not ldoco2:                # no need to update co2 data
        return st

    tab = gases_table(ctx)
    tab["kmonsav"] = imon
    st.kmonsav = imon
    if ictmflg < 0:
        lextpl = False
        idyr = iyear
    else:
        lextpl = (ictmflg % 10 == 1)
        idyr = ictmflg // 10
        if idyr == 0:
            idyr = iyear
    if st.kyrsav == iyear:
        return st
    st.kyrsav = iyear
    iyr = iyear

    if idyr < MINYEAR and ictmflg > 0:
        # Historical semi-yearly global means (Fortran 610-674).
        if not os.path.exists(co2gbl_file):
            raise RuntimeError(f'Requested co2 data file "{co2gbl_file}" not found - Stopped in '
                               f"subroutine gas_update!!")
        with open(co2gbl_file) as f:
            lines = f.read().splitlines()
        hdr = _record(lines, 0).ljust(60)
        iyr1, iyr2 = _i(hdr[0:4]), _i(hdr[8:12])              # read (NICO2CN, 24) iyr1, iyr2
        if idyr < iyr1:
            iyr = iyr1
        i = iyr2
        pos = 1
        while i >= iyr1:
            tok, pos = _list_record(lines, pos)                  # read (NICO2CN, *) jyr, g1, g2
            jyr = int(tok[0])
            g1, g2 = np.float32(tok[1]), np.float32(tok[2])
            if i == iyr and iyr == jyr:
                st.co2_glb = (g1 + g2) * np.float32(0.5e-6)
                if ico2flg == 2:
                    _launch(ctx, MODE_HIST, g1=g1, g2=g2)
                break
            i -= 1
        _publish(ctx, st, tab)
        return st

    # Monthly 2-D data (Fortran 680-833).
    cfile1 = co2_file_name(co2dat_file, idyr)
    if not os.path.exists(cfile1):
        if ictmflg > 10:
            raise RuntimeError(f"Specified co2 data for year {idyr} not found !! Need to change "
                               f"namelist ICTM !! *** Stopped in subroutine gas_update !!")
        found = False
        while iyr >= MINYEAR:                                    # look for the latest available
            iyr -= 1
            cfile1 = co2_file_name(co2dat_file, iyr)
            if os.path.exists(cfile1):
                found = True
                break
        if not found:
            raise RuntimeError("Can not find co2 data source file *** Stopped in subroutine "
                               "gas_update !!")

    with open(cfile1) as f:
        lines = f.read().splitlines()
    # read (NICO2CN, 36) iyr, cline, co2g1, co2g2 with (i4, a94, f7.2, 16x, f5.2).
    # The header year overwrites iyr, which the rate below uses (Fortran 731, 743).
    hdr = _record(lines, 0).ljust(126)
    iyr = _i(hdr[0:4])
    g1 = np.float32(_f(hdr[98:105]))
    g2 = np.float32(_f(hdr[121:126]))

    if lextpl:
        rate = np.float32(2.00) * np.float32(iyear - iyr)     # avg rate for recent period
    else:
        rate = np.float32(0.0)
    st.co2_glb = (g1 + rate) * np.float32(1.0e-6)

    pos = 1
    if ictmflg == -2:             # annual mean first, then the seasonal cycle (Fortran 754-786)
        if ico2flg == 2:
            dat = np.empty((12, JMXCO2, IMXCO2), np.float32)
            for imo in range(12):
                dat[imo], pos = _read_grid(lines, pos)
            _launch(ctx, MODE_ANNUAL, dat=dat)
    else:                         # monthly data directly (Fortran 805-828)
        if ico2flg == 2:
            dat = np.empty((12, JMXCO2, IMXCO2), np.float32)
            for imo in range(12):
                dat[imo], pos = _read_grid(lines, pos)
            _launch(ctx, MODE_MONTHLY, rate=rate, dat=dat)
        _dev_array(tab, "gco2cyc", (12,))[:] = 0.0      # Fortran 828: direct branch only
    _publish(ctx, st, tab)
    return st


def _publish(ctx, st: GasState, tab: dict) -> None:
    """Device copy of the global mean that getgases reads (co2_glb [1])."""
    _dev_array(tab, "co2_glb", (1,))[:] = np.float32(st.co2_glb)
