"""phys-rrtm-gas-init: ozone and CO2 setup of NMMB gas_init (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_gases_nmmb.f:186-483, gas_init.
Kernel: namgpu/kernels/phys_rrtm_gas_init.cu (gas_init_pkstr, gas_init_o3r, gas_init_co2_scale,
gas_init_co2_glb). The Fortran line-by-line mapping is in the kernel header.

Host (file I/O, Fortran OPEN/READ/INQUIRE and every branch choice):
  - ioznflg == 0 (climatology): reads the ozone climatology (NIO3CLM = unit 48, fort.48 ->
    global_o3clim.txt). LOZ == 17 uses the formatted operational layout; other LOZ uses the
    unformatted sequential layout.
  - ico2flg 1/2 with ictmflg -1: reads the user CO2 file (header line, then 12 monthly 24x12 blocks
    for ico2flg 2).
  - ictmflg -2: reads the CO2 seasonal-cycle file (header, 12 skipped annual blocks, then per month a
    gco2cyc line and a 24x12 block).
  - A missing file stops the run, as the Fortran does.
Device (every arithmetic term): pkstr, o3r, co2_glb, co2vmr_sav, co2cyc_sav, gco2cyc.

Out (ctx.tables, float32 = kind_phys single build):
  gas_pkstr      (LOZ,)            ozone pressure-level kappa powers (ioznflg 0 only)
  gas_o3r        (12, LOZ, JMR)    ozone climatology, [imo][k][j] (Fortran o3r(JMR,LOZ,12))
  gas_co2_glb    (1,)              global CO2 volume mixing ratio
  gas_co2vmr_sav (12, 12, 24)      monthly 2-d CO2, [imo][j][i] (Fortran co2vmr_sav(24,12,12))
  gas_co2cyc_sav (12, 12, 24)      seasonal-cycle CO2 (ictmflg -2)
  gas_gco2cyc    (12,)             seasonal-cycle monthly factors
run() returns a dict with the same arrays plus the host value read (co2g1).

Defaults are the NMMB physparam defaults: ico2flg 0, ictmflg 0, ioznflg 1. The NAM run's real
ico2 / ictm / ntoz come from the caller of rad_initialize and are not in parm/nam_configfile, so the
caller passes them in.
One launch per Fortran section on ctx.stream, no host sync inside the kernels.
Not bitwise-checked (not required). The text layouts follow the Fortran formats exactly:
  f10.3 (pstr), (i2,i4,10f6.2) and (6x,10f6.2) (ozone), (24f7.2) (CO2 blocks),
  (i4,a94,f7.2,16x,f5.2) and (a98,f7.2,16x,f5.2) (headers), (a58,f7.2) (cycle factor lines).
"""
from __future__ import annotations

import contextlib
import os
import struct

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_gas_init"
IMXCO2 = 24  # radiation_gases_nmmb.f IMXCO2
JMXCO2 = 12  # radiation_gases_nmmb.f JMXCO2
NMONTH = 12
LOZ_DEFAULT = 17  # levozc (module_SOLVER_GRID_COMP.F90:10380)
JMR_DEFAULT = 18  # latsozc (module_SOLVER_GRID_COMP.F90:10381)
O3CLIM_FILE = "global_o3clim.txt"  # exglobal_fcst_nems.sh O3CLIM default (linked as fort.48)
GASES_KEY = "rrtm_gases"  # the one CO2 state table: gas_init and gas_update write it, getgases reads it


def gases_table(ctx) -> dict:
    """ctx.tables['rrtm_gases']: co2_glb [1], gco2cyc [12], co2vmr_sav and co2cyc_sav [12, 12, 24]
    (device), ico2flg, ivflip and kmonsav (ints). Created on first use."""
    t = ctx.tables.get(GASES_KEY)
    if t is None:
        t = {}
        ctx.tables[GASES_KEY] = t
    return t


def _dev_array(tab: dict, key: str, shape):
    """Device float32 array in tab, zero-filled on first use."""
    import cupy as cp
    v = tab.get(key)
    if v is None:
        v = cp.zeros(shape, dtype=cp.float32)
        tab[key] = v
    return v


def physparam(ctx, given: dict) -> dict:
    """Each physparam value: the explicit argument if given, else ctx.cfg (physparam.f defaults)."""
    out = {}
    for name, v in given.items():
        if v is None:
            cfg = getattr(ctx, "cfg", None)
            if cfg is None:
                raise RuntimeError(f"phys-rrtm-gas: {name} not given and ctx.cfg is missing")
            v = getattr(cfg, name)
        out[name] = v
    return out


class _Lines:
    """Formatted sequential reader: one READ statement consumes one record (line)."""

    def __init__(self, path: str):
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            self.lines = fh.read().splitlines()
        self.i = 0

    def next(self) -> str:
        if self.i >= len(self.lines):
            raise EOFError("gas_init: end of formatted record data")
        s = self.lines[self.i].ljust(130)
        self.i += 1
        return s


class _Unformatted:
    """gfortran sequential unformatted reader: each record is int32 length, data, int32 length."""

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            self.buf = fh.read()
        self.p = 0

    def read(self) -> bytes:
        n = struct.unpack_from("<i", self.buf, self.p)[0]
        self.p += 4
        data = self.buf[self.p:self.p + n]
        self.p += n + 4
        return data


def _f(s: str, d: int = 2) -> float:
    """Fortran Fw.d input edit of one field (gfortran rules).

    Blanks read as zero (BN default). A field with no decimal point and no exponent has the
    descriptor d as an implied decimal point: '  350' under f7.2 reads as 3.50, not 350.
    Fields that carry their own point or exponent are read as written."""
    s = s.strip()
    if not s:
        return 0.0
    if "." in s or any(c in s for c in "eEdD"):
        return float(s.replace("d", "e").replace("D", "E"))
    return float(s) / float(10 ** d)


def _read_ozone_formatted(path: str, loz: int, jmr: int):
    """LOZ == 17 operational climatology (gas_init, formatted branch)."""
    rd = _Lines(path)
    pstr4 = np.zeros(loz, dtype=np.float32)
    for k in range(loz):
        pstr4[k] = _f(rd.next()[0:10], 3)  # read (NIO3CLM,15) pstr4(k), format f10.3
    o3clim4 = np.zeros((NMONTH, loz, jmr), dtype=np.float32)  # [imo][k][j]
    for imo in range(NMONTH):
        for j in range(jmr):
            hdr = rd.next()  # format (i2,i4,10f6.2): imond, ilat, k = 1..10
            for m in range(10):
                o3clim4[imo, m, j] = _f(hdr[6 + 6 * m:12 + 6 * m])
            body = rd.next()  # format (6x,10f6.2): k = 11..LOZ
            for m in range(loz - 10):
                o3clim4[imo, 10 + m, j] = _f(body[6 + 6 * m:12 + 6 * m])
    return pstr4, o3clim4


def _read_ozone_unformatted(path: str, loz: int, jmr: int):
    """LOZ != 17 newer climatology: unformatted sequential records (gas_init, else branch)."""
    rd = _Unformatted(path)
    rd.read()  # read (NIO3CLM): header record
    pstr4 = np.zeros(loz, dtype=np.float32)
    for k in range(loz):
        pstr4[k] = np.frombuffer(rd.read(), dtype="<f4")[0]
    o3clim4 = np.zeros((NMONTH, loz, jmr), dtype=np.float32)
    for imo in range(NMONTH):
        for k in range(loz):
            o3clim4[imo, k, :] = np.frombuffer(rd.read(), dtype="<f4")[:jmr]
    return pstr4, o3clim4


def _read_co2_block(rd: _Lines) -> np.ndarray:
    """read (NICO2CN,cform) co2dat with cform (24f7.2): 288 values in 12 records, i fastest."""
    vals = []
    for _ in range(JMXCO2):
        line = rd.next()
        for m in range(IMXCO2):
            vals.append(_f(line[7 * m:7 * m + 7]))
    return np.asarray(vals, dtype=np.float32).reshape(JMXCO2, IMXCO2)  # [j][i]


def run(ctx, *, ictmflg: int | None = None, ico2flg: int | None = None,
        ioznflg: int | None = None, timeozc: int = 12,
        loz: int = LOZ_DEFAULT, jmr: int = JMR_DEFAULT, base_dir: str | None = None,
        o3clim_file: str = O3CLIM_FILE, co2usr_file: str | None = None,
        co2cyc_file: str | None = None):
    """gas_init. ictmflg, ico2flg, co2usr_file, co2cyc_file and ivflip come from ctx.cfg (the physparam
    values gas_update reads too) unless given; ioznflg comes from the radinit host flags."""
    import cupy as cp

    base = base_dir if base_dir is not None else os.getcwd()

    def path(name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(base, name)

    pp = physparam(ctx, dict(ictmflg=ictmflg, ico2flg=ico2flg, co2usr_file=co2usr_file,
                             co2cyc_file=co2cyc_file, ivflip=None))
    ictmflg, ico2flg = int(pp["ictmflg"]), int(pp["ico2flg"])
    co2usr_file, co2cyc_file = pp["co2usr_file"], pp["co2cyc_file"]

    T = ctx.tables
    tab = gases_table(ctx)
    tab["ico2flg"] = ico2flg          # read by phys-rrtm-getgases
    tab["ivflip"] = int(pp["ivflip"])

    def dev(key: str, shape):
        if key not in T:
            T[key] = cp.zeros(shape, dtype=cp.float32)
        return T[key]

    if ioznflg is None:
        from namgpu.physics.rrtm.phys_rrtm_radinit import host_ctl
        ioznflg = int(host_ctl(ctx)["ioznflg"])   # NAM ntoz (module_SOLVER_GRID_COMP.F90:10691)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    k_pk = kernels.get(UNIT, "gas_init_pkstr")
    k_o3 = kernels.get(UNIT, "gas_init_o3r")
    k_sc = kernels.get(UNIT, "gas_init_co2_scale")
    k_gl = kernels.get(UNIT, "gas_init_co2_glb")
    out = {}

    with scope:
        # ---- climatology ozone: Fortran "if (ioznflg > 0) interactive else climatology" (line 268),
        # so every ioznflg <= 0 reads the climatology (NTOZ=0 is the NAM value)
        if ioznflg <= 0:
            if timeozc != 12:
                raise ValueError(f"gas_init: timeozc={timeozc} is not monthly mean (NMMB stops)")
            fname = path(o3clim_file)
            if not os.path.isfile(fname):
                raise FileNotFoundError(f"gas_init: ozone climatology not found: {fname}")
            if loz == LOZ_DEFAULT:
                pstr4, o3clim4 = _read_ozone_formatted(fname, loz, jmr)
            else:
                pstr4, o3clim4 = _read_ozone_unformatted(fname, loz, jmr)
            pstr_d = cp.asarray(pstr4)  # pstr(k) = pstr4(k)
            o3_in = cp.asarray(o3clim4.reshape(-1))
            pkstr = dev("gas_pkstr", (loz,))
            o3r = dev("gas_o3r", (NMONTH, loz, jmr))
            k_pk(((loz + 127) // 128,), (128,), (pstr_d, pkstr, np.int32(loz)))
            n_o3 = NMONTH * loz * jmr
            k_o3(((n_o3 + 255) // 256,), (256,), (o3_in, o3r.reshape(-1), np.int32(n_o3)))
            out["pkstr"] = pkstr
            out["o3r"] = o3r

        # ---- CO2 (ico2flg, ictmflg) ----------------------------------------------------------
        co2_glb = _dev_array(tab, "co2_glb", (1,))
        co2g1 = np.float32(0.0)
        use_file = 0
        vmr_sav = _dev_array(tab, "co2vmr_sav", (NMONTH, JMXCO2, IMXCO2))
        cyc_sav = _dev_array(tab, "co2cyc_sav", (NMONTH, JMXCO2, IMXCO2))
        gco2cyc = _dev_array(tab, "gco2cyc", (NMONTH,))
        n_sav = NMONTH * JMXCO2 * IMXCO2

        if ico2flg != 0:
            if ictmflg == -1:  # user-provided data (gas_init 'input user provided data')
                fname = path(co2usr_file)
                if not os.path.isfile(fname):
                    raise FileNotFoundError(f"gas_init: user CO2 file not found: {fname}")
                rd = _Lines(fname)
                hdr = rd.next()  # read (NICO2CN,25) iyr, cline, co2g1, co2g2 : (i4,a94,f7.2,16x,f5.2)
                co2g1 = np.float32(_f(hdr[98:105]))
                use_file = 1
                if ico2flg == 2:
                    host = np.zeros((NMONTH, JMXCO2, IMXCO2), dtype=np.float32)
                    for imo in range(NMONTH):
                        host[imo] = _read_co2_block(rd)
                    src = cp.asarray(host.reshape(-1))
                    k_sc(((n_sav + 255) // 256,), (256,), (src, vmr_sav.reshape(-1), np.int32(n_sav)))
                elif ico2flg != 1:
                    raise ValueError(f"gas_init: ICO2={ico2flg} is not a valid selection")
            else:  # observed data (ictmflg yyyy#, -2 or the other flags)
                if ico2flg not in (1, 2):
                    raise ValueError(f"gas_init: ICO2={ico2flg} is not a valid selection")
                if ictmflg == -2:  # seasonal cycle superimposed on the mean
                    fname = path(co2cyc_file)
                    if not os.path.isfile(fname):
                        raise FileNotFoundError(f"gas_init: CO2 seasonal cycle file not found: {fname}")
                    rd = _Lines(fname)
                    hdr = rd.next()  # read (NICO2CN,35) cline, co2g1, co2g2 : (a98,f7.2,16x,f5.2)
                    co2g1 = np.float32(_f(hdr[98:105]))
                    _read_co2_block(rd)  # skip the annual mean part (read (NICO2CN,cform) co2dat)
                    cyc_host = np.zeros((NMONTH, JMXCO2, IMXCO2), dtype=np.float32)
                    gco_host = np.zeros(NMONTH, dtype=np.float32)
                    for imo in range(NMONTH):
                        line = rd.next()  # read (NICO2CN,45) cline, gco2cyc(imo) : (a58,f7.2)
                        gco_host[imo] = _f(line[58:65])
                        cyc_host[imo] = _read_co2_block(rd)
                    gsrc = cp.asarray(gco_host)
                    k_sc(((NMONTH + 255) // 256,), (256,), (gsrc, gco2cyc, np.int32(NMONTH)))
                    csrc = cp.asarray(cyc_host.reshape(-1))
                    k_sc(((n_sav + 255) // 256,), (256,), (csrc, cyc_sav.reshape(-1), np.int32(n_sav)))
                # other ictmflg: co2vmr_sav is allocated but not filled (zeros), as NMMB leaves it

        # co2_glb = co2vmr_def, or co2g1*1e-6 when the user file supplied it (ictmflg -1)
        k_gl((1,), (1,), (np.int32(use_file), co2g1, co2_glb))

    out.update({"co2_glb": co2_glb, "co2vmr_sav": vmr_sav, "co2cyc_sav": cyc_sav,
                "gco2cyc": gco2cyc, "co2g1": float(co2g1),
                "ico2flg": ico2flg, "ivflip": int(tab["ivflip"]),
                "kmonsav": tab.get("kmonsav")})   # kmonsav is set by phys-rrtm-gas-update
    return out
