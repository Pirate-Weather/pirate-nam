"""phys-rrtm-clim-aerinit: CLIM_AERINIT, SET_AERCOEF and OPTAVG (RRTM group, aerosol optics init).

NOAA / NCEP / EMC, NAM nam.v4.2.11 (83df4e13c1), phys/radiation_aerosols_nmmb.f:821-1498 (clim_aerinit,
set_aercoef, optavg). NOAA US government work, no licence file; the NOAA header is kept as the notice in
namgpu/kernels/phys_rrtm_clim_aerinit.cu.
Kernel: namgpu/kernels/phys_rrtm_clim_aerinit.cu (clim_aerinit_bands, then clim_aerinit_optavg).

Runs once at initialisation. aer_init calls clim_aerinit only when iaerflg is not 0 or 100 and iaermdl == 0
(radiation_aerosols_nmmb.f:515, 539, 549); run() raises otherwise. No per-step work.

Host part (init I/O, once per process): parm/nam_climaeropac_global.txt is read the way the Fortran reads
it. The 12 monthly global distributions (each 2 header lines + IMXAE*JMXAE = 2664 integer lines) are
skipped; the optical tables start at line 31993 (the IENDWV header). Each table block is one header line
followed by records laid out by the Fortran edit descriptors (13i6, 20f4.1, 10f7.2, 8e10.3, 8e10.3).

In:   ctx.tables['rrtm_aer_init']['solfwv']  float32[57600]  solar flux per wavenumber (NWVTOT), from phys-rrtm-aer-init
      ctx.tables['rrtm_aer_init']['eirfwv']  float32[4000]   ir flux per wavenumber (NWVTIR), from phys-rrtm-aer-init
      flags laswflg, lalwflg, lalw1bd (physparam defaults: True, True, False)
Out:  ctx.tables (CuPy, Fortran names; shapes are Fortran reversed, CONTRACT section 1):
      iendwv int32[61]; haer, prsref, sigref float32[NAE=7, NDM=5]
      rhidext0/rhidsca0/rhidssa0/rhidasy0 float32[NCM1=6, 61]
      rhdpext0/rhdpsca0/rhdpssa0/rhdpasy0 float32[NCM2=4, NRHLEV=8, 61]; straext0 float32[61]
      solbnd float32[14]; solwaer float32[61, 14]; nv1, nv2 int32[14]
      eirbnd float32[16]; eirwaer float32[61, 16]; nr1, nr2 int32[16]
      extrhi/scarhi/ssarhi/asyrhi float32[nb, 6]; extrhd/scarhd/ssarhd/asyrhd float32[nb, 4, 8]
      extstra float32[nb]          nb = nswbnd + nlwbnd rows: sw 0..nswbnd-1 then lw (Fortran ib = NSWBND + nb).
                                   Default (laswflg, lalwflg True, lalw1bd False): nb = 30, sw rows 0..13, lw 14..29.
                                   laswflg False: nswbnd 0, so lw band nb is row nb. lalw1bd: nlwbnd 1.
Scratch: the constant tables are uploaded once; the band-output tables are reallocated when nb changes.
Launch: two kernel launches on ctx.stream, no host sync inside.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_clim_aerinit"
K_BANDS = "clim_aerinit_bands"
K_OPTAVG = "clim_aerinit_optavg"

from namgpu.paths import PARM  # noqa: E402
DATA_FILE = PARM / "nam_climaeropac_global.txt"

NAERBND = 61
NRHLEV = 8
NCM1 = 6
NCM2 = 4
NDM = 5
NAE = 7
NSW = 14        # NBDSW (radsw_param_nmmb.f NBANDS = NBHGH - NBLOW + 1 = 29 - 16 + 1)
NLW = 16        # NBDLW (radlw_param_nmmb.f NBANDS)
NSWLWBD = NSW + NLW
NSLOT = NCM1 + NCM2 * NRHLEV + 1
NWVTOT = 57600
NWVTIR = 4000

MONTH_LINES = 2 + 72 * 37            # MONTH header + ID/MXRT header + IMXAE*JMXAE integer lines
OPTICAL_START = 12 * MONTH_LINES     # 0-based index of the IENDWV header (file line 31993)

# Band data statements (Fortran module data, copied from the NAM source).
# radsw_param_nmmb.f:179-185 (wvnum1 / wvnum2), radlw_param_nmmb.f:165-171 (wvnlw1 / wvnlw2).
WVNSW1 = [2600.0, 3251.0, 4001.0, 4651.0, 5151.0, 6151.0, 7701.0,
          8051.0, 12851.0, 16001.0, 22651.0, 29001.0, 38001.0, 820.0]
WVNSW2 = [3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0,
          12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 50000.0, 2600.0]
WVNLW1 = [10., 251., 501., 631., 701., 821., 981., 1081.,
          1181., 1391., 1481., 1801., 2081., 2251., 2381., 2601.]
WVNLW2 = [250., 500., 630., 700., 820., 980., 1080., 1180.,
          1390., 1480., 1800., 2080., 2250., 2380., 2600., 3000.]

_HOST = {}


def _fortran_float(s: str) -> float:
    """Fortran F/E edit input: blank field is zero; E without the exponent letter is accepted."""
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        k = max(s.rfind("+"), s.rfind("-"))
        return float(s[:k] + "E" + s[k:])


def _block(lines, pos: int, n: int, per_rec: int, width: int, conv):
    """One table: header record (format 21, a80), then records of per_rec fields of width chars."""
    pos += 1
    vals = []
    while len(vals) < n:
        line = lines[pos]
        pos += 1
        take = min(per_rec, n - len(vals))
        for j in range(take):
            vals.append(conv(line[j * width:(j + 1) * width]))
    return pos, vals


def parse_optical(path) -> dict:
    """Host read of the optical part of the data file (once per process)."""
    if not Path(path).is_file():     # radiation_aerosols_nmmb.f:1008-1013 print and stop
        raise FileNotFoundError(f'Requested aerosol data file "{path}" not found!\n'
                                "*** Stopped in subroutine aero_init !!")
    lines = Path(path).read_text().splitlines()
    if not lines[OPTICAL_START].startswith("ENDING WAVENUMBER"):
        raise RuntimeError(f"{path}: IENDWV header not at line {OPTICAL_START + 1}; layout differs from the NAM file")
    pos = OPTICAL_START
    pos, iendwv = _block(lines, pos, NAERBND, 13, 6, lambda s: int(s))
    pos, haer = _block(lines, pos, NAE * NDM, 20, 4, _fortran_float)
    pos, prsref = _block(lines, pos, NAE * NDM, 10, 7, _fortran_float)
    rhid = []
    for _ in range(4):                       # rhidext0, rhidsca0, rhidssa0, rhidasy0 (61,6)
        pos, v = _block(lines, pos, NAERBND * NCM1, 8, 10, _fortran_float)
        rhid.append(np.array(v, dtype=np.float32).reshape(NCM1, NAERBND))
    rhdp = []
    for _ in range(4):                       # rhdpext0, rhdpsca0, rhdpssa0, rhdpasy0 (61,8,4)
        pos, v = _block(lines, pos, NAERBND * NRHLEV * NCM2, 8, 10, _fortran_float)
        rhdp.append(np.array(v, dtype=np.float32).reshape(NCM2, NRHLEV, NAERBND))
    if not lines[pos].startswith("STRATOSPERIC"):
        raise RuntimeError(f"{path}: STRAEXT header not at line {pos + 1}; layout differs from the NAM file")
    pos, straext = _block(lines, pos, NAERBND, 8, 10, _fortran_float)
    return {
        "iendwv": np.array(iendwv, dtype=np.int32),
        "haer": np.array(haer, dtype=np.float32).reshape(NAE, NDM),
        "prsref": np.array(prsref, dtype=np.float32).reshape(NAE, NDM),
        "rhidext0": rhid[0], "rhidsca0": rhid[1], "rhidssa0": rhid[2], "rhidasy0": rhid[3],
        "rhdpext0": rhdp[0], "rhdpsca0": rhdp[1], "rhdpssa0": rhdp[2], "rhdpasy0": rhdp[3],
        "straext0": np.array(straext, dtype=np.float32),
    }


def _host(path) -> dict:
    key = str(path)
    if key not in _HOST:
        _HOST[key] = parse_optical(path)
    return _HOST[key]


def _allocate(ctx, path):
    """First call only: upload the optical tables and the wavenumber bands, allocate the fixed-size outputs."""
    import cupy as cp

    tb = ctx.tables
    if "rhidext0" not in tb:
        for name, arr in _host(path).items():
            tb[name] = cp.asarray(arr)
        tb["wvnsw1"] = cp.asarray(np.array(WVNSW1, dtype=np.float32))
        tb["wvnsw2"] = cp.asarray(np.array(WVNSW2, dtype=np.float32))
        tb["wvnlw1"] = cp.asarray(np.array(WVNLW1, dtype=np.float32))
        tb["wvnlw2"] = cp.asarray(np.array(WVNLW2, dtype=np.float32))
    if "sigref" in tb:
        return
    f32 = cp.float32
    tb["sigref"] = cp.zeros((NAE, NDM), dtype=f32)
    tb["solbnd"] = cp.zeros(NSW, dtype=f32)
    tb["solwaer"] = cp.zeros((NAERBND, NSW), dtype=f32)
    tb["nv1"] = cp.zeros(NSW, dtype=cp.int32)
    tb["nv2"] = cp.zeros(NSW, dtype=cp.int32)
    tb["eirbnd"] = cp.zeros(NLW, dtype=f32)
    tb["eirwaer"] = cp.zeros((NAERBND, NLW), dtype=f32)
    tb["nr1"] = cp.zeros(NLW, dtype=cp.int32)
    tb["nr2"] = cp.zeros(NLW, dtype=cp.int32)


def _band_outputs(ctx, nbt):
    """Band-output tables sized by the real band count nbt = nswbnd + nlwbnd (reallocated if it changes)."""
    import cupy as cp

    tb = ctx.tables
    if "extstra" in tb and tb["extstra"].shape == (nbt,):
        return
    f32 = cp.float32
    for name in ("extrhi", "scarhi", "ssarhi", "asyrhi"):
        tb[name] = cp.zeros((nbt, NCM1), dtype=f32)
    for name in ("extrhd", "scarhd", "ssarhd", "asyrhd"):
        tb[name] = cp.zeros((nbt, NCM2, NRHLEV), dtype=f32)
    tb["extstra"] = cp.zeros(nbt, dtype=f32)


def run(ctx, laswflg: bool = True, lalwflg: bool = True, lalw1bd: bool = False, aeros_file=None,
        iaerflg: int = 11, iaermdl: int = 0):
    """Two launches: band mapping, then optical averages. Outputs land in ctx.tables.

    The Fortran calls clim_aerinit only for iaerflg not 0 or 100 and iaermdl == 0 (radiation_aerosols_nmmb.f
    :515, :539, :549). The caller gates it (phys_rrtm_radinit); run() asserts the same condition.
    """
    import cupy as cp

    if iaerflg in (0, 100) or iaermdl != 0:
        raise RuntimeError(f"{UNIT}: clim_aerinit runs only for iaerflg not 0 or 100 and iaermdl == 0 "
                           f"(radiation_aerosols_nmmb.f:515, 539, 549); got iaerflg={iaerflg} iaermdl={iaermdl}")

    tb = ctx.tables
    aer = tb.get("rrtm_aer_init")
    if aer is None:
        raise RuntimeError(f"{UNIT} needs ctx.tables['rrtm_aer_init'] from phys-rrtm-aer-init "
                           "(aer_init computes solfwv/eirfwv, radiation_aerosols_nmmb.f:445-816)")
    solfwv, eirfwv = aer["solfwv"], aer["eirfwv"]
    if solfwv.shape != (NWVTOT,) or eirfwv.shape != (NWVTIR,):
        raise RuntimeError(f"solfwv {solfwv.shape} / eirfwv {eirfwv.shape}: want ({NWVTOT},) / ({NWVTIR},)")

    path = aeros_file or DATA_FILE
    _allocate(ctx, path)

    nswbnd = NSW if laswflg else 0
    nlwbnd = (1 if lalw1bd else NLW) if lalwflg else 0
    nbt = nswbnd + nlwbnd
    _band_outputs(ctx, nbt)

    kb = kernels.get(UNIT, K_BANDS)
    ko = kernels.get(UNIT, K_OPTAVG)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        n1 = NSW + NLW + NAE * NDM
        kb(((n1 + 127) // 128,), (128,), (
            solfwv, eirfwv, tb["iendwv"], tb["wvnsw1"], tb["wvnsw2"], tb["wvnlw1"], tb["wvnlw2"],
            tb["prsref"], tb["sigref"], tb["solbnd"], tb["solwaer"], tb["nv1"], tb["nv2"],
            tb["eirbnd"], tb["eirwaer"], tb["nr1"], tb["nr2"],
            np.int32(nswbnd), np.int32(nlwbnd), np.int32(int(laswflg)), np.int32(int(lalwflg))))
        n2 = nbt * NSLOT
        if n2 == 0:
            return tb
        ko(((n2 + 127) // 128,), (128,), (
            tb["rhidext0"], tb["rhidsca0"], tb["rhidssa0"], tb["rhidasy0"],
            tb["rhdpext0"], tb["rhdpsca0"], tb["rhdpssa0"], tb["rhdpasy0"],
            tb["straext0"], tb["solbnd"], tb["solwaer"], tb["nv1"], tb["nv2"],
            tb["eirbnd"], tb["eirwaer"], tb["nr1"], tb["nr2"],
            tb["extrhi"], tb["scarhi"], tb["ssarhi"], tb["asyrhi"],
            tb["extrhd"], tb["scarhd"], tb["ssarhd"], tb["asyrhd"], tb["extstra"],
            np.int32(nswbnd), np.int32(nlwbnd), np.int32(int(laswflg)), np.int32(int(lalwflg))))
    return tb
