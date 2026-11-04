"""phys-rrtm-aer-update: aer_update, trop_update, volc_update (RRTM group, climatological aerosols).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_aerosols_nmmb.f:1504-1839 (aer_update and its
contained trop_update and volc_update). NOAA US government work; the NOAA header block of the Fortran
file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_aer_update.cu (aer_trop_fill, one thread per (i,j) cell, one launch).

What runs where:
  device  trop_update's per-cell fill: idxcg, cmixg, denng, kprfg (aer_trop_fill, one launch).
  host    the file reads. Fortran reads text on the host (inquire/open/read); the device cannot do that.
          parse_trop_month reads the selected month's 72 x 37 records (once per monthly update, not per
          step) and uploads them as raw arrays. volc_update reads the decade volcanic table (10 x 4 x 12
          integers) and uploads it. Both are a few thousand lines once a month, and neither is on the step path.
  not ported: the me == 0 print statements (diagnostics, write no state) and the sample-volcanic print.

Calendar control: radupdate (phys-rrtm-radupdate) sets flags[F_CALL_AER]; the driver calls run() then.

In:   ctx.clock.idate (idate[0] = year, idate[1] = month) unless iyear/imon are passed.
      lalwflg, laswflg, lavoflg from ctx.cfg (physparam.f:112-116 defaults, all .true.).
      aeros_path: the tropospheric climatology file. The Fortran opens 'aerosol.dat' in the run directory
      (physparam.f:120); this wrapper defaults to parm/nam_climaeropac_global.txt (sha256 pinned; identity to the
      AERODIR file the NCO script stages is NOT verified, AERODIR is not available). volc_dir: directory for the volcanic decade files
      (default: the working directory, as the Fortran opens them). cfg.aeros_file, when set, names the staged
      aerosol.dat and is used instead of the shipped copy; the sha256 of every file read is recorded in
      ctx.tables["aer_trop_provenance"] for the receipt.
Out:  ctx.tables['aer_idxcg']  int32   [JMXAE, IMXAE, NXC]   idxcg(NXC,IMXAE,JMXAE)
      ctx.tables['aer_cmixg']  float64 [JMXAE, IMXAE, NXC]   cmixg(NXC,IMXAE,JMXAE), kind_phys
      ctx.tables['aer_denng']  float64 [JMXAE, IMXAE, 2]     denng(2,IMXAE,JMXAE)
      ctx.tables['aer_kprfg']  int32   [JMXAE, IMXAE]        kprfg(IMXAE,JMXAE)
      ctx.tables['aer_ivolae'] int32   [10, 4, 12]           ivolae(12,4,10), reversed per CONTRACT section 1
      ctx.tables['aer_volc']   VolcState (host scalars kyrstr, kyrend, kyrsav, kmonsav)
Scratch: the output arrays are allocated once on first call and reused. The raw upload arrays are allocated
per monthly update (month changes only, never per step).

Fortran state that matters downstream: kyrstr = kyrend = kyrsav = kmonsav = 1 at aer_init (line 500-503), so
the first volc_update always reads the decade file for the first requested year.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_aer_update"
KERNEL = "aer_trop_fill"
ROOT = Path(__file__).resolve().parents[3]

NXC = 5            # physparam NXC (radiation_aerosols_nmmb.f:237)
IMXAE = 72         # radiation_aerosols_nmmb.f:240
JMXAE = 37         # radiation_aerosols_nmmb.f:241
MINVYR = 1850      # radiation_aerosols_nmmb.f:222
MAXVYR = 1999      # radiation_aerosols_nmmb.f:223

HDR_LINES = 2                          # read(NIAERCM,12) with format a80/ consumes the month title and column head
RECS_PER_MONTH = IMXAE * JMXAE         # 2664 records, one per (i,j)
MONTH_LINES = HDR_LINES + RECS_PER_MONTH   # 2666 lines per month block (checked: month starts 1, 2667, 5333, ...)
REC_WIDTH_PAIR = 13                    # i2 + e11.4 per component pair (5 pairs = 65 columns)

TROP_FILE = ROOT / "parm" / "nam_climaeropac_global.txt"
# sha256 of the shipped copy above. The NCO run script stages AERODIR/global_climaeropac_global.txt as aerosol.dat
# (exglobal_fcst_nems.sh:946-954); that AERODIR file is not available, so identity to it is NOT verified. The pin
# guarantees the wrapper reads the same bytes every run and makes any drift a loud error.
TROP_FILE_SHA256 = "ac5e965cc70a25cbb2845848e30c62569d7ea33e719a3f463ba5f56e86755727"
PROVENANCE_KEY = "aer_trop_provenance"

KEY_IDX, KEY_CMIX, KEY_DENN, KEY_KPRF = "aer_idxcg", "aer_cmixg", "aer_denng", "aer_kprfg"
KEY_IVOL, KEY_VOLC = "aer_ivolae", "aer_volc"


@dataclass
class VolcState:
    """volc_update's module state (radiation_aerosols_nmmb.f:229 kyrstr, kyrend, kyrsav, kmonsav)."""
    kyrstr: int = 1
    kyrend: int = 1
    kyrsav: int = 1
    kmonsav: int = 1


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _alloc(ctx):
    import cupy as cp

    t = ctx.tables
    if KEY_IDX not in t:
        t[KEY_IDX] = cp.zeros((JMXAE, IMXAE, NXC), dtype=cp.int32)
        t[KEY_CMIX] = cp.zeros((JMXAE, IMXAE, NXC), dtype=cp.float64)
        t[KEY_DENN] = cp.zeros((JMXAE, IMXAE, 2), dtype=cp.float64)
        t[KEY_KPRF] = cp.zeros((JMXAE, IMXAE), dtype=cp.int32)
    if KEY_IVOL not in t:
        t[KEY_IVOL] = cp.zeros((10, 4, 12), dtype=cp.int32)
    if KEY_VOLC not in t:
        t[KEY_VOLC] = VolcState()


def parse_trop_month(path, imon):
    """Host read of the selected month, as trop_update reads it.

    Fortran: for months before imon, one list-directed read per record (skipped); for imon, the format
    5(i2,e11.4),i2,f8.2,i3,1x,a3 per record. Records run j outer, i inner, so record r = (j-1)*IMXAE + (i-1).
    Columns: component pair k (0..4) = line[13k:13k+2] (i2) and line[13k+2:13k+13] (e11.4); kprf = line[65:67];
    denn = line[67:75] (f8.2). nc, ctyp and the lat/lon columns are read by Fortran but unused.
    """
    idx = np.zeros((JMXAE, IMXAE, NXC), dtype=np.int32)
    cmix = np.zeros((JMXAE, IMXAE, NXC), dtype=np.float64)
    kprf = np.zeros((JMXAE, IMXAE), dtype=np.int32)
    denn = np.zeros((JMXAE, IMXAE), dtype=np.float64)
    start = (imon - 1) * MONTH_LINES + HDR_LINES
    with open(path, "r") as f:
        for _ in range(start):
            f.readline()
        for j in range(JMXAE):
            for i in range(IMXAE):
                line = f.readline()
                if not line:
                    raise EOFError(f"aer_update: {path} ends before month {imon} record ({i + 1},{j + 1})")
                for k in range(NXC):
                    b = REC_WIDTH_PAIR * k
                    idx[j, i, k] = int(line[b:b + 2])
                    cmix[j, i, k] = float(line[b + 2:b + 13])
                kprf[j, i] = int(line[65:67])
                denn[j, i] = float(line[67:75])
    return idx, cmix, kprf, denn


def trop_update(ctx, imon, aeros_path):
    import cupy as cp

    _alloc(ctx)
    digest = file_sha256(aeros_path)
    if os.path.abspath(aeros_path) == str(TROP_FILE) and digest != TROP_FILE_SHA256:
        raise ValueError(f"aer_update: {aeros_path} sha256 {digest} != pinned {TROP_FILE_SHA256}; "
                         "the shipped climatology changed, refusing to run with unknown aerosol inputs")
    # Receipt record of the exact input bytes used (provenance for every monthly update).
    ctx.tables.setdefault(PROVENANCE_KEY, {})["trop_file"] = str(aeros_path)
    ctx.tables[PROVENANCE_KEY]["trop_sha256"] = digest
    idx, cmix, kprf, denn = parse_trop_month(aeros_path, imon)
    t = ctx.tables
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(IMXAE, JMXAE), (kernels.BLOCK_COL[0], 1, 1),
          (cp.asarray(idx), cp.asarray(cmix), cp.asarray(kprf), cp.asarray(denn),
           t[KEY_IDX], t[KEY_CMIX], t[KEY_DENN], t[KEY_KPRF]))
    return t[KEY_IDX], t[KEY_CMIX], t[KEY_DENN], t[KEY_KPRF]


def volc_update(ctx, iyear, imon, volc_dir):
    """volc_update (radiation_aerosols_nmmb.f:1800-1839 region, contained in aer_update)."""
    import cupy as cp

    _alloc(ctx)
    vs = ctx.tables[KEY_VOLC]
    vs.kmonsav = imon
    if vs.kyrstr <= iyear <= vs.kyrend:            # use previously input data
        vs.kyrsav = iyear
        return ctx.tables[KEY_IVOL]
    vs.kyrsav = iyear
    vs.kyrstr = iyear - (iyear % 10)               # mod(iyear,10) for calendar years (positive)
    vs.kyrend = vs.kyrstr + 9
    if iyear < MINVYR or iyear > MAXVYR:
        ctx.tables[KEY_IVOL].fill(1)               # lowest value, as the Fortran sets it
        return ctx.tables[KEY_IVOL]
    name = f"volcanic_aerosols_{vs.kyrstr:04d}-{vs.kyrend:04d}.txt"
    base = volc_dir if volc_dir is not None else os.getcwd()
    path = os.path.join(base, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'aer_update: volcanic data file "{name}" not found in {base} (Fortran stops here)')
    ctx.tables.setdefault(PROVENANCE_KEY, {})["volc_file"] = path
    ctx.tables[PROVENANCE_KEY]["volc_sha256"] = file_sha256(path)
    arr = np.zeros((10, 4, 12), dtype=np.int32)   # [k][j][i] = ivolae(i,j,k)
    with open(path, "r") as f:
        f.readline()                               # read(NIAERCM,62) cline, format a80: title record
        for k in range(10):
            for j in range(4):
                line = f.readline()
                arr[k, j, :] = [int(line[5 * i:5 * i + 5]) for i in range(12)]   # format 12i5
    ctx.tables[KEY_IVOL][...] = cp.asarray(arr)
    return ctx.tables[KEY_IVOL]


def aer_update(ctx, iyear, imon, lalwflg=True, laswflg=True, lavoflg=True,
               aeros_path=None, volc_dir=None):
    """aer_update (radiation_aerosols_nmmb.f:1504-1539 region)."""
    if imon < 1 or imon > 12:
        raise ValueError(f"aer_update: ERROR in specifying requested month, imon={imon}")
    if lalwflg or laswflg:                         # monthly tropospheric aerosol data
        trop_update(ctx, imon, str(aeros_path or TROP_FILE))
    if lavoflg:                                    # yearly stratospheric volcanic aerosol data
        volc_update(ctx, iyear, imon, volc_dir)
    return (ctx.tables[KEY_IDX], ctx.tables[KEY_CMIX], ctx.tables[KEY_DENN],
            ctx.tables[KEY_KPRF], ctx.tables[KEY_IVOL])


def run(ctx, iyear=None, imon=None, aeros_path=None, volc_dir=None):
    """Driver entry: called by radupdate when flags[F_CALL_AER] is set."""
    clock = getattr(ctx, "clock", None)
    if iyear is None:
        iyear = int(clock.idate[0])
    if imon is None:
        imon = int(clock.idate[1])
    cfg = getattr(ctx, "cfg", None)
    if aeros_path is None:                         # explicit staged file (AERODIR aerosol.dat) wins over the shipped copy
        aeros_path = getattr(cfg, "aeros_file", None)
    lalw = bool(getattr(cfg, "lalwflg", True))
    lasw = bool(getattr(cfg, "laswflg", True))
    lavo = bool(getattr(cfg, "lavoflg", True))
    return aer_update(ctx, iyear, imon, lalw, lasw, lavo, aeros_path, volc_dir)
