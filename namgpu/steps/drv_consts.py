"""drv-consts: derived dynamics constants of NMMB CONSTS, into ctx.tables['consts'].

Ported from nmm/module_CONTROL.F90:123-345 (NAM nam.v4.2.11, 83df4e13c1), CONSTS.
Kernel: namgpu/kernels/drv_consts.cu (consts_scalars, one thread, one launch).

CONSTS calls grid_consts, exptbl, tablep, tablet and psitbl. grid_consts is unit init-grid-consts
(its rows are already in ctx.tables['grid'] when this unit runs, see init_case_driver.CHAIN).
exptbl, tablep, tablet and psitbl are units drv-exptbl, drv-tablep, drv-tablet and drv-psitbl and
are not ported yet; pl (the input to tablep) is written here at CF_PL for them.

In:      ctx.cfg (dt, smag2, codamp, wcor, pt, lnsh, global_, im, jm), ctx.grid (im, jm),
         ctx.case_dir (boco.0000 written by the boundary preparation, optional).
Out:     ctx.tables['consts'] = {
             'f': float32[8] device array, indices CF_*;
             'i': int32[8] device array, indices CI_*;
             'tboco': float (s), 'tboco_source': 'boco.0000' | 'nam_namelist.nps interval_seconds'
                      | 'none (global branch)'}
Scratch: none on the step path. Runs once at init; the output is read by the units that need it.

tboco: the Fortran reads it from the record ``read (nbc) runbc,idatbc,ihrstbc,tboco`` of boco.0000
(module_CONTROL.F90:283), an unformatted sequential record: 4-byte length, payload (logical runbc,
int32 idatbc(3), int32 ihrstbc, real32 tboco; 24 bytes, little endian), 4-byte length. The Fortran
reads it unconditionally and stops when the file is missing. Here, when boco.0000 is missing (a
check run or a fresh case before boundary preparation), tboco comes from parm/nam_namelist.nps
interval_seconds, which is the value NPS writes as tboco (gridinfo_module.F90:223). That is a
deliberate deviation from the Fortran, it is printed on every run, and tboco_source names it.
The 24-byte layout was checked against module_CONTROL.F90:95-104 and :283. No boco.0000 file exists
available yet, so the layout has not been checked against a real file.
"""
from __future__ import annotations

import contextlib
import re
import struct
from pathlib import Path

import numpy as np

from namgpu import kernels, slab
from namgpu.config import PARM

UNIT = "drv_consts"
KERNEL = "consts_scalars"

# Single process, one domain (CONTRACT section 1): the domain id and index limits the Fortran
# call receives. The kernel takes them as arguments, so a nest can pass its own.
MY_DOMAIN_ID = 1

# float32 slots (kernel outf)
CF_DT, CF_EF4T, CF_PL, CF_BOFAC, CF_SMAG2, CF_CODAMP, CF_WCOR, CF_PT = range(8)
# int32 slots (kernel outi)
CI_LNSBC, CI_LNSAD, CI_ADV_STANDARD, CI_ADV_UPSTREAM, CI_NBOCO = range(5)

_PAYLOAD_BYTES = 4 + 3 * 4 + 4 + 4


def read_tboco_boco(path: Path) -> float:
    """tboco from a boco.#### file: one Fortran unformatted sequential record (module_CONTROL:283)."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    if n != _PAYLOAD_BYTES or len(raw) < 4 + n + 4:
        raise ValueError(f"{path}: first record is {n} bytes, expected {_PAYLOAD_BYTES} (runbc,idatbc,ihrstbc,tboco)")
    (tboco,) = struct.unpack_from("<f", raw, 4 + _PAYLOAD_BYTES - 4)
    return float(tboco)


def tboco_from_nps(path: Path = PARM / "nam_namelist.nps") -> float:
    """interval_seconds of the NPS namelist: the value NPS writes as tboco (gridinfo_module.F90:223)."""
    m = re.search(r"interval_seconds\s*=\s*([0-9.eE+-]+)", Path(path).read_text())
    if m is None:
        raise ValueError(f"{path}: no interval_seconds, cannot set tboco")
    return float(m.group(1))


def _tboco(ctx):
    case_dir = Path(ctx.case_dir) if ctx.case_dir is not None else Path.cwd()
    # init-gfs-bocos writes <case>/boco and sets ctx.boco_dir (the run directory of the Fortran job)
    dirs = [Path(d) for d in (getattr(ctx, "boco_dir", None),) if d] + [case_dir / "boco", case_dir]
    for d in dirs:
        boco = d / "boco.0000"
        if boco.exists():
            return read_tboco_boco(boco), "boco.0000"
    return tboco_from_nps(), "nam_namelist.nps interval_seconds"


def run(ctx):
    import cupy as cp

    cfg = ctx.cfg
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    is_global = bool(cfg.global_)
    if is_global:
        tboco, src = 0.0, "none (global branch)"
    else:
        tboco, src = _tboco(ctx)
        # Fortran write(0,*) equivalent (module_CONTROL:291), plus the deviation notice.
        print(f"drv-consts: tboco = {tboco} s from {src}", flush=True)
        if src != "boco.0000":
            where = Path(ctx.case_dir) if ctx.case_dir is not None else Path.cwd()
            print(f"drv-consts: WARNING no boco.0000 in {where}; the Fortran would stop here. "
                  f"Using nam_namelist.nps interval_seconds.", flush=True)

    outf = cp.zeros(8, dtype=cp.float32)
    outi = cp.zeros(8, dtype=cp.int32)

    kern = kernels.get(UNIT, KERNEL)
    # domain and tile limits of the single process: ids=jds=1, ide=IM, jde=JM, its=ids, ite=ide, ...
    args = (
        np.int32(1 if is_global else 0),
        np.int32(cfg.lnsh),
        np.int32(MY_DOMAIN_ID),
        # row slab: the advection-region flags are the whole-domain ones (every rank computes all its
        # memory rows against GLOBAL domain tests), so the tile passed here is the global domain.
        np.int32(1), np.int32(im), np.int32(1), np.int32(slab.rows(ctx)[1]),   # ids, ide, jds, jde
        np.int32(1), np.int32(im), np.int32(1), np.int32(slab.rows(ctx)[1]),   # its, ite, jts, jte
        np.float32(cfg.dt), np.float32(cfg.smag2), np.float32(cfg.codamp),
        np.float32(cfg.wcor), np.float32(cfg.pt), np.float32(tboco),
        outf, outi,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern((1, 1, 1), (1, 1, 1), args)

    ctx.tables["consts"] = {"f": outf, "i": outi, "tboco": tboco, "tboco_source": src}
    return ctx.tables["consts"]
