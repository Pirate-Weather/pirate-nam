"""phys-myj-vdifh: VDIFH, vertical diffusion of TH, Q and CWM in the MYJ PBL column.

Fortran: phys/module_BL_MYJPBL.F90:552-625 (MYJPBL main_integration: caller column prep and
the primary-variable tendencies) and 1425-1522 (SUBROUTINE VDIFH). nam.v4.2.11, 83df4e13c1.
Kernels: namgpu/kernels/phys_myj_vdifh.cu, one thread per column:
  phys_myj_vdifh_prep_column  (caller prep, Fortran 552-605)
  phys_myj_vdifh_column       (VDIFH, Fortran 1425-1522)
  phys_myj_vdifh_tend_column  (tendencies, Fortran 619-622)

Inputs (ctx.st; the caller state names, CONTRACT sections 1 and 4):
  3-D [LM, JM, IM]: TH, Q, QC (the Fortran CWM), T, PHMID (PMID), AKH via MYJ_AKH (phys-myj-difcof);
    RTHBLTEN, RQBLTEN, RQCBLTEN (outputs).
  3-D [LM+1, JM, IM]: PHINT (PINH), MYJ_ZINT (ZINT).
  2-D [JM, IM]: TSFC (TSK), XLAND, SICE, SNOW, CHKLOWQ, ELFLX, AKHS, CT, THZ0 (inout), QZ0 (inout),
    QSH (inout, the Fortran QSFC).
  Scratch (st.add on first use): THK, QK, CWMK, AKHK, RHOK [LM, JM, IM]; ZHK [LM+1, JM, IM];
    AKHS_DENS [JM, IM].
  MYJ_ZINT is the Fortran ZINT of the MYJPBL prep (ZINT(K)=ZINT(K+1)+DZ(K), ZINT(LM+1)=HT). No
  wrapper writes it yet; it must be present, and a missing one raises.
Scalars: dtdif (DTTURBL = DT*NPHS, from ctx.cfg), lmh (= LM, the Fortran LMH=LM at line 605).

Loop extent: every column of the full IM x JM domain, as the fused MYJPBL kernel does. The
Fortran DO J=JTS,JTE / DO I=ITS,ITE bounds are the caller's tile limits; whether the NAM driver
passes the full domain interior or excludes boundary rows is not confirmed here.

Refusals (no silent fallback):
  - LM must be between 2 and 64 (VDIFH_LMAX in the kernel).
  - LMH must be in 2..LM.
  - Any named input missing from ctx.st raises.

Launch: three kernels on ctx.stream, no host sync, no per-step allocation (scratch added once).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_vdifh"
PREP_KERNEL = "phys_myj_vdifh_prep_column"
KERNEL = "phys_myj_vdifh_column"
TEND_KERNEL = "phys_myj_vdifh_tend_column"
VDIFH_LMAX = 64


def _field(st, name: str):
    if name not in st.fields:
        raise RuntimeError(f"phys-myj-vdifh: {name} is not in ctx.st; the MYJPBL caller state must provide it")
    return st.fields[name]


def run(ctx, dtdif: float | None = None, lmh: int | None = None):
    """Caller prep, VDIFH and tendencies on every column. Updates the caller state in place."""
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > VDIFH_LMAX:
        raise RuntimeError(f"phys-myj-vdifh: lm={lm} exceeds the kernel's {VDIFH_LMAX} levels")
    if lm < 2:
        raise RuntimeError(f"phys-myj-vdifh: lm={lm} is below 2 levels")
    lmh = lm if lmh is None else int(lmh)
    if not 2 <= lmh <= lm:
        raise RuntimeError(f"phys-myj-vdifh: lmh={lmh} outside 2..{lm}")
    if dtdif is None:
        # DTTURBL = DT * NPHS, in float32 as the fused MYJPBL kernel computes it
        dtdif = float(np.float32(np.float32(cfg.dt) * np.float32(cfg.nphs)))

    f32 = np.float32
    i32 = np.int32
    lev3 = (lm, jm, im)
    lev3p = (lm + 1, jm, im)
    lev2 = (jm, im)

    # caller state (inputs, and the inout 2-D fields)
    th = _field(st, "TH")
    q = _field(st, "Q")
    cwm = _field(st, "QC")
    t = _field(st, "T")
    pmid = _field(st, "PHMID")
    pinh = _field(st, "PHINT")
    zint = _field(st, "MYJ_ZINT")
    akh = _field(st, "MYJ_AKH")
    tsk = _field(st, "TSFC")
    xland = _field(st, "XLAND")
    sice = _field(st, "SICE")
    snow = _field(st, "SNOW")
    chklowq = _field(st, "CHKLOWQ")
    elflx = _field(st, "ELFLX")
    akhs = _field(st, "AKHS")
    ct = _field(st, "CT")
    thz0 = _field(st, "THZ0")
    qz0 = _field(st, "QZ0")
    qsh = _field(st, "QSH")

    # scratch (created once) and the tendency outputs
    thk = st.add("THK", lev3)
    qk = st.add("QK", lev3)
    cwmk = st.add("CWMK", lev3)
    zhk = st.add("ZHK", lev3p)
    rhok = st.add("RHOK", lev3)
    akhk = st.add("AKHK", lev3)
    akhs_dens = st.add("AKHS_DENS", lev2)
    rthblten = st.add("RTHBLTEN", lev3)
    rqblten = st.add("RQBLTEN", lev3)
    rqcblten = st.add("RQCBLTEN", lev3)

    prep = kernels.get(UNIT, PREP_KERNEL)
    diff = kernels.get(UNIT, KERNEL)
    tend = kernels.get(UNIT, TEND_KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        prep(grid, block, (
            th, q, cwm, t, pmid, pinh, zint, akh,
            tsk, xland, sice, snow, chklowq, elflx, akhs,
            thk, qk, cwmk, zhk, rhok, akhk, akhs_dens,
            thz0, qz0, qsh,
            i32(im), i32(jm), i32(lm),
        ))
        diff(grid, block, (
            thk, qk, cwmk, akhk, rhok, zhk, thz0, qz0, akhs_dens, chklowq, ct,
            f32(dtdif), i32(lmh), i32(im), i32(jm), i32(lm),
        ))
        tend(grid, block, (
            th, q, cwm, thk, qk, cwmk, rthblten, rqblten, rqcblten,
            f32(dtdif), i32(im), i32(jm), i32(lm),
        ))
    return st
