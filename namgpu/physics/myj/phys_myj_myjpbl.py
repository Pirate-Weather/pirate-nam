"""phys-myj-myjpbl: NAM MYJPBL, the MYJ level 2.5 PBL driver, one thread per column.

Fortran: phys/module_BL_MYJPBL.F90:180-753 (SUBROUTINE MYJPBL), with MIXLEN, PRODQ2, DIFCOF,
VDIFQ, VDIFH and VDIFV called per column (758-1607). nam.v4.2.11, 83df4e13c115d197db67b972cb3befec8511291a.
Kernel: namgpu/kernels/phys_myj_myjpbl.cu, entry phys_myj_myjpbl_column, one thread per column.
Called from the TURBL driver glue (namgpu/physics/driver/phys_turbl_driver.py, step 6) on the
PBL gate, i.e. when NTSD == 1 or NTSD % NPHS == 0, with DTTURBL = NPHS * DT.

Inputs (ctx.st, NMMB and glue names; CONTRACT sections 1 and 4):
  3-D [LM, JM, IM]: DZ, EXNER, PHMID, Q, QC (cloud condensate, the Fortran CWM), U_PHY, V_PHY,
    T, TH; PHINT [LM+1, JM, IM]; Q2 and EXCH_H (inout).
  2-D [JM, IM]: SFCZ (terrain height, the Fortran HT), TSFC, XLAND, SICE, SNOW, CHKLOWQ, ELFLX,
    AKHS, AKMS, USTAR, UZ0, VZ0, THZ0 (inout), QZ0 (inout), QSH (inout), CT (inout).
  1-D: EPSL [LM-1 used], EPSQ2 [LM].
  HSTDV (2-D): the Fortran STDH. Only used by the orography DCOL term, which MYJPBL sets to
    zero; created as zeros when the glue has not written it.
  Tables from the phys-myj-init unit in ctx.tables: myj_ptbl [JTBL, ITBL] (PTBL), myj_qs0 (QS0),
    myj_sqs (SQS), and the scalars myj_rdth (RDTH) and myj_rdq (RDQ).
Outputs (ctx.st): Q2, EXCH_H (3-D, inout); XLEN_MIX (EL_MYJ, [LM, JM, IM]); DUDT, DVDT,
  RTHBLTEN, RQBLTEN, RQCBLTEN (tendencies, [LM, JM, IM]); PBLH, MIXHT, QSH, CT, THZ0, QZ0 (2-D);
  LPBL (int32, [JM, IM]; the Fortran KPBL).
Scalars: DT and NPHS from ctx.cfg.

Refusals (no silent fallback):
  - LM must be between 4 and 128 (the kernel's per-column arrays are sized for 128 levels).
  - Any named input missing from ctx.st, or any table missing from ctx.tables, raises.

Launch: one kernel on ctx.stream, no host sync, no per-step allocation (HSTDV added once).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_myjpbl"
KERNEL = "phys_myj_myjpbl_column"
MAXL = 128


def _field(st, name: str):
    if name not in st.fields:
        raise RuntimeError(f"phys-myj-myjpbl: {name} is not in ctx.st; the TURBL driver must create it")
    return st.fields[name]


def _table(ctx, key: str):
    if key not in ctx.tables:
        raise RuntimeError(f"phys-myj-myjpbl: ctx.tables['{key}'] is missing; phys-myj-init must write it")
    return ctx.tables[key]


def _scalar(ctx, key: str):
    """phys-myj-tables keeps RDTH/RDQ as 1-element device arrays; the kernel takes them by value.
    Read once (init-time host copy cached in ctx.tables), never per step."""
    hk = key + "_host"
    if hk not in ctx.tables:
        v = _table(ctx, key)
        ctx.tables[hk] = np.float32(float(v.get().ravel()[0]) if hasattr(v, "get") else float(np.ravel(v)[0]))
    return ctx.tables[hk]


def run(ctx):
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > MAXL or lm < 4:
        raise RuntimeError(f"phys-myj-myjpbl: lm={lm}; the kernel supports 4..{MAXL} levels")

    f32 = np.float32
    i32 = np.int32

    ptbl = _table(ctx, "myj_ptbl")
    qs0 = _table(ctx, "myj_qs0")
    sqs = _table(ctx, "myj_sqs")
    rdth = _scalar(ctx, "myj_rdth")
    rdq = _scalar(ctx, "myj_rdq")

    if "HSTDV" not in st.fields:
        st.add("HSTDV", (jm, im))

    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm),
            f32(cfg.dt), i32(cfg.nphs),
            rdth, rdq,
            ptbl, qs0, sqs,
            _field(st, "EPSL"), _field(st, "EPSQ2"),
            _field(st, "DZ"), _field(st, "EXNER"),
            _field(st, "PHMID"), _field(st, "Q"),
            _field(st, "QC"), _field(st, "U_PHY"),
            _field(st, "V_PHY"), _field(st, "T"),
            _field(st, "TH"), _field(st, "PHINT"),
            _field(st, "Q2"), _field(st, "EXCH_H"),
            _field(st, "XLEN_MIX"),
            _field(st, "DUDT"), _field(st, "DVDT"),
            _field(st, "RTHBLTEN"), _field(st, "RQBLTEN"), _field(st, "RQCBLTEN"),
            _field(st, "SFCZ"), _field(st, "HSTDV"),
            _field(st, "TSFC"), _field(st, "XLAND"),
            _field(st, "SICE"), _field(st, "SNOW"),
            _field(st, "CHKLOWQ"), _field(st, "ELFLX"),
            _field(st, "AKHS"), _field(st, "AKMS"),
            _field(st, "USTAR"), _field(st, "UZ0"),
            _field(st, "VZ0"),
            _field(st, "THZ0"), _field(st, "QZ0"),
            _field(st, "QSH"), _field(st, "CT"),
            _field(st, "PBLH"), _field(st, "MIXHT"),
            _field(st, "LPBL"),
        ))
