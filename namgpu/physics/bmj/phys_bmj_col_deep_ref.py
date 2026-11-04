"""phys-bmj-col-deep-ref: BMJ deep-convection reference profiles, freezing level, first adjustment setup.

Ported from phys/module_CU_BMJ.F90:940-1084 (NAM nam.v4.2.11, 83df4e13c1), the deep path of subroutine
bmj: depth and plume test, deep/shallow flags, label 300 setup, the reference temperature profile
(TTBLEX on the moist adiabat, then the below- and above-freezing descent), depwl, depth, sm1, pbotfc.
Kernel: namgpu/kernels/phys_bmj_col_deep_ref.cu (phys_bmj_col_deep_ref, one thread per column, one launch).

Inputs (ctx.st, CuPy, float32 unless noted; Fortran (i,j,l) -> [l, j, i]):
  T                      [LM,JM,IM]  state T (bmjdrv tcol)
  BMJ_QCOL               [LM,JM,IM]  bmjdrv qcol = max(EPSQ, Q); written by phys-bmj-col-prep
  BMJ_PRSMID             [LM,JM,IM]  bmjdrv pcol = phmid (mid-layer pressure, Pa)
  BMJ_RXNER              [LM,JM,IM]  1/exner (bmj prep)
  BMJ_THES               [LM,JM,IM]  thes(l) as left by cloud-base/cloud-top (theta-e, K)
  BMJ_CT_LBOT, BMJ_CT_LTOP [JM,IM]   int32 inout, cloud base / top level (1-based), from cloud-top
  BMJ_CT_PBOT, BMJ_CT_PTOP [JM,IM]   float32 inout, cloud base / top pressure (Pa), from cloud-top.
                         The quick exit (Fortran 931-936) rewrites these four and CLDEFI in place,
                         as the Fortran does to its bmj locals and cldefi; later units read them.
  BMJ_CT_CAPE            [JM,IM]     cloud-top CAPE = capecnv (phys-bmj-col-cloudtop)
  SM                     [JM,IM]     land mask sm (the cloud-top input)
  PINT                   [LM+1,JM,IM] state; psfc = PINT[LM] (bmjdrv psfc = phint(lm+1))
  CLDEFI                 [JM,IM]     state cloud efficiency (inout, see quick exit)
  BMJ_TAB_*              the BMJ_INIT tables: ctx.tables["bmj_sthe"], ["bmj_the0"], ["bmj_ttbl"],
                         ["bmj_stheq"], ["bmj_the0q"], ["bmj_ttblq"] (phys-bmj-init). Fortran
                         ttbl(jtb,itb) = [itb, jtb].
  Constants: ELWV, ELIV, TIW (module_CONSTANTS, namgpu config).

Outputs (ctx.st, scratch added once with st.add):
  [LM,JM,IM] float32: BMJ_DR_TREFK, BMJ_DR_QREFK, BMJ_DR_THERK, BMJ_DR_RXNERK, BMJ_DR_EL, BMJ_DR_PK,
                      BMJ_DR_PSK, BMJ_DR_QK, BMJ_DR_TK, BMJ_DR_DIFT, BMJ_DR_DIFQ, BMJ_DR_TREF
  [JM,IM] int32:      BMJ_DR_LB, BMJ_DR_L0, BMJ_DR_DEEP, BMJ_DR_PLUME, BMJ_DR_SHALLOW
  [JM,IM] float32:    BMJ_DR_PK0, BMJ_DR_PKB, BMJ_DR_PKT, BMJ_DR_STABDL, BMJ_DR_EFI, BMJ_DR_DEPTH,
                      BMJ_DR_DEPWL, BMJ_DR_PBOTFC, BMJ_DR_SM1
Shallow columns (DEEP = 0) get only PLUME, DEEP, SHALLOW written; the other outputs are stale there.

Launch: one kernel, grid from kernels.grid_cols, on ctx.stream; no host sync, no per-step allocation.
run(ctx): the bmjdrv-chain entry. It refuses (RuntimeError naming the missing fields or tables) when an
upstream input or a BMJ_INIT table is absent. It never fills inputs with synthetic values.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_deep_ref"
KERNEL = "phys_bmj_col_deep_ref"

ITB, JTB, ITBQ, JTBQ = 76, 134, 152, 440   # module_CU_BMJ.F90 parameters
ELWV, ELIV, TIW = 2.501e6, 2.850e6, 273.15  # module_CONSTANTS (as in gpuwm kernels/nmmb_common.cuh)

OUT3 = ("BMJ_DR_TREFK", "BMJ_DR_QREFK", "BMJ_DR_THERK", "BMJ_DR_RXNERK", "BMJ_DR_EL", "BMJ_DR_PK",
        "BMJ_DR_PSK", "BMJ_DR_QK", "BMJ_DR_TK", "BMJ_DR_DIFT", "BMJ_DR_DIFQ", "BMJ_DR_TREF")
OUT2_INT = ("BMJ_DR_LB", "BMJ_DR_L0", "BMJ_DR_DEEP", "BMJ_DR_PLUME", "BMJ_DR_SHALLOW")
OUT2_F = ("BMJ_DR_PK0", "BMJ_DR_PKB", "BMJ_DR_PKT", "BMJ_DR_STABDL", "BMJ_DR_EFI", "BMJ_DR_DEPTH",
          "BMJ_DR_DEPWL", "BMJ_DR_PBOTFC", "BMJ_DR_SM1")


def _ensure_outputs(st, IM, JM, LM):
    for n in OUT3:
        if n not in st.fields:
            st.add(n, (LM, JM, IM))
    for n in OUT2_INT:
        if n not in st.fields:
            st.add(n, (JM, IM), dtype="int32")
    for n in OUT2_F:
        if n not in st.fields:
            st.add(n, (JM, IM))


def phys_bmj_col_deep_ref(ctx, *, IM, JM, LM, tables, inputs):
    """One launch over every column. inputs: dict of the BMJ input arrays (see module docstring)."""
    st = ctx.st
    _ensure_outputs(st, IM, JM, LM)
    k = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(IM, JM)
    block = kernels.BLOCK_COL
    o3 = [st[n] for n in OUT3]
    oi = [st[n] for n in OUT2_INT]
    of = [st[n] for n in OUT2_F]
    args = (
        np.int32(IM), np.int32(JM), np.int32(LM),
        np.float32(ELWV), np.float32(ELIV), np.float32(TIW),
        inputs["T"], inputs["BMJ_QCOL"], inputs["BMJ_PRSMID"], inputs["BMJ_RXNER"], inputs["BMJ_THES"],
        inputs["BMJ_LBOT"], inputs["BMJ_LTOP"], inputs["BMJ_PBOT"], inputs["BMJ_PTOP"],
        inputs["PSFC"], inputs["BMJ_SEAMASK"], inputs["CLDEFI"], inputs["BMJ_CAPE"],
        tables["sthe"], tables["the0"], tables["ttbl"],
        tables["stheq"], tables["the0q"], tables["ttblq"],
        *o3, *oi, *of,
    )
    k(grid, block, args, stream=ctx.stream if ctx.stream is not None else 0)
    return {n: st[n] for n in OUT3 + OUT2_INT + OUT2_F}


_REQ_FIELDS = ("T", "BMJ_QCOL", "BMJ_PRSMID", "BMJ_RXNER", "BMJ_THES", "BMJ_CT_LBOT",
               "BMJ_CT_LTOP", "BMJ_CT_PBOT", "BMJ_CT_PTOP", "BMJ_CT_CAPE", "SM", "CLDEFI", "PINT")
_REQ_TABLES = {"sthe": "bmj_sthe", "the0": "bmj_the0", "ttbl": "bmj_ttbl",
               "stheq": "bmj_stheq", "the0q": "bmj_the0q", "ttblq": "bmj_ttblq"}


def run(ctx):
    """Entry for the bmjdrv chain. One launch over the full grid; returns the dict of outputs."""
    IM, JM, LM = ctx.grid.im, ctx.grid.jm, ctx.grid.lm
    st = ctx.st
    missing = [n for n in _REQ_FIELDS if n not in st.fields]
    if missing:
        raise RuntimeError(f"phys-bmj-col-deep-ref: state fields missing (upstream unit not run): {missing}")
    missing_t = [v for v in _REQ_TABLES.values() if v not in ctx.tables]
    if missing_t:
        raise RuntimeError(f"phys-bmj-col-deep-ref: ctx.tables missing (phys-bmj-init): {missing_t}")
    inputs = {
        "T": st["T"], "BMJ_QCOL": st["BMJ_QCOL"], "BMJ_PRSMID": st["BMJ_PRSMID"],
        "BMJ_RXNER": st["BMJ_RXNER"], "BMJ_THES": st["BMJ_THES"],
        "BMJ_LBOT": st["BMJ_CT_LBOT"], "BMJ_LTOP": st["BMJ_CT_LTOP"],
        "BMJ_PBOT": st["BMJ_CT_PBOT"], "BMJ_PTOP": st["BMJ_CT_PTOP"],
        "BMJ_CAPE": st["BMJ_CT_CAPE"], "BMJ_SEAMASK": st["SM"], "CLDEFI": st["CLDEFI"],
        "PSFC": st["PINT"][LM],
    }
    tables = {k: ctx.tables[v] for k, v in _REQ_TABLES.items()}
    return phys_bmj_col_deep_ref(ctx, IM=IM, JM=JM, LM=LM, tables=tables, inputs=inputs)
