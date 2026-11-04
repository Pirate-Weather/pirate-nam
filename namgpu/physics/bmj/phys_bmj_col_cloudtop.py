"""phys-bmj-col-cloudtop: BMJ cloud-top search, entrainment, buoyancy (CAPE) and maximum-instability
selection, with the post-loop quick exit (module_CU_BMJ.F90:697-939, NAM nam.v4.2.11, 83df4e13c1).

Kernel: namgpu/kernels/phys_bmj_col_cloudtop.cu (phys_bmj_col_cloudtop, one thread per column,
one launch, grid_cols). The per-column max_buoy_loop (671-905) and the post-loop section (906-939)
run in that thread; the device functions are bmjct_kb_body (697-905), bmjct_post (906-939) and
bmjct_cloudbase_kb (671-696, carried because the cloud-top body reads its loop-carried lbot).

Reads (ctx.st, NMMB names). Every input is required; the wrapper refuses to run without it and never
fills a stand-in (review of phys-bmj-col-cloudtop, 2026-11-04):
    T, Q                 float32 [LM, JM, IM]   tk, qk
    PINT                 float32 [LM+1, JM, IM] for DPRS (dprs = PINT(k+1)-PINT(k))
    PHMID, EXNER         float32 [LM, JM, IM]   prsmid (pcol = phmid), exner (exnercol). Written by the
                                                BMJ driver / phys-bmj-col-prep (the same contract
                                                phys-bmj-col-cloudbase enforces for PHMID).
    XLAND                float32 [JM, IM]       NMMB land/sea flag. The BMJ seamask is XLAND - 1
                                                (module_CU_BMJ.F90 driver line 286, seamask=xland(i,j)-1,
                                                 1.0 for sea, 0.0 for land). Derived here as BMJ_SEAMASK.
    BMJ_RHK              float32 [LM, JM, IM]   rhk (phys-bmj-col-prep, the BMJ prelude)
    BMJ_PSK, BMJ_RXNERSK, BMJ_THESP   float32 [LM, JM, IM]   prep_loop outputs of the cloud-base unit
                         (phys-bmj-col-cloudbase, Fortran 598-670)
    CLDEFI               float32 [JM, IM]       inout (phys-bmj-init writes it)
    BMJ tables           ctx.tables["bmj_sthe"], ["bmj_the0"], ["bmj_ttbl"], ["bmj_stheq"],
                         ["bmj_the0q"], ["bmj_ttblq"] (phys-bmj-init, built here when missing).
Config flags (cfg.raw): entrain, newswap, nodeep (nam_configfile, the bmjdrv arguments).

Writes (scratch, st.add; zero on first use):
    CLDEFI (in place); BMJ_CT_LBOT, BMJ_CT_LTOP int32 [JM, IM]; BMJ_CT_PBOT, BMJ_CT_PTOP, BMJ_CT_PSP,
    BMJ_CT_THBT, BMJ_CT_QBT, BMJ_CT_RXNERS, BMJ_CT_CAPE float32 [JM, IM] (CAPE = capecnv);
    BMJ_CPE, BMJ_DTV, BMJ_THES, BMJ_QBTK float32 [LM, JM, IM] for levels 1..LMH.

Launch: one kernel, on ctx.stream, no host sync, no per-step allocation after the first call (scratch
is added once). Scalars come from the NMMB constants (nmmb_common.cuh) and the config flags.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.bmj.phys_bmj_bmjdrv import _flag

UNIT = "phys_bmj_col_cloudtop"
KERNEL = "phys_bmj_col_cloudtop"

# Kernel column-array capacity (BMJCT_LMAX in phys_bmj_col_cloudtop.cu). Above it the kernel's
# outputs would not be written, so the wrapper refuses before launch.
BMJCT_LMAX = 64

# NMMB constants (nmmb_common.cuh, module_CONSTANTS), Fortran literals.
A2 = np.float32(17.2693882)
A3 = np.float32(273.15)
A4 = np.float32(35.86)
CP = np.float32(1004.6)
R_D = np.float32(287.04)
R_V = np.float32(461.6)
P608 = np.float32(R_V / R_D - np.float32(1.0))
PQ0 = np.float32(379.90516)
ELIWV = np.float32(2.683e6)
P0 = np.float32(1.0e5)
CAPPA = np.float32(R_D / CP)
ELOCP = np.float32(ELIWV / CP)

# rh_deep: 0.90 if nodeep else 0.75 (module_CU_BMJ.F90:557-561).
RH_DEEP_NODEEP = np.float32(0.90)
RH_DEEP = np.float32(0.75)

OUT_2D_I32 = ("BMJ_CT_LBOT", "BMJ_CT_LTOP")
OUT_2D = ("BMJ_CT_PBOT", "BMJ_CT_PTOP", "BMJ_CT_PSP", "BMJ_CT_THBT", "BMJ_CT_QBT",
          "BMJ_CT_RXNERS", "BMJ_CT_CAPE")
OUT_3D = ("BMJ_CPE", "BMJ_DTV", "BMJ_THES", "BMJ_QBTK")

# (state name, producer) for every input the unit reads. Missing any one is a refusal, never a fill.
REQUIRED = (
    ("PHMID", "BMJ driver / phys-bmj-col-prep"),
    ("EXNER", "BMJ driver / phys-bmj-col-prep"),
    ("BMJ_RHK", "phys-bmj-col-prep (BMJ prelude)"),
    ("BMJ_PSK", "phys-bmj-col-cloudbase"),
    ("BMJ_RXNERSK", "phys-bmj-col-cloudbase"),
    ("BMJ_THESP", "phys-bmj-col-cloudbase"),
    ("CLDEFI", "phys-bmj-init"),
    ("XLAND", "the initial state (NMMB land/sea flag)"),
    ("PINT", "the dynamics state"),
    ("T", "the dynamics state"),
    ("Q", "the dynamics state"),
)


def _ensure_tables(ctx):
    """BMJ_INIT tables (phys-bmj-init) built once into ctx.tables."""
    from namgpu.physics.bmj import phys_bmj_init

    if "bmj_ttbl" not in ctx.tables:
        phys_bmj_init.run(ctx)
    t = ctx.tables
    return {
        "sthe": t["bmj_sthe"], "the0": t["bmj_the0"], "ttbl": t["bmj_ttbl"],
        "stheq": t["bmj_stheq"], "the0q": t["bmj_the0q"], "ttblq": t["bmj_ttblq"],
    }


def _require_inputs(st):
    missing = [f"{name} (from {who})" for name, who in REQUIRED if name not in st.fields]
    if missing:
        raise RuntimeError(
            "phys-bmj-col-cloudtop: inputs missing from ctx.st, run the upstream units first: "
            + "; ".join(missing))


def run(ctx):
    """Done-rule entry point: one launch over the full [LM, JM, IM] domain on ctx.stream."""
    import cupy as cp

    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm > BMJCT_LMAX:
        raise RuntimeError(
            f"phys-bmj-col-cloudtop: lm={lm}; the kernel column arrays hold {BMJCT_LMAX} levels "
            "(BMJCT_LMAX). Outputs would not be written, so the run is refused.")
    lmh = lm
    shape3 = (lm, jm, im)
    shape2 = (jm, im)

    _require_inputs(st)

    raw = cfg.raw
    entrain = _flag(raw, "entrain")
    newswap = _flag(raw, "newswap")
    nodeep = bool(_flag(raw, "nodeep"))
    rh_deep = RH_DEEP_NODEEP if nodeep else RH_DEEP

    tb = _ensure_tables(ctx)

    # BMJ seamask (Fortran sm): seamask = xland(i,j) - 1 (module_CU_BMJ.F90 driver line 286).
    seamask = st.add("BMJ_SEAMASK", shape2)
    cp.subtract(st["XLAND"], np.float32(1.0), out=seamask)

    # DPRS(k) = PINT(k+1) - PINT(k), the bmjdrv DPCOL (phys_bmj_bmjdrv.cu, Fortran 189-196).
    dprs = st.add("BMJ_CT_DPRS", shape3)
    cp.subtract(st["PINT"][1:], st["PINT"][:-1], out=dprs)

    lbot = st.add("BMJ_CT_LBOT", shape2, dtype="int32")
    ltop = st.add("BMJ_CT_LTOP", shape2, dtype="int32")
    outs2 = [st.add(n, shape2) for n in OUT_2D]
    outs3 = [st.add(n, shape3) for n in OUT_3D]

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm), np.int32(lmh),
            entrain, newswap, np.float32(rh_deep),
            A2, A3, A4, P608, PQ0, R_D,
            st["T"], st["Q"], st["PHMID"], dprs, st["EXNER"], st["BMJ_THESP"], st["BMJ_RHK"],
            st["BMJ_PSK"], st["BMJ_RXNERSK"],
            seamask, st["CLDEFI"],
            tb["sthe"], tb["the0"], tb["ttbl"], tb["stheq"], tb["the0q"], tb["ttblq"],
            lbot, ltop, *outs2, *outs3,
        ))
    return {"lbot": lbot, "ltop": ltop, **dict(zip(OUT_2D, outs2)), **dict(zip(OUT_3D, outs3))}
