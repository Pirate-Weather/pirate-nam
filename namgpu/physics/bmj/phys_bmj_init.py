"""phys-bmj-init: BMJ_INIT look-up tables and cldefi, Betts-Miller-Janjic convection.

Ported from phys/module_CU_BMJ.F90:2158-2425 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE bmj_init,
with the SPLINE routine of the same module. Kernels: namgpu/kernels/phys_bmj_init.cu (four
launches, one per table section, plus cldefi). Built once at init; the tables live in
ctx.tables and never change after that.

Writes (ctx.tables, CuPy float32; Fortran module-private arrays of module_CU_BMJ):
    bmj_sthe, bmj_the0     [ITB]        Fortran sthe(itb), the0(itb)
    bmj_qs0, bmj_sqs       [JTB]        Fortran qs0(jtb), sqs(jtb)
    bmj_qs0_exp, bmj_sqs_exp [JTB]      module-share copies (module_bl_myjpbl)
    bmj_ptbl               [JTB, ITB]   Fortran ptbl(itb,jtb): element (kp,kth) at [kth-1, kp-1]
    bmj_ptbl_exp           [JTB, ITB]   module-share copy of ptbl
    bmj_ttbl               [ITB, JTB]   Fortran ttbl(jtb,itb): element (kth,kp) at [kp-1, kth-1]
    bmj_stheq, bmj_the0q   [ITBQ]       Fortran stheq(itbq), the0q(itbq)
    bmj_ttblq              [ITBQ, JTBQ] Fortran ttblq(jtbq,itbq): element (kth,kp) at [kp-1, kth-1]
    ITB=76, JTB=134, ITBQ=152, JTBQ=440 (module parameters, module_CU_BMJ.F90:65-66).
Writes the state field CLDEFI (Fortran cldefi(ims:ime,jms:jme), CuPy [JM, IM]) when restart is
False: avgefi on the interior (i < IM, j < JM in 1-based, as the Fortran itf/jtf limits).

Scalars a2, a3, a4, cappa, cp, pq0 are the bmj_init arguments; the NMMB values are passed here
(nmmb_common.cuh constants, single precision).

Launch: one kernel per routine section, on ctx.stream, no host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_init"
ITB, JTB, ITBQ, JTBQ = 76, 134, 152, 440
BLOCK = 64

# bmj_init scalar arguments (NMMB module_CONSTANTS values, single precision).
A2 = np.float32(17.2693882)
A3 = np.float32(273.15)
A4 = np.float32(35.86)
PQ0 = np.float32(379.90516)
CP = np.float32(1004.6)
R_D = np.float32(287.04)
CAPPA = np.float32(R_D / CP)


def _alloc(tables):
    import cupy as cp

    f = cp.float32
    tables["bmj_sthe"] = cp.zeros((ITB,), f)
    tables["bmj_the0"] = cp.zeros((ITB,), f)
    tables["bmj_qs0"] = cp.zeros((JTB,), f)
    tables["bmj_sqs"] = cp.zeros((JTB,), f)
    tables["bmj_qs0_exp"] = cp.zeros((JTB,), f)
    tables["bmj_sqs_exp"] = cp.zeros((JTB,), f)
    tables["bmj_ptbl"] = cp.zeros((JTB, ITB), f)
    tables["bmj_ptbl_exp"] = cp.zeros((JTB, ITB), f)
    tables["bmj_ttbl"] = cp.zeros((ITB, JTB), f)
    tables["bmj_stheq"] = cp.zeros((ITBQ,), f)
    tables["bmj_the0q"] = cp.zeros((ITBQ,), f)
    tables["bmj_ttblq"] = cp.zeros((ITBQ, JTBQ), f)


def run(ctx, restart: bool = False):
    """Build the BMJ tables into ctx.tables and set CLDEFI (unless restart). Returns the tables."""
    tables = ctx.tables
    if "bmj_ttbl" not in tables:
        _alloc(tables)

    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    cldefi = ctx.st.add("CLDEFI", (jm, im))

    k_sat = kernels.get(UNIT, "phys_bmj_init_sat")
    k_ttbl = kernels.get(UNIT, "phys_bmj_init_ttbl")
    k_ttblq = kernels.get(UNIT, "phys_bmj_init_ttblq")
    k_cldefi = kernels.get(UNIT, "phys_bmj_init_cldefi")

    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_sat(((JTB + BLOCK - 1) // BLOCK,), (BLOCK,), (
            tables["bmj_qs0"], tables["bmj_sqs"], tables["bmj_qs0_exp"], tables["bmj_sqs_exp"],
            tables["bmj_ptbl"], tables["bmj_ptbl_exp"],
            A2, A3, A4, CAPPA, PQ0,
        ))
        k_ttbl(((ITB + BLOCK - 1) // BLOCK,), (BLOCK,), (
            tables["bmj_ttbl"], tables["bmj_the0"], tables["bmj_sthe"],
            A2, A3, A4, CAPPA, CP, PQ0,
        ))
        k_ttblq(((ITBQ + BLOCK - 1) // BLOCK,), (BLOCK,), (
            tables["bmj_ttblq"], tables["bmj_the0q"], tables["bmj_stheq"],
            A2, A3, A4, CAPPA, CP, PQ0,
        ))
        n = im * jm
        k_cldefi(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (
            cldefi, np.int32(im), np.int32(jm), np.int32(1 if restart else 0),
        ))
    return tables
