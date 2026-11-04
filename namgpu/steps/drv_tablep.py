"""drv-tablep: TABLEP, the saturation-pressure / potential-temperature table for the moist processes.

Ported from nmm/module_CONTROL.F90:1868-1964 (NAM nam.v4.2.11, 83df4e13c1); the CALL SPLINE
(module_CONTROL.F90:2074-2183) is inlined in the kernel.  Kernel: namgpu/kernels/drv_tablep.cu
(drv_tablep).

Writes (ctx.tables, built once at init and never rebuilt per step):
    ptbl  float32 [JTB=601, ITB=201]   Fortran ptbl(itb,jtb), CuPy [kth, kp]
    qs0   float32 [JTB=601]            Fortran qs0(jtb)
    sqs   float32 [JTB=601]            Fortran sqs(jtb)
    rdth  float                        module scalar 1/dth, dth = (thh-thl)/(jtb-1)
    rdq   float                        module scalar 1/dqs, dqs = 1/(itb-1)
Reads : ctx.grid.pt (the model top pressure PT, Pa; pl = max(pt, 1000) as in SOLVER consts).
Launch: one kernel, JTB threads (one per kth row, each row a sequential spline chain), on
        ctx.stream.  No host sync; the two scalars are float32 host arithmetic done once.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_tablep"
ITB = 201
JTB = 601
BLOCK = 128


def run(ctx):
    import cupy as cp

    pl = np.float32(max(float(ctx.grid.pt), 1000.0))   # module_CONTROL.F90:338, pl = max(pt, 1000)

    tables = ctx.tables
    if "ptbl" not in tables:
        tables["ptbl"] = cp.zeros((JTB, ITB), dtype=cp.float32)
        tables["qs0"] = cp.zeros((JTB,), dtype=cp.float32)
        tables["sqs"] = cp.zeros((JTB,), dtype=cp.float32)

    # Module scalars with the Fortran's float32 operations (kfpt = single).
    f32 = np.float32
    dth = (f32(350.0) - f32(200.0)) / f32(JTB - 1)     # thh - thl, / real(jtb-1)
    rdth = f32(1.0) / dth
    dqs = f32(1.0) / f32(ITB - 1)
    rdq = f32(1.0) / dqs
    tables["rdth"] = float(rdth)
    tables["rdq"] = float(rdq)

    k_tablep = kernels.get(UNIT, "drv_tablep")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_tablep(((JTB + BLOCK - 1) // BLOCK,), (BLOCK,), (
            tables["ptbl"], tables["qs0"], tables["sqs"], pl,
        ))
