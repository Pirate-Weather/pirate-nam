"""phys-fer-gpvs: GPVS_hr, the saturation vapour pressure lookup table for Ferrier-Aligo microphysics.

Ported from phys/module_MP_FER_HIRES.F90:2791-2836 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE GPVS_hr.
Kernel: namgpu/kernels/phys_fer_gpvs.cu (gpvs_hr, one launch, one thread per table entry).

Writes (ctx.tables, rebuilt on each call; the call is once at init, the table is 7501 entries):
    fer_gpvs_tbpvs   float32 [NX=7501]   Fortran TBPVS(1:NX)  (FPVSX over 180..330 K)
    fer_gpvs_tbpvs0  float32 [NX=7501]   Fortran TBPVS0(1:NX) (FPVSX0 over 180..330 K)
    fer_gpvs_coef    float32 [4]         {C1XPVS, C2XPVS, C1XPVS0, C2XPVS0}
Consumers: phys-fer-fpvs (FPVS, FPVS0 lookups), phys-fer-calmict, phys-fer-condense, phys-fer-deposit,
phys-rrtm-driver-call and phys-fer-col-satcheck, all reading these exact keys.
Restart: Fortran saves TBPVS/TBPVS0/C1XPVS*/C2XPVS* to MP_RESTART_STATE and restores them at init; this port
does not write that state. It rebuilds the table instead. The rebuild is deterministic (one launch, no atomics,
same inputs), so a restarted run reproduces the same table on the same card. Not claimed bitwise to Fortran.
No host arithmetic; no host sync; scratch allocated on first call and reused.
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "phys_fer_gpvs"
KERNEL = "gpvs_hr"
NX = 7501
BLOCK = 128
KEY_TBPVS = "fer_gpvs_tbpvs"
KEY_TBPVS0 = "fer_gpvs_tbpvs0"
KEY_COEF = "fer_gpvs_coef"


def run(ctx):
    import cupy as cp

    tables = ctx.tables
    if KEY_TBPVS not in tables:
        tables[KEY_TBPVS] = cp.zeros(NX, dtype=cp.float32)
        tables[KEY_TBPVS0] = cp.zeros(NX, dtype=cp.float32)
        tables[KEY_COEF] = cp.zeros(4, dtype=cp.float32)
    tbpvs = tables[KEY_TBPVS]
    tbpvs0 = tables[KEY_TBPVS0]
    coef = tables[KEY_COEF]

    k = kernels.get(UNIT, KERNEL)
    grid = ((NX + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1), (tbpvs, tbpvs0, coef))
    return tbpvs, tbpvs0, coef
