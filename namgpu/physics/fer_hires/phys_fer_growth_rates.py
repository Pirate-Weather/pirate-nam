"""phys-fer-growth-rates: MY_GROWTH_RATES_NMM_hr, the ice growth table scaled by the step DTPH.

Ported from phys/module_MP_FER_HIRES.F90:2746-2785 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_fer_growth_rates.cu (fer_growth_rates), one launch, 35 threads.
Caller: FERRIER_INIT_hr (module_MP_FER_HIRES.F90:2589) with DTPH = GSMDT (line 2505), and
GSMDT = DT_MICRO = int_state%NPRECIP*DT (module_SOLVER_GRID_COMP.F90:10974/10993, passed at 10997).

Writes ctx.tables["fer_growth"]: float32[35], Fortran MY_GROWTH_NMM(1:35) (MY_T1=1, MY_T2=35).
The consumer is phys-fer-init (namgpu/physics/fer_hires/phys_fer_init.py): it calls run() here and
binds the returned array as fer["growth"] for phys_fer_col_processes (Fortran PINIT, 1436-1439) and
for the MP_RESTART_STATE build (phys_fer_init_scalars). This is the only growth kernel in the port.

Reads: dtph = GSMDT in seconds. Default: float32(cfg.nprecip) * float32(cfg.dt), the same expression
phys_fer_init.run uses for DT_MICRO, so the default tracks the run's NPRECIP and DT (50 s at NPRECIP=2,
DT=25 s).

Restart: Fortran restores MY_GROWTH_NMM from MP_RESTART_STATE(1:35) (module line 281) and writes it back
(589, 2720). This unit always recomputes the table from dtph; the restart words are carried by
phys-fer-init (mp_restart built from growth in phys_fer_init_scalars) and the restart writer, not here.
Numerically the same whenever dtph matches the run that wrote the restart.

Launch: one kernel on ctx.stream, no host sync; the scalar is passed by value.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_growth_rates"
KERNEL = "fer_growth_rates"
KEY = "fer_growth"
N = 35
BLOCK = 64


def default_dtph(cfg) -> np.float32:
    """DTPH = GSMDT = NPRECIP * DT in single precision (module_SOLVER_GRID_COMP.F90:10974/10993)."""
    return np.float32(np.float32(cfg.nprecip) * np.float32(cfg.dt))


def run(ctx, dtph: float | None = None):
    import cupy as cp

    if dtph is None:
        dtph = default_dtph(ctx.cfg)
    if KEY not in ctx.tables:
        ctx.tables[KEY] = cp.zeros((N,), dtype=cp.float32)
    out = ctx.tables[KEY]
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(((N + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1), (out, np.float32(dtph)))
    return out
