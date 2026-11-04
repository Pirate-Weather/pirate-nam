"""drv-tablet: TABLET temperature look-up table (init only, built once).

Ported from nmm/module_CONTROL.F90:1970-2068 (NAM nam.v4.2.11, 83df4e13c1).
Kernel:  namgpu/kernels/drv_tablet.cu (tablet_build; one thread per pressure row, knots only).
Spline:  drv-spline's batched kernel (namgpu/steps/drv_spline.py spline_batch), one spline per
         row, nold = nnew = JTB, the same CALL SPLINE that TABLET makes for every kp.
Reads:   ctx.cfg.pt (model top pressure, Pa); pl = max(pt, 1000.) as the consts routine sets it.
Writes (ctx.tables; these are the TABLET set of module_CONTROL, not the BMJ set):
    tablet_ttbl   [ITB, JTB] float32   Fortran ttbl(jtb, itb): CuPy [kp, kth]
    tablet_the0   [ITB]      float32   Fortran the0(itb)
    tablet_sthe   [ITB]      float32   Fortran sthe(itb)
    tablet_rdp    [1]        float32   Fortran module rdp  = 1/dp
    tablet_rdthe  [1]        float32   Fortran module rdthe = 1/dthe
    tablet_pl     [1]        float32   pl used to build the table (max(pt, 1000))

Consumers: none in the NMM Fortran. module_CONTROL's ttbl/the0/sthe/rdp/rdthe are referenced only
inside module_CONTROL (TABLET itself). module_CU_BMJ.F90 declares its own ttbl/the0/sthe and its own
pl/rdp/rdthe parameters (module_CU_BMJ.F90:62,100-102) built by bmj_init, and the port reads those
via phys-bmj-init. The two sets have different shapes (201x601 here, 76x134 in BMJ) and never share
state, so they cannot diverge.

Scratch: work [4, ITB, JTB] float32 (knots) plus p, q [ITB, JTB] (spline scratch), released after.
Launch:  one build kernel and one spline batch on ctx.stream, no host sync. Built once; later calls
         return at once.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps import drv_spline

UNIT = "drv_tablet"
ITB = 201
JTB = 601


def run(ctx):
    if "tablet_ttbl" in ctx.tables:
        return
    import cupy as cp

    pt = float(ctx.cfg.pt)
    pl = np.float32(max(pt, 1000.0))

    f32 = cp.float32
    ttbl = cp.zeros((ITB, JTB), dtype=f32)
    the0 = cp.zeros((ITB,), dtype=f32)
    sthe = cp.zeros((ITB,), dtype=f32)
    scal = cp.zeros((2,), dtype=f32)
    work = cp.zeros((4, ITB, JTB), dtype=f32)
    p = cp.zeros((ITB, JTB), dtype=f32)
    q = cp.zeros((ITB, JTB), dtype=f32)

    k_build = kernels.get(UNIT, "tablet_build")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_build((1,), (256,), (work, the0, sthe, scal, pl))
        drv_spline.spline_batch(work[0], work[1], work[2], work[3], ynew=ttbl, p=p, q=q)

    ctx.tables["tablet_ttbl"] = ttbl
    ctx.tables["tablet_the0"] = the0
    ctx.tables["tablet_sthe"] = sthe
    ctx.tables["tablet_rdp"] = scal[0:1]
    ctx.tables["tablet_rdthe"] = scal[1:2]
    ctx.tables["tablet_pl"] = cp.asarray([pl], dtype=f32)
