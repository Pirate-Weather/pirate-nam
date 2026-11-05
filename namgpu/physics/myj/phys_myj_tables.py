"""phys-myj-tables: HERFTBL, TABLEPT and TABLETT, the MYJ PBL look-up tables (module_BL_MYJPBL.F90:1661-2029, nam.v4.2.11, 83df4e13c1).

Called by the MYJ init path right after phys-myj-init (MYJPBL_INIT calls HERFTBL, TABLEPT, TABLETT).
Table build only: once per process at init, never on the step path. The SPLINE that TABLEPT and
TABLETT call is inlined in the kernel file as the __device__ function myj_spline.

Kernel: namgpu/kernels/phys_myj_tables.cu
  phys_myj_herftbl  one thread (sequential prefix sum over 301 values, as the Fortran)
  phys_myj_tablept  one thread per kth row (JTBL = 1201 threads), block 64
  phys_myj_tablett  one thread per kp column (ITBL = 401 threads), block 64
Three launches on ctx.stream. No host sync, no host math on the tables.

Outputs (CuPy float32 device arrays, names kept in ctx.tables when a ctx is given):
  myj_herff  [KERFM=301]         HERFF (half erf table; HERF is the __device__ myj_herf in the kernel file)
  myj_rdxe   [1]                 RDXE   = 1/DXE (ERF table scale)
  myj_ptbl   [JTBL=1201, ITBL=401]  PTBL(ITBL,JTBL): ptbl[kth, kp] (saturation pressure, Pa)
  myj_qs0    [JTBL=1201]         QS0 (base for saturation specific humidity)
  myj_sqs    [JTBL=1201]         SQS (range for saturation specific humidity)
  myj_rdth   [1]                 RDTH = 1/DTH (TABLEPT potential temperature scale)
  myj_rdq    [1]                 RDQ  = 1/DQS (TABLEPT humidity scale)
  myj_ttbl   [ITBL=401, JTBL=1201]  TTBL(JTBL,ITBL): ttbl[kp, kth] (temperature, K)
  myj_the0   [ITBL=401]          THE0 (base for equivalent potential temperature)
  myj_sthe   [ITBL=401]          STHE (range for equivalent potential temperature)
  myj_rdp    [1]                 RDP  = 1/DP (TABLETT pressure scale)
  myj_rdthe  [1]                 RDTHE = 1/DTHE (TABLETT equivalent potential temperature scale)

The scalar outputs are 1-element device views (no host readback). Layout follows CONTRACT section 1
(the Fortran PTBL(KP,KTH) is ptbl[kth, kp]; TTBL(KTH,KP) is ttbl[kp, kth]).

Done rule (CONTRACT section 6): run(None) builds the tables once and returns them. The unit has
no state input, so there is no random State; the check is that every output is finite.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_myj_tables"
BLOCK = 64

ITBL = 401
JTBL = 1201
KERFM = 301
TT_WORK = 7 * (JTBL + 1)     # floats of spline work per TABLETT column (kernel MYJ_TT_WORK)

_SCAL_RDXE = 0
_SCAL_RDTH = 1
_SCAL_RDQ = 2
_SCAL_RDP = 3
_SCAL_RDTHE = 4


def build_tables(stream=None) -> dict:
    """Allocate the MYJ tables on the device and fill them with the three kernels."""
    import cupy as cp

    herff = cp.zeros((KERFM,), dtype=cp.float32)
    qs0 = cp.zeros((JTBL,), dtype=cp.float32)
    sqs = cp.zeros((JTBL,), dtype=cp.float32)
    ptbl = cp.zeros((JTBL, ITBL), dtype=cp.float32)
    the0 = cp.zeros((ITBL,), dtype=cp.float32)
    sthe = cp.zeros((ITBL,), dtype=cp.float32)
    ttbl = cp.zeros((ITBL, JTBL), dtype=cp.float32)
    scal = cp.zeros((8,), dtype=cp.float32)

    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k_herf = kernels.get(UNIT, "phys_myj_herftbl")
        k_herf((1, 1, 1), (1, 1, 1), (herff, scal))

        k_pt = kernels.get(UNIT, "phys_myj_tablept")
        grid_pt = ((JTBL + BLOCK - 1) // BLOCK, 1, 1)
        k_pt(grid_pt, (BLOCK, 1, 1), (ptbl, qs0, sqs, scal))

        k_tt = kernels.get(UNIT, "phys_myj_tablett")
        grid_tt = ((ITBL + BLOCK - 1) // BLOCK, 1, 1)
        # per-column spline work arrays in global memory (13.5 MB), not 28.8 KB/thread of local stack that
        # the driver reserved for every resident thread of the card (~7.5 GB; kernel header)
        work = cp.empty((ITBL * TT_WORK,), dtype=cp.float32)
        k_tt(grid_tt, (BLOCK, 1, 1), (ttbl, the0, sthe, scal, work))

    return {
        "myj_herff": herff,
        "myj_rdxe": scal[_SCAL_RDXE:_SCAL_RDXE + 1],
        "myj_ptbl": ptbl,
        "myj_qs0": qs0,
        "myj_sqs": sqs,
        "myj_rdth": scal[_SCAL_RDTH:_SCAL_RDTH + 1],
        "myj_rdq": scal[_SCAL_RDQ:_SCAL_RDQ + 1],
        "myj_ttbl": ttbl,
        "myj_the0": the0,
        "myj_sthe": sthe,
        "myj_rdp": scal[_SCAL_RDP:_SCAL_RDP + 1],
        "myj_rdthe": scal[_SCAL_RDTHE:_SCAL_RDTHE + 1],
    }


def run(ctx=None) -> dict:
    """Build the MYJ tables. With a ctx, the tables are stored in ctx.tables; returns the dict."""
    stream = getattr(ctx, "stream", None) if ctx is not None else None
    tables = build_tables(stream=stream)
    if ctx is not None and getattr(ctx, "tables", None) is not None:
        ctx.tables.update(tables)
    return tables
