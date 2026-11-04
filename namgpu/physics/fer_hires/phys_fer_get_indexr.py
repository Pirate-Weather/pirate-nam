"""phys-fer-get-indexr: GET_INDEXR, the rain-drop mean diameter index (module_MP_FER_HIRES.F90:2297-2380).

Kernel: namgpu/kernels/phys_fer_get_indexr.cu (fer_get_indexr). One launch, one thread per element,
no host sync, no per-step allocation after the first call.

Inputs:
  RR   : float32 device array of rain rates (kg m-2 s-1), any shape (the Fortran scalar argument,
         applied elementwise). Default: ctx.st["RR"] if the caller has put it in the state.
  thr  : float32[7] device array ctx.tables["fer_rr_thresh"] = (RR_DRmin, RR_DR1, RR_DR2, RR_DR3,
         RR_DR4, RR_DR5, RR_DRmax), computed by FER_HIRES_INIT from N0r0*RRATE(MDR*), set by the
         phys-fer-init unit. Passed explicitly with thresh= if the caller has them.
Output: int32 array of the same shape (microns), ctx.tables["fer_indexr_out"], allocated once on
first call or when the shape changes.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_get_indexr"
KERNEL = "fer_get_indexr"
BLOCK = 256


def run(ctx, RR=None, thresh=None, out=None):
    import cupy as cp

    if RR is None:
        RR = ctx.st["RR"]
    if thresh is None:
        thresh = ctx.tables["fer_rr_thresh"]
    # Guards (review of phys-fer-get-indexr): the kernel reads RR and thr as flat float32 buffers and
    # writes a flat int32 buffer. A float64, non-contiguous or mismatched buffer would be read from the
    # wrong memory without an error, so each is refused here.
    if thresh.shape != (7,) or thresh.dtype != cp.float32:
        raise TypeError(f"thresh must be float32[7], got {thresh.dtype}{tuple(thresh.shape)}")
    if RR.dtype != cp.float32:
        raise TypeError(f"RR must be float32 (kernel reads float32), got {RR.dtype}")
    if not RR.flags.c_contiguous:
        raise ValueError("RR must be C-contiguous (the kernel indexes RR as a flat buffer)")
    shape = tuple(RR.shape)
    if out is None:
        out = ctx.tables.get("fer_indexr_out")
        if out is None or out.shape != shape:
            out = cp.empty(shape, dtype=cp.int32)
            ctx.tables["fer_indexr_out"] = out
    elif out.shape != shape or out.dtype != cp.int32 or not out.flags.c_contiguous:
        raise ValueError(f"out must be C-contiguous int32 with shape {shape}")
    n = int(RR.size)
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(((n + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1),
          (RR, thresh, out, np.int32(n)))
    return out
