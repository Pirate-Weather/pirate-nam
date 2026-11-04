"""phys-fer-fpvs: FPVS, FPVS0, FPVSX, FPVSX0 (module_MP_FER_HIRES.F90:2840-2989).

Kernel: namgpu/kernels/phys_fer_fpvs.cu (FPVS and FPVS0 table lookups); FPVSX and FPVSX0 come from
namgpu/kernels/fer_fpvs_common.cuh, shared with phys-fer-gpvs. The production callers (phys-fer-condense,
phys-fer-deposit, the column kernels) include phys_fer_fpvs.cu and call nam_fpvs / nam_fpvs0 / nam_fpvsx /
nam_fpvsx0 directly with the GPVS tables. This wrapper runs the evaluation kernel fer_fpvs_eval once over an
array of temperatures (one launch, one thread per element).

Inputs (ctx):
  T                    : ctx.st["T"] by default (float32 [LM, JM, IM], Kelvin, C-contiguous) or the explicit argument.
  ctx.tables["fer_gpvs_tbpvs"], ctx.tables["fer_gpvs_tbpvs0"] : float32[NX=7501] from GPVS_hr (phys-fer-gpvs);
                                       built by phys-fer-gpvs.run() if absent.
  ctx.tables["fer_gpvs_coef"]         : float32[4] = (c1xpvs, c2xpvs, c1xpvs0, c2xpvs0) from GPVS_hr; read on device.
Outputs (scratch in ctx.st, CONTRACT section 1, allocated once by st.add and reused):
  ctx.st["fer_fpvs"], ctx.st["fer_fpvs0"], ctx.st["fer_fpvsx"], ctx.st["fer_fpvsx0"]: float32, shape of T,
  holding FPVS, FPVS0, FPVSX and FPVSX0 of T.
No host sync inside the wrapper; the coefficients are read from the device array fer_gpvs_coef.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_fpvs"
KERNEL = "fer_fpvs_eval"
BLOCK = 256
NX = 7501
OUT_NAMES = ("fer_fpvs", "fer_fpvs0", "fer_fpvsx", "fer_fpvsx0")


def run(ctx, T=None):
    if T is None:
        T = ctx.st["T"]
    from namgpu.physics.fer_hires import phys_fer_gpvs

    if T.dtype.name != "float32":
        raise ValueError(f"phys-fer-fpvs: T must be float32, got {T.dtype}")
    if not T.flags.c_contiguous:
        # the kernel reads T as a flat C-contiguous buffer (n = T.size); a strided view would be read wrongly
        raise ValueError("phys-fer-fpvs: T must be C-contiguous (pass cp.ascontiguousarray(T))")

    tables = ctx.tables
    if "fer_gpvs_tbpvs" not in tables:
        phys_fer_gpvs.run(ctx)
    tbpvs = tables["fer_gpvs_tbpvs"]
    tbpvs0 = tables["fer_gpvs_tbpvs0"]
    coef = tables["fer_gpvs_coef"]
    assert tbpvs.shape == (NX,) and tbpvs0.shape == (NX,) and coef.shape == (4,)

    shape = tuple(T.shape)
    out = tuple(ctx.st.add(name, shape) for name in OUT_NAMES)
    for name, a in zip(OUT_NAMES, out):
        if a.shape != shape:
            raise ValueError(f"phys-fer-fpvs: ctx.st[{name!r}] has shape {a.shape}, T has {shape}")

    n = int(T.size)
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(((n + BLOCK - 1) // BLOCK, 1, 1), (BLOCK, 1, 1),
          (T, tbpvs, tbpvs0, coef, out[0], out[1], out[2], out[3], np.int32(n)))
    return out
