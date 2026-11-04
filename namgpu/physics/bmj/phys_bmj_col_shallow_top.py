"""phys-bmj-col-shallow-top: shallow cloud top search and shallow-point gather for the BMJ scheme.

Ported from phys/module_CU_BMJ.F90:1317-1449 (NAM nam.v4.2.11, 83df4e13c1), the failed-deep branch of
subroutine bmj (the else of the deep test at line 1223), through the gather return at 1444. The per-column
state the Fortran sets around that branch is written too: the goto-600 shallow-only columns (953-955), and
the deep-path columns (555, 953, 1225-1226).
Kernel: namgpu/kernels/phys_bmj_col_shallow_top.cu. The device body is the __device__ function
``bmj_col_shallow_top``, which the column scheme (phys-bmj-col-*) inlines per column; the extern "C"
entry ``phys_bmj_col_shallow_top_columns`` runs it over a whole [JM, IM] field, one thread per column.

Inputs (all device arrays; NMMB names in parentheses):
  psfc [JM, IM], sm [JM, IM] (SM), deep_ok [JM, IM] int32 (0/1, the 1223 test, meaningful where bypass=0),
  bypass [JM, IM] int32 (0/1, depth < depmin: the goto-600 shallow-only columns, 953-955),
  lbot [JM, IM] int32 (LBOT), pk [LM, JM, IM] (PK, from phys-bmj-col-deep-ref), dtv [LM, JM, IM] (DTV, from
  phys-bmj-col-cloudtop), efi [JM, IM] (EFI, 1210-1213, read on deep columns only).
In/out: ltop [JM, IM] int32 (LTOP), cldefi [JM, IM] (CLDEFI), deep [JM, IM] int32 (DEEP),
  shallow [JM, IM] int32 (SHALLOW, the gather mask read by phys-bmj-col-shallow-adjust).
Out: iswap [JM, IM] int32 (ISWAP), ptop [JM, IM] (PTOP).
Scalars: lm, lmh (LMH = LM, module_CU_BMJ.F90:303), newswap (configfile), psh = 20000., pshu = 45000.,
  rsfcp = 1/101300., efimn (0.20, module_CU_BMJ.F90:30), stefi (bmj_init values).

Caller contracts (not checked in the kernel): pk = prsmid copy (996), dtv post-restore (919-921), lbot
post-cloud-base (914). Refusals, additions to the Fortran: lbot < 1 leaves the column unshallow (the Fortran
would read pk(0)); ltop < 1 is clamped to 1, as cloudtop (885) does for its own output.

Launch: one kernel on ctx.stream, grid_cols, no host sync, no per-step allocation, no Python loop over cells.

run(ctx): done-rule entry. Builds a full-size random column state (954 x 835 x 60) and runs the kernel once,
with all three column paths present. Nothing is written to ctx.st; the outputs come back in a dict. No
bitwise comparison (rule 0).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_col_shallow_top"
KERNEL = "phys_bmj_col_shallow_top_columns"

# Module defaults, module_CU_BMJ.F90:30 (efimn), :39 (psh, pshu) and :103 (rsfcp). See nmmb_common.cuh.
EFIMN = np.float32(0.20)
PSH = np.float32(20000.0)
PSHU = np.float32(45000.0)
RSFCP = np.float32(1.0 / 101300.0)


def shallow_top_columns(IM, JM, LM, lmh, newswap, efimn, stefi, psfc, sm, deep_ok, bypass, lbot, ltop,
                        pk, dtv, efi, cldefi, deep, shallow, iswap, ptop, stream=None):
    """Run the shallow cloud top search over every column. All arrays are device arrays; no host sync."""
    grid, block = kernels.grid_cols(IM, JM), kernels.BLOCK_COL
    kern = kernels.get(UNIT, KERNEL)
    args = (
        np.int32(IM), np.int32(JM), np.int32(LM), np.int32(lmh), np.int32(1 if newswap else 0),
        PSH, PSHU, RSFCP, np.float32(efimn), np.float32(stefi),
        psfc, sm, deep_ok, bypass, lbot, ltop, pk, dtv, efi, cldefi, deep, shallow, iswap, ptop,
    )
    kern(grid, block, args, stream=stream)
    return ltop, cldefi, deep, shallow, iswap, ptop


def run(ctx=None, *, newswap=True, efimn=EFIMN, stefi=1.0, arrays=None):
    """Done-rule entry: one launch on a full-size random column state (954 x 835 x 60).

    With ``arrays`` given (a dict with the same keys as the inputs and outputs), runs on those arrays.
    Returns the dict of output arrays (ltop, cldefi, deep, shallow, iswap, ptop).
    """
    import cupy as cp

    if arrays is None:
        IM, JM, LM = (954, 835, 60) if ctx is None else (int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm))
        cp.random.seed(7)
        # pressure increasing with level index (l=1 model top); small perturbation keeps it monotone
        base = cp.linspace(1000.0, 100000.0, LM, dtype=cp.float32).reshape(LM, 1, 1)
        pk = (base * (1.0 + 0.002 * (cp.random.random((LM, JM, IM), dtype=cp.float32) - 0.5))).astype(cp.float32)
        dtv = cp.random.standard_normal((LM, JM, IM), dtype=cp.float32)
        psfc = cp.random.uniform(85000.0, 102000.0, (JM, IM)).astype(cp.float32)
        sm = cp.random.random((JM, IM), dtype=cp.float32)
        # the three column paths: shallow-only (bypass), deep (deep_ok on non-bypass), failed deep (rest)
        bypass = (cp.random.random((JM, IM)) < 0.25).astype(cp.int32)
        deep_ok = ((cp.random.random((JM, IM)) < 0.5) & (bypass == 0)).astype(cp.int32)
        efi = cp.random.uniform(float(efimn), 1.0, (JM, IM)).astype(cp.float32)
        lmh = LM
        lbot = cp.random.randint(2, lmh + 1, size=(JM, IM)).astype(cp.int32)
        ltop = cp.random.randint(1, LM + 1, size=(JM, IM)).astype(cp.int32)
        cldefi = cp.zeros((JM, IM), dtype=cp.float32)
        deep = cp.ones((JM, IM), dtype=cp.int32)
        shallow = cp.zeros((JM, IM), dtype=cp.int32)
        iswap = cp.zeros((JM, IM), dtype=cp.int32)
        ptop = cp.zeros((JM, IM), dtype=cp.float32)
        arrays = dict(psfc=psfc, sm=sm, deep_ok=deep_ok, bypass=bypass, lbot=lbot, ltop=ltop, pk=pk, dtv=dtv,
                      efi=efi, cldefi=cldefi, deep=deep, shallow=shallow, iswap=iswap, ptop=ptop)
    else:
        LM, JM, IM = (int(s) for s in arrays["pk"].shape)
        lmh = LM
    a = arrays
    stream = None if ctx is None else ctx.stream
    shallow_top_columns(IM, JM, LM, lmh, newswap, efimn, stefi, a["psfc"], a["sm"], a["deep_ok"],
                        a["bypass"], a["lbot"], a["ltop"], a["pk"], a["dtv"], a["efi"], a["cldefi"],
                        a["deep"], a["shallow"], a["iswap"], a["ptop"], stream=stream)
    return {k: a[k] for k in ("ltop", "cldefi", "deep", "shallow", "iswap", "ptop")}
