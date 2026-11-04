"""phys-noah-penman: PENMAN, potential evaporation (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3390-3468. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file.
Start point: ArWen namgpu/third_party/arwen/core/noah.py and kernels/noah.cu (Apache-2.0, notice kept
in the kernel header). Kernel: namgpu/kernels/phys_noah_penman.cu (phys_noah_penman, one
thread per (i, j), one launch). No host sync, no per-step allocation, current stream.

Outputs (CONTRACT names): T24, ETP, RCH, RR, EPSCA, FLX2, all float32 [JM, IM].
Inputs: SFCTMP, SFCPRS, CH, T2V, TH2, PRCP, FDOWN, SSOIL, Q2, Q2SAT, DQSDT2, SNOWNG (int32
0/1), FRZGRA (int32 0/1), EMISSI_IN, SNEQV; all float32 [JM, IM] except the two logicals.

PENMAN's arguments are local to SFLX/SNOPAC (not state-table fields), so with ctx they come
from scratch arrays allocated once. Without ctx it builds a full-size 954 x 835 plausible batch.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_penman"
KERNEL = "phys_noah_penman"
BLOCK = (32, 8, 1)

FLOAT_INPUTS = ("SFCTMP", "SFCPRS", "CH", "T2V", "TH2", "PRCP", "FDOWN", "SSOIL", "Q2",
                "Q2SAT", "DQSDT2", "EMISSI_IN", "SNEQV")
INT_INPUTS = ("SNOWNG", "FRZGRA")
OUTPUTS = ("T24", "ETP", "RCH", "RR", "EPSCA", "FLX2")

_scratch: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def penman(args: dict, out: dict):
    """Launch PENMAN on every column. args/out map CONTRACT names to device arrays [JM, IM]."""
    jm, im = (int(s) for s in args["SFCTMP"].shape)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK, (
        args["SFCTMP"], args["SFCPRS"], args["CH"], args["T2V"], args["TH2"], args["PRCP"],
        args["FDOWN"], args["SSOIL"], args["Q2"], args["Q2SAT"], args["DQSDT2"],
        args["SNOWNG"], args["FRZGRA"], args["EMISSI_IN"], args["SNEQV"],
        out["T24"], out["ETP"], out["RCH"], out["RR"], out["EPSCA"], out["FLX2"],
        np.int32(im), np.int32(jm)))
    return out


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        ins = {n: cp.zeros((jm, im), dtype=cp.float32) for n in FLOAT_INPUTS}
        ins.update({n: cp.zeros((jm, im), dtype=cp.int32) for n in INT_INPUTS})
        outs = {n: cp.zeros((jm, im), dtype=cp.float32) for n in OUTPUTS}
        _scratch[key] = (ins, outs)
    return _scratch[key]


def _plausible_batch(jm: int, im: int, seed: int = 1):
    """Random but physically plausible full-size batch (no column test, no reduced size)."""
    import cupy as cp

    rng = np.random.default_rng(seed)

    def u(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(jm, im)).astype(np.float32))

    ins = {
        "SFCTMP": u(250.0, 310.0), "SFCPRS": u(70000.0, 102000.0), "CH": u(0.001, 0.02),
        "T2V": u(250.0, 315.0), "TH2": u(250.0, 315.0), "PRCP": u(0.0, 0.002),
        "FDOWN": u(100.0, 450.0), "SSOIL": u(-50.0, 50.0), "Q2": u(0.001, 0.02),
        "Q2SAT": u(0.001, 0.025), "DQSDT2": u(1.0e-5, 1.0e-3), "EMISSI_IN": u(0.9, 0.99),
        "SNEQV": u(0.0, 0.5),
    }
    ins["SNEQV"] = cp.where(ins["SNEQV"] < 0.1, 0.0, ins["SNEQV"]).astype(cp.float32)
    ins["SNOWNG"] = cp.asarray((rng.uniform(size=(jm, im)) < 0.2).astype(np.int32))
    ins["FRZGRA"] = cp.asarray((rng.uniform(size=(jm, im)) < 0.05).astype(np.int32))
    return ins


def run(ctx=None, *, args=None, out=None):
    """Done-rule entry: one launch on a full-size 954 x 835 field. Returns the arrays it wrote."""
    import cupy as cp

    if args is None:
        if ctx is not None:
            jm, im = int(ctx.grid.jm), int(ctx.grid.im)
            args, out = _scratch_arrays(jm, im)
            if not _scratch.get(("seeded", jm, im)):
                args.update(_plausible_batch(jm, im))
                _scratch[("seeded", jm, im)] = True
        else:
            jm, im = 835, 954
            args = _plausible_batch(jm, im)
            out = {n: cp.zeros((jm, im), dtype=cp.float32) for n in OUTPUTS}
    if out is None:
        jm, im = (int(s) for s in args["SFCTMP"].shape)
        out = {n: cp.zeros((jm, im), dtype=cp.float32) for n in OUTPUTS}
    penman(args, out)
    return dict(out)
