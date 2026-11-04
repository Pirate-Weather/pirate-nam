"""phys-noah-canres: CANRES, canopy resistance and plant coefficient (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2331-2469, called under the
caller guard IF (SHDFAC > 0 .AND. XLAI > 0) at lines 2154-2158. NOAA US government work; the
module's NOAA header is kept as the notice in the kernel file. Start point: ArWen core/kernels/
noah.cu (Apache-2.0, namgpu/third_party/arwen/LICENSE), CP changed to the NAM value.
Kernel: namgpu/kernels/phys_noah_canres.cu (phys_noah_canres, one thread per (i, j), one launch).

RC and PC (plus RCS, RCT, RCQ, RCSOIL, which the Fortran also assigns) per column. No host sync,
no per-step allocation, everything on the current stream.

Inputs (CONTRACT names where they exist): SOLAR (solar), CH, SFCTMP, Q2, SFCPRS, SH2O [NSOIL,JM,IM]
(state table), ZSOIL [NSOIL] (column independent), NROOT (int32 [JM,IM]), SHDFAC, SMCWLT, SMCREF,
RSMIN, RSMAX, RGL, HS, XLAI, TOPT, Q2SAT, DQSDT2, EMISSI.

Caller guard: the kernel reproduces IF (SHDFAC > 0 .AND. XLAI > 0) itself. Columns outside it keep
their previous RC, PC, RCS, RCT, RCQ and RCSOIL, exactly as the Fortran leaves them. No active-mask
input is needed from the caller.

NROOT outside 1..NSOIL: the kernel poisons that column's outputs with NaN rather than indexing out
of range. Clamps are NaN-propagating, so bad SH2O, SMCREF or RCS is visible in the output.

Parameters come only from ctx.st. With ctx, run() refuses loudly if any input is missing and names
its producer. Test fills are used only on the ctx-less path.

run(ctx): the done-rule entry point. Without ctx it builds a full-size 954 x 835 test batch and
checks that active columns are finite and inactive columns keep the sentinel.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_canres"
KERNEL = "phys_noah_canres"
NSOIL = 4
BLOCK = kernels.BLOCK_2D
SENTINEL = np.float32(-1.0)   # outputs before the call on the ctx-less test path

# 2-D float32 inputs read from ctx.st by name.
_F2D_INPUTS = ("SOLAR", "CH", "SFCTMP", "Q2", "SFCPRS", "SMCWLT", "SMCREF", "RSMIN", "RSMAX",
               "RGL", "HS", "XLAI", "TOPT", "Q2SAT", "DQSDT2", "EMISSI", "SHDFAC")
# Who produces each input when a ctx is given (named in the refusal).
_PRODUCER = {
    "SOLAR": "the radiation driver", "CH": "the surface layer (phys-mynn-sfc)",
    "SFCTMP": "the forcing state", "Q2": "the forcing state", "SFCPRS": "the forcing state",
    "SMCWLT": "phys-noah-parm-tables/init", "SMCREF": "phys-noah-parm-tables/init",
    "RSMIN": "phys-noah-redprm", "RSMAX": "phys-noah-redprm", "RGL": "phys-noah-redprm",
    "HS": "phys-noah-redprm", "XLAI": "the greenness state", "TOPT": "phys-noah-redprm",
    "Q2SAT": "the surface humidity state", "DQSDT2": "the surface humidity state",
    "EMISSI": "the radiation state", "SHDFAC": "phys-noah-redprm",
    "SH2O": "the Noah state table (phys-noah-init)",
    "ZSOIL": "phys-noah-redprm (SFLX layer setup)", "NROOT": "phys-noah-redprm",
}
_OUTPUTS = ("RC", "PC", "RCS", "RCT", "RCQ", "RCSOIL")

# Test-only plausible ranges, used only on the ctx-less path.
_RANGES = {
    "SOLAR": (0.0, 900.0), "CH": (1.0e-3, 1.0e-2), "SFCTMP": (260.0, 305.0),
    "Q2": (0.002, 0.02), "SFCPRS": (80000.0, 102000.0),
    "SMCWLT": (0.02, 0.10), "SMCREF": (0.15, 0.30), "RSMIN": (40.0, 200.0),
    "RSMAX": (500.0, 5000.0), "RGL": (30.0, 100.0), "HS": (36.0, 60.0),
    "XLAI": (0.5, 6.0), "TOPT": (293.0, 298.0), "Q2SAT": (0.004, 0.03),
    "DQSDT2": (5.0e-4, 3.0e-3), "EMISSI": (0.90, 0.98), "SHDFAC": (0.05, 0.95),
}


def _grid(im: int, jm: int):
    return kernels.grid_2d(im, jm, 1)


def canres(solar, ch, sfctmp, q2, sfcprs, sh2o, zsoil, nroot, smcwlt, smcref, rsmin, rsmax,
           rgl, hs, xlai, topt, q2sat, dqsdt2, emissi, shdfac, rc, pc, rcs, rct, rcq, rcsoil):
    """All 2-D arrays float32 [JM, IM] on device (nroot int32); sh2o float32 [NSOIL, JM, IM];
    zsoil float32 [NSOIL]. Outputs rc, pc, rcs, rct, rcq, rcsoil are written in place where
    SHDFAC > 0 and XLAI > 0 (and poisoned with NaN where NROOT is out of range)."""
    jm, im = int(smcwlt.shape[0]), int(smcwlt.shape[1])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (solar, ch, sfctmp, q2, sfcprs, sh2o, zsoil, nroot, smcwlt, smcref, rsmin, rsmax,
       rgl, hs, xlai, topt, q2sat, dqsdt2, emissi, shdfac, rc, pc, rcs, rct, rcq, rcsoil,
       np.int32(im), np.int32(jm)))
    return rc, pc


def _require(st, name):
    """ctx.st.<name>, or a refusal naming the field and its producer. Nothing is invented."""
    if name not in st.fields:
        raise RuntimeError(
            f"phys-noah-canres needs the state field {name} (produced by {_PRODUCER[name]}); "
            "it is not in ctx.st, so CANRES will not run on invented parameters")
    return st.fields[name]


def _ctx_inputs(ctx):
    st = ctx.st
    jm, im = int(st.fields["T"].shape[1]), int(st.fields["T"].shape[2])
    cols = {nm: _require(st, nm) for nm in _F2D_INPUTS}
    sh2o = _require(st, "SH2O")
    zsoil = _require(st, "ZSOIL")
    nroot = _require(st, "NROOT")
    if tuple(zsoil.shape) != (NSOIL,):
        raise RuntimeError(f"phys-noah-canres: ZSOIL has shape {tuple(zsoil.shape)}, need ({NSOIL},)")
    outs = {nm: st.add(nm, (jm, im)) for nm in _OUTPUTS}
    return cols, sh2o, zsoil, nroot, outs


def run(ctx=None):
    """Done-rule entry. ctx: all inputs from ctx.st, outputs added to it. Without ctx: a full-size
    954 x 835 test batch; active columns (SHDFAC > 0 and XLAI > 0) must be finite, inactive ones
    must keep the sentinel. Returns the output arrays."""
    import cupy as cp

    if ctx is not None:
        cols, sh2o, zsoil, nroot, outs = _ctx_inputs(ctx)
    else:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        cols = {nm: cp.asarray(rng.uniform(*_RANGES[nm], size=(jm, im)).astype(np.float32))
                for nm in _RANGES}
        # Exercise the guarded branch: about one column in five has no leaf area or no green
        # fraction. The Fortran leaves those columns alone, so they must keep SENTINEL.
        inactive = cp.asarray((rng.random((jm, im)) < 0.1) | (rng.random((jm, im)) < 0.1))
        cols["XLAI"] = cp.where(inactive, cp.float32(0.0), cols["XLAI"])
        cols["SHDFAC"] = cp.where(inactive, cp.float32(0.0), cols["SHDFAC"])
        cols["SMCREF"] = cp.maximum(cols["SMCREF"], cols["SMCWLT"] + 0.05)
        sh2o = cp.asarray(rng.uniform(0.02, 0.45, size=(NSOIL, jm, im)).astype(np.float32))
        zsoil = cp.asarray(np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32))
        nroot = cp.asarray(rng.integers(1, NSOIL + 1, size=(jm, im)).astype(np.int32))
        outs = {nm: cp.full((jm, im), SENTINEL, dtype=cp.float32) for nm in _OUTPUTS}
    canres(cols["SOLAR"], cols["CH"], cols["SFCTMP"], cols["Q2"], cols["SFCPRS"], sh2o, zsoil,
           nroot, cols["SMCWLT"], cols["SMCREF"], cols["RSMIN"], cols["RSMAX"], cols["RGL"],
           cols["HS"], cols["XLAI"], cols["TOPT"], cols["Q2SAT"], cols["DQSDT2"], cols["EMISSI"],
           cols["SHDFAC"], outs["RC"], outs["PC"], outs["RCS"], outs["RCT"], outs["RCQ"],
           outs["RCSOIL"])
    if ctx is None:
        active = (cols["SHDFAC"] > 0) & (cols["XLAI"] > 0)
        for nm in _OUTPUTS:
            a = outs[nm]
            if not bool(cp.all(cp.isfinite(cp.where(active, a, 0.0)))):
                raise AssertionError(f"phys-noah-canres: {nm} is not finite on active columns")
            if not bool(cp.all(cp.where(active, True, a == SENTINEL))):
                raise AssertionError(f"phys-noah-canres: {nm} changed on inactive columns")
    return outs
