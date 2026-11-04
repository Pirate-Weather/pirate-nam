"""phys-gwd-col-blocking: GWD_col part 1 (setup, Lott-Miller mountain blocking, mountain drag DB).

NOTICE: NOAA/NCEP NMMB source (nam.v4.2.11, 83df4e13c1), US government work. Ported from
nmm/module_GWD.F90:368-897. Kernel: namgpu/kernels/phys_gwd_col_blocking.cu.

Inputs are GWD_col arrays in GWD_col order (bottom-up: index 0 is the surface, Fortran K = 1), in
double precision (kind_phys = real*8). phys-gwd-driver builds them from the NAM layers (flip,
Pa -> centibars, geopotential). Shapes: [LM, JM, IM] for U1 V1 T1 Q1 DEL PRSL PRSLK PHIL;
[LM+1, JM, IM] for PRSI PHII; [4, JM, IM] for OA4 CLX4; [JM, IM] for HPRIME THETA SIGMA GAMMA
ELVMAX OC (float64) and KPBL (int32).

INDEX CONTRACT. The kernel reads and writes every field at the true column (j, i) (IDX3/IDX2 by
j0). The Fortran stores DB(I,K), RO(I,K), BNV2(I,K), RI_N(I,K) at the compact index I of the point
list (module_GWD.F90:607, 537, 832-835, 874). Here the same values sit at the column index, so
part 2 (phys-gwd-col-wave) must read them by column j, e.g. Fortran 126 RO(I,KP1), 170 DEL(ipt(I),K),
207-208 DB(I,K). Level order is bottom-up in every array, as the Fortran K.

OUTPUTS. ELEVMX, IDXZB, KREF, OA, CLX, UBAR, VBAR, ROLL, BNV2BAR, DELKS, DELKS1, DB, BNV2, RI_N, RO,
TAUP (zeroed, filled by part 2), KBPS, KMPS, DUsfc, DVsfc (zeroed), ZBK, WK (diagnostics), GUARD
(one int32 counter, must be 0), and OC (a pass-through of the input, carried for part 2). BNV2 and
RI_N are NaN at level LM by design (the Fortran never defines them; see the kernel).

BINDINGS OUTSIDE THIS UNIT. Part 2 takes OC (Fortran 971), CLEFF (969), FACTOP (1077) and RLOLEV
(1087) from its own wrapper (phys_gwd_col_wave.py). CLEFF, FACTOP and RLOLEV are declared but unused
in GWD_col 368-897, so this unit does not take them.

PRECISION. CDMB, SIGFAC, DPMIN are REAL*8 in the as-compiled NMM build (module_GWD.f90 is built with
-fdefault-real-8; see the kernel header), so they are float64 here.

GATING. The Fortran GWD_driver skips columns with kpbl<=1 or kpbl>LM and TERRtest < TERRmin before it
calls GWD_col. Those gates live in phys-gwd-driver (namgpu/kernels/phys_gwd_driver.cu:102-109). This
unit is called only through that driver and has no gate of its own.

run(ctx): takes the GWD inputs from ctx.gwd when the driver sets it, else builds a full-size random
plausible column set (954 x 835 x 60) for the done rule. One launch, no host sync.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_gwd_col_blocking"
KERNEL = "phys_gwd_col_blocking"
BLOCK = (128, 1, 1)
LMAX = 64

# nam_configfile gwd tunables (CONTRACT section 5): cdmb 0.1, sigfac 3.0, dpmin 0 (REAL*8 under -r8).
CDMB = np.float64(0.1)
SIGFAC = np.float64(3.0)
DPMIN = np.float64(0.0)

_scratch: dict = {}


def _grid(im: int, jm: int):
    return (((im * jm) + BLOCK[0] - 1) // BLOCK[0], 1, 1)


def _outputs(jm: int, im: int, lm: int):
    """Scratch outputs, allocated once per shape (device arrays, float64 as kind_phys)."""
    import cupy as cp

    key = (jm, im, lm)
    if key not in _scratch:
        f2 = lambda: cp.zeros((jm, im), dtype=cp.float64)
        f3 = lambda n: cp.zeros((n, jm, im), dtype=cp.float64)
        i2 = lambda: cp.zeros((jm, im), dtype=cp.int32)
        _scratch[key] = {
            "DUsfc": f2(), "DVsfc": f2(), "ELEVMX": f2(), "IDXZB": i2(), "KREF": i2(),
            "OA": f2(), "CLX": f2(), "UBAR": f2(), "VBAR": f2(), "ROLL": f2(),
            "BNV2BAR": f2(), "DELKS": f2(), "DELKS1": f2(), "ZBK": f2(), "WK": f2(),
            "DB": f3(lm), "BNV2": f3(lm), "RI_N": f3(lm), "RO": f3(lm), "TAUP": f3(lm + 1),
            "KBPS": cp.zeros((1,), dtype=cp.int32), "KMPS": cp.zeros((1,), dtype=cp.int32),
            "GUARD": cp.zeros((1,), dtype=cp.int32),
            "OC": f2(),
        }
    return _scratch[key]


def _random_inputs(jm: int, im: int, lm: int, seed: int = 1):
    """Full-size random-but-plausible GWD_col inputs (bottom-up), float64 on the device."""
    import cupy as cp

    if lm > LMAX:
        raise ValueError(f"LM={lm} exceeds GWD_LMAX={LMAX}")
    rs = cp.random.RandomState(seed)
    ps = 90.0 + 12.0 * rs.random_sample((jm, im))                       # surface, 1000 Pa units
    frac = cp.linspace(0.0, 1.0, lm + 1, dtype=cp.float64)[:, None, None]
    prsi = ps[None] * (0.002 ** frac)                                    # (LM+1, JM, IM), bottom-up
    prsl = 0.5 * (prsi[:-1] + prsi[1:])
    dele = prsi[:-1] - prsi[1:]
    prslk = (prsl / 100.0) ** (287.04 / 1004.6)
    dz = 50.0 + 350.0 * rs.random_sample((lm, jm, im))
    phii = cp.concatenate([cp.zeros((1, jm, im)), 9.806 * cp.cumsum(dz, axis=0)], axis=0)
    phil = 0.5 * (phii[:-1] + phii[1:])
    u1 = 30.0 * (rs.random_sample((lm, jm, im)) - 0.5)
    v1 = 30.0 * (rs.random_sample((lm, jm, im)) - 0.5)
    t1 = 220.0 + 80.0 * rs.random_sample((lm, jm, im))
    q1 = 1.0e-4 + 1.0e-2 * rs.random_sample((lm, jm, im))
    elvmax = 3000.0 * rs.random_sample((jm, im))
    hprime = 800.0 * rs.random_sample((jm, im))
    return {
        "U1": u1, "V1": v1, "T1": t1, "Q1": q1, "DEL": dele, "PRSL": prsl, "PRSLK": prslk,
        "PHIL": phil, "PRSI": prsi, "PHII": phii,
        "OA4": 2.0 * rs.random_sample((4, jm, im)) - 1.0,
        "CLX4": 1.0 + 9.0 * rs.random_sample((4, jm, im)),
        "HPRIME": hprime,
        "THETA": 180.0 * rs.random_sample((jm, im)),
        "SIGMA": 0.1 * rs.random_sample((jm, im)),
        "GAMMA": 0.2 + 0.8 * rs.random_sample((jm, im)),
        "ELVMAX": elvmax,
        "OC": 0.5 + 1.5 * rs.random_sample((jm, im)),
        "KPBL": (1 + (rs.random_sample((jm, im)) * (lm // 2)).astype(cp.int32)).astype(cp.int32),
    }


def gwd_col_blocking(inp: dict, out: dict, *, im: int, jm: int, lm: int,
                     cdmb=CDMB, sigfac=SIGFAC, dpmin=DPMIN):
    """One launch over every column. inp: GWD_col inputs (see module docstring). out: scratch dict."""
    if lm > LMAX:
        raise ValueError(f"LM={lm} exceeds GWD_LMAX={LMAX}")
    k = kernels.get(UNIT, KERNEL)
    # Fortran KBPS = 1 and KMPS = KM before the point loop (device fill, no host sync).
    out["KBPS"].fill(1)
    out["KMPS"].fill(lm)
    out["GUARD"].fill(0)
    args = (
        inp["U1"], inp["V1"], inp["T1"], inp["Q1"], inp["DEL"], inp["PRSL"], inp["PRSLK"],
        inp["PHIL"], inp["PRSI"], inp["PHII"], inp["OA4"], inp["CLX4"], inp["HPRIME"],
        inp["THETA"], inp["SIGMA"], inp["GAMMA"], inp["ELVMAX"], inp["KPBL"],
        out["DUsfc"], out["DVsfc"], out["ELEVMX"], out["IDXZB"], out["KREF"], out["OA"],
        out["CLX"], out["UBAR"], out["VBAR"], out["ROLL"], out["BNV2BAR"], out["DELKS"],
        out["DELKS1"], out["ZBK"], out["WK"], out["DB"], out["BNV2"], out["RI_N"], out["RO"],
        out["TAUP"], out["KBPS"], out["KMPS"], out["GUARD"],
        np.float64(cdmb), np.float64(sigfac), np.float64(dpmin),
        np.int32(im), np.int32(jm), np.int32(lm),
    )
    k(_grid(im, jm), BLOCK, args)
    return out


def run(ctx=None, *, inputs: dict | None = None, outputs: dict | None = None):
    """Entry point. Returns the output dict (scratch arrays written by the kernel)."""
    if ctx is not None and getattr(ctx, "st", None) is not None:
        lm, jm, im = (int(s) for s in ctx.st.T.shape)
    else:
        lm, jm, im = 60, 835, 954
    if inputs is None:
        inputs = getattr(ctx, "gwd", None) if ctx is not None else None
    if inputs is None:
        inputs = _random_inputs(jm, im, lm)
    if outputs is None:
        outputs = _outputs(jm, im, lm)
    gwd_col_blocking(inputs, outputs, im=im, jm=jm, lm=lm)
    # Several cards: KBPS/KMPS are domain max/min of kref (diagnostics; the per-column loop does not read them).
    # Integer allreduce, so every rank holds the one-card values (MULTICARD-PLAN.md section 5). One card: no-op.
    from namgpu import parallel as _par
    _comm = _par.get(ctx) if ctx is not None else None
    if _comm is not None:
        _comm.allreduce(outputs["KBPS"], outputs["KBPS"], "max", ctx.stream)
        _comm.allreduce(outputs["KMPS"], outputs["KMPS"], "min", ctx.stream)
    outputs["OC"][...] = inputs["OC"]   # pass-through for phys-gwd-col-wave (Fortran OC, line 971)
    return outputs
