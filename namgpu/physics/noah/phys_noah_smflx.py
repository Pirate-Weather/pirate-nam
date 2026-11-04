"""phys-noah-smflx: SMFLX, soil moisture flux (Noah LSM), with SRT, SSTEP, ROSR12 and WDFCND.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3799-3916 (SMFLX) and its
callees SRT 4603-4852, SSTEP 4855-4942, ROSR12 3666-3725, WDFCND 5312-5369. NOAA US government work;
the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_smflx.cu (phys_noah_smflx, one thread per column, grid_cols, one launch).
Started from the ArWen noah_smflx family (Apache-2.0) and brought to the NAM Fortran; the differences
are listed in the kernel header.

SMFLX updates SMC, SH2O and CMC in place and returns DRIP and RUNOFF1..3.

Callers and inputs (Fortran SFLX calls SMFLX at module_LS_NOAHLSM.F90:3271, 3297 and 4329):
  * DT: the call passes DTBL = NPHS * DT (module_TURBULENCE.F90:764). The ctx path takes it from
    ctx.cfg.nphs x ctx.dt, the same value phys-noah-driver uses (12 km CONUS 25 x 2 = 50 s, the
    conusnest 6 x 2 = 12 s, firewxnest 2 x 4 = 8 s). There is no silent 50 s default on the ctx path.
  * Soil and vegetation parameters (BEXP, DKSAT, DWSAT, SMCMAX, SMCWLT, KDT, SLOPE, CMCMAX) are the
    outputs of phys-noah-redprm (REDPRM, 3471-3664) in ctx.st. FRZFACT is not a REDPRM output: SFLX
    computes it at line 3611 as (SMCMAX / SMCREF) * (0.412 / 0.468), and the wrapper does the same.
    SHDFAC is the state field SFLX passes in.
  * Per-step fluxes PRCP1, EDIR, EC and ET are computed inside SFLX (2210-2632, 3246-3338,
    4114-4173). They are not state fields yet, so the ctx path takes them as keyword arguments and
    refuses (KeyError naming SFLX) when the caller does not pass them. Nothing is zero-filled.

run(ctx): the done-rule entry point. Without ctx it builds a full-size 954 x 835 random batch that keeps
the physical invariants (SH2O <= SMC, positive precipitation, SMCWLT < SMCMAX) so both the one-pass and
the two-pass branch run. No host sync, no per-step allocation (outputs and FRZFACT are cached once per
grid shape), everything on the current stream.

Recorded NAM quirks kept as written (not defects of the port):
  * SRT caps INFMAX with MIN(INFMAX, PX), where INFMAX is a rate and PX an amount (module_LS_NOAHLSM.F90
    SRT). The kernel keeps it.
  * SRT and SSTEP use Fortran MIN/MAX; the kernel uses fminf/fmaxf, which differ only in NaN propagation.
    The ctx path now refuses to run without its real inputs, so no NaN reaches the kernel from it.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_smflx"
KERNEL = "phys_noah_smflx"
BLOCK = kernels.BLOCK_COL
NSOIL = 4
DT_TEST = 50.0           # standalone random batch only (NAM 12 km DTBL); the ctx path never uses it

ZSOIL_NAM = (-0.1, -0.4, -1.0, -2.0)
# FRZFACT = (SMCMAX / SMCREF) * (0.412 / 0.468), module_LS_NOAHLSM.F90:3611 (REAL arithmetic)
FRZ_CONST = np.float32(np.float32(0.412) / np.float32(0.468))

_scratch: dict = {}

_REDPRM_FIELDS = ("BEXP", "DKSAT", "DWSAT", "SMCMAX", "SMCWLT", "SMCREF", "KDT", "SLOPE", "CMCMAX")
_SFLX_KW = ("PRCP1", "EDIR", "EC", "ET")
_OUTPUT_NAMES = ("DRIP", "RUNOFF1", "RUNOFF2", "RUNOFF3")
_PRODUCERS_REDPRM = "phys-noah-redprm (REDPRM, module_LS_NOAHLSM.F90:3471-3664) writes these into ctx.st"
_PRODUCERS_SFLX = ("SFLX (phys-noah-sflx; Fortran 2210-2632, 3246-3338, 4114-4173) computes PRCP1, EDIR, "
                   "EC and ET each step and must pass them as keyword arguments")


def smflx(smc, sh2o, cmc, zsoil, et, prcp1, edir, ec, shdfac, slope, kdt, frzfact,
          smcmax, bexp, smcwlt, dksat, dwsat, cmcmax, drip, runoff1, runoff2, runoff3,
          dt):
    """SMC, SH2O, CMC updated in place; DRIP, RUNOFF1..3 written.

    smc, sh2o, et: float32 [NSOIL, JM, IM]; cmc and every column array: float32 [JM, IM];
    zsoil: float32 [NSOIL].
    """
    nsoil, jm, im = (int(s) for s in smc.shape)
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_cols(im, jm), BLOCK,
      (smc, sh2o, cmc, zsoil, et, prcp1, edir, ec, shdfac, slope, kdt, frzfact,
       smcmax, bexp, smcwlt, dksat, dwsat, cmcmax, drip, runoff1, runoff2, runoff3,
       np.float32(dt), np.int32(im), np.int32(jm)))
    return smc, sh2o, cmc


def _cache(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        d = {n: cp.zeros((jm, im), dtype=cp.float32) for n in _OUTPUT_NAMES}
        d["FRZFACT"] = cp.zeros((jm, im), dtype=cp.float32)
        _scratch[key] = d
    return _scratch[key]


def _random_batch(jm: int, im: int, seed: int = 1):
    import cupy as cp

    rng = np.random.default_rng(seed)
    f = lambda lo, hi, shape: cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))  # noqa: E731
    col = (jm, im)
    soil = (NSOIL, jm, im)
    smcmax = f(0.40, 0.50, col)
    smc = f(0.10, 0.40, soil)
    sh2o = smc * f(0.0, 0.9, soil)                 # SH2O <= SMC (SICE >= 0)
    d = {
        "SMC": smc, "SH2O": sh2o,
        "CMC": f(0.0, 0.2e-3, col),
        "ET": f(0.0, 2.0e-6, soil),
        "PRCP1": f(0.0, 1.0e-4, col),
        "EDIR": f(0.0, 1.0e-6, col),
        "EC": f(0.0, 1.0e-6, col),
        "SHDFAC": f(0.0, 0.9, col),
        "SLOPE": f(0.0, 1.0, col),
        "KDT": f(1.0e-4, 5.0e-4, col),
        "FRZFACT": f(0.1, 0.5, col),
        "SMCMAX": smcmax,
        "BEXP": f(2.0, 12.0, col),
        "SMCWLT": smcmax * f(0.05, 0.3, col),
        "DKSAT": f(1.0e-6, 1.0e-4, col),
        "DWSAT": f(1.0e-6, 1.0e-4, col),
        "CMCMAX": f(0.5e-3, 1.0e-3, col),
    }
    return d


def _ctx_dt(ctx) -> float:
    cfg = getattr(ctx, "cfg", None)
    nphs = getattr(cfg, "nphs", None) if cfg is not None else None
    dt = getattr(ctx, "dt", None)
    if nphs is None or dt is None:
        raise KeyError("phys-noah-smflx needs ctx.cfg.nphs and ctx.dt to form DTBL = NPHS * DT "
                       "(module_TURBULENCE.F90:764); the 50 s default is not used on the ctx path")
    return float(int(nphs)) * float(dt)


def _ctx_inputs(ctx, jm: int, im: int, kw: dict):
    """Per-point inputs for the ctx path: REDPRM parameters and SHDFAC from ctx.st, FRZFACT derived,
    per-step fluxes from keyword arguments. Raises KeyError naming the producer when one is missing."""
    import cupy as cp

    st = ctx.st
    missing = [n for n in _REDPRM_FIELDS + ("SHDFAC",) if n not in st.fields]
    if missing:
        raise KeyError(f"phys-noah-smflx needs {missing} in the state; {_PRODUCERS_REDPRM}; "
                       "SHDFAC comes from SFLX/init")
    absent = [n for n in _SFLX_KW if kw.get(n) is None]
    if absent:
        raise KeyError(f"phys-noah-smflx needs {absent} from its caller; {_PRODUCERS_SFLX}")
    for n in _REDPRM_FIELDS + ("SHDFAC",):
        if tuple(st.fields[n].shape) != (jm, im):
            raise ValueError(f"phys-noah-smflx: state {n} has shape {tuple(st.fields[n].shape)}, "
                             f"expected {(jm, im)}")
    sc = _cache(jm, im)
    fz = sc["FRZFACT"]
    cp.divide(st.fields["SMCMAX"], st.fields["SMCREF"], out=fz)
    cp.multiply(fz, FRZ_CONST, out=fz)
    inputs = {
        "PRCP1": kw["PRCP1"], "EDIR": kw["EDIR"], "EC": kw["EC"], "SHDFAC": st.fields["SHDFAC"],
        "SLOPE": st.fields["SLOPE"], "KDT": st.fields["KDT"], "FRZFACT": fz,
        "SMCMAX": st.fields["SMCMAX"], "BEXP": st.fields["BEXP"], "SMCWLT": st.fields["SMCWLT"],
        "DKSAT": st.fields["DKSAT"], "DWSAT": st.fields["DWSAT"], "CMCMAX": st.fields["CMCMAX"],
    }
    return inputs, kw["ET"]


def run(ctx=None, *, dt=None, **arrays):
    """Done-rule entry. Returns the arrays it wrote.

    With ctx: SMC, SH2O, CMC and ZSOIL come from ctx.st; the REDPRM parameters and SHDFAC come from
    ctx.st; PRCP1, EDIR, EC and ET are required keyword arguments; dt = ctx.cfg.nphs * ctx.dt.
    Without ctx: the standalone random batch (954 x 835, dt 50 s); keyword arrays override any input.
    """
    import cupy as cp

    if ctx is not None and "SMC" not in arrays:
        st = ctx.st
        smc, sh2o, cmc = st.SMC, st.SH2O, st.CMC
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        if "ZSOIL" in st.fields:
            zsoil = st.fields["ZSOIL"]
        else:
            zsoil = st.add("ZSOIL", (NSOIL,))
            zsoil[...] = cp.asarray(np.array(ZSOIL_NAM, dtype=np.float32))
        inputs, et = _ctx_inputs(ctx, jm, im, {k: arrays.get(k) for k in _SFLX_KW})
        if dt is None:
            dt = _ctx_dt(ctx)
        sc = _cache(jm, im)
    else:
        if dt is None:
            dt = DT_TEST
        if "SMC" in arrays:
            smc, sh2o, cmc = arrays["SMC"], arrays["SH2O"], arrays["CMC"]
            jm, im = int(smc.shape[1]), int(smc.shape[2])
            zsoil = arrays.get("ZSOIL")
            if zsoil is None:
                zsoil = cp.asarray(np.array(ZSOIL_NAM, dtype=np.float32))
            et = arrays["ET"]
            inputs = {n: arrays[n] for n in ("PRCP1", "EDIR", "EC", "SHDFAC", "SLOPE", "KDT",
                                             "FRZFACT", "SMCMAX", "BEXP", "SMCWLT", "DKSAT",
                                             "DWSAT", "CMCMAX")}
        else:
            jm, im = 835, 954
            b = _random_batch(jm, im)
            smc, sh2o, cmc, et = b["SMC"], b["SH2O"], b["CMC"], b["ET"]
            zsoil = cp.asarray(np.array(ZSOIL_NAM, dtype=np.float32))
            inputs = {n: b[n] for n in ("PRCP1", "EDIR", "EC", "SHDFAC", "SLOPE", "KDT",
                                        "FRZFACT", "SMCMAX", "BEXP", "SMCWLT", "DKSAT",
                                        "DWSAT", "CMCMAX")}
        sc = _cache(jm, im)
    outs = {n: sc[n] for n in _OUTPUT_NAMES}
    smflx(smc, sh2o, cmc, zsoil, et,
          inputs["PRCP1"], inputs["EDIR"], inputs["EC"], inputs["SHDFAC"], inputs["SLOPE"],
          inputs["KDT"], inputs["FRZFACT"], inputs["SMCMAX"], inputs["BEXP"], inputs["SMCWLT"],
          inputs["DKSAT"], inputs["DWSAT"], inputs["CMCMAX"],
          outs["DRIP"], outs["RUNOFF1"], outs["RUNOFF2"], outs["RUNOFF3"], dt=dt)
    return {"SMC": smc, "SH2O": sh2o, "CMC": cmc, **outs}
