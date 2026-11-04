"""phys-noah-evapo: EVAPO, total evapotranspiration (Noah LSM), with DEVAP and TRANSP.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2557-2641, with the callees
DEVAP (2514-2554) and TRANSP (5200-5309). NOAA US government work; the module's NOAA header is
kept as the notice in the kernel file. The CUDA structure is adapted from ArWen's noah.cu
(Apache-2.0, see the kernel header).

Kernel: namgpu/kernels/phys_noah_evapo.cu (phys_noah_evapo, one thread per (i, j), one launch).
This launch is the NOPAC call site (module_LS_NOAHLSM.F90:3264). The SNOPAC call site (4155) is
nam_evapo inside namgpu/kernels/phys_noah_snopac.cu, not this launch.

Outputs: ETA1 (total), EDIR (bare soil), EC (canopy), ETT (transpiration), ET [NSOIL, JM, IM],
and DEW [JM, IM] (the ETP <= 0 dew branch that NOPAC and SNOPAC take around EVAPO).

Inputs, each bound from ctx.st, the field its producer writes (a missing producer is a RuntimeError
that names it; there is no stand-in value on the ctx path):
    ETP [JM, IM]          potential evaporation, kg m-2 s-1 (phys-noah-penman). The kernel forms
                          ETP1 = ETP * 0.001 per point per step, as NOPAC does (3247).
    CMC [JM, IM], SH2O [NSOIL, JM, IM]           (NAM state; SH2O from phys-noah-sflx / phys-noah-init)
    RTDIS [NSOIL, JM, IM], NROOT [JM, IM] int32, SHDFAC, SMCMAX, SMCWLT, SMCREF, SMCDRY, CMCMAX,
    CFACTR, FXEXP [JM, IM]                       (phys-noah-redprm)
    PC [JM, IM]                                  (phys-noah-canres)
DT = NPHS * ctx.dt, with NPHS from ctx.cfg as phys-noah-driver derives it (default 2).

Outputs are unit-owned state fields (ETA1, EDIR, EC, ETT, ET, DEW), created once. No host sync and
no per-step allocation after the first call.

Without ctx, run() builds a full-size 954 x 835 synthetic batch for the done-rule check only.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_evapo"
KERNEL = "phys_noah_evapo"
BLOCK = (32, 8, 1)
NSOIL = 4
SYNTH_DT = 50.0          # synthetic-batch DT only (DTBL = NPHS x 25 s); the ctx path uses ctx.dt

# state field -> producer, for the ctx path.
REQUIRED = {
    "ETP": "phys-noah-penman (ETP, kg m-2 s-1)",
    "CMC": "NAM state (CMC)",
    "SH2O": "phys-noah-sflx / phys-noah-init",
    "RTDIS": "phys-noah-redprm",
    "NROOT": "phys-noah-redprm",
    "SHDFAC": "phys-noah-redprm",
    "SMCMAX": "phys-noah-redprm",
    "SMCWLT": "phys-noah-redprm",
    "SMCREF": "phys-noah-redprm",
    "SMCDRY": "phys-noah-redprm",
    "CMCMAX": "phys-noah-redprm",
    "CFACTR": "phys-noah-redprm",
    "FXEXP": "phys-noah-redprm",
    "PC": "phys-noah-canres",
}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def evapo(etp, cmc, sh2o, rtdis, shdfac, smcmax, smcwlt, smcref, smcdry, cmcmax, pc, cfactr,
          fxexp, nroot, dt, eta1, edir, ec, ett, et, dew, *, im: int | None = None,
          jm: int | None = None):
    """One launch. etp, cmc, shdfac.. [JM, IM] (nroot int32), sh2o and rtdis [NSOIL, JM, IM].

    Writes eta1, edir, ec, ett [JM, IM], et [NSOIL, JM, IM] and dew [JM, IM]."""
    if im is None or jm is None:
        jm, im = int(etp.shape[0]), int(etp.shape[1])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (etp, cmc, sh2o, rtdis, shdfac, smcmax, smcwlt, smcref, smcdry, cmcmax, pc, cfactr,
       fxexp, nroot, np.float32(dt), eta1, edir, ec, ett, et, dew, np.int32(im), np.int32(jm)))
    return eta1, edir, ec, ett, et, dew


def _require(st, names):
    missing = [n for n in names if n not in st.fields]
    if missing:
        detail = "; ".join(f"{n} from {REQUIRED[n]}" for n in missing)
        raise RuntimeError(f"phys-noah-evapo: state fields not produced yet: {detail}")


def _dt(ctx) -> np.float32:
    """DT = NPHS x ctx.dt, as phys-noah-driver derives DTBL."""
    nphs = int(getattr(ctx.cfg, "nphs", 2)) if ctx.cfg is not None else 2
    return np.float32(nphs * float(ctx.dt))


def run(ctx=None):
    """Done-rule entry. With ctx: bind ctx.st by producer name and write the unit outputs into ctx.st.
    Without ctx: one launch on a full-size 954 x 835 synthetic batch. Returns the arrays it wrote."""
    import cupy as cp

    if ctx is not None:
        st = ctx.st
        _require(st, tuple(REQUIRED))
        jm, im = int(st.SH2O.shape[1]), int(st.SH2O.shape[2])
        outs = {
            "ETA1": st.add("ETA1", (jm, im)),
            "EDIR": st.add("EDIR", (jm, im)),
            "EC": st.add("EC", (jm, im)),
            "ETT": st.add("ETT", (jm, im)),
            "ET": st.add("ET", (NSOIL, jm, im)),
            "DEW": st.add("DEW", (jm, im)),
        }
        evapo(st.ETP, st.CMC, st.SH2O, st.RTDIS, st.SHDFAC, st.SMCMAX, st.SMCWLT, st.SMCREF,
              st.SMCDRY, st.CMCMAX, st.PC, st.CFACTR, st.FXEXP, st.NROOT, _dt(ctx),
              outs["ETA1"], outs["EDIR"], outs["EC"], outs["ETT"], outs["ET"], outs["DEW"],
              im=im, jm=jm)
        return outs

    jm, im = 835, 954
    rng = np.random.default_rng(1)

    def u(lo, hi, shape=(jm, im)):
        return cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))

    smcwlt = u(0.02, 0.15)
    smcref = cp.maximum(u(0.2, 0.35), smcwlt + 0.05)
    rt = rng.uniform(0.0, 1.0, size=(NSOIL, jm, im)).astype(np.float32)
    rtdis = cp.asarray(rt / rt.sum(axis=0, keepdims=True))
    outs = {
        "ETA1": cp.zeros((jm, im), cp.float32), "EDIR": cp.zeros((jm, im), cp.float32),
        "EC": cp.zeros((jm, im), cp.float32), "ETT": cp.zeros((jm, im), cp.float32),
        "ET": cp.zeros((NSOIL, jm, im), cp.float32), "DEW": cp.zeros((jm, im), cp.float32),
    }
    evapo(u(-2.0e-5, 3.0e-4), u(0.0, 0.0005), u(0.02, 0.44, (NSOIL, jm, im)), rtdis,
          u(0.0, 1.0), u(0.35, 0.55), smcwlt, smcref, u(0.01, 0.05), u(0.0002, 0.0005),
          u(0.1, 0.9), u(0.2, 0.6), u(0.5, 2.0),
          cp.asarray(rng.integers(1, NSOIL + 1, size=(jm, im)).astype(np.int32)),
          np.float32(SYNTH_DT),
          outs["ETA1"], outs["EDIR"], outs["EC"], outs["ETT"], outs["ET"], outs["DEW"],
          im=im, jm=jm)
    return outs
