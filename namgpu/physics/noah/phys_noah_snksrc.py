"""phys-noah-snksrc: SNKSRC, phase-change sink/source of the soil heat equation (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3960-4047. NOAA US government
work; the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_snksrc.cu (phys_noah_snksrc, one launch over (i, j, soil layer)).
Started from the ArWen noah_snksrc (namgpu/third_party/arwen/core/kernels/noah.cu, Apache-2.0). FRH2O is the
device function in namgpu/kernels/phys_noah_frh2o.cuh.

Per-column inputs, as the NAM call sites use them:
  TAVG   TMPAVG of the layer (phys-noah-tmpavg; the live ITAVG = .TRUE. path, module_LS_NOAHLSM.F90:2828).
  QTOT   heat flux into the layer from HRT (phys-noah-hrt).
  SMC, SH2O  state table; SH2O is updated in place.
  STC    state table; used by the call-site gate.
  TBND   TBND interface temperature below each layer (phys-noah-tbnd); used by the gate.
  TSURF  (YY + (ZZ1-1)*STC(1))/ZZ1 from SFLX (top-layer gate).
  ICE    sea-ice flag from SFLX [JM, IM] int32. ZSOIL depends on it (module_LS_NOAHLSM.F90:1822-1827):
         ICE == 1 columns use ZSOIL(KZ) = -3*KZ/NSOIL, other columns use the cumulative NAM depths.
  SMCMAX, BEXP, PSIS  REDPRM soil-type values (phys-noah-redprm / parm tables).
  DT     SNKSRC time step = DTBL = DT*NPHS (module_TURBULENCE.F90:764 and 886): 25 s * 2 = 50 s.

The call-site gate is evaluated in the kernel: the Fortran calls SNKSRC only when the layer's
conditions hold (see the kernel header). The kernel's NAM_NSOIL is 4; any other NSOIL is refused.
The STC-variant of SNKSRC (ITAVG = .FALSE.) is dead in NAM (ITAVG is .TRUE. unconditionally) and is
not ported.

run(ctx): every input is bound from ctx.st.fields or ctx.sflx. A missing input raises KeyError naming
the field and the unit that produces it. Nothing is defaulted on the ctx path.
Without ctx, run() builds a full-size 954 x 835 x NSOIL random state for the done rule.
No host sync, no per-step allocation, everything on the current stream.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_snksrc"
KERNEL = "phys_noah_snksrc"
NSOIL = 4
DT_PHYS = 50.0  # DTBL = DT*NPHS = 25 s * 2 (module_TURBULENCE.F90:764), passed as DT at :886
ZSOIL_NAM = np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32)  # non-ice cumulative SLDPTH

_PRODUCER = {
    "TAVG": "phys-noah-tmpavg", "QTOT": "phys-noah-hrt", "SMC": "the state table",
    "SH2O": "the state table", "STC": "the state table", "TBND": "phys-noah-tbnd",
    "TSURF": "SFLX (YY, ZZ1 from sflx and STC)", "ICE": "SFLX (sflx['ICE'])",
    "YY": "SFLX (module_LS_NOAHLSM.F90 YY)", "ZZ1": "SFLX (module_LS_NOAHLSM.F90 ZZ1)",
    "SMCMAX": "phys-noah-redprm", "BEXP": "phys-noah-redprm", "PSIS": "phys-noah-redprm",
}


def _need(ctx, name: str):
    """A named binding from ctx.st.fields or ctx.sflx; missing -> KeyError naming the producer."""
    if name in ctx.st.fields:
        return ctx.st.fields[name]
    sflx = getattr(ctx, "sflx", None) or {}
    if name in sflx:
        return sflx[name]
    raise KeyError(f"phys-noah-snksrc needs {name} (produced by {_PRODUCER[name]}); "
                   "not bound in the state or sflx yet")


def snksrc(tavg, qtot, smc, sh2o, stc, tbnd, tsurf, zsoil, ice, smcmax, bexp, psis, dt, tsnsr):
    """TSNSR[l, j, i], SH2O[l, j, i] = SNKSRC(...) for every soil layer, with the call-site gate.

    float32 [NSOIL, JM, IM]: tavg, qtot, smc, sh2o, stc, tbnd, tsnsr. float32 [JM, IM]: tsurf,
    smcmax, bexp, psis. int32 [JM, IM]: ice. float32 [NSOIL]: zsoil (non-ice depths).
    sh2o is updated in place. Refuses any NSOIL other than NAM's 4.
    """
    nsoil, jm, im = (int(s) for s in smc.shape)
    if nsoil != NSOIL:
        raise ValueError(f"phys-noah-snksrc: NSOIL={nsoil}, but the kernel is NAM's NSOIL={NSOIL} "
                         "(hard-coded layer count; other layer counts are not ported)")
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_2d(im, jm, nsoil), kernels.BLOCK_2D,
      (tavg, qtot, smc, sh2o, stc, tbnd, tsurf, zsoil, ice, smcmax, bexp, psis,
       np.float32(dt), tsnsr, np.int32(im), np.int32(jm)))
    return tsnsr, sh2o


def run(ctx=None, *, tavg=None, qtot=None, smc=None, sh2o=None, stc=None, tbnd=None, tsurf=None,
        zsoil=None, ice=None, smcmax=None, bexp=None, psis=None, dt=DT_PHYS, tsnsr=None):
    """Done-rule entry: one launch over every soil layer of a full-size field set."""
    import cupy as cp

    if ctx is not None and tavg is None:
        st = ctx.st
        smc = _need(ctx, "SMC")
        sh2o = _need(ctx, "SH2O")
        tavg = _need(ctx, "TAVG")
        qtot = _need(ctx, "QTOT")
        stc = _need(ctx, "STC")
        tbnd = _need(ctx, "TBND")
        ice = _need(ctx, "ICE")
        smcmax = _need(ctx, "SMCMAX")
        bexp = _need(ctx, "BEXP")
        psis = _need(ctx, "PSIS")
        if tsurf is None:
            yy = _need(ctx, "YY")
            zz1 = _need(ctx, "ZZ1")
            tsurf = (yy + (zz1 - 1.0) * stc[0]) / zz1
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        zsoil = cp.asarray(ZSOIL_NAM)
        tsnsr = st.fields["TSNSR"] if "TSNSR" in st.fields else st.add("TSNSR", (NSOIL, jm, im))
    elif tavg is None:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        tavg = cp.asarray(rng.uniform(240.0, 300.0, size=(NSOIL, jm, im)).astype(np.float32))
        qtot = cp.asarray(rng.uniform(-60.0, 60.0, size=(NSOIL, jm, im)).astype(np.float32))
        smc = cp.asarray(rng.uniform(0.05, 0.45, size=(NSOIL, jm, im)).astype(np.float32))
        sh2o = cp.asarray(rng.uniform(0.0, 0.9, size=(NSOIL, jm, im)).astype(np.float32)) * smc
        stc = cp.asarray(rng.uniform(250.0, 285.0, size=(NSOIL, jm, im)).astype(np.float32))
        tbnd = cp.asarray(rng.uniform(250.0, 285.0, size=(NSOIL, jm, im)).astype(np.float32))
        tsurf = cp.asarray(rng.uniform(250.0, 285.0, size=(jm, im)).astype(np.float32))
        ice = cp.asarray((rng.uniform(size=(jm, im)) < 0.3).astype(np.int32))
        zsoil = cp.asarray(ZSOIL_NAM)
        smcmax = cp.asarray(rng.uniform(0.4, 0.5, size=(jm, im)).astype(np.float32))
        bexp = cp.asarray(rng.uniform(2.0, 12.0, size=(jm, im)).astype(np.float32))
        psis = cp.asarray(rng.uniform(0.05, 0.7, size=(jm, im)).astype(np.float32))
        tsnsr = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    else:
        jm, im = int(smc.shape[1]), int(smc.shape[2])
        if zsoil is None:
            zsoil = cp.asarray(ZSOIL_NAM)
        if tsnsr is None:
            tsnsr = cp.zeros((NSOIL, jm, im), dtype=cp.float32)
    snksrc(tavg, qtot, smc, sh2o, stc, tbnd, tsurf, zsoil, ice, smcmax, bexp, psis, dt, tsnsr)
    return {"TSNSR": tsnsr, "SH2O": sh2o}
