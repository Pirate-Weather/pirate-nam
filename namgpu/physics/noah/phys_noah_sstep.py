"""phys-noah-sstep: SSTEP, the soil moisture step with the ROSR12 tridiagonal solve (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:4855-4942 (SSTEP) and
:3666-3725 (ROSR12). NOAA US government work; the module header is kept in the kernel file as the
notice. Start point: ArWen core/kernels/noah.cu (Apache-2.0, namgpu/third_party/arwen/LICENSE).
Kernel: namgpu/kernels/phys_noah_sstep.cu (phys_noah_sstep, one thread per (i, j) column, one launch,
all NSOIL layers looped inside the thread).

Arrays: soil fields float32 [NSOIL=4, JM, IM] (layer 1 at the top); column fields float32 [JM, IM];
ZSOIL float32 [NSOIL, JM, IM], per column. SFLX builds ZSOIL per point (module_LS_NOAHLSM.F90:1823-1837):
sea-ice points (ICE==1) use ZSOIL(KZ) = -3*KZ/NSOIL, land points use the caller's SLDPTH (DZS) cumulated.
zsoil_columns() builds it; the caller passes its own sldpth and the per-point ice flag.

INOUT Fortran arguments are written back in place: RHSTT (= DELTA), AI and BI (scaled by DT), CI
(= ROSR12 P), CMC. SMC, SH2OOUT and RUNOFF3 are outputs. SH2OIN and SH2OOUT may be the same array
(SFLX calls SSTEP(SH2O,SH2O,...)). SFLX's first SSTEP call passes DUMMY as CMC; the SFLX device code
(phys_noah_sflx.cu, noah_sstep call near line 627) passes a function-local dummy there, so that
obligation is met inside the SFLX kernel. This kernel always updates CMC, so any caller that must not
change the canopy water passes a scratch array.

Soil ice is its own field, SOIL_ICE [NSOIL, JM, IM]. The state table's SICE is the H-point sea-ice
fraction [JM, IM] (parm/nam_phy_state.txt:114) and is used only for the ICE flag here.

No host sync, no per-step allocation beyond the scratch arrays allocated once, current stream.

run(ctx): the done-rule entry point. With ctx it reads the state fields listed in REQUIRED_FIELDS and
raises, naming every missing one, if any is absent. Nothing is filled with test values on this path.
RUNOFF3 and SMC are outputs and are allocated zero in the state when absent. Without ctx it builds a
full-size random 954 x 835 batch with a sea-ice subset so both ZSOIL branches run.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_sstep"
KERNEL = "phys_noah_sstep"
BLOCK = (32, 8, 1)
NSOIL = 4

# Land-point soil layer thicknesses (SLDPTH; module_LS_NOAHLSM.F90:31 and DZS default). run() takes the
# caller's values through its sldpth argument; this is only the NAM default for callers that pass none.
SLDPTH_NAM = (0.1, 0.3, 0.6, 1.0)
# Sea-ice case (ICE==1) when SICE exceeds this, the same threshold as phys-noah-sflx and phys-noah-alcalc.
XICE_THRESHOLD = 0.5

# State fields the ctx path needs. SMC and RUNOFF3 are outputs and are not in this list.
REQUIRED_FIELDS = ("SH2O", "CMC", "RHSTT", "RHSCT", "SMCMAX", "SMCWLT", "CMCMAX", "SOIL_ICE",
                   "AI", "BI", "CI", "SICE")

# Test-only plausible ranges for the random batch of run() without ctx.
_RANGES = {
    "SMCMAX": (0.40, 0.50),
    "SMCWLT": (0.02, 0.10),
    "CMCMAX": (0.0, 5.0e-4),
    "RHSCT": (-1.0e-6, 1.0e-6),
    "RHSTT": (-1.0e-5, 1.0e-5),
    "AI": (-0.05, 0.05),
    "BI": (0.0, 0.05),
    "CI": (-0.05, 0.05),
    "SH2O": (0.05, 0.40),
    "SOIL_ICE": (0.0, 0.05),
    "CMC": (0.0, 5.0e-4),
}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def zsoil_columns(sldpth, ice, xp):
    """ZSOIL[NSOIL, JM, IM] per column, as SFLX sets it (module_LS_NOAHLSM.F90:1823-1837).

    sldpth: the caller's per-layer thickness (m), length NSOIL. ice: [JM, IM] bool or 0/1, the ICE flag.
    Land: ZSOIL(1) = -SLDPTH(1); ZSOIL(K) = -SLDPTH(K) + ZSOIL(K-1).
    Sea ice: ZSOIL(KZ) = -3.*FLOAT(KZ)/FLOAT(NSOIL).
    """
    sl = np.asarray(sldpth, dtype=np.float32)
    if sl.shape != (NSOIL,):
        raise ValueError(f"phys-noah-sstep sldpth must have {NSOIL} layers, got shape {sl.shape}")
    land = np.empty(NSOIL, dtype=np.float32)
    acc = np.float32(0.0)
    for k in range(NSOIL):
        acc = np.float32(acc - sl[k])
        land[k] = acc
    ice_vals = np.array([np.float32(-3.0) * np.float32(kz) / np.float32(NSOIL)
                         for kz in range(1, NSOIL + 1)], dtype=np.float32)
    ice_mask = xp.asarray(ice).astype(bool)
    land_d = xp.asarray(land).reshape(NSOIL, 1, 1)
    ice_d = xp.asarray(ice_vals).reshape(NSOIL, 1, 1)
    return xp.ascontiguousarray(xp.where(ice_mask[None], ice_d, land_d).astype(xp.float32))


def _check(name, arr, shape):
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"phys-noah-sstep {name} has shape {tuple(arr.shape)}, expected {tuple(shape)}")


def sstep(sh2oin, sh2oout, cmc, rhstt, rhsct, dt, smcmax, smcwlt, cmcmax, zsoil,
          smc, soil_ice, ai, bi, ci, runoff3):
    """One SSTEP over the whole [JM, IM] domain. Writes SH2OOUT, SMC, RUNOFF3 and the INOUT arrays.

    dt is the time step in seconds. Soil arrays are [NSOIL, JM, IM] float32, column arrays [JM, IM],
    zsoil is [NSOIL, JM, IM] (see zsoil_columns).
    """
    _, jm, im = (int(s) for s in sh2oin.shape)
    soil, col = (NSOIL, jm, im), (jm, im)
    for name, arr in (("SH2OIN", sh2oin), ("SH2OOUT", sh2oout), ("RHSTT", rhstt), ("SMC", smc),
                      ("SOIL_ICE", soil_ice), ("AI", ai), ("BI", bi), ("CI", ci), ("ZSOIL", zsoil)):
        _check(name, arr, soil)
    for name, arr in (("CMC", cmc), ("RHSCT", rhsct), ("SMCMAX", smcmax), ("SMCWLT", smcwlt),
                      ("CMCMAX", cmcmax), ("RUNOFF3", runoff3)):
        _check(name, arr, col)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (sh2oin, sh2oout, cmc, rhstt, rhsct, np.float32(dt), smcmax, smcwlt, cmcmax, zsoil,
       smc, soil_ice, ai, bi, ci, runoff3, np.int32(im), np.int32(jm)))
    return sh2oout


def _need(st, names):
    """The named state fields, or one error naming every missing one. Nothing is substituted."""
    missing = [n for n in names if n not in st.fields]
    if missing:
        raise RuntimeError(
            f"phys-noah-sstep needs state fields {missing}; each is produced by its named unit "
            f"(SH2O/SOIL_ICE: phys-noah-srt and the soil init, CMC/SMCMAX/SMCWLT/CMCMAX/RHSCT/RHSTT: "
            f"the canopy and soil producers, AI/BI/CI: phys-noah-srt, SICE: state table). "
            f"No scratch values are used on the ctx path.")
    return {n: st.fields[n] for n in names}


def run(ctx=None, *, dt: float = 25.0, sldpth=SLDPTH_NAM, xice_threshold: float = XICE_THRESHOLD):
    """Done-rule entry: one launch on a full-size 954 x 835 column set. Returns the arrays it wrote."""
    import cupy as cp

    rng = np.random.default_rng(1)
    if ctx is None:
        jm, im = 835, 954
        soil = (NSOIL, jm, im)
        col = (jm, im)

        def rnd(name, shape):
            return cp.asarray(rng.uniform(*_RANGES[name], size=shape).astype(np.float32))

        sh2o = rnd("SH2O", soil)
        cmc = rnd("CMC", col)
        rhstt = rnd("RHSTT", soil)
        rhsct = rnd("RHSCT", col)
        smcmax = rnd("SMCMAX", col)
        smcwlt = rnd("SMCWLT", col)
        cmcmax = rnd("CMCMAX", col)
        smc = cp.zeros(soil, dtype=cp.float32)
        soil_ice = rnd("SOIL_ICE", soil)
        ai = rnd("AI", soil)
        bi = rnd("BI", soil)
        ci = rnd("CI", soil)
        runoff3 = cp.zeros(col, dtype=cp.float32)
        ice = cp.asarray(rng.random(col) < 0.10)
        sh2oout = sh2o
    else:
        st = ctx.st
        f = _need(st, REQUIRED_FIELDS)
        sh2o = f["SH2O"]
        jm, im = int(sh2o.shape[1]), int(sh2o.shape[2])
        soil = (NSOIL, jm, im)
        col = (jm, im)
        cmc = f["CMC"]
        rhstt = f["RHSTT"]
        rhsct = f["RHSCT"]
        smcmax = f["SMCMAX"]
        smcwlt = f["SMCWLT"]
        cmcmax = f["CMCMAX"]
        soil_ice = f["SOIL_ICE"]
        ai, bi, ci = f["AI"], f["BI"], f["CI"]
        sice = f["SICE"]
        smc = st.add("SMC", soil)
        runoff3 = st.add("RUNOFF3", col)
        _check("SICE", sice, col)
        ice = sice > np.float32(xice_threshold)
        sh2oout = sh2o            # SSTEP(SH2O, SH2O, ...) in SFLX: in place
    zsoil = zsoil_columns(sldpth, ice, cp)
    sstep(sh2o, sh2oout, cmc, rhstt, rhsct, dt, smcmax, smcwlt, cmcmax, zsoil,
          smc, soil_ice, ai, bi, ci, runoff3)
    return {"SH2O": sh2oout, "SMC": smc, "CMC": cmc, "RHSTT": rhstt, "AI": ai, "BI": bi,
            "CI": ci, "RUNOFF3": runoff3}
