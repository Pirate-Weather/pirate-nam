"""phys-noah-hrt: HRT, soil heat-equation RHS and tridiagonal coefficients (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:2785-3035, with the helpers
FRH2O, SNKSRC, TBND, TDFCND and TMPAVG from the same module (one source: noah_soil_helpers.cuh).
NOAA US government work; the module's NOAA header is kept as the notice in the kernel file. The
column structure was transcribed from ArWen noah_hrt (Apache-2.0); the notice is in the kernel file.
Kernel: namgpu/kernels/phys_noah_hrt.cu (phys_noah_hrt, one thread per (i, j) column, one launch).

Outputs per column: RHSTS, AI, BI, CI ([NSOIL, JM, IM]) for ICE == 0 columns and SH2O updated in
place by SNKSRC. ICE == 1 columns belong to HRTICE (module_LS_NOAHLSM.F90:3762-3764): the wrapper
runs phys-noah-hrtice on them first, and the kernel skips them, so SH2O on sea ice is untouched.

No host sync, no per-step allocation, everything on the current stream.

Bindings in run(ctx). Nothing is defaulted on the ctx path; a missing producer raises.
  STC, SMC, SH2O     ctx.st.fields (state table).
  ISLTYP             ctx.st.fields (state table); soil index = ISLTYP - 1.
  SMCMAX, PSISAT, BEXP, QUARTZ
                     soil table gather at ISLTYP: MAXSMC, SATPSI, BB, QTZ (SOILPARM.TBL, via
                     phys-noah-redprm's bound tables). Fortran: REDPRM (module_LS_NOAHLSM.F90:3582-3589).
  CSOIL              GENPARM CSOIL_DATA (scalar, REDPRM line 3581), the same for every column.
  ZBOT               GENPARM ZBOT_DATA (scalar, REDPRM line 3596). Production value -8.0 m.
  ZSOIL              cumulative -SLDPTH from ctx.st.fields["SLDPTH"] (SFLX lines 1835-1837), on device.
  TBOT               ctx.st.fields["TBOT"] (solver state table: TMN where ICE == 0, 271.16 K where ICE == 1).
  VEGTYP             ctx.sflx["VEGTYP"] or ctx.st.fields["VEGTYP"].
  YY, ZZ1, DF1       ctx.sflx or ctx.st.fields. SFLX produces them (module_LS_NOAHLSM.F90:3363-3371,
                     final DF1 after the VEGTYP override and EXP(SBETA*SHDFAC)). phys-noah-sflx is todo.
  ICE                ctx.sflx["ICE"]. SFLX's ICE flag (module_LS_NOAHLSM.F90:744-746).
  HRTICE inputs      SFCTMP, TH2, FDOWN, EMISSI, T24, RCH, RR, BETA, EPSCA (ctx.sflx), the phys-noah-hrtice
                     contract.
  RHSTS, AI, BI, CI  unit-owned scratch, allocated once per (JM, IM).

Without ctx it builds a full-size 954 x 835 x 4 random batch (plausible soil states, ~30 % sea-ice
columns, SFLX-consistent intermediates) and returns the outputs. The no-ctx smoke has no parm tables,
so its soil parameters are random ranges; the ctx path is the production binding and is tested by
tools/check-phys-noah-hrt.py.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_hrtice as _hrtice
from namgpu.physics.noah import phys_noah_parm_tables as parm
from namgpu.physics.noah import phys_noah_redprm as _redprm

UNIT = "phys_noah_hrt"
KERNEL = "phys_noah_hrt"
BLOCK = (128, 1, 1)
NSOIL = 4
IVEGSRC = 1                                                     # IGBP: urban IW = 13
DT = 50.0                                                       # DTPHS, NPHS = 2 x 25 s
ZBOT_SMOKE = -8.0                                               # GENPARM ZBOT_DATA, no-ctx smoke only
CSOIL_SMOKE = 1.26e6                                            # GENPARM CSOIL_DATA, no-ctx smoke only
HRTICE_SFLX = ("SFCTMP", "TH2", "FDOWN", "EMISSI", "T24", "RCH", "RR", "BETA", "EPSCA")

_scratch: dict = {}


def _grid(im: int, jm: int):
    n = im * jm
    return ((n + BLOCK[0] - 1) // BLOCK[0], 1, 1)


def hrt(rhsts, stc, smc, sh2o, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, bexp, df1,
        quartz, csoil, vegtyp, ice, ai, bi, ci, ivegsrc=IVEGSRC, dt=DT):
    """One launch: RHSTS, AI, BI, CI for ICE == 0 columns; SH2O updated in place.

    Shapes: soil fields [NSOIL, JM, IM] float32; per-column [JM, IM] float32; VEGTYP and ICE int32
    [JM, IM]; zsoil [NSOIL] float32 on device. Columns with ICE == 1 are skipped (HRTICE owns them).
    """
    jm, im = int(stc.shape[1]), int(stc.shape[2])
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (rhsts, stc, smc, sh2o, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, bexp, df1,
       quartz, csoil, vegtyp, ice, ai, bi, ci, np.int32(ivegsrc), np.float32(dt),
       np.int32(im), np.int32(jm)))
    return rhsts, ai, bi, ci


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        soil = lambda: cp.zeros((NSOIL, jm, im), dtype=cp.float32)   # noqa: E731
        col = lambda: cp.zeros((jm, im), dtype=cp.float32)           # noqa: E731
        _scratch[key] = {
            "RHSTS": soil(), "AI": soil(), "BI": soil(), "CI": soil(),
            "SMCMAX": col(), "YY": col(), "ZZ1": col(), "TBOT": col(), "ZBOT": col(),
            "PSISAT": col(), "BEXP": col(), "DF1": col(), "QUARTZ": col(), "CSOIL": col(),
            "VEGTYP": cp.zeros((jm, im), dtype=cp.int32),
            "ICE": cp.zeros((jm, im), dtype=cp.int32),
            "ZSOIL": cp.zeros((NSOIL,), dtype=cp.float32),
        }
    return _scratch[key]


def _need(name: str, st, sflx: dict, unit: str = "phys-noah-hrt"):
    """A bound input: ctx.sflx first (SFLX producer outputs), then the state table. Never defaulted."""
    if sflx and name in sflx:
        return sflx[name]
    if name in st.fields:
        return st.fields[name]
    raise RuntimeError(
        f"{unit}: {name} is not bound. Its producer is phys-noah-sflx (todo) or the state table; "
        "no default is used, because a default would give a plausible but wrong soil column.")


def _bind_ctx(ctx):
    """Production bindings (see module docstring). Returns the kernel inputs and the HRTICE dict."""
    import cupy as cp

    st = ctx.st
    sflx = getattr(ctx, "sflx", None) or {}
    stc = _need("STC", st, {})
    smc = _need("SMC", st, {})
    sh2o = _need("SH2O", st, {})
    jm, im = int(stc.shape[1]), int(stc.shape[2])
    sc = _scratch_arrays(jm, im)

    # soil tables gathered at ISLTYP (1-based Fortran index -> 0-based)
    tabs, scal = _redprm._bound_tables(ctx)
    host = ctx.tables[parm.KEY_HOST]
    isltyp = cp.asarray(_need("ISLTYP", st, {}))
    idx = isltyp.astype(cp.int32) - 1
    sc["SMCMAX"][...] = tabs["MAXSMC"][idx]
    sc["PSISAT"][...] = tabs["SATPSI"][idx]
    sc["BEXP"][...] = tabs["BB"][idx]
    sc["QUARTZ"][...] = tabs["QTZ"][idx]
    sc["CSOIL"][...] = float(host.gen["CSOIL_DATA"])
    sc["ZBOT"][...] = float(host.gen["ZBOT_DATA"])

    # ZSOIL(K) = -SLDPTH(K) + ZSOIL(K-1), SFLX lines 1835-1837 (device cumulative sum, no host read)
    sc["ZSOIL"][...] = -cp.cumsum(cp.asarray(_need("SLDPTH", st, {})).astype(cp.float32))

    sc["TBOT"][...] = _need("TBOT", st, {})
    sc["VEGTYP"][...] = _need("VEGTYP", st, sflx).astype(cp.int32)
    sc["YY"][...] = _need("YY", st, sflx)
    sc["ZZ1"][...] = _need("ZZ1", st, sflx)
    sc["DF1"][...] = _need("DF1", st, sflx)
    sc["ICE"][...] = _need("ICE", st, sflx).astype(cp.int32)

    sf = {k: _need(k, st, sflx) for k in HRTICE_SFLX}
    sf["DF1"] = sc["DF1"]
    ivegsrc = int(ctx.cfg.raw["ivegsrc"])                 # IVEGSRC from the run config, not a constant
    return stc, smc, sh2o, sc, sf, ivegsrc


def run(ctx=None, *, stc=None, smc=None, sh2o=None):
    """Done-rule entry: one HRTICE launch and one HRT launch on a full-size 954 x 835 x 4 column set.

    Returns the arrays it wrote."""
    import cupy as cp

    ivegsrc = IVEGSRC
    if ctx is not None and stc is None:
        stc, smc, sh2o, sc, sf, ivegsrc = _bind_ctx(ctx)
        jm, im = int(stc.shape[1]), int(stc.shape[2])
    elif stc is None:
        ivegsrc = IVEGSRC
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        smcmax = rng.uniform(0.35, 0.50, size=(jm, im)).astype(np.float32)
        smc = (smcmax[None] * rng.uniform(0.3, 1.0, size=(NSOIL, jm, im))).astype(np.float32)
        sh2o = (smc * rng.uniform(0.5, 1.0, size=(NSOIL, jm, im))).astype(np.float32)
        stc = rng.uniform(262.0, 292.0, size=(NSOIL, jm, im)).astype(np.float32)
        stc, smc, sh2o = cp.asarray(stc), cp.asarray(smc), cp.asarray(sh2o)
        sc = _scratch_arrays(jm, im)
        sc["SMCMAX"][...] = cp.asarray(smcmax)
        sc["YY"][...] = cp.asarray(rng.uniform(265.0, 295.0, size=(jm, im)).astype(np.float32))
        sc["ZZ1"][...] = 1.0
        sc["TBOT"][...] = cp.asarray(rng.uniform(270.0, 290.0, size=(jm, im)).astype(np.float32))
        sc["ZBOT"][...] = ZBOT_SMOKE
        sc["PSISAT"][...] = cp.asarray(rng.uniform(0.1, 0.8, size=(jm, im)).astype(np.float32))
        sc["BEXP"][...] = cp.asarray(rng.uniform(2.0, 11.0, size=(jm, im)).astype(np.float32))
        sc["DF1"][...] = cp.asarray(rng.uniform(0.5, 3.0, size=(jm, im)).astype(np.float32))
        sc["QUARTZ"][...] = cp.asarray(rng.uniform(0.05, 0.9, size=(jm, im)).astype(np.float32))
        sc["CSOIL"][...] = CSOIL_SMOKE
        sc["VEGTYP"][...] = cp.asarray(rng.integers(1, 21, size=(jm, im)).astype(np.int32))
        sc["ICE"][...] = cp.asarray((rng.uniform(size=(jm, im)) < 0.3).astype(np.int32))
        sc["ZSOIL"][...] = cp.asarray(np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32))
        sfctmp = rng.uniform(260.0, 300.0, size=(jm, im)).astype(np.float32)
        sf = {
            "SFCTMP": cp.asarray(sfctmp),
            "TH2": cp.asarray(rng.uniform(260.0, 300.0, size=(jm, im)).astype(np.float32)),
            "FDOWN": cp.asarray(rng.uniform(200.0, 400.0, size=(jm, im)).astype(np.float32)),
            "EMISSI": cp.asarray(np.full((jm, im), 0.95, dtype=np.float32)),
            "T24": cp.asarray((sfctmp * sfctmp * sfctmp * sfctmp).astype(np.float32)),
            "RCH": cp.asarray(rng.uniform(0.002, 0.006, size=(jm, im)).astype(np.float32)),
            "RR": cp.asarray(rng.uniform(1.0, 2.0, size=(jm, im)).astype(np.float32)),
            "BETA": cp.asarray(rng.uniform(0.0, 1.0, size=(jm, im)).astype(np.float32)),
            "EPSCA": cp.asarray(rng.uniform(-5.0, 5.0, size=(jm, im)).astype(np.float32)),
            "DF1": sc["DF1"],
        }
    else:
        jm, im = int(stc.shape[1]), int(stc.shape[2])
        sc = _scratch_arrays(jm, im)
        raise RuntimeError("phys-noah-hrt: pass ctx (the bound state), not raw arrays")

    # HRTICE first: writes the ICE == 1 cells (and zeros the others); HRT then fills the ICE == 0 cells
    # and skips ICE == 1, so the two routines cover every column exactly once (SFLX 3762-3772).
    _hrtice.hrtice(stc, sc["ICE"], sf, sc["RHSTS"], sc["AI"], sc["BI"], sc["CI"])
    hrt(sc["RHSTS"], stc, smc, sh2o, sc["SMCMAX"], sc["ZSOIL"], sc["YY"], sc["ZZ1"],
        sc["TBOT"], sc["ZBOT"], sc["PSISAT"], sc["BEXP"], sc["DF1"], sc["QUARTZ"],
        sc["CSOIL"], sc["VEGTYP"], sc["ICE"], sc["AI"], sc["BI"], sc["CI"], ivegsrc=ivegsrc)
    return {"RHSTS": sc["RHSTS"], "AI": sc["AI"], "BI": sc["BI"], "CI": sc["CI"], "SH2O": sh2o}
