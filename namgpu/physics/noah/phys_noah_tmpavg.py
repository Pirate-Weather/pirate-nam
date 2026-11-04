"""phys-noah-tmpavg: TMPAVG, soil layer average temperature in the freezing/thawing layer (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:5095-5197. NOAA US government
work; the module header is kept as the notice in the kernel file. Start point: ArWen
core/kernels/noah.cu noah_tmpavg (Apache-2.0, namgpu/third_party/arwen/LICENSE).
Kernel: namgpu/kernels/phys_noah_tmpavg.cu (phys_noah_tmpavg, one thread per (i, j, soil layer),
one launch, grid.z = NSOIL). No host sync, no per-step allocation beyond the first-call scratch,
everything on the current stream. Fast numerics: NVRTC may contract a*b+c (fmad), so the output is
not bit-identical to NAM's compiled Fortran; it is held to a tolerance (tests/test_phys_noah_tmpavg.py).

Inputs: TUP, TM, TDN, TAVG [NSOIL, JM, IM] float32 (C-contiguous); ZSOIL [NSOIL] float32 (layer bottom
depth, m, negative down). TAVG entry k (0-based) is the Fortran TAVG for K = k + 1.

Caller contract (SFLX, ITAVG branch, module_LS_NOAHLSM.F90:2883-3003), implemented in compute():
  TSURF = (YY + (ZZ1 - 1) * STC(1)) / ZZ1                       (line 2883)
  TBK(K) = TBND(STC(K), STC(K+1), K) for K < NSOIL; TBK(NSOIL) = TBND(STC(NSOIL), TBOT, NSOIL)
           (phys-noah-tbnd gives the bottom interface of each layer)
  K = 1:   TUP = TSURF,        TDN = TBK(1)                      (lines 2890, 2894)
  K > 1:   TUP = TBK(K-1),     TDN = TBK(K)                      (lines 2958, 3003)
So TUP[0] = TSURF, TUP[k] = TDN[k-1], TDN[k] = TBND bottom interface of layer k.
ZSOIL is the land vector from SLDPTH (module_LS_NOAHLSM.F90:1835-1837) or, on sea-ice points, the
vector -3*KZ/NSOIL (lines 1822-1826). Both are built here. TBOT and ZBOT come from the bound state.

run(ctx): the done-rule entry point. With ctx it reads ctx.st.STC, ctx.st.fields YY, ZZ1, TBOT and
(optional) SLDPTH, ZBOT from the bound parm tables (ZBOT_DATA), and ICE from ctx.sflx when present.
A missing field is a refusal naming it. Without ctx it builds a full-size 954 x 835 x 4 random batch
with explicit test constants (ZBOT -2.0 m, land ZSOIL).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.physics.noah import phys_noah_redprm as _redprm
from namgpu.physics.noah import phys_noah_tbnd as _tbnd

UNIT = "phys_noah_tmpavg"
KERNEL = "phys_noah_tmpavg"
BLOCK = (32, 8, 1)
NSOIL = 4
# DZSOIL 0.1, 0.3, 0.6, 1.0 m, the module default (module_LS_NOAHLSM.F90:31, phys-noah-redprm).
SLDPTH_DEFAULT = (0.1, 0.3, 0.6, 1.0)
ZBOT_TEST = -2.0   # only for the no-ctx test batch

_scratch: dict = {}
_zsoil_cache: dict = {}


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, NSOIL)


def zsoil_vector(ice: int = 0, sldpth=SLDPTH_DEFAULT) -> np.ndarray:
    """ZSOIL as SFLX builds it: sea-ice points -3*KZ/NSOIL (lines 1822-1826), else from SLDPTH (1835-1837)."""
    if int(ice) == 1:
        return np.array([-3.0 * kz / NSOIL for kz in range(1, NSOIL + 1)], dtype=np.float32)
    return np.asarray(_redprm._sldpth_zsoil(np, tuple(float(x) for x in sldpth))[1], dtype=np.float32)


def _zsoil_dev(ice: int, sldpth):
    import cupy as cp

    key = (int(ice), tuple(float(x) for x in sldpth))
    if key not in _zsoil_cache:
        _zsoil_cache[key] = cp.asarray(zsoil_vector(ice, sldpth))
    return _zsoil_cache[key]


def _check(name, a, shape):
    import cupy as cp

    if not isinstance(a, cp.ndarray):
        raise TypeError(f"phys-noah-tmpavg: {name} must be a CuPy array")
    if a.dtype != cp.float32:
        raise TypeError(f"phys-noah-tmpavg: {name} is {a.dtype}, need float32 (Fortran REAL, kfpt=single)")
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"phys-noah-tmpavg: {name} has shape {tuple(a.shape)}, need {tuple(shape)}")
    if not a.flags.c_contiguous:
        raise ValueError(f"phys-noah-tmpavg: {name} must be C-contiguous")


def tmpavg(tavg, tup, tm, tdn, zsoil):
    """TAVG[k] = TMPAVG(TUP[k], TM[k], TDN[k], ZSOIL, K = k + 1). One launch, float32 device arrays."""
    if tm.ndim != 3 or int(tm.shape[0]) != NSOIL:
        raise ValueError(f"phys-noah-tmpavg: TM must be [{NSOIL}, JM, IM], got {tuple(tm.shape)}")
    _check("TM", tm, tm.shape)
    _check("TUP", tup, tm.shape)
    _check("TDN", tdn, tm.shape)
    _check("TAVG", tavg, tm.shape)
    _check("ZSOIL", zsoil, (NSOIL,))
    _, jm, im = (int(s) for s in tm.shape)
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (tup, tm, tdn, zsoil, tavg, np.int32(im), np.int32(jm), np.int32(NSOIL)))
    return tavg


def interface_temps(stc, tsurf, tbot, zsoil, zbot, tup, tdn, tbnd_out):
    """SFLX interface temperatures for the ITAVG branch (see the module docstring for the Fortran lines)."""
    _tbnd.tbnd(stc, tbot, zsoil, zbot, tbnd_out)   # tbnd_out[k] = bottom interface of layer k
    tup[0] = tsurf
    tup[1:] = tbnd_out[:-1]
    tdn[...] = tbnd_out
    return tup, tdn


def _scratch_arrays(jm: int, im: int):
    import cupy as cp

    key = (jm, im)
    if key not in _scratch:
        soil = lambda: cp.zeros((NSOIL, jm, im), dtype=cp.float32)   # noqa: E731
        _scratch[key] = {
            "TAVG": soil(), "TAVG_ICE": soil(), "TUP": soil(), "TDN": soil(), "TBND": soil(),
            "TSURF": cp.zeros((jm, im), dtype=cp.float32),
        }
    return _scratch[key]


def compute(stc, yy, zz1, tbot, zbot, ice=0, sldpth=SLDPTH_DEFAULT):
    """TAVG for every layer from SFLX inputs. ice: int (one zsoil class) or an int array [JM, IM]
    (sea-ice points use the -3*KZ/NSOIL zsoil, the rest the land zsoil; both run, then selected)."""
    import cupy as cp

    jm, im = int(stc.shape[1]), int(stc.shape[2])
    sc = _scratch_arrays(jm, im)
    tsurf = sc["TSURF"]
    tsurf[...] = (yy + (zz1 - 1.0) * stc[0]) / zz1
    if np.ndim(ice) == 0:
        zs = _zsoil_dev(int(ice), sldpth)
        interface_temps(stc, tsurf, tbot, zs, zbot, sc["TUP"], sc["TDN"], sc["TBND"])
        return tmpavg(sc["TAVG"], sc["TUP"], stc, sc["TDN"], zs)
    mask = cp.asarray(ice) == 1
    zs_land = _zsoil_dev(0, sldpth)
    interface_temps(stc, tsurf, tbot, zs_land, zbot, sc["TUP"], sc["TDN"], sc["TBND"])
    tmpavg(sc["TAVG"], sc["TUP"], stc, sc["TDN"], zs_land)
    zs_ice = _zsoil_dev(1, sldpth)
    interface_temps(stc, tsurf, tbot, zs_ice, zbot, sc["TUP"], sc["TDN"], sc["TBND"])
    tmpavg(sc["TAVG_ICE"], sc["TUP"], stc, sc["TDN"], zs_ice)
    cp.copyto(sc["TAVG"], sc["TAVG_ICE"], where=mask[None])
    return sc["TAVG"]


def run(ctx=None, *, ice=None):
    """Done-rule entry: one full-size 954 x 835 x 4 SFLX-style pass. Returns the arrays it wrote."""
    import cupy as cp

    if ctx is not None:
        st = ctx.st
        for name in ("YY", "ZZ1", "TBOT"):
            if name not in st.fields:
                raise RuntimeError(
                    f"phys-noah-tmpavg needs the state field {name} (SFLX per-column input); "
                    "it is not in ctx.st, so TMPAVG will not run on invented interface temperatures")
        stc = st.STC
        sldpth = tuple(cp.asnumpy(st.fields["SLDPTH"])) if "SLDPTH" in st.fields else SLDPTH_DEFAULT
        zbot = float(_redprm._bound_tables(ctx)[1]["ZBOT_DATA"])
        if ice is None:
            ice = st.fields["ICE"] if "ICE" in st.fields else getattr(ctx, "sflx", {}).get("ICE", 0)
        yy, zz1, tbot = st.fields["YY"], st.fields["ZZ1"], st.fields["TBOT"]
    else:
        jm, im = 835, 954
        rng = np.random.default_rng(1)
        stc = cp.asarray(rng.uniform(250.0, 295.0, size=(NSOIL, jm, im)).astype(np.float32))
        yy = cp.asarray(rng.uniform(265.0, 295.0, size=(jm, im)).astype(np.float32))
        zz1 = cp.ones((jm, im), dtype=cp.float32)
        tbot = cp.asarray(rng.uniform(265.0, 285.0, size=(jm, im)).astype(np.float32))
        sldpth = SLDPTH_DEFAULT
        zbot = ZBOT_TEST
        ice = 0 if ice is None else ice
    tavg = compute(stc, yy, zz1, tbot, zbot, ice=ice, sldpth=sldpth)
    return {"TAVG": tavg}
