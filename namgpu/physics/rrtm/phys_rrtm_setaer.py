"""phys-rrtm-setaer: SETAER, aerosol optical properties per layer (RRTM group, nam-gpu).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1). NOAA US government work; the NOAA header of
phys/radiation_aerosols_nmmb.f is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_setaer.cu). Source: phys/radiation_aerosols_nmmb.f:1845-2371.

Launch sequence (Fortran order, one launch per stage, no host sync inside):
  1. phys_rrtm_setaer_zero   zero aerosw / aerolw                      (Fortran 1871-1895)
  2. phys_rrtm_setaer_prep   alon, alat; dz, hz per column              (Fortran 1956-2051)
  3. aer_property stage      Fortran 2045, unit phys-rrtm-aer-property, module
                             namgpu.physics.rrtm.phys_rrtm_aer_property (NOT part of this unit)
  4. phys_rrtm_setaer_volc   stratospheric volcanic aerosol, lavoflg    (Fortran 2092-2352)

The aer_property stage is called exactly as the Fortran call passes its arguments, so the unit
that ports it implements `run_call` with these keywords: prsi, prsl, prslk, tvly, rhlay, dz, hz,
tracer, alon, alat, slmsk, laersw, laerlw, imax, nlay, nlp1, ivflip, aerosw, aerolw, stream.
Until that module exists, run_call raises for any call that reaches step 3 (iaerflg > 0),
naming the dependency. Nothing is skipped silently.

Inputs (device float32, CONTRACT names; level-major flattened columns, column = j*IM + i):
  prsi  [NLP1, IMAX] interface pressure (prsi)     prsl [NLAY, IMAX] layer pressure (prsl)
  tlay, qlay [NLAY, IMAX] layer temperature, specific humidity (old-code arguments)
  xlon, xlat [IMAX] radians; slmsk [IMAX]; rhlay, tvly, prslk [NLAY, IMAX]; tracer [NTRAC, NLAY, IMAX]
  (tracer and prslk are part of the Fortran argument list; setaer itself does not read them, and
  aer_property does not read tracer in the NMMB source either. They are passed through unchanged.)
Outputs (device float32, written): aerosw [NF_AESW=3, NBDSW=14, NLAY, IMAX], aerolw [NF_AELW=3,
  NBDLW=16, NLAY, IMAX]; field 0 = optical depth, 1 = single scattering albedo, 2 = asymmetry.
  With the grrad_nmmb call the CuPy shapes are [NF, NBDSW, LMK, JM, IM], which is the same memory.

Module state (set by aer_init / aer_update / volc_update, units owned elsewhere), passed in
ctx.tables["aer_state"] by the driver:
  iaerflg (int, physparam default 0 = setaer returns with zero outputs), laswflg, lalwflg,
  lavoflg (logicals), ivflip (int, physparam default 1 = input surface->top), nlwbnd (NLWBND,
  16 with lalw1bd = .false.), ivolae (int32 [10, 4, 12] = [year, zone, month]), kyrsav, kmonsav.
  iaerflg, laswflg, lalwflg and lavoflg are set in rad_initialize_nmmb.f:142-148 from the iaer
  argument. The NAM exnam scripts in nam-gh/scripts do not set IAER (grep), so the NAM value is
  NOT CONFIRMED by any file read; it is whatever the caller passes.

Done rule (CONTRACT section 6): selftest() compiles the kernels and runs the setaer stages on a
full-size random state (954 x 835 x 60) and asserts no NaN/Inf. selftest does not include the
aer_property stage (other unit) and says so in its return value.
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_setaer"
AER_PROPERTY_MODULE = "namgpu.physics.rrtm.phys_rrtm_aer_property"

NF_AESW = 3
NF_AELW = 3
NBDSW = 14
NBDLW = 16

_SCRATCH: dict = {}


def _scratch(imax: int, nlay: int, nlp1: int):
    """alon, alat, dz, hz: allocated once per shape, reused every call (CONTRACT section 1)."""
    import cupy as cp

    key = (imax, nlay, nlp1)
    s = _SCRATCH.get(key)
    if s is None:
        s = {
            "alon": cp.zeros((imax,), dtype=cp.float32),
            "alat": cp.zeros((imax,), dtype=cp.float32),
            "dz": cp.zeros((nlay, imax), dtype=cp.float32),
            "hz": cp.zeros((nlp1, imax), dtype=cp.float32),
        }
        _SCRATCH[key] = s
    return s


def _zero(aerosw, aerolw, stream_scope):
    kern = kernels.get(UNIT, "phys_rrtm_setaer_zero")
    with stream_scope:
        kern((1024, 1, 1), (256, 1, 1), (
            aerosw, aerolw, np.uint64(aerosw.size), np.uint64(aerolw.size),
        ))


def _prep(prsi, prsl, tlay, qlay, xlon, xlat, s, imax, nlay, nlp1, ivflip, do_dz, stream_scope):
    kern = kernels.get(UNIT, "phys_rrtm_setaer_prep")
    with stream_scope:
        kern(kernels.grid_cols(imax, 1), kernels.BLOCK_COL, (
            prsi, prsl, tlay, qlay, xlon, xlat, s["alon"], s["alat"], s["dz"], s["hz"],
            np.int32(imax), np.int32(nlay), np.int32(nlp1), np.int32(ivflip), np.int32(1 if do_dz else 0),
        ))


def _volc(aerosw, aerolw, prsi, alat, ivolae, imax, nlay, ivflip, year1, month,
          do_sw, do_lw, nlwbnd1, stream_scope):
    kern = kernels.get(UNIT, "phys_rrtm_setaer_volc")
    with stream_scope:
        kern(kernels.grid_cols(imax, 1), kernels.BLOCK_COL, (
            aerosw, aerolw, prsi, alat, ivolae,
            np.int32(imax), np.int32(nlay), np.int32(ivflip), np.int32(year1), np.int32(month),
            np.int32(1 if do_sw else 0), np.int32(1 if do_lw else 0), np.int32(1 if nlwbnd1 else 0),
        ))


def _aer_property_run_call():
    """The aer_property stage entry. Named failure until that unit exposes run_call: its wrapper
    (phys_rrtm_aer_property.py) currently exposes launch() with a table dict, and the table binding
    (haer, prsref, sigref, rhlev, extrhi..., cmixg, denng, idxcg, kprfg) is not wired from aer_init /
    aer_update / clim_aerinit. iaerflg > 0 (NAM iaer=11) always reaches this stage."""
    mod = _aer_property_module()
    fn = getattr(mod, "run_call", None)
    if fn is None:
        raise RuntimeError(
            "phys-rrtm-setaer: aer_property stage (Fortran 2045) has no run_call in "
            f"{AER_PROPERTY_MODULE}; unit phys-rrtm-aer-property must bind its tables and expose run_call")
    return fn


def _aer_property_module():
    try:
        return importlib.import_module(AER_PROPERTY_MODULE)
    except ImportError as e:  # the dependency is a named gap, not a silent skip
        raise RuntimeError(
            f"phys-rrtm-setaer needs {AER_PROPERTY_MODULE} (unit phys-rrtm-aer-property, not in this unit): {e}"
        ) from e


def _check_ivolae(ivolae):
    """ivolae must be the device (CuPy) int32 array [10, 4, 12] = [year, zone, month]. A host numpy
    array is not a valid kernel argument, so it is refused here rather than failing inside the kernel."""
    import cupy as cp
    if not isinstance(ivolae, cp.ndarray) or ivolae.dtype != cp.int32 or tuple(ivolae.shape) != (10, 4, 12):
        raise TypeError(
            "phys-rrtm-setaer: ivolae must be a device CuPy int32 array of shape (10, 4, 12); got "
            f"{type(ivolae).__name__} dtype={getattr(ivolae, 'dtype', None)} "
            f"shape={getattr(ivolae, 'shape', None)}")


def run_call(prsi, prsl, prslk, tvly, rhlay, slmsk, tracer, xlon, xlat, imax, nlay, nlp1,
             lsswr, lslwr, tlay, qlay, aerosw, aerolw, *, iaerflg=0, laswflg=True, lalwflg=True,
             lavoflg=True, ivflip=1, nlwbnd=NBDLW, ivolae=None, kyrsav=1, kmonsav=1, stream=None,
             tables=None):
    """SETAER call. Arguments follow the Fortran call; the keyword arguments are the module state
    listed in the docstring (defaults are the physparam.f values). Outputs are written in place.
    One launch per stage; no host sync."""
    stream_scope = stream if stream is not None else contextlib.nullcontext()

    # Fortran 1871-1895: outputs are zeroed first, on every call.
    _zero(aerosw, aerolw, stream_scope)

    # Fortran 1962-1969: nothing else runs unless a radiation call is due and aerosols are on.
    if not (lsswr or lslwr) or iaerflg <= 0:
        return aerosw, aerolw

    laersw = bool(lsswr and laswflg)
    laerlw = bool(lslwr and lalwflg)
    s = _scratch(imax, nlay, nlp1)

    # Fortran 1956-2051: alon/alat always; dz/hz only in the laswflg-or-lalwflg block.
    do_dz = bool(laswflg or lalwflg)
    _prep(prsi, prsl, tlay, qlay, xlon, xlat, s, imax, nlay, nlp1, ivflip, do_dz, stream_scope)

    # Fortran 2045: aer_property, inside the same laswflg-or-lalwflg block.
    if do_dz:
        with stream_scope:
            _aer_property_run_call()(
                prsi=prsi, prsl=prsl, prslk=prslk, tvly=tvly, rhlay=rhlay, dz=s["dz"], hz=s["hz"],
                tracer=tracer, alon=s["alon"], alat=s["alat"], slmsk=slmsk, laersw=laersw,
                laerlw=laerlw, imax=imax, nlay=nlay, nlp1=nlp1, ivflip=ivflip,
                aerosw=aerosw, aerolw=aerolw, stream=stream, tables=tables,
                laswflg=laswflg, lalwflg=lalwflg, lalw1bd=(nlwbnd == 1),
            )

    # Fortran 2092-2352: volcanic aerosol, lavoflg branch, after aer_property.
    if lavoflg:
        # laddsw / laddlw: iaerflg == 100 uses lsswr/lslwr alone (Fortran 2097-2102).
        if iaerflg == 100:
            laddsw = bool(lsswr)
            laddlw = bool(lslwr)
        else:
            laddsw = bool(lsswr and laswflg)
            laddlw = bool(lslwr and lalwflg)
        if laddsw or laddlw:
            if ivolae is None:
                raise RuntimeError(
                    "phys-rrtm-setaer: lavoflg needs ivolae (unit aer_update / volc_update, not yet landed)"
                )
            _check_ivolae(ivolae)
            year1 = int(kyrsav) % 10 + 1          # Fortran i1 = mod(kyrsav, 10) + 1
            _volc(aerosw, aerolw, prsi, s["alat"], ivolae, imax, nlay, ivflip, year1, int(kmonsav),
                  laddsw, laddlw, nlwbnd == 1, stream_scope)
    return aerosw, aerolw


def run(ctx, *, lsswr: bool, lslwr: bool):
    """Driver entry. Inputs are the grrad-setup RAD_* arrays and XLON/XLAT/SLMSK, under the same
    uppercase CONTRACT names phys_rrtm_grrad_aerosols.grrad_aerosols reads. Module state comes from
    ctx.tables['aer_state'] (phys_rrtm_grrad_aerosols.aer_state) and ctx.tables['aer_ivolae'] (device
    int32 [10, 4, 12]). lsswr / lslwr are this step's radiation-call flags (driver schedule; the
    config has no such fields)."""
    aer = ctx.tables.get("aer_state")
    if aer is None:
        raise RuntimeError("phys_rrtm_setaer needs ctx.tables['aer_state'] (phys_rrtm_grrad_aerosols.aer_state)")
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk, lmp = lm + 1, lm + 2
    cols = im * jm
    st = ctx.st
    rs = lambda x, *shape: x.reshape(*shape)  # contiguous [.., JM, IM] -> [.., JM*IM]
    ntrac = int(st["RAD_TRACER1"].shape[0])
    faersw = ctx.tables["rrtm_faersw"]
    faerlw = ctx.tables["rrtm_faerlw"]
    return run_call(
        prsi=rs(st["RAD_PLVL"], lmp, cols), prsl=rs(st["RAD_PLYR"], lmk, cols),
        prslk=rs(st["RAD_PRSLK1"], lmk, cols), tvly=rs(st["RAD_TVLY"], lmk, cols),
        rhlay=rs(st["RAD_RHLY"], lmk, cols), slmsk=rs(st["SLMSK"], cols),
        tracer=rs(st["RAD_TRACER1"], ntrac, lmk, cols),
        xlon=rs(st["XLON"], cols), xlat=rs(st["XLAT"], cols),
        imax=cols, nlay=lmk, nlp1=lmp,
        lsswr=bool(lsswr), lslwr=bool(lslwr),
        tlay=rs(st["RAD_TLYR"], lmk, cols), qlay=rs(st["RAD_QLYR"], lmk, cols),
        aerosw=rs(faersw, NF_AESW, NBDSW, lmk, cols), aerolw=rs(faerlw, NF_AELW, NBDLW, lmk, cols),
        iaerflg=aer["iaerflg"], laswflg=aer["laswflg"], lalwflg=aer["lalwflg"],
        lavoflg=aer["lavoflg"], ivflip=aer["ivflip"], nlwbnd=aer["nlwbnd"],
        ivolae=ctx.tables.get("aer_ivolae"), kyrsav=aer["kyrsav"], kmonsav=aer["kmonsav"],
        stream=getattr(ctx, "stream", None),
    )


def run_path_check(im: int = 4, jm: int = 3, lm: int = 8):
    """The run(ctx) binding, once, on a small grid: uppercase RAD_*/SLMSK/XLON/XLAT state names,
    the [LMK,JM,IM] reshapes and the device ivolae. laswflg and lalwflg are off so the stages that
    need the aer_property unit (not landed) are not reached; the attribute lookups all run."""
    import cupy as cp
    from types import SimpleNamespace

    lmk, lmp = lm + 1, lm + 2
    rng = np.random.default_rng(7)

    class _St:
        def __init__(self, d):
            self._d = d

        def __getitem__(self, k):
            return self._d[k]

    k = np.arange(lmp, dtype=np.float64)
    pint = (1000.0 * (0.002 ** (k / lmk))).astype(np.float32)
    plvl = cp.asarray(np.broadcast_to(pint[:, None, None], (lmp, jm, im)).copy())
    plyr = 0.5 * (plvl[:-1] + plvl[1:])
    fields = {
        "RAD_PLVL": plvl, "RAD_PLYR": plyr,
        "RAD_PRSLK1": cp.asarray(rng.uniform(0.5, 1.0, (lmk, jm, im)).astype(np.float32)),
        "RAD_TVLY": cp.asarray(rng.uniform(230.0, 300.0, (lmk, jm, im)).astype(np.float32)),
        "RAD_RHLY": cp.asarray(rng.uniform(0.0, 1.0, (lmk, jm, im)).astype(np.float32)),
        "RAD_TLYR": cp.asarray(rng.uniform(230.0, 300.0, (lmk, jm, im)).astype(np.float32)),
        "RAD_QLYR": cp.asarray(rng.uniform(1.0e-4, 2.0e-2, (lmk, jm, im)).astype(np.float32)),
        "RAD_TRACER1": cp.asarray(rng.uniform(0.0, 2.0e-3, (2, lmk, jm, im)).astype(np.float32)),
        "SLMSK": cp.asarray(rng.integers(0, 2, (jm, im)).astype(np.float32)),
        "XLON": cp.asarray(rng.uniform(-np.pi, np.pi, (jm, im)).astype(np.float32)),
        "XLAT": cp.asarray(rng.uniform(-np.pi / 2, np.pi / 2, (jm, im)).astype(np.float32)),
    }
    ctx = SimpleNamespace(
        grid=SimpleNamespace(im=im, jm=jm, lm=lm), st=_St(fields), stream=None,
        tables={
            "aer_state": {"iaerflg": 11, "laswflg": False, "lalwflg": False, "lavoflg": False,
                          "ivflip": 1, "nlwbnd": NBDLW, "kyrsav": 1, "kmonsav": 6},
            "rrtm_faersw": cp.zeros((NF_AESW, NBDSW, lmk, jm, im), dtype=cp.float32),
            "rrtm_faerlw": cp.zeros((NF_AELW, NBDLW, lmk, jm, im), dtype=cp.float32),
            "aer_ivolae": cp.asarray(rng.integers(0, 11, (10, 4, 12)).astype(np.int32)),
        },
    )
    sw, lw = run(ctx, lsswr=True, lslwr=True)
    cp.cuda.Stream.null.synchronize()
    alon = _scratch(im * jm, lmk, lmp)["alon"].get()
    assert np.all(np.isfinite(alon)) and np.all(np.isfinite(sw.get())) and np.all(np.isfinite(lw.get()))
    return {"run_path": "ok", "shape": (im, jm, lm)}


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """Compile the kernels and run the setaer stages (zero, prep, volc) once on a full-size random
    state. The aer_property stage is another unit and is not run here. Asserts no NaN/Inf and
    non-negative optical depth."""
    import cupy as cp

    run_path = run_path_check()
    imax = im * jm
    nlp1 = nlay + 1
    rng = np.random.default_rng(seed)

    # Surface-to-top pressures in mb (ivflip=1): 1000 mb at the surface, ~2 mb at the top.
    k = np.arange(nlp1, dtype=np.float64)
    pint = 1000.0 * (0.002 ** (k / nlay))
    jitter = rng.uniform(0.995, 1.005, (nlp1, imax))
    prsi = (pint[:, None] * jitter).astype(np.float32)
    prsi = np.minimum.accumulate(prsi, axis=0)  # keep monotone, surface first
    prsl = (0.5 * (prsi[:-1] + prsi[1:])).astype(np.float32)
    tlay = rng.uniform(230.0, 300.0, (nlay, imax)).astype(np.float32)
    qlay = rng.uniform(1.0e-4, 2.0e-2, (nlay, imax)).astype(np.float32)
    xlon = rng.uniform(-np.pi, np.pi, (imax,)).astype(np.float32)
    xlat = rng.uniform(-np.pi / 2, np.pi / 2, (imax,)).astype(np.float32)
    ivolae = rng.integers(0, 11, (10, 4, 12), dtype=np.int32)

    prsi_d = cp.asarray(prsi)
    prsl_d = cp.asarray(prsl)
    tlay_d = cp.asarray(tlay)
    qlay_d = cp.asarray(qlay)
    xlon_d = cp.asarray(xlon)
    xlat_d = cp.asarray(xlat)
    ivolae_d = cp.asarray(ivolae)
    aerosw = cp.empty((NF_AESW, NBDSW, nlay, imax), dtype=cp.float32)
    aerolw = cp.empty((NF_AELW, NBDLW, nlay, imax), dtype=cp.float32)
    s = _scratch(imax, nlay, nlp1)

    scope = contextlib.nullcontext()
    _zero(aerosw, aerolw, scope)
    _prep(prsi_d, prsl_d, tlay_d, qlay_d, xlon_d, xlat_d, s, imax, nlay, nlp1, 1, True, scope)
    _volc(aerosw, aerolw, prsi_d, s["alat"], ivolae_d, imax, nlay, 1, 1, 6,
          True, True, False, scope)
    cp.cuda.Stream.null.synchronize()

    sw = aerosw.get()
    lw = aerolw.get()
    dz = s["dz"].get()
    hz = s["hz"].get()
    for name, arr in (("aerosw", sw), ("aerolw", lw), ("dz", dz), ("hz", hz)):
        assert np.all(np.isfinite(arr)), f"NaN/Inf in setaer {name}"
    assert sw[0].min() >= 0.0 and lw[0].min() >= 0.0, "negative optical depth"
    return {
        "aerosw_tau_mean": float(sw[0].mean()),
        "aerolw_tau_mean": float(lw[0].mean()),
        "dz_mean": float(dz.mean()),
        "run_path": run_path["run_path"],
        "stages": "zero+prep+volc (aer_property stage not included: other unit)",
    }


if __name__ == "__main__":
    print(selftest())
