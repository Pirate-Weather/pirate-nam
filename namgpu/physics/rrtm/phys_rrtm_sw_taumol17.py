# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-taumol17: NMMB RRTM shortwave band 17 optical depths (taumol group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:3495-3627, subroutine taumol17 (internal
to taumol). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the header is kept in the kernel file
(namgpu/kernels/phys_rrtm_sw_taumol17.cu). Kernel: phys_rrtm_sw_taumol17, one thread per column, one launch.

Band 17 (3250-4000 cm-1): H2O and CO2 binary-species interpolation below the tropopause (laytrop) and above
it, the H2O self continuum (lower layers only) and the H2O foreign continuum, plus Rayleigh scattering for
every layer. The band has NG17 = 12 g-points at NS17 = 6. Outputs are the 12-row band slab; the caller places
it at 0-based g-points 6..17 of its taug and taur (NGPT = 112).

Tables: module_radsw_kgb17 / module_radsw_kgb17_nmmb (phys/radsw_datatb.f:3287 and radsw_datatb_nmmb.f:2533;
numerically identical), extracted by tools/extract-rrtm-sw-kgb17.py to namgpu/data/rrtm_sw/kgb17.npz.
strrat(17) = 0.364641 (radsw_datatb_nmmb.f:1108, strrat has bounds NBLOW:NBHGH = 16:29, so strrat(17) is the
second value of its data statement).

Reads ctx.tables[IN_KEY] (dict of device arrays in CONTRACT layout; Fortran 1-based index values, the keys
phys-rrtm-sw-setcoef and the SW taumol driver write):
  laytrop       int32   [JM, IM]
  jp, jt, jt1   int32   [NLAY, JM, IM]   (setcoef outputs; jt/jt1 1..4, jp 1..58)
  indself       int32   [NLAY, JM, IM]   (0 above the self-continuum range, 1..9 below)
  indfor        int32   [NLAY, JM, IM]   (1..3)
  colamt        float32 [6, NLAY, JM, IM]   colamt(k, m); only m = 1 (H2O) and 2 (CO2) are read
  colmol        float32 [NLAY, JM, IM]
  fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac   float32 [NLAY, JM, IM]
Writes ctx.tables[OUT_KEY] = {"taug17": [12, NLAY, JM, IM], "taur17": [12, NLAY, JM, IM]} (float32,
allocated once and reused). No host sync, no per-step allocation after the first call.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_taumol17"
KERNEL = "phys_rrtm_sw_taumol17"
IN_KEY = "rrtm_sw_taumol17_in"
OUT_KEY = "rrtm_sw_taugb17_out"
KGB_KEY = "rrtm_sw_kgb17"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "kgb17.npz"

NG17 = 12          # radsw_param_nmmb.f:161, NG17
NS17 = 6           # radsw_param_nmmb.f, NS17 = NS16 + NG16 (slab offset in taug / taur)
MAXGAS = 6         # module_radsw_parameters_nmmb maxgas
STRRAT17 = 0.364641  # strrat(17), radsw_datatb_nmmb.f:1108 data statement (second value, bounds 16:29)


def load_kgb17():
    """Band-17 k-tables as device float32 arrays, uploaded once (host read at init only)."""
    import cupy as cp

    with np.load(NPZ) as d:
        return {
            "absa": cp.asarray(d["absa"].astype(np.float32)),        # [585, 12]  absa(ind, g) -> [ind-1, g-1]
            "absb": cp.asarray(d["absb"].astype(np.float32)),        # [1175, 12]
            "selfref": cp.asarray(d["selfref"].astype(np.float32)),  # [10, 12]
            "forref": cp.asarray(d["forref"].astype(np.float32)),    # [4, 12]
        }


def alloc_outputs(nlay: int, jm: int, im: int):
    """taug17 and taur17 slabs [NG17, NLAY, JM, IM], allocated once and reused."""
    import cupy as cp

    return {"taug17": cp.zeros((NG17, nlay, jm, im), dtype=cp.float32),
            "taur17": cp.zeros((NG17, nlay, jm, im), dtype=cp.float32)}


def _check_shapes(inp: dict, nlay: int, jm: int, im: int):
    """Shape and dtype checks. The kernel reads the index arrays as int32 and the rest as float32; a wrong
    dtype would be read as garbage with no error, so it is refused here (review gap, phys-rrtm-sw-taumol17)."""
    import cupy as cp

    int_names = ("laytrop", "jp", "jt", "jt1", "indself", "indfor")
    float_names = ("colamt", "colmol", "fac00", "fac01", "fac10", "fac11",
                   "selffac", "selffrac", "forfac", "forfrac")
    shapes = {"laytrop": (jm, im)}
    for name in int_names[1:] + float_names:
        shapes[name] = (MAXGAS, nlay, jm, im) if name == "colamt" else (nlay, jm, im)
    for name, shape in shapes.items():
        arr = inp[name]
        if not isinstance(arr, cp.ndarray):
            raise TypeError(f"taumol17: {name} must be a CuPy array, got {type(arr).__name__}")
        if tuple(arr.shape) != shape:
            raise ValueError(f"taumol17: {name} shape {tuple(arr.shape)} != {shape}")
        want = cp.int32 if name in int_names else cp.float32
        if arr.dtype != cp.dtype(want):
            raise TypeError(f"taumol17: {name} dtype {arr.dtype} != {cp.dtype(want)} (kernel reads it as that type)")


def taumol17(inp: dict, kgb: dict, out: dict, nlay: int, jm: int, im: int, stream=None):
    """One launch over every (i, j) column. Arrays are device arrays in CONTRACT layout."""
    _check_shapes(inp, nlay, jm, im)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    ncol = im * jm
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(nlay), np.int32(ncol), np.float32(STRRAT17),
            inp["laytrop"], inp["jp"], inp["jt"], inp["jt1"], inp["indself"], inp["indfor"],
            inp["colamt"], inp["colmol"],
            inp["fac00"], inp["fac01"], inp["fac10"], inp["fac11"],
            inp["selffac"], inp["selffrac"], inp["forfac"], inp["forfrac"],
            kgb["absa"], kgb["absb"], kgb["selfref"], kgb["forref"],
            out["taug17"], out["taur17"],
        ))
    return out


def run(ctx):
    """Band-17 optical depths for the current SW inputs; returns ctx.tables[OUT_KEY]."""
    if ctx.tables is None:
        ctx.tables = {}
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_sw_taumol17 needs ctx.tables[{IN_KEY!r}] from the SW taumol driver")
    nlay, jm, im = inp["jp"].shape
    kgb = ctx.tables.get(KGB_KEY)
    if kgb is None:
        kgb = load_kgb17()
        ctx.tables[KGB_KEY] = kgb
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["taug17"].shape != (NG17, nlay, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return taumol17(inp, kgb, out, nlay, jm, im, stream=ctx.stream)


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """One launch on a full-size plausible state (954 x 835 x 60). Asserts no NaN/Inf in the outputs and
    the index ranges the Fortran produces. Values are random but in the ranges setcoef produces."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    f32 = np.float32
    i32 = np.int32
    lt = rng.integers(20, 50, (jm, im)).astype(i32)
    jp = rng.integers(1, 59, (nlay, jm, im)).astype(i32)
    jt = rng.integers(1, 5, (nlay, jm, im)).astype(i32)
    jt1 = rng.integers(1, 5, (nlay, jm, im)).astype(i32)
    indself = rng.integers(0, 10, (nlay, jm, im)).astype(i32)
    indfor = rng.integers(1, 4, (nlay, jm, im)).astype(i32)
    colamt = rng.uniform(0.0, 1.0e24, (MAXGAS, nlay, jm, im)).astype(f32)
    colmol = rng.uniform(1.0e20, 1.0e25, (nlay, jm, im)).astype(f32)
    frac = lambda: rng.uniform(0.0, 1.0, (nlay, jm, im)).astype(f32)  # noqa: E731
    fac00, fac01, fac10, fac11 = frac(), frac(), frac(), frac()
    selffac, selffrac, forfac, forfrac = frac(), frac(), frac(), frac()

    inp = {
        "laytrop": cp.asarray(lt), "jp": cp.asarray(jp), "jt": cp.asarray(jt), "jt1": cp.asarray(jt1),
        "indself": cp.asarray(indself), "indfor": cp.asarray(indfor), "colamt": cp.asarray(colamt),
        "colmol": cp.asarray(colmol),
        "fac00": cp.asarray(fac00), "fac01": cp.asarray(fac01), "fac10": cp.asarray(fac10),
        "fac11": cp.asarray(fac11), "selffac": cp.asarray(selffac), "selffrac": cp.asarray(selffrac),
        "forfac": cp.asarray(forfac), "forfrac": cp.asarray(forfrac),
    }
    kgb = load_kgb17()
    out = alloc_outputs(nlay, jm, im)
    taumol17(inp, kgb, out, nlay, jm, im)
    cp.cuda.Stream.null.synchronize()

    for k, a in out.items():
        assert bool(cp.isfinite(a).all()), f"NaN/Inf in sw taumol17 output {k}"
    return {k: float(out[k].mean()) for k in out}
