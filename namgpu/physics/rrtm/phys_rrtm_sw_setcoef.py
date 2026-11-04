# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-setcoef: SETCOEF, per-layer interpolation coefficients for the NMMB RRTM shortwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_main_nmmb.f:1989-2139 (SUBROUTINE setcoef).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the sw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_sw_setcoef.cu. Entries: rrtm_sw_setcoef (full domain) and
rrtm_sw_setcoef_chunk (the driver's chunk, used by run_chunk).

What it computes, per column and layer k: the reference pressure index jp (clamped 1..58) and the
temperature indices jt / jt1 (clamped 1..4), the four interpolation factors fac00/fac01/fac10/fac11,
the forfac scale factor and forfrac/indfor (foreign continuum), selffac/selffrac/indself (self
continuum; zero in the upper atmosphere), and laytrop (the last layer with ln(p) > 4.56, or nlay).

Pressure unit: pavel is the layer pressure in mb (hPa). The Fortran caller passes plyr in mb
(radsw_main_nmmb.f:392 and :523). tavel is in K; h2ovmr is the water vapour volume mixing ratio.

Reference tables: preflog [59] and tref [59] from phys_rrtm_sw_table_ref (module_radsw_ref DATA
statements, radsw_datatb.f). Loaded once into ctx.tables["rrtm_sw_setcoef_tables"].

Driver contract (phys-rrtm-sw-swrad, _seam): run_chunk(ctx, ch, cstart, nc) is called once per chunk of
day columns. ch is the chunk scratch: pavel, tavel, h2ovmr [nc, nlay] (filled by swrad_prep in the
Fortran internal layer order); chunk column t is day column cstart + t. The outputs laytrop [nc] and
jp, jt, jt1, indself, indfor, fac00..fac11, selffac, selffrac, forfac, forfrac [nc, nlay] are written into
ch. ctx.st["SW_NDAY"] (device int32[1]) gates the live columns, read on the device.

The Fortran caller invokes setcoef once per day column j1 inside DO j1 = 1,NDAY (radsw_main_nmmb.f:1005).
The chunk path does all of its day columns in one launch.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_sw_table_ref as tables_unit

UNIT = "phys_rrtm_sw_setcoef"
KERNEL = "rrtm_sw_setcoef"
KERNEL_CHUNK = "rrtm_sw_setcoef_chunk"
NREF = 59
TAB_KEY = tables_unit.TAB_KEY
OUT_KEYS = ("jp", "jt", "jt1", "indself", "indfor", "fac00", "fac01", "fac10", "fac11",
            "selffac", "selffrac", "forfac", "forfrac")


def alloc_outputs(nlay: int, jm: int, im: int):
    """Full-domain outputs [nlay, JM, IM], allocated once and reused on every call."""
    import cupy as cp

    def f(*shape):
        return cp.zeros(shape, dtype=cp.float32)

    def i(*shape):
        return cp.zeros(shape, dtype=cp.int32)

    return {
        "laytrop": i(jm, im),
        "jp": i(nlay, jm, im), "jt": i(nlay, jm, im), "jt1": i(nlay, jm, im),
        "fac00": f(nlay, jm, im), "fac01": f(nlay, jm, im),
        "fac10": f(nlay, jm, im), "fac11": f(nlay, jm, im),
        "selffac": f(nlay, jm, im), "selffrac": f(nlay, jm, im),
        "indself": i(nlay, jm, im),
        "forfac": f(nlay, jm, im), "forfrac": f(nlay, jm, im),
        "indfor": i(nlay, jm, im),
    }


def _launch(kern_name: str, args: tuple, grid_n: int, stream):
    kern = kernels.get(UNIT, kern_name)
    block = kernels.BLOCK_COL
    grid = ((grid_n + block[0] - 1) // block[0], 1, 1)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, args)


def setcoef(inp: dict, tab: dict, out: dict, stream=None):
    """Full domain: one launch over every (i, j) column. inp: pavel (mb), tavel, h2ovmr [nlay, JM, IM]."""
    nlay, jm, im = inp["pavel"].shape
    for name in ("tavel", "h2ovmr"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"sw setcoef: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if tab["preflog"].shape != (NREF,) or tab["tref"].shape != (NREF,):
        raise ValueError("sw setcoef: preflog/tref must have 59 entries")
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["pavel"], inp["tavel"], inp["h2ovmr"],
            tab["preflog"], tab["tref"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["laytrop"], out["jp"], out["jt"], out["jt1"],
            out["fac00"], out["fac01"], out["fac10"], out["fac11"],
            out["selffac"], out["selffrac"], out["indself"],
            out["forfac"], out["forfrac"], out["indfor"],
        ))
    return out


def run_chunk(ctx, ch: dict, cstart: int, nc: int):
    """Driver seam entry: setcoef for one chunk of day columns. Reads ch pavel, tavel, h2ovmr [nc, nlay]
    and writes laytrop [nc] and the per-layer outputs [nc, nlay] into ch. Live columns are
    cstart + t < SW_NDAY (device), so the chunk is safe when the last chunk runs past NDAY."""
    tab = tables_unit.run(ctx)
    ndayp = ctx.st["SW_NDAY"]
    pav = ch["pavel"]
    if pav.shape[0] != nc or ch["tavel"].shape != pav.shape or ch["h2ovmr"].shape != pav.shape:
        raise ValueError(f"sw setcoef: chunk inputs {pav.shape} do not match nc={nc}")
    nlay = int(pav.shape[1])
    args = (
        pav, ch["tavel"], ch["h2ovmr"], tab["preflog"], tab["tref"],
        ndayp, np.int32(cstart), np.int32(nc), np.int32(nlay),
        ch["laytrop"], ch["jp"], ch["jt"], ch["jt1"],
        ch["fac00"], ch["fac01"], ch["fac10"], ch["fac11"],
        ch["selffac"], ch["selffrac"], ch["indself"],
        ch["forfac"], ch["forfrac"], ch["indfor"],
    )
    _launch(KERNEL_CHUNK, args, nc, getattr(ctx, "stream", None))
    return ch


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1, nc: int = 4096):
    """Full-size plausible state (954 x 835 columns, 60 layers). One full-domain launch, then the driver
    chunk path over every column with the SW_NDAY gate open. Asserts no NaN/Inf and the index ranges
    the Fortran clamps to. Tables come from the module_radsw_ref unit, not stand-ins."""
    import cupy as cp
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    f32 = np.float32
    # Pressures in mb: log-uniform 0.01 to 1000 mb. Temperatures 190-310 K. Water vapour mixing ratio.
    pav = np.exp(rng.uniform(np.log(0.01), np.log(1000.0), (nlay, jm, im))).astype(f32)
    tav = rng.uniform(190.0, 310.0, (nlay, jm, im)).astype(f32)
    h2o = rng.uniform(1.0e-6, 2.0e-2, (nlay, jm, im)).astype(f32)

    ctx = SimpleNamespace(tables={}, st={}, stream=None)
    tab = tables_unit.run(ctx)
    inp = {"pavel": cp.asarray(pav), "tavel": cp.asarray(tav), "h2ovmr": cp.asarray(h2o)}
    out = alloc_outputs(nlay, jm, im)
    setcoef(inp, tab, out)
    cp.cuda.Stream.null.synchronize()
    _check(out, nlay, full=True)

    # Driver chunk path over every column (all day columns live).
    ncol = jm * im
    pc = cp.asarray(pav.reshape(nlay, ncol).T.copy())
    tc = cp.asarray(tav.reshape(nlay, ncol).T.copy())
    hc = cp.asarray(h2o.reshape(nlay, ncol).T.copy())
    ctx.st["SW_NDAY"] = cp.asarray([ncol], dtype=cp.int32)
    for c0 in range(0, ncol, nc):
        c1 = min(c0 + nc, ncol)
        ch = {"pavel": pc[c0:c1], "tavel": tc[c0:c1], "h2ovmr": hc[c0:c1]}
        ch.update(_alloc_chunk(c1 - c0, nlay))
        run_chunk(ctx, ch, c0, c1 - c0)
        cp.cuda.Stream.null.synchronize()
        _check(ch, nlay, full=False)
    return {k: float(out[k].mean()) for k in out if out[k].dtype == cp.float32}


def _alloc_chunk(nc: int, nlay: int) -> dict:
    import cupy as cp

    f = lambda *s: cp.zeros((nc,) + s, dtype=cp.float32)  # noqa: E731
    i = lambda *s: cp.zeros((nc,) + s, dtype=cp.int32)    # noqa: E731
    return {
        "laytrop": i(),
        "jp": i(nlay), "jt": i(nlay), "jt1": i(nlay), "indself": i(nlay), "indfor": i(nlay),
        "fac00": f(nlay), "fac01": f(nlay), "fac10": f(nlay), "fac11": f(nlay),
        "selffac": f(nlay), "selffrac": f(nlay), "forfac": f(nlay), "forfrac": f(nlay),
    }


def _check(out: dict, nlay: int, full: bool):
    import cupy as cp

    for k, a in out.items():
        if a.dtype == cp.float32:
            assert bool(cp.isfinite(a).all()), f"NaN/Inf in sw setcoef output {k} (full={full})"
    assert out["jp"].min() >= 1 and out["jp"].max() <= 58
    assert out["jt"].min() >= 1 and out["jt"].max() <= 4
    assert out["jt1"].min() >= 1 and out["jt1"].max() <= 4
    assert out["indself"].min() >= 0 and out["indself"].max() <= 9
    assert out["indfor"].min() >= 1 and out["indfor"].max() <= 3
    lt = out["laytrop"]
    assert lt.min() >= 1 and lt.max() <= nlay
