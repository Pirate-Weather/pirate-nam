# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-setcoef: SETCOEF, per-layer interpolation coefficients for the NMMB RRTM longwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_main_nmmb.f:1863-2121 (SUBROUTINE setcoef).
NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the lw-rrtm3 header of that file is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_lw_setcoef.cu (rrtm_lw_setcoef, one thread per (i, j) column, one launch).

What it computes, per column and layer k: the Planck functions pklay/pklev at the layer and level
temperatures (plus the surface level k = 0), the reference pressure index jp and temperature indices
jt/jt1, the four interpolation factors fac00/fac01/fac10/fac11, the water vapour self and foreign
continuum factors, the minor gas factors, the binary species ratios rfrate, and laytrop (count of
layers with ln(p) > 4.56).

VERTICAL ORDER: SURFACE FIRST. Every per-layer input and output here is in radlw's internal order,
index 0 = the lowest layer and tz plane 0 = the surface level, as in radlw_main_nmmb.f:660 ("vertical
index of internal array is from surface to top"). This is NOT the NMMB top-first order of CONTRACT
section 1. The flip is done once, in lwrad_prep (namgpu/kernels/phys_rrtm_lw_lwrad.cu, kin = nlay-1-k0),
which writes the lwrad_* scratch that the seam passes here. This unit does no flip of its own.

Inputs (device float32, Fortran (i,j,l) -> [l, j, i], surface-first layer index):
  pavel, tavel, h2ovmr, coldry, colbrd  [nlay, JM, IM]
  tz                                    [nlay+1, JM, IM] (plane 0 = tz(0), the surface level)
  stemp                                 [JM, IM]
  colamt                                [MAXGAS=6, nlay, JM, IM]
Tables (device float32, read from ctx.tables; each table is the one the NMMB build links):
  totplnk [NBANDS=16, NPLNK=181]  module_radlw_avplank, radlw_datatb.f:58-728 (phys-rrtm-lw-table-avplank)
  delwave [16]                    radlw_param.f:133 (module_radlw_parameters), see DELWAVE below
  preflog [59], tref [59]         module_radlw_ref, radlw_datatb.f:761-790 (rrtm_lw_ref.load_preflog_tref)
  chi_mls [59, 7]                 module_radlw_ref, radlw_datatb.f:792-905 (rrtm_lw_ref.load_chi_mls)
Outputs (device, allocated once, see alloc_outputs): see the kernel header for shapes.

PRECISION: float32 throughout, per the kfpt = single ruling. The NAM configure file that the makefiles
include (src/conf/configure.nems) does not define SINGLE_PREC, so machine.f selects kind_phys = 8 there.
The compile flags of the pinned reference build are not recorded in the tree. Open: project lead to rule which
precision the reference build uses. Until then this unit is single precision and says so here.

Caller: the LW driver (phys-rrtm-lw-lwrad, radlw_main_nmmb.f:957) calls setcoef once per row chunk, with
the lwrad_* scratch as its inputs. The LW driver fills ctx.tables["rrtm_lw_setcoef_in"] (dict of the
inputs above, plus "nlay" implied by the shapes) and the tables under "rrtm_lw_setcoef_tables"; this unit
fills ctx.tables["rrtm_lw_setcoef_out"].
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.rrtm_lw_ref import load_chi_mls, load_preflog_tref

UNIT = "phys_rrtm_lw_setcoef"
KERNEL = "rrtm_lw_setcoef"
NBANDS = 16
NPLNK = 181
NREF = 59
NGAS_REF = 7
MAXGAS = 6
NRATES = 6

# radlw_param.f:133, data delwave (module_radlw_parameters, the module radlw_main_nmmb.f:246 uses).
DELWAVE = np.array([340., 150., 130., 70., 120., 160., 100., 100.,
                    210., 90., 320., 280., 170., 130., 220., 650.], dtype=np.float32)

IN_KEY = "rrtm_lw_setcoef_in"
TAB_KEY = "rrtm_lw_setcoef_tables"
OUT_KEY = "rrtm_lw_setcoef_out"


def alloc_outputs(nlay: int, jm: int, im: int):
    """Scratch and outputs, allocated once. rfrate entries a branch does not set stay zero."""
    import cupy as cp

    def f(*shape):
        return cp.zeros(shape, dtype=cp.float32)

    def i(*shape):
        return cp.zeros(shape, dtype=cp.int32)

    return {
        "laytrop": i(jm, im),
        "pklay": f(nlay + 1, NBANDS, jm, im),
        "pklev": f(nlay + 1, NBANDS, jm, im),
        "jp": i(nlay, jm, im), "jt": i(nlay, jm, im), "jt1": i(nlay, jm, im),
        "rfrate": f(2, NRATES, nlay, jm, im),
        "fac00": f(nlay, jm, im), "fac01": f(nlay, jm, im),
        "fac10": f(nlay, jm, im), "fac11": f(nlay, jm, im),
        "selffac": f(nlay, jm, im), "selffrac": f(nlay, jm, im),
        "indself": i(nlay, jm, im),
        "forfac": f(nlay, jm, im), "forfrac": f(nlay, jm, im),
        "indfor": i(nlay, jm, im),
        "minorfrac": f(nlay, jm, im), "scaleminor": f(nlay, jm, im),
        "scaleminorn2": f(nlay, jm, im),
        "indminor": i(nlay, jm, im),
    }


def _check_shapes(inp: dict, tab: dict):
    nlay, jm, im = inp["pavel"].shape
    for name in ("tavel", "h2ovmr", "coldry", "colbrd"):
        if inp[name].shape != (nlay, jm, im):
            raise ValueError(f"setcoef: {name} shape {inp[name].shape} != {(nlay, jm, im)}")
    if inp["tz"].shape != (nlay + 1, jm, im):
        raise ValueError(f"setcoef: tz shape {inp['tz'].shape} != {(nlay + 1, jm, im)}")
    if inp["stemp"].shape != (jm, im):
        raise ValueError(f"setcoef: stemp shape {inp['stemp'].shape} != {(jm, im)}")
    if inp["colamt"].shape != (MAXGAS, nlay, jm, im):
        raise ValueError(f"setcoef: colamt shape {inp['colamt'].shape} != {(MAXGAS, nlay, jm, im)}")
    if tab["totplnk"].shape != (NBANDS, NPLNK):
        raise ValueError(f"setcoef: totplnk shape {tab['totplnk'].shape}")
    if tab["delwave"].shape != (NBANDS,):
        raise ValueError(f"setcoef: delwave shape {tab['delwave'].shape}")
    if tab["preflog"].shape != (NREF,) or tab["tref"].shape != (NREF,):
        raise ValueError("setcoef: preflog/tref must have 59 entries")
    if tab["chi_mls"].shape != (NREF, NGAS_REF):
        raise ValueError(f"setcoef: chi_mls shape {tab['chi_mls'].shape} != {(NREF, NGAS_REF)}")
    return nlay, jm, im


def setcoef(inp: dict, tab: dict, out: dict, stream=None):
    """One launch over every (i, j) column. All arrays are device float32/int32, surface-first layers."""
    nlay, jm, im = _check_shapes(inp, tab)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            inp["pavel"], inp["tavel"], inp["tz"], inp["stemp"], inp["h2ovmr"], inp["colamt"],
            inp["coldry"], inp["colbrd"],
            tab["totplnk"], tab["delwave"], tab["preflog"], tab["tref"], tab["chi_mls"],
            np.int32(nlay), np.int32(im), np.int32(jm),
            out["laytrop"], out["pklay"], out["pklev"],
            out["jp"], out["jt"], out["jt1"], out["rfrate"],
            out["fac00"], out["fac01"], out["fac10"], out["fac11"],
            out["selffac"], out["selffrac"], out["indself"],
            out["forfac"], out["forfrac"], out["indfor"],
            out["minorfrac"], out["scaleminor"], out["scaleminorn2"], out["indminor"],
        ))
    return out


def run(ctx):
    """Driver entry. The LW driver (phys-rrtm-lw-lwrad) puts the inputs and tables in ctx.tables
    before calling this; the outputs are allocated once and reused on every call."""
    inp = ctx.tables.get(IN_KEY)
    if inp is None:
        raise RuntimeError(f"phys_rrtm_lw_setcoef needs ctx.tables[{IN_KEY!r}] from the LW driver (phys-rrtm-lw-lwrad)")
    tab = ctx.tables.get(TAB_KEY)
    if tab is None:
        raise RuntimeError(f"phys_rrtm_lw_setcoef needs ctx.tables[{TAB_KEY!r}] from the LW table units")
    nlay, jm, im = inp["pavel"].shape
    out = ctx.tables.get(OUT_KEY)
    if out is None or out["pklay"].shape != (nlay + 1, NBANDS, jm, im):
        out = alloc_outputs(nlay, jm, im)
        ctx.tables[OUT_KEY] = out
    return setcoef(inp, tab, out, stream=ctx.stream)


def load_tables():
    """The four setcoef tables, each from the file the NMMB build links, as device float32 arrays."""
    import cupy as cp

    from namgpu.physics.rrtm.phys_rrtm_lw_table_avplank import load_host as load_totplnk

    preflog, tref = load_preflog_tref()
    return {"totplnk": cp.asarray(load_totplnk()), "delwave": cp.asarray(DELWAVE),
            "preflog": cp.asarray(preflog), "tref": cp.asarray(tref),
            "chi_mls": cp.asarray(load_chi_mls())}


def selftest(im: int = 954, jm: int = 835, nlay: int = 60, seed: int = 1):
    """One launch on a full-size plausible state (954 x 835 x 60 layers), surface-first. Asserts no NaN/Inf,
    the surface-first order of the inputs, and the index ranges the Fortran clamps to. All four tables are
    the real ones (load_tables); only the atmospheric state is random."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    f32 = np.float32
    # Pressure ladder surface first: index 0 is ~1000 hPa (lowest layer), index nlay-1 the top.
    pav = np.geomspace(1000.0, 2.0, nlay).astype(f32)
    assert pav[0] > pav[-1], "selftest pressure must be surface first"
    pav = np.broadcast_to(pav[:, None, None], (nlay, jm, im)).astype(f32)
    pav = (pav * rng.uniform(0.98, 1.02, (nlay, jm, im)).astype(f32)).astype(f32)
    tav = rng.uniform(190.0, 310.0, (nlay, jm, im)).astype(f32)
    tz = rng.uniform(190.0, 310.0, (nlay + 1, jm, im)).astype(f32)
    stemp = rng.uniform(250.0, 312.0, (jm, im)).astype(f32)
    h2o = rng.uniform(1.0e-6, 2.0e-2, (nlay, jm, im)).astype(f32)
    colamt = rng.uniform(1.0e18, 1.0e24, (MAXGAS, nlay, jm, im)).astype(f32)
    coldry = rng.uniform(1.0e20, 1.0e24, (nlay, jm, im)).astype(f32)
    colbrd = rng.uniform(1.0e20, 1.0e24, (nlay, jm, im)).astype(f32)

    inp = {"pavel": cp.asarray(pav), "tavel": cp.asarray(tav), "tz": cp.asarray(tz),
           "stemp": cp.asarray(stemp), "h2ovmr": cp.asarray(h2o), "colamt": cp.asarray(colamt),
           "coldry": cp.asarray(coldry), "colbrd": cp.asarray(colbrd)}
    tab = load_tables()
    out = alloc_outputs(nlay, jm, im)
    setcoef(inp, tab, out)
    cp.cuda.Stream.null.synchronize()

    floats = [k for k, a in out.items() if a.dtype == cp.float32]
    for k in floats:
        assert bool(cp.isfinite(out[k]).all()), f"NaN/Inf in setcoef output {k}"
    jp = out["jp"].get()
    assert jp.min() >= 1 and jp.max() <= 58, (jp.min(), jp.max())
    assert out["jt"].get().min() >= 1 and out["jt"].get().max() <= 4
    assert out["indself"].get().min() >= 0 and out["indself"].get().max() <= 9
    assert out["indminor"].get().min() >= 1 and out["indminor"].get().max() <= 18
    return {k: float(out[k].mean()) for k in floats}
