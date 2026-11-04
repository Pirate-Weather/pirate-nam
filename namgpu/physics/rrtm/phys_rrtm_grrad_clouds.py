"""phys-rrtm-grrad-clouds: GRRAD_NMMB cloud-property dispatch, clw and IALB=2 surface albedo (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:1556-1606 (clw and the np3d dispatch),
1607-1717 (cloud branches), 1722-1794 (albedo). NOAA US government work; the NOAA header of
grrad_nmmb.f is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_grrad_clouds.cu (one launch per routine section).

Scope: clw, the cloud dispatch and the albedo block of grrad_nmmb. Dependencies called from here, not
ported in this unit: progcld1 (np3d 4), progcld2 (np3d 3, phys-rrtm-progcld2), progcld8 (np3d 8),
diagcld1 (diagnostic scheme). They are called as dep(name, ctx, inputs); inputs is the named
contract built by _contract() below (every Fortran argument of the call, bound to a state array or
a driver table entry, shape-checked here). A dependency returns a dict with clouds
[NF_CLDS, LMK, JM, IM] in this unit's local layout (kd / lya / lyb as below) and may return
cldsa, mtopa, mbota. With no dep for a branch that needs one, run() raises instead of skipping it.

Inputs are read by name (CuPy float32, Fortran (i,j,l) -> [l, j, i]; LM = model levels, LMK = LM+LTP,
LMP = LMK+1; zero-based local levels):
  state (ctx.st, from phys-rrtm-grrad-setup, RAD_* names, LMK/LMP levels; the setup kernel already
         writes the extra top layer for tracer1, cldfk1, gcice/grain/grime, qlyr, tvly
         (phys_rrtm_grrad_setup.cu:201-210)):
         RAD_TRACER1 [NTRAC, LMK, JM, IM], NTRAC = NUM_TRACERS_TOTAL of the microphysics (thompson 10)
  state (ctx.st): CLDFRA [LM,JM,IM] (Fortran cldf), SM [JM,IM], GLAT, GLON [JM,IM]
  driver tables (ctx.tables["rrtm_driver_call"], phys-rrtm-driver-call): TAUCLOUDS [LM,JM,IM]
         (Fortran tauclouds), VVEL [LM,JM,IM] (Fortran vvl), CV, CVT, CVB, SFCALBEDO (the caller's
         SFCALBEDO, overwritten by the IALB=2 block), SLMSK, FLGMIN_L [JM,IM]
  tables (ctx.tables["coszen"]): coszen_loc, the daytime-average cosine from phys-rrtm-coszmn; it must be
         produced earlier in the same step.
Outputs (ctx.st, allocated once with st.add, names are the consumers' names):
  RAD_CLOUDS [NF_CLDS, LMK, JM, IM]   (read by phys-rrtm-grrad-sw as RAD_CLOUDS)
  RAD_CLW [LMK,JM,IM], RAD_VVEL [LMK,JM,IM], RAD_CVT1, RAD_CVB1 [JM,IM],
  RAD_QC2D, RAD_QI2D, RAD_QS2D, RAD_NI2D [LMK,JM,IM] (np3d 8)
  ALVSF, ALNSF, ALVWF, ALNWF, FACSF, FACWF [JM,IM] (the names phys-rrtm-setalb reads)
  RAD_SFCALBEDO, RAD_SMX, RAD_CZMEAN [JM,IM]
"""
from __future__ import annotations

import os
import re

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_grrad_clouds"
NF_CLDS = 9          # radiation_clouds_nmmb.f: NF_CLDS
NF_ALBD = 4          # radiation_surface_nmmb.f: NF_ALBD (not used by this unit's arrays)
LTP = 1              # grrad_nmmb.f: LTP = 1 (extra top layer)
EPSQ = 1.0e-12       # grrad_nmmb.f: EPSQ
NCLD = 1             # module_RA_RRTM.F90:220 NCLDX = 1 (grrad receives it as ncld)
NTRAC_MIN = 9        # Ferrier/Thompson species index 9 (ni, Fortran tracer 9); thompson NUM_TRACERS_TOTAL = 10
DRIVER_TABLE = "rrtm_driver_call"
PREP_TABLE = "rrtm_prep:out"

# Fortran argument -> state name, for the dependency contract. Shapes are checked in _contract().
_STATE_OF = {
    "plyr": "RAD_PLYR", "plvl": "RAD_PLVL", "tlyr": "RAD_TLYR", "tvly": "RAD_TVLY",
    "qlyr": "RAD_QLYR", "qstl": "RAD_QSTL", "rhly": "RAD_RHLY", "cldfk1": "RAD_CLDFK1",
    "gcice": "RAD_GCICE", "grain": "RAD_GRAIN", "grime": "RAD_GRIME", "clw": "RAD_CLW",
    "vvel": "RAD_VVEL", "cvt1": "RAD_CVT1", "cvb1": "RAD_CVB1", "qc2d": "RAD_QC2D",
    "qi2d": "RAD_QI2D", "qs2d": "RAD_QS2D", "ni2d": "RAD_NI2D",
    "xlat": "GLAT", "xlon": "GLON", "slmsk": "SLMSK", "flgmin": "FLGMIN_L", "cv": "CV",
}
_KIND = {  # 3-D kind (LMK/LMP levels) or 2-D [JM, IM]
    "plyr": "lmk", "tlyr": "lmk", "tvly": "lmk", "qlyr": "lmk", "qstl": "lmk", "rhly": "lmk",
    "cldfk1": "lmk", "gcice": "lmk", "grain": "lmk", "grime": "lmk", "clw": "lmk", "vvel": "lmk",
    "qc2d": "lmk", "qi2d": "lmk", "qs2d": "lmk", "ni2d": "lmk", "plvl": "lmp",
    "cvt1": "2d", "cvb1": "2d", "xlat": "2d", "xlon": "2d", "slmsk": "2d", "flgmin": "2d", "cv": "2d",
}
# Fortran argument lists of the dependency calls (grrad_nmmb.f 1589-1592, 1600-1606, 1671-1672, 1681-1698).
_PROGCLD_ARGS = {
    4: ("plyr", "plvl", "tlyr", "tvly", "qlyr", "qstl", "rhly", "clw", "xlat", "xlon", "slmsk"),
    3: ("plyr", "plvl", "tlyr", "tvly", "qlyr", "qstl", "rhly", "clw", "xlat", "xlon", "slmsk",
        "gcice", "grain", "grime", "flgmin", "cldfk1"),
    8: ("plyr", "plvl", "tlyr", "qlyr", "qc2d", "qi2d", "qs2d", "ni2d", "xlat", "cldfk1"),
    "diag": ("plyr", "plvl", "tlyr", "rhly", "vvel", "cv", "cvt1", "cvb1", "xlat", "xlon", "slmsk"),
}
_DEP_OF = {"progcld1": 4, "progcld2": 3, "progcld8": 8, "diagcld1": "diag"}


def _k(name):
    return kernels.get(UNIT, name)


def _scratch(st, name, shape):
    arr = st.fields.get(name)
    if arr is None:
        arr = st.add(name, shape)
    return arr


def _get(ctx, key):
    """Named input: ctx.st first, then the driver tables, then ctx.tables. Raises with the name when absent."""
    arr = ctx.st.fields.get(key)
    if arr is None:
        for table in (DRIVER_TABLE, PREP_TABLE):
            d = ctx.tables.get(table)
            if isinstance(d, dict) and d.get(key) is not None:
                return d[key]
        arr = ctx.tables.get(key)
        if isinstance(arr, dict):
            arr = None
    if arr is None:
        raise RuntimeError(
            f"{UNIT} needs {key!r} in ctx.st or ctx.tables: the producing unit must run earlier in "
            "this step (coszen: phys-rrtm-coszmn; RAD_*: phys-rrtm-grrad-setup; driver tables: "
            "phys-rrtm-driver-call)")
    return arr


def _check(name, arr, shape):
    if tuple(arr.shape) != tuple(shape):
        raise RuntimeError(f"{UNIT}: {name!r} has shape {tuple(arr.shape)}, the contract needs {tuple(shape)}")
    return arr


def _contract(ctx, key, cld_fraction, dims):
    """Named inputs for one dependency call: every Fortran argument of the call, shape-checked."""
    lmk, lmp, jm, im = dims
    inputs = {}
    for fname in _PROGCLD_ARGS[key]:
        shape = {"lmk": (lmk, jm, im), "lmp": (lmp, jm, im), "2d": (jm, im)}[_KIND[fname]]
        inputs[fname] = _check(_STATE_OF[fname], _get(ctx, _STATE_OF[fname]), shape)
    inputs.update({"IM": im, "LMK": lmk, "LMP": lmp, "cld_fraction": cld_fraction})
    return inputs


def _dep(dep, name, ctx, key, cld_fraction, clouds, dims):
    """Call a dependency with its contract inputs and write its clouds into RAD_CLOUDS."""
    if dep is None:
        raise RuntimeError(
            f"phys-rrtm-grrad-clouds needs dependency {name!r} for this cloud branch; "
            "its unit is not ported yet (see the wrapper docstring)")
    lmk, _lmp, jm, im = dims
    inputs = _contract(ctx, key, cld_fraction, dims)
    res = dep(name, ctx, inputs)
    if not isinstance(res, dict) or "clouds" not in res:
        raise RuntimeError(f"{UNIT}: dependency {name!r} must return a dict with 'clouds'")
    clouds[...] = _check(f"{name}.clouds", res["clouds"], (NF_CLDS, lmk, jm, im))
    # progcld* also return the grrad locals cldsa (clds), mtopa, mbota (grrad_nmmb.f:1589-1597), which the
    # LW block (fluxr cloud fractions) and the driver post read as CLDS, MTOP, MBOT.
    for key_res, key_st, shape, dt in (("clds", "CLDS", (5, jm, im), "float32"),
                                       ("mtopa", "MTOP", (3, jm, im), "int32"),
                                       ("mbota", "MBOT", (3, jm, im), "int32")):
        if key_res in res:
            st_arr = ctx.st.add(key_st, shape, dtype=dt)
            st_arr[...] = res[key_res]
    return res


def run(ctx, dep=None):
    """Driver entry. ctx.grid (im, jm, lm), ctx.cfg (np3d, ntcw, ialbflg, ivflip, cld_fraction), ctx.st, ctx.tables."""
    g, cfg, st = ctx.grid, ctx.cfg, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk = lm + LTP
    lmp = lmk + 1
    dims = (lmk, lmp, jm, im)
    np3d, ntcw, ialb = int(cfg.np3d), int(cfg.ntcw), int(cfg.ialbflg)
    cld_fraction = int(getattr(cfg, "cld_fraction", 0))
    ivflip = int(getattr(cfg, "ivflip", 1))
    # Zero-based forms of the Fortran kd, lya, lyb (grrad_nmmb.f:1168-1194, lextop branch).
    if ivflip == 0:
        kd, lya0, lyb0 = 1, 1, 0
    else:
        kd, lya0, lyb0 = 0, lm - 1, lmk - 1
    lextop = LTP > 0

    grid3 = kernels.grid_2d(im, jm, lm)
    grid_lmk = kernels.grid_2d(im, jm, lmk)
    grid2 = kernels.grid_2d(im, jm, 1)
    blk = kernels.BLOCK_2D
    i32 = np.int32

    clouds = _scratch(st, "RAD_CLOUDS", (NF_CLDS, lmk, jm, im))
    vvel = _scratch(st, "RAD_VVEL", (lmk, jm, im))
    cvt1 = _scratch(st, "RAD_CVT1", (jm, im))
    cvb1 = _scratch(st, "RAD_CVB1", (jm, im))
    clw = _scratch(st, "RAD_CLW", (lmk, jm, im))
    qc2d = _scratch(st, "RAD_QC2D", (lmk, jm, im))
    qi2d = _scratch(st, "RAD_QI2D", (lmk, jm, im))
    qs2d = _scratch(st, "RAD_QS2D", (lmk, jm, im))
    ni2d = _scratch(st, "RAD_NI2D", (lmk, jm, im))

    if ntcw > 0:
        # Fortran 1556-1571: clw = sum of the ncld condensate tracers (tracer index ntcw.. in 1-based
        # Fortran), then clw < EPSQ -> 0. Zero-based tracer index is ntcw-1+j.
        tracer1 = _get(ctx, "RAD_TRACER1")
        ntrac = int(tracer1.shape[0])
        if ntcw - 1 + NCLD - 1 >= ntrac:
            raise RuntimeError(f"{UNIT}: ntcw={ntcw} with ncld={NCLD} needs {ntcw + NCLD - 1} tracers, "
                               f"RAD_TRACER1 has {ntrac}")
        _k("grrad_clouds_clw")(
            grid_lmk, blk,
            (tracer1, clw, i32(im), i32(jm), i32(lmk), i32(ntcw - 1), i32(NCLD), np.float32(EPSQ)))

        if np3d == 4:
            _dep(dep, "progcld1", ctx, 4, cld_fraction, clouds, dims)
        elif np3d == 3:
            _dep(dep, "progcld2", ctx, 3, cld_fraction, clouds, dims)
        elif np3d == 5:
            # Fortran 1599-1617 (GFDL type diagnostic clouds), then the extra top layer (1635-1640).
            _k("grrad_clouds_gfdl_fill")(
                grid3, blk,
                (clouds, _get(ctx, "CLDFRA"), _get(ctx, "TAUCLOUDS"), i32(im), i32(jm), i32(lm),
                 i32(lmk), i32(kd)))
            if lextop:
                _k("grrad_top_copy")(
                    grid2, blk,
                    (clouds, i32(im), i32(jm), i32(lmk), i32(4), i32(lya0), i32(lyb0)))
        elif np3d == 8:
            # Fortran 1650-1674: qc2d, qi2d, qs2d (EPSQ floor) and ni2d = max(1e-8, tracer 9).
            if ntrac < NTRAC_MIN:
                raise RuntimeError(
                    f"{UNIT}: np3d 8 (Thompson) reads tracer index {NTRAC_MIN} (ni); RAD_TRACER1 has "
                    f"{ntrac} tracers. Thompson NUM_TRACERS_TOTAL is 10.")
            _k("grrad_clouds_np8_prep")(
                grid_lmk, blk,
                (tracer1, qc2d, qi2d, qs2d, ni2d, i32(im), i32(jm), i32(lmk), np.float32(EPSQ)))
            _dep(dep, "progcld8", ctx, 8, cld_fraction, clouds, dims)
    else:
        # Diagnostic scheme, Fortran 1677-1698.
        _k("grrad_clouds_diag_cvt")(
            grid2, blk, (_get(ctx, "CVT"), _get(ctx, "CVB"), cvt1, cvb1, i32(im), i32(jm)))
        _k("grrad_clouds_diag_vvel")(
            grid3, blk, (_get(ctx, "VVEL"), vvel, i32(im), i32(jm), i32(lm), i32(kd)))
        if lextop:
            _k("grrad_top_copy")(
                grid2, blk, (vvel, i32(im), i32(jm), i32(lmk), i32(1), i32(lya0), i32(lyb0)))
        _dep(dep, "diagcld1", ctx, "diag", cld_fraction, clouds, dims)

    if ialb == 2:
        # Fortran 1722-1794, RRTM 2012 albedo (BSF). Outputs carry the names setalb reads.
        sfcalbedo = _scratch(st, "RAD_SFCALBEDO", (jm, im))
        smx = _scratch(st, "RAD_SMX", (jm, im))
        czmean = _scratch(st, "RAD_CZMEAN", (jm, im))
        alvsf = _scratch(st, "ALVSF", (jm, im))
        alnsf = _scratch(st, "ALNSF", (jm, im))
        alvwf = _scratch(st, "ALVWF", (jm, im))
        alnwf = _scratch(st, "ALNWF", (jm, im))
        facsf = _scratch(st, "FACSF", (jm, im))
        facwf = _scratch(st, "FACWF", (jm, im))
        _k("grrad_clouds_albedo")(
            grid2, blk,
            (_get(ctx, "SFCALBEDO"), _get(ctx, "SM"), _get(ctx, "GRRAD_COSZEN_V"), sfcalbedo, smx, czmean,
             alvsf, alnsf, alvwf, alnwf, facsf, facwf, i32(im), i32(jm)))
    return clouds


# ---------------------------------------------------------------------------------------------
# Selftest: a contract test for every branch (every dependency input bound and shape-checked, all
# outputs finite) plus value checks against numpy references for the parts this unit computes:
# clw, the GFDL fill, the Thompson prep, the diagnostic scheme and the IALB=2 albedo. The progcld and
# diagcld dependencies are other units; here they only receive the contract and return zeros.
# ---------------------------------------------------------------------------------------------
_CU = os.path.join(os.path.dirname(__file__), "..", "..", "kernels", "phys_rrtm_grrad_clouds.cu")


def _parse_c(name, n):
    with open(_CU) as f:
        text = f.read()
    m = re.search(name + r"\[\d+\]\s*=\s*\{(.*?)\};", text, re.S)
    vals = [float(v) for v in re.findall(r"(-?\d*\.\d+)f?", m.group(1))]
    if len(vals) != n:
        raise RuntimeError(f"selftest: parsed {len(vals)} values for {name}, want {n}")
    return np.array(vals, dtype=np.float64)


def _albedo_reference(sm, salb, cz):
    """Fortran 1722-1794 in numpy, 1-based table indexing via (IQ-1) and (JX-1); JX clamp as the kernel."""
    trn = _parse_c("c_TRN", 21)
    za = _parse_c("c_ZA", 20)
    dza = _parse_c("c_DZA", 19)
    albd = _parse_c("c_ALBD", 420).reshape(20, 21).T       # albd[iq-1, jx-1] == ALBD(IQ, JX)
    sfc = salb.astype(np.float64).copy()
    iq = 11                                                  # INT(TWENTY*HP537+ONE)
    mask = (cz > 0.0) & (sm > 0.5)
    c = np.clip(cz[mask], 0.0, 1.0)
    zen = (180.0 / 3.14159265) * np.arccos(c)
    jx = np.where(zen >= 74.0, np.floor(0.5 * (90.0 - zen) + 1.0),
                  np.where(zen >= 50.0, np.floor(0.25 * (74.0 - zen) + 9.0),
                           np.floor(0.1 * (50.0 - zen) + 15.0))).astype(np.int64)
    jx = np.minimum(jx, 19)                                  # the documented deviation (JX = 20)
    dzen = -(zen - za[jx - 1]) / dza[jx - 1]
    alb1 = albd[iq - 1, jx - 1] + dzen * (albd[iq - 1, jx] - albd[iq - 1, jx - 1])
    alb2 = albd[iq, jx - 1] + dzen * (albd[iq, jx] - albd[iq, jx - 1])
    sfc[mask] = alb1 + 20.0 * (alb2 - alb1) * (0.537 - trn[iq - 1])
    return sfc


def selftest():
    import types

    import cupy as cp

    from namgpu.state import State
    from namgpu.steps.drv_tracer_index import host_index

    ntrac = int(host_index("thompson", 0)["NUM_TRACERS_TOTAL"])     # 10
    grid = types.SimpleNamespace(im=954, jm=835, lm=60)
    im, jm, lm = grid.im, grid.jm, grid.lm
    lmk, lmp = lm + LTP, lm + LTP + 1
    st = State(grid, ntracers=ntrac, names=set())
    rng = cp.random.default_rng(7)

    def rnd(shape, lo, hi):
        return rng.uniform(lo, hi, shape, dtype=cp.float32)

    st.add("CLDFRA", (lm, jm, im))[...] = rnd((lm, jm, im), 0.0, 1.0)
    st.add("SM", (jm, im))[...] = (rnd((jm, im), 0.0, 1.0) > 0.3).astype(cp.float32)
    st.add("GLAT", (jm, im))[...] = rnd((jm, im), -1.2, 1.2)
    st.add("GLON", (jm, im))[...] = rnd((jm, im), -2.0, 2.0)
    for name, shape, lo, hi in (
            ("RAD_PLYR", (lmk, jm, im), 1.0, 1000.0), ("RAD_PLVL", (lmp, jm, im), 1.0, 1000.0),
            ("RAD_TLYR", (lmk, jm, im), 200.0, 320.0), ("RAD_TVLY", (lmk, jm, im), 200.0, 320.0),
            ("RAD_QLYR", (lmk, jm, im), 0.0, 0.02), ("RAD_QSTL", (lmk, jm, im), 0.0, 0.02),
            ("RAD_RHLY", (lmk, jm, im), 0.0, 1.0), ("RAD_CLDFK1", (lmk, jm, im), 0.0, 1.0),
            ("RAD_GCICE", (lmk, jm, im), 0.0, 1.0), ("RAD_GRAIN", (lmk, jm, im), 0.0, 1.0),
            ("RAD_GRIME", (lmk, jm, im), 0.0, 1.0)):
        st.add(name, shape)[...] = rnd(shape, lo, hi)
    st.add("RAD_TRACER1", (ntrac, lmk, jm, im))[...] = rnd((ntrac, lmk, jm, im), 0.0, 2.0e-3)
    drv = {
        "TAUCLOUDS": rnd((lm, jm, im), 0.0, 50.0), "VVEL": rnd((lm, jm, im), -1.0, 1.0),
        "CV": rnd((jm, im), 0.0, 1.0), "CVT": rnd((jm, im), 0.0, 100.0),
        "CVB": rnd((jm, im), 0.0, 100.0), "SFCALBEDO": rnd((jm, im), 0.0, 1.0),
        "SLMSK": (rnd((jm, im), 0.0, 1.0) > 0.3).astype(cp.float32),
        "FLGMIN_L": rnd((jm, im), 0.0, 0.2),
    }
    tables = {DRIVER_TABLE: drv, "coszen": rnd((jm, im), -0.2, 1.0)}

    calls = []

    def dep(name, ctx_, inputs):
        calls.append({"name": name, "inputs": sorted(inputs.keys())})
        for fname in _PROGCLD_ARGS[_DEP_OF[name]]:
            assert fname in inputs, (name, fname)
        return {"clouds": cp.zeros((NF_CLDS, lmk, jm, im), dtype=cp.float32)}

    arms = [
        ("np3d4_progcld1", dict(np3d=4, ntcw=1)),
        ("np3d3_progcld2", dict(np3d=3, ntcw=1)),
        ("np3d5_gfdl", dict(np3d=5, ntcw=1)),
        ("np3d8_thompson", dict(np3d=8, ntcw=1)),
        ("diagnostic", dict(np3d=3, ntcw=0)),
    ]
    written = ["RAD_CLOUDS", "RAD_VVEL", "RAD_CVT1", "RAD_CVB1", "RAD_CLW", "RAD_QC2D", "RAD_QI2D",
               "RAD_QS2D", "RAD_NI2D", "RAD_SFCALBEDO", "RAD_SMX", "RAD_CZMEAN", "ALVSF", "ALNSF",
               "ALVWF", "ALNWF", "FACSF", "FACWF"]
    report = {}
    cfg_base = dict(ialbflg=2, ivflip=0, cld_fraction=0)
    for label, c in arms:
        cfg = types.SimpleNamespace(np3d=c["np3d"], ntcw=c["ntcw"], **cfg_base)
        ctx = types.SimpleNamespace(grid=grid, cfg=cfg, st=st, tables=tables)
        calls.clear()
        run(ctx, dep=dep)
        cp.cuda.Stream.null.synchronize()
        for name in written:
            assert bool(cp.isfinite(st.fields[name]).all()), (label, name)
        report[label] = {"deps_called": [x["name"] for x in calls], "finite": True}
        if calls:
            report[label]["contract_inputs"] = calls[0]["inputs"]

    def run_arm(np3d, ntcw):
        cfg = types.SimpleNamespace(np3d=np3d, ntcw=ntcw, **cfg_base)
        run(types.SimpleNamespace(grid=grid, cfg=cfg, st=st, tables=tables), dep=dep)
        cp.cuda.Stream.null.synchronize()

    run_arm(3, 1)
    tr = cp.asnumpy(st.fields["RAD_TRACER1"])
    clw_ref = tr[0].copy()
    clw_ref[clw_ref < np.float32(EPSQ)] = 0.0
    report["clw_matches_reference"] = bool(np.array_equal(cp.asnumpy(st.fields["RAD_CLW"]), clw_ref))
    assert report["clw_matches_reference"], "clw differs from the numpy reference"

    run_arm(5, 1)
    cl = cp.asnumpy(st.fields["RAD_CLOUDS"])
    cldf = cp.asnumpy(st.fields["CLDFRA"])
    ok_gfdl = bool(np.array_equal(cl[0, 1:1 + lm], cldf) and np.all(cl[2, 1:1 + lm] == np.float32(0.99))
                   and np.array_equal(cl[:, 0], cl[:, 1]))
    report["gfdl_fill_matches_reference"] = ok_gfdl
    assert ok_gfdl, "GFDL fill differs from the Fortran 1599-1617 reference"

    run_arm(8, 1)
    ref = tr[3].copy()
    ref[ref <= np.float32(EPSQ)] = 0.0
    ok_np8 = bool(np.array_equal(cp.asnumpy(st.fields["RAD_QC2D"]), ref)
                  and np.all(cp.asnumpy(st.fields["RAD_NI2D"]) >= np.float32(1.0e-8)))
    report["np8_prep_matches_reference"] = ok_np8
    assert ok_np8, "Thompson prep differs from the Fortran 1650-1674 reference"

    run_arm(3, 0)
    cvt1 = cp.asnumpy(st.fields["RAD_CVT1"])
    vv = cp.asnumpy(st.fields["RAD_VVEL"])
    ok_diag = bool(np.array_equal(cvt1, np.float32(10.0) * cp.asnumpy(drv["CVT"]))
                   and np.array_equal(vv[1:1 + lm], np.float32(10.0) * cp.asnumpy(drv["VVEL"]))
                   and np.array_equal(vv[0], vv[1]))
    report["diagnostic_matches_reference"] = ok_diag
    assert ok_diag, "diagnostic scheme differs from the Fortran 1677-1698 reference"

    run_arm(3, 1)
    alb = cp.asnumpy(st.fields["ALVSF"]).astype(np.float64)
    alb_ref = _albedo_reference(cp.asnumpy(st.fields["SM"]).astype(np.float64),
                                cp.asnumpy(drv["SFCALBEDO"]).astype(np.float64),
                                cp.asnumpy(tables["coszen"]).astype(np.float64))
    bad_frac = float(np.mean(np.abs(alb - alb_ref) > 1.0e-4))
    report["albedo_mismatch_fraction"] = bad_frac
    assert bad_frac < 1.0e-3, f"albedo differs from the reference at {bad_frac:.2%} of points"
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(selftest(), indent=1))
