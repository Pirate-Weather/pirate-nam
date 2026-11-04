# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-cldprop: CLDPROP of the NMMB shortwave RRTM (phys/radsw_main_nmmb.f:1400-1822).

NOAA/NCEP NAM (nam.v4.2.11, 83df4e13c1) source; header notice kept in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_sw_cldprop.cu, phys_rrtm_sw_cldprop, one thread per column
(kernels.grid_cols over IM*JM), the layer loop inside the thread. One launch, no host sync, no
per-step allocation (outputs are scratch in ctx.tables, allocated on first call).

Layer axis: the seam arrays are in radsw internal order, surface first (k0 = 0 is the lowest layer,
Fortran cldprop k = 1). The swrad seam reverses the NMMB top-first state arrays before this unit. The
CONTRACT section 1 rule (l = 0 is the model top) is for state arrays only. mcica-subcol, spcvrtc and
taumol consume the same surface-first order.

Inputs (CuPy float32). nlay = LM, JM, IM from ctx.st["T"]:
    cfrac   [LM, JM, IM]  layer cloud fraction (Fortran cfrac(1:nlay) of the radsw call, k0 = 0 is Fortran k = 1)
    cliqp   [LM, JM, IM]  in-cloud liquid water path, g/m2
    reliq   [LM, JM, IM]  liquid effective radius, micron
    cicep   [LM, JM, IM]  in-cloud ice water path, g/m2
    reice   [LM, JM, IM]  ice effective radius, micron
    cdat1   [LM, JM, IM]  rain water path (iswcice < 4) / cloud optical depth (iswcliq = 0)
    cdat2   [LM, JM, IM]  rain effective radius (unused by the iswcliq > 0 branch) / cloud ssa (iswcliq = 0)
    cdat3   [LM, JM, IM]  snow water path (iswcice < 4, iswcice = 4 snow) / cloud asymmetry (iswcliq = 0)
    cdat4   [LM, JM, IM]  snow effective radius, micron
    cf1     [JM, IM]      zcf1, effective total cloud cover of the column. The kernel gates cf1 > 0
                          itself (radsw_main_nmmb.f:985-1002): a column with cf1 <= 0 (or NaN) gets
                          zero outputs, ssacw = 0 included, exactly as the Fortran clear branch.
Tables (ctx.tables["rrtm_sw_cldprtb"], from namgpu/data/rrtm_sw_cldprtb.npz, built by
tools/extract_rrtm_sw_cldprtb.py from radsw_datatb.f module_radsw_cldprtb):
    extliq1, ssaliq1, asyliq1 [14, 58]; extice2, ssaice2, asyice2 [14, 43]; extice3, ssaice3, asyice3 [14, 46]
Outputs:
    taucw, ssacw, asycw [14, LM, JM, IM]  cloud optical properties per SW band (bands 16..29) and layer (scratch)
    cldfrc [LM, JM, IM]   cfrac / cf1, written only when isubcsw <= 0 (else None)
    cldf   [LM, JM, IM]   McICA cloud fraction, written only when isubcsw > 0 and kept in
                          ctx.tables["rrtm_sw_cldf"] for phys-rrtm-sw-mcica-subcol (else None)

NAM configuration: iswcliq=1 and iswcice=3 (physparam.f:53, 57; module_SOLVER_GRID_COMP.F90 ICICE_SW=3),
isubcsw=0 (module_SOLVER_GRID_COMP.F90:10697 ISUBCSW=0). These are the defaults here. ICICE_SW=4 is set
only for Thompson with RRTM (not the NAM default).
isubcsw > 0 is refused in run() until the swrad seam calls phys-rrtm-sw-mcica-subcol (the only consumer of
cldf). Without that call, cldf would be written and never used, so the sub-column step would silently
go missing.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sw_cldprop"
KERNEL = "phys_rrtm_sw_cldprop"
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw_cldprtb.npz"
NBDSW = 14
TAB_NAMES = ("extliq1", "ssaliq1", "asyliq1", "extice2", "ssaice2", "asyice2",
             "extice3", "ssaice3", "asyice3")


def tables(ctx):
    import cupy as cp
    t = ctx.tables
    if "rrtm_sw_cldprtb" not in t:
        z = np.load(TABLE_FILE)
        t["rrtm_sw_cldprtb"] = {k: cp.asarray(z[k]) for k in TAB_NAMES}
    return t["rrtm_sw_cldprtb"]


def _scratch(ctx, key, shape):
    import cupy as cp
    a = ctx.tables.get(key)
    if a is None or tuple(a.shape) != tuple(shape):
        a = cp.empty(shape, dtype=cp.float32)
        ctx.tables[key] = a
    return a


def _random_inputs(lm, jm, im, seed):
    import cupy as cp
    rs = cp.random.RandomState(seed)
    cell = (lm, jm, im)
    u = rs.uniform(0.0, 1.0, cell).astype(cp.float32)
    cfrac = rs.uniform(0.0, 1.0, cell).astype(cp.float32)
    cfrac[cfrac < 0.4] = 0.0                       # plausible clear fraction
    cf1 = (0.5 + 0.5 * rs.uniform(0.0, 1.0, (jm, im))).astype(cp.float32)
    liq = (rs.uniform(0.0, 1.0, cell) * 200.0 * (u > 0.3)).astype(cp.float32)
    ice = (rs.uniform(0.0, 1.0, cell) * 100.0 * (u > 0.5)).astype(cp.float32)
    return dict(
        cfrac=cfrac,
        cliqp=liq,
        reliq=(5.0 + 15.0 * u).astype(cp.float32),
        cicep=ice,
        reice=(20.0 + 60.0 * u).astype(cp.float32),
        cdat1=(0.5 * u).astype(cp.float32),
        cdat2=(0.9 * u).astype(cp.float32),
        cdat3=(5.0 * u).astype(cp.float32),
        cdat4=(10.0 + 100.0 * u).astype(cp.float32),
        cf1=cf1,
    )


def run(ctx, inputs=None, iswcliq: int = 1, iswcice: int = 3, isubcsw: int = 0, seed: int = 1):
    if isubcsw > 0:
        raise RuntimeError(
            "phys-rrtm-sw-cldprop: isubcsw > 0 writes cldf for phys-rrtm-sw-mcica-subcol, but the swrad "
            "seam does not call mcica-subcol yet, so the McICA sub-column cloud fraction would go unused. "
            "Wire mcica-subcol into swrad before enabling isubcsw > 0.")
    lm, jm, im = (int(x) for x in ctx.st["T"].shape)   # LM, JM, IM from the state (CONTRACT section 1)
    nlay = lm
    if inputs is None:
        inputs = _random_inputs(lm, jm, im, seed)
    tab = tables(ctx)

    taucw = _scratch(ctx, "rrtm_sw_cldprop_taucw", (NBDSW, nlay, jm, im))
    ssacw = _scratch(ctx, "rrtm_sw_cldprop_ssacw", (NBDSW, nlay, jm, im))
    asycw = _scratch(ctx, "rrtm_sw_cldprop_asycw", (NBDSW, nlay, jm, im))
    if isubcsw > 0:
        cldf = _scratch(ctx, "rrtm_sw_cldf", (nlay, jm, im))
        cldfrc = ctx.tables.get("rrtm_sw_cldprop_cldfrc_dummy")
        if cldfrc is None:
            import cupy as cp
            cldfrc = cp.empty(1, dtype=cp.float32)
            ctx.tables["rrtm_sw_cldprop_cldfrc_dummy"] = cldfrc
    else:
        cldfrc = _scratch(ctx, "rrtm_sw_cldprop_cldfrc", (nlay, jm, im))
        cldf = ctx.tables.get("rrtm_sw_cldprop_cldf_dummy")
        if cldf is None:
            import cupy as cp
            cldf = cp.empty(1, dtype=cp.float32)
            ctx.tables["rrtm_sw_cldprop_cldf_dummy"] = cldf

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (inputs["cfrac"], inputs["cliqp"], inputs["reliq"], inputs["cicep"], inputs["reice"],
           inputs["cdat1"], inputs["cdat2"], inputs["cdat3"], inputs["cdat4"], inputs["cf1"],
           tab["extliq1"], tab["ssaliq1"], tab["asyliq1"],
           tab["extice2"], tab["ssaice2"], tab["asyice2"],
           tab["extice3"], tab["ssaice3"], tab["asyice3"],
           taucw, ssacw, asycw, cldfrc, cldf,
           np.int32(nlay), np.int32(iswcliq), np.int32(iswcice), np.int32(isubcsw),
           np.int32(im), np.int32(jm)))
    return (taucw, ssacw, asycw,
            (cldfrc if isubcsw <= 0 else None),
            (cldf if isubcsw > 0 else None))
