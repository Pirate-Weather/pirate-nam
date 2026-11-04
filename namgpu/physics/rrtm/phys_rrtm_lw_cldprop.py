# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-cldprop: CLDPROP of the NMMB LW radiation (phys/radlw_main_nmmb.f:1344-1697).

NOAA/NCEP NAM (nam.v4.2.11, 83df4e13c1) source; header notice kept in the kernel file.

Kernel: namgpu/kernels/phys_rrtm_lw_cldprop.cu, phys_rrtm_lw_cldprop, one thread per column
(kernels.grid_cols over IM*JM), the level loop inside the thread. One launch, no host sync,
no per-step allocation (taucld and cldf are scratch in ctx.tables, allocated on first call).

Inputs, read from model state (no synthetic data in this wrapper):
    ctx.st.fields["RAD_CLOUDS"]  [NF_CLDS=9, LMK, JM, IM] float32, written by phys-rrtm-grrad-clouds
                                 (CONTRACT section 1, model top first, layer 0 = top). LMK = LM + 1
                                 (grrad LTP extra top layer); nlay is read from the array shape.
    Channels: 0 cfrac, 1 cliqp, 2 reliq, 3 cicep, 4 reice, 5 cda1, 6 cda2 (unused), 7 cda3, 8 cda4.
    With ilwcliq == 0 the input optical depth is channel 1 (cda1 of the ilwcliq > 0 branch is 5).
    The kernel does the lwrad_prep layer flip itself (ivflip = 0, NAM): radlw internal layer k0 = 0 is
    the surface, state layer nlay-1-k0.

Outputs (radlw internal order, k0 = 0 is the SURFACE, Fortran k = 1; rtrn and mcica_subcol read them in
this order; they are scratch, not state arrays):
    taucld  [16, nlay, JM, IM]  cloud optical depth per LW band and layer, ctx.tables["rrtm_lw_cldprop_taucld"]
    cldf    [nlay, JM, IM]      layer cloud fraction for the McICA generator, written only when isubclw > 0
                                (ctx.tables["rrtm_lw_cldprop_cldf"], the key phys-rrtm-lw-mcica-subcol reads);
                                otherwise None.
Config: isubclw and ilwcliq come from ctx.tables["rrtm_lw_ctl_host"] (rlwinit owns them); ilwcice is an argument.

Gate: the kernel applies the lwrad lcf1 gate per column (cldprop runs only if some layer has
cfrac > 1e-6, radlw_main_nmmb.f:924-945). A column with no such layer gets taucld = 0 and cldf = 0.

Named gap: cldfmc. Fortran zeroes cldfmc at radlw_main_nmmb.f:1477-1481, calls mcica_subcol at 1676 and
writes the 1/0 cldfmc at 1683-1691 (isubclw > 0 only). That is owned by phys-rrtm-lw-mcica-subcol. With
isubclw = 0 (NAM default) rtrn reads taucld only, so nothing is lost on the NAM path.

Undefined Fortran inputs are refused here (radlw_main_nmmb.f:1531-1541 for tauliq, 1555-1611 for tauice):
ilwcliq must be 0 or 1; when ilwcliq > 0, ilwcice must be 1, 2, 3 or 4. ilwcice is ignored when ilwcliq == 0
(radlw_main_nmmb.f:451).

Table binding: ctx.tables["rrtm_lw_cldprlw"] from namgpu/data/rrtm_lw_cldprlw.npz, extracted from radlw_datatb.f
DATA statements (module_radlw_cldprlw, the module radlw_main_nmmb.f:1444 uses). The NMMB module
(module_radlw_cldprlw_nmmb, namgpu/data/rrtm_lw/cldprlw.npz, radlw_datatb_nmmb.f:733) is unused by NAM and is
not read here.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_cldprop"
KERNEL = "phys_rrtm_lw_cldprop"
TABLE_FILE = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw_cldprlw.npz"
NBANDS = 16
NF_CLDS = 9
CLOUDS_KEY = "RAD_CLOUDS"            # written by phys-rrtm-grrad-clouds (RAD_CLOUDS, CONTRACT order)
CLDF_KEY = "rrtm_lw_cldprop_cldf"    # the key phys-rrtm-lw-mcica-subcol reads (both units must agree)
ILWCLIQ_OK = (0, 1)                  # ilwcliq >= 2 leaves tauliq undefined in Fortran (radlw_main_nmmb.f:1531-1541)
ILWCICE_OK = (1, 2, 3, 4)            # ilwcice = 0 with ilwcliq > 0 leaves tauice undefined in Fortran


def tables(ctx):
    import cupy as cp
    t = ctx.tables
    if "rrtm_lw_cldprlw" not in t:
        z = np.load(TABLE_FILE)
        t["rrtm_lw_cldprlw"] = {k: cp.asarray(z[k]) for k in ("absliq1", "absice1", "absice2", "absice3", "ipat")}
    return t["rrtm_lw_cldprlw"]


def run(ctx, ilwcliq=None, ilwcice: int = 1, isubclw=None):
    """Driver entry. Reads ctx.st.fields[RAD_CLOUDS] (phys-rrtm-grrad-clouds must run earlier in the step).
    isubclw and ilwcliq come from the rlwinit host copy ctx.tables["rrtm_lw_ctl_host"] (RLWINIT owns them,
    radlw_main_nmmb.f:1229-1265). Passing them explicitly is for tests only. A missing rlwinit copy is a
    refusal, never a silent default. Returns (taucld, cldf); cldf is None unless isubclw > 0."""
    import cupy as cp

    ctl = ctx.tables.get("rrtm_lw_ctl_host")
    if (ctl is None) and (isubclw is None or ilwcliq is None):
        raise RuntimeError(f"{UNIT}: ctx.tables[rrtm_lw_ctl_host] missing; phys-rrtm-lw-rlwinit must run first")
    if isubclw is None:
        isubclw = int(ctl["isubclw"])
    if ilwcliq is None:
        ilwcliq = int(ctl["ilwcliq"])
    if ilwcliq not in ILWCLIQ_OK:
        raise ValueError(f"{UNIT}: ilwcliq={ilwcliq} refused: Fortran leaves tauliq undefined for it "
                         f"(radlw_main_nmmb.f:1531-1541); NAM uses ilwcliq=1")
    if ilwcliq > 0 and ilwcice not in ILWCICE_OK:
        raise ValueError(f"{UNIT}: ilwcice={ilwcice} refused with ilwcliq={ilwcliq}: Fortran leaves tauice "
                         f"undefined for it (radlw_main_nmmb.f:1555-1611); NAM uses ilwcice=1")
    if isubclw < 0:
        raise ValueError(f"{UNIT}: isubclw={isubclw} must be >= 0")

    clouds = ctx.st.fields.get(CLOUDS_KEY)
    if clouds is None:
        raise RuntimeError(f"{UNIT} needs ctx.st[{CLOUDS_KEY!r}] from phys-rrtm-grrad-clouds: "
                           f"that unit must run earlier in the same step")
    if clouds.ndim != 4 or clouds.shape[0] != NF_CLDS or clouds.dtype != cp.float32 or not clouds.flags.c_contiguous:
        raise ValueError(f"{UNIT}: {CLOUDS_KEY} must be C-contiguous float32 [{NF_CLDS}, nlay, JM, IM], "
                         f"got shape {tuple(clouds.shape)} dtype {clouds.dtype}")
    nlay, jm, im = int(clouds.shape[1]), int(clouds.shape[2]), int(clouds.shape[3])

    tab = tables(ctx)

    taucld = ctx.tables.get("rrtm_lw_cldprop_taucld")
    if taucld is None or tuple(taucld.shape) != (NBANDS, nlay, jm, im):
        taucld = cp.empty((NBANDS, nlay, jm, im), dtype=cp.float32)
        ctx.tables["rrtm_lw_cldprop_taucld"] = taucld
    if isubclw > 0:
        cldf = ctx.tables.get(CLDF_KEY)
        if cldf is None or tuple(cldf.shape) != (nlay, jm, im):
            cldf = cp.empty((nlay, jm, im), dtype=cp.float32)
            ctx.tables[CLDF_KEY] = cldf
    else:
        cldf = ctx.tables.get("rrtm_lw_cldprop_cldf_dummy")
        if cldf is None:
            cldf = cp.empty(1, dtype=cp.float32)
            ctx.tables["rrtm_lw_cldprop_cldf_dummy"] = cldf

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (clouds,
           tab["absliq1"], tab["absice1"], tab["absice2"], tab["absice3"], tab["ipat"],
           taucld, cldf,
           np.int32(nlay), np.int32(ilwcliq), np.int32(ilwcice), np.int32(isubclw),
           np.int32(im), np.int32(jm)))
    return taucld, (cldf if isubclw > 0 else None)
