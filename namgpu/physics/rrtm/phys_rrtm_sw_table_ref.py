# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-table-ref: module_radsw_ref reference pressure and temperature tables for the NMMB RRTM
shortwave (preflog, tref, 59 entries each).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radsw_datatb.f module_radsw_ref (DATA statements for
preflog and tref). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the NOAA header of radsw_datatb.f is the notice. The values
are extracted once by tools/extract_rrtm_sw_ref.py into namgpu/data/rrtm_sw/ref.npz (float32, the
single-precision values Fortran builds from its decimal literals). No stand-in values are used.

No kernel: constant data. Out: ctx.tables["rrtm_sw_setcoef_tables"] = {"preflog": f32[59], "tref": f32[59]}
on the device, one host-to-device copy on the first call, reused afterwards.
Fortran preflog(k) / tref(k) (k = 1..59) is out[...][k-1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "phys_rrtm_sw_table_ref"
TAB_KEY = "rrtm_sw_setcoef_tables"
NREF = 59
DATA = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "ref.npz"


def load_host() -> dict:
    with np.load(DATA) as d:
        out = {k: np.asarray(d[k]) for k in ("preflog", "tref")}
    for k, a in out.items():
        if a.shape != (NREF,) or a.dtype != np.float32:
            raise ValueError(f"{UNIT}: {k} has {a.shape} {a.dtype}, want ({NREF},) float32")
        if not np.isfinite(a).all():
            raise ValueError(f"{UNIT}: non-finite {k} in {DATA}")
    return out


def run(ctx) -> dict:
    import cupy as cp

    if TAB_KEY not in ctx.tables:
        h = load_host()
        ctx.tables[TAB_KEY] = {k: cp.asarray(v) for k, v in h.items()}
    return ctx.tables[TAB_KEY]
