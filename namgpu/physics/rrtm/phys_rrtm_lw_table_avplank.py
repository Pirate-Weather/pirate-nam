# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-lw-table-avplank: RRTM LW Planck-flux table (module_radlw_avplank, radlw_datatb.f).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_datatb.f:58-728 (module_radlw_avplank, the module radlw_main_nmmb.f
links): the DATA statements that
set totplnk(NPLNK=181, NBANDS=16). NCEP's modifications are US Government work, RRTMG code AER BSD-3-Clause (see NOTICE); the NOAA header of radlw_datatb.f
is kept as the notice. module_radlw_avplank_nmmb (radlw_datatb_nmmb.f:57-727)
has different values and is not referenced by radlw_main_nmmb.f, so it is not used.

No kernel: totplnk is constant data (the Fortran only sets it with DATA statements, no arithmetic),
so there is no GPU computation to port. The table is extracted once by
tools/extract_rrtm_lw_avplank.py into namgpu/data/rrtm_lw_avplank.npz (values parsed as Fortran
decimal literals, stored float32 as in every other nam-gpu table) and uploaded once here.

Out: ctx.tables["rrtm_lw_totplnk"], float32 [NBANDS, NPLNK] = [16, 181], CuPy C order.
     Fortran totplnk(k, ib) (k = 1..181, ib = 1..16) is out[ib-1, k-1].
Launch: none. One host-to-device copy of 11.6 KB on the first call, reused afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "phys_rrtm_lw_table_avplank"
NPLNK = 181
NBANDS = 16
DATA = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw_avplank.npz"


def load_host() -> np.ndarray:
    with np.load(DATA) as d:
        t = d["totplnk"]
    if t.shape != (NBANDS, NPLNK) or t.dtype != np.float32:
        raise ValueError(f"{UNIT}: {DATA} has {t.shape} {t.dtype}, want ({NBANDS}, {NPLNK}) float32")
    if not np.isfinite(t).all():
        raise ValueError(f"{UNIT}: non-finite values in {DATA}")
    return t


def run(ctx):
    import cupy as cp

    if "rrtm_lw_totplnk" not in ctx.tables:
        ctx.tables["rrtm_lw_totplnk"] = cp.asarray(load_host())
    return ctx.tables["rrtm_lw_totplnk"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import cupy as cp

    ctx = SimpleNamespace(cfg=SimpleNamespace(), tables={}, stream=None)
    out = run(ctx)
    assert out.shape == (NBANDS, NPLNK) and out.dtype == cp.float32
    assert bool(cp.isfinite(out).all()), "NaN/Inf in totplnk"
    print(UNIT, "ok", out.shape, out.dtype, float(out.min()), float(out.max()))
