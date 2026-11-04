# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The nam-gpu port of this file is Apache-2.0.
"""phys-rrtm-sw-table-sflux: NMMB RRTM shortwave solar-flux reference tables.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f:1070-1378, module_radsw_sflux_nmmb
(NOAA/NCEP, NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE). The module is DATA statements only, so this unit has no kernel
and no per-step work (same shape as phys-rrtm-lw-table-cldprlw). The tables live in
namgpu/data/rrtm_sw/sflux.npz, extracted by tools/extract_rrtm_sw_sflux_nmmb.py, and are uploaded
once to ctx.tables at init.

Out: ctx.tables entries (cupy arrays), keys prefixed "rrtm_sw_sflux_":
  sfluxref01 [7, 1, 16]   sfluxref02 [2, 5, 16]   sfluxref03 [5, 9, 16]  float32
      Fortran sfluxref0x(g, s, b) -> [b-1, s-1, g-1]  (g = g-point, s = solar-cycle index,
      b = band slot in the order given by band_sfluxref0x)
  band_sfluxref01 [7], band_sfluxref02 [2], band_sfluxref03 [5]  int32 band numbers (16..29)
  strrat [14], specwt [14]  float32       entry k-1 = Fortran band NBLOW-1+k (band 16 + k - 1)
  layreffr [14], ix1 [14], ix2 [14], ibx [14]  int32
  scalekur  0-d float32 = 50.15/48.37 (the Fortran comment applies it to band-27 sfluxref01 values;
            the DATA values here are unscaled, as in the Fortran)
The upload is a single host-to-device copy at the first call (init), never per step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "phys_rrtm_sw_table_sflux"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "sflux.npz"
KEYS = ("sfluxref01", "sfluxref02", "sfluxref03",
        "band_sfluxref01", "band_sfluxref02", "band_sfluxref03",
        "strrat", "specwt", "layreffr", "ix1", "ix2", "ibx", "scalekur")
PREFIX = "rrtm_sw_sflux_"
SHAPES = {"sfluxref01": (7, 1, 16), "sfluxref02": (2, 5, 16), "sfluxref03": (5, 9, 16),
          "band_sfluxref01": (7,), "band_sfluxref02": (2,), "band_sfluxref03": (5,),
          "strrat": (14,), "specwt": (14,), "layreffr": (14,), "ix1": (14,), "ix2": (14,),
          "ibx": (14,), "scalekur": ()}
INT_KEYS = ("band_sfluxref01", "band_sfluxref02", "band_sfluxref03", "layreffr", "ix1", "ix2", "ibx")


def _load_host(path: Path = NPZ) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in KEYS}
    for k, shp in SHAPES.items():
        if out[k].shape != shp:
            raise ValueError(f"{UNIT}: {k} has shape {out[k].shape}, expected {shp}")
        want = np.int32 if k in INT_KEYS else np.float32
        if out[k].dtype != want:
            raise ValueError(f"{UNIT}: {k} has dtype {out[k].dtype}, expected {np.dtype(want)}")
    return out


def run(ctx):
    """Upload the sflux tables into ctx.tables once; later calls return the cached entries."""
    import cupy as cp

    t = ctx.tables
    if PREFIX + "sfluxref01" not in t:
        host = _load_host()
        for k in KEYS:
            t[PREFIX + k] = cp.asarray(host[k])
    return t


if __name__ == "__main__":
    from types import SimpleNamespace

    ctx = SimpleNamespace(tables={}, stream=None)
    out = run(ctx)
    for k in sorted(out):
        print(k, out[k].shape, out[k].dtype)
