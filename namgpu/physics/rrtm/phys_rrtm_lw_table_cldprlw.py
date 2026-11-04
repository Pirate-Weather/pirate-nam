"""phys-rrtm-lw-table-cldprlw: NMMB RRTM longwave cloud-property coefficient tables.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radlw_datatb_nmmb.f:733-1144, module_radlw_cldprlw_nmmb
(NOAA/NCEP, US government work). The module is DATA statements only, so this unit has no kernel
and no per-step work. The tables live in namgpu/data/rrtm_lw/cldprlw.npz, extracted by
tools/extract_rrtm_lw_cldprlw_nmmb.py, and are uploaded once to ctx.tables at init.

Binding (checked against the nam.v4.2.11 source on 2026-11-03): the NMMB longwave driver
radlw_main_nmmb.f reads its cloud tables through `use module_radlw_cldprlw` (line 1444), which is
radlw_datatb.f:910, not module_radlw_cldprlw_nmmb. Nothing in the NAM tree uses
module_radlw_cldprlw_nmmb, so the arrays below have no consumer. phys-rrtm-lw-cldprop therefore keeps
the radlw_datatb.f tables (namgpu/data/rrtm_lw_cldprlw.npz), which are what NAM's NMMB longwave
reads. This unit is reference data: it is extracted with correct float32 rounding
(tools/f32_literal.py), it is not on any step path, and nothing calls run() in the forecast.

Out: ctx.tables entries (cupy arrays, float32 except ipat int32), keys prefixed "rrtm_lw_cldprlw_":
  absice0  [2]        absice1 [5, 2] (absice1[ia-1, m-1] = Fortran absice1(m, ia))
  absice2  [16, 40]   absliq3 [16, 58] (row = band ib-1, column = effective-radius index)
  absliq1, absliq2, absrain, abssnow0, abssnow1  0-d float32
  ipat     [16] int32 (band index for the Ebert-Curry ice cloud, iflagice = 1)
Fortran (i,j) maps to [j, i] per CONTRACT section 1; the 1-D Fortran tables map to the same order.
The upload is a single host-to-device copy at the first call (init), never per step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "phys_rrtm_lw_table_cldprlw"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "cldprlw.npz"
KEYS = ("absice0", "absice1", "absice2", "absliq1", "absliq2", "absliq3",
        "absrain", "abssnow0", "abssnow1", "ipat")
PREFIX = "rrtm_lw_cldprlw_"


def _load_host(path: Path = NPZ) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in KEYS}
    shapes = {"absice0": (2,), "absice1": (5, 2), "absice2": (16, 40), "absliq3": (16, 58),
              "ipat": (16,), "absliq1": (), "absliq2": (), "absrain": (), "abssnow0": (),
              "abssnow1": ()}
    for k, shp in shapes.items():
        if out[k].shape != shp:
            raise ValueError(f"{UNIT}: {k} has shape {out[k].shape}, expected {shp}")
    if out["ipat"].dtype != np.int32:
        raise ValueError(f"{UNIT}: ipat must be int32, got {out['ipat'].dtype}")
    return out


def run(ctx):
    """Upload the cldprlw tables into ctx.tables once; later calls return the cached entries."""
    import cupy as cp

    t = ctx.tables
    if PREFIX + "absice2" not in t:
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
