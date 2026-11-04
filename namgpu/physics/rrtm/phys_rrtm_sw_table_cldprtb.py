"""phys-rrtm-sw-table-cldprtb: NMMB RRTM shortwave cloud-property coefficient tables.

Source: NAM nam.v4.2.11 (83df4e13c1) phys/radsw_datatb_nmmb.f:52-1064, module_radsw_cldprtb_nmmb
(NOAA/NCEP, US government work). The module is DATA statements and parameters only, so this unit has
no kernel and no per-step work. The tables live in namgpu/data/rrtm_sw/cldprtb.npz, extracted by
tools/extract_rrtm_sw_cldprtb_nmmb.py, and are uploaded once to ctx.tables at init.

Out: ctx.tables entries (cupy arrays, float32), keys prefixed "rrtm_sw_cldprtb_":
  extdatliq1, ssadatliq1, asydatliq1   [14, 58]  row = band - NBLOW (band 16..29 -> row 0..13);
                                                  column k0 = Fortran first index k - 1 (k = 1..58)
  extdatice3, ssadatice3, asydatice3, fdldatice3  [14, 27]
  b0r, c0r, b0s, b1s, c0s              [14]       band vectors (band = 16..29)
  a0r, a1r, a0s, a1s, absdatliq1       0-d float32
The upload is a single host-to-device copy at the first call (init), never per step.

Precision: the Fortran declares kind_phys. NAM's production build defines -DSINGLE_PREC
(configure.nems.Wcoss_C.intel_ESMF_630r_gsm CPPFLAGS), which makes kind_phys = real*4 (phys/machine.f:20).
Without that define kind_phys is real*8 (phys/machine.f:10). This unit stores float32 because nam-gpu runs
kfpt=single (CONTRACT.md section 3). Recorded declared-precision deviation: a real*8 build would need float64.
NBLOW/NBHGH come from module_radsw_parameters_nmmb, phys/radsw_param_nmmb.f:147-148.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

UNIT = "phys_rrtm_sw_table_cldprtb"
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_sw" / "cldprtb.npz"
SHAPES = {
    "extdatliq1": (14, 58), "ssadatliq1": (14, 58), "asydatliq1": (14, 58),
    "extdatice3": (14, 27), "ssadatice3": (14, 27), "asydatice3": (14, 27), "fdldatice3": (14, 27),
    "b0r": (14,), "c0r": (14,), "b0s": (14,), "b1s": (14,), "c0s": (14,),
    "a0r": (), "a1r": (), "a0s": (), "a1s": (), "absdatliq1": (),
}
PREFIX = "rrtm_sw_cldprtb_"


def _load_host(path: Path = NPZ) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in SHAPES}
    for k, shp in SHAPES.items():
        if out[k].shape != shp:
            raise ValueError(f"{UNIT}: {k} has shape {out[k].shape}, expected {shp}")
        if out[k].dtype != np.float32:
            raise ValueError(f"{UNIT}: {k} must be float32, got {out[k].dtype}")
    return out


def run(ctx):
    """Upload the cldprtb tables into ctx.tables once; later calls return the cached entries."""
    import cupy as cp

    t = ctx.tables
    if PREFIX + "extdatliq1" not in t:
        host = _load_host()
        for k in SHAPES:
            t[PREFIX + k] = cp.asarray(host[k])
    return t


if __name__ == "__main__":
    from types import SimpleNamespace

    host = _load_host()
    for k in sorted(host):
        print(k, host[k].shape, host[k].dtype)
    ctx = SimpleNamespace(tables={}, stream=None)
    out = run(ctx)
    for k in sorted(out):
        print(k, out[k].shape, out[k].dtype)
