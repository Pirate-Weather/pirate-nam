"""Done-rule check for phys-rrtm-sw-taumol26 (CONTRACT.md section 6, parts 1 and 2).

    source env.sh && python tools/check-phys-rrtm-sw-taumol26.py

Compiles the kernel, then runs the wrapper ONCE on a full-size random but plausible colmol
(NLAY = 60, JM = 835, IM = 954) and asserts: no NaN/Inf in taug26 and taur26; taug26 is exactly zero (the
routine is a zero fill); taur26 equals colmol * rayl(g) for every g-point (checked on a column sample of the
full array, so the host does not copy the whole 191 MB plane set).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cupy as cp
    import numpy as np

    from namgpu import kernels
    from types import SimpleNamespace
    from namgpu.physics.rrtm import phys_rrtm_sw_taumol26 as unit

    nlay, jm, im = 60, 835, 954
    rng = cp.random.default_rng(1)
    kernels.module(unit.UNIT).compile()

    colmol = cp.exp(rng.uniform(np.log(1e20), np.log(1e25), (nlay, jm, im))).astype(cp.float32)
    tables = {unit.IN_KEY: {"colmol": colmol}}
    # run() reads only ctx.tables and ctx.stream; driver.Ctx needs a full cfg (uncommitted driver change).
    ctx = SimpleNamespace(tables=tables, stream=cp.cuda.Stream(non_blocking=True))
    with ctx.stream:
        out = unit.run(ctx)
    ctx.stream.synchronize()

    taug, taur = out["taug26"], out["taur26"]
    finite = bool(cp.isfinite(taug).all()) and bool(cp.isfinite(taur).all())
    zero_taug = bool((taug == 0).all())
    rayl = unit.load_tables()["rayl"]
    # column sample: 4 layers x a 64 x 64 patch corner, every g-point
    sl = (slice(None), slice(0, 60, 15), slice(0, 64), slice(0, 64))  # layers 0,15,30,45; 64x64 corner
    ref = colmol[0:60:15, 0:64, 0:64][None] * rayl[:, None, None, None]
    got = taur[sl]
    err = float(cp.max(cp.abs(got - ref) / cp.maximum(cp.abs(ref), 1e-30)))
    print(f"phys_rrtm_sw_taumol26: taug {tuple(taug.shape)} taur {tuple(taur.shape)} finite={finite} "
          f"taug_zero={zero_taug} taur_max_rel_err_sample={err:.3g}")
    ok = finite and zero_taug and err < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
