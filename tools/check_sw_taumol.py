"""Done-rule check for phys-rrtm-sw-taumol (CONTRACT.md section 6, parts 1 and 2).

    source env.sh && python tools/check_sw_taumol.py

Compiles the kernel, then runs the wrapper ONCE on full-size random but plausible inputs
(NLAY = 60, JM = 835, IM = 954) and checks the output for NaN/Inf and that every g-point of every
column was written (TOA flux is strictly positive in every band of the tables used).
Only the six inputs of the unit are allocated (about 1.5 GB), not the full model state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cupy as cp
    import numpy as np

    from namgpu import kernels
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_sw_taumol as unit

    nlay, jm, im = 60, 835, 954
    rng = cp.random.default_rng(1)
    kernels.module(unit.UNIT).compile()

    tables = {
        "rrtm_sw_colamt": rng.uniform(1e16, 1e22, (6, nlay, jm, im), dtype=cp.float32),
        "rrtm_sw_jp": rng.integers(1, 58, (nlay, jm, im), dtype=cp.int32),
        "rrtm_sw_laytrop": rng.integers(1, nlay + 1, (jm, im), dtype=cp.int32),
        "rrtm_sw_sflux_sfluxref01": rng.uniform(0.1, 100.0, (7, 1, 16), dtype=cp.float32),
        "rrtm_sw_sflux_sfluxref02": rng.uniform(0.1, 100.0, (2, 5, 16), dtype=cp.float32),
        "rrtm_sw_sflux_sfluxref03": rng.uniform(0.1, 100.0, (5, 9, 16), dtype=cp.float32),
    }
    ctx = Ctx(cfg=SimpleNamespace(raw={"restart": False}), grid=None, st=None, stream=cp.cuda.Stream(non_blocking=True), tables=tables)
    with ctx.stream:
        out = unit.run(ctx)
    ctx.stream.synchronize()

    finite = bool(cp.isfinite(out).all())
    lo = float(out.min())
    hi = float(out.max())
    print(f"phys_rrtm_sw_taumol: shape {tuple(out.shape)}, finite={finite}, min={lo:.4g}, max={hi:.4g}")
    # Positive everywhere: every g-point of every column was written with a positive flux.
    ok = finite and lo > 0.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
