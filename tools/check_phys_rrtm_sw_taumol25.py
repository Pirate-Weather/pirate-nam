"""Done-rule check for phys-rrtm-sw-taumol25 (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh && python tools/check_phys_rrtm_sw_taumol25.py

Compiles the kernel, then runs the wrapper once over the full 954 x 835 x 60 domain (one launch, all columns
live). The outputs are this unit's six planes of taug and taur (Fortran planes 74..79). Checks that every
written value is finite, and that the buffer holds exactly the six planes the unit owns.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NLAY, JM, IM = 60, 835, 954


def main() -> int:
    import cupy as cp

    from namgpu import kernels
    from types import SimpleNamespace
    from namgpu.physics.rrtm import phys_rrtm_sw_taumol25 as unit

    kernels.module(unit.UNIT).compile()
    rng = cp.random.default_rng(25)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = SimpleNamespace(tables={})
    ctx.tables[unit.TAB_KEY] = unit.load_tables()
    out = unit.alloc_outputs(NLAY, JM, IM)
    shp = (NLAY, JM, IM)
    inp = {
        "colamt": rng.uniform(1e16, 1e22, (6,) + shp, dtype=cp.float32),
        "colmol": rng.uniform(1e19, 1e25, shp, dtype=cp.float32),
        "fac00": rng.uniform(0.0, 1.0, shp, dtype=cp.float32),
        "fac01": rng.uniform(0.0, 1.0, shp, dtype=cp.float32),
        "fac10": rng.uniform(0.0, 1.0, shp, dtype=cp.float32),
        "fac11": rng.uniform(0.0, 1.0, shp, dtype=cp.float32),
        "laytrop": rng.integers(1, NLAY + 1, (JM, IM), dtype=cp.int32),
        "jp": rng.integers(1, 59, shp, dtype=cp.int32),
        "jt": rng.integers(1, 5, shp, dtype=cp.int32),
        "jt1": rng.integers(1, 5, shp, dtype=cp.int32),
    }
    kmin_buf = cp.zeros((1,), dtype=cp.int32)

    t0 = time.time()
    with stream:
        unit.taumol25(inp, out, ctx.tables[unit.TAB_KEY], kmin_buf, stream=stream)
    stream.synchronize()
    elapsed = time.time() - t0

    bad = []
    for name in ("taug", "taur"):
        if out[name].shape != (unit.NG25, NLAY, JM, IM):
            bad.append(f"{name} shape {out[name].shape}")
        if not bool(cp.isfinite(out[name]).all()):
            bad.append(f"{name} non-finite")
        print(f"{name}: planes 0..5 (Fortran 74..79) min {float(out[name].min()):.4g} "
              f"max {float(out[name].max()):.4g} GB {out[name].nbytes / 1e9:.2f}", flush=True)
    print(f"phys_rrtm_sw_taumol25: full domain {NLAY}x{JM}x{IM} in {elapsed:.2f} s")
    print("PASS" if not bad else f"FAIL: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
