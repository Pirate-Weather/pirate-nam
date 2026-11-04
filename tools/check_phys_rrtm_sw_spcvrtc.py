"""Done-rule check for phys-rrtm-sw-spcvrtc (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check_phys_rrtm_sw_spcvrtc.py

1. compiles the unit's kernel (NVRTC through CuPy);
2. runs the wrapper ONCE on a full-size random state (954 x 835 x 60, all 112 g-points) and asserts
   that no output array holds NaN or Inf.
Take the GPU pool lease first (README.md). No column test, no reduced size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

from namgpu import kernels  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_sw_spcvrtc as unit  # noqa: E402

import os
JM = int(os.environ.get("CHECK_JM", 835))
IM = int(os.environ.get("CHECK_IM", 954))
NLAY = 60


def U(shape, lo, hi):
    return (lo + (hi - lo) * cp.random.random(shape, dtype=cp.float32)).astype(cp.float32)


def main() -> int:
    kernels.module(unit.UNIT).compile()
    print("compile: ok", flush=True)

    cp.random.seed(1)
    cz = U((JM, IM), 0.0, 1.0)
    cz = cp.where(cp.random.random((JM, IM), dtype=cp.float32) < 0.3, 0.0, cz).astype(cp.float32)
    sntz = (1.0 / cp.maximum(cz, 0.01)).astype(cp.float32)
    cf1 = U((JM, IM), 0.0, 1.0)
    cf0 = (1.0 - cf1).astype(cp.float32)
    cldfrc = (U((NLAY, JM, IM), 0.0, 1.0)
              * (cp.random.random((NLAY, JM, IM), dtype=cp.float32) < 0.4)).astype(cp.float32)
    inp = {
        "cosz": cz, "sntz": sntz, "ssolar": U((JM, IM), 1000.0, 1400.0),
        "cf1": cf1, "cf0": cf0,
        "albbm": U((2, JM, IM), 0.05, 0.6), "albdf": U((2, JM, IM), 0.05, 0.6),
        "cldfrc": cldfrc,
    }

    def gpoint(jg):
        return U((NLAY, JM, IM), 0.0, 0.5), U((NLAY, JM, IM), 0.0, 0.05)

    def band(ib):
        return {
            "tauae": U((NLAY, JM, IM), 0.0, 0.3),
            "ssaae": U((NLAY, JM, IM), 0.5, 1.0),
            "asyae": U((NLAY, JM, IM), 0.0, 0.9),
            "taucw": U((NLAY, JM, IM), 0.0, 10.0),
            "ssacw": U((NLAY, JM, IM), 0.9, 1.0),
            "asycw": U((NLAY, JM, IM), 0.7, 0.9),
        }

    sfluxzen = U((112, JM, IM), 0.02, 0.08)  # per-column TOA flux, [NGPTSW, JM, IM] (taumol output)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = SimpleNamespace(tables={"rrtm_sw_in": inp, "rrtm_sw_gpoint": gpoint,
                                  "rrtm_sw_band": band, "rrtm_sw_sfluxzen": sfluxzen}, stream=stream)
    t0 = time.perf_counter()
    out = unit.run(ctx)
    stream.synchronize()
    print(f"run: {time.perf_counter() - t0:.2f} s over {NLAY} layers, {JM} x {IM} columns, "
          f"112 g-points", flush=True)

    bad = [name for name, a in out.items() if not bool(cp.isfinite(a).all())]
    for name in sorted(out):
        a = out[name]
        print(f"  {name:8s} shape={tuple(a.shape)} min={float(a.min()):.4g} max={float(a.max()):.4g}")
    if bad:
        print("FAIL non-finite:", bad)
        return 1
    print("PASS: no NaN/Inf in any output array")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
