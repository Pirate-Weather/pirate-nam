"""Done-rule check for phys-rrtm-lw-taugb06 (CONTRACT section 6).

Compiles the kernel, runs the wrapper once on a full-size 954 x 835 x 60 random state with index ranges
that the Fortran can produce (jp 1..12, jt/jt1 1..4, inds 1..9, indf 1..3, indm 1..18, laytrop 0..nlay),
and checks that both band-6 outputs are finite.

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-lw-taugb06.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IM, JM, NLAY = 954, 835, 60


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb06 as glue

    kernels.module(glue.UNIT).compile()

    def ints(lo, hi, shape):
        return (cp.random.randint(lo, hi + 1, size=shape, dtype=cp.int32))

    def unif(shape, scale=1.0):
        return (cp.random.random(shape, dtype=cp.float32) * cp.float32(scale)).astype(cp.float32)

    col = (JM, IM)
    lay = (NLAY, JM, IM)
    coldry = (cp.random.random(lay, dtype=cp.float32) * cp.float32(1.0e20) + cp.float32(1.0e18)).astype(cp.float32)
    inp = {
        "laytrop": cp.random.randint(0, NLAY + 1, size=col, dtype=cp.int32),
        "jp": ints(1, 12, lay), "jt": ints(1, 4, lay), "jt1": ints(1, 4, lay),
        "indself": ints(1, 9, lay), "indfor": ints(1, 3, lay), "indminor": ints(1, 18, lay),
        "coldry": coldry,
        "colamt": (cp.random.random((glue.MAXGAS, NLAY, JM, IM), dtype=cp.float32) * cp.float32(1.0e20)),
        "wx": (cp.random.random((glue.MAXXSEC, NLAY, JM, IM), dtype=cp.float32) * cp.float32(1.0e18)),
        "selffac": unif(lay), "selffrac": unif(lay), "forfac": unif(lay), "forfrac": unif(lay),
        "minorfrac": unif(lay), "fac00": unif(lay), "fac01": unif(lay), "fac10": unif(lay), "fac11": unif(lay),
    }
    chi = (cp.random.random((glue.NREF, glue.NGAS_REF), dtype=cp.float32) * cp.float32(1.0e-4)
           + cp.float32(1.0e-6)).astype(cp.float32)
    ctx = SimpleNamespace(tables={glue.IN_KEY: inp, glue.TAB_KEY: {"chi_mls": chi}}, stream=None)

    t0 = time.time()
    out = glue.run(ctx)
    cp.cuda.Stream.null.synchronize()
    print(f"wrapper run {time.time() - t0:.2f} s on {IM} x {JM} x {NLAY}")

    ok = True
    for name in ("taug6", "fracs6"):
        a = out[name]
        bad = int((~cp.isfinite(a)).sum())
        print(f"{name} shape {tuple(a.shape)} nonfinite {bad} min {float(a.min()):.4e} max {float(a.max()):.4e}")
        ok = ok and bad == 0
    if not ok:
        raise SystemExit("phys-rrtm-lw-taugb06: NaN/Inf in output")
    print("phys-rrtm-lw-taugb06: ok")


if __name__ == "__main__":
    main()
