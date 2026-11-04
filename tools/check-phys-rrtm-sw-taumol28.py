"""Done-rule check for phys-rrtm-sw-taumol28 (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-sw-taumol28.py

Compiles the kernel and runs the wrapper once at full NAM 12 km size (954 x 835 x 60) on random but
plausible inputs. Checks that the planes this unit writes (taug and taur, the six band-28 g-point
planes) hold no NaN or Inf. Only the arrays this unit reads or writes are allocated, to fit a short
pool lease.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_sw_taumol28 as u

    im, jm, nlay = 954, 835, 60
    kernels.module(u.UNIT).compile()
    rng = cp.random.default_rng(28)
    f = lambda: rng.random((nlay, jm, im), dtype=cp.float32)
    ints = lambda lo, hi: rng.integers(lo, hi + 1, size=(nlay, jm, im), dtype=cp.int32)

    laytrop = rng.integers(1, nlay + 1, size=(jm, im), dtype=cp.int32)
    inp = {
        "colamt": rng.random((6, nlay, jm, im), dtype=cp.float32) * cp.float32(1.0e20),
        "colmol": f() * cp.float32(1.0e25),
        "laytrop": laytrop,
        "jp": ints(1, 58), "jt": ints(1, 5), "jt1": ints(1, 5),
        "fac00": f(), "fac01": f(), "fac10": f(), "fac11": f(),
    }
    tab = u.load_tables()
    out = u.alloc_outputs(nlay, jm, im)

    t0 = time.perf_counter()
    u.taumol28(inp, out, tab)
    cp.cuda.Device().synchronize()
    dt = time.perf_counter() - t0

    bad = 0
    for name in ("taug", "taur"):
        a = out[name]
        n_bad = int((~cp.isfinite(a)).sum())
        bad += n_bad
        print(f"{name}: shape {tuple(a.shape)} nonfinite {n_bad} max {float(cp.max(a)):.6g}")
    print(f"launch + sync {dt:.3f} s, minval(laytrop) {int(cp.min(laytrop))}, nonfinite total {bad}")
    if bad:
        raise SystemExit(1)
    print("PASS phys-rrtm-sw-taumol28: compiled, full-size run, no NaN/Inf")


if __name__ == "__main__":
    main()
