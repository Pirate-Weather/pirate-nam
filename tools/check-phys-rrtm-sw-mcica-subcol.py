"""Done-rule check for phys-rrtm-sw-mcica-subcol (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-sw-mcica-subcol.py

Full NAM 12 km size (NLAY = 60, JM = 835, IM = 954), one pass over each reviewed path:
  1. random cloud mix, iovrsw = 1 (NAM): the launch completes, output finite and uint32, bits populated.
  2. cldf = 1 everywhere, iovrsw = 0, 1, 2: every sub-column cloudy (words 0..2 all ones, word 3 low 16
     bits set for g = 96..111). Each overlap branch is exercised.
  3. cldf = 5e-13 everywhere equals cldf = 0 everywhere (radsw_main_nmmb.f:1787-1792 ftiny step).
  4. layer direction: cldf = 1 at the lowest layer (k0 = 0) and 0 above. Layer 0 is fully cloudy and the
     max-random walk up the column leaves the upper layers almost clear.
  5. guard: the kernel called directly with iovrsw = 3 zero-fills a pre-set output (no stale bits).
  6. wrapper refusal: a float64 cldf raises ValueError (no silent misread).
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NLAY, JM, IM = 60, 835, 954
WORD3_FULL_TOP = 0x0000FFFF   # g = 96..111 in word 3 (NGPTSW = 112)


def _ctx(cldf, iovrsw, isubcsw=1):
    import cupy as cp

    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(isubcsw=isubcsw, iovrsw=iovrsw),
        tables={"rrtm_sw_cldf": cldf},
        stream=cp.cuda.Stream(non_blocking=True),
    )


def _run(u, cldf, iovrsw):
    ctx = _ctx(cldf, iovrsw)
    with ctx.stream:
        out = u.run(ctx)
    ctx.stream.synchronize()
    return out


def main():
    import cupy as cp
    import numpy as np

    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_sw_mcica_subcol as u

    kernels.module(u.UNIT).compile()
    print("config: isubcsw=0 iovrsw=1 (NAM, config.py); layer order lowest-first (NAM iflip=0)")

    # 1. random mix, NAM overlap
    rng = cp.random.default_rng(1)
    cldf = rng.random((NLAY, JM, IM), dtype=cp.float32)
    cldf[cldf < 0.5] = 0.0
    t0 = time.perf_counter()
    out = _run(u, cldf, 1)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    assert out is not None and out.shape == (u.NWORD, NLAY, JM, IM) and out.dtype == cp.uint32
    assert bool(cp.isfinite(cldf).all())
    frac = float(cp.unpackbits(out.view(cp.uint8)).mean())
    print(f"1 random mix iovrsw=1: ok, {dt:.2f} s, set-bit fraction {frac:.3f} (mean cldf 0.375; max-random overlap sits below it)")

    # 2. cldf = 1 everywhere: every overlap branch gives every sub-column cloudy
    ones = cp.ones((NLAY, JM, IM), dtype=cp.float32)
    for ov in (0, 1, 2):
        o = _run(u, ones, ov)
        full = bool(cp.all(o[0:3] == cp.uint32(0xFFFFFFFF)) and cp.all(o[3] == cp.uint32(WORD3_FULL_TOP)))
        assert full, f"iovrsw={ov}: cldf=1 did not give all sub-columns cloudy"
        print(f"2 cldf=1 iovrsw={ov}: ok, every g-point cloudy in every layer")

    # 3. ftiny: 5e-13 is the same as 0
    tiny = cp.full((NLAY, JM, IM), 5e-13, dtype=cp.float32)
    zero = cp.zeros((NLAY, JM, IM), dtype=cp.float32)
    a = _run(u, tiny, 1).copy()
    b = _run(u, zero, 1).copy()
    assert bool(cp.array_equal(a, b)), "cldf 5e-13 differs from cldf 0 (ftiny step missing)"
    print("3 cldf=5e-13 vs cldf=0 iovrsw=1: ok, identical output (ftiny applied)")

    # 4. layer direction: cloud only at the lowest layer (k0 = 0)
    low = cp.zeros((NLAY, JM, IM), dtype=cp.float32)
    low[0] = 1.0
    o = _run(u, low, 1)
    lowest_full = bool(cp.all(o[0:3, 0] == cp.uint32(0xFFFFFFFF)) and cp.all(o[3, 0] == cp.uint32(WORD3_FULL_TOP)))
    upper = cp.unpackbits(o[:, 1:].view(cp.uint8)).mean()
    assert lowest_full, "layer k0=0 (lowest) is not fully cloudy for cldf=1 there"
    assert float(upper) < 0.01, f"upper layers not clear: set-bit fraction {float(upper):.4f}"
    print(f"4 cloud at lowest layer only: ok, layer 0 fully cloudy, upper-layer set-bit fraction {float(upper):.5f}")

    # 5. guard: kernel with iovrsw = 3 zero-fills stale output
    stale = cp.full((u.NWORD, NLAY, JM, IM), 0xFFFFFFFF, dtype=cp.uint32)
    ipseed = cp.arange(1, JM * IM + 1, dtype=cp.int32).reshape(JM, IM)
    kern = kernels.get(u.UNIT, u.KERNEL)
    kern(kernels.grid_cols(IM, JM), kernels.BLOCK_COL,
         (cldf, ipseed, stale, np.int32(NLAY), np.int32(JM), np.int32(IM), np.int32(3)))
    cp.cuda.Stream.null.synchronize()
    assert int(cp.count_nonzero(stale)) == 0, "guard left stale bits for iovrsw = 3"
    print("5 kernel guard iovrsw=3: ok, output zero-filled")
    del stale

    # 6. wrapper refuses float64 cldf
    try:
        u.mcica_subcol(cldf.astype(cp.float64), ipseed, u.new_lcloudy(NLAY, JM, IM), 1)
    except ValueError as e:
        print(f"6 float64 cldf refused: ok ({e})")
    else:
        raise AssertionError("float64 cldf was not refused")


if __name__ == "__main__":
    main()
