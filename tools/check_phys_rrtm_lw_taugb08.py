"""Done-rule check for phys-rrtm-lw-taugb08 (CONTRACT section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_phys_rrtm_lw_taugb08.py

Compiles the kernel (NVRTC) and runs the wrapper once over the FULL 954 x 835 x 60 column set on
random but physically ranged inputs, in column tiles (the full taug and fracs are 26.7 GB each,
more than the shared card has, so the wrapper writes them tile by tile). Every tile's output is
checked for NaN/Inf. No column test, no bitwise comparison.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LM, JM, IM = 60, 835, 954
NTILES = 32


def random_inputs(seed: int):
    import cupy as cp
    rs = cp.random.RandomState(seed)
    cell = (LM, JM, IM)
    # reference-level indices in their physical ranges (lower atmosphere jp 1..11, upper 13..57)
    jp_lo = rs.randint(1, 12, size=cell, dtype=cp.int32)
    jp_up = rs.randint(13, 58, size=cell, dtype=cp.int32)
    laytrop = rs.randint(15, 41, size=(JM, IM), dtype=cp.int32)
    lev = cp.arange(LM, dtype=cp.int32).reshape(LM, 1, 1)
    lower = lev < laytrop[None, :, :]
    jp = cp.where(lower, jp_lo, jp_up).astype(cp.int32)
    jt = rs.randint(1, 6, size=cell, dtype=cp.int32)
    jt1 = rs.randint(1, 6, size=cell, dtype=cp.int32)
    indself = rs.randint(1, 10, size=cell, dtype=cp.int32)
    indfor = rs.randint(1, 4, size=cell, dtype=cp.int32)
    indminor = rs.randint(1, 19, size=cell, dtype=cp.int32)

    def u(lo, hi, shape=cell):
        return (lo + (hi - lo) * rs.uniform(0.0, 1.0, shape)).astype(cp.float32)

    coldry = u(1.0e21, 2.5e25)
    colamt = cp.empty((6,) + cell, dtype=cp.float32)
    for n in range(6):   # one gas at a time: float64 temporaries stay one array deep
        colamt[n] = u(0.0, 1.0) * coldry * 1.0e-2
    wx = cp.empty((4,) + cell, dtype=cp.float32)
    for n in range(4):
        wx[n] = u(0.0, 1.0e-3) * coldry
    cp.get_default_memory_pool().free_all_blocks()
    return dict(
        laytrop=laytrop, jp=jp, jt=jt, jt1=jt1, indself=indself, indfor=indfor, indminor=indminor,
        coldry=coldry, colamt=colamt, wx=wx,
        selffac=u(0.0, 1.0), selffrac=u(0.0, 1.0), forfac=u(0.0, 1.0), forfrac=u(0.0, 1.0),
        minorfrac=u(0.0, 1.0), fac00=u(0.0, 1.0), fac10=u(0.0, 1.0), fac01=u(0.0, 1.0),
        fac11=u(0.0, 1.0),
    )


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb08 as wrap

    kernels.module("phys_rrtm_lw_taugb08").compile()
    print("compiled phys_rrtm_lw_taugb08")
    ctx = types.SimpleNamespace(tables={}, stream=cp.cuda.Stream(non_blocking=True))

    t0 = time.perf_counter()
    inputs = random_inputs(seed=1)
    cp.cuda.Device().synchronize()
    print(f"inputs ready on device: {time.perf_counter() - t0:.2f} s, "
          f"free {cp.cuda.runtime.memGetInfo()[0] / 2**30:.2f} GiB")

    nc = JM * IM
    tile = -(-nc // NTILES)
    bad = 0
    band_min, band_max = float("inf"), float("-inf")
    frac_min, frac_max = float("inf"), float("-inf")
    for c0 in range(0, nc, tile):
        ncol = min(tile, nc - c0)
        t1 = time.perf_counter()
        with ctx.stream:
            taug, fracs = wrap.run(ctx, inputs, c0=c0, ncol=ncol)
        ctx.stream.synchronize()
        band = taug[88:96]
        fb = fracs[88:96]
        nb = int((~cp.isfinite(band)).sum()) + int((~cp.isfinite(fb)).sum())
        bad += nb
        band_min = min(band_min, float(band.min()))
        band_max = max(band_max, float(band.max()))
        frac_min = min(frac_min, float(fb.min()))
        frac_max = max(frac_max, float(fb.max()))
        if c0 == 0 or nb:
            print(f"tile c0={c0} ncol={ncol}: nonfinite={nb} ({time.perf_counter() - t1:.2f} s)")
    print(f"all tiles: nonfinite={bad} taug[band 8] min={band_min:.4g} max={band_max:.4g} "
          f"fracs[band 8] min={frac_min:.4g} max={frac_max:.4g}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
