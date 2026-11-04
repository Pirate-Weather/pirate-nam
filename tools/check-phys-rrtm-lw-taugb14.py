"""Done-rule check for phys-rrtm-lw-taugb14 (CONTRACT section 6).

Compiles the kernel, runs the wrapper once on a full-size 954 x 835 x 60 random state with index ranges
that the Fortran can produce (lower layers jp 1..12, upper layers jp 13..58 so that the absb indices
stay in 1..235; jt/jt1 1..4; inds 1..9; indf 1..3; laytrop 0..nlay), and checks that both band-14
outputs are finite.

    source env.sh && python tools/check-phys-rrtm-lw-taugb14.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IM, JM, NLAY = 954, 835, 60
TILE = 32768   # columns per tile: the full 140-row band output would be 26.7 GB


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb14 as glue

    kernels.module(glue.UNIT).compile()

    lay = (NLAY, JM, IM)
    col = (JM, IM)

    def ints(lo, hi, shape):
        return cp.random.randint(lo, hi + 1, size=shape, dtype=cp.int32)

    def unif(shape, scale=1.0):
        return (cp.random.random(shape, dtype=cp.float32) * cp.float32(scale)).astype(cp.float32)

    laytrop = cp.random.randint(0, NLAY + 1, size=col, dtype=cp.int32)
    below = cp.arange(NLAY, dtype=cp.int32).reshape(NLAY, 1, 1) < laytrop[None]
    jp = cp.where(below, ints(1, 12, lay), ints(13, 58, lay)).astype(cp.int32)
    inp = {
        "laytrop": laytrop, "jp": jp,
        "jt": ints(1, 4, lay), "jt1": ints(1, 4, lay),
        "indself": ints(1, 9, lay), "indfor": ints(1, 3, lay),
        "colamt": cp.random.random((glue.MAXGAS, NLAY, JM, IM), dtype=cp.float32) * cp.float32(1.0e20),
        "selffac": unif(lay), "selffrac": unif(lay), "forfac": unif(lay), "forfrac": unif(lay),
        "fac00": unif(lay), "fac01": unif(lay), "fac10": unif(lay), "fac11": unif(lay),
    }
    ctx = SimpleNamespace(tables={glue.IN_KEY: inp}, stream=None)

    t0 = time.time()
    band = [glue.NS14, glue.NS14 + 1]
    ok = True
    nc = JM * IM
    for c0 in range(0, nc, TILE):
        n = min(TILE, nc - c0)
        out = glue.run(ctx, c0=c0, ncol=n)
        for name in ("taug", "fracs"):
            a = out[name]
            bad = int((~cp.isfinite(a)).sum())
            # placement: band 14 lands at Fortran rows 134..135 and no other g-point row is written
            stray = int((cp.delete(a, band, axis=0) != 0).sum())
            if bad or stray:
                print(f"{name} tile {c0}: nonfinite {bad} stray {stray}")
            ok = ok and bad == 0 and stray == 0
            if c0 == 0:
                print(f"{name} tile0 shape {tuple(a.shape)} rows {band} min {float(a[band].min()):.4e} max {float(a[band].max()):.4e}")
    cp.cuda.Stream.null.synchronize()
    print(f"wrapper run {time.time() - t0:.2f} s on {IM} x {JM} x {NLAY} in {-(-nc // TILE)} tiles; rows {band} written, other rows zero: {ok}")

    ok_all = ok
    if not ok_all:
        raise SystemExit("phys-rrtm-lw-taugb14: NaN/Inf in output")
    print("phys-rrtm-lw-taugb14: ok")


if __name__ == "__main__":
    main()
