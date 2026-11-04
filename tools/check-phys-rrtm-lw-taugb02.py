"""Done-rule check for phys-rrtm-lw-taugb02 (CONTRACT.md section 6).

1. The kernel compiles (NVRTC through CuPy).
2. The launch runs over the FULL 954 x 835 x 60 domain (796,590 columns, 60 levels) on random but
   physically shaped inputs, in column chunks. Every written value is finite.
3. Placement: band-2 values land only in gpoint rows NS02..NS02+NG02-1 (10..21) of the full
   [LM, NGPTLW=140, ...] arrays. Rows outside that band stay zero.
4. The run() glue (ctx.st names -> launch) runs once on a small grid: finite outputs, same placement.

Inputs follow setcoef's clamps: jp lower 1..12, upper 13..58 (radlw_main_nmmb.f:2006 clamps jp to
1..58), jt and jt1 1..4 (jt1 at radlw_main_nmmb.f:2023), indself 1..9, indfor 1..3, laytrop 1..nlay-1.
"""
import sys
from types import SimpleNamespace

import cupy as cp
import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_lw_taugb02 as u

IM, JM, LM = 954, 835, 60
CHUNK = 8192
BAND = slice(u.NS02, u.NS02 + u.NG02)


def rand_inputs(ncol, nlay, rng):
    laytrop = rng.integers(1, nlay, size=ncol, dtype=np.int32)
    k = np.arange(nlay, dtype=np.int32)[:, None]
    lower = k < laytrop[None, :]
    jp = np.where(lower, rng.integers(1, 13, size=(nlay, ncol), dtype=np.int32),
                  rng.integers(13, 59, size=(nlay, ncol), dtype=np.int32)).astype(np.int32)
    jt = rng.integers(1, 5, size=(nlay, ncol), dtype=np.int32)
    jt1 = rng.integers(1, 5, size=(nlay, ncol), dtype=np.int32)
    indself = rng.integers(1, 10, size=(nlay, ncol), dtype=np.int32)
    indfor = rng.integers(1, 4, size=(nlay, ncol), dtype=np.int32)
    f = lambda: rng.random((nlay, ncol), dtype=np.float32)  # noqa: E731
    return dict(
        laytrop=cp.asarray(laytrop), jp=cp.asarray(jp), jt=cp.asarray(jt), jt1=cp.asarray(jt1),
        pavel=cp.asarray((1.0 + 999.0 * f()).astype(np.float32)),
        colamt=cp.asarray((1e20 * f()).astype(np.float32)),
        fac00=cp.asarray(f()), fac01=cp.asarray(f()), fac10=cp.asarray(f()), fac11=cp.asarray(f()),
        selffac=cp.asarray(f()), selffrac=cp.asarray(f()), indself=cp.asarray(indself),
        forfac=cp.asarray(f()), forfrac=cp.asarray(f()), indfor=cp.asarray(indfor))


def outside_band(a):
    """Count of nonzero values outside the band-2 rows (must be zero: placement)."""
    return int(cp.count_nonzero(a[:, :u.NS02])) + int(cp.count_nonzero(a[:, u.NS02 + u.NG02:]))


def main():
    kernels.module(u.UNIT).compile()
    print("compile: ok")
    ctx = SimpleNamespace(tables={}, stream=None)
    tabs = u.init_tables(ctx)

    rng = np.random.default_rng(0)
    ncol_total = IM * JM
    nonfinite = 0
    leak = 0
    band_nonzero = 0
    done = 0
    while done < ncol_total:
        n = min(CHUNK, ncol_total - done)
        inp = rand_inputs(n, LM, rng)
        taug = cp.zeros((LM, u.NGPTLW, n), dtype=cp.float32)
        fracs = cp.zeros((LM, u.NGPTLW, n), dtype=cp.float32)
        u.launch(tabs, ncol=n, nlay=LM, taug_lw=taug, fracs_lw=fracs, **inp)
        nonfinite += int(cp.count_nonzero(~cp.isfinite(taug))) + int(cp.count_nonzero(~cp.isfinite(fracs)))
        leak += outside_band(taug) + outside_band(fracs)
        band_nonzero += int(cp.count_nonzero(taug[:, BAND])) + int(cp.count_nonzero(fracs[:, BAND]))
        done += n
    cp.cuda.Stream.null.synchronize()
    print(f"full domain {ncol_total} columns x {LM} levels: nonfinite = {nonfinite}, "
          f"outside-band nonzero = {leak}, band-2 nonzero values = {band_nonzero}")

    class FakeSt:
        def __init__(self, fields):
            self.fields = fields

        def add(self, name, shape, dtype="float32"):
            if name not in self.fields:
                self.fields[name] = cp.zeros(shape, dtype=dtype)
            return self.fields[name]

    im, jm, lm = 60, 40, LM
    ncol = im * jm
    inp = rand_inputs(ncol, lm, rng)
    f = {}
    for name, key in [("JP", "jp"), ("JT", "jt"), ("JT1", "jt1"), ("INDSELF", "indself"),
                      ("INDFOR", "indfor"), ("PAVEL", "pavel"), ("COLAMT", "colamt"),
                      ("FAC00", "fac00"), ("FAC01", "fac01"), ("FAC10", "fac10"), ("FAC11", "fac11"),
                      ("SELFFAC", "selffac"), ("SELFFRAC", "selffrac"), ("FORFAC", "forfac"),
                      ("FORFRAC", "forfrac")]:
        f[name] = inp[key].reshape(lm, jm, im)
    f["LAYTROP"] = inp["laytrop"].reshape(jm, im)
    gctx = SimpleNamespace(st=FakeSt(f), grid=SimpleNamespace(im=im, jm=jm, lm=lm),
                           tables=ctx.tables, stream=None)
    out = u.run(gctx)
    bad = {k: int(cp.count_nonzero(~cp.isfinite(v))) for k, v in out.items()}
    leak_glue = {k: outside_band(v.reshape(lm, u.NGPTLW, ncol)) for k, v in out.items()}
    print("run() glue shapes:", {k: tuple(v.shape) for k, v in out.items()},
          "nonfinite:", bad, "outside-band:", leak_glue)
    ok = nonfinite == 0 and leak == 0 and band_nonzero > 0 and not any(bad.values()) \
        and not any(leak_glue.values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
