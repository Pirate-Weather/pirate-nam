"""Done-rule check for phys-rrtm-lw-taugb01 (CONTRACT section 6, parts 1 and 2).

Compiles the kernel with NVRTC, then runs the wrapper once on a FULL-SIZE random state
(954 x 835 x 60), and checks taug and fracs for NaN/Inf. Also checks the upper/lower split on the
output: fracs equals fracrefa for layers at or below laytrop and fracrefb above it.

The value check against the NAM Fortran TAUGB01 is tools/check-phys-rrtm-lw-taugb01-ref.py.

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-lw-taugb01.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IM, JM, LM = 954, 835, 60
MAXGAS = 6


class Store:
    """Minimal ctx.st: named device arrays with the State.add / __getitem__ calls the wrapper uses."""

    def __init__(self):
        self.d = {}

    def put(self, name, arr):
        self.d[name] = arr

    def add(self, name, shape, dtype="float32"):
        import cupy as cp

        if name not in self.d:
            self.d[name] = cp.zeros(shape, dtype=dtype)
        return self.d[name]

    def __getitem__(self, name):
        return self.d[name]


def build_state(cp, seed=1):
    """Full-size random TAUGB01 inputs. Returns (st, below) where below[k, j, i] is k <= laytrop."""
    rng = cp.random.default_rng(seed)
    st = Store()
    shape = (LM, JM, IM)
    laytrop = rng.integers(1, LM + 1, size=(JM, IM), dtype=cp.int32)
    kidx = cp.arange(1, LM + 1, dtype=cp.int32)[:, None, None]
    below = kidx <= laytrop[None]
    st.put("laytrop", laytrop)
    st.put("pavel", (rng.uniform(5.0, 1000.0, size=shape)).astype(cp.float32))
    # Fortran colamt(MAXLAY, MAXGAS) convention, [6, LM, JM, IM]; TAUGB01 reads index 0 (H2O).
    st.put("colamt", (rng.uniform(1e18, 1e23, size=(MAXGAS,) + shape)).astype(cp.float32))
    st.put("colbrd", (rng.uniform(1e18, 1e19, size=shape)).astype(cp.float32))
    st.put("scaleminorn2", (rng.uniform(0.5, 1.5, size=shape)).astype(cp.float32))
    for n in ("selffac", "selffrac", "forfac", "forfrac", "minorfrac", "fac00", "fac01", "fac10", "fac11"):
        st.put(n, rng.uniform(0.0, 1.0, size=shape, dtype=cp.float32))
    st.put("indself", rng.integers(1, 10, size=shape, dtype=cp.int32))     # inds, indsp <= MSF01 = 10
    st.put("indfor", rng.integers(1, 4, size=shape, dtype=cp.int32))       # indf, indfp <= MFR01 = 4
    st.put("indminor", rng.integers(1, 19, size=shape, dtype=cp.int32))    # indm, indmp <= MMN01 = 19
    jlo = rng.integers(1, 13, size=shape, dtype=cp.int32)                  # lower: jp 1..12
    jup = rng.integers(13, 59, size=shape, dtype=cp.int32)                 # upper: jp 13..58
    st.put("jp", cp.where(below, jlo, jup))
    st.put("jt", rng.integers(1, 6, size=shape, dtype=cp.int32))
    st.put("jt1", rng.integers(1, 6, size=shape, dtype=cp.int32))
    return st, below


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb01 as glue

    kernels.module(glue.UNIT).compile()
    st, below = build_state(cp)

    ctx = SimpleNamespace(cfg=None, tables={}, st=st, stream=cp.cuda.Stream(non_blocking=True), grid=None)
    t0 = time.perf_counter()
    with ctx.stream:
        glue.run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    taug = st["taug_b01"]
    fracs = st["fracs_b01"]
    for name, a in (("taug_b01", taug), ("fracs_b01", fracs)):
        assert bool(cp.isfinite(a).all()), f"NaN/Inf in {name}"
    # fracs must be the reference fractions per layer side (TAUGB01 :3669 and :3707).
    z = glue._tables(ctx)
    fa = z["rrtm_lw_kgb01_fracrefa"].reshape(-1, glue.NG01)[0]
    fb = z["rrtm_lw_kgb01_fracrefb"].reshape(-1, glue.NG01)[0]
    ref = cp.where(below[:, None], fa[None, :, None, None], fb[None, :, None, None])
    # fa/fb are flat over ig (length NG01); the reshape above takes the first row of [NG01]
    assert bool(cp.allclose(fracs, cp.broadcast_to(ref, fracs.shape))), "fracs does not match fracrefa/fracrefb"
    print(f"{glue.UNIT}: compiled, wrapper ran on {LM}x{JM}x{IM} in {dt:.2f} s, "
          f"taug/fracs finite, fracs split by laytrop ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
