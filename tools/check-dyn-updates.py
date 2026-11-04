"""Done-rule check for dyn-updates (CONTRACT.md section 6), with the review's gaps.

    source env.sh
    python tools/check-dyn-updates.py

Checks, on a full-size random State (954 x 835 x 60, configured microphysics):
  1. the kernel compiles;
  2. the driver's q_tracer cadence is every step for a regional run (IDTADTQ=1);
  3. the driver resolves the two UPDATES ranges from INDX names: Q call 1..1, tracers call 2..3;
  4. after each call, in-range interior points satisfy s_new == s_old + tend_old exactly, tend is
     zeroed there, out-of-range slots are untouched, and the boundary rows/columns are untouched
     for both arrays;
  5. no NaN or Inf in TRACERS or TRACERS_TEND.
The GPU is taken through the pool lease (namgpu/steps/drv_gpu_lease.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.dynamics import dyn_updates
    from namgpu.steps.drv_gpu_lease import gpu_lease
    from namgpu.steps.drv_tracer_index import host_index

    cfg = config.load()
    assert cfg.global_ is False, "the driver check assumes the regional parent"
    g = gridmod.build(cfg)
    kernels.module("dyn_updates").compile()
    print("kernel compiled", flush=True)

    with gpu_lease(vram_mib=12000, est_s=600, what="check-dyn-updates full-size"):
        st = statemod.State(g).randomize(seed=1)
        ntr = int(st.TRACERS.shape[0])
        tab = host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))
        assert ntr == tab["NUM_TRACERS_TOTAL"], (ntr, tab["NUM_TRACERS_TOTAL"])
        lm, jm, im = int(g.lm), int(g.jm), int(g.im)
        tend = st.add("TRACERS_TEND", (ntr, lm, jm, im))
        host = np.random.default_rng(3).random(tend.shape, dtype=np.float32)
        tend[...] = cp.asarray(host)
        del host
        cp.cuda.get_current_stream().synchronize()
        stream = cp.cuda.Stream(non_blocking=True)
        ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
        drv = Driver(ctx)

        # 2 and 3: driver-side cadence and range resolution
        ctx.ntimestep = 1
        assert drv._due("q_tracer"), "IDTADTQ must be 1 for a regional run"
        ranges = {}
        for e in drv.plan:
            if e[0] == "dyn-updates":
                ranges[e[1]] = {k: drv._indx(v) for k, v in e[2].items()}
        assert ranges == {"q_tracer": {"kss": 1, "kse": 1}, "tracers": {"kss": 2, "kse": 3}}, ranges
        print("driver: q_tracer every step; dyn-updates ranges", ranges, flush=True)

        # 4: exact update, out-of-range and edge untouched, per call
        edge = cp.ones((jm, im), dtype=bool)
        edge[1:jm - 1, 1:im - 1] = False          # zero-based interior [1, JM-2] x [1, IM-2]

        def check(kss, kse):
            s0 = st.TRACERS.copy()
            t0 = tend.copy()
            # the snapshots run on the null stream, the kernel on ctx.stream: order them first
            cp.cuda.get_current_stream().synchronize()
            dyn_updates.run(ctx, kss=kss, kse=kse)
            stream.synchronize()
            k0, k1 = kss - 1, kse - 1
            inr = slice(k0, k1 + 1)
            out = [k for k in range(ntr) if not (k0 <= k <= k1)]
            ii, jj = slice(1, im - 1), slice(1, jm - 1)
            exp_s = s0[inr, :, jj, ii] + t0[inr, :, jj, ii]
            assert bool(cp.array_equal(st.TRACERS[inr, :, jj, ii], exp_s)), f"interior s, slots {kss}..{kse}"
            assert float(cp.abs(tend[inr, :, jj, ii]).max()) == 0.0, f"interior tend not zeroed, slots {kss}..{kse}"
            if out:
                assert bool(cp.array_equal(st.TRACERS[out], s0[out])), "out-of-range TRACERS changed"
                assert bool(cp.array_equal(tend[out], t0[out])), "out-of-range TRACERS_TEND changed"
            for a, b in ((st.TRACERS[inr], s0[inr]), (tend[inr], t0[inr])):
                changed = (a != b).any(axis=(0, 1))   # [jm, im]
                assert not bool((changed & edge).any()), f"edge points changed, slots {kss}..{kse}"
            print(f"update slots {kss}..{kse}: exact in interior, tend zeroed, edges and other slots untouched",
                  flush=True)

        check(2, 3)   # tracers call
        check(1, 1)   # q call
        for name, a in (("TRACERS", st.TRACERS), ("TRACERS_TEND", tend)):
            assert bool(cp.isfinite(a).all()), f"NaN/Inf in {name}"
        print("no NaN/Inf in TRACERS or TRACERS_TEND", flush=True)
    print("CHECK PASS dyn-updates")


if __name__ == "__main__":
    main()
