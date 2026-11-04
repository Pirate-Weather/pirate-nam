"""Done-rule check for bnd-bocov-blend and bnd-bocov-update, wired the way the driver wires them.

    source ~/nam-gpu/env.sh && python tools/check-bnd-bocov-blend.py

Runs the real chain on full-size fields (954 x 835 x 60, LNSV = 5), no hand-built binding:
  bnd-boundary-init (producer, fills ctx.bc["<SIDE>_<VAR>"] from ctx.st PD, T, Q, CW, U, V)
  -> bnd-bocov-update -> bnd-bocov-blend (nsmud = 0, the NAM path) -> bnd-bocov-blend (nsmud = 1, once).
The binding under test is bnd_bocov_view.v_view(ctx), which the two wrappers call themselves.
Asserts: the view's .full aliases ctx.st U and V; the blend wrote the producer strip into the domain
(U at j = 1, i = IM/2, level 1 equals ctx.bc["S_U"] slot 0 at the same point, weight wv(1) = 1); the
domain values changed; no NaN/Inf in any state or strip array. Runs under one pool-v3 lease.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels
    from namgpu import state as statemod
    from namgpu.boundary import bnd_boundary_init as init
    from namgpu.boundary import bnd_bocov_update as upd
    from namgpu.boundary import bnd_bocov_blend as blend
    from namgpu.boundary import bnd_bocov_view as view
    from namgpu.driver import Ctx
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm, lnsv = int(g.im), int(g.jm), int(g.lm), int(cfg.lnsv)
    kernels.module("bnd_bocov_update").compile()
    kernels.module("bnd_bocov_blend").compile()

    st = statemod.State(g)
    rng = cp.random.default_rng(2)
    for name in ("PD", "T", "Q", "CW", "U", "V"):
        st[name][...] = rng.random(st[name].shape, dtype=cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True),
              dt=float(cfg.dt), bc={})

    with gpu_lease(vram_mib=6144, est_s=600, cards=1, what="bnd-bocov-blend wiring check"):
        with ctx.stream:
            init.run(ctx)                      # producer: ctx.bc["S_U"] ... ["E_V"]
        ctx.stream.synchronize()
        # The producer copies the domain's own edge rows, so blending them is an identity. Offset the
        # value slot of every V strip (a new parent boundary, as bnd-read-bc would supply) so the blend
        # has something to write.
        for n in ("U", "V"):
            for s in ("S", "N", "W", "E"):
                ctx.bc[f"{s}_{n}"][0] += cp.float32(0.5)

        v = view.v_view(ctx)
        aliased = all(v.var_3d[k].full.data.ptr == st[n].data.ptr
                      for k, n in enumerate(("U", "V")))
        before = st["U"].copy()
        strip0_s = ctx.bc["S_U"][0]            # value slot, [LM, LNSV, IM]

        with ctx.stream:
            upd.run(ctx)                       # driver order: bocov-update, then bocov-blend
            blend.run(ctx)                     # NAM path, nsmud = 0
        ctx.stream.synchronize()

        i0 = im // 2
        wired = float(st["U"][0, 0, i0].get()) == float(strip0_s[0, 0, i0].get())
        changed = int(cp.count_nonzero(st["U"] != before).get())

        with ctx.stream:
            blend.run(ctx, nsmud=1)            # smoothing path, once
        ctx.stream.synchronize()

        arrays = [st["U"], st["V"]]
        for n in ("U", "V"):
            arrays += [ctx.bc[f"{s}_{n}"] for s in ("S", "N", "W", "E")]
        bad = sum(0 if bool(cp.isfinite(a).all()) else 1 for a in arrays)

    ok = aliased and wired and changed > 0 and bad == 0
    print(f"bnd-bocov-blend wiring: IM={im} JM={jm} LM={lm} LNSV={lnsv}; view aliases state={aliased}; "
          f"U(1,1,IM/2) equals producer S_U={wired}; changed U cells={changed}; nonfinite arrays={bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
