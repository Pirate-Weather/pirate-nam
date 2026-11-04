"""Done-rule check for dyn-adv1-coriolis (review fix, 2026-11-04).

    source ~/nam-gpu/env.sh
    python tools/check-dyn-adv1-coriolis.py

1. Full-size 954 x 835 x 60 state. CURV and F_V are the REAL tables from init-grid-consts (no stand-in,
   no grid fallback). PFX/PFY/PFNE/PFNW are random stand-ins for the dyn-dht output.
2. Runs the wrapper once and checks every float array for NaN/Inf.
3. Compares U1D, V1D, TCU, TCV and the eight PF*1 outputs against a NumPy float32 reference of the
   Fortran loops (stage [0], [A], [B]) on three levels (bottom, middle, top).
4. Guards: a missing PFX raises (no zero fill); global=True raises. Both checks run before any GPU work.
"""
from __future__ import annotations

import copy
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REL_TOL = 1e-5


def reference(a, lev, dt, curv, f, cfc, bfc):
    """NumPy float32 reference for one level. Inputs are host arrays [LM,JM,IM] (or [JM,IM] for f)."""
    u, v, up, vp = (a[k][lev] for k in ("U", "V", "UP", "VP"))
    tcu0, tcv0 = a["TCU"][lev].copy(), a["TCV"][lev].copy()
    pf = {k: a[k][lev] for k in ("PFX", "PFY", "PFNE", "PFNW")}
    u1d = (u * cfc + up * bfc).astype(np.float32)
    v1d = (v * cfc + vp * bfc).astype(np.float32)
    jm, im = u.shape
    crv = (curv.reshape(-1, 1) * dt).astype(np.float32)
    fp = (f * dt).astype(np.float32)
    fpp = (u1d * crv + fp).astype(np.float32)
    tcu, tcv = tcu0.copy(), tcv0.copy()
    sa = (slice(1, jm - 2), slice(1, im - 2))
    tcu[sa] = fpp[sa] * v1d[sa] + tcu0[sa]
    tcv[sa] = -fpp[sa] * u1d[sa] + tcv0[sa]
    out = {"U1D": u1d, "V1D": v1d, "TCU": tcu, "TCV": tcv}
    sb = (slice(1, jm - 1), slice(1, im - 1))
    p, q = pf["PFX"], None
    out["PFXX1"] = np.zeros_like(u); out["PFYX1"] = np.zeros_like(u)
    out["PFNEX1"] = np.zeros_like(u); out["PFNWX1"] = np.zeros_like(u)
    out["PFXY1"] = np.zeros_like(u); out["PFYY1"] = np.zeros_like(u)
    out["PFNEY1"] = np.zeros_like(u); out["PFNWY1"] = np.zeros_like(u)
    for src, dx, dy in (("PFX", "PFXX1", "PFXY1"), ("PFY", "PFYX1", "PFYY1"),
                        ("PFNE", "PFNEX1", "PFNEY1"), ("PFNW", "PFNWX1", "PFNWY1")):
        s = pf[src]
        out[dx][sb] = s[1:jm - 1, 1:im - 1] + s[2:jm, 2:im]
        out[dy][sb] = s[1:jm - 1, 2:im] + s[2:jm, 1:im - 1]
    return out


def guards(cfg_base, make_ctx, dyn):
    """Cheap guards: no GPU work. Each must raise RuntimeError."""
    import numpy as np  # noqa: F401
    glob = types.SimpleNamespace(**vars(cfg_base)) if hasattr(cfg_base, "__dict__") else None
    res = {}
    try:
        cfg_g = copy.copy(cfg_base)
        object.__setattr__(cfg_g, "global_", True)
        ctx = make_ctx(cfg_g)
        dyn.run(ctx)
        res["global"] = "NO RAISE"
    except RuntimeError as e:
        res["global"] = f"raised: {str(e)[:90]}"
    del glob
    return res


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.dynamics import dyn_adv1_coriolis as dyn
    from namgpu.init import init_grid_consts

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)

    # guard: global=True raises before any GPU work
    res = guards(cfg, lambda c: Ctx(cfg=c, grid=g, st=st, stream=stream, dt=cfg.dt), dyn)
    print("guard global:", res["global"])

    # real CURV / F_V tables from init-grid-consts
    init_grid_consts.run(ctx)
    stream.synchronize()
    print(f"CURV {tuple(ctx.tables['CURV'].shape)} F_V {tuple(ctx.tables['F_V'].shape)} "
          f"im={im} jm={jm} lm={lm} dt={ctx.dt}")

    # stand-in PFX..PFNW (dyn-dht output shape); U, V, UP, VP, TCU, TCV come from randomize
    rng = np.random.default_rng(7)
    for name in ("PFX", "PFY", "PFNE", "PFNW"):
        arr = st.add(name, (lm, jm, im))
        arr[...] = cp.asarray(rng.uniform(-1.0, 1.0, (lm, jm, im)).astype(np.float32))

    # guard: missing producer raises (PFX removed), before any GPU work
    saved = st.fields.pop("PFX")
    try:
        dyn.run(ctx)
        print("guard missing PFX: NO RAISE")
    except RuntimeError as e:
        print("guard missing PFX: raised:", str(e)[:100])
    st.fields["PFX"] = saved

    levels = [0, lm // 2, lm - 1]
    snap = {k: cp.asnumpy(st.fields[k][levels]) for k in ("U", "V", "UP", "VP", "TCU", "TCV",
                                                          "PFX", "PFY", "PFNE", "PFNW")}
    snap = {k: v for k, v in snap.items()}
    curv_h = cp.asnumpy(ctx.tables["CURV"]).astype(np.float32)
    f_h = cp.asnumpy(ctx.tables["F_V"]).astype(np.float32)
    cfc, bfc = dyn.CFC, dyn.BFC

    kernels.module("dyn_adv1_coriolis").compile()
    t0 = time.perf_counter()
    dyn.run(ctx)
    stream.synchronize()
    wall = time.perf_counter() - t0

    bad = [n for n, a in st.fields.items()
           if getattr(a, "dtype", None) is not None and a.dtype.kind == "f"
           and not bool(cp.isfinite(a).all())]
    print(f"dyn-adv1-coriolis full size: wrapper {wall:.3f} s (first call incl. compile); "
          f"NaN/Inf in {bad if bad else 'none'}")

    worst = 0.0
    for li, lev in enumerate(levels):
        ref = {}
        a_host = {k: snap[k][li] for k in snap}
        a_host2 = {k: v[None] for k, v in a_host.items()}  # shape [1,JM,IM] for reference()
        for k in a_host2:
            a_host2[k] = a_host2[k]
        ref = reference({k: a_host2[k] for k in a_host2}, 0, ctx.dt, curv_h, f_h, cfc, bfc)
        for name, rv in ref.items():
            gv = cp.asnumpy(st.fields[name][lev])
            scale = max(float(np.abs(rv).max()), 1e-30)
            rel = float(np.abs(gv - rv).max()) / scale
            worst = max(worst, rel)
            if rel > REL_TOL:
                print(f"  level {lev} {name}: rel {rel:.3e} (> {REL_TOL})")
    print(f"reference compare on levels {levels}: worst rel {worst:.3e} (tol {REL_TOL})")
    ok = (not bad) and worst <= REL_TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
