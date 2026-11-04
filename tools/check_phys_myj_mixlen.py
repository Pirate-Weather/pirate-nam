"""Done-rule check for unit phys-myj-mixlen (CONTRACT section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_phys_myj_mixlen.py

1. compiles namgpu/kernels/phys_myj_mixlen.cu through NVRTC;
2. runs the wrapper once on a full-size State (954 x 835 x 60) built by State(grid).randomize(),
   then sets the column geometry to physically ordered profiles (heights and pressures decrease
   with the level index, theta increases downward) so the square roots see plausible arguments;
   asserts that no float output contains NaN or Inf.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.myj import phys_myj_mixlen

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    # Physically ordered column geometry (top-down): heights fall with K, pressure rises with K.
    k = cp.arange(lm + 1, dtype=cp.float32)
    zint = (20000.0 - k * 330.0)[:, None, None] * cp.ones((1, jm, im), cp.float32)
    zint = zint + 50.0 * cp.random.default_rng(2).random((1, jm, im), dtype=cp.float32)
    st.add("MYJ_ZINT", (lm + 1, jm, im))[...] = zint
    pk = (1000.0 * 100.0 - k[:lm] * 1200.0)[:, None, None] * cp.ones((1, jm, im), cp.float32)
    st.add("MYJ_P", (lm, jm, im))[...] = pk
    st.add("MYJ_PSK", (lm, jm, im))[...] = pk + 600.0
    th = (300.0 + 0.004 * (k[:lm] * 330.0))[:, None, None] * cp.ones((1, jm, im), cp.float32)
    st.fields["THE"] = th + 0.5 * cp.random.default_rng(3).random((lm, jm, im), dtype=cp.float32)
    st.fields["THV"] = st.fields["THE"] - 1.0
    st.fields["Q2"][...] = 0.5 * st.fields["Q2"] + 0.05; st.fields["Q2"][40:] *= 0.01
    st.fields["EPSQ2"][...] = 0.02
    st.add("MYJ_LMH", (jm, im), dtype="int32")[...] = lm    # driver sets LMH = LM (MYJPBL)
    st.fields["EPSL"][...] = 0.5 + 0.01 * cp.arange(lm, dtype=cp.float32)
    cp.random.seed(4)
    st.fields["U"][...] = 10.0 * (st.fields["U"] - 0.5)
    st.fields["V"][...] = 10.0 * (st.fields["V"] - 0.5)

    print("compile:", kernels.module("phys_myj_mixlen").compile() or "ok", flush=True)
    outs = phys_myj_mixlen.run(ctx)
    ctx.stream.synchronize()

    bad = [n for n, a in outs.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print("LPBL range", int(outs["MYJ_LPBL"].min()), int(outs["MYJ_LPBL"].max()),
          "LMXL range", int(outs["MYJ_LMXL"].min()), int(outs["MYJ_LMXL"].max()))
    if bad:
        print("NaN/Inf in:", bad)
        return 1
    print("ok: no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
