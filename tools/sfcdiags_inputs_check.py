"""One-shot done-rule check for phys-noah-sfcdiags: the full-size random State plus the turbl
scratch inputs SFCDIAGS reads (TSFC, CHS2, CQS2, TWBS, QWBS, QSH), filled with plausible values."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    shape2 = (int(g.jm), int(g.im))
    rng = cp.random.default_rng(7)
    if "PHINT" not in st.fields:
        st.add("PHINT", (int(g.lm) + 1, shape2[0], shape2[1]), "float32")[...] = st.fields["PINT"]
    st.add("TSFC", shape2, "float32")[...] = (250.0 + 60.0 * rng.random(shape2)).astype(cp.float32)
    st.add("CHS2", shape2, "float32")[...] = (1e-4 + 4e-3 * rng.random(shape2)).astype(cp.float32)
    st.add("CQS2", shape2, "float32")[...] = (1e-4 + 4e-3 * rng.random(shape2)).astype(cp.float32)
    st.add("TWBS", shape2, "float32")[...] = (-200.0 + 600.0 * rng.random(shape2)).astype(cp.float32)
    st.add("QWBS", shape2, "float32")[...] = (-2e-4 + 4e-4 * rng.random(shape2)).astype(cp.float32)
    st.add("QSH", shape2, "float32")[...] = (0.002 + 0.018 * rng.random(shape2)).astype(cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    kernels.module("phys_noah_sfcdiags").compile()
    with ctx.stream:
        out = importlib.import_module("namgpu.physics.noah.phys_noah_sfcdiags").run(ctx)
    ctx.stream.synchronize()
    bad = [n for n in ("T2", "TH2X", "Q2X") if not bool(cp.isfinite(st.fields[n]).all())]
    print("outputs:", sorted(out), "shape", tuple(out["TH2X"].shape))
    print("TH2X min/max:", float(st.fields["TH2X"].min()), float(st.fields["TH2X"].max()))
    print("Q2X min/max:", float(st.fields["Q2X"].min()), float(st.fields["Q2X"].max()))
    print("nonfinite:", bad or "none")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
