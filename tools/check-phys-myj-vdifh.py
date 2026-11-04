"""Done-rule check for unit phys-myj-vdifh (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-myj-vdifh.py

1. compiles namgpu/kernels/phys_myj_vdifh.cu through NVRTC (prep, VDIFH and tendency kernels);
2. runs the wrapper once on the full-size state (954 x 835 x 60): State.randomize fills the NAM
   tables, and the MYJPBL inputs the unit does not own (MYJ_ZINT, MYJ_AKH) are set with plausible
   values (interface heights decreasing with level index, so the Fortran divisions are defined;
   AKH zero at level LM, which the Fortran never reads). Asserts that every array the wrapper
   writes contains no NaN or Inf.
Takes the GPU pool lease first (namgpu.steps.drv_gpu_lease.gpu_lease).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WRITTEN = ("THK", "QK", "CWMK", "ZHK", "RHOK", "AKHK", "AKHS_DENS",
           "THZ0", "QZ0", "QSH", "RTHBLTEN", "RQBLTEN", "RQCBLTEN")


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.myj import phys_myj_vdifh as w
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    rng = cp.random.default_rng(1)
    f = cp.float32
    # caller fields that the turbulence glue (phys_turbl_driver) creates before MYJPBL
    pinh = st.add("PHINT", (lm + 1, jm, im))
    pinh[...] = cp.linspace(2000.0, 101000.0, lm + 1, dtype=f).reshape(lm + 1, 1, 1)  # PINH
    st.add("PHMID", (lm, jm, im))[...] = 0.5 * (pinh[1:] + pinh[:-1])
    st.add("TH", (lm, jm, im))[...] = 280 + 40 * rng.random((lm, jm, im), dtype=f)
    st.add("T", (lm, jm, im))[...] = 200 + 110 * rng.random((lm, jm, im), dtype=f)
    st.add("Q", (lm, jm, im))[...] = 1e-3 + 1e-2 * rng.random((lm, jm, im), dtype=f)
    st.add("QC", (lm, jm, im))[...] = 1e-4 * rng.random((lm, jm, im), dtype=f)
    st.add("TSFC", (jm, im))[...] = 270 + 30 * rng.random((jm, im), dtype=f)
    st.add("XLAND", (jm, im))[...] = 1 + (rng.random((jm, im), dtype=f) > 0.3).astype(f)
    st.add("SICE", (jm, im))[...] = 0.5 * rng.random((jm, im), dtype=f)
    st.add("SNOW", (jm, im))[...] = rng.random((jm, im), dtype=f)
    st.add("CHKLOWQ", (jm, im))[...] = 1 + 0.1 * rng.random((jm, im), dtype=f)
    st.add("ELFLX", (jm, im))[...] = 1e-4 * rng.random((jm, im), dtype=f)
    st.add("CT", (jm, im))[...] = 0.05 * rng.random((jm, im), dtype=f)
    st.add("AKHS", (jm, im))[...] = 1 + 5 * rng.random((jm, im), dtype=f)
    st.add("THZ0", (jm, im))[...] = 290 + 5 * rng.random((jm, im), dtype=f)
    st.add("QZ0", (jm, im))[...] = 0.008 * rng.random((jm, im), dtype=f)
    st.add("QSH", (jm, im))[...] = 0.008 * rng.random((jm, im), dtype=f)
    # MYJ_ZINT: the MYJPBL ZINT, decreasing with level index; MYJ_AKH: difcof output, zero at LM
    z = st.add("MYJ_ZINT", (lm + 1, jm, im))
    z[...] = cp.linspace(25000, 0, lm + 1, dtype=f).reshape(lm + 1, 1, 1)
    akh = st.add("MYJ_AKH", (lm, jm, im))
    akh[...] = 0.5 + 50 * rng.random((lm, jm, im), dtype=f)
    akh[lm - 1] = 0                                                         # Fortran never reads LM
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    print("compile:", kernels.module("phys_myj_vdifh").compile() or "ok", flush=True)
    with gpu_lease(lane="nam-gpu-phys-myj-vdifh", vram_mib=4000, est_s=300,
                   what="phys-myj-vdifh done-rule check"):
        t0 = time.perf_counter()
        with ctx.stream:
            w.run(ctx)
        ctx.stream.synchronize()
        dt = time.perf_counter() - t0

    bad = [n for n in WRITTEN if not bool(cp.isfinite(st[n]).all())]
    for n in ("THK", "QK", "CWMK", "RTHBLTEN", "RQBLTEN", "RQCBLTEN"):
        a = st[n]
        print(f"{n}: min={float(a.min()):.6g} max={float(a.max()):.6g}")
    print(f"grid {im} x {jm} x {lm}, run {dt:.3f} s, NaN/Inf in {bad or 'none'}")
    if bad:
        return 1
    print("ok: no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
