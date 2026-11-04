"""Done-rule check for phys-fer-driver (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-fer-driver.py

1. compiles namgpu/kernels/phys_fer_driver.cu;
2. runs the real driver ONCE on a full-size random state (954 x 835 x 60) with the real FER_HIRES
   unit (phys-fer-fer-hires) and the real Ferrier init, and checks every float array for NaN/Inf.
No stub: the FER_HIRES launch needs fer_egcp01column_hr (EGCP01COLUMN_hr), which is not yet defined
in the tree; until it lands this check fails at the FER_HIRES kernel load, and it says so.
Takes the pool lease first (namgpu.steps.drv_gpu_lease.gpu_lease).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Every state field the driver and FER_HIRES read (the rest are created by st.add on first call).
NEEDED = {"T", "Q", "CW", "PD", "TRAIN", "DFI_TTEN", "PREC", "ACPREC", "ACPREC_TOT", "AVRAIN",
          "DSG2", "SGML2", "QC", "QR", "QS", "F_ICE", "F_RAIN", "F_RIMEF", "REFL_10CM", "SR"}


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.driver import phys_fer_driver as unit
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    if g.dsg1 is None:
        gridmod.vcoord_from_config(g, cfg)
    st = statemod.State(g, names=NEEDED).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    with gpu_lease(lane="nam-gpu-phys-fer-driver", vram_mib=6000, est_s=300,
                   what="phys-fer-driver done-rule check") as card:
        print("lease granted:", card, flush=True)
        kernels.module(unit.UNIT).compile()
        print("compiled phys_fer_driver.cu", flush=True)
        t0 = time.perf_counter()
        with ctx.stream:
            unit.run(ctx, use_radar=False)
        ctx.stream.synchronize()
        dt = time.perf_counter() - t0

    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print(f"driver run: {dt:.2f} s (includes first NVRTC compile)")
    if bad:
        print("NaN/Inf in:", bad)
        return 1
    print("no NaN/Inf in", sorted(n for n in st.fields if st[n].dtype.kind == "f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
