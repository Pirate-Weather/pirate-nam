"""Done-rule check for phys-fer-calmict (CONTRACT section 6, parts 1 and 2). Full-size state, one run.

    source ~/nam-gpu/env.sh && python tools/check_phys_fer_calmict.py

Compiles namgpu/kernels/phys_fer_calmict.cu through NVRTC, then runs the wrapper once on a full-size
float32 [60, 835, 954] random but plausible state and asserts the output is finite. The MASSI, TBPVS and
scalar tables are fixtures (phys-fer-init is a separate unit); the test needs only finite inputs and a
finite result.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

LM, JM, IM = 60, 835, 954
RD = 287.04


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.fer_hires import phys_fer_calmict as unit

    rng = np.random.default_rng(7)
    shape = (LM, JM, IM)
    f32 = np.float32
    T = rng.uniform(205.0, 300.0, size=shape).astype(f32)
    P = rng.uniform(2.0e4, 1.0e5, size=shape).astype(f32)
    Q = rng.uniform(1.0e-6, 2.0e-2, size=shape).astype(f32)
    C = (rng.uniform(0.0, 1.0e-3, size=shape) * (rng.uniform(size=shape) < 0.5)).astype(f32)
    FI = rng.uniform(0.0, 1.0, size=shape).astype(f32)
    FR = rng.uniform(0.0, 1.0, size=shape).astype(f32)
    FS = rng.uniform(1.0, 12.0, size=shape).astype(f32)
    CUREFL = (rng.uniform(0.0, 40.0, size=(JM, IM)) * (rng.uniform(size=(JM, IM)) < 0.2)).astype(f32)

    # Fixtures. MASSI index k = m - 50 for m = 50..1000 microns.
    m = np.arange(50, 1001, dtype=np.float64)
    massi = (500.0 * np.pi / 6.0 * (m * 1.0e-6) ** 3).astype(f32)
    # TBPVS on the NX=7501 grid, T = 180 + (j-1)*0.02 K (C1XPVS + C2XPVS*T = j).
    tgrid = 180.0 + (np.arange(7501, dtype=np.float64)) * 0.02
    tbpvs = (0.61078 * np.exp(17.2694 * (tgrid - 273.16) / (tgrid - 35.86))).astype(f32)
    scalars = {
        "C1XPVS": 1.0 - 50.0 * 180.0, "C2XPVS": 50.0,
        "RQR_DRMIN": 1.0e-4, "RQR_DRMAX": 1.0e-2, "CN0R0": 1.0e6 / (1.0e7 ** 0.25),
        "CN0R_DMRMIN": 1.0 / (np.pi * 1000.0 * (0.05e-3) ** 4),
        "CN0R_DMRMAX": 1.0 / (np.pi * 1000.0 * (1.0e-3) ** 4),
    }

    ctx = SimpleNamespace(
        stream=cp.cuda.Stream(non_blocking=True),
        tables={"fer_massi": cp.asarray(massi), "fer_gpvs_tbpvs": cp.asarray(tbpvs),
                "fer_scalars": scalars},
    )
    args = [cp.asarray(a) for a in (P, T, Q, C, FI, FR, FS, CUREFL)]

    t0 = time.perf_counter()
    kernels.module(unit.UNIT).compile()
    with ctx.stream:
        out = unit.run(ctx, *args)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    finite = bool(cp.isfinite(out).all())
    dbz = cp.asnumpy(out) if finite else None
    print(f"phys-fer-calmict: compiled, finite={finite}, {dt:.2f} s, "
          f"dBZ range {float(out.min()):.1f} .. {float(out.max()):.1f}" if finite
          else f"phys-fer-calmict: NaN/Inf in output ({dt:.2f} s)")
    return 0 if finite and dbz is not None else 1


if __name__ == "__main__":
    sys.exit(main())
