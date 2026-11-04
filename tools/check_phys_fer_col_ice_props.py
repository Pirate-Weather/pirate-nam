"""Done-rule check for phys-fer-col-ice-props (CONTRACT section 6, parts 1 and 2). Full-size state, one run.

    source ~/nam-gpu/env.sh && python tools/check_phys_fer_col_ice_props.py

Compiles namgpu/kernels/phys_fer_col_ice_props.cu through NVRTC, then runs the wrapper once on a full-size
float32 [60, 835, 954] random but plausible state and asserts every output slot is finite. The MASSI,
VSNOWI, VEL_RF and scalar tables are fixtures here (phys-fer-init is a separate unit); the test needs only
finite inputs and a finite result.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

LM, JM, IM = 60, 835, 954
NSLOT = 37
RD = 287.04


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.fer_hires import phys_fer_col_ice_props as unit

    rng = np.random.default_rng(7)
    shape = (LM, JM, IM)
    f32 = np.float32
    TK = rng.uniform(205.0, 300.0, size=shape).astype(f32)
    PMID = rng.uniform(2.0e4, 1.0e5, size=shape).astype(f32)
    RHO = (PMID / (RD * TK)).astype(f32)
    THICK = rng.uniform(100.0, 800.0, size=shape).astype(f32)
    QI = (rng.uniform(0.0, 2.0e-4, size=shape) * (rng.uniform(size=shape) < 0.7)).astype(f32)
    QR = (rng.uniform(0.0, 1.0e-3, size=shape) * (rng.uniform(size=shape) < 0.5)).astype(f32)
    ARAIN = rng.uniform(0.0, 1.0e-3, size=shape).astype(f32)
    ASNOW = rng.uniform(0.0, 1.0e-3, size=shape).astype(f32)
    RIMEF = rng.uniform(1.0, 12.0, size=shape).astype(f32)
    CLEAR = (rng.uniform(size=shape) < 0.3).astype(np.int32)   # CLEAR levels exercise the early-return path
    ASNOW = np.where(rng.uniform(size=shape) < 0.5, ASNOW * 1.0e-9, ASNOW).astype(f32)  # some ASNOW <= CLIMIT

    # Fixtures. Index k = m - 50 for m = MDImin..MDImax (50..1000 microns).
    m = np.arange(50, 1001, dtype=np.float64)
    d = m * 1.0e-6
    massi = (500.0 * np.pi / 6.0 * d ** 3).astype(f32)
    vsnowi = (0.5 + 0.002 * m).astype(f32)
    vel_rf = rng.uniform(1.0, 2.0, size=(41, 8)).astype(f32)

    ctx = SimpleNamespace(
        st={
            "T": cp.asarray(TK), "PMID": cp.asarray(PMID), "RHO": cp.asarray(RHO),
            "THICK": cp.asarray(THICK), "QI": cp.asarray(QI), "QR": cp.asarray(QR),
            "ARAIN_COL": cp.asarray(ARAIN), "ASNOW_COL": cp.asarray(ASNOW), "F_RIMEF": cp.asarray(RIMEF),
            "FER_SC_RHO": cp.asarray(RHO), "FER_SC_CLEAR": cp.asarray(CLEAR),
        },
        stream=cp.cuda.Stream(non_blocking=True),
        cfg={"dtph": 50.0},
        tables={
            "fer_massi": cp.asarray(massi), "fer_vsnowi": cp.asarray(vsnowi),
            "fer_vel_rf": cp.asarray(vel_rf),
            "fer_scalars": {"RFmx1": 1.0e4, "RFmax": 40.0, "RH_NgC": 1.0e4, "RH_NgT": 5.0e3,
                            "RQhail": 1.0e-3, "AVhail": 3.0, "BVhail": 0.5},
        },
    )
    from namgpu.steps.drv_gpu_lease import gpu_lease

    kernels.module(unit.UNIT).compile()
    t0 = time.perf_counter()
    with gpu_lease(vram_mib=12288, est_s=600, cards=1, what="phys-fer-col-ice-props full-size NaN check"):
        with ctx.stream:
            out = unit.run(ctx)
        ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    assert out.shape == (NSLOT,) + shape, out.shape
    bad = [s for s in range(NSLOT) if not bool(cp.isfinite(out[s]).all())]
    print(f"output slots {NSLOT}, shape {tuple(out.shape)}, non-finite slots: {bad or 'none'}")
    print(f"ICE_logical fraction {float(out[0].mean()):.3f}, RAIN_logical fraction {float(out[1].mean()):.3f}")
    print(f"RimeF1 min/max {float(out[3].min()):.3f} / {float(out[3].max()):.3f}")
    print(f"wrapper wall {dt:.3f} s, full size {TK.size} points")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
