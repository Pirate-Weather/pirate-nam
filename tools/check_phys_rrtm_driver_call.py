"""Done-rule check for phys-rrtm-driver-call (CONTRACT section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_phys_rrtm_driver_call.py

Compiles the kernel and runs the wrapper ONCE on a full-size (954 x 835 x 60) random-but-plausible
state, then checks every float output for NaN/Inf. The inputs the generic harness does not carry
(P_PHY, TAUTOTAL, CUTOP, CUBOT, the GPVS tables) are built here at full size and placed where the
caller chain puts them (ctx.st and the prep output), so the bindings are what is exercised.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_driver_call as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    st = {}   # the unit reads its inputs from the explicit dict below; no full State needed
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    rng = cp.random.default_rng(7)
    s3 = (lm, jm, im)
    s2 = (jm, im)
    # Plausible pressures: interfaces from ~1 hPa top to ~1000 hPa, Pa; mid-layer between neighbours.
    pint = cp.linspace(100.0, 101000.0, lm + 1, dtype=cp.float32)
    p8w = cp.broadcast_to(pint[:, None, None], (lm + 1, jm, im)).astype(cp.float32).copy()
    p_phy = (0.5 * (p8w[:-1] + p8w[1:])).astype(cp.float32)
    q = (rng.random(s3, dtype=cp.float32) * 0.02).astype(cp.float32)
    t = (220.0 + rng.random(s3, dtype=cp.float32) * 80.0).astype(cp.float32)
    inputs = {
        "PINT": p8w, "P_PHY": p_phy, "T": t, "Q": q,
        "CW": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "O3": (rng.random(s3, dtype=cp.float32) * 1e-6).astype(cp.float32),
        "OMGALF": (rng.random(s3, dtype=cp.float32) * 0.1 - 0.05).astype(cp.float32),
        "CLDFRA": rng.random(s3, dtype=cp.float32),
        "TAUTOTAL": cp.zeros(s3, dtype=cp.float32),
        "F_ICE": rng.random(s3, dtype=cp.float32), "F_RAIN": rng.random(s3, dtype=cp.float32),
        "F_RIMEF": 1.0 + rng.random(s3, dtype=cp.float32),
        "QC": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "QI": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "QS": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "QR": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "QG": (rng.random(s3, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "NI": (rng.random(s3, dtype=cp.float32) * 1e3).astype(cp.float32),
        "CUTOP": (rng.random(s2, dtype=cp.float32) * lm).astype(cp.float32),
        "CUBOT": (rng.random(s2, dtype=cp.float32) * lm).astype(cp.float32),
        "CUPPT": (rng.random(s2, dtype=cp.float32) * 1e-4).astype(cp.float32),
        "GLON": (rng.random(s2, dtype=cp.float32) * 6.28).astype(cp.float32),
        "GLAT": (rng.random(s2, dtype=cp.float32) * 1.2 - 0.6).astype(cp.float32),
        "TSKIN": (260.0 + rng.random(s2, dtype=cp.float32) * 40.0).astype(cp.float32),
        "Z0": (rng.random(s2, dtype=cp.float32) * 0.1).astype(cp.float32),
        "SICE": rng.random(s2, dtype=cp.float32), "SI": rng.random(s2, dtype=cp.float32) * 100.0,
        "SNOWC": rng.random(s2, dtype=cp.float32), "MXSNAL": rng.random(s2, dtype=cp.float32),
        "STDH": rng.random(s2, dtype=cp.float32) * 100.0,
        "SM": (rng.random(s2, dtype=cp.float32) > 0.5).astype(cp.float32),
        "ALBEDO": rng.random(s2, dtype=cp.float32),
    }

    # Saturation-vapour table: the phys-fer-gpvs unit builds the real GPVS table; run it here.
    from namgpu.physics.fer_hires import phys_fer_gpvs
    phys_fer_gpvs.run(ctx)

    # Caller-chain binding (the defect this check guards): P_PHY is in ctx.st (phys-rrtm-grrad-setup
    # writes it each call), TAUTOTAL/CUTOP/CUBOT are in the phys-rrtm-driver-prep output dict. None of
    # the four is passed explicitly, and the control values come from cfg and ctx.dt.
    ctx.st["P_PHY"] = inputs.pop("P_PHY")
    prep_out = {k: inputs.pop(k) for k in ("TAUTOTAL", "CUTOP", "CUBOT")}
    ctx.tables[unit.PREP_KEY] = prep_out
    t0 = time.perf_counter()
    out = unit.run(ctx, inputs=inputs)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    bad = [k for k, a in out.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print(f"phys-rrtm-driver-call: grid {im}x{jm}x{lm}, {dt:.2f} s (compile + run), "
          f"microphysics={getattr(cfg, 'microphysics', '')!r}")
    print("NaN/Inf in:", bad if bad else "none")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
