"""Done-rule check for dyn-cdwdt-diffusion after the review fix (CONTRACT.md section 6).

    source ~/nam-gpu/env.sh
    python tools/check-dyn-cdwdt-diffusion.py

Runs dyn-hdiff-coef first (it writes DEF3D, the only deformation array), then the wrapper, on a
full-size random State (954 x 835 x 60). Checks:
  1. the wrapper binds DEF3D and creates no zero-filled "DEF" array;
  2. DEF3D is nonzero, so the face fluxes are live;
  3. the DWDT increment is nonzero and equals the Fortran formula (module_DYNAMICS_ROUTINES.F90
     3558-3587) recomputed on the full interior from the same W, DEF3D, HDACX, HDACY;
  4. outside the interior DWDT is untouched; no NaN/Inf anywhere.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.dynamics import dyn_cdwdt_diffusion, dyn_hdiff_coef

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=2)
    stream = cp.cuda.Stream(non_blocking=True)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    ctx.tables["LPT2"] = int(np.sum(g.sgml2 == 0))
    ctx.tables["EPSQ2"] = cp.full((g.lm,), 0.02, dtype=cp.float32)

    kernels.module("dyn_hdiff_coef").compile()
    kernels.module("dyn_cdwdt_diffusion").compile()
    with stream:
        dyn_hdiff_coef.run(ctx)
    stream.synchronize()

    # Snapshot the inputs of the diffusion term as the step will see them.
    deff = st["DEF3D"].copy()
    w = st["W"].copy()
    hx = st["HDACX"].copy()
    hy = st["HDACY"].copy()
    dwdt0 = st["DWDT"].copy()

    t0 = time.perf_counter()
    dyn_cdwdt_diffusion.run(ctx)
    stream.synchronize()
    dt_wall = time.perf_counter() - t0

    # 1. binding: no zero-filled DEF scratch, and the wrapper read DEF3D (same object).
    no_def_scratch = "DEF" not in st.fields
    bound = st["DEF3D"] is st.fields["DEF3D"] and bool(cp.array_equal(st["DEF3D"], deff))

    # 2. deformation is live.
    def_max = float(cp.abs(deff).max())

    # 3. Fortran formula on the interior, Fortran i = 2..IM-1, j = 2..JM-1 (0-based 1..n-2).
    dtv = np.float32(cfg.dt)
    rdt = np.float32(1.0) / dtv
    W, D = w, deff
    c = (slice(None), slice(1, -1), slice(1, -1))  # (l, j, i) interior
    wx_e = (W[:, 1:-1, 2:] - W[:, 1:-1, 1:-1]) * (D[:, 1:-1, 2:] + D[:, 1:-1, 1:-1])
    wx_w = (W[:, 1:-1, 1:-1] - W[:, 1:-1, :-2]) * (D[:, 1:-1, :-2] + D[:, 1:-1, 1:-1])
    wy_n = (W[:, 2:, 1:-1] - W[:, 1:-1, 1:-1]) * (D[:, 2:, 1:-1] + D[:, 1:-1, 1:-1])
    wy_s = (W[:, 1:-1, 1:-1] - W[:, :-2, 1:-1]) * (D[:, :-2, 1:-1] + D[:, 1:-1, 1:-1])
    hxi = hx[1:-1, 1:-1][None]
    hyi = hy[1:-1, 1:-1][None]
    inc = -((wx_e - wx_w) * hxi + (wy_n - wy_s) * hyi) * rdt
    got = st["DWDT"][c] - dwdt0[c]
    inc_max = float(cp.abs(inc).max())
    err = float(cp.abs(got - inc).max())
    nonzero = int(cp.count_nonzero(inc))

    # 4. outside the interior DWDT is untouched.
    dw = st["DWDT"]
    edge_ok = bool(cp.array_equal(dw[:, 0, :], dwdt0[:, 0, :])
                   and cp.array_equal(dw[:, -1, :], dwdt0[:, -1, :])
                   and cp.array_equal(dw[:, :, 0], dwdt0[:, :, 0])
                   and cp.array_equal(dw[:, :, -1], dwdt0[:, :, -1]))

    bad = [n for n in ("DWDT", "DEF3D") if not bool(cp.isfinite(st[n]).all())]

    print(f"wrapper {dt_wall:.2f} s; DEF3D max|.|={def_max:.4e}; no DEF scratch={no_def_scratch}; "
          f"DEF3D bound={bound}")
    print(f"diffusion increment: max|inc|={inc_max:.4e}, nonzero points={nonzero}, "
          f"max|got-ref|={err:.3e}, rel={err / max(inc_max, 1e-30):.2e}")
    print(f"edges untouched={edge_ok}; NaN/Inf in {bad if bad else 'none'}")

    ok = (no_def_scratch and bound and def_max > 0.0 and nonzero > 0
          and err <= 1e-4 * inc_max and edge_ok and not bad)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
