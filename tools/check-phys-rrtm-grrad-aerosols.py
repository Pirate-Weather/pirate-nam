"""Review check for unit phys-rrtm-grrad-aerosols (the fixes for the 2026-11-04 review).

    source env.sh && python tools/check-phys-rrtm-grrad-aerosols.py

1. grrad-setup on the full-size random State (954 x 835 x 60), ivflip 0 (NAM iflip = 0): the new
   RAD_QLYR and RAD_TVLY are compared against a numpy reference of the Fortran top-to-surface recurrence
   (grrad_nmmb.f 1437-1460) on a corner of columns, and the extra-top-layer copies (lines 1297-1330)
   are checked on the full arrays.
2. rad_initialize_nmmb flags with NAM values (iaer 11, iflip 0) and the aerosol module state the wrapper reads.
3. the aerosol wrapper on the full-size state: the setaer stages run, then the aer_property stage (iaerflg 11)
   must stop with its named RuntimeError. That stage is not wired yet; the check reports it, not a pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_grrad_setup as setup
    from namgpu.physics.rrtm import phys_rrtm_rad_initialize as radinit
    from namgpu.physics.rrtm import phys_rrtm_grrad_aerosols as aers

    kernels.module("phys_rrtm_grrad_setup").compile()
    cfg = config.load()
    g = gridmod.build(cfg)
    names = {"PINT", "PD", "SGML2", "T", "Q", "TRACERS", "CLDFRA", "F_ICE", "F_RAIN", "F_RIMEF",
             "SLMSK", "XLON", "XLAT"}
    st = statemod.State(g, names=names).randomize(seed=1)
    st["PD"][...] = cp.clip(st["PD"], 50000.0, 100000.0)
    st["PINT"][...] = 1000.0 + 99000.0 * st["PINT"]
    jm, im = int(g.jm), int(g.im)
    st.add("SLMSK", (jm, im))  # sea/land mask, supplied by the driver init (not a spec field)
    st["SLMSK"][...] = cp.random.randint(0, 3, (jm, im)).astype(cp.float32)
    st.add("XLON", (jm, im))
    st["XLON"][...] = cp.random.uniform(-3.1, 3.1, (jm, im)).astype(cp.float32)
    st.add("XLAT", (jm, im))
    st["XLAT"][...] = cp.random.uniform(-1.5, 1.5, (jm, im)).astype(cp.float32)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    with ctx.stream:
        out = setup.run(ctx, dtsw=300.0, dtlw=300.0, ivflip=0, np3d=3, ntoz=0)
        radinit.run(ctx)  # NAM defaults: iaer 11, iflip 0
    ctx.stream.synchronize()

    # 1. qlyr / tvly against the Fortran recurrence, corner of columns on the host
    lm = int(g.lm)
    fvirt = 461.5 / 287.05 - 1.0
    tq = (st["T"][:, :8, :8].get(), st["Q"][:, :8, :8].get())
    qlyr = out["RAD_QLYR"][:, :8, :8].get()
    tvly = out["RAD_TVLY"][:, :8, :8].get()
    tem1d = np.float32(1.0e-7)
    qref = np.zeros((lm + 1, 8, 8), np.float32)
    tref = np.zeros((lm + 1, 8, 8), np.float32)
    for k in range(lm):
        ql = np.maximum(tem1d, tq[1][k]).astype(np.float32)
        tem1d = np.minimum(np.float32(1.0e-7), ql)
        qref[k + 1] = ql
        tref[k + 1] = (tq[0][k] * (1.0 + np.float32(fvirt) * ql)).astype(np.float32)
    qref[0], tref[0] = qref[1], tref[1]  # extra top layer copies lya -> lyb
    err_q = float(np.max(np.abs(qlyr - qref) / np.maximum(np.abs(qref), 1e-12)))
    err_t = float(np.max(np.abs(tvly - tref) / np.maximum(np.abs(tref), 1e-12)))
    top_q = bool(cp.array_equal(out["RAD_QLYR"][0], out["RAD_QLYR"][1]))
    top_t = bool(cp.array_equal(out["RAD_TVLY"][0], out["RAD_TVLY"][1]))
    print(f"qlyr max rel err vs Fortran recurrence (8x8 corner): {err_q:.3e}")
    print(f"tvly max rel err vs Fortran recurrence (8x8 corner): {err_t:.3e}")
    print(f"extra top layer copies qlyr/tvly: {top_q}/{top_t}")
    nonfin = [n for n in ("RAD_QLYR", "RAD_TVLY") if not bool(cp.isfinite(out[n]).all())]
    print(f"NaN/Inf in setup outputs: {nonfin or 'none'}")

    # 2. module state the aerosol wrapper reads
    a = aers.aer_state(ctx)
    print(f"aer_state: iaerflg={a['iaerflg']} laswflg={a['laswflg']} lalwflg={a['lalwflg']} "
          f"lavoflg={a['lavoflg']} ivflip={a['ivflip']} nlwbnd={a['nlwbnd']}")

    # 3. aerosol wrapper on the full-size state
    faersw, faerlw = aers.new_outputs(lm + 1, int(g.jm), int(g.im))
    try:
        with ctx.stream:
            aers.grrad_aerosols(ctx, faersw, faerlw, lsswr=True, lslwr=True)
        ctx.stream.synchronize()
        print("aerosol wrapper: full chain ran")
    except RuntimeError as e:
        print(f"aerosol wrapper stopped at a named gap: {e}")
    return 0 if not nonfin and err_q < 1e-6 and err_t < 1e-6 and top_q and top_t else 1


if __name__ == "__main__":
    sys.exit(main())
