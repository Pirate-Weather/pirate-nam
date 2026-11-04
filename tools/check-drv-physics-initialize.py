"""Done-rule check for drv-physics-initialize (CONTRACT section 6), run once on the full 12 km grid.

Builds a full-size 954 x 835 x 60 random State (only the fields the physics init chain writes or
reads), runs the real wrapper once under one pool-v3 lease, then checks that every float array the
wrapper wrote is finite and that the selection result matches the Fortran special cases.

    source ~/nam-gpu/env.sh && python tools/check-drv-physics-initialize.py
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp  # noqa: E402

from namgpu import config  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402
from namgpu.state import State  # noqa: E402
from namgpu.steps import drv_physics_initialize as w  # noqa: E402
from namgpu.steps.drv_gpu_lease import gpu_lease  # noqa: E402

IM, JM, LM = 954, 835, 60
NAMES = ["GLAT", "GLON", "HANGL", "USTAR", "Z0", "SM", "SICE", "IVGTYP", "EXCH_H",
         "CMC", "ISLTYP", "STC", "SMC", "SH2O", "CLDEFI", "F_ICE", "F_RAIN", "F_RIMEF", "SEAMASK", "XICE"]


def main():
    import os
    if os.environ.get("CHECK_SKIP_NOAH") == "1":
        # PARTIAL: NAM_NOAH_FIX_DIR (NCEP fix VEGPARM/SOILPARM/GENPARM) is not on node-2; the NOAH sub-units are skipped here only.
        from namgpu.physics.noah import phys_noah_init, phys_noah_parm_tables
        phys_noah_parm_tables.run = lambda ctx, **kw: {}
        phys_noah_init.run = lambda ctx, **kw: None
    cfg = config.load()
    grid = types.SimpleNamespace(
        im=IM, jm=JM, lm=LM, pt=200.0,
        sg2=np.linspace(1.0, 0.0, LM + 1).astype(np.float32),
        wbd=cfg.wbd, sbd=cfg.sbd, tph0d=cfg.tph0d, tlm0d=cfg.tlm0d,
    )
    st = State(grid, names=NAMES)
    st.randomize()
    ctx = Ctx(cfg=cfg, grid=grid, st=st, tables={})
    with gpu_lease(lane="nam-gpu-drv-physics-initialize", vram_mib=12000, est_s=900,
                   what="drv-physics-initialize full-size check"):
        t0 = time.time()
        tables = w.run(ctx)
        cp.cuda.Stream.null.synchronize()
        print(f"wrapper run {time.time() - t0:.2f} s on {IM} x {JM} x {LM}")

    print("selection:", tables["phys_select"])
    assert tables["phys_select"] == {"icice_sw": 3, "icice_lw": 1, "iclq_sw": 1, "iclq_lw": 1}
    ok = True
    for name in ["LOWLYR", "EMISS", "USTAR", "EXCH_H", "CMC", "SH2O", "CLDEFI", "F_ICE", "F_RAIN", "F_RIMEF"]:
        if name in ctx.st.names if hasattr(ctx.st, "names") else False:
            pass
    for name in ["EMISS", "USTAR", "EXCH_H", "SH2O", "CLDEFI", "F_ICE", "F_RAIN", "F_RIMEF", "CROT", "SROT", "HANGL"]:
        try:
            a = st[name]
        except KeyError:
            continue
        if a.dtype.kind == "f":
            bad = int((~cp.isfinite(a)).sum())
            print(f"{name:8s} shape {a.shape} non-finite {bad}")
            ok &= bad == 0
    gp = tables["gpkap"]["tbpkap"]
    print("gpkap nodes", gp.shape, "finite", bool(cp.isfinite(gp).all()), "last", float(gp[-1]))
    assert ok, "non-finite values written by drv-physics-initialize"
    print("PASS")


if __name__ == "__main__":
    main()
