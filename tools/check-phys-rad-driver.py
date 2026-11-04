"""Done-rule check for phys-rad-driver (CONTRACT section 6): compile the kernel, run the wrapper once
at a radiation step on a FULL-SIZE state (954 x 835 x 60), check the outputs it writes for NaN/Inf.

The RRTM driver call and radupdate_nmmb are other units, not written yet. They are passed as
recording stubs: the check covers the driver's own terms (radupdate call, zeroing kernel, default
cloud-input branch, RRTM argument list, radiation-step test). Not a physics test.
Everything, including the State allocation, runs inside one pool lease (the card is shared).

    source ~/nam-gpu/env.sh
    python tools/check-phys-rad-driver.py
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEEDED = {
    "CLDFRA", "CFRACH", "CFRACL", "CFRACM", "CZMEAN", "SIGT4", "GLAT", "GLON",
    "DSG2", "SGML2", "PDSG1", "PSGML1", "PD", "T", "Q", "CW", "O3", "ALBEDO",
    "F_ICE", "F_RAIN", "QC", "QI", "QS", "QR", "QG", "NI", "SM",
    "RLWTT", "RSWTT", "RLWIN", "RSWIN", "RSWINC", "RSWOUT", "RLWTOA", "RSWTOA",
    "ACFRST", "NCFRST", "ACFRCV", "NCFRCV", "CUPPT", "SNOWC", "SI", "HTOP", "HBOT",
    "TSKIN", "Z0", "SICE", "F_RIMEF", "MXSNAL", "SGM", "STDH", "OMGALF",
}


def run_check(cp, kernels, config, gridmod, statemod, Ctx, pr) -> bool:
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=NEEDED).randomize(seed=1)
    # STDH is not in the NMMB state tables; the init unit will load the terrain standard deviation.
    # Test input only: a constant 50 m stands in for it.
    st.add("STDH", (g.jm, g.im))[...] = 50.0
    print("state fields:", len(st.fields), "bytes:", round(st.nbytes() / 2**30, 2), "GiB", flush=True)

    seen = {}

    def radupdate_stub(ctx, jdat, dtsw, dtx, lssw):
        seen["radupdate"] = {"dtsw": dtsw, "dtx": dtx, "lssw": bool(lssw)}
        return {"SOLCON": 1361.0}   # stand-in value for the call's wiring; not the physics

    def rrtm_stub(ctx, a):
        seen["rrtm"] = {k: a[k] for k in ("NTIMESTEP", "CLD_FRACTION", "I_S", "I_E", "J_S", "J_E",
                                          "NUM_WATER", "F_QI", "F_NI", "F_QS", "SOLCON")}
        seen["rrtm_qtdum_is_cw"] = a["QTdum"] is ctx.st["CW"]
        seen["rrtm_nci_zero"] = bool(cp.all(a["NCI"] == 0))
        seen["rrtm_qci_zero"] = bool(cp.all(a["QCI"] == 0))
        seen["rrtm_nargs"] = len(a)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt,
              ntimestep=48, start=datetime.datetime(2025, 3, 14, 12))
    ctx.tables["JDAT"] = [2025, 3, 14, 0, 12, 0, 0, 0]

    with ctx.stream:
        kernels.module("phys_rad_driver").compile()
        print("kernel compiled", flush=True)
        t0 = time.perf_counter()
        pr.run(ctx, radupdate=radupdate_stub, rrtm_call=rrtm_stub)
    ctx.stream.synchronize()
    wall = time.perf_counter() - t0

    print("wall", round(wall, 3), "s", flush=True)
    print("radupdate call:", seen.get("radupdate"))
    print("rrtm call:", seen.get("rrtm"), "qtdum_is_cw", seen.get("rrtm_qtdum_is_cw"),
          "qci_zero", seen.get("rrtm_qci_zero"), "nci_zero", seen.get("rrtm_nci_zero"),
          "nargs", seen.get("rrtm_nargs"))

    ok = True
    for name in ("CLDFRA", "CFRACH", "CFRACL", "CFRACM", "CZMEAN", "SIGT4"):
        mx = float(cp.abs(st[name]).max())
        print(f"{name}: max|x| after zeroing = {mx}")
        ok &= mx == 0.0
    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print("NaN/Inf fields:", bad)
    ok &= (not bad) and ("rrtm" in seen) and ("radupdate" in seen)
    return ok


def main() -> int:
    import cupy as cp

    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.driver import phys_rad_driver as pr
    from namgpu.steps.drv_gpu_lease import gpu_lease

    with gpu_lease(lane="nam-gpu-phys-rad-driver", vram_mib=6000, est_s=600,
                   what="phys-rad-driver check"):
        ok = run_check(cp, kernels, config, gridmod, statemod, Ctx, pr)
    print("RESULT", "ok" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
