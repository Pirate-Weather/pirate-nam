"""Done-rule check for unit io-max-fields (CONTRACT.md section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check-io-max-fields.py

1. compiles kernels/io_max_fields.cu (and io-uphlcy through its wrapper);
2. runs the wrapper on a full-size random State (954 x 835 x 60) with NTSD > 0 so the shelter and
   10 m block runs, then checks every float array for NaN/Inf;
3. runs it again at NTSD = avg_max_length/DT (the hourly reset step) and checks that the reset
   restarted NCOUNT, so T10AVG equals T10 (one call in the window).

The GPVS_hr table (TBPVS0) is the real one from phys-fer-gpvs (built by phys_fer_gpvs.run).
Take the GPU pool lease first (README.md). No column test, no reduced size.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.io import io_max_fields
    from namgpu.physics.fer_hires import phys_fer_gpvs

    cfg = config.load()
    g = gridmod.build(cfg)
    from namgpu.io.io_max_fields import _OUT_2D, RESET_VALUES
    # Only the fields this unit and io-uphlcy read or write, at full grid size. The full State
    # is ~13 GB and does not fit the shared card beside the other lanes.
    names = {"T", "W_TOT", "Z", "REFL_10CM", "PINT", "PD", "PREC", "CPRATE", "T2", "U10", "V10",
             "PSHLTR", "TSHLTR", "QSHLTR", "SGML2", "SR", "FIS", "TH10", "AKHS", "AKMS", "SNO",
             "U", "V", "UPHLMAX25", "UPHLMIN25", "UPHLMAX03", "UPHLMIN03"}
    names |= set(_OUT_2D) | set(RESET_VALUES)
    st = statemod.State(g, names=names).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    nrst = io_max_fields.nsteps_per_reset(cfg)
    nchk = io_max_fields.nsteps_per_check(cfg.dt)
    print(f"NSTEPS_PER_RESET={nrst} NSTEPS_PER_CHECK={nchk}", flush=True)

    t0 = time.perf_counter()
    from namgpu import kernels
    kernels.module("io_max_fields").compile()
    print(f"compiled io_max_fields.cu: {time.perf_counter() - t0:.2f} s", flush=True)

    with ctx.stream:
        phys_fer_gpvs.run(ctx)
    ctx.stream.synchronize()

    t1 = time.perf_counter()
    ctx.ntimestep = 1
    with ctx.stream:
        io_max_fields.run(ctx)
    ctx.stream.synchronize()
    print(f"wrapper ran once on full-size state (io-uphlcy + io-max-fields): "
          f"{time.perf_counter() - t1:.2f} s", flush=True)

    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    if bad:
        print(f"NaN/Inf in {bad[:12]}")
        return 1
    for n in ("REFDMAX", "REFDMAX263K", "T10", "T02MAX", "RH02MAX", "SPD10MAX", "PRATEMAX",
              "UPHLMAX25", "UPHLMIN03", "UPVVELMAX", "DNVVELMAX"):
        a = st[n]
        print(f"{n}: min {float(a.min()):.4g} max {float(a.max()):.4g}")

    # Hourly reset step: NCOUNT must restart, so the window holds one call.
    ctx.ntimestep = nrst
    with ctx.stream:
        io_max_fields.run(ctx)
    ctx.stream.synchronize()
    ncount = ctx.tables["MAX_FIELDS_NCOUNT"]
    diff = float(cp.abs(st["T10AVG"] - st["T10"]).max())
    print(f"reset step {nrst}: NCOUNT={ncount}, max |T10AVG - T10| = {diff:.3g}", flush=True)
    if ncount != 1 or diff != 0.0:
        print("FAIL: hourly reset did not restart the running average")
        return 1
    print("ok: no NaN/Inf in any array; hourly reset restarts NCOUNT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
