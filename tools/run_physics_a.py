"""Physics A on the REAL case state: one full physics call (Fortran SOLVER_RUN physics block order).

    source env.sh
    ~/gpuwm-work/pool-v3/run.sh --cards 1 --vram-mib 20000 --est 900 nam-physics-a -- \
        python tools/run_physics_a.py --case 2025-03-14_12z [--step 0] [--with-rad]

Builds the real initial state from the native NAM analysis (init-case-driver, boundary/GFS stages
skipped: physics does not read them), runs PHYSICS_INITIALIZE, then the physics-A part of the step
plan (driver.PHYSICS minus radiation unless --with-rad) at ntimestep --step (0 = the first physics
call of the forecast: NPHS, NPRECIP and the NTSD==0 blocks all fire). Then a NaN/Inf census of every
float array and a short table of the physics outputs. Not a column test: the whole 954 x 835 x 60 state.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import config, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import PHYSICS, Ctx, Driver  # noqa: E402

SKIP_INIT = ("init-fetch-case", "init-gfs-bocos", "init-gfs-decode", "init-gfs-hinterp", "init-gfs-vinterp",
             "init-gfs-wind-rotate", "init-boco-tendencies")
REPORT = ("T", "Q", "CW", "Q2", "U", "V", "PBLH", "USTAR", "TSKIN", "THS", "QSH", "TSFC", "Z0",
          "SFCSHX", "SFCLHX", "TWBS", "QWBS", "STC", "SMC", "SH2O", "CMC", "SNO", "TSHLTR",
          "QSHLTR", "U10", "V10", "TH10", "Q10", "AKHS", "AKMS", "EL_PBL", "DUDT", "DVDT",
          "DUDT_GWD", "DVDT_GWD", "CUPREC", "ACPREC", "PREC", "CPRATE", "CNVTOP", "CNVBOT",
          "F_ICE", "F_RAIN", "F_RIMEF", "QC", "QR", "QS", "QG", "REFL_10CM", "TRAIN", "TCUCN",
          "SR", "AVRAIN", "RTHBLTEN", "RQBLTEN")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2025-03-14_12z")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--with-rad", action="store_true")
    ap.add_argument("--gwd-dem", default="", help="raw 30-arcsec int16 terrain for init-gwd-orog "
                    "(default $NAMGPU_GWD_DEM or /work/nam-static/gmted2010.30sec.int)")
    ap.add_argument("--calls", type=int, default=1, help="physics calls (steps) to run")
    ap.add_argument("--discover", default="", help="comma list of units to stub OUT for a discovery run only "
                    "(phys-noah-driver, phys-gwd-driver, phys-bmj-driver, phys-fer-driver); never a result")
    a = ap.parse_args(argv)

    import cupy as cp
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.dt = cfg.dt
    drv = Driver(ctx)
    ctx.driver = drv

    from namgpu.init import init_case_driver  # noqa: PLC0415
    import os  # noqa: PLC0415
    dem = Path(a.gwd_dem or os.environ.get("NAMGPU_GWD_DEM", "/work/nam-static/gmted2010.30sec.int"))
    if dem.is_file():                     # raw 30-arcsec int16 (NOAA fix/orog gmted2010.30sec.int)
        from namgpu.init.init_gwd_orog import load_gwd_dem  # noqa: PLC0415
        ctx.tables["gwd_dem"] = load_gwd_dem(str(dem))
        print(f"gwd dem: {dem}", flush=True)
    t0 = time.perf_counter()
    with ctx.stream:
        init_case_driver.run(ctx, only=[u for u in init_case_driver.CHAIN if u not in SKIP_INIT])
    ctx.stream.synchronize()
    print(f"init {time.perf_counter() - t0:.1f} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    before = {n: (st.fields[n].copy() if n in st.fields else None) for n in ("T", "Q", "CW", "U", "V", "Q2")}

    disc = [u for u in a.discover.split(",") if u]
    if disc:
        print(f"DISCOVERY RUN: {disc} stubbed out; not a physics result", flush=True)
        import types
        from namgpu.physics.driver import phys_turbl_driver as td
        stub = types.SimpleNamespace(run=lambda ctx, **kw: None)
        orig = td._dep
        td._dep = lambda m: stub if (m.split(".")[-1].replace("_", "-") in disc) else orig(m)
    plan = [e for e in PHYSICS if a.with_rad or e[0] != "phys-rad-driver"]
    for call in range(a.calls):
        ctx.ntimestep = a.step + call
        ctx.first_step = ctx.ntimestep == 0
        ctx.ntimestep_rad = 0 if ctx.ntimestep == 0 else ctx.ntimestep + 1
        for entry in plan:
            uid, key = entry[0], entry[1]
            if drv.units[uid]["mode"] == "skip" or not drv._due(key) or uid in disc:
                continue
            kw = ({k: (v if k == "branch" else drv._resolve(v)) for k, v in entry[2].items()}
                  if len(entry) > 2 else {})
            t1 = time.perf_counter()
            with ctx.stream:
                drv._wrapper(uid).run(ctx, **kw)
            ctx.stream.synchronize()
            bad = init_case_driver.census(ctx)
            print(f"  step {ctx.ntimestep} {uid:<22} {key:<12} {time.perf_counter() - t1:7.3f} s"
                  + (f"  NONFINITE {bad}" if bad else ""), flush=True)
            if bad:
                raise SystemExit(f"NaN/Inf after {uid}: {bad}")

    print("\nfield            min          max         mean", flush=True)
    for n in REPORT:
        if n not in st.fields:
            print(f"{n:<10} (absent)")
            continue
        x = st.fields[n].astype(cp.float64)
        print(f"{n:<10} {float(x.min()):12.5g} {float(x.max()):12.5g} {float(x.mean()):12.5g}")
    print("\nchange over the physics call (max |new-old|, mean new-old):")
    for n, old in before.items():
        if old is None:
            continue
        d = (st.fields[n] - old).astype(cp.float64)
        print(f"{n:<4} {float(cp.abs(d).max()):12.5g} {float(d.mean()):12.5g}")
    print("OK: physics A ran on the real state with no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
