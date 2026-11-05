"""Dynamics + boundary + driver integration run on the REAL case state (dyn integrator, 2026-11-04).

    source env.sh
    python tools/run-dyn-step.py --case 2025-03-14_12z [--steps 2] [--physics] [--vram 20000]

Builds the real state with init-case-driver (native B-grid analysis + GFS boundaries) and
drv-physics-initialize, then runs Driver.step() --steps times (step 0 is the SOLVER_RUN firststep
block, step 1 is the first not_firststep step with HDIFF, BOCOH/BOCOV, PGFORCE, DDAMP, ADV1, ADV2,
CDZDT, CDWDT, VSOUND).  After every unit it syncs and scans every float array of ctx.st for NaN/Inf
and names the first unit that makes one (diagnosis mode: the per-unit sync is only in this tool,
never in the forecast loop).  Physics groups run only with --physics.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def nonfinite(st, cp):
    return [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case folder name under cases/, e.g. YYYY-MM-DD_HHz")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--vram", type=int, default=20000)
    ap.add_argument("--no-lease", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--stats", default="T,U,V,PD,PINT,Q,W,DWDT,CW,Q2")
    a = ap.parse_args(argv)

    from namgpu.steps.drv_gpu_lease import gpu_lease
    lease = contextlib.nullcontext("") if a.no_lease else gpu_lease(
        lane="nam-gpu-dyn-integrator", vram_mib=a.vram, est_s=900, what="nam dyn step on real state")
    with lease:
        import cupy as cp
        from namgpu import config, grid as gridmod, state as statemod
        from namgpu.driver import Ctx, Driver
        from namgpu.init import init_case_driver
        from namgpu.steps import drv_physics_initialize

        cfg = config.load()
        g = gridmod.build(cfg)
        st = statemod.State(g)
        ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
        ctx.physics_on = a.physics
        drv = Driver(ctx)
        drv.check_ready()
        ctx.case_dir = ROOT / "cases" / a.case
        ctx.start = datetime.strptime(a.case[:13], "%Y-%m-%d_%H")
        ctx.driver = drv
        ctx.dt = cfg.dt
        ctx.forecast_hours = max(3, 3 * int(-(-a.steps * cfg.dt // 10800)))   # boco files the run needs
        t0 = time.perf_counter()
        # The case is already on disk (no fetch under a GPU lease).
        skip = ("init-fetch-case",)
        if not a.physics:
            skip += ("drv-physics-initialize", "init-gwd-orog")   # physics-only inputs
        only = [u for u in init_case_driver.CHAIN if u not in skip]
        with ctx.stream:
            init_case_driver.run(ctx, only=only)
            if a.physics and "drv-physics-initialize" not in init_case_driver.CHAIN:
                drv_physics_initialize.run(ctx)
        ctx.stream.synchronize()
        print(f"boundary files: {getattr(ctx, 'boco_dir', None)}", flush=True)

        def stats(tag):
            out = []
            for n in a.stats.split(","):
                if n in st.fields:
                    x = st.fields[n]
                    out.append(f"{n}[{float(x.min()):.4g},{float(x.max()):.4g}]")
            print(f"  {tag}: " + " ".join(out), flush=True)

        stats("init")
        orig = drv._wrapper
        seen = set()

        class W:
            def __init__(self, uid, mod):
                self.uid, self.mod = uid, mod

            def run(self, c, **kw):
                tt = time.perf_counter()
                r = self.mod.run(c, **kw)
                c.stream.synchronize()
                dt = time.perf_counter() - tt
                bad = [b for b in nonfinite(st, cp) if b not in seen]
                print(f"  step {c.ntimestep} {self.uid} {kw if kw else ''} {dt * 1e3:.1f} ms"
                      + (f"  NONFINITE {bad}" if bad else ""), flush=True)
                if bad:
                    seen.update(bad)
                    if not a.keep_going:
                        raise SystemExit(f"first NaN/Inf after {self.uid} at step {c.ntimestep}: {bad}")
                return r

        cache = {}
        drv._wrapper = lambda uid: cache.setdefault(uid, W(uid, orig(uid)))
        with ctx.stream:
            for _ in range(a.steps):
                ts = time.perf_counter()
                drv.step()
                ctx.stream.synchronize()
                print(f"step done -> ntimestep {ctx.ntimestep}, {time.perf_counter() - ts:.2f} s", flush=True)
                stats(f"after step {ctx.ntimestep}")
        print("RESULT: ran", a.steps, "steps;", "nonfinite:", sorted(seen) or "none")
        return 0 if not seen else 1


if __name__ == "__main__":
    sys.exit(main())
