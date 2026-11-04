"""Dynamics + boundary + driver on the REAL case state: init chain, physics init, then N whole
NMMB steps (first step + not_first steps, physics on, wet) through Driver's plan.

    source /work/nam-gpu/env-box.sh
    python tools/run_dyn_step.py --case 2025-03-14_12z --source native --steps 3 \
        --out scratch/dyn-int/step_native [--skip-init init-gwd-orog] [--census-each] [--keep-going]

--census-each: after every plan unit, count NaN/Inf in every float32 state array (one device read
per unit; diagnostic only, never used for timing) and name the first unit that makes one.
--keep-going: a unit that raises is recorded and the step goes on, so one run collects every red.
Writes <out>/dyn-step-receipt.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-init", default="", help="comma list of init chain stages to leave out")
    ap.add_argument("--census-each", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--bc-hours", type=int, default=6, help="boundary span built by init-gfs-bocos")
    ap.add_argument("--leave-out", default="", help="comma list of plan units not to run (diagnostic only: isolates a red owned by another group; the receipt lists them)")
    ap.add_argument("--dfi", action="store_true", help="run the TDFI filter before the steps")
    ap.add_argument("--watch-min", default="", help="FIELD: after every unit of the first --watch-steps steps, print the "
                    "field's min, its (k,j,i) and lat/lon whenever the min moves (diagnostic; one small device read per unit)")
    ap.add_argument("--watch-steps", type=int, default=12)
    a = ap.parse_args(argv)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver
    from namgpu.steps import drv_timing_receipt as timing

    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    ctx.out_dir = out
    ctx.forecast_hours = a.bc_hours
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    skip = [s for s in a.skip_init.split(",") if s]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    rec = {"case": a.case, "source": a.source, "skipped_init": skip, "steps": [], "reds": []}
    t0 = time.perf_counter()
    with ctx.stream:
        rec["init"] = init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    rec["init_s"] = round(time.perf_counter() - t0, 2)
    print(f"init done {rec['init_s']} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)

    def census():
        return init_case_driver.census(ctx)

    SANE = ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "PINT", "DWDT", "OMGALF")

    def stats():
        """min/max/mean of the main prognostics (one small device read per field; diagnostic run)."""
        out = {}
        for nm in SANE:
            a_ = st.fields.get(nm)
            if a_ is None:
                continue
            v = a_[-1] if nm == "PINT" else a_      # PINT: the surface interface
            out[nm] = [round(float(v.min()), 4), round(float(v.max()), 4), round(float(v.mean()), 4)]
        return out

    rec["stats0"] = stats()
    print("stats init", json.dumps(rec["stats0"]), flush=True)

    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()                      # the NMMB run directory, as drv-time-loop runs the forecast

    state = {"first_bad": None, "wmin": None}
    rec["watch"] = []

    def watch(n, tag):
        import numpy as np
        f = st.fields[a.watch_min]
        idx = int(cp.argmin(f))
        val = float(f.ravel()[idx])
        kji = np.unravel_index(idx, f.shape)
        prev = state["wmin"]
        if prev is None or abs(val - prev) > 0.05 or not np.isfinite(val):
            j, i = int(kji[-2]), int(kji[-1])
            ll = {}
            for nm in ("GLAT", "GLON"):
                if nm in st.fields:
                    ll[nm] = round(float(np.degrees(float(st.fields[nm][j, i]))), 2)
            col = {}
            for nm in ("T", "Q", "PD", "W", "U", "V", "PINT"):
                if nm in st.fields:
                    arr = st.fields[nm]
                    col[nm] = (cp.asnumpy(arr[:, j, i]).round(3).tolist() if arr.ndim == 3 else round(float(arr[j, i]), 3))
            e = {"step": n, "unit": tag, "min": round(val, 3), "kji": [int(x) for x in kji], **ll}
            rec["watch"].append({**e, "col": col})
            print(f"WATCH {a.watch_min} step {n} after {tag}: min {val:.3f} at kji {e['kji']} {ll}", flush=True)
        state["wmin"] = val
    leave_out = {u for u in a.leave_out.split(",") if u}
    rec["left_out"] = sorted(leave_out)

    def step_collect():
        """Driver.step with every red collected (and, with --census-each, the first NaN located).
        Installed as drv.step so the TDFI legs use it too."""
        n = ctx.ntimestep
        ctx.ntimestep_rad = 0 if n == 0 else n + 1
        if not ctx.tables.get("filter_consts_bound"):
            from namgpu.steps import drv_filter_consts
            drv_filter_consts.bind(ctx)
        ran = []
        for name, entries in drv.groups:
            if name == "phys" and not ctx.physics_on:
                continue
            for entry in entries:
                uid, key = entry[0], entry[1]
                if drv.units[uid]["mode"] == "skip" or not drv._due(key) or uid in leave_out:
                    continue
                kw = ({k: (v if k == "branch" else drv._resolve(v)) for k, v in entry[2].items()}
                      if len(entry) > 2 else {})
                tag = f"{uid}/{key}" + (f" {kw}" if kw else "")
                timing.stamp(ctx, f"{uid}/{key}")
                try:
                    with ctx.stream:
                        drv._wrapper(uid).run(ctx, **kw)
                except Exception as e:  # noqa: BLE001
                    red = {"step": n, "filter": bool(ctx.filter_active), "unit": tag,
                           "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
                    if not any(r["unit"] == tag and r["error"] == red["error"] for r in rec["reds"]):
                        rec["reds"].append(red)
                        print(f"RED step {n}{' (filter)' if ctx.filter_active else ''} {tag}: {red['error']}", flush=True)
                    if not a.keep_going:
                        (out / "dyn-step-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
                        raise
                    continue
                ran.append(tag)
                if a.watch_min and n < a.watch_steps:
                    watch(n, tag)
                if a.census_each and state["first_bad"] is None:
                    bad = census()
                    if bad:
                        state["first_bad"] = {"step": n, "filter": bool(ctx.filter_active), "unit": tag, "arrays": bad}
                        print(f"FIRST NaN/Inf: step {n} after {tag}: {bad}", flush=True)
        ctx.first_step = False
        ctx.ntimestep += 1
        return ran

    drv.step = step_collect

    if a.dfi:
        from namgpu.steps import drv_dfi_tdfi_driver
        ctx.filter_active = True
        t1 = time.perf_counter()
        with ctx.stream:
            drv_dfi_tdfi_driver.run(ctx)
        ctx.stream.synchronize()
        ctx.filter_active = False
        ctx.ntimestep, ctx.first_step = 0, True
        bad = census()
        rec["dfi"] = {"s": round(time.perf_counter() - t1, 2), "nonfinite": bad,
                      "accum_max_after": {nm: float(st.fields[nm].max()) for nm in ("ACPREC", "CUPREC", "ACPREC_TOT", "CUPREC_TOT")
                                          if nm in st.fields}}
        print(f"TDFI accumulators after the filter (max, m): {rec['dfi']['accum_max_after']}", flush=True)
        print(f"TDFI done {rec['dfi']['s']} s nonfinite {bad}", flush=True)

    first_bad = None
    for _ in range(a.steps):
        n = ctx.ntimestep
        ts = time.perf_counter()
        ran = step_collect()
        ctx.stream.synchronize()
        bad = census()
        dt = time.perf_counter() - ts
        sx = stats()
        rec["steps"].append({"step": n, "units": len(ran), "s": round(dt, 2), "nonfinite": bad,
                             "ran": ran, "stats": sx})
        print(f"step {n}: {len(ran)} units, {dt:.2f} s, nonfinite {bad or 'none'}", flush=True)
        print(f"stats step {n}", json.dumps(sx), flush=True)
    rec["first_bad"] = state["first_bad"]
    rec["peak_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    (out / "dyn-step-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {out / 'dyn-step-receipt.json'}; reds {len(rec['reds'])}", flush=True)
    return 1 if (rec["reds"] or any(s["nonfinite"] for s in rec["steps"])) else 0


if __name__ == "__main__":
    sys.exit(main())
