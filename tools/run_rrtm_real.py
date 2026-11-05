"""Run the RRTM radiation subsystem (phys-rad-driver -> rrtm-call chain) on the REAL case state.

    source env.sh
    python tools/run_rrtm_real.py --case 2025-03-14_12z --out runs/rrtm-real

Builds the real 954 x 835 x 60 state from the native NAM analysis (init-case-driver stages up to
drv-physics-initialize; the GFS boundary stages are not needed for one physics call), then makes two
full radiation calls (SW + LW): NTIMESTEP_RAD = 0 (12Z, the first step) and NTIMESTEP_RAD = 864
(+6 h, 18Z, the sun over CONUS) on the same analysis state. Reports NaN/Inf counts and ranges of every
radiation output, the per-stage wall time, and saves the 2-D outputs to <out>/rrtm_<tag>.npz.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from namgpu import config, grid as gridmod, state as statemod  # noqa: E402
from namgpu.driver import Ctx  # noqa: E402

INIT_STAGES = [
    "init-fetch-case", "init-grib2-decode", "init-grid-geo", "init-grid-consts", "drv-loop-limits",
    "drv-consts", "drv-spline", "drv-exptbl", "drv-psitbl", "drv-tablet", "drv-tablep",
    "drv-tracer-index", "init-native-sfc", "init-native-atm", "init-vcoord-recover",
    "init-static-missing", "init-derive-state", "init-gwd-orog", "drv-physics-initialize",
]
OUT3 = ("RLWTT", "RSWTT", "CLDFRA")
OUT2 = ("RLWIN", "RSWIN", "RSWINC", "RSWOUT", "RLWTOA", "RSWTOA", "CZMEAN", "SIGT4",
        "CFRACL", "CFRACM", "CFRACH", "ACFRST", "ACFRCV")


def stats(cp, name, a):
    a = a.astype(cp.float32) if a.dtype != cp.float32 else a
    bad = int(cp.count_nonzero(~cp.isfinite(a)))
    f = a[cp.isfinite(a)]
    lo, hi, mean = (float(f.min()), float(f.max()), float(f.mean())) if f.size else (0, 0, 0)
    return bad, f"{name:8s} nonfinite {bad:8d}  min {lo: .4e}  max {hi: .4e}  mean {mean: .4e}"


SW = True


def one_call(cp, ctx, ntimestep, tag, out):
    from namgpu.physics.driver import phys_rad_driver
    from namgpu.physics.rrtm import rrtm_call
    ctx.ntimestep = ntimestep
    ctx.ntimestep_rad = 0 if ntimestep == 0 else ntimestep + 1
    events = []

    def stamp(name):
        e = cp.cuda.Event()
        e.record(ctx.stream)
        events.append((name, e))

    stamp("rad-driver")
    phys_rad_driver.run(ctx, rrtm_call=lambda c, a: rrtm_call.run(c, a, stamp=stamp, sw=SW))
    stamp("end")
    ctx.stream.synchronize()
    print(f"--- {tag}: NTIMESTEP_RAD {ctx.ntimestep_rad}, JDAT {ctx.tables['JDAT']}, "
          f"flags {ctx.tables.get('rrtm_call_flags')}", flush=True)
    for (n0, e0), (n1, e1) in zip(events, events[1:]):
        print(f"  stage {n0:16s} {cp.cuda.get_elapsed_time(e0, e1):9.2f} ms (GPU)")
    print(f"  total radiation call {cp.cuda.get_elapsed_time(events[0][1], events[-1][1]):9.2f} ms")
    st = ctx.st
    nbad = 0
    for n in OUT3 + OUT2:
        b, line = stats(cp, n, st[n])
        nbad += b
        print("  " + line)
    # interior rows only for heating: column-mean and level stats of the heating rates (K/day)
    for n in ("RLWTT", "RSWTT"):
        a = st[n][:, 1:-1, :] * 86400.0
        print(f"  {n} K/day by level (top..sfc) every 10: "
              + " ".join(f"{float(a[l].mean()):.2f}" for l in range(0, a.shape[0], 10)))
    save = {n: cp.asnumpy(st[n]) for n in OUT2}
    save["RLWTT_SUM"] = cp.asnumpy((st["RLWTT"] * 86400.0).mean(axis=0))
    save["RSWTT_SUM"] = cp.asnumpy((st["RSWTT"] * 86400.0).mean(axis=0))
    save["GLAT"] = cp.asnumpy(st["GLAT"])
    save["GLON"] = cp.asnumpy(st["GLON"])
    np.savez_compressed(out / f"rrtm_{tag}.npz", **save)
    return nbad


def rrtm_physics_init(ctx):
    """The PHYSICS_INITIALIZE pieces RRTM reads (drv-physics-initialize steps 1-5), for a cached state."""
    import cupy as cp
    from namgpu.physics.fer_hires import phys_fer_gpvs
    from namgpu.physics.rrtm import phys_rrtm_init, phys_rrtm_radinit, phys_rrtm_rad_initialize, phys_rrtm_sfc_init
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs
    from namgpu.steps import drv_physics_initialize as dpi
    import os
    cfg = ctx.cfg
    fix = Path(getattr(ctx, "fix_dir", None) or os.environ.get("NAMGPU_FIX_DIR") or getattr(dpi, "FIX_DIR", ROOT / "fix"))
    cwd = os.getcwd()
    os.chdir(fix)   # the Fortran opens the radiation fix files by name in the run directory
    try:
        _rrtm_physics_init(ctx, cp, dpi, cfg)
    finally:
        os.chdir(cwd)


def _rrtm_physics_init(ctx, cp, dpi, cfg):
    from namgpu.physics.fer_hires import phys_fer_gpvs
    from namgpu.physics.rrtm import phys_rrtm_init, phys_rrtm_radinit, phys_rrtm_rad_initialize, phys_rrtm_sfc_init
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs
    with ctx.stream:
        sel = dpi._select(ctx, cp)
        ctx.tables["phys_select"] = sel
        dpi._gpkap(ctx, cp)
        phys_fer_gpvs.run(ctx)
        args = RadInitArgs(num_p3d=int(cfg.np3d), iaer=int(cfg.iaer), iems=int(cfg.iems),
                           ntcw=int(cfg.ntcw), ntoz=int(cfg.ntoz),
                           icice_sw=sel["icice_sw"], icice_lw=sel["icice_lw"],
                           icliq_sw=sel["iclq_sw"], icliq_lw=sel["iclq_lw"])
        ctx.tables["rad_init_args"] = args
        phys_rrtm_rad_initialize.run(ctx, args)
        phys_rrtm_sfc_init.run(ctx, mode=0, ialb=args.ialb, iems=args.iems)
        phys_rrtm_radinit.run(ctx, dep_calls=True)
        phys_rrtm_init.run(ctx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case folder name under cases/, e.g. YYYY-MM-DD_HHz")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rrtm-real")
    ap.add_argument("--dump", type=Path, default=None,
                    help="after init, save every state field and the RRTM init tables here")
    ap.add_argument("--lw-only", action="store_true", help="test knob: skip the SW block (loudly)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="load the real state saved by --dump instead of running init")
    a = ap.parse_args(argv)
    global SW
    SW = not a.lw_only
    if not SW:
        print("WARNING: --lw-only: the SW block is NOT run in this test", flush=True)
    import cupy as cp
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    t0 = time.perf_counter()
    if a.cache is not None:
        names = sorted(p.stem for p in a.cache.glob("*.npy"))
        st = statemod.State(g, names=set(names))
        for n in names:
            arr = np.load(a.cache / f"{n}.npy")
            if n in st.fields and tuple(st.fields[n].shape) == arr.shape:
                st.fields[n][...] = cp.asarray(arr)
            else:
                st.fields[n] = cp.asarray(arr)
    else:
        st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / a.case
    ctx.start = datetime.strptime(a.case[:14], "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.dt = cfg.dt
    if a.cache is None:
        from namgpu.init import init_case_driver
        with ctx.stream:
            init_case_driver.run(ctx, only=INIT_STAGES)
        ctx.stream.synchronize()
        print(f"init {time.perf_counter() - t0:.1f} s", flush=True)
        if a.dump is not None:
            a.dump.mkdir(parents=True, exist_ok=True)
            for n, arr in st.fields.items():
                if getattr(arr, "ndim", 0) >= 1 and arr.size:
                    np.save(a.dump / f"{n}.npy", cp.asnumpy(arr))
            print(f"dumped {len(st.fields)} fields to {a.dump}", flush=True)
    else:
        print(f"loaded {len(st.fields)} cached fields {time.perf_counter() - t0:.1f} s", flush=True)
        from namgpu.steps import drv_physics_initialize as dpi
        rrtm_physics_init(ctx)
    print(f"state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    if a.dump is not None and a.cache is None:
        return 0
    if not SW:   # driver-post reads the SW outputs; with the SW block skipped they are zero
        lm, jm, im = int(g.lm), int(g.jm), int(g.im)
        st.add("HTRSW", (lm, jm, im)); st.add("SFCFSW", (4, jm, im)); st.add("TOPFSW", (3, jm, im))
    nbad = 0
    with ctx.stream:
        nbad += one_call(cp, ctx, 0, "t00_12z", a.out)
        nbad += one_call(cp, ctx, 863, "t06_18z", a.out)
    print("RESULT", "NAN-FREE" if nbad == 0 else f"{nbad} NONFINITE")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
