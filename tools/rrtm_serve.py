"""Physics B dev harness: build the real 2025-03-14 12Z state ONCE, then run full radiation calls on request.

    source env-box.sh; NAMGPU_GWD_DEM=/work/nam-static/gmted2010.30sec.int python tools/rrtm_serve.py &
    echo "0 863" > /work/nam-scratch/rrtm-serve/cmd     # NTIMESTEPs to run; result in .../result.log

Each request reloads namgpu.physics.rrtm.*, phys_rad_driver and the kernel cache, drops every ctx.tables key
created after init (so scratch is reallocated) and runs tools/run_rrtm_real.one_call for each step. A file
named 'quit' ends it. Development tool only; the proof command is tools/run_rrtm_real.py.
"""
from __future__ import annotations

import importlib
import io
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
D = Path("/work/nam-scratch/rrtm-serve")


def main():
    import cupy as cp
    from datetime import datetime
    import run_rrtm_real as R
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    D.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = ROOT / "cases" / "2025-03-14_12z"
    ctx.start = datetime(2025, 3, 14, 12)
    ctx.start_time = ctx.start
    ctx.dt = cfg.dt
    from namgpu.init import init_case_driver
    with ctx.stream:
        init_case_driver.run(ctx, only=R.INIT_STAGES)
    ctx.stream.synchronize()
    keep_tables = set(ctx.tables)
    keep_fields = set(st.fields)
    print("READY", flush=True)
    (D / "ready").write_text(time.strftime("%H:%M:%SZ", time.gmtime()))
    while True:
        if (D / "quit").exists():
            (D / "quit").unlink()
            return 0
        ex = D / "exec.py"
        if ex.exists():
            code = ex.read_text()
            ex.unlink()
            buf = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(buf):
                try:
                    exec(compile(code, "exec.py", "exec"), {"ctx": ctx, "st": st, "cp": cp, "R": R, "np": __import__("numpy")})
                except BaseException:
                    traceback.print_exc()
            (D / "exec.log").write_text(buf.getvalue())
            (D / "exec.done").write_text("1")
            continue
        cmd = D / "cmd"
        if not cmd.exists():
            time.sleep(1)
            continue
        steps = [int(x) for x in cmd.read_text().split()]
        cmd.unlink()
        buf = io.StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                from namgpu import kernels
                kernels.module.cache_clear()
                for name in sorted(m for m in list(sys.modules) if m.startswith("namgpu.physics.rrtm.")):
                    importlib.reload(sys.modules[name])
                importlib.reload(sys.modules["namgpu.physics.rrtm"]) if "namgpu.physics.rrtm" in sys.modules else None
                for name in ("namgpu.physics.driver.phys_rad_driver",):
                    if name in sys.modules:
                        importlib.reload(sys.modules[name])
                importlib.reload(R)
                for k in list(ctx.tables):
                    if k not in keep_tables:
                        del ctx.tables[k]
                for k in list(st.fields):
                    if k not in keep_fields:
                        del st.fields[k]
                cp.get_default_memory_pool().free_all_blocks()
                out = ROOT / "runs" / "rrtm-real"
                out.mkdir(parents=True, exist_ok=True)
                nbad = 0
                with ctx.stream:
                    for s in steps:
                        nbad += R.one_call(cp, ctx, s, f"serve_{s}", out)
                print("RESULT", "NAN-FREE" if nbad == 0 else f"{nbad} NONFINITE")
            except BaseException:
                traceback.print_exc()
        cp.get_default_memory_pool().free_all_blocks()
        (D / "result.log").write_text(buf.getvalue() + f"\nwall {time.perf_counter() - t0:.1f} s\n")
        (D / "done").write_text(" ".join(map(str, steps)))


if __name__ == "__main__":
    sys.exit(main())
