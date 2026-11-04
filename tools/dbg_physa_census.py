"""Debug: run init then phys-turbl-driver with a NaN census after every kernel and dependency unit."""
import sys, time, types
from pathlib import Path
sys.argv = [sys.argv[0]] + sys.argv[1:]
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
from namgpu.init import init_case_driver as icd
from namgpu.physics.driver import phys_turbl_driver as td
import cupy as cp

CTX = {}
def census(tag):
    ctx = CTX["ctx"]
    ctx.stream.synchronize()
    bad = icd.census(ctx)
    print(f"    after {tag:<24} {('NONFINITE ' + str(bad)) if bad else 'ok'}", flush=True)
    if bad and "first" not in CTX:
        CTX["first"] = tag

orig_k = td._k
def k(name):
    f = orig_k(name)
    def g(*a, **kw):
        f(*a, **kw); census(name)
    return g
td._k = k
orig_dep = td._dep
def dep(m):
    mod = orig_dep(m)
    def run(ctx, **kw):
        r = mod.run(ctx, **kw); census(m.split(".")[-1]); return r
    return types.SimpleNamespace(run=run)
td._dep = dep
orig_run = td.run
def trun(ctx, **kw):
    CTX["ctx"] = ctx
    census("before turbl")
    return orig_run(ctx, **kw)
td.run = trun
R.main()
