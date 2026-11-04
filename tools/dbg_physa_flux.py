import sys, types
from pathlib import Path
ROOT = Path("/work/nam-gpu"); sys.path.insert(0, str(ROOT))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
from namgpu.physics.driver import phys_turbl_driver as td
import numpy as np
CT = {}
def pct(a, m=None):
    a = a[m] if m is not None else a.ravel()
    return " ".join(f"{np.percentile(a, q):.4g}" for q in (0, 5, 50, 95, 100))
def show(tag):
    st = CT["ctx"].st; CT["ctx"].stream.synchronize()
    sm = st["SM"].get(); land = sm < 0.5; sea = ~land
    b = np.zeros_like(land); b[1:-1, 1:-1] = True
    print(f"[{tag}]")
    for n in ("QWBS", "TWBS", "CHKLOWQ", "QSH", "AKHS", "TSFC", "ELFLX"):
        if n in st.fields:
            a = st[n].get(); print(f"   {n:8} land {pct(a, land & b)} | sea {pct(a, sea & b)}")
    q = st["Q"][-1].get(); print(f"   Qlow     land {pct(q, land & b)} | sea {pct(q, sea & b)}")
orig_k = td._k
def k(name):
    f = orig_k(name)
    def g(*a, **kw):
        if name == "turbl_flux_accum": show("before flux_accum")
        f(*a, **kw)
        if name in ("turbl_sfc_pre",): show("after " + name)
    return g
td._k = k
orig_dep = td._dep
def dep(m):
    mod = orig_dep(m)
    def run(ctx, **kw):
        r = mod.run(ctx, **kw)
        if m.endswith(("jsfc_jsfc", "noah_driver")): show("after " + m.split(".")[-1])
        return r
    return types.SimpleNamespace(run=run)
td._dep = dep
orig = td.run
def run(ctx, **kw):
    CT["ctx"] = ctx; return orig(ctx, **kw)
td.run = run
R.main()
