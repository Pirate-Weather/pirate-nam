"""Debug: snapshot Noah inputs, run, dump NaN columns' inputs."""
import sys, types
from pathlib import Path
ROOT = Path("/work/nam-gpu"); sys.path.insert(0, str(ROOT))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
from namgpu.physics.driver import phys_turbl_driver as td
from namgpu.physics.noah import phys_noah_driver as nd
import cupy as cp, numpy as np

orig_dep = td._dep
def dep(m):
    mod = orig_dep(m)
    if not m.endswith("phys_noah_driver"):
        return mod
    def run(ctx, **kw):
        st = ctx.st
        names = nd.REQUIRED_2D + ("TWBS", "QWBS", "SMSTAV", "SMSTOT", "SSROFF", "BGROFF", "SNOWC")
        snap = {n: st[n].get() for n in names if n in st.fields}
        soil = {n: st[n].get() for n in nd.REQUIRED_SOIL}
        low = {n: st[n][-1].get() for n in ("DZ", "T", "Q")}
        phint = st["PHINT"][-2:].get()
        r = mod.run(ctx, **kw)
        ctx.stream.synchronize()
        stc = st["STC"].get(); ts = st["TSFC"].get()
        bad_soil = ~np.isfinite(stc).all(axis=0); bad_ts = ~np.isfinite(ts)
        print("bad soil cols", bad_soil.sum(), "bad tsfc", bad_ts.sum(), "both", (bad_soil & bad_ts).sum())
        xl = snap["XLAND"]
        print("XLAND values among bad soil:", np.unique(xl[bad_soil])[:10], " land cells XLAND==1:", (xl == 1).sum())
        for nm in ("IVGTYP", "ISLTYP"):
            v = snap[nm][bad_soil]; print(nm, "bad unique", np.unique(v)[:30])
            g = snap[nm][(~bad_soil) & (xl < 1.5)]; print(nm, "good-land unique", np.unique(g)[:30])
        for label, mask in (("soil-only", bad_soil & ~bad_ts), ("both", bad_soil & bad_ts)):
            jj, ii = np.nonzero(mask)
            if len(jj) == 0: continue
            j, i = jj[len(jj)//2], ii[len(ii)//2]
            print(f"--- {label} column j={j} i={i}")
            for n, a in snap.items(): print(f"   {n:10} {a[j, i]}")
            for n, a in soil.items(): print(f"   {n:10} {a[:, j, i]}")
            for n, a in low.items(): print(f"   low {n:6} {a[j, i]}")
            print("   PHINT[-2:]", phint[:, j, i])
            print("   out STC", stc[:, j, i], "TSFC", ts[j, i], "SH2O", st["SH2O"][:, j, i].get(), "SMC", st["SMC"][:, j, i].get())
        # also input-side NaN check
        for n, a in {**snap, **soil}.items():
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                print("INPUT nonfinite", n, (~np.isfinite(a)).sum())
        raise SystemExit("dbg done")
    return types.SimpleNamespace(run=run)
td._dep = dep
R.main()
