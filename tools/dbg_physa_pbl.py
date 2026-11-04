import sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
from namgpu.physics.driver import phys_turbl_driver as td
import numpy as np
def pct(a, m=None):
    a = a[m] if m is not None else a.ravel()
    return " ".join(f"{np.percentile(a, q):.4g}" for q in (0, 5, 50, 95, 100))
orig = td.run
def run(ctx, **kw):
    st = ctx.st
    sm = st["SM"].get(); land = sm < 0.5; sea = ~land
    q2 = st["Q2"].get()
    print("init Q2 lowest 5 layers p0/5/50/95/100:", [pct(q2[-k]) for k in (1, 2, 3, 5, 10)])
    print("init PBLH:", pct(st["PBLH"].get()))
    r = orig(ctx, **kw); ctx.stream.synchronize()
    pb = st["PBLH"].get(); qs = st["QSH"].get(); tw = st["TWBS"].get(); qw = st["QWBS"].get()
    print("PBLH land:", pct(pb, land), " sea:", pct(pb, sea))
    print("Q2 after lowest:", pct(st["Q2"].get()[-1]), " Q2 L-5:", pct(st["Q2"].get()[-5]))
    print("EXCH_H lowest:", pct(st["EXCH_H"].get()[-2]))
    print("QSH land:", pct(qs, land), " sea:", pct(qs, sea), " n<0 land", int((qs[land] < 0).sum()), "sea", int((qs[sea] < 0).sum()))
    print("TWBS land:", pct(tw, land), " sea:", pct(tw, sea))
    print("QWBS land:", pct(qw, land), " sea:", pct(qw, sea))
    print("TSKIN land:", pct(st["TSKIN"].get(), land), "sea:", pct(st["TSKIN"].get(), sea))
    print("USTAR:", pct(st["USTAR"].get()), "AKHS:", pct(st["AKHS"].get()))
    j, i = np.unravel_index(np.argmin(np.where(land, qs, 9)), qs.shape)
    print("most negative land QSH at", j, i, qs[j, i], "TSKIN", st["TSKIN"][j, i].get(), "SNO", st["SNO"][j, i].get(), "SICE", st["SICE"][j, i].get(), "ISLTYP", st["ISLTYP"][j,i].get(), "IVGTYP", st["IVGTYP"][j,i].get())
    lp = st["LPBL"].get(); print("LPBL:", pct(lp.astype(float)))
    return r
td.run = run
R.main()
