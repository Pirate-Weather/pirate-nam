import sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import importlib.util as _u
_s = _u.spec_from_file_location("rpa", ROOT / "tools" / "run_physics_a.py"); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
from namgpu.physics.driver import phys_turbl_driver as td
import numpy as np
orig_dep = td._dep
def dep(m):
    mod = orig_dep(m)
    if not m.endswith("phys_noah_driver"):
        return mod
    def run(ctx, **kw):
        st = ctx.st
        print("SLDPTH in fields before:", "SLDPTH" in st.fields, st["SLDPTH"].get() if "SLDPTH" in st.fields else None)
        from namgpu.physics.noah import phys_noah_driver as nd
        args = nd._tbl_args(ctx)
        names = "ALBTBL Z0TBL SHDTBL RSTBL RGLTBL HSTBL SNUPTBL LAITBL NROTBL BB DRYSMC F11 MAXSMC REFSMC SATPSI SATDK SATDW WLTSMC QTZ SLOPE".split()
        for n, a in zip(names, args):
            print(f"{n:8}", np.array2string(a.get()[:20], precision=4, max_line_width=250))
        print("scalars", args[20:])
        print("EPSR range", float(st["EPSR"].min()), float(st["EPSR"].max()))
        for n in ("SMC","SH2O","STC"):
            a = st[n].get(); land = st["XLAND"].get() < 1.5
            print(n, [ (float(a[k][land].min()), float(a[k][land].max())) for k in range(4)])
        print("CHS land range", float(st["CHS"].get()[st["XLAND"].get()<1.5].min()), float(st["CHS"].get()[st["XLAND"].get()<1.5].max()))
        raise SystemExit("dbg done")
    return types.SimpleNamespace(run=run)
td._dep = dep
R.main()
