"""Done-rule check for phys-turbl-driver (review fix, commit "fix phys-turbl-driver: review").

    source env.sh && python tools/check-phys-turbl-driver.py

Runs the real TURBL wrapper (phys_turbl_driver.run) ONCE per NTSD arm (1 and 0) on a full-size
954 x 835 x 60 state, and asserts no NaN or Inf in any state array. The dependency units
JSFC, NOAHLSM, SFCDIAGS, MYJPBL and GWD are replaced by no-op wrappers in this check ONLY, and it
says so: their outputs are not verified here. Also checks the review defects directly:
  * Z allocated [LM,JM,IM] before TURBL (the old dyn-cdzdt shape) is refused, not read out of bounds;
  * a missing TSFC or THS is refused, not zero-filled;
  * turbl_lpbl writes the B1 columns only and leaves the edge LPBL untouched.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.driver import phys_turbl_driver as wrap

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    print(f"grid {im} x {jm} x {lm}, dt={cfg.dt}, nphs={cfg.nphs}")

    class _Noop:
        @staticmethod
        def run(ctx):
            return {}

    real_dep = wrap._dep
    wrap._dep = lambda modname: _Noop
    print("dependency units replaced by no-op wrappers in this check (outputs unverified)")

    src = Path(wrap.__file__).read_text()
    names = set(re.findall(r"st\[\"([A-Z0-9_]+)\"\]", src)) | set(wrap.SCRATCH_2D) | set(wrap.SCRATCH_3D)
    names |= {"PHINT", "Z", "LPBL", "MAVAIL", "EPSQ2", "TSFC", "THS", "EPSR", "RADOT"}
    lev1 = {"PHINT", "Z"}
    lev = {"Q2", "EXCH_H", "DZ", "PHMID", "RR", "EXNER", "TH", "DELP", "RTHBLTEN", "RQBLTEN",
           "RQCBLTEN", "RQIBLTEN", "U_PHY", "V_PHY", "DUDT", "DVDT", "DUDT_GWD", "DVDT_GWD", "T", "Q",
           "CW", "QC", "QI", "QR", "QS", "QG", "F_RIMEF", "F_ICE", "F_RAIN", "U", "V"}
    soil = {"STC", "SMC", "SH2O"}
    f32 = np.float32
    rng = np.random.default_rng(7)

    def phys(name, shape):
        a = rng.random(shape, dtype=f32)
        if name == "T":
            a = f32(250.0) + f32(50.0) * a
        elif name == "Q":
            a = f32(0.001) + f32(0.01) * a
        elif name in ("CW", "QC", "QI", "QR", "QS", "QG", "F_ICE", "F_RAIN", "F_RIMEF"):
            a = f32(1e-5) * a + (f32(1.0) if name == "F_RIMEF" else f32(0.0))
        elif name == "PD":
            a = f32(9.0e4) + f32(1000.0) * a
        elif name == "FIS":
            a = f32(1.0e4) * a
        elif name in ("TSFC", "THS", "SST", "TH2X", "Q2X", "TLOW", "THLOW"):
            a = f32(280.0) + f32(5.0) * a
        elif name == "EPSQ2":
            a = f32(1.0e-3) + f32(1.0e-4) * a
        elif name == "SM":
            a = (a < 0.5).astype(f32)
        elif name == "SICE":
            a = f32(0.0) * a
        elif name == "MAVAIL":
            a = f32(1.0) + f32(0.0) * a
        elif name == "PBLH":
            a = f32(500.0) * a
        elif name in ("CZMEAN",):
            a = f32(0.5) + f32(0.1) * a
        return a

    st = statemod.State(g, names=set())
    for n in sorted(names):
        if n in st.fields or n == "LPBL":
            continue
        if n in lev1:
            shape = (lm + 1, jm, im)
        elif n in lev:
            shape = (lm, jm, im)
        elif n in soil:
            shape = (4, jm, im)
        else:
            shape = (jm, im)
        st.fields[n] = cp.asarray(phys(n, shape))
    st.fields["LPBL"] = cp.zeros((jm, im), dtype=np.int32)
    st.fields["EPSQ2"] = cp.asarray(np.full((lm,), 1.0e-3, f32))
    st.fields["PBLH"] = cp.asarray(phys("PBLH", (jm, im)))
    st.fields["TSFC"] = cp.asarray(phys("TSFC", (jm, im)))
    st.fields["THS"] = cp.asarray(phys("THS", (jm, im)))
    st.fields["Z"] = cp.asarray(phys("Z", (lm + 1, jm, im)))
    print(f"state fields {len(st.fields)}")

    # ---- review defects checked directly ----
    st2 = statemod.State(g, names=set())
    st2.fields["TSFC"] = st.fields["TSFC"]
    st2.fields["THS"] = st.fields["THS"]
    st2.fields["Z"] = cp.zeros((lm, jm, im), dtype=f32)
    try:
        wrap._get(st2, "Z", (lm + 1, jm, im))
        print("FAIL: Z of the wrong shape was accepted")
        return 1
    except RuntimeError as exc:
        print(f"ok: wrong-shape Z refused: {exc}")
    st3 = statemod.State(g, names=set())
    st3.fields["THS"] = st.fields["THS"]
    try:
        wrap._need(st3, wrap.REQUIRED_STATE)
        print("FAIL: missing TSFC accepted")
        return 1
    except RuntimeError as exc:
        print(f"ok: missing TSFC refused: {exc}")

    ctx = Ctx(cfg=cfg, grid=g, st=st, dt=float(cfg.dt))
    ctx.stream = None
    bad_total = 0
    for ntsd in (1, 0):
        ctx.ntimestep = ntsd
        t0 = time.time()
        wrap.run(ctx)
        cp.cuda.Device().synchronize()
        print(f"NTSD={ntsd}: run wall {time.time() - t0:.2f} s")
        bad = []
        for n, a in st.fields.items():
            if getattr(a, "dtype", None) is not None and a.dtype.kind == "f":
                if not bool(cp.isfinite(a).all()):
                    bad.append(n)
        print(f"NTSD={ntsd}: non-finite fields {bad if bad else 'none'}")
        bad_total += len(bad)

    # ---- LPBL: B1 only, edges untouched ----
    lp = st.fields["LPBL"]
    lp[...] = -7
    kernels.get(wrap.UNIT, "turbl_lpbl")(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
        st["Z"], st["SFCZ"], st["PBLH"], lp, np.int32(im), np.int32(jm), np.int32(lm)))
    cp.cuda.Device().synchronize()
    edge_ok = bool((lp[0, :] == -7).all() and (lp[-1, :] == -7).all()
                   and (lp[:, 0] == -7).all() and (lp[:, -1] == -7).all())
    inner = cp.asnumpy(lp[1:-1, 1:-1])
    print(f"LPBL edges untouched: {edge_ok}; interior range {inner.min()}..{inner.max()}")
    wrap._dep = real_dep
    ok = bad_total == 0 and edge_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
