"""Per-column value check for phys-rrtm-lw-taugb08 against a scalar re-implementation.

    source env.sh
    python tools/check_taugb08_values.py

Runs the GPU wrapper on the random full-size inputs of tools/check_phys_rrtm_lw_taugb08.py (the same
generator), then recomputes the band-8 taug and fracs for four columns on the CPU in float32 with a
scalar loop that follows radlw_main_nmmb.f:4946-5065 statement by statement (independent of the CUDA
index helpers). Also checks the table values against hand-read Fortran DATA literals. Passes when the
max relative difference is below 1e-5 and no value is NaN.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

COLS = [(0, 0), (0, 1), (5, 17), (20, 900)]   # (j, i) column picks across the grid


def reference(inp, tab, j, i):
    """Scalar float32 re-implementation of taugb08 for one column; returns taug, fracs [140, LM]."""
    f = np.float32
    lm = inp["coldry"].shape[0]
    taug = np.zeros((140, lm), dtype=f)
    fracs = np.zeros((140, lm), dtype=f)
    lt = int(inp["laytrop"][j, i])
    g = lambda name, x: inp[name][x]
    for k in range(lm):   # k = 0-based layer
        jp = int(g("jp", (k, j, i))); jt = int(g("jt", (k, j, i))); jt1 = int(g("jt1", (k, j, i)))
        ins = int(g("indself", (k, j, i))); inf = int(g("indfor", (k, j, i)))
        inm = int(g("indminor", (k, j, i)))
        cd = f(g("coldry", (k, j, i)))
        ca = [f(inp["colamt"][n, k, j, i]) for n in range(6)]
        wx = [f(inp["wx"][n, k, j, i]) for n in range(4)]
        mf = f(g("minorfrac", (k, j, i)))
        temp = f(cd * tab["chi_mls_co2"][jp])
        ratco2 = f(ca[1] / temp)
        adjcol = f(f(2.0) + f(np.power(f(ratco2 - 2.0), f(0.65)))) * temp if ratco2 > 3.0 else ca[1]
        if k < lt:   # lower atmosphere, jp 1..11, 1-based indices from setcoef
            r0 = (jp - 1) * 5 + (jt - 1); r1 = jp * 5 + (jt1 - 1)
            s0 = ins - 1; f0 = inf - 1; m0 = inm - 1
            f00, f10, f01, f11 = (f(inp[n][k, j, i]) for n in ("fac00", "fac10", "fac01", "fac11"))
            sf, sfr = f(inp["selffac"][k, j, i]), f(inp["selffrac"][k, j, i])
            ff, ffr = f(inp["forfac"][k, j, i]), f(inp["forfrac"][k, j, i])
            for ig in range(8):
                ts = sf * (tab["selfref"][s0, ig] + sfr * (tab["selfref"][s0 + 1, ig] - tab["selfref"][s0, ig]))
                tf = ff * (tab["forref"][f0, ig] + ffr * (tab["forref"][f0 + 1, ig] - tab["forref"][f0, ig]))
                def mix(a):
                    return a[m0, ig] + mf * (a[m0 + 1, ig] - a[m0, ig])
                aco2, ao3, an2o = mix(tab["ka_mco2"]), mix(tab["ka_mo3"]), mix(tab["ka_mn2o"])
                adj = adjcol
                gl = (ca[0] * (f00 * tab["absa"][r0, ig] + f10 * tab["absa"][r0 + 1, ig]
                               + f01 * tab["absa"][r1, ig] + f11 * tab["absa"][r1 + 1, ig])
                      + ts + tf + adj * aco2 + ca[2] * ao3 + ca[3] * an2o
                      + wx[2] * tab["cfc12"][ig] + wx[3] * tab["cfc22adj"][ig])
                taug[88 + ig, k] = f(gl); fracs[88 + ig, k] = tab["fracrefa"][ig]
        else:        # upper atmosphere, jp 13..57
            r0 = (jp - 13) * 5 + (jt - 1); r1 = (jp - 12) * 5 + (jt1 - 1)
            m0 = inm - 1
            f00, f10, f01, f11 = (f(inp[n][k, j, i]) for n in ("fac00", "fac10", "fac01", "fac11"))
            mf = f(inp["minorfrac"][k, j, i])
            for ig in range(8):
                def mix(a):
                    return a[m0, ig] + mf * (a[m0 + 1, ig] - a[m0, ig])
                aco2, an2o = mix(tab["kb_mco2"]), mix(tab["kb_mn2o"])
                gu = (ca[2] * (f00 * tab["absb"][r0, ig] + f10 * tab["absb"][r0 + 1, ig]
                               + f01 * tab["absb"][r1, ig] + f11 * tab["absb"][r1 + 1, ig])
                      + adjcol * aco2 + ca[3] * an2o
                      + wx[2] * tab["cfc12"][ig] + wx[3] * tab["cfc22adj"][ig])
                taug[88 + ig, k] = f(gu); fracs[88 + ig, k] = tab["fracrefb"][ig]
    return taug, fracs


def main():
    import cupy as cp
    from check_phys_rrtm_lw_taugb08 import random_inputs, JM, IM
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb08 as wrap

    z = np.load(wrap.TABLE_FILE)
    tab = {k: z[k].astype(np.float32) for k in z.files}
    # table value checks against the Fortran DATA literals (radlw_datatb.f:24575-24580)
    assert tab["absa"][0, 1] == np.float32(3.183247e-05), "absa(2,1) scrambled"
    assert tab["absa"][1, 1] == np.float32(4.539081e-05), "absa(2,2) scrambled"
    assert tab["absa"][1, 0] == np.float32(1.958781e-05), "absa(1,2) scrambled"

    kernels.module("phys_rrtm_lw_taugb08").compile()
    ctx = types.SimpleNamespace(tables={}, stream=cp.cuda.Stream(non_blocking=True))
    inp = random_inputs(seed=1)
    host = {k: cp.asnumpy(v) for k, v in inp.items()}
    worst = 0.0
    for j, i in COLS:
        # one column per launch: the full 140 x LM x NC output is 26.7 GB and does not fit beside the inputs
        with ctx.stream:
            taug, fracs = wrap.run(ctx, inp, c0=j * IM + i, ncol=1)
        ctx.stream.synchronize()
        got_t = cp.asnumpy(taug[88:96, :, 0])        # [8, LM]
        got_f = cp.asnumpy(fracs[88:96, :, 0])
        ref_t, ref_f = reference(host, tab, j, i)
        assert not np.isnan(got_t).any() and not np.isnan(got_f).any(), (j, i)
        d = np.abs(got_t - ref_t[88:96]) / np.maximum(np.abs(ref_t[88:96]), 1e-30)
        df = np.abs(got_f - ref_f[88:96])
        worst = max(worst, float(d.max()), float(df.max()))
        print(f"col (j={j}, i={i}): max rel taug diff {d.max():.3g}, max abs fracs diff {df.max():.3g}")
    assert worst < 1e-5, worst
    print(f"PASS worst {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
