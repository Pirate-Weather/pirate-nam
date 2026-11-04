"""Done-rule check for init-derive-state, full size (review fixes: cold-start values, restart gating, PINT, Z).

    source ~/nam-gpu/env.sh
    python tools/check-init-derive-state.py

1. compiles the kernel file (NVRTC through CuPy);
2. cold start: runs the wrapper once on a full-size (IM x JM x LM from the config) state with plausible values;
   checks every array the wrapper writes for NaN/Inf, and the cold-start values the review named:
   Q2 = 0.02, W = 0, DWDT = 1, O3 = 10 in the 13x13 box and 0 elsewhere, PINT(1) = PT and the native recursion,
   REFL_10CM = DBZmin, TADJ = 0, QSH = 0, TRACERS_PREV(q2) = sqrt(0.02);
3. restart: runs the wrapper with ctx.restart = True on sentinel-filled file-read fields and checks none of them
   was overwritten (PINT, Z, DWDT, OMGALF, UP, VP, TP, TOLD, W, Q2, O3, TRACERS_PREV, REFL_10CM, TADJ, QSH, THS).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESTART_READ = ["PINT", "Z", "DWDT", "OMGALF", "UP", "VP", "TP", "TOLD", "W", "Q2", "O3",
                "TRACERS_PREV", "REFL_10CM", "TADJ", "QSH", "THS"]


def _make_state(g, cp, cfg, restart: bool):
    from namgpu import state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps.drv_tracer_index import host_index

    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    st = statemod.State(g, names=set())
    rng = cp.random.default_rng(7)

    def f3(lo, hi):
        return (lo + (hi - lo) * rng.random((lm, jm, im), dtype=cp.float32)).astype(cp.float32)

    def f2(lo, hi):
        return (lo + (hi - lo) * rng.random((jm, im), dtype=cp.float32)).astype(cp.float32)

    sg2 = np.linspace(0.0, 1.0, lm + 1).astype(np.float32)
    sg1 = (1.0 - sg2).astype(np.float32)
    st.add("SG1", (lm + 1,))[...] = cp.asarray(sg1)
    st.add("SG2", (lm + 1,))[...] = cp.asarray(sg2)
    st.add("DSG1", (lm,))[...] = cp.asarray((-np.diff(sg1)).astype(np.float32))
    st.add("DSG2", (lm,))[...] = cp.asarray((-np.diff(sg2)).astype(np.float32))
    st.add("EPSQ2", (lm,))
    st.add("PD", (jm, im))[...] = f2(80000.0, 85000.0)
    st.add("FIS", (jm, im))[...] = f2(0.0, 2.0e4)
    st.add("TSKIN", (jm, im))[...] = f2(280.0, 300.0)
    st.add("T", (lm, jm, im))[...] = f3(220.0, 300.0)
    st.add("Q", (lm, jm, im))[...] = f3(0.0, 0.02)
    st.add("CW", (lm, jm, im))[...] = f3(0.0, 1.0e-3)
    st.add("U", (lm, jm, im))[...] = f3(-20.0, 20.0)
    st.add("V", (lm, jm, im))[...] = f3(-20.0, 20.0)
    st.add("Q2", (lm, jm, im))[...] = f3(0.0, 1.0)
    tab = host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))
    # Only the three slots this unit writes. The full NUM_TRACERS_TOTAL array would not fit the shared card.
    ntr = max(int(tab["INDX_Q"]), int(tab["INDX_CW"]), int(tab["INDX_Q2"])) + 1
    st.add("TRACERS", (ntr, lm, jm, im))
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt, restart=restart)
    ctx.tables = {}
    return ctx, st


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels
    from namgpu.init import init_derive_state as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    kernels.module(unit.UNIT).compile()
    fails = []

    # 2. cold start
    ctx, st = _make_state(g, cp, cfg, restart=False)
    unit.run(ctx)
    ctx.stream.synchronize()
    written = ["PINT", "Z", "RTOP", "DWDT", "OMGALF", "UP", "VP", "TP", "TOLD", "W", "O3", "Q2",
               "TRACERS", "TRACERS_PREV", "THS", "REFL_10CM", "TADJ", "QSH", "EPSQ2"]
    bad = [n for n in written if not bool(cp.isfinite(st.fields[n]).all())]
    print("cold NaN/Inf arrays:", bad if bad else "none")
    fails += [f"nonfinite {n}" for n in bad]

    def chk(name, ok):
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    chk("Q2 == 0.02", bool((st["Q2"] == np.float32(0.02)).all()))
    chk("W == 0", bool((st["W"] == 0).all()))
    chk("DWDT == 1", bool((st["DWDT"] == 1).all()))
    ci, cj = im // 2 + 1, (jm * 3) // 4 + 1
    o3 = st["O3"]
    box = cp.zeros((jm, im), dtype=bool)
    box[cj - 7:cj + 6, ci - 7:ci + 6] = True          # 1-based i in [ci-6, ci+6], j in [cj-6, cj+6]
    expect = cp.where(box[None], np.float32(10.0), np.float32(0.0))
    chk("O3 box 10, else 0 (every level)", bool((o3 == expect).all()))
    pint = st["PINT"]
    chk("PINT(1) == PT", bool((pint[0] == np.float32(g.pt)).all()))
    dsg1, dsg2 = st["DSG1"], st["DSG2"]
    pd = st["PD"]
    exp_step = dsg2[:, None, None] * pd[None] + (dsg1 * np.float32(g.pdtop))[:, None, None]
    rel = float(cp.max(cp.abs((pint[1:] - pint[:-1]) - exp_step) / cp.maximum(cp.abs(exp_step), 1.0)))
    chk(f"PINT native recursion (max rel err {rel:.2e})", rel < 1e-5)
    chk("REFL_10CM == DBZmin", bool((st["REFL_10CM"] == np.float32(-20.0)).all()))
    chk("TADJ == 0", bool((st["TADJ"] == 0).all()))
    chk("QSH == 0", bool((st["QSH"] == 0).all()))
    from namgpu.init import init_derive_state as ids
    n_q, n_cw, n_q2 = ids._tracer_slots(ctx)
    tp = st["TRACERS_PREV"]
    chk("TRACERS_PREV(q2) == sqrt(0.02)", bool((tp[n_q2] == np.float32(np.sqrt(np.float32(0.02)))).all()))
    chk("TRACERS(q2) == Q2", bool((st["TRACERS"][n_q2] == st["Q2"]).all()))

    # 3. restart: file-read fields must keep their sentinel values
    del ctx, st
    cp.get_default_memory_pool().free_all_blocks()
    ctx_r, st_r = _make_state(g, cp, cfg, restart=True)
    sentinel = np.float32(7.0)
    shapes = {
        "PINT": (lm + 1, jm, im), "Z": (lm + 1, jm, im), "THS": (jm, im), "QSH": (jm, im),
        "TRACERS_PREV": tuple(st_r["TRACERS"].shape),
    }
    for n in RESTART_READ:
        if n not in st_r.fields:
            st_r.add(n, shapes.get(n, (lm, jm, im)))
        st_r[n][...] = sentinel
    unit.run(ctx_r)
    ctx_r.stream.synchronize()
    moved = [n for n in RESTART_READ if not bool((st_r[n] == sentinel).all())]
    chk(f"restart leaves file-read fields alone (overwritten: {moved if moved else 'none'})", not moved)
    bad_r = [n for n in ["RTOP", "TRACERS", "TRACERS_PREV"] if not bool(cp.isfinite(st_r[n]).all())]
    print("restart NaN/Inf arrays:", bad_r if bad_r else "none")
    fails += [f"restart nonfinite {n}" for n in bad_r]

    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
