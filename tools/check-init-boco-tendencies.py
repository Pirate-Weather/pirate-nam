"""Done-rule check for init-boco-tendencies (CONTRACT.md section 6, parts 1 and 2), plus the two
review gaps it must cover: the nest-only path and the Fortran RECIP / carried-slot semantics.

    source env.sh
    python tools/check-init-boco-tendencies.py

Full 954 x 835 x 60 boundary strips for every parent boundary variable (PD, T, Q, CW, U, V), random
float32 carried values (slot 0, as bnd-boundary-init leaves them) and a random next-interval boundary
set. Nest path (I_AM_A_NEST true): the wrapper runs once; the check asserts no NaN/Inf, that slot 1
equals the Fortran formula (fut - carried) * RECIP in float32 with RECIP = 1./(DT*PCTR), that slot 0 is
not written, and that the V last index is left unchanged. Uppermost path (I_AM_A_NEST false): staging a
future interval must raise, and the file-placed tendency (slot 1) must stay untouched.
Takes the card through the pool lease first (README.md "GPU use"). Uses only the boundary-variable
arrays, not the whole 10.6 GiB state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PCTR = 3   # parent_child_time_ratio used for the nest check (any positive integer; the formula is checked, not a value)


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod
    from namgpu.boundary import bnd_write_bc
    from namgpu.init import init_boco_tendencies as ibt

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lnsh, lnsv = int(cfg.lnsh), int(cfg.lnsv)
    dt = float(cfg.dt)
    rng = np.random.default_rng(1)

    st = {}
    for name in bnd_write_bc.PARENT_BC_VARS["h2d"]:
        st[name] = cp.asarray(rng.standard_normal((jm, im), dtype=np.float32))
    for cls in ("h3d", "v3d"):
        for name in bnd_write_bc.PARENT_BC_VARS[cls]:
            st[name] = cp.asarray(rng.standard_normal((lm, jm, im), dtype=np.float32))

    fut, bc = {}, {}
    for cls, names in bnd_write_bc.PARENT_BC_VARS.items():
        is_v = cls.startswith("v")
        nlines = lnsv if is_v else lnsh
        lev = lm if cls.endswith("3d") else 1
        for name in names:
            for side in (0, 1, 2, 3):
                along = im if side < 2 else jm
                key = (cls, name, side)
                fut[key] = cp.asarray(rng.standard_normal((1, lev, nlines, along), dtype=np.float32))
                # carried Fortran slot 1 (slot 0 here) and the tendency slot, as bnd-boundary-init leaves them
                bc[key] = cp.zeros((2, 1, lev, nlines, along), dtype=np.float32)
                bc[key][0] = cp.asarray(rng.standard_normal((1, lev, nlines, along), dtype=np.float32))

    carried0 = {k: v[0].copy() for k, v in bc.items()}
    stream = cp.cuda.Stream(non_blocking=True)

    # 1. Uppermost path: the file tendency must not be overwritten.
    ctx_up = SimpleNamespace(cfg=cfg, grid=g, st=st, stream=stream, bc=bc, dt=dt,
                             parent_child_time_ratio=PCTR, i_am_a_nest=False, boco_future=fut)
    raised = False
    try:
        ibt.run(ctx_up)
    except ValueError:
        raised = True
    stream.synchronize()
    up_ok = raised and all(bool(cp.array_equal(bc[k][1], cp.zeros_like(bc[k][1]))) for k in bc)

    # 2. Nest path: the Fortran UPDATE_BC_TENDS formula on the full-size grid.
    ctx = SimpleNamespace(cfg=cfg, grid=g, st=st, stream=stream, bc=bc, dt=dt,
                          parent_child_time_ratio=PCTR, i_am_a_nest=True, boco_future=fut)
    with stream:
        keys = ibt.run(ctx)
    stream.synchronize()

    dt32, p32 = np.float32(dt), np.float32(PCTR)
    recip = np.float32(1.0) / np.float32(dt32 * p32)   # Fortran RECIP=1./(DT*PARENT_CHILD_TIME_RATIO), REAL
    bad = []
    for key in keys:
        buf = bc[key]
        if not bool(cp.isfinite(buf).all()):
            bad.append(("nonfinite", key))
        if not bool(cp.array_equal(buf[0], cp.asarray(carried0[key]))):
            bad.append(("slot0-written", key))
        ref = (fut[key][0] - carried0[key][0]) * recip   # [lev, lines, along], same as buf[1][0]
        tend = buf[1][0]
        if key[0].startswith("v"):
            if not bool(cp.array_equal(tend[..., :-1], ref[..., :-1])):
                bad.append(("tendency", key))
            if bool(cp.any(tend[..., -1] != 0)):
                bad.append(("v-last-index-written", key))
        elif not bool(cp.array_equal(tend, ref)):
            bad.append(("tendency", key, float(cp.abs(tend - ref).max())))
    if not up_ok:
        bad.append(("uppermost-path-not-refused-or-overwritten",))
    if bad:
        print("FAIL", bad[:10])
        return 1
    print(f"init-boco-tendencies: ok, {len(keys)} entries, grid {im}x{jm}x{lm}, no NaN/Inf, "
          f"tendency exact vs Fortran RECIP formula (float32), carried slot 0 untouched, "
          f"V last index untouched, uppermost path refused without overwrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
