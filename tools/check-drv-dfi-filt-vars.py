"""Done-rule check for drv-dfi-filt-vars (review fix): full-size, the exact defects of the review.

    source ~/nam-gpu/env.sh
    python tools/check-drv-dfi-filt-vars.py

One full-size random State (954 x 835 x 60). Runs drv-dfi-filt-vars on the real table with the
restart flag false and true, then checks: the kernel's device actions equal the host mirror; every
bundle member has its allocated extent and halo recorded; no NaN/Inf in the bundle fields; and the
refusals named by the review (unknown name on an ignored line, malformed line, rank 4 filter,
unsubstituted REST, cached call with different restart) raise.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _expect_raise(fn, label):
    try:
        fn()
    except ValueError as e:
        print(f"refused ({label}): {str(e)[:110]}")
        return
    raise SystemExit(f"FAIL: {label} was accepted")


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps import drv_dfi_filt_vars as fv

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    stream = cp.cuda.Stream(non_blocking=True)

    kernels.module(fv.UNIT).compile()
    for restart in (False, True):
        cfg.raw["restart"] = restart
        ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
        out = fv.run(ctx)
        stream.synchronize()
        entries = fv.read_entries(fv.DEFAULT_TABLE)
        dev = cp.asnumpy(ctx.tables["DFI_FILT_ACT"])[:len(entries)]
        host = np.asarray(fv.host_actions([fv.CODE.get(c, 0) for _, c in entries], restart),
                          dtype=np.int32)
        assert np.array_equal(dev, host), "device actions differ from host mirror"
        for f in out["bundle_fields"]:
            arr = st.fields[f["name"]]
            assert arr.ndim == f["rank"]
            assert f["halo"] == (fv.IHALO, fv.JHALO)
            if f["rank"] == 3:
                assert f["ungridded"] == (1, int(arr.shape[0]))
            assert bool(cp.isfinite(arr).all()), f"NaN/Inf in {f['name']}"
        print(f"restart={restart}: filter 2D/3D {len(out['filter_2d'])}/{len(out['filter_3d'])}, "
              f"restore 2D/3D {len(out['restore_2d'])}/{len(out['restore_3d'])}, "
              f"bundle members {len(out['bundle_fields'])}, device==host, finite")

    # Cached call with a different restart must rebuild, not return the stale lists.
    cfg.raw["restart"] = True
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
    a = fv.run(ctx)
    cfg.raw["restart"] = False
    b = fv.run(ctx)
    assert a["restart"] is True and b["restart"] is False, "stale cache"
    print("cache: restart change rebuilt the lists")

    # Refusals, each on a small table or config.
    tmp = Path(tempfile.mkdtemp(prefix="dfi-filt-check-"))
    t_unknown_ignored = tmp / "unknown_ignored.txt"
    t_unknown_ignored.write_text("'NOSUCHVAR'  X  'unused code, unknown name'\n")
    _expect_raise(lambda: fv.run(Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt),
                                 table=t_unknown_ignored), "unknown name, ignored code")
    t_malformed = tmp / "malformed.txt"
    t_malformed.write_text("'PD'\n")
    _expect_raise(lambda: fv.run(Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt),
                                 table=t_malformed), "malformed line")
    t_rank4 = tmp / "rank4.txt"
    rank4 = [n for n, s in statemod.all_specs().items() if s.rank == 4]
    if rank4:
        t_rank4.write_text(f"'{rank4[0]}'  F  'rank 4 filter'\n")
        _expect_raise(lambda: fv.run(Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt),
                                     table=t_rank4), f"rank-4 filter {rank4[0]}")
    else:
        print("no rank-4 state field in the tables; rank-4 refusal not exercised")
    cfg.raw["restart"] = "REST"
    _expect_raise(lambda: fv.run(Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt),
                                 table=None), "unsubstituted REST")
    print("check-drv-dfi-filt-vars: PASS")


if __name__ == "__main__":
    main()
