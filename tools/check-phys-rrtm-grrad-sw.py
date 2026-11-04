"""Done-rule check for phys-rrtm-grrad-sw (CONTRACT section 6).

Runs the glue wrapper once on the full-size random State, with the real phys-rrtm-dayparts unit
(COSZMN over the SW interval) as the day-predicate source. The two dependency units
(phys-rrtm-setalb, phys-rrtm-sw-swrad) are not built yet, so this check installs stand-ins that fill
their CONTRACT outputs with random values (FDNCMP always, as the Fortran scmpsw argument is
unconditional). The glue arithmetic is then checked on the device result against numpy.

    source ~/nam-gpu/env.sh && python tools/check-phys-rrtm-grrad-sw.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# solar state for the dayparts call (values as phys-rrtm-dayparts selftest: late March, dec ~ +3 deg)
SOL = {"sollag": 0.0, "sindec": float(np.sin(np.deg2rad(3.0))),
       "cosdec": float(np.cos(np.deg2rad(3.0)))}


def _install_stand_ins():
    import namgpu.physics.rrtm as rrtm_pkg

    def setalb_run(ctx):
        st = ctx.st
        rng = np.random.default_rng(3)
        import cupy as cp
        st.SFCALB[...] = cp.asarray(rng.random(st.SFCALB.shape, dtype=np.float32))

    def swrad_run(ctx, has_htrswb=False):
        import cupy as cp
        st = ctx.st
        rng = np.random.default_rng(4)
        st.HTSWC[...] = cp.asarray(rng.random(st.HTSWC.shape, dtype=np.float32))
        st.TOPFLX[...] = cp.asarray(rng.random(st.TOPFLX.shape, dtype=np.float32))
        st.SFCFLX[...] = cp.asarray(rng.random(st.SFCFLX.shape, dtype=np.float32))
        st.FDNCMP[...] = cp.asarray(rng.random(st.FDNCMP.shape, dtype=np.float32))
        if has_htrswb:
            st.HTSWB[...] = cp.asarray(rng.random(st.HTSWB.shape, dtype=np.float32))

    for name, fn in (("phys_rrtm_setalb", setalb_run), ("phys_rrtm_sw_swrad", swrad_run)):
        mod = types.ModuleType(f"namgpu.physics.rrtm.{name}")
        mod.run = fn
        sys.modules[mod.__name__] = mod
        setattr(rrtm_pkg, name, mod)


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_grrad_sw as glue

    _install_stand_ins()
    cfg = config.load()
    cfg.lfdncmp = True          # swrad must fill FDNCMP (scmpsw is unconditional in grrad_nmmb)
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    # longitudes over the globe so the day/night mask has both kinds of column (randomize gives 0..1 rad)
    lon = np.random.default_rng(5).uniform(-np.pi, np.pi, st.GLON.shape).astype(np.float32)
    st.GLON[...] = cp.asarray(lon)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    # solar state and solhr for DAYPARTS (phys-rrtm-sol-update writes SOL_STATE; solhr is ctx.clock.solhr)
    from types import SimpleNamespace
    ctx.clock = SimpleNamespace(solhr=13.0)
    st.add("SOL_STATE", (4,))
    st.SOL_STATE[...] = cp.asarray(np.array([SOL["sollag"], SOL["sindec"], SOL["cosdec"], 0.0], np.float32))
    glue.run(ctx, has_htrswb=True)
    ctx.stream.synchronize()

    bad = [n for n, a in st.fields.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    assert not bad, f"NaN/Inf in {bad}"

    # arithmetic checks against numpy on the device results
    sfcalb = st.SFCALB.get()
    sfalb_ref = np.maximum(np.float32(0.01), np.float32(0.5) * (sfcalb[1] + sfcalb[3]))
    assert np.array_equal(st.SFALB.get(), sfalb_ref.astype(np.float32)), "sfalb"

    # day predicate is the DAYPARTS coszen_loc (interval mean), not the instantaneous CZEN
    coszen = ctx.tables["rrtm_dayparts"]["coszen"].get().reshape(st.SW_COSZEN.shape)
    assert np.array_equal(st.SW_COSZEN.get(), coszen), "sw_coszen"
    day = (coszen >= np.float32(0.0001)).ravel()
    ndays = int(st.SW_NDAY.get()[0])
    assert ndays == int(day.sum()), "nday"
    idx = st.SW_IDXDAY.get()[:ndays]
    assert np.array_equal(idx, np.flatnonzero(day).astype(np.int32)), "idxday"

    # kd = 1: htrsw[l] = htswc[l + 1] for l = 0..LM-1 (grrad_nmmb.f:1177-1178, 1918-1925)
    htswc = st.HTSWC.get()
    htrsw = st.HTRSW.get()
    lm = int(g.lm)
    dayg = day.reshape(coszen.shape)
    assert np.array_equal(htrsw, np.where(dayg[None], htswc[1:lm + 1], np.float32(0.0))), "htrsw"
    topf = st.TOPFLX.get()
    assert np.array_equal(st.TOPFSW.get(), np.where(dayg[None], topf, np.float32(0.0))), "topfsw"
    sfcf = st.SFCFLX.get()
    assert np.array_equal(st.SFCFSW.get(), np.where(dayg[None], sfcf, np.float32(0.0))), "sfcfsw"
    fd = st.FDNCMP.get()
    assert np.array_equal(st.SCMPSW.get(), np.where(dayg[None], fd, np.float32(0.0))), "scmpsw"
    htswb = st.HTSWB.get()
    htrswb = st.HTRSWB.get()
    assert np.array_equal(htrswb, np.where(dayg[None, None], htswb[:, 1:lm + 1], np.float32(0.0))), "htrswb"

    print(f"phys-rrtm-grrad-sw: ok, IM={g.im} JM={g.jm} LM={lm}, day columns={ndays} of {day.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
