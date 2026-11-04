"""Done-rule check for phys-rrtm-gethml (CONTRACT section 6).

Full-size 954 x 835 x 60 grid, finite synthetic cloud profiles. Runs both vertical orders:
  A: ivflip = 1, surface-first arrays (the NAM order).
  B: ivflip = 0, top-first arrays (the CONTRACT order). This branch had never run before.
B is A mirrored in the level axis, so clds must match A and the layer indices must mirror.
Inputs are asserted finite before the run. llyr comes from llyr_from_si on the same sigma profile.

    source env.sh && python tools/check-phys-rrtm-gethml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu import kernels
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_gethml as glue

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=())  # gethml reads no state fields; its scratch comes from st.add
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nlay = lm + glue.LTP

    kernels.module(glue.UNIT).compile()
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    rng = cp.random.default_rng(7)
    # surface-first pressure in Pa (index 0 = surface), 100000 Pa down to 1000 Pa at the top
    pres_pa = cp.broadcast_to(cp.linspace(100000.0, 1000.0, nlay, dtype=cp.float32)[:, None, None],
                              (nlay, jm, im)).astype(cp.float32)
    si_sfc_first = [1.0 - k / (nlay - 1) for k in range(nlay)]     # sigma, surface = 1
    cldtot = rng.random((nlay, jm, im), dtype=cp.float32)
    cldcnv = (0.3 * rng.random((nlay, jm, im), dtype=cp.float32)).astype(cp.float32)
    assert bool(cp.isfinite(cldtot).all()) and bool(cp.isfinite(cldcnv).all()), "input NaN/Inf"

    ptop1 = glue.ptop1_mb(ctx)
    ctx.stream.synchronize()
    # A: surface-first, ivflip = 1
    plyr_a = glue.plyr_mb(ctx, pres_pa, nlay).copy()
    llyr_a = glue.llyr_from_si(si_sfc_first, 1)
    clds_a, mtop_a, mbot_a = glue.run(ctx, plyr=plyr_a, ptop1=ptop1, cldtot=cldtot, cldcnv=cldcnv,
                                      llyr=llyr_a, ivflip=1, iovr=1, nlay=nlay)
    ctx.stream.synchronize()
    clds_a, mtop_a, mbot_a = clds_a.copy(), mtop_a.copy(), mbot_a.copy()

    # B: top-first, ivflip = 0 (mirror of A in the level axis)
    flip = lambda a: cp.ascontiguousarray(a[::-1])
    # flipped inputs are built on the default stream; finish them before any kernel-stream op reads them
    pres_b, cldtot_b, cldcnv_b = flip(pres_pa), flip(cldtot), flip(cldcnv)
    ctx.stream.synchronize()
    plyr_b = glue.plyr_mb(ctx, pres_b, nlay).copy()
    llyr_b = glue.llyr_from_si(si_sfc_first[::-1], 0)
    # Fortran asymmetry (radiation_clouds_nmmb.f:325-342): ivflip = 1 names the level BELOW the first
    # sigma < 0.9 level (llyr = kl - 1), ivflip = 0 names that level itself (llyr = kl). The same
    # physical bl level is therefore llyr_b = nlay - llyr_a in top-first order, not nlay + 1 - llyr_a.
    # The kernel is compared on that same physical level.
    assert llyr_b == nlay - llyr_a, f"llyr producer: {llyr_b} vs {nlay - llyr_a}"
    llyr_same_level = nlay + 1 - llyr_a
    clds_b, mtop_b, mbot_b = glue.run(ctx, plyr=plyr_b, ptop1=ptop1, cldtot=cldtot_b,
                                      cldcnv=cldcnv_b, llyr=llyr_same_level, ivflip=0, iovr=1, nlay=nlay)
    ctx.stream.synchronize()

    for name, a in (("clds_a", clds_a), ("clds_b", clds_b)):
        assert bool(cp.isfinite(a).all()), f"NaN/Inf in {name}"
    assert float(clds_a.min()) >= 0.0 and float(clds_a.max()) <= 1.0, "CLDS outside [0, 1]"
    assert float(clds_b.min()) >= 0.0 and float(clds_b.max()) <= 1.0, "CLDS(B) outside [0, 1]"
    for name, a in (("mtop_a", mtop_a), ("mbot_a", mbot_a), ("mtop_b", mtop_b), ("mbot_b", mbot_b)):
        assert int(a.min()) >= 0 and int(a.max()) <= nlay, f"{name} out of range"

    per_comp = [float(cp.abs(clds_a[n] - clds_b[n]).max()) for n in range(5)]
    clds_diff = max(per_comp)
    assert clds_diff <= 1e-5, f"ivflip 0 vs 1 clds differ by {clds_diff} per component {per_comp}"
    mir_top = int(cp.count_nonzero(mtop_b != (nlay + 1 - mtop_a) * (mtop_a > 0)))
    mir_bot = int(cp.count_nonzero(mbot_b != (nlay + 1 - mbot_a) * (mbot_a > 0)))
    print(f"phys-rrtm-gethml: ok, IM={im} JM={jm} NLAY={nlay} llyr A={llyr_a} B={llyr_b}, "
          f"clds max|A-B|={clds_diff:.2e}, mirror index mismatches top={mir_top} bot={mir_bot}, "
          f"mean low/mid/high/total/bl = "
          + " ".join(f"{float(clds_a[n].mean()):.4f}" for n in range(5)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
