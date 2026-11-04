"""Done-rule check for init-vcoord-recover (CONTRACT.md section 6, parts 1 and 2), full size.

    source env.sh
    python tools/check-init-vcoord-recover.py

1. compiles the kernel file;
2. recovers a known, valid hybrid coordinate at full size (954 x 835 x 60): interfaces SG1, SG2
   built with SG(1)=0, SG(LM+1)=1, PRES = PSGML1 + SGML2*PD from those interfaces. Checks that
   the recovered SG1/SG2 equal the truth, that the surface identity holds (PINT(LM+1) = PSFC
   with PD = PSFC - PDTOP - PT), and that no float array is NaN/Inf after the unit;
3. constant PD (the named breakage): the unit must refuse, not write a zero slope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.init import init_vcoord_recover as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    im, jm, lm = g.im, g.jm, g.lm
    rng = cp.random.default_rng(7)

    pt = float(cfg.pt)
    pdtop = float(cfg.ptsgm) - pt
    psfc = (80000.0 + 21000.0 * rng.random((jm, im), dtype=cp.float32)).astype(cp.float32)
    st.fields["PSFC"][...] = psfc
    pd = psfc - np.float32(pdtop) - np.float32(pt)

    # Truth: valid interfaces (0 at top, 1 at surface, monotone), mid-layer values from them.
    t = np.linspace(0.0, 1.0, lm + 1)
    sg1_true = t.copy()
    sg2_true = t ** 2
    psgml1_true = (0.5 * (sg1_true[1:] + sg1_true[:-1])) * pdtop + pt
    sgml2_true = 0.5 * (sg2_true[1:] + sg2_true[:-1])
    a_true = cp.asarray(psgml1_true, dtype=cp.float32)
    b_true = cp.asarray(sgml2_true, dtype=cp.float32)
    pres = (a_true[:, None, None] + b_true[:, None, None] * pd[None, :, :]).astype(cp.float32)
    st.add("PRES", (lm, jm, im))[...] = pres

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    kernels.module(unit.UNIT).compile()
    unit.run(ctx)
    ctx.stream.synchronize()

    bad = [n for n, a in st.fields.items()
           if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    print("NaN/Inf arrays:", bad[:10] if bad else "none")

    err_sg1 = float(np.max(np.abs(g.sg1.astype(np.float64) - sg1_true)))
    err_sg2 = float(np.max(np.abs(g.sg2.astype(np.float64) - sg2_true)))
    surf_pint = float(g.sg2[-1]) * float(pd[0, 0]) + float(g.sg1[-1]) * pdtop + pt
    print(f"max |SG1 - true| = {err_sg1:.3g}, max |SG2 - true| = {err_sg2:.3g}")
    print(f"surface: SG1(LM+1)={g.sg1[-1]}, SG2(LM+1)={g.sg2[-1]}, "
          f"PINT(LM+1)-PSFC at column 0 = {surf_pint - float(psfc[0, 0]):.3g} Pa")
    print("receipt:", ctx.tables.get(unit.RECEIPT_KEY))
    ok_main = (not bad) and err_sg1 < 1e-5 and err_sg2 < 1e-5 and abs(surf_pint - float(psfc[0, 0])) < 0.5

    # Constant PD: the unit must refuse (named breakage: silent SG2(LM+1)=0).
    st.fields["PSFC"][...] = np.float32(90000.0)
    refused = False
    try:
        unit.run(ctx)
        ctx.stream.synchronize()
    except RuntimeError as e:
        refused = "undetermined" in str(e)
        print("constant PD refused:", str(e)[:120])
    print("constant PD refused:", refused)
    return 0 if (ok_main and refused) else 1


if __name__ == "__main__":
    sys.exit(main())
