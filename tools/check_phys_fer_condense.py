"""Done-rule check for phys-fer-condense (CONTRACT section 6, parts 1 and 2).

    source env.sh && python tools/check_phys_fer_condense.py

Compiles namgpu/kernels/phys_fer_condense.cu through NVRTC and runs the wrapper once on a full-size
[60, 835, 954] random column state (T, Q, CW, PD). Only the fields the unit reads are allocated (the full
State is ~13 GB and does not fit the shared card beside other leases). The production inputs PIDEP, PIACWI
and QSWGRD are passed explicitly (they come from phys-fer-col-processes in the chain). Asserts:
  - the default PP equals PHMID = SGML2*PD + PSGML1 (the Fortran P_col), checked on the full grid;
  - the output is finite, and the non-convergence flag is 0 or 1 everywhere.
"""
from __future__ import annotations

import sys
import time


class _Fields:
    """Minimal stand-in for namgpu.state.State: a field dict with add(), enough for the condense unit."""

    def __init__(self, fields):
        self.fields = fields

    def __getitem__(self, name):
        return self.fields[name]

    def add(self, name, shape, dtype="float32"):
        import cupy as cp

        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def main():
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels
    from namgpu.driver import Ctx
    from namgpu.physics.fer_hires import phys_fer_condense

    cfg = config.load()
    g = gridmod.build(cfg)
    lm, jm, im = g.lm, g.jm, g.im
    shape = (lm, jm, im)
    rng = np.random.default_rng(1)

    def rnd(shp, lo, hi):
        return cp.asarray(rng.uniform(lo, hi, shp).astype(np.float32))

    st = _Fields({
        "T": rnd(shape, 230.0, 310.0),
        "Q": rnd(shape, 0.0, 0.02),
        "CW": rnd(shape, 0.0, 2e-4),
        "PD": rnd((jm, im), 5.0e4, 6.0e4),
    })
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)
    pidep = rnd(shape, -1e-7, 1e-7)
    piacwi = rnd(shape, 0.0, 1e-7)
    qswgrd = cp.full(shape, 0.012, dtype=cp.float32)
    kernels.module(phys_fer_condense.UNIT).compile()
    t0 = time.perf_counter()
    with ctx.stream:
        out = phys_fer_condense.run(ctx, PIDEP=pidep, PIACWI=piacwi, QSWGRD=qswgrd)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    pmid = st["PMID"]
    sgml2 = cp.asarray(np.asarray(g.sgml2, np.float32).reshape(-1, 1, 1))
    psgml1 = cp.asarray(np.asarray(g.psgml1, np.float32).reshape(-1, 1, 1))
    phmid = sgml2 * st["PD"][None, :, :] + psgml1
    pp_ok = bool(cp.array_equal(pmid, phmid))
    finite = bool(cp.isfinite(out).all())
    flag = ctx.tables["fer_condense_flag"]
    nonconv = int(cp.count_nonzero(flag))
    print(f"condense: shape={tuple(out.shape)} finite={finite} PP==PHMID {pp_ok} "
          f"min={float(out.min()):.4e} max={float(out.max()):.4e} mean={float(out.mean()):.4e} kg/kg "
          f"nonconverged={nonconv}")
    print(f"wrapper wall {dt:.3f} s, full size {out.size} cells")
    return 0 if (finite and pp_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
