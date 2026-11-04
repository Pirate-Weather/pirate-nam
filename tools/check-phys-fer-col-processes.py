"""Done-rule check for phys-fer-col-processes (CONTRACT section 6): compile the kernel, run the producer
chain and the unit once on a FULL-SIZE (954 x 835 x 60) random column state, check every array it writes
for NaN/Inf, and check that the table-index and non-convergence flags are as expected.

Producers run first, in the order the column needs them: phys-fer-init tables, phys-fer-fpvs (inside
bind_inputs), phys-fer-col-ice-props (its outputs are inputs of this unit), then this unit. The base
column fields (T, PMID, FER_Q, FER_QW, FER_SC_RHO, ...) are random but physically shaped, standing in for
the column prep and satcheck outputs. Runs inside one pool lease (the card is shared).

    source ~/nam-gpu/env.sh
    python tools/check-phys-fer-col-processes.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.steps.drv_gpu_lease import gpu_lease  # noqa: E402  (no cupy import before the card is granted)


class Fields(dict):
    """ctx.st stand-in: named device fields with st[name] and st.add(name, shape, dtype)."""

    def add(self, name, shape, dtype="float32"):
        import cupy as cp

        if name not in self:
            self[name] = cp.zeros(shape, dtype=dtype)
        return self[name]


def main() -> int:
    with gpu_lease(lane="nam-gpu-phys-fer-col-processes", vram_mib=20000, est_s=600,
                   what="phys-fer-col-processes done-rule") as card:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
        import cupy as cp
        import numpy as np

        from namgpu import kernels
        from namgpu.physics.fer_hires import phys_fer_col_ice_props as ice_props
        from namgpu.physics.fer_hires import phys_fer_col_processes as unit
        from namgpu.physics.fer_hires import phys_fer_init

        grid = SimpleNamespace(im=954, jm=835, lm=60)
        shape = (grid.lm, grid.jm, grid.im)
        st = Fields()
        ctx = SimpleNamespace(tables={}, grid=grid, stream=None, st=st, cfg=None, dt=25.0)
        dtph = np.float32(50.0)
        fer = phys_fer_init._build(ctx, dtph, False)
        ctx.tables["fer"] = fer
        # ice-props reads its tables under the names below; phys-fer-init publishes them under fer[...].
        ctx.tables["fer_massi"] = fer["massi"]
        ctx.tables["fer_vsnowi"] = fer["vsnowi"]
        ctx.tables["fer_vel_rf"] = fer["vel_rf"]
        s = fer["scalars"]
        ctx.tables["fer_scalars"] = {
            "RFmx1": s["RFMX1"], "RFmax": s["RFMAX"], "RH_NgC": s["RH_NGC"], "RH_NgT": s["RH_NGT"],
            "RQhail": s["RQHAIL"], "AVhail": s["AVHAIL"], "BVhail": s["BVHAIL"],
        }

        def U(lo, hi):
            return (lo + (hi - lo) * cp.random.random(shape, dtype=cp.float32)).astype(cp.float32)

        def mask(p):
            return (cp.random.random(shape, dtype=cp.float32) > p).astype(cp.float32)

        cp.random.seed(3)
        tk = U(230.0, 300.0)
        pp = U(3.0e4, 1.0e5)
        rho = (pp / (287.04 * tk)).astype(cp.float32)
        st["T"] = tk
        st["PMID"] = pp
        st["FER_Q"] = U(1.0e-4, 1.0e-2)
        # Clear levels (satcheck CLEAR=1) carry no condensate and RHO=0, as satcheck writes them.
        # A clear level has WC <= EPSQ by definition, so QI, QR, QW are zero there.
        clear = cp.random.random(shape, dtype=cp.float32) > 0.8
        notclear = (~clear).astype(cp.float32)
        st["FER_QW"] = U(0.0, 2.0e-3) * mask(0.3) * notclear
        st["FER_SC_RHO"] = rho * notclear                       # 0 on clear levels (satcheck)
        st["FER_SC_CLEAR"] = clear.astype(cp.int32)            # 1 = CYCLE big_loop
        st["THICK"] = U(100.0, 400.0)
        st["QI"] = U(0.0, 1.0e-4) * mask(0.3) * notclear
        st["QR"] = U(0.0, 1.0e-4) * mask(0.3) * notclear
        st["ARAIN_COL"] = U(0.0, 1.0e-3)
        st["ASNOW_COL"] = U(0.0, 1.0e-3)
        st["F_RIMEF"] = U(1.0, 2.0)
        cp.cuda.Stream.null.synchronize()
        print("state built", flush=True)

        t0 = time.perf_counter()
        ice_props.run(ctx, dtph=float(dtph))                    # producer of the unit's ice inputs
        cp.cuda.Stream.null.synchronize()
        print(f"ice-props wall {time.perf_counter() - t0:.3f} s (includes first NVRTC load)", flush=True)

        kernels.module(unit.UNIT).compile()
        print("kernel compiled", flush=True)

        cols = unit.alloc(ctx, shape)
        t0 = time.perf_counter()
        unit.run(ctx, cols=cols)                                # binds its inputs from the producers, then launches
        cp.cuda.Stream.null.synchronize()
        print(f"unit wall {time.perf_counter() - t0:.3f} s (includes first NVRTC load)", flush=True)

        flags = [int(v) for v in cp.asnumpy(ctx.tables[unit.FLAGS_KEY])]
        print("flags [deposit nonconv, INDEXS out of range, condense nonconv]:", flags[:3], flush=True)
        bad = [name for name in unit.ALL_F32 if not bool(cp.isfinite(cols[name]).all())]
        print("NaN/Inf arrays:", bad, flush=True)
        for name in unit.OUT_F32:
            a = cols[name]
            print(f"{name}: max|x| {float(cp.abs(a).max()):.4g}, nonzero fraction "
                  f"{float(cp.count_nonzero(a)) / a.size:.3f}", flush=True)
        ok = not bad and flags[1] == 0
        print("PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
