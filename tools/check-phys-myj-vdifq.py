"""Done-rule check for phys-myj-vdifq (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-myj-vdifq.py

Compiles namgpu/kernels/phys_myj_vdifq.cu (NVRTC) and runs the wrapper once on a FULL-SIZE random
state (954 x 835 x 60) under one pool-v3 lease on the shared card. Checks Q2 for NaN/Inf and that the EPSQ2 floor holds at every level (Q2 >= EPSQ2).
No bitwise comparison and no column test (CONTRACT section 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.myj import phys_myj_vdifq as unit
    from namgpu.steps.drv_gpu_lease import gpu_lease

    kernels.module(unit.UNIT).compile()
    with gpu_lease(vram_mib=2048, est_s=600, cards=1, what="phys-myj-vdifq check"):
        out = unit.run(None)
        cp.cuda.Device().synchronize()
    q2 = out["Q2"]
    n_bad = int(cp.count_nonzero(~cp.isfinite(q2)))
    print("shape", tuple(q2.shape), "min", float(q2.min()), "max", float(q2.max()))
    print("non-finite Q2 values:", n_bad)
    epsq2 = out["EPSQ2"]
    n_under = int(cp.count_nonzero(q2 < epsq2[:, None, None]))
    print("Q2 below EPSQ2 floor:", n_under)
    if n_bad or n_under:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
