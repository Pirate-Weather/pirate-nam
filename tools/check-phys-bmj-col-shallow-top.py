"""Done-rule check for phys-bmj-col-shallow-top (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-bmj-col-shallow-top.py

Compiles namgpu/kernels/phys_bmj_col_shallow_top.cu (NVRTC) and runs the wrapper once on a FULL-SIZE
random column state (954 x 835 x 60) under one pool-v3 lease on the shared card. Checks every array the
kernel writes for NaN/Inf. No bitwise comparison and no column test (CONTRACT section 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.bmj import phys_bmj_col_shallow_top as unit
    from namgpu.steps.drv_gpu_lease import gpu_lease

    kernels.module(unit.UNIT).compile()
    with gpu_lease(vram_mib=2048, est_s=600, cards=1, what="phys-bmj-col-shallow-top check"):
        out = unit.run(None, newswap=True)
        cp.cuda.Device().synchronize()
        out_old = unit.run(None, newswap=False)
        cp.cuda.Device().synchronize()
    bad = []
    for label, o in (("newswap=1", out), ("newswap=0", out_old)):
        for name, arr in o.items():
            if arr.dtype.kind == "f" and not bool(cp.isfinite(arr).all()):
                bad.append(f"{label}:{name}")
        print(f"phys-bmj-col-shallow-top {label}: shallow={int(o['shallow'].sum())} "
              f"iswap={int(o['iswap'].sum())} ltop min {int(o['ltop'].min())} max {int(o['ltop'].max())}")
    print("NaN/Inf in: " + (", ".join(bad) if bad else "none"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
