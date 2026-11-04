"""Done-rule check for phys-rrtm-sw-cldprop (CONTRACT section 6, parts 1 and 2).

    source env.sh
    python tools/check_phys_rrtm_sw_cldprop.py

Compiles the kernel (NVRTC) and runs the wrapper once at full size (954 x 835 x 60) for the standard
cloud scheme (isubcsw=0), checking the outputs for NaN/Inf. Then it runs the clear-column gate
(radsw_main_nmmb.f:985-1002): columns with cf1 = 0 and cfrac = 0 must give zero taucw/ssacw/asycw/cldfrc,
with no NaN. The McICA path (isubcsw > 0) is refused by the wrapper until the swrad seam calls
mcica-subcol, so it is not run here. The wrapper only reads the grid shape from ctx.st["T"], so this check
allocates only this unit's arrays, not the whole 10.6 GiB State. No column test, no bitwise comparison.
"""
from __future__ import annotations

import importlib
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LM, JM, IM = 60, 835, 954


def main():
    import cupy as cp
    from namgpu import kernels

    st = {"T": cp.empty((LM, JM, IM), dtype=cp.float32)}
    ctx = types.SimpleNamespace(st=st, tables={}, stream=cp.cuda.Stream(non_blocking=True))
    wrap = importlib.import_module("namgpu.physics.rrtm.phys_rrtm_sw_cldprop")
    kernels.module("phys_rrtm_sw_cldprop").compile()
    print("compiled phys_rrtm_sw_cldprop", flush=True)
    ok = True
    for isub in (0,):
        t0 = time.perf_counter()
        with ctx.stream:
            taucw, ssacw, asycw, cldfrc, cldf = wrap.run(ctx, isubcsw=isub)
        ctx.stream.synchronize()
        arrs = [taucw, ssacw, asycw] + [a for a in (cldfrc, cldf) if a is not None]
        nbad = sum(int((~cp.isfinite(a)).sum()) for a in arrs)
        dt = time.perf_counter() - t0
        print(f"isubcsw={isub}: nonfinite={nbad} taucw min={float(taucw.min()):.4g} "
              f"max={float(taucw.max()):.4g} mean={float(taucw.mean()):.4g} "
              f"cldfrc={'ok' if cldfrc is not None else 'none'} cldf={'ok' if cldf is not None else 'none'} "
              f"({dt:.2f} s)", flush=True)
        if nbad:
            ok = False
    # clear-column gate: first 100 columns have cf1 = 0 and cfrac = 0 (as swrad_zero_cld leaves them)
    inp = wrap._random_inputs(LM, JM, IM, seed=1)
    inp["cf1"][:, :100] = 0.0
    inp["cfrac"][:, :, :100] = 0.0
    with ctx.stream:
        taucw, ssacw, asycw, cldfrc, _ = wrap.run(ctx, inputs=inp, isubcsw=0)
    ctx.stream.synchronize()
    clear = [taucw[:, :, :, :100], ssacw[:, :, :, :100], asycw[:, :, :, :100], cldfrc[:, :, :100]]
    nbad = sum(int((~cp.isfinite(a)).sum()) for a in [taucw, ssacw, asycw, cldfrc])
    nnonzero = sum(int((a != 0).sum()) for a in clear)
    print(f"clear columns (cf1 = 0): nonfinite={nbad} nonzero_in_clear={nnonzero} "
          f"ssacw_clear_max={float(ssacw[:, :, :, :100].max()):.4g}", flush=True)
    if nbad or nnonzero:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
