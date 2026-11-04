"""Value check for unit io-uphlcy (review: the kernel had only a NaN/Inf check).

    source env.sh && python tools/check-io-uphlcy.py

Run under the GPU pool lease. Full-size NAM 12 km grid (954 x 835 x 60), random state where W_TOT
(the Fortran W dummy) and the bare W differ, so the W<0 kill test and the term are both exercised.

Asserts:
1. The io_uphlcy wrapper binds W_TOT, not W: its output equals a host reference of CALC_UPHLCY
   (module_DIAGNOSE.F90:2114-2186) that reads W_TOT, for both layers, in float32, with the same
   loop order and EXIT semantics.
2. No NaN/Inf in the four outputs.
Only the fields the unit reads are allocated (about 1.3 GB), so it runs beside other lanes.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

G = np.float32(9.8060226)   # NMMB_G, module_CONSTANTS.F90:28


def reference(U, V, W, Z, FIS, DXH, DYH, lo, hi, UMAX, UMIN):
    """Host CALC_UPHLCY, vectorised over (j, i), level loop in Fortran order. Arrays are [LM,JM,IM]."""
    LM, JM, IM = Z.shape
    f32 = np.float32
    zint = FIS / G                                   # ZINTSFC, [JM,IM]
    uphl = np.zeros((JM, IM), f32)
    done = np.zeros((JM, IM), bool)
    # interior cells: I in [2, IM-1], J in [2, JM-1] (1-based) -> [1, IM-2], [1, JM-2] 0-based
    jrows = np.arange(1, JM - 1)
    ok_row = DXH[jrows] >= f32(0.1)
    r2dy = f32(1.0) / (f32(2.0) * DYH)
    rdy = f32(2.0) * r2dy
    dxj = DXH[jrows].astype(f32)[:, None]
    r2dx = f32(1.0) / (f32(2.0) * dxj)
    rdx = f32(2.0) * r2dx
    sl = (slice(1, JM - 1), slice(1, IM - 1))
    zi = zint[sl]
    d = done[sl]
    u = uphl[sl]
    for l in range(LM - 1):
        zmid = Z[l][sl]
        zagl = zmid - zi
        act = ((zagl >= f32(lo)) & (zagl <= f32(hi))) & ~d
        if not act.any():
            continue
        wv = W[l][sl]
        dz = Z[l][sl] - Z[l + 1][sl]
        kill = act & (wv < f32(0.0))
        u[kill] = f32(0.0)
        d[kill] = True
        go = act & ~kill
        if not go.any():
            continue
        vm1 = f32(0.5) * (V[l][1:JM - 1, 1:IM - 1] + V[l][0:JM - 2, 1:IM - 1])
        vm2 = f32(0.5) * (V[l][1:JM - 1, 0:IM - 2] + V[l][0:JM - 2, 0:IM - 2])
        dvdx = (vm1 - vm2) * rdx
        um1 = f32(0.5) * (U[l][1:JM - 1, 0:IM - 2] + U[l][1:JM - 1, 1:IM - 1])
        um2 = f32(0.5) * (U[l][0:JM - 2, 0:IM - 2] + U[l][0:JM - 2, 1:IM - 1])
        dudy = (um1 - um2) * rdy
        term = (dvdx - dudy) * wv * dz
        u[go] = u[go] + term[go]
    uphl[sl] = u
    keep = np.zeros((JM, IM), bool)
    keep[sl] = ok_row[:, None]
    mx = UMAX.copy()
    mn = UMIN.copy()
    mx[keep] = np.maximum(uphl[keep], UMAX[keep])
    mn[keep] = np.minimum(uphl[keep], UMIN[keep])
    return mx, mn, keep


def main() -> int:
    import cupy as cp
    from namgpu import config, grid as gridmod
    from namgpu.io import io_uphlcy

    cfg = config.load()
    g = gridmod.build(cfg)
    IM, JM, LM = int(g.im), int(g.jm), int(g.lm)
    rng = np.random.default_rng(7)
    f32 = np.float32
    Z = (15000.0 - 280.0 * np.arange(LM, dtype=f32))[:, None, None] + rng.normal(0, 20, (LM, JM, IM)).astype(f32)
    FIS = rng.uniform(0, 3000.0 * 9.8, (JM, IM)).astype(f32)
    U = rng.normal(0, 15, (LM, JM, IM)).astype(f32)
    V = rng.normal(0, 15, (LM, JM, IM)).astype(f32)
    W_TOT = rng.normal(0, 1, (LM, JM, IM)).astype(f32)                  # W + BARO stand-in
    W = W_TOT + rng.normal(0.5, 0.5, (LM, JM, IM)).astype(f32)           # a different bare W
    DXH = np.full(JM, 12000.0, f32)
    DYH = f32(12000.0)
    init_max = np.full((JM, IM), -1.0e30, f32)
    init_min = np.full((JM, IM), 1.0e30, f32)

    st = types.SimpleNamespace(
        U=cp.asarray(U), V=cp.asarray(V), W=cp.asarray(W), W_TOT=cp.asarray(W_TOT),
        Z=cp.asarray(Z), FIS=cp.asarray(FIS),
        UPHLMAX25=cp.asarray(init_max), UPHLMIN25=cp.asarray(init_min),
        UPHLMAX03=cp.asarray(init_max), UPHLMIN03=cp.asarray(init_min))
    gr = types.SimpleNamespace(im=IM, jm=JM, lm=LM, dxh=DXH, dyh=DYH)
    ctx = types.SimpleNamespace(st=st, grid=gr, tables={}, stream=None)
    io_uphlcy.run(ctx)
    cp.cuda.Device().synchronize()

    bad = [n for n in ("UPHLMAX25", "UPHLMIN25", "UPHLMAX03", "UPHLMIN03")
           if not bool(cp.isfinite(getattr(st, n)).all())]
    if bad:
        print(f"NaN/Inf in {bad}")
        return 1

    fails = 0
    for tag, (lo, hi), mxn, mnn in (("25", io_uphlcy.LAYER_25, "UPHLMAX25", "UPHLMIN25"),
                                    ("03", io_uphlcy.LAYER_03, "UPHLMAX03", "UPHLMIN03")):
        ref_mx, ref_mn, keep = reference(U, V, W_TOT, Z, FIS, DXH, DYH, lo, hi, init_max, init_min)
        got_mx = cp.asnumpy(getattr(st, mxn))
        got_mn = cp.asnumpy(getattr(st, mnn))
        scale = max(float(np.abs(ref_mx[keep]).max()), 1.0)
        dmx = float(np.abs(got_mx[keep] - ref_mx[keep]).max())
        dmn = float(np.abs(got_mn[keep] - ref_mn[keep]).max())
        alt_mx, _, _ = reference(U, V, W, Z, FIS, DXH, DYH, lo, hi, init_max, init_min)
        dalt = float(np.abs(alt_mx[keep] - ref_mx[keep]).max())
        print(f"layer {tag}: max |kernel-ref| UPHLMAX {dmx:.3g}, UPHLMIN {dmn:.3g} (scale {scale:.3g}); "
              f"reference with bare W differs by {dalt:.3g}", flush=True)
        if dmx > 1e-4 * scale or dmn > 1e-4 * scale:
            fails += 1
    if fails:
        print("FAIL: io-uphlcy kernel does not match the CALC_UPHLCY reference on W_TOT")
        return 1
    print("ok: io-uphlcy matches the CALC_UPHLCY reference (W_TOT), no NaN/Inf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
