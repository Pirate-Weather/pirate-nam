"""Done-rule check for bnd-bocoh-blend (CONTRACT.md section 6, parts 1 and 2), plus a value check
against the Fortran it ports (module_FLTBNDS.F90:5355-5962).

    source env.sh
    python tools/check-bnd-bocoh-blend.py

Builds the same full-size random State (954 x 835 x 60), adds random ctx.bc value arrays in the
bnd-boundary-init layout ("<SIDE>_<VAR>", leading slot axis of 2), copies the pre-blend H fields,
runs the wrapper ONCE, then:
  1. checks every float array for NaN/Inf (done-rule part 2);
  2. compares PD, T, Q, CW and PINT with a full-size numpy transcription of the Fortran blend
     (S, N, W, E strips in Fortran order, then the four corners, then the outer PINT rows).
     Float32 with fused multiply-add on the GPU, so the comparison is at a relative tolerance,
     not bitwise.
Compiles the kernel file first. Take the GPU pool lease first (README.md).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RTOL = 1e-5


def fortran_reference(pre, bc, dsg2, pdsg1, pt, im, jm, lm, lnsh):
    """Numpy transcription of module_FLTBNDS.F90:5355-5962 for one domain (1-based Fortran
    indices converted to 0-based here). pre holds PD [JM,IM], T/Q/CW/PINT arrays; returns new arrays."""
    wh = np.zeros(lnsh + 1, dtype=np.float32)          # wh[k], k = 1..lnsh
    for k in range(1, lnsh + 1):
        wh[k] = np.float32(1.0) - (np.float32(0.9) / np.float32(lnsh - 1)) * np.float32(k - 1)
    pd = pre["PD"].copy()
    f3 = {n: pre[n].copy() for n in ("T", "Q", "CW")}
    pint = pre["PINT"].copy()

    def blend_pint(pidx, w_arr_idx, cell_j, cell_i):
        # PINT recursion on the selected cells (1-D index arrays), per level, with weight w.
        w = wh[w_arr_idx]
        wc = np.float32(1.0) - w
        pdc = pd[cell_j, cell_i]
        pint[0, cell_j, cell_i] = pt
        for l in range(1, lm + 1):
            pl = pint[l - 1, cell_j, cell_i]
            pl1 = pint[l, cell_j, cell_i]
            pint[l, cell_j, cell_i] = (pl + (dsg2[l - 1] * pdc + pdsg1[l - 1])) * w + pl1 * wc

    # S boundary (Fortran 5364-5452): jb = j = 1..lnsh, i in [jb, IM+1-jb].
    for jb in range(1, lnsh + 1):
        i = np.arange(jb, im + 1 - jb + 1) - 1          # 0-based i
        j = jb - 1
        w = wh[jb]; wc = np.float32(1.0) - w
        pd[j, i] = bc["S_PD"][0][jb - 1, i] * w + pd[j, i] * wc
        for n, name in (("T", "T"), ("Q", "Q"), ("CW", "CW")):
            f3[name][:, j, i] = bc[f"S_{n}"][0][:, jb - 1, i] * w + f3[name][:, j, i] * wc
        blend_pint(None, jb, np.full(i.shape, j), i)

    # N boundary (Fortran 5453-5541): j = JM-lnsh+jb, i in [1-jb+lnsh, IM+jb-lnsh], k = lnsh+1-jb.
    for jb in range(1, lnsh + 1):
        i = np.arange(1 - jb + lnsh, im + jb - lnsh + 1) - 1
        j = jm - lnsh + jb - 1
        k = lnsh + 1 - jb
        w = wh[k]; wc = np.float32(1.0) - w
        pd[j, i] = bc["N_PD"][0][jb - 1, i] * w + pd[j, i] * wc
        for name in ("T", "Q", "CW"):
            f3[name][:, j, i] = bc[f"N_{name}"][0][:, jb - 1, i] * w + f3[name][:, j, i] * wc
        blend_pint(None, k, np.full(i.shape, j), i)

    # W boundary (Fortran 5542-5630): i = ib = jb, j in [1+ib, JM-ib], k = ib.
    for ib in range(1, lnsh + 1):
        j = np.arange(1 + ib, jm - ib + 1) - 1
        i = ib - 1
        w = wh[ib]; wc = np.float32(1.0) - w
        pd[j, i] = bc["W_PD"][0][j, ib - 1] * w + pd[j, i] * wc
        for name in ("T", "Q", "CW"):
            f3[name][:, j, i] = bc[f"W_{name}"][0][:, j, ib - 1] * w + f3[name][:, j, i] * wc
        blend_pint(None, ib, j, np.full(j.shape, i))

    # E boundary (Fortran 5631-5720): i = IM-lnsh+jb, j in [2+lnsh-jb, JM-lnsh-1+jb], k = lnsh+1-jb.
    for jb in range(1, lnsh + 1):
        i = im - lnsh + jb - 1
        j = np.arange(2 + lnsh - jb, jm - lnsh - 1 + jb + 1) - 1
        k = lnsh + 1 - jb
        w = wh[k]; wc = np.float32(1.0) - w
        pd[j, i] = bc["E_PD"][0][j, jb - 1] * w + pd[j, i] * wc
        for name in ("T", "Q", "CW"):
            f3[name][:, j, i] = bc[f"E_{name}"][0][:, j, jb - 1] * w + f3[name][:, j, i] * wc
        blend_pint(None, k, j, np.full(j.shape, i))

    # Corners (Fortran 5721-5897): average of the four neighbours, 0.25 weight, order as written.
    def corner(x, y):
        def avg(a):
            s = ((a[y - 1, x] + a[y, x - 1]) + a[y, x + 1]) + a[y + 1, x]
            return s * np.float32(0.25)
        pd[y, x] = avg(pd)
        for name in ("T", "Q", "CW"):
            a = f3[name]
            a[:, y, x] = ((a[:, y - 1, x] + a[:, y, x - 1]) + a[:, y, x + 1]) + a[:, y + 1, x]
            a[:, y, x] = a[:, y, x] * np.float32(0.25)

    corner(1, 1); corner(im - 2, 1); corner(1, jm - 2); corner(im - 2, jm - 2)

    # Outer PINT rows and columns (Fortran 5898-5961): no blend weight, after the corners.
    rows = [0, 1, jm - 2, jm - 1]
    cols = [0, 1, im - 2, im - 1]
    edge_mask = np.zeros((jm, im), dtype=bool)
    edge_mask[rows, :] = True
    edge_mask[:, cols] = True
    jj, ii = np.nonzero(edge_mask)
    pdc = pd[jj, ii]
    pint[0, jj, ii] = pt
    for l in range(1, lm + 1):
        pint[l, jj, ii] = pint[l - 1, jj, ii] + (dsg2[l - 1] * pdc + pdsg1[l - 1])

    out = {"PD": pd, "PINT": pint}
    out.update(f3)
    return out


def main():
    import cupy as cp
    import os
    from namgpu import config, grid as gridmod, kernels, state as statemod
    # Optional: a directory holding a clean nmmb_common.cuh (the shared header in the tree is mid-edit
    # by another lane). It goes ahead of the kernels dir on the include path.
    hdr = os.environ.get("NAMGPU_HDR_DIR")
    if hdr:
        kernels.OPTIONS = (kernels.OPTIONS[0], kernels.OPTIONS[1], f"-I{hdr}") + tuple(kernels.OPTIONS[2:])
    from namgpu.driver import Ctx
    from namgpu.boundary import bnd_bocoh_blend as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g).randomize(seed=1)
    im, jm, lm, lnsh = int(g.im), int(g.jm), int(g.lm), int(cfg.lnsh)
    rng = cp.random.default_rng(2)

    # Boundary layout of bnd-boundary-init: "<SIDE>_<VAR>", leading slot axis of 2, value in slot 0.
    bc = {}
    for side in ("S", "N", "W", "E"):
        s2 = (lnsh, im) if side in ("S", "N") else (jm, lnsh)
        s3 = (lm, lnsh, im) if side in ("S", "N") else (lm, jm, lnsh)
        bc[f"{side}_PD"] = (50000.0 + 50000.0 * rng.random((2,) + s2, dtype=cp.float32))
        for v in ("T", "Q", "CW"):
            scale = 0.02 if v in ("Q", "CW") else 1.0
            bc[f"{side}_{v}"] = scale * rng.random((2,) + s3, dtype=cp.float32)
        bc[f"{side}_T"] = 200.0 + 110.0 * rng.random((2,) + s3, dtype=cp.float32)

    # Pre-blend copies (host) for the Fortran reference.
    names = ("PD", "T", "Q", "CW", "PINT")
    pre = {n: cp.asnumpy(st[n]).copy() for n in names}
    for n in ("PD",):
        pre[n] = pre[n].reshape(jm, im)
    for n in ("T", "Q", "CW"):
        pre[n] = pre[n].reshape(lm, jm, im)
    pre["PINT"] = pre["PINT"].reshape(lm + 1, jm, im)
    bc_host = {k: cp.asnumpy(v) for k, v in bc.items()}
    dsg2 = cp.asnumpy(st["DSG2"]).astype(np.float32).reshape(-1)[:lm]
    pdsg1 = np.asarray(g.pdsg1, dtype=np.float32).reshape(-1)[:lm]
    pt = np.float32(g.pt)

    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt, bc=bc)
    t0 = time.perf_counter()
    kernels.module(unit.UNIT).compile()
    with ctx.stream:
        unit.run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0

    bad = [n for n, a in st.fields.items()
           if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
    if bad:
        print(f"{unit.UNIT}: NaN/Inf in {bad[:10]} ({dt:.2f} s)")
        return 1

    ref = fortran_reference(pre, bc_host, dsg2, pdsg1, pt, im, jm, lm, lnsh)
    fails = []
    for n in ("PD", "T", "Q", "CW", "PINT"):
        got = cp.asnumpy(st[n]).reshape(ref[n].shape)
        want = ref[n]
        err = np.abs(got - want)
        scale = np.maximum(np.abs(want), np.float32(1.0))
        worst = float((err / scale).max())
        ndiff = int((err > RTOL * scale).sum())
        print(f"  {n}: max rel err {worst:.3g}, cells over rtol {ndiff} of {got.size}")
        if ndiff:
            fails.append(n)
    if fails:
        print(f"{unit.UNIT}: value check FAILED for {fails} ({dt:.2f} s)")
        return 1
    print(f"{unit.UNIT}: ok, compiled, {im}x{jm}x{lm} random state, no NaN/Inf, matches the "
          f"Fortran transcription at rtol {RTOL}, {dt:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
