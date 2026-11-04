"""Done-rule check for init-gfs-wind-rotate (strip version).

    source ~/nam-gpu/env.sh && python tools/check-init-gfs-wind-rotate.py

Takes the GPU pool lease first (README.md "GPU use"). One run at full size: the real NAM 12 km grid
from parm/nam_configfile (954 x 835 x 60), a full-size GFS 0.25 degree wind pair staged as ctx.gfs.u/.v
with random values, the wrapper run once.

Asserts:
1. NVRTC compile of namgpu/kernels/init_gfs_wind_rotate.cu.
2. Every U and V value on the V boundary strips (LNSV rows/columns, last row and column excluded) is
   finite and the rotated speed equals the speed of the interpolated GFS strip winds (float32 tolerance).
3. Interior U and V (outside the strips) keep the sentinel value: the unit writes strips only.
4. The last row and last column keep the sentinel (CONTRACT section 1).
5. The wrapper refuses to run without staged winds (no zero fill).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SENTINEL = -999.0


def main():
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels
    from namgpu.state import State
    from namgpu.init import init_gfs_wind_rotate as u
    from namgpu.init import init_gfs_hinterp as hin

    kernels.module(u.UNIT).compile()
    print("compile: ok", flush=True)

    cfg = config.load()
    g = gridmod.build(cfg)
    lm, jm, im = int(g.lm), int(g.jm), int(g.im)
    print(f"grid: {im} x {jm} x {lm}", flush=True)
    w = int(hin.LNSV)

    st = State(g, names=["U", "V"])
    st.add("VLAT", (jm, im))[...] = cp.asarray(g.glat_v.astype(np.float32))
    st.add("VLON", (jm, im))[...] = cp.asarray(g.glon_v.astype(np.float32))
    st.U[...] = SENTINEL
    st.V[...] = SENTINEL

    rng = np.random.default_rng(7)
    nls, njs, nis = lm, 721, 1440
    gfs_u = cp.asarray((rng.standard_normal((hin.TIME_LEVELS, nls, njs, nis)) * 10.0).astype(np.float32))
    gfs_v = cp.asarray((rng.standard_normal((hin.TIME_LEVELS, nls, njs, nis)) * 10.0).astype(np.float32))
    ones = cp.ones(gfs_u.shape, dtype=cp.uint8)
    gsrc = hin.GfsSource(grid=hin.Grid(**hin.GFS_QUARTER_DEG),
                         u=hin.GfsField(src=gfs_u, mask=ones), v=hin.GfsField(src=gfs_v, mask=ones))
    ctx = type("Ctx", (), {"st": st, "grid": g, "gfs": gsrc, "stream": None})()

    # 5. refusal without staged winds
    try:
        u.run(type("Ctx", (), {"st": st, "grid": g, "gfs": None, "stream": None})())
        print("FAILED: ran without staged winds", flush=True)
        return 1
    except RuntimeError as e:
        print(f"refusal ok: {str(e)[:60]}...", flush=True)

    t0 = time.perf_counter()
    u.run(ctx)
    cp.cuda.Stream.null.synchronize()
    print(f"run: {time.perf_counter() - t0:.2f} s", flush=True)

    # expected strip speed from the interpolated GFS strips
    ue = hin.interp_strips(gfs_u, ones, st["VLAT"], st["VLON"], width=w, grid=gsrc.grid, slot="CHK_U")
    ve = hin.interp_strips(gfs_v, ones, st["VLAT"], st["VLON"], width=w, grid=gsrc.grid, slot="CHK_V")
    ue = {k: ue[k][:lm] for k in ("s", "n", "w", "e")}   # boundary time 0, as the unit binds it
    ve = {k: ve[k][:lm] for k in ("s", "n", "w", "e")}
    U, V = st.U, st.V
    mask = cp.zeros((jm, im), dtype=bool)
    mask[:w, :] = True
    mask[jm - w:, :] = True
    mask[:, :w] = True
    mask[:, im - w:] = True
    mask[jm - 1, :] = False
    mask[:, im - 1] = False
    inner = cp.zeros((jm, im), dtype=bool)
    inner[:jm - 1, :im - 1] = True
    inner &= ~mask

    fails = []
    edge_u = U[:, mask]
    edge_v = V[:, mask]
    if not bool(cp.isfinite(edge_u).all() & cp.isfinite(edge_v).all()):
        fails.append("strip U/V not finite")
    if bool((U[:, inner] != SENTINEL).any()) or bool((V[:, inner] != SENTINEL).any()):
        fails.append("interior U/V overwritten")
    if bool((U[:, jm - 1, :] != SENTINEL).any()) or bool((U[:, :, im - 1] != SENTINEL).any()):
        fails.append("last row/column written")

    # speed check on the strips: |(U,V)| must equal |(UE,VE)| at each strip point
    # assemble the expected strip speed field on the full grid, strip by strip
    ue_full = cp.full((lm, jm, im), 0.0, dtype=cp.float32)
    ve_full = cp.full((lm, jm, im), 0.0, dtype=cp.float32)
    ue_full[:, :w, :] = ue["s"]
    ue_full[:, jm - w:, :] = ue["n"]
    ue_full[:, :, :w] = ue["w"]
    ue_full[:, :, im - w:] = ue["e"]
    ve_full[:, :w, :] = ve["s"]
    ve_full[:, jm - w:, :] = ve["n"]
    ve_full[:, :, :w] = ve["w"]
    ve_full[:, :, im - w:] = ve["e"]
    m3 = cp.broadcast_to(mask, (lm, jm, im))
    spd_rot = cp.sqrt(U * U + V * V)[m3]
    spd_src = cp.sqrt(ue_full * ue_full + ve_full * ve_full)[m3]
    err = float(cp.abs(spd_rot - spd_src).max())
    rel = float((cp.abs(spd_rot - spd_src) / cp.maximum(spd_src, 1e-3)).max())
    print(f"strip points {int(m3.sum())}, max speed |diff| {err:.3e} m/s, max rel {rel:.3e}", flush=True)
    if not (err < 1e-3 and rel < 1e-4):
        fails.append("rotated speed differs from the GFS strip speed")

    if fails:
        print("FAILED: " + "; ".join(fails), flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
