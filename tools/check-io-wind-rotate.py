"""Check for io-wind-rotate (review gaps: angle, operation order, sign, notice).

    source ~/nam-gpu/env.sh && python tools/check-io-wind-rotate.py

Run under the GPU pool lease (README.md "GPU use"). Full-size NAM 12 km grid from parm/nam_configfile.

Asserts:
1. NVRTC compile of namgpu/kernels/io_wind_rotate.cu.
2. Round trip against init-gfs-wind-rotate's map (sign and inverse): a constant earth-relative wind
   (10 m/s east, then 10 m/s north) is forward-mapped to the V points with init-gfs-wind-rotate's
   coefficients, rotated back at every H point (3-D winds and 10 m winds), and must return the same earth
   wind (float32 tolerance).
3. Closed-form cross-check of the angle (host): the post's SINALP = SINPH0*sin(TLON)/cos(STNLAT) with
   DLM = STNLON - TLM0 and TLON = atan2(YY, XX) equals sin(theta) of the grid at every V point with
   lat < 85 N (within 1e-5).
Reports (evidence for the rotate-then-average decision): the V cells where the grid-i direction jumps by more
than 0.2 rad between neighbours, their latitude range, the domain max latitude, and the fraction of V points
with cos(theta) < 0 (where the post's COSALP = COS(ASIN) cannot be right).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels
    from namgpu.state import State
    from namgpu.init import init_gfs_wind_rotate as iw
    from namgpu.io import io_wind_rotate as wr

    kernels.module(wr.UNIT).compile()
    print("compile: ok", flush=True)

    cfg = config.load()
    g = gridmod.build(cfg)
    lm, jm, im = int(g.lm), int(g.jm), int(g.im)
    print(f"grid: {im} x {jm} x {lm}", flush=True)

    st = State(g, names=["U", "V"])
    st.add("U10", (jm, im))
    st.add("V10", (jm, im))
    ctx = type("Ctx", (), {"st": st, "grid": g, "gfs": None, "stream": None})()

    # 2. round trip: forward map (init unit, V points) then inverse (this unit, H points)
    ce_e, ce_n, cn_e, cn_n = (np.asarray(a.get()).astype(np.float64) for a in iw._coefficients(ctx))
    for name, ue0, ve0 in (("east", 10.0, 0.0), ("north", 0.0, 10.0)):
        u_v = ue0 * ce_e + ve0 * ce_n
        v_v = ue0 * cn_e + ve0 * cn_n
        st.U[...] = cp.asarray(np.broadcast_to(u_v, (lm, jm, im)).astype(np.float32))
        st.V[...] = cp.asarray(np.broadcast_to(v_v, (lm, jm, im)).astype(np.float32))
        st.U10[...] = cp.asarray(u_v.astype(np.float32))
        st.V10[...] = cp.asarray(v_v.astype(np.float32))
        wr.run(ctx)
        cp.cuda.Stream.null.synchronize()
        ue = np.asarray(st.UEARTH.get()).astype(np.float64)
        ve = np.asarray(st.VEARTH.get()).astype(np.float64)
        ue10 = np.asarray(st.UE10.get()).astype(np.float64)
        ve10 = np.asarray(st.VE10.get()).astype(np.float64)
        e3 = max(np.abs(ue - ue0).max(), np.abs(ve - ve0).max())
        e10 = max(np.abs(ue10 - ue0).max(), np.abs(ve10 - ve0).max())
        print(f"round trip {name} 10 m/s: 3-D max err {e3:.3e} m/s, 10 m max err {e10:.3e} m/s", flush=True)
        assert e3 < 0.05 and e10 < 0.05, f"round trip {name} failed: sign or inverse wrong"

    # 3. closed-form cross-check of the angle at V points (host)
    jj = np.arange(jm)[:, None]
    ii = np.arange(im)[None, :]
    tlm_v = np.broadcast_to(g.wb + ii * g.dlm + 0.5 * g.dlm, (jm, im)).astype(np.float64)
    tph_v = np.broadcast_to(g.sb + jj * g.dph + 0.5 * g.dph, (jm, im)).astype(np.float64)
    glon, glat = gridmod.rotated_to_geographic(tlm_v, tph_v, g.tlm0, g.tph0)
    sp, cp0 = np.sin(g.tph0), np.cos(g.tph0)
    dlm = glon - g.tlm0
    xx = cp0 * np.cos(glat) * np.cos(dlm) + sp * np.sin(glat)
    yy = -np.cos(glat) * np.sin(dlm)
    s_closed = sp * np.sin(np.arctan2(yy, xx)) / np.cos(glat)
    lat_deg = np.degrees(glat)
    ok = lat_deg < 85.0
    d_sin = np.abs(s_closed - ce_n)[ok].max()
    print(f"closed-form sine vs grid sine (V points, lat < 85): max diff {d_sin:.3e}", flush=True)
    assert d_sin < 1e-5, "angle sine does not match the closed form"

    # evidence for the rotate-then-average decision
    ang = np.arctan2(ce_n, ce_e)
    jx = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=1)))) > 0.2
    jy = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=0)))) > 0.2
    njump = int(jx.sum() + jy.sum())
    print(f"domain max lat {lat_deg.max():.2f} N; V cells with a > 0.2 rad grid-i angle jump: {njump}", flush=True)
    if njump:
        print(f"  jump cells at lat {lat_deg[:, :-1][jx].min():.2f} to {lat_deg[:, :-1][jx].max():.2f} N", flush=True)
    print(f"V points with cos(theta) < 0: {(ce_e < 0).mean() * 100:.2f} %", flush=True)
    print("PASS io-wind-rotate", flush=True)


if __name__ == "__main__":
    main()
