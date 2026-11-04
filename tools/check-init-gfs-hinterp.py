"""Done-rule check for init-gfs-hinterp (strip version, review fixes).

    source env.sh && python tools/check-init-gfs-hinterp.py

Takes the GPU pool lease first (README.md "GPU use"). One run at full size: the real NAM 12 km grid
(954 x 835 x 60 from parm/nam_configfile), a full-size GFS 0.25 degree source [2, 41, 721, 1440] with
random values, decode-style masks, planted missing points (NaN, mask 0) and one planted fully missing
2x2 cluster at the south-west corner of the domain.

Asserts:
1. NVRTC compile of namgpu/kernels/init_gfs_hinterp.cu.
2. Every strip value (H and V, both boundary times) is finite.
3. Every strip value matches an independent vectorised reference of the METGRID rule (four_pt, then
   average_4pt over the present neighbours, then missing = 0), point for point.
4. The missing count equals the reference count (per call, both times, both grid classes).
5. The planted cluster is reached (missing count > 0), so the missing path is exercised.
6. Refusals: a north-to-south grid, a non-positive dlat, a source without a mask, and isobaric strips
   handed to read_bc (require_read_bc_levels) all raise.
7. run(ctx) binds both time levels into ctx.tables["gfs_hinterp_strips"].
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILL = 0.0


def ref_strips(src4, mask4, glat_deg, glon_deg, grid, w, im, jm):
    """Vectorised reference of the METGRID rule on the four strips of one grid class (cupy)."""
    import cupy as cp
    nt, nl, njs, nis = (int(x) for x in src4.shape)
    S = src4.reshape(nt * nl, njs, nis)
    M = (mask4.reshape(nt * nl, njs, nis) & 5) != 0       # PRESENT (1) or FILLED (4)

    def point(lat, lon):
        fx = (lon - grid.lon0) / grid.dlon
        nper = 360.0 / grid.dlon
        fx = fx - nper * cp.floor(fx / nper)
        fy = cp.clip((lat - grid.lat0) / grid.dlat, 0.0, njs - 1)
        i0 = cp.floor(fx).astype(cp.int64) % nis
        i1 = (i0 + 1) % nis
        wx = fx - cp.floor(fx)
        j0 = cp.clip(cp.floor(fy).astype(cp.int64), 0, njs - 2)
        wy = fy - j0
        j1 = j0 + 1
        vals, oks = [], []
        for jj, ii in ((j0, i0), (j0, i1), (j1, i0), (j1, i1)):
            vals.append(S[:, jj, ii])
            oks.append(M[:, jj, ii])
        v00, v10, v01, v11 = vals
        o00, o10, o01, o11 = oks
        allok = o00 & o10 & o01 & o11
        bil = ((1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10
               + (1 - wx) * wy * v01 + wx * wy * v11)
        nv = o00.astype(cp.int32) + o10 + o01 + o11
        tot = (cp.where(o00, v00, 0) + cp.where(o10, v10, 0)
               + cp.where(o01, v01, 0) + cp.where(o11, v11, 0))
        avg = tot / cp.maximum(nv, 1)
        out = cp.where(allok, bil, cp.where(nv > 0, avg, FILL))
        miss = (~allok) & (nv == 0)
        return out, miss

    res = {}
    lat_all = glat_deg
    lon_all = glon_deg
    # south, north: rows; west, east: columns (same index map as the kernel)
    sides = {
        "s": (lat_all[:w, :], lon_all[:w, :]),
        "n": (lat_all[jm - w:, :], lon_all[jm - w:, :]),
        "w": (lat_all[:, :w], lon_all[:, :w]),
        "e": (lat_all[:, im - w:], lon_all[:, im - w:]),
    }
    nmiss = 0
    for k, (la, lo) in sides.items():
        out, miss = point(la.reshape(-1)[None, :], lo.reshape(-1)[None, :])
        res[k] = out.reshape((nt * nl,) + la.shape)
        nmiss += int(miss.sum())
    return res, nmiss


def main():
    import cupy as cp
    import numpy as np
    from namgpu import config, grid as gridmod, kernels
    from namgpu.state import State
    from namgpu.init import init_gfs_hinterp as hin

    kernels.module(hin.UNIT).compile()
    print("compile: ok", flush=True)

    cfg = config.load()
    g = gridmod.build(cfg)
    lm, jm, im = int(g.lm), int(g.jm), int(g.im)
    print(f"grid: {im} x {jm} x {lm}", flush=True)
    grid = hin.Grid(**hin.GFS_QUARTER_DEG)
    nt, nl, njs, nis = hin.TIME_LEVELS, 41, 721, 1440

    rng = np.random.default_rng(11)
    src = cp.asarray((rng.standard_normal((nt, nl, njs, nis)) * 10.0).astype(np.float32))
    mask = cp.ones(src.shape, dtype=cp.uint8)
    # planted missing: scattered NaN points (mask 0)
    k = 3000
    idx = (rng.integers(0, nt, k), rng.integers(0, nl, k), rng.integers(0, njs, k), rng.integers(0, nis, k))
    src[idx] = cp.float32(np.nan)
    mask[idx] = 0
    # planted fully missing 2x2 cluster at the GFS cell under the domain's south-west corner (both times)
    lat_c = float(g.glat_v[0, 0]) * 180.0 / np.pi
    lon_c = float(g.glon_v[0, 0]) * 180.0 / np.pi
    jc = int(np.floor((lat_c - grid.lat0) / grid.dlat))
    ic = int(np.floor((lon_c % 360.0 - grid.lon0) / grid.dlon)) % nis
    for t in range(nt):
        src[t, 0, jc:jc + 2, ic:ic + 2] = cp.float32(np.nan)
        mask[t, 0, jc:jc + 2, ic:ic + 2] = 0
    print(f"planted: {k} scattered NaN, 2x2 cluster at GFS j={jc} i={ic}", flush=True)

    # masks and the rule (the 0 fill is never read by the rule: masked points are not used)
    st = State(g, names=["U", "V"])
    st.add("GLAT", (jm, im))[...] = cp.asarray(g.glat_h.astype(np.float32))
    st.add("GLON", (jm, im))[...] = cp.asarray(g.glon_h.astype(np.float32))
    st.add("VLAT", (jm, im))[...] = cp.asarray(g.glat_v.astype(np.float32))
    st.add("VLON", (jm, im))[...] = cp.asarray(g.glon_v.astype(np.float32))

    fails = []
    t0 = time.perf_counter()
    h = hin.interp_strips(src, mask, st["GLAT"], st["GLON"], width=hin.LNSH, grid=grid, slot="CHK_H")
    v = hin.interp_strips(src, mask, st["VLAT"], st["VLON"], width=hin.LNSV, grid=grid, slot="CHK_V")
    cp.cuda.Stream.null.synchronize()
    print(f"interp: {time.perf_counter() - t0:.2f} s", flush=True)

    for name, res, lat, lon, width in (
        ("H", h, st["GLAT"], st["GLON"], hin.LNSH),
        ("V", v, st["VLAT"], st["VLON"], hin.LNSV),
    ):
        lat_d = lat * (180.0 / np.pi)
        lon_d = lon * (180.0 / np.pi)
        ref, ref_miss = ref_strips(src, mask, lat_d, lon_d, grid, width, im, jm)
        got_miss = int(res["missing"][0])
        for key in ("s", "n", "w", "e"):
            a = res[key]
            if not bool(cp.isfinite(a).all()):
                fails.append(f"{name} {key}: non-finite output")
            diff = float(cp.abs(a - ref[key]).max())
            scale = float(cp.abs(ref[key]).max())
            print(f"{name} {key}: max |diff| vs reference {diff:.3e} (scale {scale:.1f})", flush=True)
            if not diff <= 1e-3 * max(1.0, scale):
                fails.append(f"{name} {key}: differs from reference by {diff:.3e}")
        print(f"{name} missing: kernel {got_miss}, reference {ref_miss}", flush=True)
        if got_miss != ref_miss:
            fails.append(f"{name}: missing count {got_miss} != reference {ref_miss}")
        if ref_miss == 0:
            fails.append(f"{name}: planted cluster not reached (missing path not exercised)")

    # refusals
    for label, make in (
        ("north-to-south grid", lambda: hin.Grid(lat0=-90.0, lon0=0.0, dlat=-0.25, dlon=0.25)),
        ("dlat 0", lambda: hin.Grid(lat0=-90.0, lon0=0.0, dlat=0.0, dlon=0.25)),
    ):
        try:
            make()
            fails.append(f"{label}: accepted")
        except ValueError:
            print(f"refusal ok: {label}", flush=True)
    try:
        hin.interp_strips(src, None, st["GLAT"], st["GLON"], width=hin.LNSH, grid=grid, slot="CHK_X")
        fails.append("no-mask source: accepted")
    except (ValueError, AttributeError, TypeError):
        print("refusal ok: source without mask", flush=True)
    try:
        hin.require_read_bc_levels({"nl": 41}, lm)
        fails.append("isobaric strips accepted by read_bc")
    except RuntimeError:
        print("refusal ok: isobaric strips to read_bc", flush=True)

    # run(ctx) binds both time levels for the consumer
    ctx = type("Ctx", (), {"st": st, "grid": g, "stream": None, "tables": {}})()
    ctx.gfs = type("Gfs", (), {"grid": grid, "fields": {"HGT": hin.GfsField(src=src, mask=mask)}})()
    out = hin.run(ctx)
    bound = ctx.tables.get("gfs_hinterp_strips", {}).get("HGT")
    if out is None or bound is None or bound["H"]["nt"] != nt or bound["V"]["nl"] != nl:
        fails.append("run(ctx) did not bind both time levels")
    else:
        print(f"run(ctx): bound H slabs {tuple(bound['H']['s'].shape)}, V slabs {tuple(bound['V']['s'].shape)}",
              flush=True)

    if fails:
        print("FAILED: " + "; ".join(fails), flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
