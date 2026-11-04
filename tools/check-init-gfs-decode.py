"""Done-rule check for init-gfs-decode (CONTRACT.md section 6).

    source ~/nam-gpu/env.sh && python tools/check-init-gfs-decode.py

1. NVRTC-compiles namgpu/kernels/init_gfs_decode.cu (via the kernel loader).
2. Reads the isobaric level set from the real GFS 0.25 deg f006 inventory (levels_from_inventory).
3. Builds a FULL-SIZE input (10 x nlev x 721 x 1440 and 2 x 721 x 1440) with planted NaN / Inf /
   9.999e20 fills at known indices, and negative species. The planted masks are the truth.
4. Runs the wrapper ONCE and checks against the planted truth, not against the kernel's own test:
   - every output finite;
   - PRESENT bit equals the complement of the planted fills, exactly, for pl and sfc;
   - every present value unchanged (species clamped to >= 0 only);
   - every planted point with a present 8-neighbour equals the mean of those neighbours (recomputed
     here with shifted arrays, a different computation), bit FILLED set; the others stay 0 and unset;
   - BELOW bit exactly where plev > surface pressure (surface known); PS_UNKNOWN exactly where it is not.
Takes the GPU pool lease first (README.md). Full size, one run, no column test.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GRIB = Path(os.environ.get("NAMGPU_GFS_F006",
                           str(Path.home() / "nam-gpu/cases/2025-03-14_12z/raw/gfs/gfs.t06z.f006.grib2")))
FILL = 9.999e20


def planted_input(u, plev, nlat, nlon, seed=1):
    import cupy as cp
    rng = cp.random.default_rng(seed)
    nv, nlev = len(u.PL_VARS), len(plev)
    pl = rng.random((nv, nlev, nlat, nlon), dtype=cp.float32)
    pl[0] *= 40000.0                          # HGT (gpm)
    pl[1] = 180.0 + 140.0 * pl[1]             # TMP (K)
    pl[2] *= 0.02                             # SPFH
    pl[3] *= 1.2                              # RH (%)
    pl[4:6] = 120.0 * pl[4:6] - 60.0          # UGRD, VGRD
    pl[6:] = 2.0e-3 * pl[6:] - 5.0e-4         # condensate, some negative (roundoff)
    planted_pl = cp.zeros(pl.shape, dtype=bool)
    for v, val, sl in ((1, cp.nan, (slice(None), slice(None, None, 97), slice(None, None, 101))),
                       (4, cp.inf, (slice(None), slice(None, None, 89), slice(None, None, 103))),
                       (0, FILL, (slice(None), slice(None, None, 113), slice(None, None, 107))),
                       (3, FILL, (slice(None), slice(5, None, 211), slice(9, None, 199)))):
        pl[(v,) + sl] = val
        planted_pl[(v,) + sl] = True
    sfc = cp.empty((len(u.SFC_VARS), nlat, nlon), dtype=cp.float32)
    sfc[0] = 50000.0 + 52000.0 * rng.random((nlat, nlon), dtype=cp.float32)   # PRES (Pa)
    sfc[1] = 4000.0 * rng.random((nlat, nlon), dtype=cp.float32)             # HGT (gpm)
    planted_sfc = cp.zeros(sfc.shape, dtype=bool)
    sfc[0, ::127, ::131] = cp.nan
    planted_sfc[0, ::127, ::131] = True
    sfc[1, ::109, ::113] = FILL
    planted_sfc[1, ::109, ::113] = True
    sfc[0, 300:306, 700:712] = FILL           # a small patch of surface-pressure fills
    planted_sfc[0, 300:306, 700:712] = True
    return pl, sfc, planted_pl, planted_sfc


def neighbour_mean(vals, present):
    """Mean of the present 8-neighbours (lon periodic, lat edge-clipped) over the last two axes.
    Returns (mean, count). Computed with shifted arrays, independent of the kernel's loop."""
    import cupy as cp
    s = cp.zeros(vals.shape, dtype=cp.float32)
    c = cp.zeros(vals.shape, dtype=cp.float32)
    pv = cp.where(present, vals, 0.0).astype(cp.float32)
    pc = present.astype(cp.float32)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            rv = cp.roll(pv, -di, axis=-1)
            rc = cp.roll(pc, -di, axis=-1)
            if dj == 1:
                rv2 = cp.zeros_like(rv); rc2 = cp.zeros_like(rc)
                rv2[..., :-1, :] = rv[..., 1:, :]; rc2[..., :-1, :] = rc[..., 1:, :]
            elif dj == -1:
                rv2 = cp.zeros_like(rv); rc2 = cp.zeros_like(rc)
                rv2[..., 1:, :] = rv[..., :-1, :]; rc2[..., 1:, :] = rc[..., :-1, :]
            else:
                rv2, rc2 = rv, rc
            s += rv2
            c += rc2
    mean = cp.where(c > 0, s / cp.maximum(c, 1.0), 0.0)
    return mean, c


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.init import init_gfs_decode as u
    from namgpu.init import init_grib2_decode as g

    kernels.module(u.KSTEM).compile()
    print("compile: ok", flush=True)

    plev = u.levels_from_inventory(g.inventory(str(GRIB)))
    nlev, nlat, nlon = len(plev), u.NLAT_FULL, u.NLON_FULL
    print(f"levels from {GRIB.name}: {nlev} ({plev[0]:.0f} .. {plev[-1]:.0f} Pa)", flush=True)

    pl, sfc, planted_pl, planted_sfc = planted_input(u, plev, nlat, nlon)
    pl_in = pl.copy()                          # decode is in place; keep the input for the checks
    sfc_in = sfc.copy()
    print(f"planted fills: {int(planted_pl.sum())} pl / {int(planted_sfc.sum())} sfc", flush=True)

    t0 = time.perf_counter()
    raw = {"pl": pl, "plev": plev, "sfc": sfc, "valid": "check"}
    out = u.run(None, raw=raw)
    cp.cuda.Stream.null.synchronize()
    print(f"run: {time.perf_counter() - t0:.2f} s, pl {out.pl.shape}, sfc {out.sfc.shape}", flush=True)

    fails = []
    if not bool(cp.isfinite(out.pl).all()) or not bool(cp.isfinite(out.sfc).all()):
        fails.append("outputs not finite")

    present_pl = (out.pl_mask & u.MASK_PRESENT) != 0
    present_sfc = (out.sfc_mask & u.MASK_PRESENT) != 0
    if not bool(cp.array_equal(present_pl, ~planted_pl)):
        fails.append("pl PRESENT bit differs from the planted fills")
    if not bool(cp.array_equal(present_sfc, ~planted_sfc)):
        fails.append("sfc PRESENT bit differs from the planted fills")

    # present values unchanged (species and RH clamped at zero only)
    nonneg = u.NONNEG
    changed = 0
    for v in range(len(u.PL_VARS)):
        exp = pl_in[v]
        if nonneg[v]:
            exp = cp.maximum(exp, 0.0)
        bad = (~planted_pl[v]) & (out.pl[v] != exp)
        changed += int(bad.sum())
    for v in range(len(u.SFC_VARS)):
        bad = (~planted_sfc[v]) & (out.sfc[v] != sfc_in[v])
        changed += int(bad.sum())
    print(f"present values changed: {changed}", flush=True)
    if changed:
        fails.append(f"{changed} present values changed")

    # planted points: FILLED bit and value equal the independent neighbour mean
    def check_fill(name, planted, mask, new, vals_in_clamped, present_bit_ok):
        nonlocal_fails = []
        filled_count = 0
        unfilled = 0
        worst = 0.0
        for s in range(new.shape[0]):
            pres = ~planted[s]
            mean, cnt = neighbour_mean(vals_in_clamped[s], pres)
            want = planted[s] & (cnt > 0)
            got_filled = (mask[s] & u.MASK_FILLED) != 0
            if not bool(cp.array_equal(got_filled, want)):
                nonlocal_fails.append(f"{name} slab {s}: FILLED bit set on the wrong points")
                break
            diff = cp.abs(new[s].astype(cp.float64) - mean)
            scale = cp.maximum(cp.abs(mean), 1.0)
            rel = cp.where(want, diff / scale, 0.0)
            worst = max(worst, float(rel.max()))
            zero_unfilled = planted[s] & (cnt == 0)
            if int(cp.abs(new[s][zero_unfilled]).sum()) != 0:
                nonlocal_fails.append(f"{name} slab {s}: unfilled point not zero")
                break
            filled_count += int(want.sum())
            unfilled += int(zero_unfilled.sum())
        if worst > 1e-5:
            nonlocal_fails.append(f"{name} fill differs from neighbour mean (max rel {worst:.2e})")
        return nonlocal_fails, filled_count, unfilled, worst

    clamped_pl = pl_in
    for v in range(len(u.PL_VARS)):
        if nonneg[v]:
            clamped_pl[v] = cp.maximum(clamped_pl[v], 0.0)
    f_pl, nf_pl, un_pl, w_pl = [], 0, 0, 0.0
    for v in range(len(u.PL_VARS)):
        f, n, un, w = check_fill(f"pl[{u.PL_VARS[v]}]", planted_pl[v:v + 1].reshape((1,) + planted_pl.shape[1:]),
                                 out.pl_mask[v:v + 1], out.pl[v:v + 1], clamped_pl[v:v + 1], True)
        f_pl += f; nf_pl += n; un_pl += un; w_pl = max(w_pl, w)
    f_sfc, nf_sfc, un_sfc, w_sfc = check_fill("sfc", planted_sfc, out.sfc_mask, out.sfc, sfc_in, True)
    fails += f_pl + f_sfc
    print(f"filled from neighbours: {nf_pl} pl / {nf_sfc} sfc (max rel err {w_pl:.1e} / {w_sfc:.1e}); "
          f"no present neighbour, left missing: {un_pl} pl / {un_sfc} sfc", flush=True)

    # surface and below-ground flags
    ps_known = (out.sfc_mask[0] & (u.MASK_PRESENT | u.MASK_FILLED)) != 0
    got_unknown = ((out.pl_mask & u.MASK_PS_UNKNOWN) != 0)
    if not bool(cp.array_equal(got_unknown, cp.broadcast_to(~ps_known[None, None], got_unknown.shape))):
        fails.append("PS_UNKNOWN bit disagrees with the surface pressure mask")
    got_below = (out.pl_mask & u.MASK_BELOW) != 0
    want_below = (plev_dev(cp, plev)[None, :, None, None] > out.sfc[0][None, None]) & ps_known[None, None]
    want_below = cp.broadcast_to(want_below, got_below.shape)
    n_below = int(got_below.sum())
    print(f"below-ground points flagged: {n_below}; surface unknown at {int((~ps_known).sum())} columns",
          flush=True)
    if n_below == 0:
        fails.append("no below-ground flags")
    if not bool(cp.array_equal(got_below, want_below)):
        fails.append("below-ground bit disagrees with plev > psfc (surface known)")

    if fails:
        print("FAILED: " + "; ".join(fails), flush=True)
        return 1
    print("init-gfs-decode: ok (compiled; full-size run; present values unchanged; fills equal the "
          "neighbour mean; masks consistent with the planted truth)", flush=True)
    return 0


def plev_dev(cp, plev):
    return cp.asarray(plev, dtype=cp.float32)


if __name__ == "__main__":
    sys.exit(main())
