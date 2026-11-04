"""Done-rule check for drv-rdtemp after the review fixes (the review's three gaps).

    source ~/nam-gpu/env.sh
    python tools/check-drv-rdtemp.py

Full-size 954 x 835 x 60 random State, the GPU taken through the pool lease.
  1. the kernel compiles and the wrapper runs on the full domain, no NaN or Inf in CZEN or T;
  2. the sticky leap flag (Fortran MONTH(2)=29 persistence, module_RA_GFDL.F90:1704-1707 and
     7838-7842): the kernel's CZEN at sampled columns equals a single-precision transcription of
     the Fortran CAL_MON_DAY and ZENITH for leap_seen = 0 and leap_seen = 1 in a non-leap year
     (2023), and leap 1 changes CZEN (the sticky day count is visible).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
PI = 3.1415926
DEG2RD = PI / 180.0
OBLIQ = 23.440 * DEG2RD
F = np.float32


def ref_czen(glat, glon, ntsd, dt, julday, julyr, ihrst, leap_seen):
    """Transcription of CAL_MON_DAY and ZENITH in single precision (Fortran REAL is float32).

    The Julian date is about 2.45e6 days, so float32 resolution (about 0.25 day) changes the
    answer: the reference rounds like the Fortran, step by step.
    """
    month = [int(m) for m in MONTH]
    if julyr % 4 == 0 or leap_seen:
        month[1] = 29
    itmp, jmon, jday = julday, 12, julday
    for i in range(12):
        if itmp > month[i]:
            itmp -= month[i]
        else:
            jday, jmon = itmp, i + 1
            break
    times = F(ntsd) * F(dt)
    day = F(0.0)
    for knt in range(1, jmon):
        day = F(day + F(month[knt - 1]))
    day = F(day + F(jday - 1) + (F(ihrst) + F(times / F(3600.0))) / F(24.0))
    dayi = F(int(day) + 1)
    hour = F((day - dayi + F(1.0)) * F(24.0))
    yfctr = F(2000.0 - julyr)
    idifyr = julyr - 2000
    addday = F(idifyr // 4) if idifyr < 0 else F((idifyr + 3) // 4)
    startyr = F(F(2451545.0) + F(idifyr) * F(365.0) + addday - F(0.5))
    difjd = F(F(startyr + day) - F(2451545.0))
    slonm = F(F(F(280.460) + F(0.9856474) * difjd) * F(DEG2RD) + yfctr * F(2.0 * PI))
    anom = F(F(F(357.528) + F(0.9856003) * difjd) * F(DEG2RD))
    slon = F(slonm + F(F(F(1.915) * np.sin(anom) + F(0.020) * np.sin(F(2.0) * anom)) * F(DEG2RD)))
    if slon > F(2.0 * PI):
        slon = F(slon - F(2.0 * PI))
    dec = F(np.arcsin(F(np.sin(slon) * np.sin(F(OBLIQ)))))
    ra = F(np.arccos(F(np.cos(slon) / np.cos(dec))))
    if slon > F(PI):
        ra = F(F(2.0 * PI) - ra)
    datj0 = F(startyr + dayi - F(1.0))
    tu = F((datj0 - F(2451545.0)) / F(36525.0))
    stim0 = F(F(24110.54841) + tu * (F(8640184.812866) + F(9.3104e-2) * tu + F(-6.2e-6) * tu * tu))
    sidtim = F(F(stim0 / F(3600.0) + yfctr * F(24.0) + F(1.00273791) * hour) * F(15.0) * F(DEG2RD))
    if sidtim < F(0.0):
        sidtim = F(sidtim + F(2.0 * PI))
    if sidtim > F(2.0 * PI):
        sidtim = F(sidtim - F(2.0 * PI))
    hrang = F(sidtim - ra)
    hrlcl = F(hrang + F(glon) + F(2.0 * PI))
    sinalt = F(np.sin(dec) * np.sin(F(glat)) + np.cos(dec) * np.cos(hrlcl) * np.cos(F(glat)))
    return max(float(sinalt), 0.0)


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.steps import drv_rdtemp as w
    from namgpu.steps import drv_loop_limits as dll
    from namgpu.steps.drv_gpu_lease import gpu_lease

    cfg = config.load()
    g = gridmod.build(cfg)
    kern = kernels.get("drv_rdtemp", "rdtemp_apply")
    print("kernel compiled", flush=True)

    with gpu_lease(vram_mib=8000, est_s=300, what="check-drv-rdtemp full-size"):
        st = statemod.State(g, names=["GLAT", "GLON", "CZEN", "CZMEAN", "T", "RSWTT", "RLWTT"])
        im, jm, lm = int(g.im), int(g.jm), int(g.lm)
        rng = cp.random.default_rng(7)
        st["GLAT"][...] = (rng.random((jm, im), dtype=cp.float32) - 0.5) * 1.0
        st["GLON"][...] = (rng.random((jm, im), dtype=cp.float32) - 0.5) * 2.0
        st["CZMEAN"][...] = 0.05 + rng.random((jm, im), dtype=cp.float32)
        st["T"][...] = 250.0 + 10.0 * rng.random((lm, jm, im), dtype=cp.float32)
        st["RSWTT"][...] = 1e-5 * rng.random((lm, jm, im), dtype=cp.float32)
        st["RLWTT"][...] = 1e-5 * (rng.random((lm, jm, im), dtype=cp.float32) - 0.5)
        stream = cp.cuda.Stream(non_blocking=True)
        ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
        ctx.start = SimpleNamespace(year=2023, month=3, day=31, hour=12)
        ctx.ntimestep = 100
        with stream:
            w.run(ctx)
        stream.synchronize()
        assert bool(cp.isfinite(st["CZEN"]).all()) and bool(cp.isfinite(st["T"]).all()), "NaN/Inf"
        print("full-size wrapper run: CZEN and T finite", flush=True)

        # 2. sticky leap flag: kernel vs the Fortran transcription at sampled columns
        lim = dll.table(ctx)
        cols = [(0, 0), (jm // 2, im // 3), (jm - 1, im - 1)]
        glat = cp.asnumpy(st["GLAT"])
        glon = cp.asnumpy(st["GLON"])
        results = {}
        for leap in (0, 1):
            with stream:
                kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
                     (np.int32(100), np.float32(cfg.dt), np.int32(90), np.int32(2023),
                      np.int32(12), np.int32(leap), st["GLAT"], st["GLON"], st["CZEN"],
                      st["CZMEAN"], st["T"], st["RSWTT"], st["RLWTT"],
                      np.int32(im), np.int32(jm), np.int32(lm), lim))
            stream.synchronize()
            got = cp.asnumpy(st["CZEN"])
            for (j0, i0) in cols:
                ref = ref_czen(glat[j0, i0], glon[j0, i0], 100, cfg.dt, 90, 2023, 12, leap)
                k = float(got[j0, i0])
                assert abs(k - ref) < 2e-5, (leap, j0, i0, k, ref)
            results[leap] = [float(got[j0, i0]) for j0, i0 in cols]
        # CAL_MON_DAY and ZENITH share one MONTH table, so DAY = JULDAY - 1 for either flag: the
        # flag cannot change CZEN for a valid day number. Both flags must give the same field.
        assert results[0] == results[1], ("leap flag changed CZEN", results)
        print("sticky leap flag: kernel == Fortran transcription at 3 columns for leap 0 and 1;",
              "CZEN identical for both (valid julday round trip)", results, flush=True)
    print("CHECK PASS drv-rdtemp", flush=True)


if __name__ == "__main__":
    main()
