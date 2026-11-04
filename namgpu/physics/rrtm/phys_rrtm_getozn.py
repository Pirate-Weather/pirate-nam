"""phys-rrtm-getozn: getozn, climatological ozone profile for the NMMB RRTM call (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_gases_nmmb.f:984-1094, getozn.
NOAA US government work, no licence file; the NOAA/NCEP routine header is kept as the notice in
the kernel file namgpu/kernels/phys_rrtm_getozn.cu (one thread per column, every level loop inside).

Caller: grrad_nmmb.f:1397-1407, the ntoz == 0 branch: call getozn(prslk1, xlat, IM, LMK, olyr).
So the level count is LMK = LM + 1 (LTP = 1).

Inputs (ctx.st names):
  RAD_PRSLK1 [LMK, JM, IM]  exner function (grrad setup output, prslk1), index 0 = model top
  XLAT       [JM, IM]       latitude, radians (state table)
Output (ctx.st name):
  RAD_OLYR   [LMK, JM, IM]  ozone mass mixing ratio, same vertical order as the input

Climatology, bound from the names the producing units write (no stand-in, no fallback):
  ctx.tables["gas_o3r"], ctx.tables["gas_pkstr"]   written by phys-rrtm-gas-init when ioznflg == 0
                            (reads global_o3clim.txt). o3r device float32 [12, LOZ, JMR], month
                            first; pkstr device float32 [LOZ].
  ctx.tables["rrtm_gas_update"]                    the GasState written by phys-rrtm-gas-update when
                            ioznflg == 0: k1oz, k2oz (1-based month indices), facoz (time factor).
  bind_climatology() reads these names on every call. A missing name raises RuntimeError naming the
  unit that must run first.

ozne_def grid constants (module_SOLVER_GRID_COMP.F90:10380-10382: levozc = 17, latsozc = 18,
blatc = -85.0). dphiozc is not set in that block; module_SOLVER_GRID_COMP.f90:9419 derives it as
-(blatc + blatc) / (latsozc - 1) = 10.0. blte = blatc, dlte = dphiozc.
ivflip: the NAM setting is iflip = 0 (toa -> sfc), passed explicitly.

Missing inputs refuse, they do not stand in:
  - XLAT all zero (never written; State.add zero-fills): RuntimeError, checked once per ctx.
  - RAD_PRSLK1 not strictly positive (phys-rrtm-grrad-setup has not run): RuntimeError, once per ctx.
  - LOZ > 64 (kernel level storage): ValueError.

Precision: the kernel is single precision. The Fortran kind_phys is 8 in the default NAM build
(phys/machine.f, #ifndef SINGLE_PREC) and 4 only under -DSINGLE_PREC. The receipt names the flag
KIND_PHYS_BUILD_FLAG; the bitwise claim applies only to a reference built with that flag.

run(ctx) is the done-rule entry (tools/check_unit.py). With ctx=None, run() is a SELF-TEST only: a
full-size 954 x 835 x 60 set with an explicitly labelled test climatology, returned as
climatology_source == "selftest". A forecast never takes that path. Launch: one grid_cols launch,
no host sync (the one-time input checks sync once per ctx).
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_getozn"
KERNEL = "phys_rrtm_getozn"
JMR_NAM = 18           # ozne_def latsozc (module_SOLVER_GRID_COMP.F90:10381)
LOZ_NAM = 17           # ozne_def levozc (module_SOLVER_GRID_COMP.F90:10380)
BLTE_NAM = -85.0       # ozne_def blatc (module_SOLVER_GRID_COMP.F90:10382)
DLTE_NAM = 10.0        # ozne_def dphiozc = -(2*blatc)/(latsozc-1) (module_SOLVER_GRID_COMP.f90:9419)
GETOZN_MAXLOZ = 64     # kernel local o3i storage (phys_rrtm_getozn.cu)
NAM_ROCP = 0.2857142   # con_rocp, Rd/cp
PRESS_TOP = 200.0      # Pa, PT (contract section 5), self-test profile only
PRESS_SFC = 100000.0   # Pa, self-test profile only
KIND_PHYS_BUILD_FLAG = "SINGLE_PREC"  # single-precision build; the default NAM build is kind_phys=8


def _selftest_climatology(xp, jmr: int, loz: int):
    """Explicit self-test climatology. Used only when ctx is None (the done-rule full-size check)."""
    pstr = np.geomspace(1.0, 1000.0, loz).astype(np.float32)          # hPa, top to bottom
    pkstr = (pstr / np.float32(1000.0)) ** np.float32(NAM_ROCP)
    prof = (3.0e-6 + 6.0e-6 * np.exp(-((pstr - 30.0) / 40.0) ** 2)).astype(np.float32)
    o3r = np.broadcast_to(prof[None, :, None], (12, loz, jmr)).astype(np.float32).copy()
    return {
        "o3r": xp.asarray(o3r),
        "pkstr": xp.asarray(pkstr.astype(np.float32)),
        "k1oz": 1, "k2oz": 2, "facoz": 0.5,
        "blte": BLTE_NAM, "dlte": DLTE_NAM, "jmr": jmr, "loz": loz,
    }


def _selftest_prslk(xp, lmk: int, jm: int, im: int):
    """Exner function (p/1e5)^(Rd/cp) on the layers, same in every column (self-test input only)."""
    p = np.linspace(PRESS_TOP, PRESS_SFC, lmk).astype(np.float32)
    ex = (p / np.float32(1.0e5)) ** np.float32(NAM_ROCP)
    return xp.asarray(np.broadcast_to(ex[:, None, None], (lmk, jm, im)).copy())


def bind_climatology(tables) -> dict:
    """Bind the ozone climatology from the names gas_init and gas_update write (no stand-in)."""
    tables = tables if tables is not None else {}
    o3r = tables.get("gas_o3r")
    pkstr = tables.get("gas_pkstr")
    gu = tables.get("rrtm_gas_update")
    missing = [n for n, v in (("gas_o3r", o3r), ("gas_pkstr", pkstr), ("rrtm_gas_update", gu))
               if v is None]
    if missing:
        raise RuntimeError(
            f"phys-rrtm-getozn: ozone climatology not bound (missing ctx.tables {missing}). Run "
            "phys-rrtm-gas-init with ioznflg=0 (reads global_o3clim.txt) and phys-rrtm-gas-update "
            "with ioznflg=0 first. No stand-in climatology is used.")
    if o3r.ndim != 3 or o3r.shape[0] != 12:
        raise RuntimeError(f"phys-rrtm-getozn: gas_o3r must be [12, LOZ, JMR], got {tuple(o3r.shape)}")
    loz, jmr = int(o3r.shape[1]), int(o3r.shape[2])
    if tuple(pkstr.shape) != (loz,):
        raise RuntimeError(f"phys-rrtm-getozn: gas_pkstr shape {tuple(pkstr.shape)} != ({loz},)")
    if (loz, jmr) != (LOZ_NAM, JMR_NAM):
        raise RuntimeError(f"phys-rrtm-getozn: ozne_def NAM grid is LOZ={LOZ_NAM}, JMR={JMR_NAM}; "
                           f"gas_init produced LOZ={loz}, JMR={jmr}")
    if loz > GETOZN_MAXLOZ:
        raise ValueError(f"phys-rrtm-getozn: LOZ={loz} exceeds kernel storage {GETOZN_MAXLOZ}")
    if int(gu.k1oz) < 1 or int(gu.k2oz) < 1:
        raise RuntimeError("phys-rrtm-getozn: gas_update has not set the ozone time indices "
                           f"(k1oz={gu.k1oz}, k2oz={gu.k2oz}); run phys-rrtm-gas-update with ioznflg=0")
    return {
        "o3r": o3r, "pkstr": pkstr,
        "k1oz": int(gu.k1oz), "k2oz": int(gu.k2oz), "facoz": float(gu.facoz),
        "blte": BLTE_NAM, "dlte": DLTE_NAM, "jmr": jmr, "loz": loz,
    }


def run(ctx=None, *, ivflip: int = 0):
    """Launch the unit once. Returns the arrays it wrote (dict of cupy arrays)."""
    import cupy as cp

    if ctx is None:
        # SELF-TEST ONLY (done-rule full-size check): explicit test inputs, labelled in the return.
        im, jm, lm = 954, 835, 60
        lmk = lm + 1
        prslk = _selftest_prslk(cp, lmk, jm, im)
        xlat = cp.zeros((jm, im), dtype=cp.float32)
        xlat[...] = cp.linspace(-1.2, 1.2, jm, dtype=cp.float32)[:, None]
        oz = _selftest_climatology(cp, JMR_NAM, LOZ_NAM)
        source = "selftest"
        olyr = cp.zeros((lmk, jm, im), dtype=cp.float32)
    else:
        im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
        lmk = lm + 1
        st = ctx.st
        tables = ctx.tables
        if "RAD_PRSLK1" not in st.fields or "XLAT" not in st.fields:
            raise RuntimeError("phys-rrtm-getozn: state lacks RAD_PRSLK1 or XLAT")
        prslk = st.fields["RAD_PRSLK1"]
        xlat = st.fields["XLAT"]
        # One-time checks per ctx, cached in tables. State.add zero-fills, so an unwritten input is
        # all zeros; the exner function is strictly positive once phys-rrtm-grrad-setup has run.
        if not tables.get("getozn_prslk_checked"):
            if not bool(cp.all(prslk > 0).get()):
                raise RuntimeError("phys-rrtm-getozn: RAD_PRSLK1 is not positive everywhere; "
                                   "phys-rrtm-grrad-setup must run first (no synthetic profile)")
            tables["getozn_prslk_checked"] = True
        if not tables.get("getozn_xlat_checked"):
            if not bool(cp.any(xlat != 0).get()):
                raise RuntimeError("phys-rrtm-getozn: XLAT is all zero (latitude never set); "
                                   "refusing to run every column at latitude 0")
            tables["getozn_xlat_checked"] = True
        oz = bind_climatology(tables)
        source = "gas_init/gas_update"
        olyr = st.add("RAD_OLYR", (lmk, jm, im))

    jmr, loz = int(oz["jmr"]), int(oz["loz"])
    grid = kernels.grid_cols(im, jm)
    k = kernels.get(UNIT, KERNEL)
    k(grid, kernels.BLOCK_COL,
      (prslk, xlat, oz["o3r"], oz["pkstr"], olyr,
       np.int32(im), np.int32(jm), np.int32(lmk),
       np.int32(jmr), np.int32(loz),
       np.int32(oz["k1oz"]), np.int32(oz["k2oz"]), np.float32(oz["facoz"]),
       np.float32(oz["blte"]), np.float32(oz["dlte"]),
       np.int32(ivflip)))
    return {"RAD_OLYR": olyr, "climatology_source": source,
            "kind_phys": "single", "kind_phys_build_flag": KIND_PHYS_BUILD_FLAG}
