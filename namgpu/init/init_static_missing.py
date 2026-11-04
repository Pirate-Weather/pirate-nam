"""init-static-missing: fill the static surface fields the analysis does not carry.

Mode "new" (no NAM Fortran counterpart for the fill). Kernel: namgpu/kernels/init_static_missing.cu,
one launch over the 2-D (IM x JM) surface, no host sync on the step path (one count line at init).

In:      ctx.st IVGTYP, ISLTYP (int32 [JM, IM]); ALBASE, MXSNAL, EPSR, SM, SICE (float32 [JM, IM]).
         EPSR: NPS sea 0.97 / land 1.0 where the analysis has none (see _epsr).
Out:     the same arrays, with every missing or out-of-range entry filled (kernel header has the
         rules, the NAM line citations and the declared divergences).
Missing: ALBASE and MXSNAL count as missing unless they are a valid albedo fraction in (0, 1].
         NAM's nemsio reader leaves ALBASE=0 when the record is absent (module_INIT_READ_NEMSIO.F90:1772),
         and the port's allocation zeros carry no sentinel, so the test is on the value, not on a sentinel.
Tables:  the Noah VEGPARM / SOILPARM / GENPARM tables in namgpu/data/noah_tables, parsed once by
         ArWen's core/noah.py (load_tables, MODIFIED_IGBP_MODIS_NOAH, STAS) and uploaded once to
         ctx.tables['noah_static'].
Uncertain: the soil fallback (SOILPARM row 6, LOAM) and the mid-range albedo fill are a choice, not
         a NAM value. The NAM tree carries no Noah table, so the category columns are unverified.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "init_static_missing"
KERNEL = "init_static_missing"

ROOT = Path(__file__).resolve().parents[2]
from namgpu import paths  # noqa: E402
ARWEN_NOAH = paths.ARWEN / "core" / "noah.py"
TBL_DIR = paths.DATA / "noah_tables"
SOIL_DEFAULT_LOAM = 6          # SOILPARM order: sand, loamy sand, sandy loam, silt loam, silt, loam, ...


def _noah():
    spec = importlib.util.spec_from_file_location("arwen_noah", ARWEN_NOAH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses looks the module up by name
    spec.loader.exec_module(mod)
    return mod


def _tables(ctx):
    """Upload the packed Noah veg table and its column indices once (scratch on ctx.tables)."""
    key = "noah_static"
    if key in ctx.tables:
        return ctx.tables[key]
    import cupy as cp
    noah = _noah()
    t = noah.load_tables(mminlu="MODIFIED_IGBP_MODIS_NOAH", mminsl="STAS", tbl_dir=TBL_DIR)
    p = noah.pack_params(t)
    cols = noah.VEG_COLS
    if not (1 <= int(p.bare) <= int(p.lucats)):
        # Named breakage: the BARE fill for out-of-range IVGTYP would index outside the veg table.
        raise ValueError(f"{UNIT}: VEGPARM BARE category {p.bare} is outside 1..{p.lucats}")
    tab = {
        "veg": cp.asarray(np.ascontiguousarray(p.veg, dtype=np.float32)),
        "ncol": len(cols),
        "lucats": int(p.lucats),
        "slcats": int(p.slcats),
        "bare": int(p.bare),
        "c_albmin": cols.index("albedomin"),
        "c_albmax": cols.index("albedomax"),
        "c_maxalb": cols.index("maxalb"),
    }
    ctx.tables[key] = tab
    return tab


def run(ctx):
    import cupy as cp
    st, g = ctx.st, ctx.grid
    im, jm = int(g.im), int(g.jm)
    tab = _tables(ctx)
    kern = kernels.get(UNIT, KERNEL)

    # Count what the fill touches (one reduction at init, reported; nothing is switched off).
    iv, isl = st.IVGTYP, st.ISLTYP
    n_veg = int(cp.count_nonzero((iv < 1) | (iv > tab["lucats"])))
    n_slt = int(cp.count_nonzero((isl < 1) | (isl > tab["slcats"])))
    n_alb = int(cp.count_nonzero(~((st.ALBASE > 0) & (st.ALBASE <= 1))))
    n_mxs = int(cp.count_nonzero(~((st.MXSNAL > 0) & (st.MXSNAL <= 1))))
    print(f"{UNIT}: filled IVGTYP={n_veg} ISLTYP={n_slt} ALBASE={n_alb} MXSNAL={n_mxs} "
          f"of {im * jm} cells", flush=True)

    with ctx.stream:
        kern(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D,
             (st.IVGTYP, st.ISLTYP, st.ALBASE, st.MXSNAL,
              tab["veg"],
              np.int32(tab["lucats"]), np.int32(tab["ncol"]),
              np.int32(tab["c_albmin"]), np.int32(tab["c_albmax"]), np.int32(tab["c_maxalb"]),
              np.int32(tab["slcats"]), np.int32(tab["bare"]), np.int32(SOIL_DEFAULT_LOAM),
              np.int32(im), np.int32(jm)))
        _epsr(st, cp)
        _stdh(st, cp)
    return None


def _stdh(st, cp):
    """STDH (terrain standard deviation, the RRTM HPRIME input, module_RA_RRTM.F90:911). NAM never reads it:
    it is absent from NAM's own parm/nam_phy_state.txt, so it keeps the cold-start value
    int_state%STDH = -1.E6 of module_SOLVER_GRID_COMP.F90:1075. The radiation path sees that value (with
    iems = 0 setemis does not use it; the snow-roughness term clamps it). Set where never loaded (all zero)."""
    if "STDH" not in st.fields:
        return
    a = st.fields["STDH"]
    if not bool(cp.any(a != 0.0)):
        a.fill(-1.0e6)


def _epsr(st, cp):
    """EPSR (surface emissivity) where the analysis has none: NPS nemsinterp
    module_assorted_definitions.F90:134-155 writes 0.97 over sea (SM > 0.9, sea ice included, which
    NPS turns into SM = 0, SICE = 1 afterwards) and 1.0 over land. The bgrd3d00 analysis carries no
    EPSR record and no NMMB routine computes it during the run (Noah only reads EMISS, the RRTM path
    has its own sfcemis), so without this fill Noah sees EMISS = 0 (no downward longwave reaches the
    surface: LWDN = GLW * EMISSI, module_LS_NOAHLSM.F90:705-706). Cells already in (0, 1] are kept."""
    e = st.EPSR
    missing = ~((e > 0) & (e <= 1))
    sea = (st.SM > 0.9) | (st.SICE > 0.5)
    e[...] = cp.where(missing, cp.where(sea, cp.float32(0.97), cp.float32(1.0)), e)
