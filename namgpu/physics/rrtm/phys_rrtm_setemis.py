"""phys-rrtm-setemis: setemis, surface emissivity for the NMMB RRTM longwave call (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_surface_nmmb.f:730-897, setemis.
NOAA US government work; the routine header is kept as the notice in
namgpu/kernels/phys_rrtm_setemis.cu (kernel phys_rrtm_setemis, one thread per column, one launch).

Caller: module_RA_RRTM.F90:1113 -> grrad_nmmb.f:1963 (call setemis with IM columns); consumer:
phys_rrtm_grrad_lw reads st.fields["sfcemis"].

Module state comes from phys-rrtm-sfc-init, which must run first (RuntimeError otherwise):
  ctx.tables['sfc_flags']   int32[3] = [ialbflg, iemsflg, iemslw]. The kernel reads [0] (ialbflg)
                            and [2] (iemslw, already mod(iemsflg,10) with the missing-file fallback).
  ctx.tables['sfc_idxems']  int32 [JMXEMS=180, IMXEMS=360], the emissivity type map read from
                            semis_file. The kernel reads it only when iemslw == 1, and sfc_init
                            produces the table exactly when that case can occur. When the table is
                            absent, a zero array of the same shape is passed; it is never read then.

Inputs (ctx.st, device float32 [JM, IM]), NMMB driver mapping in module_RA_RRTM.F90:
  GLON, GLAT   xlon, xlat (radians)
  SM           sea mask; slmsk = 1 - SM (:917), except SICE > 0.5 forces slmsk = 2 (:913-914)
  SICE         sea-ice fraction (state table); SICE > 0.5 is the sea-ice branch
  SI           snow depth in mm = SNWDPH (:908)
  SNOWC        snow cover fraction, used when ialbflg == 1
  Z0           roughness length in m; zorlf = 100*Z0 cm (:907)
  STDH         topographic standard deviation = HPRIME_V (:911), state table NAME 'STDH' (int_state%STDH,
               module_SOLVER_INTERNAL_STATE.F90:1612). The unit refuses while STDH is all zero: no init unit
               loads it yet (gap, owner: init-static-missing), so zeros would silently drop the snow hrgh term.
  TSKIN        surface temperature tsknf. module_RA_RRTM passes TSEA; TSKIN is used here (see note).
Output: st.fields["sfcemis"], device float32 [JM, IM].

Notes: tairf is an argument of setemis that the Fortran body never reads, so it is not passed.
Sea-ice and sea points use emsref(7) and emsref(1) exactly as the routine does.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_sfc_init import KEY_FLAGS, KEY_IDXEMS

STDH_CHECK_KEY = "rad_setemis_stdh_checked"
UNIT = "phys_rrtm_setemis"
KERNEL = "phys_rrtm_setemis"
IMXEMS = 360
JMXEMS = 180


def _scope(stream):
    return stream if stream is not None else contextlib.nullcontext()


def run(ctx):
    """Launch the unit once. Returns {"sfcemis": device array}. No host sync."""
    import cupy as cp

    st = ctx.st
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    f = st.fields

    flags = ctx.tables.get(KEY_FLAGS)
    if flags is None:
        raise RuntimeError(
            "phys-rrtm-setemis: ctx.tables['sfc_flags'] is missing; run phys-rrtm-sfc-init first")

    idx = ctx.tables.get(KEY_IDXEMS)
    if idx is None:
        idx = cp.zeros((JMXEMS, IMXEMS), dtype=cp.int32)   # never read unless sfc_flags[2] == 1 (see docstring)

    stdh = f["STDH"]
    if STDH_CHECK_KEY not in ctx.tables:               # one reduction at first call, not per step
        if not bool(cp.any(stdh != 0.0)):
            raise RuntimeError("phys-rrtm-setemis: state STDH (int_state%STDH, module_SOLVER_INTERNAL_STATE.F90:1612) "
                               "is all zero, so it was never loaded. No init unit loads it yet (gap, owner: "
                               "init-static-missing). Refusing rather than zeroing the snow-depth term")
        ctx.tables[STDH_CHECK_KEY] = True

    out = st.add("sfcemis", (jm, im))
    kern = kernels.get(UNIT, KERNEL)
    with _scope(ctx.stream):
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            f["GLON"], f["GLAT"], f["SM"], f["SICE"], f["SI"], f["SNOWC"], f["Z0"], stdh, f["TSKIN"],
            idx, flags, out, np.int32(im), np.int32(jm)))
    return {"sfcemis": out}
