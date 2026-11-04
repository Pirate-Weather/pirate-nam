"""phys-rrtm-grrad-sw: the SW branch of GRRAD_NMMB (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:1818-1962. NOAA US government work;
the NOAA header of grrad_nmmb.f is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_grrad_sw.cu).

Fortran order, one radiation call (inside `if (lsswr)`):
  1. setalb (unit phys-rrtm-setalb)          -> ctx.st.SFCALB [4, JM, IM]
  2. sfalb = max(0.01, 0.5*(SFCALB[2]+SFCALB[4]))           kernel grrad_sw_sfalb
  3. day mask on CZEN >= 0.0001, device compaction into idxday/nday   daymask + scatter
  4. swrad (unit phys-rrtm-sw-swrad) over the day list      -> HTSWC, TOPFLX, SFCFLX,
                                                               HTSWB (if band heating requested)
  5. htrsw, htrswb, topfsw, sfcfsw from the day flags       kernels grrad_sw_htr, grrad_sw_fluxes

Day predicate (review fix): the Fortran isday is the DAYPARTS flag, coszen_loc >= 0.0001 with
coszen_loc = COSZMN_NMMB over the SW interval (grrad_nmmb.f:626-648, 1414, 1834). That is the
phys-rrtm-dayparts output ctx.tables["rrtm_dayparts"]["coszen"], not the instantaneous CZEN. Every
column of a daypart has the same isday, so the per-column mask below is exact membership.
swrad takes the interval mean cosine (coszen_loc, grrad_nmmb.f:1878, 1904) as its cosz input, which
is SW_COSZEN here, not CZEN.

Arrays come from ctx.st by CONTRACT name (namgpu/state.py). Scratch this unit adds with st.add:
  SFCALB  [4, JM, IM]    setalb output (added here if setalb has not created it)
  SFALB   [JM, IM]       grrad output, sfalb
  SW_DAY  int32[JM*IM]   day flag per column (zero-based flat c = j*IM + i)
  SW_CSUM int32[JM*IM]   inclusive cumsum of SW_DAY
  SW_IDXDAY int32[JM*IM] day column list, first SW_NDAY entries valid
  SW_NDAY int32[1]       number of day columns (device, never read on the host)
  HTRSW   [LM, JM, IM]   SW heating rate (state RSWTT is set from it by the radiation driver)
  TOPFSW  [3, JM, IM]    upfxc, dnfxc, upfx0 at TOA
  SFCFSW  [4, JM, IM]    upfxc, dnfxc, upfx0, dnfx0 at the surface
  SW_COSZEN [JM, IM]     coszen_loc, interval-mean cosine (dayparts output); swrad's cosz input
  FDNCMP  [6, JM, IM]    swrad output (always requested, grrad_nmmb.f:1910)
  SCMPSW  [6, JM, IM]    scmpsw (cmpfsw_type) = FDNCMP on day columns, zero on night columns
                         (grrad_nmmb.f:1935, read at 2111-2119 by the fluxr diagnostics)
  HTRSWB  [14, LM, JM, IM]  only when has_htrswb (optional band heating)

Interfaces this unit assumes (documented here because the two dependencies are not built yet):
  setalb:  namgpu.physics.rrtm.phys_rrtm_setalb.run(ctx) fills ctx.st.SFCALB.
  dayparts: namgpu.physics.rrtm.phys_rrtm_dayparts.run(ctx) gives coszen_loc (the day predicate source).
  swrad:   namgpu.physics.rrtm.phys_rrtm_sw_swrad.run(ctx, has_htrswb) reads ctx.st.SW_NDAY,
           SW_IDXDAY, SW_COSZEN (interval mean, the Fortran coszen_loc), the full 4-component SFCALB,
           icsdsw, faersw, clouds, solcon and the profiles; it writes HTSWC [LMK, JM, IM],
           TOPFLX [3, JM, IM], SFCFLX [4, JM, IM], FDNCMP [6, JM, IM] (always: the Fortran scmpsw
           argument is unconditional, grrad_nmmb.f:1910) and, when has_htrswb, HTSWB [14, LMK, JM, IM].
           swrad compacts with SW_IDXDAY and never evaluates a night column, so the Fortran CHK
           pad-out (grrad_nmmb.f:1841-1867) has no effect on any day output.
  Both are imported at call time; a missing one raises RuntimeError (named gap, not a skip).

Not ported from this range: the CHK pad-out (grrad_nmmb.f:1841-1867). It keeps swrad's unused tail
entries finite (radsw_main_nmmb.f:714-720 loops over CHK entries); the device compaction evaluates
day columns only. scmpsw is bound as SCMPSW and read by the driver post (fluxr 21-27, 2111-2119).

One pass of launches, no host sync, no per-step allocation (scratch is created on the first call).
"""
from __future__ import annotations

import contextlib
import importlib

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_dayparts as dayparts_unit

UNIT = "phys_rrtm_grrad_sw"
SETALB_MODULE = "namgpu.physics.rrtm.phys_rrtm_setalb"
SWRAD_MODULE = "namgpu.physics.rrtm.phys_rrtm_sw_swrad"

NBDSW = 14       # radsw_param_nmmb.f:156
NF_ALBD = 4      # radiation_surface_nmmb.f:94
NF_TOPFSW = 3
NF_SFCFSW = 4
NF_SCMP = 6      # cmpfsw_type, six components (swrad FDNCMP order)
LTP = 1          # grrad_nmmb.f:210, extra top layer
# NAM NMMB path: module_SOLVER_GRID_COMP.F90:10714 sets IFLIP=0 (TOA to surface) in the non-GFS
# branch, and rad_initialize_nmmb sets ivflip = iflip (rad_initialize_nmmb.f:183). So ivflip = 0,
# NOT the physparam default 1. grrad_nmmb.f:1168-1178 then gives kd = 1 for lextop (LTP > 0).
IVFLIP = 0
KD = 1 if IVFLIP == 0 else 0   # grrad_nmmb.f:1177-1178


def _dep(module_name: str, unit_label: str):
    """Import a sibling unit's wrapper. The dependency is a named gap until that unit lands."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"phys-rrtm-grrad-sw needs {module_name} (unit {unit_label}): {e}"
        ) from e


def _scratch(st, im: int, jm: int, lm: int, has_htrswb: bool):
    """Scratch arrays, created once (State.add is idempotent)."""
    lmk = lm + LTP
    st.add("SFCALB", (NF_ALBD, jm, im))
    st.add("SFALB", (jm, im))
    st.add("SW_DAY", (jm * im,), dtype="int32")
    st.add("SW_CSUM", (jm * im,), dtype="int32")
    st.add("SW_IDXDAY", (jm * im,), dtype="int32")
    st.add("SW_NDAY", (1,), dtype="int32")
    st.add("HTRSW", (lm, jm, im))
    st.add("TOPFSW", (NF_TOPFSW, jm, im))
    st.add("SFCFSW", (NF_SFCFSW, jm, im))
    st.add("SW_COSZEN", (jm, im))
    st.add("FDNCMP", (NF_SCMP, jm, im))
    st.add("SCMPSW", (NF_SCMP, jm, im))
    st.add("HTSWC", (lmk, jm, im))
    st.add("TOPFLX", (NF_TOPFSW, jm, im))
    st.add("SFCFLX", (NF_SFCFSW, jm, im))
    if has_htrswb:
        st.add("HTRSWB", (NBDSW, lm, jm, im))
        st.add("HTSWB", (NBDSW, lmk, jm, im))


def run(ctx, has_htrswb: bool = False):
    """Driver entry, one call per SW radiation step (every NRADS = 48 steps). Only when lsswr."""
    import cupy as cp

    if not bool(getattr(getattr(ctx, "cfg", None), "lfdncmp", False)):
        raise RuntimeError("phys-rrtm-grrad-sw needs cfg.lfdncmp: scmpsw is read by the driver post "
                           "(grrad_nmmb.f:2111-2119) and swrad must fill FDNCMP on every call")
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    lmk = lm + LTP
    n = jm * im
    st = ctx.st
    _scratch(st, im, jm, lm, has_htrswb)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    setalb = _dep(SETALB_MODULE, "phys-rrtm-setalb")
    swrad = _dep(SWRAD_MODULE, "phys-rrtm-sw-swrad")

    with scope:
        # 0. DAYPARTS / COSZMN_NMMB over the SW interval (grrad_nmmb.f:1414 and dayparts 633).
        #    Its coszen is the Fortran coszen_loc; isday is coszen_loc >= 0.0001 per column.
        cz = dayparts_unit.run(ctx)["coszen"]
        st.SW_COSZEN[...] = cz.reshape(jm, im)

        # 1. setalb fills SFCALB (Fortran call at grrad_nmmb.f:1818-1820)
        setalb.run(ctx)

        # 2. sfalb (grrad_nmmb.f:1822-1824)
        k_sfalb = kernels.get(UNIT, "grrad_sw_sfalb")
        k_sfalb(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
                (im, jm, st.SFCALB, st.SFALB))

        # 3. day list on device (isday, grrad_nmmb.f:1834; predicate grrad_nmmb.f:640 on coszen_loc)
        k_mask = kernels.get(UNIT, "grrad_sw_daymask")
        k_mask(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
               (im, jm, cz, st.SW_DAY))
        cp.cumsum(st.SW_DAY, dtype=cp.int32, out=st.SW_CSUM)
        k_scat = kernels.get(UNIT, "grrad_sw_scatter")
        k_scat(((n + kernels.BLOCK_COL[0] - 1) // kernels.BLOCK_COL[0],), kernels.BLOCK_COL,
               (n, st.SW_DAY, st.SW_CSUM, st.SW_IDXDAY, st.SW_NDAY))

        # 4. swrad over the day list (grrad_nmmb.f:1872-1916); reads SW_NDAY / SW_IDXDAY on device
        swrad.run(ctx, has_htrswb=has_htrswb)

        # 5a. htrsw (and htrswb) per column and level (grrad_nmmb.f:1918-1947), kd = 1
        k_htr = kernels.get(UNIT, "grrad_sw_htr")
        # htswb / htrswb are only touched when has_htrswb; otherwise HTRSW stands in as a valid pointer
        htswb = st.HTSWB if has_htrswb else st.HTRSW
        htrswb = st.HTRSWB if has_htrswb else st.HTRSW
        k_htr(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
              (im, jm, lm, lmk, KD, st.SW_DAY, st.HTSWC, st.HTRSW,
               1 if has_htrswb else 0, htswb, htrswb))

        # 5b. topfsw / sfcfsw / scmpsw (grrad_nmmb.f:1933-1935, 1952-1953)
        k_flx = kernels.get(UNIT, "grrad_sw_fluxes")
        k_flx(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
              (im, jm, st.SW_DAY, st.TOPFLX, st.SFCFLX, st.FDNCMP,
               st.TOPFSW, st.SFCFSW, st.SCMPSW))
    return st.HTRSW
