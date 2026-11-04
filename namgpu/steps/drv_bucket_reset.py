"""drv-bucket-reset: accumulator resets by NCLOD, NRDLW, NRDSW, NSRFC, NPREC.

Fortran: nmm/module_SOLVER_GRID_COMP.F90 (NAM nam.v4.2.11, 83df4e13c1), two sites:
  Site A, 5367-5470 (before and just after RDTEMP): NCLOD, NRDLW, NRDSW, NSRFC, NPREC blocks.
  Site B, 6348-6384 (after GSMDRIVE, CLTEND and TQADJUST, before the GFS physics branch): NRDLW,
          NRDSW and a 9-field NSRFC block. TURBL (5490) increments ARDLW, ARDSW and ASRFC, so Site B
          is what resets those counters after the turbulence increments.
Kernels: namgpu/kernels/drv_bucket_reset.cu (bucket_reset for Site A, bucket_reset_post for Site B),
one thread per (i, j), grid_2d with lm = 1.

Intervals (Fortran int_state%NCLOD etc., set at SOLVER_INITIALIZE 10299-10305):
  NSTEPS_PER_HOUR = NINT(3600./DT); NPREC = NSTEPS_PER_HOUR*NHRS_PREC, and likewise NCLOD (nhrs_clod),
  NRDLW (nhrs_rdlw), NRDSW (nhrs_rdsw), NSRFC (nhrs_srfc). With nhrs_udef false every NHRS is 1 hour
  (module_GET_CONFIG.F90:1670-1765, "IF(.NOT.NHRS_UDEF) NHRS_x=1"). The configfile values are read
  from cfg.raw after the placeholder substitution (LWBUCKET, SWBUCKET, EVAPBUCKET, CLOUDBUCKET,
  PRECIPBUCKET = 3 h in namgpu.config.SCRIPT_SUBSTITUTIONS), so the 12 km parent resets every 432
  steps (3 h x 144).

run(ctx, blocks):  the MOD(NTIMESTEP, N) == 0 tests are done on the host from ctx.ntimestep (scalar
           arithmetic, no device access). If none fires the step launches nothing. Otherwise one
           launch zeroes the blocks that fire. No host sync, no per-step allocation.
           The 24 Site A accumulators and the 17 Site B accumulators are all 2-D [JM, IM] in ctx.st
           (NCFRST and NCFRCV are int32).
With ctx.ntimestep == 0 (tools/check_unit.py) every block fires once, so the full-size check exercises
every branch.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "drv_bucket_reset"
KERNEL = "bucket_reset"          # Site A
KERNEL_POST = "bucket_reset_post"  # Site B

# Site A kernel argument order after (im, jm, do_nclod, do_nrdlw, do_nrdsw, do_nsrfc, do_nprec).
ARRAY_ORDER = (
    "ACFRST", "ACFRCV", "NCFRST", "NCFRCV",                  # NCLOD block, 5367-5376
    "ALWIN", "ALWOUT", "ALWTOA", "ARDLW",                    # NRDLW block, 5418-5428
    "ASWIN", "ASWOUT", "ASWTOA", "ARDSW",                    # NRDSW block, 5430-5440
    "SFCSHX", "SFCLHX", "SUBSHX", "SNOPCX", "POTFLX", "ASRFC",  # NSRFC block, 5442-5454
    "ACSNOW", "ACSNOM", "SSROFF", "BGROFF", "SFCEVP", "POTEVP",  # NPREC block, 5456-5468
)

# Site B kernel argument order after (im, jm, do_nrdlw, do_nrdsw, do_nsrfc).
POST_ARRAY_ORDER = (
    "ALWIN", "ALWOUT", "ALWTOA", "ARDLW",                    # NRDLW block, 6348-6357
    "ASWIN", "ASWOUT", "ASWTOA", "ARDSW",                    # NRDSW block, 6359-6368
    "ACSNOW", "POTEVP", "SFCEVP", "SFCLHX", "SFCSHX", "SUBSHX", "BGROFF", "SSROFF", "ASRFC",  # 6370-6384
)


def _nint(x: float) -> int:
    """Fortran NINT for positive x (round half up)."""
    return int(math.floor(x + 0.5))


def _hours(raw: dict, key: str) -> int:
    if not raw.get("nhrs_udef", False):
        return 1  # module_GET_CONFIG.F90:1670-1765: IF(.NOT.NHRS_UDEF) NHRS_x=1
    if key not in raw:
        raise RuntimeError(f"drv-bucket-reset: configfile has no {key}")
    h = int(raw[key])
    if h <= 0:
        raise RuntimeError(f"drv-bucket-reset: {key} must be > 0, got {h}")
    return h


def intervals(cfg) -> dict:
    """Step intervals NCLOD, NRDLW, NRDSW, NSRFC, NPREC (SOLVER_INITIALIZE 10299-10305)."""
    dt = float(cfg.dt)
    if dt <= 0.0:
        raise RuntimeError(f"drv-bucket-reset: dt must be > 0, got {dt}")
    nsph = _nint(3600.0 / dt)
    if nsph <= 0:
        raise RuntimeError(f"drv-bucket-reset: NSTEPS_PER_HOUR is {nsph} for dt={dt}")
    raw = cfg.raw
    return {
        "nclod": nsph * _hours(raw, "nhrs_clod"),
        "nrdlw": nsph * _hours(raw, "nhrs_rdlw"),
        "nrdsw": nsph * _hours(raw, "nhrs_rdsw"),
        "nsrfc": nsph * _hours(raw, "nhrs_srfc"),
        "nprec": nsph * _hours(raw, "nhrs_prec"),
    }


# blocks: "nclod"      = Site A NCLOD reset, before RDTEMP (5367-5376);
#         "post_rdtemp" = Site A NRDLW, NRDSW, NSRFC, NPREC resets, after RDTEMP (5418-5468);
#         "post_turb"   = Site B NRDLW, NRDSW, NSRFC resets, after the microphysics and CLTEND
#                         calls (6348-6384), after the turbulence increments;
#         None          = all five Site A blocks (the check_unit default for a bare call).
BLOCK_SETS = {
    None: ("nclod", "nrdlw", "nrdsw", "nsrfc", "nprec"),
    "nclod": ("nclod",),
    "post_rdtemp": ("nrdlw", "nrdsw", "nsrfc", "nprec"),
    "post_turb": ("nrdlw", "nrdsw", "nsrfc", "nprec"),
}


def run(ctx, blocks=None):
    if blocks not in BLOCK_SETS:
        raise ValueError(f"drv-bucket-reset: blocks must be one of {sorted(map(str, BLOCK_SETS))}, got {blocks!r}")
    st = ctx.st
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    iv = intervals(ctx.cfg)
    ntsd = int(ctx.ntimestep)
    active = BLOCK_SETS[blocks]

    # MOD(NTIMESTEP, N) == 0 is the same zero test in Fortran and Python for any sign of NTIMESTEP.
    flags = {k: int(k in active and ntsd % iv[k] == 0) for k in ("nclod", "nrdlw", "nrdsw", "nsrfc", "nprec")}
    if not any(flags.values()):
        return

    grid = kernels.grid_2d(im, jm, 1)
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    if blocks == "post_turb":
        kern = kernels.get(UNIT, KERNEL_POST)
        ptrs = tuple(st[name] for name in POST_ARRAY_ORDER)
        args = (np.int32(im), np.int32(jm),
                np.int32(flags["nrdlw"]), np.int32(flags["nrdsw"]), np.int32(flags["nsrfc"])) + ptrs
    else:
        kern = kernels.get(UNIT, KERNEL)
        ptrs = tuple(st[name] for name in ARRAY_ORDER)
        args = (np.int32(im), np.int32(jm)) + tuple(
            np.int32(flags[k]) for k in ("nclod", "nrdlw", "nrdsw", "nsrfc", "nprec")) + ptrs

    with scope:
        kern(grid, block, args)
        # Fortran empties ACPREC and CUPREC on the NPREC step (SOLVER_RUN 5673-5676 and 6393-6395). The kernels
        # above do not carry them, so they are cleared here on the same flag and stream.
        if flags["nprec"]:
            st["ACPREC"][...] = 0.0
            st["CUPREC"][...] = 0.0
