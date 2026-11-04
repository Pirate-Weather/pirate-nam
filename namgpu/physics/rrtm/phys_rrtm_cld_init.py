"""phys-rrtm-cld-init: cld_init, the NMMB RRTM cloud-scheme flags and boundary-layer cloud top.

Fortran: phys/radiation_clouds_nmmb.f:225-349 (SUBROUTINE cld_init, nam.v4.2.11, 83df4e13c1), called from
radinit_nmmb (phys/grrad_nmmb.f:417) which is called from rad_initialize_nmmb (phys/rad_initialize_nmmb.f:210).
Kernel: namgpu/kernels/phys_rrtm_cld_init.cu (phys_rrtm_cld_init, one thread, one launch).

Arguments, all bound (none defaulted here):
  si      = SFULLD, the NAM SFULL = int_state%SGM (module_SOLVER_GRID_COMP.F90:10265, passed as SFULLD
            at :10751), LM+1 values, TOP FIRST. Read from the NAM header SGM via phys_rrtm_init.sfull_from_header
            (module_INIT_READ_BIN.F90:752). grid.sg2 is NOT used: it is a different array (SG2, not SGM).
  ivflip  = iflip of the rad_initialize call (module_SOLVER_GRID_COMP.F90:10716 sets IFLIP=0; rad_initialize_nmmb.f:183
            sets ivflip = iflip). ivflip 0 is the toa->sfc branch; the kernel returns llyr in the same top-first index
            as the NAM arrays, which is the index the cloud consumers (gethml, progcld2) take with ivflip = 0.
  icmphys = num_p3d of the rad_initialize call (rad_initialize_nmmb.f: icmphys = num_p3d; NAM NP3D).
  icldflg = 1 if ntcw > 0 else 0 (rad_initialize_nmmb.f:151-154; NAM NTCW = 3, :10684).
  iovrsw/iovrlw = iovr_sw/iovr_lw of the rad_initialize call (NAM IOVR_SW/IOVR_LW = 1, :10693/:10695).
All of them are read from the RadInitArgs that drv_physics_initialize recorded in ctx.tables["rad_init_args"]
(the same object rad_initialize ran with); the wrapper refuses if that record is absent.

Writes:
  ctx.tables["rrtm_cld"]   dict {"out": device int32[2] = [llyr, iovr]}. llyr is the Fortran module variable llyr
                           (1-based, top-first index, the same index space as the NAM arrays); iovr = max(iovrsw, iovrlw).
                           No consumer in namgpu reads this table yet: gethml and progcld2 take llyr as an argument, so
                           the caller that drives them must pass check_status(ctx)["llyr"].

Refusals (name the breakage):
  SGM absent: refused (phys_rrtm_init.sfull_from_header). A stand-in would put llyr at the wrong level.
  icldflg == 0 is refused: the diagnostic branch calls rhtable (phys-rrtm-rhtable, mode skip), which would
  leave the diagnostic RH table unset. The NAM configuration uses ntcw = 3, so icldflg 1, and the branch is not reached.
  icmphys not in {3, 4, 5, 8} is refused: the Fortran `stop`s on it (radiation_clouds_nmmb.f:317-318).

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation. Runs once at forecast start.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_init import sfull_from_header

UNIT = "phys_rrtm_cld_init"
KERNEL = "phys_rrtm_cld_init"
KEY = "rrtm_cld"
ARGS_KEY = "rad_init_args"
VALID_ICMPHYS = (3, 4, 5, 8)


def _rad_args(ctx):
    a = ctx.tables.get(ARGS_KEY) if ctx.tables else None
    if a is None:
        raise RuntimeError("phys-rrtm-cld-init: ctx.tables['rad_init_args'] is missing; the RadInitArgs that "
                           "rad_initialize_nmmb ran with must be recorded first (drv_physics_initialize)")
    return a


def run(ctx):
    import cupy as cp

    g = ctx.grid
    lm = int(g.lm)
    a = _rad_args(ctx)

    icmphys = int(a.num_p3d)
    if icmphys not in VALID_ICMPHYS:
        raise RuntimeError(f"phys-rrtm-cld-init: icmphys (num_p3d) = {icmphys}; the Fortran stops on it "
                           f"(valid: {VALID_ICMPHYS})")
    icldflg = 1 if int(a.ntcw) > 0 else 0          # rad_initialize_nmmb.f:151-154
    if icldflg == 0:
        raise RuntimeError("phys-rrtm-cld-init: icldflg == 0 needs rhtable (phys-rrtm-rhtable, mode skip); "
                           "the diagnostic RH table would be unset")
    ivflip = int(a.iflip)                           # rad_initialize_nmmb.f:183
    if ivflip not in (0, 1):
        raise RuntimeError(f"phys-rrtm-cld-init: iflip = {ivflip}; must be 0 or 1")
    iovrsw = int(a.iovr_sw)                         # rad_initialize_nmmb.f:165
    iovrlw = int(a.iovr_lw)                         # rad_initialize_nmmb.f:166

    si = np.ascontiguousarray(sfull_from_header(ctx, lm), dtype=np.float32)   # SFULLD, top first, refuses without SGM

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    out = cp.zeros(2, dtype=cp.int32)
    ctx.tables[KEY] = {"out": out}

    k = kernels.get(UNIT, KERNEL)
    with scope:
        k((1, 1, 1), (1, 1, 1), (cp.asarray(si), np.int32(lm), np.int32(ivflip),
                                 np.int32(iovrsw), np.int32(iovrlw), out))
    return out


def check_status(ctx):
    """Host read of the device outputs for tests only (not on the step path)."""
    out = ctx.tables[KEY]["out"].get()
    return {"llyr": int(out[0]), "iovr": int(out[1])}
