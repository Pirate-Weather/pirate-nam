"""phys-rrtm-rad-initialize: rad_initialize_nmmb physparam flags for the RRTM radiation.

Ported from phys/rad_initialize_nmmb.f:2-224 (NAM nam.v4.2.11, 83df4e13c1), subroutine
rad_initialize_nmmb. Kernel: namgpu/kernels/phys_rrtm_rad_initialize.cu (rad_initialize_flags,
one thread, one launch).

In:      RadInitArgs (the call arguments of rad_initialize_nmmb, see below). Defaults are the
         12 km NAM CONUS parent values set in module_SOLVER_GRID_COMP.F90:10646-10722 (the caller
         PHYSICS_INITIALIZE block that precedes the call at 10797).
Out:     ctx.tables['rad_physparam'] = int32[26] device array, slots RI_* below. These are the
         module-level physparam flags of the Fortran (isolar ... ivflip) that the RRTM kernels read.
Next:    the driver calls unit phys-rrtm-radinit (radinit_nmmb, phys/grrad_nmmb.f:226-451) with the
         sigma interfaces `si` and levr=LM after this unit. That unit is not run here.
Scratch: the 26-int output buffer, allocated once on first call and reused.
Launch:  one kernel launch on ctx.stream, no host sync. The flag logic is O(1) scalar work done
         once at initialisation, not per step.

The Fortran print block (me == 0) is a host diagnostic and is not ported; it writes no state.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_rad_initialize"
KERNEL = "rad_initialize_flags"
KEY = "rad_physparam"
N_OUT = 26

# Output slots (order of the kernel out writes).
RI_ISOLAR, RI_ICTMFLG, RI_ICO2FLG, RI_IOZNFLG, RI_IAERFLG, RI_LASWFLG, RI_LALWFLG, RI_LAVOFLG, \
    RI_IAERMDL, RI_ICLDFLG, RI_ICMPHYS, RI_IOVRSW, RI_IOVRLW, RI_LSASHAL, RI_LCRICK, RI_LCNORM, \
    RI_LNOPREC, RI_ISUBCSW, RI_ISUBCLW, RI_ISWCLIQ, RI_ISWCICE, RI_ILWCLIQ, RI_ILWCICE, \
    RI_IALBFLG, RI_IEMSFLG, RI_IVFLIP = range(N_OUT)


@dataclass(frozen=True)
class RadInitArgs:
    """Arguments of rad_initialize_nmmb other than si/levr/me (those belong to phys-rrtm-radinit)."""
    ictm: int = 1            # module_SOLVER_GRID_COMP.F90:10646
    isol: int = 0            # :10652
    ico2: int = 1            # :10655
    iaer: int = 11           # :10658
    iaer_mdl: int = 0        # :10674
    ialb: int = 2            # :10676 (10356 sets IALB=0 earlier; the call uses the later value)
    iems: int = 0            # :10681
    ntcw: int = 3            # :10684
    num_p3d: int = 3         # configfile np3d: 3 (Ferrier)
    ntoz: int = 0            # :10691
    iovr_sw: int = 1         # :10693
    iovr_lw: int = 1         # :10695
    isubc_sw: int = 0        # :10697
    isubc_lw: int = 0        # :10702
    icliq_sw: int = 1        # :10707
    icice_sw: int = 3        # :10708 (NP3D=3 keeps 3; NP3D=8 switches to 4 at :10777)
    icliq_lw: int = 1        # :10709
    icice_lw: int = 1        # :10710
    sashal: bool = False     # :10718-10719 (LSASHAL = SASHAL>0 .and. .not.RAS); config value not read here
    crick_proof: bool = False  # :10720
    ccnorm: bool = True      # :10721
    norad_precip: bool = False  # :10722
    iflip: int = 0           # :10714 (the call passes this local IFLIP, set to 0)


def run(ctx, args: RadInitArgs | None = None):
    import cupy as cp

    a = args or RadInitArgs()
    if KEY not in ctx.tables:
        ctx.tables[KEY] = cp.zeros(N_OUT, dtype=cp.int32)
    out = ctx.tables[KEY]
    k = kernels.get(UNIT, KERNEL)
    i32 = np.int32
    call = (
        i32(a.ictm), i32(a.isol), i32(a.ico2), i32(a.iaer), i32(a.iaer_mdl), i32(a.ialb),
        i32(a.iems), i32(a.ntcw), i32(a.num_p3d), i32(a.ntoz), i32(a.iovr_sw), i32(a.iovr_lw),
        i32(a.isubc_sw), i32(a.isubc_lw), i32(a.icliq_sw), i32(a.icice_sw), i32(a.icliq_lw),
        i32(a.icice_lw), i32(int(a.sashal)), i32(int(a.crick_proof)), i32(int(a.ccnorm)),
        i32(int(a.norad_precip)), i32(a.iflip), out,
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1), call)
    return out
