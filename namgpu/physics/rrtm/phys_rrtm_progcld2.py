"""phys-rrtm-progcld2: PROGCLD2, Ferrier prognostic cloud optical properties (np3d = 3) for RRTM.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_clouds_nmmb.f:787-1323, subroutine
progcld2, with the routines it calls, rsipath2_tmp (:2779-2964) and gethml (:2167-2494), inlined in
the kernel. NOAA US government work; the NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_progcld2.cu, entry phys_rrtm_progcld2_column, one thread per
column, one launch, no host sync, no per-step allocation (outputs are created once through st.add).

Called from grrad_nmmb.f:1589-1597 when np3d == 3 (the 12 km NAM configuration). The caller is
phys-rrtm-grrad-clouds, which builds the Fortran argument contract (phys_rrtm_grrad_clouds._contract)
and invokes this unit as its dependency through dep() below. The contract dict holds the Fortran
local arrays, named by their Fortran argument names, with LMK = LM + LTP levels (LTP = 1, the extra
top layer) and LMP = LMK + 1 interfaces:
  plyr, tlyr, tvly, qlyr, qstl, rhly, clw   [LMK, JM, IM]  (plyr hPa, tlyr/tvly K, qlyr/qstl kg/kg)
  plvl                                      [LMP, JM, IM]  interface pressure, hPa
  gcice, grain, grime                       [LMK, JM, IM]  ice fraction, rain fraction, rime
                                                           (Fortran gcice/grain/grime = f_ice/f_rain/r_rime)
  cldfk1                                    [LMK, JM, IM]  Thompson cloud fraction (cld_fraction != 0)
  xlat                                      [JM, IM]       latitude, radians
The producers are the existing units: phys-rrtm-grrad-setup (RAD_PLYR, RAD_PLVL, RAD_TLYR, RAD_TVLY,
RAD_QLYR with the QME5 floor and the extra top layer, RAD_QSTL, RAD_RHLY, RAD_CLDFK1, RAD_GCICE/GRAIN/
GRIME) and phys-rrtm-grrad-clouds (RAD_CLW: the ncld tracer sum with clw < EPSQ set to 0).

Outputs (Fortran names, LMK-sized, written once into scratch with st.add and returned as views):
  clouds [9, LMK, JM, IM] float32   Fortran clouds(IX,NLAY,NF_CLDS): 0 layer cloud fraction,
         1 liquid water path g/m2, 2 liquid eff radius um, 3 ice water path g/m2, 4 ice eff radius um,
         5 rain water path, 6 rain eff radius, 7 and 8 zero (snow is folded into ice)
  clds   [5, JM, IM] float32        Fortran clds(IX,5): low, mid, high, total, bl cloud fractions
  mtopa  [3, JM, IM] int32          Fortran mtop(IX,3): low, mid, high cloud top layer index
  mbota  [3, JM, IM] int32          Fortran mbot(IX,3): low, mid, high cloud base layer index
  (progcld2 does not write cldsa; that output belongs to progcld1, np3d 4.)

Scalar arguments (grrad_nmmb.f:1589-1597 and physparam.f):
  ivflip       0 or 1. NAM runs ivflip = 0 (rad_initialize_nmmb.f:183 ivflip = IFLIP = 0,
               module_SOLVER_GRID_COMP.F90:10714; the physparam.f:164 default 1 is overwritten), so the
               grrad arrays arrive top-first and the kernel takes the Fortran ivflip == 0 branch, which
               has its own cloud-fraction formula (1e5*clw*p/1000, capped at 1). Both branches are ported.
  llyr         REQUIRED. Fortran bl-cloud top layer set by cld_init (radiation_clouds_nmmb.f:334/342),
               read from phys-rrtm-cld-init check_status(ctx)["llyr"]. Not the module default of 2.
  iovr         max(iovrsw, iovrlw) from cld_init (physparam.f:143-144 = 1).
  lcrick, lcnorm, lnoprec, lsashal   physparam.f:146-149, all .false. in the NAM configuration.
  cld_fraction configfile 'cldfraction: default' = 0 (non-Thompson).

Not ported from this range: nothing. The routines have no MPI, halo, OpenMP or print. Fields that
progcld2 receives but never reads (slmsk, xlon, flgmin) are not passed. rsipath2_tmp's snow, res and
snden outputs are not kept (progcld2 never reads them). Unset mtop/mbot entries are written as zero
by the kernel (Fortran intent(out) leaves them undefined).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.physics.rrtm import phys_rrtm_cld_init as cld_init

UNIT = "phys_rrtm_progcld2"
KERNEL = "phys_rrtm_progcld2_column"
NF_CLDS = 9
LTP = 1               # grrad_nmmb.f: LTP = 1 (extra top layer); LMK = LM + LTP
MAXLAY = 128          # kernel per-column cloud-fraction array length (PGC_MAXLAY)

_CONTRACT = ("plyr", "plvl", "tlyr", "tvly", "qlyr", "qstl", "rhly", "clw", "gcice", "grain",
             "grime", "cldfk1", "xlat", "IM", "LMK", "LMP", "cld_fraction")


def _scratch(st, name, shape, dtype="float32"):
    arr = st.fields.get(name)
    if arr is None:
        arr = st.add(name, shape, dtype=dtype)
    return arr


def run(ctx, inputs, *, ivflip: int, llyr: int, iovr: int, lcrick: bool = False,
        lcnorm: bool = False, lnoprec: bool = False, lsashal: bool = False,
        cld_fraction: int = 0):
    """progcld2 for one full-domain call. inputs is the grrad_clouds contract dict (see module doc)."""
    for name in _CONTRACT:
        if name not in inputs:
            raise RuntimeError(f"{UNIT}: contract input {name!r} missing")
    im = int(inputs["IM"])
    lmk = int(inputs["LMK"])
    lmp = int(inputs["LMP"])
    jm = int(ctx.grid.jm)
    if im != int(ctx.grid.im):
        raise RuntimeError(f"{UNIT}: contract IM={im} differs from the grid im={ctx.grid.im}")
    if lmk != int(ctx.grid.lm) + LTP or lmp != lmk + 1:
        raise RuntimeError(f"{UNIT}: contract LMK={lmk}, LMP={lmp} must be LM+{LTP} and LMK+1")
    if lmk > MAXLAY:
        raise RuntimeError(f"{UNIT}: LMK={lmk} exceeds the kernel's per-column array ({MAXLAY})")
    if ivflip not in (0, 1):
        raise RuntimeError(f"{UNIT}: ivflip must be 0 (toa first, NAM) or 1 (surface first), got {ivflip}")
    if not 1 <= llyr <= lmk:
        raise RuntimeError(f"{UNIT}: llyr={llyr} outside 1..{lmk}")
    if cld_fraction != 0 and inputs["cldfk1"] is None:
        raise RuntimeError(f"{UNIT}: cld_fraction={cld_fraction} needs cldfk1 (Thompson)")

    st = ctx.st
    clouds = _scratch(st, "PGC2_CLOUDS", (NF_CLDS, lmk, jm, im))
    clds = _scratch(st, "PGC2_CLDS", (5, jm, im))
    mtop = _scratch(st, "PGC2_MTOP", (3, jm, im), dtype="int32")
    mbot = _scratch(st, "PGC2_MBOT", (3, jm, im), dtype="int32")

    i32 = np.int32
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lmk),
            i32(ivflip), i32(llyr), i32(iovr),
            i32(bool(lcrick)), i32(bool(lcnorm)), i32(bool(lnoprec)), i32(bool(lsashal)),
            i32(cld_fraction),
            inputs["plyr"], inputs["plvl"], inputs["tlyr"], inputs["tvly"], inputs["qlyr"],
            inputs["qstl"], inputs["rhly"], inputs["clw"], inputs["gcice"], inputs["grain"],
            inputs["grime"], inputs["cldfk1"], inputs["xlat"],
            clouds, clds, mtop, mbot,
        ))
    return {"clouds": clouds, "clds": clds, "mtopa": mtop, "mbota": mbot}


def dep(name, ctx, inputs):
    """grrad_clouds dependency entry: dep(name, ctx, inputs) -> dict with 'clouds', 'mtopa', 'mbota'."""
    if name != "progcld2":
        raise RuntimeError(f"{UNIT}: dep called for {name!r}")
    cld = cld_init.check_status(ctx)              # llyr and iovr from cld_init (rrtm_cld table)
    return run(ctx, inputs, ivflip=int(ctx.cfg.ivflip), llyr=cld["llyr"],
               iovr=cld["iovr"], cld_fraction=int(inputs["cld_fraction"]))
