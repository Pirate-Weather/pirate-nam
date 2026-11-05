# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-taugb02: NMMB RRTM longwave band 2 optical depths and Planck fractions.

Ported from phys/radlw_main_nmmb.f:3716-3788 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE taugb02.
Kernel: namgpu/kernels/phys_rrtm_lw_taugb02.cu (rrtm_lw_taugb02, one thread per column, one launch).

NOTICE: NOAA/NCEP NMMB source (NCEP's modifications US Government work; RRTMG code AER, BSD-3-Clause, see NOTICE; no licence file in the NAM clone). The package
header block of phys/radlw_main_nmmb.f (nam.v4.2.11, 83df4e13c1) is carried verbatim below.

Which k-table: taugb02 does `use module_radlw_kgb02` (radlw_main_nmmb.f:3723), which is the module in
phys/radlw_datatb.f:2385-3223 with module_radlw_parameters (radlw_param.f). The _nmmb k-table module
(radlw_datatb_nmmb.f:1753) is not referenced by taugb02. The tables are in
namgpu/data/rrtm_lw/kgb02.npz (tools/extract_rrtm_lw_kgb02.py).

Tables: init_tables(ctx) is the *-init step for this unit. It uploads the six band-2 k-tables once
into ctx.tables. phys-rrtm-lw-params (todo) owns calling it at init (CONTRACT section 3). run() does not
load tables; it raises if they are missing.

In (ctx.st.fields, Fortran names, device float32 unless noted; per-level arrays are [LM, JM, IM] with
level 0 = model top, column = j*IM + i):
  LAYTROP  [JM, IM] int32   Fortran laytrop (count of lower layers)   from setcoef
  JP, JT, JT1, INDSELF, INDFOR  [LM, JM, IM] int32   Fortran values (1-based) from setcoef
  PAVEL, COLAMT, FAC00, FAC01, FAC10, FAC11, SELFFAC, SELFFRAC, FORFAC, FORFRAC  [LM, JM, IM]
    COLAMT is the H2O column amount, Fortran colamt(k,1).
Out (ctx.st.fields, added once; zero-filled, then only rows 10..21 written here):
  TAUG_LW   [LM, NGPTLW=140, JM, IM]   taug(ngpt, k): band 2 is rows NS02+ig, ig = 1..12, NS02 = 10
  FRACS_LW  [LM, NGPTLW=140, JM, IM]   fracs(ngpt, k), same rows
  These are the full gpoint arrays taumol combines; the taumol unit writes the other bands' rows and
  runs tautot = taug + tauaer.

Launch: one kernel launch on ctx.stream, no host sync, no per-step allocation, no Python loop over cells.
The launch helper `launch` takes contiguous [nlay, ncol] per-level arrays and [nlay, NGPTLW, ncol]
output arrays, so the test can run the full domain in column chunks.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

# !!!!!  ==============================================================  !!!!!
# !!!!!               lw-rrtm3 radiation package description             !!!!!
# !!!!!  ==============================================================  !!!!!
# !                                                                          !
# !   this package includes ncep's modifications of the rrtm-lw radiation    !
# !   code from aer inc.                                                     !
# !                                                                          !
# !    the lw-rrtm3 package includes these parts:                            !
# !                                                                          !
# !       'radlw_rrtm3_param.f'                                              !
# !       'radlw_rrtm3_datatb.f'                                             !
# !       'radlw_rrtm3_main.f'                                               !
# !                                                                          !
# !    the 'radlw_rrtm3_param.f' contains:                                   !
# !                                                                          !
# !       'module_radlw_parameters'  -- band parameters set up               !
# !                                                                          !
# !    the 'radlw_rrtm3_datatb.f' contains:                                  !
# !                                                                          !
# !       'module_radlw_avplank'     -- plank flux data                      !
# !       'module_radlw_ref'         -- reference temperature and pressure   !
# !       'module_radlw_cldprlw'     -- cloud property coefficients          !
# !       'module_radlw_kgbnn'       -- absorption coeffients for 16         !
# !                                     bands, where nn = 01-16              !
# !                                                                          !
# !    the 'radlw_rrtm3_main.f' contains:                                    !
# !                                                                          !
# !       'module_radlw_main'        -- main lw radiation transfer           !
# !                                                                          !
# !    in the main module 'module_radlw_main' there are only two             !
# !    externally callable subroutines:                                      !
# !                                                                          !
# !                                                                          !
# !       'lwrad'     -- main lw radiation routine                           !
# !          inputs:                                                         !
# !           (plyr,plvl,tlyr,tlvl,qlyr,olyr,gasvmr,                         !
# !            clouds,icseed,aerosols,sfemis,sfgtmp,                         !
# !            npts, nlay, nlp1, lprnt,                                      !
# !          outputs:                                                        !
# !            hlwc,topflx,sfcflx,                                           !
# !!         optional outputs:                                               !
# !            HLW0,HLWB,FLXPRF)                                             !
# !                                                                          !
# !       'rlwinit'   -- initialization routine                              !
# !          inputs:                                                         !
# !           ( me )                                                         !
# !          outputs:                                                        !
# !           (none)                                                         !
# !                                                                          !
# !    all the lw radiation subprograms become contained subprograms         !
# !    in module 'module_radlw_main' and many of them are not directly       !
# !    accessable from places outside the module.                            !
# !                                                                          !
# !    derived data type constructs used:                                    !
# !                                                                          !
# !     1. radiation flux at toa: (from module 'module_radlw_parameters')    !
# !          topflw_type   -  derived data type for toa rad fluxes           !
# !            upfxc              total sky upward flux at toa               !
# !            upfx0              clear sky upward flux at toa               !
# !                                                                          !
# !     2. radiation flux at sfc: (from module 'module_radlw_parameters')    !
# !          sfcflw_type   -  derived data type for sfc rad fluxes           !
# !            upfxc              total sky upward flux at sfc               !
# !            upfx0              clear sky upward flux at sfc               !
# !            dnfxc              total sky downward flux at sfc             !
# !            dnfx0              clear sky downward flux at sfc             !
# !                                                                          !
# !     3. radiation flux profiles(from module 'module_radlw_parameters')    !
# !          proflw_type    -  derived data type for rad vertical prof       !
# !            upfxc              level upward flux for total sky            !
# !            dnfxc              level downward flux for total sky          !
# !            upfx0              level upward flux for clear sky            !
# !            dnfx0              level downward flux for clear sky          !
# !                                                                          !
# !    external modules referenced:                                          !
# !                                                                          !
# !       'module physparam'                                                 !
# !       'module physcons'                                                  !
# !       'mersenne_twister'                                                 !
# !                                                                          !
# !    compilation sequence is:                                              !
# !                                                                          !
# !       'radlw_rrtm3_param.f'                                              !
# !       'radlw_rrtm3_datatb.f'                                             !
# !       'radlw_rrtm3_main.f'                                               !
# !                                                                          !
# !    and all should be put in front of routines that use lw modules        !
# !                                                                          !
# !==========================================================================!
# !                                                                          !
# !    the original aer's program declarations:                              !
# !                                                                          !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !                                                                          |
# !  Copyright (c) 2002-2020, Atmospheric and Environmental                  |
# !  Research, Inc. (AER). RRTMG is distributed by AER under the             |
# !  BSD-3-Clause licence; the full text is in NOTICE. (This                 |
# !  replaces the 2002-2007 terms in NAM's copy.)                            |
# !                                                                          |
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !                                                                          !
# ! ************************************************************************ !
# !                                                                          !
# !                              rrtmg_lw                                    !
# !                                                                          !
# !                                                                          !
# !                   a rapid radiative transfer model                       !
# !                       for the longwave region                            !
# !             for application to general circulation models                !
# !                                                                          !
# !                                                                          !
# !            atmospheric and environmental research, inc.                  !
# !                        131 hartwell avenue                               !
# !                        lexington, ma 02421                               !
# !                                                                          !
# !                           eli j. mlawer                                  !
# !                        jennifer s. delamere                              !
# !                         michael j. iacono                                !
# !                         shepard a. clough                                !
# !                                                                          !
# !                                                                          !
# !                       email:  miacono@aer.com                            !
# !                       email:  emlawer@aer.com                            !
# !                       email:  jdelamer@aer.com                           !
# !                                                                          !
# !        the authors wish to acknowledge the contributions of the          !
# !        following people:  steven j. taubman, karen cady-pereira,         !
# !        patrick d. brown, ronald e. farren, luke chen, robert bergstrom.  !
# !                                                                          !
# ! ************************************************************************ !
# !                                                                          !
# !    references:                                                           !
# !    (rrtm_lw/rrtmg_lw):                                                   !
# !      clough, s.A., m.w. shephard, e.j. mlawer, j.s. delamere,            !
# !      m.j. iacono, k. cady-pereira, s. boukabara, and p.d. brown:         !
# !      atmospheric radiative transfer modeling: a summary of the aer       !
# !      codes, j. quant. spectrosc. radiat. transfer, 91, 233-244, 2005.    !
# !                                                                          !
# !      mlawer, e.j., s.j. taubman, p.d. brown, m.j. iacono, and s.a.       !
# !      clough:  radiative transfer for inhomogeneous atmospheres: rrtm,    !
# !      a validated correlated-k model for the longwave.  j. geophys. res., !
# !      102, 16663-16682, 1997.                                             !
# !                                                                          !
# !    (mcica):                                                              !
# !      pincus, r., h. w. barker, and j.-j. morcrette: a fast, flexible,    !
# !      approximation technique for computing radiative transfer in         !
# !      inhomogeneous cloud fields, j. geophys. res., 108(d13), 4376,       !
# !      doi:10.1029/2002JD003322, 2003.                                     !
# !                                                                          !
# ! ************************************************************************ !
# !                                                                          !
# !    aer's revision history:                                               !
# !     this version of rrtmg_lw has been modified from rrtm_lw to use a     !
# !     reduced set of g-points for application to gcms.                     !
# !                                                                          !
# ! --  original version (derived from rrtm_lw), reduction of g-points,      !
# !     other revisions for use with gcms.                                   !
# !        1999: m. j. iacono, aer, inc.                                     !
# ! --  adapted for use with ncar/cam3.                                      !
# !        may 2004: m. j. iacono, aer, inc.                                 !
# ! --  revised to add mcica capability.                                     !
# !        nov 2005: m. j. iacono, aer, inc.                                 !
# ! --  conversion to f90 formatting for consistency with rrtmg_sw.          !
# !        feb 2007: m. j. iacono, aer, inc.                                 !
# ! --  modifications to formatting to use assumed-shape arrays.             !
# !        aug 2007: m. j. iacono, aer, inc.                                 !
# !                                                                          !
# ! ************************************************************************ !
# !                                                                          !
# !    ncep modifications history log:                                       !
# !                                                                          !
# !       nov 1999,  ken campana       -- received the original code from    !
# !                    aer (1998 ncar ccm version), updated to link up with  !
# !                    ncep mrf model                                        !
# !       jun 2000,  ken campana       -- added option to switch random and  !
# !                    maximum/random cloud overlap                          !
# !           2001,  shrinivas moorthi -- further updates for mrf model      !
# !       may 2001,  yu-tai hou        -- updated on trace gases and cloud   !
# !                    property based on rrtm_v3.0 codes.                    !
# !       dec 2001,  yu-tai hou        -- rewritten code into fortran 90 std !
# !                    set ncep radiation structure standard that contains   !
# !                    three plug-in compatable fortran program files:       !
# !                    'radlw_param.f', 'radlw_datatb.f', 'radlw_main.f'     !
# !                    fixed bugs in subprograms taugb14, taugb2, etc. added !
# !                    out-of-bounds protections. (a detailed note of        !
# !                    up_to_date modifications/corrections by ncep was sent !
# !                    to aer in 2002)                                       !
# !       jun 2004,  yu-tai hou        -- added mike iacono's apr 2004       !
# !                    modification of variable diffusivity angles.          !
# !       apr 2005,  yu-tai hou        -- minor modifications on module      !
# !                    structures include rain/snow effect (this version of  !
# !                    code was given back to aer in jun 2006)               !
# !       mar 2007,  yu-tai hou        -- added aerosol effect for ncep      !
# !                    models using the generallized aerosol optical property!
# !                    scheme for gfs model.                                 !
# !       apr 2007,  yu-tai hou        -- added spectral band heating as an  !
# !                    optional output to support the 500 km gfs model's     !
# !                    upper stratospheric radiation calculations. and       !
# !                    restructure optional outputs for easy access by       !
# !                    different models.                                     !
# !       oct 2008,  yu-tai hou        -- modified to include new features   !
# !                    from aer's newer release v4.4-v4.7, including the     !
# !                    mcica sub-grid cloud option. add rain/snow optical    !
# !                    properties support to cloudy sky calculations.        !
# !                    correct errors in mcica cloud optical properties for  !
# !                    ebert & curry scheme (ilwcice=1) that needs band      !
# !                    index conversion. simplified and unified sw and lw    !
# !                    sub-column cloud subroutines into one module by using !
# !                    optional parameters.                                  !
# !       mar 2009,  yu-tai hou        -- replaced the original random number!
# !                    generator coming from the original code with ncep w3  !
# !                    library to simplify the program and moved sub-column  !
# !                    cloud subroutines inside the main module. added       !
# !                    option of user provided permutation seeds that could  !
# !                    be randomly generated from forecast time stamp.       !
# !       oct 2009,  yu-tai hou        -- modified subrtines "cldprop" and   !
# !                    "rlwinit" according updats from aer's rrtmg_lw v4.8.  !
# !       nov 2009,  yu-tai hou        -- modified subrtine "taumol" according
# !                    updats from aer's rrtmg_lw version 4.82. notice the   !
# !                    cloud ice/liquid are assumed as in-cloud quantities,  !
# !                    not as grid averaged quantities.                      !
# !       jun 2010,  yu-tai hou        -- optimized code to improve efficiency
# !       apr 2012,  b. ferrier and y. hou -- added conversion factor to fu's!
# !                    cloud-snow optical property scheme.                   !
# !       nov 2012,  yu-tai hou        -- modified control parameters thru   !
# !                     module 'physparam'.                                  !
# !                                                                          !
# !!!!!  ==============================================================  !!!!!
# !!!!!                         end descriptions                         !!!!!
# !!!!!  ==============================================================  !!!!!

UNIT = "phys_rrtm_lw_taugb02"
KERNEL = "rrtm_lw_taugb02"
NG02 = 12                       # module_radlw_parameters NG02 (radlw_param.f:107)
NS02 = 10                       # module_radlw_parameters NS02 (radlw_param.f:114)
NGPTLW = 140                    # radlw_param_nmmb.f:141
BLOCK = 128
NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "kgb02.npz"
PREFIX = "rrtm_lw_kgb02_"
TABLE_KEYS = ("absa", "absb", "selfref", "forref", "fracrefa", "fracrefb")
TABLE_SHAPES = {"absa": (65, NG02), "absb": (235, NG02), "selfref": (10, NG02),
                "forref": (4, NG02), "fracrefa": (NG02,), "fracrefb": (NG02,)}
INT_LEVEL_INPUTS = ("JP", "JT", "JT1", "INDSELF", "INDFOR")
FLOAT_LEVEL_INPUTS = ("PAVEL", "COLAMT", "FAC00", "FAC01", "FAC10", "FAC11",
                      "SELFFAC", "SELFFRAC", "FORFAC", "FORFRAC")
OUTPUTS = ("TAUG_LW", "FRACS_LW")


def init_tables(ctx):
    """*-init step: upload the band-2 k-tables once into ctx.tables; later calls return the cached entries."""
    import cupy as cp

    t = ctx.tables
    if PREFIX + "absa" not in t:
        with np.load(NPZ) as z:
            host = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in TABLE_KEYS}
        for k, shp in TABLE_SHAPES.items():
            if host[k].shape != shp:
                raise ValueError(f"{UNIT}: {k} has shape {host[k].shape}, expected {shp}")
        for k in TABLE_KEYS:
            t[PREFIX + k] = cp.asarray(host[k])
    return {k: t[PREFIX + k] for k in TABLE_KEYS}


def _tables(ctx):
    """Return the uploaded band-2 k-tables. Raises if the init step has not run."""
    t = ctx.tables
    missing = [PREFIX + k for k in TABLE_KEYS if PREFIX + k not in t]
    if missing:
        raise KeyError(f"{UNIT}: k-tables {missing} not in ctx.tables; init_tables(ctx) must run at init "
                       "(called by phys-rrtm-lw-params, CONTRACT section 3)")
    return {k: t[PREFIX + k] for k in TABLE_KEYS}


def launch(tabs, *, ncol, nlay, laytrop, jp, jt, jt1, pavel, colamt, fac00, fac01, fac10, fac11,
           selffac, selffrac, indself, forfac, forfrac, indfor, taug_lw, fracs_lw, stream=None,
           gfirst: bool = False):
    """One kernel launch over ncol columns. Every per-level array is contiguous [nlay, ncol];
    laytrop is [ncol]; taug_lw and fracs_lw are contiguous [nlay, NGPTLW, ncol] (full gpoint arrays,
    rows NS02..NS02+NG02-1 written), or with gfirst=True the g-first [NGPTLW, nlay, ncol] arrays the
    LW seam (lw_core) holds, written in place (same values, only the output index differs). No host sync."""
    k = kernels.get(UNIT, KERNEL)
    grid = ((ncol + BLOCK - 1) // BLOCK, 1, 1)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (np.int32(ncol), np.int32(nlay), laytrop, jp, jt, jt1, pavel, colamt,
           fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
           tabs["absa"], tabs["absb"], tabs["selfref"], tabs["forref"],
           tabs["fracrefa"], tabs["fracrefb"], taug_lw, fracs_lw,
           np.int64(ncol if gfirst else NGPTLW * ncol), np.int64(nlay * ncol if gfirst else ncol)))


def run(ctx, out_gfirst=None):
    """Band-2 optical depths for the whole domain from ctx.st (setcoef outputs), into the full gpoint arrays.

    out_gfirst = (taug, fracs), each [NGPTLW, LM, JM, IM]: write band 2 straight into these g-first arrays
    (the LW seam's canonical taug/fracs) instead of allocating the layer-first TAUG_LW/FRACS_LW pair."""
    st = ctx.st
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    ncol = im * jm
    tabs = _tables(ctx)

    f = st.fields
    missing = [n for n in INT_LEVEL_INPUTS + FLOAT_LEVEL_INPUTS + ("LAYTROP",) if n not in f]
    if missing:
        raise KeyError(f"{UNIT}: setcoef outputs missing from ctx.st: {missing}")

    def flat(name):
        a = f[name]
        return a.reshape(lm, ncol)   # C-order [LM, JM, IM] -> [LM, JM*IM], a view

    if out_gfirst is not None:
        taug, fracs = out_gfirst
        for a in (taug, fracs):
            if tuple(a.shape) != (NGPTLW, lm, jm, im) or not a.flags.c_contiguous:
                raise ValueError(f"{UNIT}: out_gfirst arrays must be contiguous [{NGPTLW}, {lm}, {jm}, {im}]")
        dst = (taug.reshape(NGPTLW, lm, ncol), fracs.reshape(NGPTLW, lm, ncol))
    else:
        taug = st.add("TAUG_LW", (lm, NGPTLW, jm, im))
        fracs = st.add("FRACS_LW", (lm, NGPTLW, jm, im))
        dst = (taug.reshape(lm, NGPTLW, ncol), fracs.reshape(lm, NGPTLW, ncol))

    launch(tabs, ncol=ncol, nlay=lm, laytrop=f["LAYTROP"].reshape(ncol),
           jp=flat("JP"), jt=flat("JT"), jt1=flat("JT1"), pavel=flat("PAVEL"), colamt=flat("COLAMT"),
           fac00=flat("FAC00"), fac01=flat("FAC01"), fac10=flat("FAC10"), fac11=flat("FAC11"),
           selffac=flat("SELFFAC"), selffrac=flat("SELFFRAC"), indself=flat("INDSELF"),
           forfac=flat("FORFAC"), forfrac=flat("FORFRAC"), indfor=flat("INDFOR"),
           taug_lw=dst[0], fracs_lw=dst[1], stream=ctx.stream, gfirst=out_gfirst is not None)
    return {"TAUG_LW": taug, "FRACS_LW": fracs}


if __name__ == "__main__":
    # Standalone smoke: compile and upload the tables only. The full-size check is
    # tools/check-phys-rrtm-lw-taugb02.py.
    from types import SimpleNamespace

    ctx = SimpleNamespace(tables={}, stream=None)
    tabs = init_tables(ctx)
    print({k: tuple(v.shape) for k, v in tabs.items()})
    print(kernels.module(UNIT).compile() is not None)
