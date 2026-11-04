"""io-max-fields: MAX_FIELDS_HR, the hourly maxima and running averages of the NMMB diagnostics.

Port of nmm/module_DIAGNOSE.F90:2190-2518 (MAX_FIELDS_HR, nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/io_max_fields.cu (io_max_fields). The updraft helicity maxima and minima are the
io-uphlcy unit (CALC_UPHLCY, called from MAX_FIELDS_HR at the two CALL sites); this wrapper runs
io_uphlcy first so the UPHL fields are updated in the same order the Fortran uses.

Reads (ctx.st, CONTRACT names, [LM,JM,IM] or [JM,IM] float32):
    T, W_TOT, Z, REFL_10CM, PINT [LM+1], PD, PREC, CPRATE, T2, U10, V10, PSHLTR, TSHLTR, QSHLTR,
    SR, FIS, TH10, AKHS, AKMS, SNO; SGML2 [LM] (state); PSGML1 [LM] (grid, uploaded once).
    W is bound to W_TOT: the Fortran passes int_state%W_TOT as the W dummy of MAX_FIELDS_HR
    (module_SOLVER_GRID_COMP.F90:5049), and of both CALC_UPHLCY calls inside it.
Reads (ctx.tables): "fer_gpvs_tbpvs0" [7501] float32, the GPVS_hr saturation table built by
    phys-fer-gpvs (built here through phys_fer_gpvs.run if absent).
Writes (ctx.st, in place, MAX/MIN with the existing value, or running averages):
    REFDMAX, REFDMAX263K, PRATEMAX, FPRATEMAX, UPVVELMAX, DNVVELMAX, TLMAX, TLMIN, T02MAX, T02MIN,
    RH02MAX, RH02MIN, U10MAX, V10MAX, SPD10MAX, T10 (written every call), T10AVG, PSFCAVG,
    AKHSAVG, AKMSAVG, SNOAVG. UPHLMAX25/UPHLMIN25/UPHLMAX03/UPHLMIN03 are written by io-uphlcy.

Hourly reset (module_SOLVER_GRID_COMP.F90:4963-4996): when NTIMESTEP == 0 or
MOD(NTIMESTEP, NSTEPS_PER_RESET) == 0 every field in RESET_VALUES is set to its start value and
NCOUNT = 0. NSTEPS_PER_RESET = NINT(AVGMAXLEN/DT) (line 817), AVGMAXLEN = avg_max_length in the
configfile (3600 s). The reset runs before the call and before io-uphlcy, as in the Fortran.

Call gate (driver.py, SOLVER_RUN 5000-5002): MOD(NTSD, NSTEPS_PER_CHECK) == 0 and FILTER_METHOD == 0,
NSTEPS_PER_CHECK = MAX(2, NINT(40/DT)) (line 818). FILTER_METHOD is 0 outside the filter legs, so the
gate is "not ctx.filter_active" plus the step cadence.

NCOUNT: the Fortran counter of calls since the hourly reset. Kept on the host as
    ctx.tables["MAX_FIELDS_NCOUNT"]; zeroed by the reset, then incremented before the launch
    (Fortran NCOUNT=NCOUNT+1 precedes the averages).
Scalars: DT, NPHS, avg_max_length from ctx.cfg; NTSD = ctx.ntimestep. FPVS constants C1XPVS0 and
    C2XPVS0 come from GPVS_hr (XINC = (XMAX-XMIN)/(NX-1), C1 = 1 - XMIN/XINC, C2 = 1/XINC).
Launch: io-uphlcy launch, then one kernel on ctx.stream. No host sync, no per-step allocation
    after the first call (the grid vectors and the table are cached in ctx.tables).
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels
from namgpu.io import io_uphlcy

UNIT = "io_max_fields"
KERNEL = "io_max_fields"
NX = 7501                        # module_MP_FER_HIRES NX
XMIN = 180.0                     # module_MP_FER_HIRES XMIN
XMAX = 330.0                     # module_MP_FER_HIRES XMAX
TABLE_KEY = "fer_gpvs_tbpvs0"    # phys_fer_gpvs.KEY_TBPVS0
NCOUNT_KEY = "MAX_FIELDS_NCOUNT"
DBZMIN = -20.0                   # module_CONSTANTS dbzmin

# Output fields the kernel writes, in kernel argument order (after the input arrays).
_OUT_2D = ("REFDMAX", "REFDMAX263K", "PRATEMAX", "FPRATEMAX", "UPVVELMAX", "DNVVELMAX",
           "TLMAX", "TLMIN", "T02MAX", "T02MIN", "RH02MAX", "RH02MIN", "U10MAX", "V10MAX",
           "SPD10MAX", "T10", "T10AVG", "PSFCAVG", "AKHSAVG", "AKMSAVG", "SNOAVG")

# Start values of the hourly reset (module_SOLVER_GRID_COMP.F90:4963-4996). UPHLMAX (no 25/03
# suffix) is set there too but has no writer in this port (its CALC_UPHLCY call is commented out
# in MAX_FIELDS_HR); the UPHLMAX25/03 and UPHLMIN25/03 values are the ones io-uphlcy writes.
RESET_VALUES = {
    "TLMAX": -999.0, "TLMIN": 999.0, "T02MAX": -999.0, "T02MIN": 999.0,
    "RH02MAX": -999.0, "RH02MIN": 999.0, "SPD10MAX": -999.0, "U10MAX": -999.0, "V10MAX": -999.0,
    "UPVVELMAX": -999.0, "DNVVELMAX": 999.0,
    "T10AVG": 0.0, "T10": 0.0, "PSFCAVG": 0.0, "AKHSAVG": 0.0, "AKMSAVG": 0.0, "SNOAVG": 0.0,
    "REFDMAX": DBZMIN, "REFDMAX263K": DBZMIN, "PRATEMAX": 0.0, "FPRATEMAX": 0.0,
    "UPHLMAX25": -999.0, "UPHLMAX03": -999.0, "UPHLMIN25": 999.0, "UPHLMIN03": 999.0,
}


def nsteps_per_check(dt) -> int:
    """NSTEPS_PER_CHECK = MAX(2, NINT(40/DT)) (module_SOLVER_GRID_COMP.F90:818)."""
    return max(2, int(math.floor(40.0 / float(dt) + 0.5)))


def nsteps_per_reset(cfg) -> int:
    """NSTEPS_PER_RESET = NINT(AVGMAXLEN/DT) (module_SOLVER_GRID_COMP.F90:817)."""
    return int(math.floor(float(cfg.raw["avg_max_length"]) / float(cfg.dt) + 0.5))


def _device_vector(ctx, key: str, host):
    """Upload a small host vector once and keep it in ctx.tables."""
    if key not in ctx.tables:
        import cupy as cp
        if host is None:
            raise RuntimeError(f"io-max-fields: host vector {key!r} is not set on the grid")
        ctx.tables[key] = cp.asarray(np.ascontiguousarray(host, dtype=np.float32))
    return ctx.tables[key]


def fpvs_constants():
    """C1XPVS0 and C2XPVS0 as GPVS_hr sets them (single precision, Fortran order of operations)."""
    xinc = np.float32(np.float32(XMAX - XMIN) / np.float32(NX - 1))
    c1 = np.float32(1.0) - np.float32(XMIN) / xinc
    c2 = np.float32(1.0) / xinc
    return np.float32(c1), np.float32(c2)


def _reset(st) -> None:
    """The hourly reset block: every RESET_VALUES field back to its start value."""
    for name, value in RESET_VALUES.items():
        st[name].fill(np.float32(value))


def run(ctx):
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    ntsd = int(ctx.ntimestep)

    nchk = nsteps_per_check(cfg.dt)
    nrst = nsteps_per_reset(cfg)
    # The driver calls this unit on check steps and on reset steps (the reset block, SOLVER_RUN 4963, has
    # its own cadence before the MAX_FIELDS_HR gate at 5000). The old refusal of NSTEPS_PER_RESET not a
    # multiple of NSTEPS_PER_CHECK (the 4 km dt 25/3 s: 432 vs 5) is retired with that fix.
    check = ntsd % nchk == 0

    table = ctx.tables.get(TABLE_KEY)
    if table is None:
        from namgpu.physics.fer_hires import phys_fer_gpvs
        phys_fer_gpvs.run(ctx)
        table = ctx.tables[TABLE_KEY]
    if int(table.shape[0]) != NX:
        raise RuntimeError(f"io-max-fields: TBPVS0 has {table.shape[0]} entries, expected {NX}")

    # Hourly reset (NTIMESTEP == 0 or MOD(NTIMESTEP, NSTEPS_PER_RESET) == 0), then NCOUNT.
    reset = (ntsd == 0) or (ntsd % nrst == 0)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    if reset:
        ctx.tables[NCOUNT_KEY] = 0
        with scope:
            _reset(st)
    if not check:
        return None
    ncount = int(ctx.tables.get(NCOUNT_KEY, 0)) + 1
    ctx.tables[NCOUNT_KEY] = ncount

    # UPHL fields first (CALC_UPHLCY, the two CALL sites in MAX_FIELDS_HR).
    io_uphlcy.run(ctx)

    psgml1 = _device_vector(ctx, "io_max_fields:psgml1", g.psgml1)
    c1, c2 = fpvs_constants()

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_cols(im, jm)
    block = kernels.BLOCK_COL
    args = (
        st["T"], st["W_TOT"], st["Z"], st["REFL_10CM"], st["PINT"],
        st["PD"], st["PREC"], st["CPRATE"], st["T2"], st["U10"], st["V10"],
        st["PSHLTR"], st["TSHLTR"], st["QSHLTR"],
        st["SGML2"], psgml1, st["SR"], st["FIS"], st["TH10"], st["AKHS"], st["AKMS"], st["SNO"],
        ctx.tables[TABLE_KEY],
        *(st[name] for name in _OUT_2D),        # kernel order
        np.float32(cfg.dt), np.int32(cfg.nphs), np.int32(ntsd),
        np.int32(ncount), c1, c2,
        np.int32(im), np.int32(jm), np.int32(lm),
    )
    with scope:
        kern(grid, block, args)
