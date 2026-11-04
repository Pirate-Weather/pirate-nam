"""phys-noah-init: NOAH_LSM_INIT, the cold-start soil and canopy initialisation (nam.v4.2.11).

Fortran: phys/module_LS_NOAHLSM.F90:1164-1293 (NOAH_LSM_INIT), with its only callee FRH2O at 2644-2782.
Kernel: namgpu/kernels/phys_noah_init.cu (one launch, one thread per column, ns-loop inside).
NMMB call site: module_SOLVER_GRID_COMP.F90:10914 (CASE 'noah'):
    CALL NOAH_LSM_INIT(int_state%CMC, int_state%ISLTYP, int_state%STC, int_state%SMC,
                       int_state%IVEGSRC, int_state%SH2O, NUM_SOIL_LAYERS, int_state%RESTART, ...)
So: CANWAT = CMC, TSLB = STC, SMOIS = SMC.

Body in the Fortran, in order:
  1. if allowed_to_read: LSM_PARM_INIT(IVEGSRC) builds BB, MAXSMC, SATPSI. That is unit
     phys-noah-parm-tables and runs BEFORE this wrapper (driver order: parm tables, then this).
     This wrapper reads those three tables from ctx.tables["noah_parm"] (soil.BB, soil.MAXSMC,
     soil.SATPSI, the keys phys_noah_parm_tables.run uploads) and refuses to run without them.
  2. if not RESTART (the wrapper returns early on restart):
     a. ISLTYP range check (IVEGSRC 0: ISLTYP < 1; IVEGSRC 1: ISLTYP < 0). Diagnostic only; the
        kernel records it, and this wrapper reads the record back once at init and logs it to stderr
        with the Fortran message, one line for the first offending column.
     b. The SH2O / CANWAT block sits behind IF(.NOT.FNDSOILW) with FNDSOILW = .true. in NAM.
        It is therefore DEAD in NAM. The kernel ports it behind the runtime flag fndsoilw;
        this wrapper passes fndsoilw = 1 (the Fortran value), so NAM never runs it.

Consequence for NAM: NOAH_LSM_INIT leaves SH2O and CMC exactly as the caller set them. The
only live effect is the range check, which writes no state.

Arrays (CuPy, float32 unless noted; NMMB names):
    reads:  ctx.st SMC, STC [NSOIL, JM, IM]; ISLTYP int32 [JM, IM]
    writes: ctx.st SH2O [NSOIL, JM, IM] and CMC [JM, IM] (only when fndsoilw == 0; skipped here)
    tables: ctx.tables["noah_parm"] soil.BB, soil.MAXSMC, soil.SATPSI [NSLTYPE = 30]
    scratch: ctx.tables noah_init_bad_isltyp int32 [4], see the kernel for the layout
Parameters: IVEGSRC = 1 (IGBP, the configfile value), NSOIL = 4.

Launch: one kernel on ctx.stream, no host sync on the step path, no per-step allocation. The one
host read-back (bad_isltyp, and the ISLTYP value at the reported column) happens at init only.
"""
from __future__ import annotations

import contextlib
import sys

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_init"
NSLTYPE = 30
NSOIL = 4
IVEGSRC = 1          # configfile: ivegsrc 1 (IGBP)
FNDSOILW = 1         # Fortran PARAMETER FNDSOILW = .true. (module_LS_NOAHLSM.F90:1183)
BLOCK = kernels.BLOCK_COL
# Table keys as phys_noah_parm_tables.run uploads them (arrays are named "soil.<NAME>").
PARM_KEYS = {"noah_bb": "soil.BB", "noah_maxsmc": "soil.MAXSMC", "noah_satpsi": "soil.SATPSI"}
BAD_INIT_LAYOUT = (0, 2**31 - 1, 0, 2**31 - 1)   # flags, first range column, count, first table column


def _soil_tables(tables) -> dict:
    """Resolve BB, MAXSMC, SATPSI from ctx.tables; the parm-tables unit stores them under noah_parm."""
    parm = tables.get("noah_parm", {})
    out = {}
    for key, parm_key in PARM_KEYS.items():
        if key in tables:
            out[key] = tables[key]
        elif parm_key in parm:
            out[key] = parm[parm_key]
        else:
            raise RuntimeError(
                f"phys-noah-init needs the Noah soil table {parm_key!r} in ctx.tables['noah_parm']; "
                "run phys-noah-parm-tables first (LSM_PARM_INIT builds BB, MAXSMC, SATPSI).")
    return out


def _report_bad_isltyp(bad, isltyp, im: int) -> None:
    """One-time init read-back of the kernel's ISLTYP diagnostics, written as the Fortran message."""
    flags, first_range, n_range, first_table = (int(x) for x in bad.get())
    if flags & 1:
        j0, i0 = divmod(first_range, im)
        val = int(isltyp[j0, i0].get())
        print(f"module_sf_noahlsm.F: lsminit: out of range ISLTYP {i0 + 1} {j0 + 1} {val} "
              f"(first of {n_range} columns; the Fortran WRITEs every one)", file=sys.stderr)
    if flags & 2:
        j0, i0 = divmod(first_table, im)
        val = int(isltyp[j0, i0].get())
        print(f"phys-noah-init: ISLTYP {val} at column ({i0 + 1},{j0 + 1}) is outside the soil table "
              f"1..{NSLTYPE}; Fortran reads BB/MAXSMC/SATPSI out of bounds there (undefined), the port "
              "takes SH2O = SMOIS for such columns (first such column shown)", file=sys.stderr)


def run(ctx, restart: bool = False):
    """Cold-start NOAH_LSM_INIT. Returns ctx.st["SH2O"], unchanged in NAM (FNDSOILW = .true.)."""
    import cupy as cp

    tables = ctx.tables
    soil = _soil_tables(tables)
    st = ctx.st
    sh2o = st["SH2O"]
    # int_state%SLDPTH(L) = SLDPTH(L) (SOLVER_INITIALIZE, module_SOLVER_GRID_COMP.F90:966), cold start
    # and restart alike: the module constant of module_LS_NOAHLSM.F90:31.
    sl = st.add("SLDPTH", (NSOIL,))
    sl[...] = cp.asarray(np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)[:NSOIL])
    if restart:
        return sh2o
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    bad = tables.get("noah_init_bad_isltyp")
    if bad is None:
        bad = tables.setdefault("noah_init_bad_isltyp", cp.zeros((4,), cp.int32))
    bad.set(np.array(BAD_INIT_LAYOUT, dtype=np.int32))
    k_init = kernels.get(UNIT, "phys_noah_init_soil")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k_init(kernels.grid_cols(im, jm), BLOCK,
               (sh2o, st["CMC"], st["SMC"], st["STC"], st["ISLTYP"],
                soil["noah_bb"], soil["noah_maxsmc"], soil["noah_satpsi"], bad,
                np.int32(im), np.int32(jm), np.int32(NSOIL), np.int32(NSLTYPE),
                np.int32(IVEGSRC), np.int32(FNDSOILW)))
    # One-time read-back at init (not on the step path): log what the kernel recorded.
    _report_bad_isltyp(bad, st["ISLTYP"], im)
    return sh2o
