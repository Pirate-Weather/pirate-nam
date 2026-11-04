"""phys-bmj-ttblex: TTBLEX table look-up for the BMJ convection scheme.

Ported from sorc/nam_nems_nmmb.fd/src/atmos/phys/module_CU_BMJ.F90:2071-2154 under NAM_SRC=~/agent-scratch/nam-port/nam-gh
(nam.v4.2.11, git 83df4e13c1). Callers in the same file: ttblex calls at lines 813, 816, 843, 846,
1004, 1007, 1844, 1847 (low set itb/jtb/pl/rdp/rdthe; moisture set itbq/jtbq/plq/rdpq/rdtheq).
The ~/agent-scratch/nam-port/cmp/module_CU_BMJ.F90.n copy has no TTBLEX and must not be used as the reference. The Fortran routine
is a device-side helper: the BMJ column units call it once per parcel. The CUDA body lives in
namgpu/kernels/phys_bmj_ttblex.cu as the __device__ function ``bmj_ttblex`` (the callers in
phys_bmj_bmjdrv inline it per column) and as the extern "C" kernel ``phys_bmj_ttblex_columns``
used here for a launchable 2-D evaluation.

The tables are the BMJ set of module_CU_BMJ, built by phys-bmj-init (BMJ_INIT) into ctx.tables as
bmj_sthe, bmj_the0, bmj_ttbl (low set) and bmj_stheq, bmj_the0q, bmj_ttblq (moisture set). The scalars
are the module_CU_BMJ parameters (module_CU_BMJ.F90:62 and 100-102), not drv-tablet's:
    low set:      pl = 2500,  rdp = (itb-1)/(ph-pl),    rdthe = jtb-1
    moisture set: plq = 70000, rdpq = (itbq-1)/(ph-plq), rdtheq = jtbq-1
with ph = 105000, itb=76, jtb=134, itbq=152, jtbq=440.
The values are the Fortran PARAMETERs at sorc/nam_nems_nmmb.fd/src/atmos/phys/module_CU_BMJ.F90:62, 65-66 and 100-102; _set() takes them from
the module constants below, never from the table dict, so the wrapper is independent of the table build. Single precision, as the Fortran parameters
are evaluated in kfpt.

Launch: one kernel, one thread per column, on ctx.stream, no host sync, no per-step allocation.
Scratch columns (BMJ_TTLEX_PRSMID, BMJ_TTLEX_THESP, BMJ_TTLEX_TREF, [JM, IM] float32) are added
once with st.add.

Scope: the 2-D evaluation entry (``ttblex_columns``) is what the column units launch for a
whole field; per-column callers use the __device__ function directly.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_ttblex"
KERNEL = "phys_bmj_ttblex_columns"
SCRATCH_2D = ("BMJ_TTLEX_PRSMID", "BMJ_TTLEX_THESP", "BMJ_TTLEX_TREF")

PH = np.float32(105000.0)
PL_LO = np.float32(2500.0)
PL_Q = np.float32(70000.0)
ITB, JTB, ITBQ, JTBQ = 76, 134, 152, 440


def ttblex_columns(IM, JM, tables, plx, rdpx, rdthex, prsmid, thesp, tref, stream=None):
    """Evaluate TTBLEX for every column of [JM, IM] parcel fields prsmid and thesp into tref.

    ``tables`` supplies itb (itbx), jtb (jtbx), sthe, the0 and ttbl for one table set. plx,
    rdpx and rdthex are that set's scalars (pl, rdp, rdthe, or plq, rdpq, rdtheq).
    """
    grid, block = kernels.grid_cols(IM, JM), kernels.BLOCK_COL
    kern = kernels.get(UNIT, KERNEL)
    args = (
        np.int32(IM), np.int32(JM),
        np.int32(tables["itb"]), np.int32(tables["jtb"]),
        np.float32(plx), np.float32(rdpx), np.float32(rdthex),
        prsmid, thesp, tables["the0"], tables["sthe"], tables["ttbl"], tref,
    )
    kern(grid, block, args, stream=stream)
    return tref


def _set(ctx, tableset: str) -> dict:
    """The BMJ table set for ``tableset`` ("lo" or "q") with module_CU_BMJ's scalars."""
    t = ctx.tables
    if tableset == "lo":
        itb, jtb, pl = ITB, JTB, PL_LO
        keys = ("bmj_sthe", "bmj_the0", "bmj_ttbl")
    elif tableset == "q":
        itb, jtb, pl = ITBQ, JTBQ, PL_Q
        keys = ("bmj_stheq", "bmj_the0q", "bmj_ttblq")
    else:
        raise ValueError(f"phys-bmj-ttblex: tableset must be 'lo' or 'q' (got {tableset!r})")
    rdp = np.float32(np.float32(itb - 1) / (PH - pl))
    rdthe = np.float32(jtb - 1)
    return {
        "itb": itb, "jtb": jtb, "pl": pl, "rdp": rdp, "rdthe": rdthe,
        "sthe": t[keys[0]], "the0": t[keys[1]], "ttbl": t[keys[2]],
    }


def run(ctx, tableset: str = "lo"):
    """Done-rule entry: one full-size launch on ctx.st's [JM, IM] scratch columns.

    The BMJ tables are built by phys-bmj-init when missing (the same pattern as phys-bmj-col-cloudtop),
    so this unit never depends on a drv-* unit having run first.
    Scratch columns are created on first use.
    """
    from namgpu.physics.bmj import phys_bmj_init

    if "bmj_ttbl" not in ctx.tables:
        phys_bmj_init.run(ctx)
    IM, JM = ctx.grid.im, ctx.grid.jm
    for name in SCRATCH_2D:
        if name not in ctx.st.fields:
            ctx.st.add(name, (JM, IM))
    tab = _set(ctx, tableset)
    ttblex_columns(IM, JM, tab, tab["pl"], tab["rdp"], tab["rdthe"],
                   ctx.st["BMJ_TTLEX_PRSMID"], ctx.st["BMJ_TTLEX_THESP"],
                   ctx.st["BMJ_TTLEX_TREF"], stream=ctx.stream)
    return ctx.st["BMJ_TTLEX_TREF"]
