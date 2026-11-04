"""phys-bmj-bmjdrv: BMJDRV per-column driver for the BMJ convection branch.

Fortran: phys/module_CU_BMJ.F90:117-395 (subroutine bmjdrv, nam.v4.2.11, 83df4e13c1).
Kernel: namgpu/kernels/phys_bmj_bmjdrv.cu, entry phys_bmj_bmjdrv_column, one thread per column.
The kernel does the per-column setup, calls the BMJ column scheme (bmj_column, units phys-bmj-col-*)
for every interior column, and writes the tendencies and precipitation back. It runs once per call on
the NCNVC steps that the phys-bmj-driver glue calls it on (CONTRACT section 2, item 18).

Inputs (ctx.st; NMMB names, all written by phys-bmj-driver or the PBL / init units):
  PHINT [LM+1, JM, IM], PHMID, EXNER, Q, T, U_PHY, V_PHY [LM, JM, IM], XLAND [JM, IM],
  KPBL [JM, IM] int32, CLDEFI [JM, IM] (inout; set by phys-bmj-init).
Outputs (ctx.st): CU_ACT_FLAG [JM, IM] int32 (added here), RAINCV, CUTOP, CUBOT [JM, IM],
  DUDT_PHY, DVDT_PHY, RTHCUTEN, RQCUTEN [LM, JM, IM].
Scalars: the bmjdrv arguments from nam_configfile (fres, fr, fsl, fss, entrain, newall, newswap,
  newupup, nodeep, nprecip, dt) and the NMMB constants of CUCNVC (nmmb_common.cuh).

Argument binding (checked against module_CU_BMJ.F90 bmjdrv, 117-395; kernel map in phys_bmj_bmjdrv.cu):
  - Every argument of the Fortran dummy list is passed, including the ones the body never reads:
    DXH (grid row spacing, from ctx.grid, feeds the dead RDXEQ at Fortran 239), DYH (ctx.grid) and TH
    (state TH, the CUCNVC theta). The flags, dt, fres/fr/fsl/fss, ntsd, ncnvc and the constants
    (A2..TIW, module_CONSTANTS values) are passed as before.
  - CU_ACT_FLAG is allocated in run() (st.add, once). CLDEFI is persistent state from phys-bmj-init.
  - Values checked against parm/nam_configfile (the port's configfile, CONTRACT item 18): fres 0.75,
    fr 1.0, fsl 0.85, fss 0.85, entrain/newall/newupup/nodeep false, newswap true.

Refusals (no silent fallback):
  - LM must be 60 (the kernel's per-column arrays are sized for the 12 km CONUS configuration).
  - CLDEFI must be in ctx.st (phys-bmj-init writes it). Without it the BMJ state would start blank.
  - The BMJ column device function must exist in the compiled module; until the phys-bmj-col-* units
    land, compile of phys_bmj_bmjdrv.cu fails with an undefined bmj_column.

Launch: one kernel on ctx.stream, no host sync, no per-step allocation (CU_ACT_FLAG added once).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_bmj_bmjdrv"
KERNEL = "phys_bmj_bmjdrv_column"
BMJ_LM = 60

# NMMB constants used by CUCNVC for the BMJDRV call (module_CONSTANTS.F90; see nmmb_common.cuh).
R_D = np.float32(287.04)
R_V = np.float32(461.6)
CP = np.float32(1004.6)
P608 = np.float32(R_V / R_D - np.float32(1.0))   # module_CONSTANTS p608 = r_v/r - 1
CAPPA = np.float32(R_D / CP)


def _flag(raw: dict, key: str) -> np.int32:
    v = raw[key]
    return np.int32(1 if bool(v) else 0)


def _dxh_device(st, g):
    """DXH [JM] (grid row spacing, Fortran dxh(jds:jde)), single precision, cached on the device."""
    if "BMJ_DXH" not in st.fields:
        import cupy as cp
        st.fields["BMJ_DXH"] = cp.asarray(np.asarray(g.dxh, dtype=np.float32))
    return st.fields["BMJ_DXH"]


def run(ctx):
    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if lm != BMJ_LM:
        raise RuntimeError(f"phys-bmj-bmjdrv: lm={lm}; the kernel is sized for {BMJ_LM} levels")
    if "CLDEFI" not in st.fields:
        raise RuntimeError("phys-bmj-bmjdrv: CLDEFI is not in ctx.st; phys-bmj-init must write it")

    ntsd = int(ctx.ntimestep)
    ncnvc = int(cfg.nprecip)
    # Fortran 128-131: the BMJ call runs when ICLDCK == 0 or NTSD == 0 (the kernel repeats this test).
    if not (ntsd == 0 or ntsd % ncnvc == 0):
        return

    st.add("CU_ACT_FLAG", (jm, im), dtype="int32")
    raw = cfg.raw
    f32 = np.float32
    i32 = np.int32

    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            i32(im), i32(jm), i32(lm),
            i32(ntsd), i32(ncnvc),
            f32(cfg.dt), f32(g.dyh),
            f32(raw["fres"]), f32(raw["fr"]), f32(raw["fsl"]), f32(raw["fss"]),
            _flag(raw, "entrain"), _flag(raw, "newall"), _flag(raw, "newswap"),
            _flag(raw, "newupup"), _flag(raw, "nodeep"),
            f32(17.2693882), f32(273.15), f32(35.86), CAPPA, CP,
            f32(2.850e6), f32(2.501e6), f32(1.0e-12), f32(9.8060226), P608,
            f32(379.90516), R_D, f32(273.15),
            st["PHINT"], st["PHMID"], st["EXNER"], st["Q"], st["T"], st["U_PHY"], st["V_PHY"],
            st["TH"], st["XLAND"], _dxh_device(st, g), st["KPBL"],
            st["CLDEFI"], st["CU_ACT_FLAG"],
            st["RAINCV"], st["CUTOP"], st["CUBOT"],
            st["DUDT_PHY"], st["DVDT_PHY"], st["RTHCUTEN"], st["RQCUTEN"],
            *[ctx.tables["bmj_" + n] for n in ("ptbl", "qs0", "sqs", "sthe", "the0", "ttbl",
                                               "stheq", "the0q", "ttblq")],
        ))
