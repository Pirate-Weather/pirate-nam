"""phys-gwd-driver: GWD_DRIVER (NAM nmm/module_GWD.F90:125-362, nam.v4.2.11, 83df4e13c1, NOAA/NCEP/EMC).

Kernel: namgpu/kernels/phys_gwd_driver.cu (phys_gwd_driver). One launch, one thread per (i, j)
mass column over the full domain (single card, halo-free, global = false). The GWD_col column
physics is the phys-gwd-col-blocking and phys-gwd-col-wave units, compiled into the same module
through phys_gwd_col.cuh (gwd_col).

Called from phys-turbl-driver (namgpu/physics/driver/phys_turbl_driver.py) on the physics cadence
(NPHS = 2, DTPHS = NPHS * DT = 50 s) when gwdflg is true. run(ctx) is the entry point.

In (ctx.st, [JM, IM] float32 unless noted): HSTDV, HCNVX, HASYW, HASYS, HASYSW, HASYNW, HLENW,
    HLENS, HLENSW, HLENNW, HANGL, HANIS, HSLOP, HZMAX (from init-gwd-orog); CROT, SROT (from
    phys-gwd-init); KPBL (int32 [JM, IM]); U, V, T, Q, DP, PMID, EXNR ([LM, JM, IM]); PINT, Z
    ([LM+1, JM, IM]).
Out (ctx.st): DUDT_GWD, DVDT_GWD ([LM, JM, IM]), zeroed for skipped columns as the Fortran does.
Scratch (ctx.st, added if absent, rewritten every call): UGWDsfc, VGWDsfc ([JM, IM]).

Scalars: DTPHS = NPHS * ctx.dt; CDMB 0.1, CLEFFAMP 1.0, CLEFF is the device scalar ctx.tables['gwd']['cleff'] written by phys-gwd-init (not recomputed here), SIGFAC 3.0, FACTOP 0.5, RLOLEV 50, DPMIN 0, IMX = IM - 1.
No host sync and no per-step allocation after the first call.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_gwd_driver"
KERNEL = "phys_gwd_driver"
NPHS = 2
LM_KERNEL = 60            # GWD_LM_MAX in the kernel: per-thread column arrays hold 60 levels
CDMB = 0.1
SIGFAC = 3.0
FACTOP = 0.5
RLOLEV = 50.0
DPMIN = 0.0

# GWD_DRIVER is called only from TURBL (module_TURBULENCE.F90:1291-1305) with the TURBL working
# arrays: U=U_PHY, V=V_PHY (H-point winds), Z=TURBL Z, DP=DELP, PINT=PHINT, PMID=PHMID, EXNR=EXNER,
# KPBL=LPBL. These are the names phys-turbl-driver writes in ctx.st.
ARGS = {"U": "U_PHY", "V": "V_PHY", "T": "T", "Q": "Q", "DP": "DELP", "PMID": "PHMID", "EXNR": "EXNER",
        "PINT": "PHINT", "Z": "TURBL_Z", "KPBL": "LPBL"}
_MASS3 = ("U", "V", "T", "Q", "DP", "PMID", "EXNR")
_INTF3 = ("PINT", "Z")
_TERRAIN2 = ("HSTDV", "HCNVX", "HASYW", "HASYS", "HASYSW", "HASYNW", "HLENW", "HLENS",
             "HLENSW", "HLENNW", "HANGL", "HANIS", "HSLOP", "HZMAX", "CROT", "SROT")


class _Args:
    """ctx.st seen through the GWD_DRIVER dummy-argument names (ARGS)."""

    def __init__(self, st):
        self.st = st

    def __getitem__(self, name):
        return self.st[ARGS.get(name, name)]

    def add(self, name, shape):
        return self.st.add(name, shape)


def run(ctx):
    st = _Args(ctx.st)
    im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
    if lm != LM_KERNEL:
        raise ValueError(f"phys-gwd-driver kernel holds LM={LM_KERNEL} per column, got {lm}")
    shape2 = (jm, im)
    shape3 = (lm, jm, im)
    for name in _TERRAIN2:
        if tuple(st[name].shape) != shape2:
            raise ValueError(f"{name} shape {st[name].shape}, expected {shape2}")
    for name in _MASS3:
        if tuple(st[name].shape) != shape3:
            raise ValueError(f"{name} shape {st[name].shape}, expected {shape3}")
    for name in _INTF3:
        if tuple(st[name].shape) != (lm + 1, jm, im):
            raise ValueError(f"{name} shape {st[name].shape}, expected {(lm + 1, jm, im)}")
    kpbl = st["KPBL"]
    if kpbl.dtype != np.int32 or tuple(kpbl.shape) != shape2:
        raise ValueError(f"KPBL dtype {kpbl.dtype} shape {kpbl.shape}, expected int32 {shape2}")
    # Written by the turbl driver's GWD call; checked here so a missing or mis-shaped
    # tendency fails loudly instead of being read as garbage.
    for name in ("DUDT_GWD", "DVDT_GWD"):
        if tuple(st[name].shape) != shape3:
            raise ValueError(f"{name} shape {st[name].shape}, expected {shape3}")

    tbl = ctx.tables.get("gwd") if ctx.tables else None
    if tbl is None or "cleff" not in tbl:
        raise RuntimeError("phys-gwd-driver needs phys-gwd-init first (ctx.tables['gwd']['cleff'])")
    cleff_dev = tbl["cleff"]

    ugwd = st.add("UGWDsfc", shape2)
    vgwd = st.add("VGWDsfc", shape2)

    dtphs = np.float32(NPHS * ctx.dt)
    imx = np.int32(im - 1)                 # GLOBAL = false: IMX = IDE - 1 (IDE == im, see note)

    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
        st["DUDT_GWD"], st["DVDT_GWD"],
        st["U"], st["V"], st["T"], st["Q"], st["DP"], st["PINT"], st["PMID"], st["EXNR"], st["Z"],
        st["KPBL"],
        st["HSTDV"], st["HCNVX"], st["HASYW"], st["HASYS"], st["HASYSW"], st["HASYNW"],
        st["HLENW"], st["HLENS"], st["HLENSW"], st["HLENNW"], st["HANGL"], st["HANIS"],
        st["HSLOP"], st["HZMAX"], st["CROT"], st["SROT"],
        ugwd, vgwd,
        dtphs, np.float32(CDMB), cleff_dev, np.float32(SIGFAC), np.float32(FACTOP),
        np.float32(RLOLEV), np.float32(DPMIN),
        imx, np.int32(im), np.int32(jm), np.int32(lm),
        # single tile = the whole domain: ITS=IDS..ITE=IDE, JTS=JDS..JTE=JDE (solver is one tile)
        np.int32(0), np.int32(im - 1), np.int32(0), np.int32(jm - 1)))
