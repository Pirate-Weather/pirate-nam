"""phys-rrtm-grrad-setup: argument, layer-array and unit-conversion setup of NMMB grrad_nmmb.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/grrad_nmmb.f:660-1396 (subroutine grrad_nmmb,
from the subroutine header through the local-array setup, before the getozn call). NOAA US
government work; the source header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_rrtm_grrad_setup.cu (rrtm_grrad_setup, one thread per column,
grid_cols, one launch, no host sync, no per-step allocation after the first call).

In (CONTRACT names, full domain, Pa for pressures):
    PINT [LM+1, JM, IM]   interface pressure           (Fortran prsi, converted to cb in the kernel)
    P_PHY [LM, JM, IM]    layer mean pressure          (Fortran prsl)
    PRSLK [LM, JM, IM]    Exner function               (Fortran prslk)
    T, Q, CLDFRA          tgrs, qgrs, cldf
    CW [LM, JM, IM]       total condensate             (Fortran GR1(:,:,3), module_RA_RRTM.F90:957)
    O3 [LM, JM, IM]       ozone, read only when ntoz > 0 (Fortran GR1(:,:,1), module_RA_RRTM.F90:951)
    TRACERS [NTRAC_STATE, LM, JM, IM]  read for the Thompson hydrometeor slots (INDX_QC, QI, QS, QR,
                          QG, NI) through the drv-tracer-index slots when microphysics is thompson
    F_ICE, F_RAIN, F_RIMEF  fcice, frain, rrime (used when np3d == 3)
    TSKIN [JM, IM]        skin temperature = the Fortran TSEA argument (module_RA_RRTM.F90:905);
                          bound by default; an explicit tsfc argument overrides it
Out (scratch, added once with st.add, names as below; the next units read them):
    RAD_PLVL [LM+2, JM, IM] mb, RAD_PLYR, RAD_TLYR, RAD_PRSLK1, RAD_RHLY, RAD_QSTL, RAD_CLDFK1
    [LM+1, JM, IM], RAD_TRACER1 [9, LM+1, JM, IM], RAD_GCICE, RAD_GRAIN, RAD_GRIME [LM+1, JM, IM]
    RAD_OLYR [LM+1, JM, IM] (ntoz > 0 only), RAD_TSKN, RAD_TSFG [JM, IM]
    RAD_QLYR, RAD_TVLY [LM+1, JM, IM] (Fortran qlyr, tvly; top-to-surface moisture floor recurrence)
Returned: dict of those arrays plus 'raddt' = min(dtsw, dtlw) (Fortran raddt, host scalar).
Filled here every call: P_PHY = SGML2*PD + PSGML1 and PRSLK = (P_PHY/1e5)^(R/CP) with R = 287.04
and CP = 1004.6 (NAM module_CONSTANTS.F90, the values the caller uses at module_RA_RRTM.F90:934).

The GR1 tracer array (Fortran NTRAC = 9, module_RA_RRTM.F90:218) is rebuilt per call in the Fortran
slot order: slot 1 = max(O3, 1e-10) when ntoz > 0, slot 2 = 0, slot 3 = CW, slots 4-9 = Thompson
QC, QI, QS, QR, QG, NI when microphysics is thompson, otherwise 0 (Fortran lines 947-986). Slots
not set are zeroed, as the Fortran zeroing loop does.

Defaults come from the configuration, not from literals: np3d = cfg.np3d, ntoz = cfg.ntoz, and
ivflip = NAM_IFLIP = 0 (module_SOLVER_GRID_COMP.F90:10714, IFLIP=0, input from toa to sfc).
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels
from namgpu.steps.drv_tracer_index import host_index

UNIT = "phys_rrtm_grrad_setup"
KERNEL = "rrtm_grrad_setup"
BLOCK = (128, 1, 1)
NAM_IFLIP = 0                # module_SOLVER_GRID_COMP.F90:10714 (IFLIP=0, input from toa to sfc)
GR1_NTRAC = 9                # module_RA_RRTM.F90:218 NTRAC=9 (GR1 slots passed to grrad_nmmb)
EPSO3 = np.float32(1.0e-10)  # module_RA_RRTM.F90 EPSO3 (line 951, MAX(O3, EPSO3))
CON_R = np.float32(287.04)   # module_CONSTANTS.F90: r = r_d = 287.04
CON_CP = np.float32(1004.6)  # module_CONSTANTS.F90: cp = 1004.6


def _gr1(ctx, st, lm, jm, im, ntoz: int):
    """Build the GR1 tracer array in the Fortran slot order: zeroed, then the slots the caller sets."""
    import cupy as cp

    gr1 = st.add("RAD_GR1", (GR1_NTRAC, lm, jm, im))
    gr1[...] = 0.0
    if ntoz > 0:                                             # module_RA_RRTM.F90:951
        gr1[0] = cp.maximum(st["O3"], EPSO3)
    gr1[2] = st["CW"]                                        # module_RA_RRTM.F90:957, total condensate
    if ctx.cfg.microphysics.rstrip() == "thompson":          # module_RA_RRTM.F90:966-986 (F_NI)
        tab = ctx.tables.get("indx")
        if tab is None:
            tab = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
        tr = st["TRACERS"]
        gr1[3] = tr[int(tab["INDX_QC"]) - 1]
        gr1[4] = tr[int(tab["INDX_QI"]) - 1]
        gr1[5] = tr[int(tab["INDX_QS"]) - 1]
        gr1[6] = tr[int(tab["INDX_QR"]) - 1]
        gr1[7] = tr[int(tab["INDX_QG"]) - 1]
        gr1[8] = tr[int(tab["INDX_NI"]) - 1]
    return gr1


def run(ctx, *, dtsw: float, dtlw: float, ivflip: int | None = None, np3d: int | None = None,
        ntoz: int | None = None, tsfc=None):
    """Run the setup over every column. Returns the dict of RAD_* arrays and 'raddt'."""
    import cupy as cp

    st, g, cfg = ctx.st, ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    lmk = lm + 1
    lmp = lm + 2
    s3 = (lm, jm, im)
    ivflip = NAM_IFLIP if ivflip is None else int(ivflip)
    np3d = int(cfg.np3d) if np3d is None else int(np3d)
    ntoz = int(cfg.ntoz) if ntoz is None else int(ntoz)

    # layer pressure and Exner function from the current PD, every call (module_RA_RRTM.F90:486-487,
    # 934); two elementwise device ops, no host work. Scratch is allocated once by st.add.
    ppy = st.add("P_PHY", s3)
    sgml2 = cp.asarray(st["SGML2"]).reshape(lm, 1, 1)
    psgml1 = cp.asarray(g.psgml1).reshape(lm, 1, 1)
    ppy[...] = sgml2 * st["PD"][None] + psgml1
    prslk = st.add("PRSLK", s3)
    prslk[...] = (ppy / np.float32(1.0e5)) ** (CON_R / CON_CP)
    if tsfc is None:
        if "TSKIN" not in st.fields:
            raise RuntimeError("phys-rrtm-grrad-setup: TSKIN is not in the state; the Fortran caller "
                               "passes TSEA=TSKIN (module_RA_RRTM.F90:905), so pass tsfc explicitly "
                               "or add TSKIN to the state")
        tsfc = st["TSKIN"]

    gr1 = _gr1(ctx, st, lm, jm, im, ntoz)
    out = {
        "RAD_PLVL": st.add("RAD_PLVL", (lmp, jm, im)),
        "RAD_PLYR": st.add("RAD_PLYR", (lmk, jm, im)),
        "RAD_TLYR": st.add("RAD_TLYR", (lmk, jm, im)),
        "RAD_PRSLK1": st.add("RAD_PRSLK1", (lmk, jm, im)),
        "RAD_RHLY": st.add("RAD_RHLY", (lmk, jm, im)),
        "RAD_QSTL": st.add("RAD_QSTL", (lmk, jm, im)),
        "RAD_CLDFK1": st.add("RAD_CLDFK1", (lmk, jm, im)),
        "RAD_TRACER1": st.add("RAD_TRACER1", (GR1_NTRAC, lmk, jm, im)),
        "RAD_GCICE": st.add("RAD_GCICE", (lmk, jm, im)),
        "RAD_GRAIN": st.add("RAD_GRAIN", (lmk, jm, im)),
        "RAD_GRIME": st.add("RAD_GRIME", (lmk, jm, im)),
        "RAD_OLYR": st.add("RAD_OLYR", (lmk, jm, im)),
        "RAD_TSKN": st.add("RAD_TSKN", (jm, im)),
        "RAD_TSFG": st.add("RAD_TSFG", (jm, im)),
        "RAD_QLYR": st.add("RAD_QLYR", (lmk, jm, im)),
        "RAD_TVLY": st.add("RAD_TVLY", (lmk, jm, im)),
    }

    k = kernels.get(UNIT, KERNEL)
    args = (
        st["PINT"], ppy, prslk, st["T"], st["Q"], gr1, st["CLDFRA"],
        st["F_ICE"], st["F_RAIN"], st["F_RIMEF"], tsfc,
        out["RAD_PLVL"], out["RAD_PLYR"], out["RAD_TLYR"], out["RAD_PRSLK1"], out["RAD_RHLY"],
        out["RAD_QSTL"], out["RAD_CLDFK1"], out["RAD_TRACER1"], out["RAD_GCICE"],
        out["RAD_GRAIN"], out["RAD_GRIME"], out["RAD_OLYR"], out["RAD_TSKN"], out["RAD_TSFG"],
        out["RAD_QLYR"], out["RAD_TVLY"],
        np.int32(im), np.int32(jm), np.int32(lm), np.int32(GR1_NTRAC),
        np.int32(ivflip), np.int32(np3d), np.int32(ntoz),
    )
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(kernels.grid_cols(im, jm), BLOCK, args)
    out["raddt"] = float(min(dtsw, dtlw))
    return out
