"""drv-tracer-index: slot indices of the NMMB TRACERS array, as SOLVER_INITIALIZE sets them.

Fortran: nmm/module_SOLVER_INTERNAL_STATE.F90:864-953 (microphysics table, NUM_TRACERS_MET/TOTAL),
nmm/module_SOLVER_INTERNAL_STATE.F90:1379-1453 (INDX_Q, INDX_CW, INDX_Q2, water slots),
nmm/module_VARS_STATE.F90:797 (INDX_Q2 = 3). NAM nam.v4.2.11, 83df4e13c1.
Kernel: namgpu/kernels/drv_tracer_index.cu (tracer_index_build, one thread, one launch).

Reads:   cfg.microphysics (parm/nam_configfile `microphysics`, fer_hires for the 12 km CONUS parent),
         cfg.raw["num_tracers_chem"] (0 for the parent).
Writes:  ctx.tables["indx"]     dict of ints: INDX_Q, INDX_CW, INDX_Q2, INDX_QC, INDX_QI, INDX_QR,
                                INDX_QS, INDX_QG, INDX_NI, INDX_NR, INDX_WATER_START,
                                INDX_WATER_END, NUM_TRACERS_TOTAL, NUM_TRACERS_MET, NUM_TRACERS_CHEM,
                                NUM_WATER (host copy, the Fortran values)
         ctx.tables["INDX_DEV"]  int32 device array [13] written by the kernel (same numbers, same order
                                 as the kernel header), for device-side consumers. INDX_DEV holds
                                 FORTRAN 1-BASED slot numbers (INDX_Q = 1). The TRACERS array is zero-based
                                 in its slot axis, so a device consumer indexing TRACERS[n] must use
                                 INDX_DEV[k] - 1 (the host path does the same in init_derive_state._tracer_slots).
Idempotent: built on the first call and reused. One launch on ctx.stream, no host sync.

NUM_TRACERS_MET is 3 here. The Fortran sets it unconditionally at module_SOLVER_INTERNAL_STATE.F90:943
(NUM_TRACERS_MET=3), so the configfile value num_tracers_met: 4 is read but not used for the slots.
Unknown microphysics names stop the Fortran (SOLVER_INTERNAL_STATE, `Unknown microphysics`), and this
wrapper refuses them the same way.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_tracer_index"
KERNEL = "tracer_index_build"

# module_SOLVER_INTERNAL_STATE.F90:880-938: NUM_WATER and the P_* water positions per scheme.
MICROPHYSICS_WATER = {
    "fer":      dict(nw=4, qc=1, qi=0, qr=2, qs=3, qg=4, ni=0, nr=0),
    "fer_hires": dict(nw=4, qc=1, qi=0, qr=2, qs=3, qg=4, ni=0, nr=0),
    "wsm6":     dict(nw=5, qc=1, qi=4, qr=2, qs=3, qg=5, ni=0, nr=0),
    "thompson": dict(nw=7, qc=1, qi=2, qr=3, qs=4, qg=5, ni=6, nr=7),
    "gfs":      dict(nw=2, qc=1, qi=2, qr=0, qs=0, qg=0, ni=0, nr=0),
}
NUM_TRACERS_MET = 3       # module_SOLVER_INTERNAL_STATE.F90:943
INDX_Q, INDX_CW, INDX_Q2 = 1, 2, 3   # module_SOLVER_INTERNAL_STATE.F90:1382,1391,1400; VARS_STATE:797
SLOT_NAMES = ("INDX_Q", "INDX_CW", "INDX_Q2", "INDX_QC", "INDX_QI", "INDX_QR", "INDX_QS",
              "INDX_QG", "INDX_NI", "INDX_NR", "INDX_WATER_START", "INDX_WATER_END",
              "NUM_TRACERS_TOTAL")


def micro_table(microphysics: str) -> dict:
    key = microphysics.rstrip()   # Fortran TRIM(): trailing blanks only; a leading blank is an unknown name
    if key not in MICROPHYSICS_WATER:
        raise ValueError(f"unknown microphysics {key!r}: the Fortran stops here "
                         "(module_SOLVER_INTERNAL_STATE.F90 Unknown microphysics)")
    return MICROPHYSICS_WATER[key]


def host_index(microphysics: str, num_tracers_chem: int) -> dict:
    """Pure-host mirror of the kernel (used for ctx.tables[indx] and for the unit test)."""
    w = micro_table(microphysics)
    met, chem, nw = NUM_TRACERS_MET, int(num_tracers_chem), w["nw"]
    wstart = met + chem + 1
    out = {"INDX_Q": INDX_Q, "INDX_CW": INDX_CW, "INDX_Q2": INDX_Q2,
           "INDX_WATER_START": wstart, "INDX_WATER_END": wstart + nw - 1,
           "NUM_TRACERS_TOTAL": met + chem + nw,
           "NUM_TRACERS_MET": met, "NUM_TRACERS_CHEM": chem, "NUM_WATER": nw}
    for name, p in (("INDX_QC", w["qc"]), ("INDX_QI", w["qi"]), ("INDX_QR", w["qr"]),
                    ("INDX_QS", w["qs"]), ("INDX_QG", w["qg"]), ("INDX_NI", w["ni"]),
                    ("INDX_NR", w["nr"])):
        out[name] = (wstart - 1 + p) if p > 0 else 0
    return out


def run(ctx):
    import cupy as cp

    tables = ctx.tables
    if "indx" in tables:
        return
    micro = ctx.cfg.microphysics
    chem = int(ctx.cfg.raw["num_tracers_chem"])
    w = micro_table(micro)
    params = np.array([NUM_TRACERS_MET, chem, w["nw"], w["qc"], w["qi"], w["qr"], w["qs"],
                       w["qg"], w["ni"], w["nr"]], dtype=np.int32)

    p_dev = cp.asarray(params)
    indx_dev = cp.zeros((13,), dtype=cp.int32)
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern((1, 1, 1), (1, 1, 1), (p_dev, indx_dev))

    tables["INDX_DEV"] = indx_dev
    tables["indx"] = host_index(micro, chem)
