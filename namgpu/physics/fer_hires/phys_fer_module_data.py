"""phys-fer-module-data: MP_FER_HIRES module constants, warn state and table layout.

Ported from phys/module_MP_FER_HIRES.F90:1-207 (NAM nam.v4.2.11, 83df4e13c1). Kernel:
namgpu/kernels/phys_fer_module_data.cu (fer_module_data_publish, fer_warn_state_init, one thread, one launch each).
Header with the constants and the statement map: namgpu/kernels/fer_hires.cuh.

Lines 1-207 are declarations only (no executable statements). What runs on the device:
  - fer_module_data_publish writes the REAL PARAMETER values (REAL_NAMES order) to a float32 vector and the
    INTEGER PARAMETER values (INT_NAMES order) to an int32 vector. Both are constants.
  - fer_warn_state_init writes the initial values of Pwarn and QTwarn (Fortran line 172) to a float32[2] vector.
    Pwarn and QTwarn are mutable SAVE state: the WARN4 path (lines 389-393) rewrites them, so they are runtime
    device state, not constants. The WARN4 path (phys-fer-fer-hires) updates the vector in place.
The table arrays (MDImin:MDImax, MDRmin:MDRmax, MY_GROWTH_NMM, VEL_RF, TBPVS) and the runtime SAVE scalars are
filled by the init unit phys-fer-init, not here.

In:      nothing from the state. Writes ctx.tables[KEY], ctx.tables[KEY_INT], ctx.tables[KEY_WARN].
Out:     float32[len(REAL_NAMES)] constants; int32[len(INT_NAMES)] constants; float32[2] warn state.
Launch:  one launch of each kernel on ctx.stream. The constant kernel runs on every call (idempotent). The warn-state
         kernel runs only when the state vector is created, so a second call never resets a live Pwarn/QTwarn.
         No host sync, no per-step work.

verify(ctx) is a test helper: it copies the device vectors to the host and compares them bit for bit with
host_values() and INTEGER_PARAMETERS. It synchronises and is never called on the step path.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_module_data"
KERNEL = "fer_module_data_publish"
KERNEL_WARN = "fer_warn_state_init"
KEY = "fer_module_data"
KEY_INT = "fer_module_data_int"
KEY_WARN = "fer_warn_state"

# Same order as FER_REAL_CONSTANT_LIST in namgpu/kernels/fer_hires.cuh.
REAL_NAMES = (
    "CP", "EPSQ", "G", "RHOL", "RD", "RV", "T0C", "EP2", "EP1", "CLIQ", "CICE", "PI", "XLV",
    "DMI_MIN", "DMI_MAX", "DELDMI", "XMI_MIN", "XMI_MAX", "XMI_EXP",
    "DMR_MIN", "DMR_MAX", "DELDMR", "XMR_MIN", "XMR_MAX", "XMIN", "XMAX",
    "CVAP", "XLF", "XLS", "EPSQ1", "RCP", "RCPRV", "RGRAV", "RRHOL",
    "XLV1", "XLF1", "XLS1", "XLV2", "XLS2", "XLV3", "XLS3",
    "CLIMIT", "CLIMIT1", "C1", "DMR1", "DMR2", "DMR3", "DMR4", "DMR5",
    "XMR1", "XMR2", "XMR3", "XMR4", "XMR5", "RQRMIX", "RQSMIX", "CDRY", "CWET",
    "T_ICE", "T_ICEK", "T_ICE_INIT", "NSI_MAX", "NLIMIN", "N0R0", "N0RMIN", "NCW",
)

# Same order as FER_INT_CONSTANT_LIST in namgpu/kernels/fer_hires.cuh.
INT_NAMES = (
    "MDI_MIN", "MDI_MAX", "MDR_MIN", "MDR_MAX", "MDR1", "MDR2", "MDR3", "MDR4", "MDR5",
    "NX", "INDEXRSTRMAX", "MY_T1", "MY_T2", "NRIME", "MAX_ITERATIONS",
)

# INTEGER PARAMETERs of the module (Fortran truncation of the REAL parents, checked on node-2 with gfortran 15.2).
INTEGER_PARAMETERS = {
    "MDI_MIN": 50, "MDI_MAX": 1000, "MDR_MIN": 50, "MDR_MAX": 1000,
    "MDR1": 100, "MDR2": 200, "MDR3": 320, "MDR4": 450, "MDR5": 670,
    "NX": 7501, "INDEXRSTRMAX": 500, "MY_T1": 1, "MY_T2": 35, "NRIME": 40, "MAX_ITERATIONS": 10,
}

# Initial values of the mutable SAVE state (Fortran line 172): [0] = Pwarn, [1] = QTwarn.
WARN_STATE_INIT = (np.float32(75.0e2), np.float32(1.0e-3))


def host_values() -> dict[str, np.float32]:
    """Host float32 evaluation, expression by expression, as the Fortran folds it."""
    f = np.float32
    CP = f(1004.6); EPSQ = f(1.0e-12); G = f(9.8060226); RHOL = f(1000.0)
    RD = f(287.04); RV = f(461.6); T0C = f(273.15)
    EP2 = f(f(287.04) / f(461.6))
    EP1 = f(f(RV / RD) - f(1.0))
    CLIQ = f(4190.0); CICE = f(2106.0); PI = f(3.141592653589793); XLV = f(2.5e6)

    DMI_MIN = f(0.05e-3); DMI_MAX = f(1.0e-3); DELDMI = f(1.0e-6)
    XMI_MIN = f(f(1.0e6) * DMI_MIN); XMI_MAX = f(f(1.0e6) * DMI_MAX); XMI_EXP = f(0.0536)
    DMR_MIN = f(0.05e-3); DMR_MAX = f(1.0e-3); DELDMR = f(1.0e-6)
    XMR_MIN = f(f(1.0e6) * DMR_MIN); XMR_MAX = f(f(1.0e6) * DMR_MAX)
    XMIN = f(180.0); XMAX = f(330.0)

    CVAP = f(1846.0); XLF = f(3.3358e5); XLS = f(XLV + XLF)
    EPSQ1 = f(f(1.001) * EPSQ)
    RCP = f(f(1.0) / CP); RCPRV = f(RCP / RV)
    RGRAV = f(f(1.0) / G); RRHOL = f(f(1.0) / RHOL)
    XLV1 = f(XLV / CP); XLF1 = f(XLF / CP); XLS1 = f(XLS / CP)
    XLV2 = f(f(XLV * XLV) / RV); XLS2 = f(f(XLS * XLS) / RV)
    XLV3 = f(f(XLV * XLV) * RCPRV); XLS3 = f(f(XLS * XLS) * RCPRV)

    CLIMIT = f(f(10.0) * EPSQ); CLIMIT1 = f(-CLIMIT); C1 = f(f(1.0) / f(3.0))
    DMR1 = f(0.1e-3); DMR2 = f(0.2e-3); DMR3 = f(0.32e-3); DMR4 = f(0.45e-3); DMR5 = f(0.67e-3)
    XMR1 = f(f(1.0e6) * DMR1); XMR2 = f(f(1.0e6) * DMR2); XMR3 = f(f(1.0e6) * DMR3)
    XMR4 = f(f(1.0e6) * DMR4); XMR5 = f(f(1.0e6) * DMR5)
    RQRMIX = f(0.05e-3); RQSMIX = f(1.0e-3); CDRY = f(1.634e13); CWET = f(f(1.0) / f(0.224))

    T_ICE = f(-40.0); T_ICEK = f(T0C + T_ICE); T_ICE_INIT = f(-12.0)
    NSI_MAX = f(250.0e3); NLIMIN = f(1.0e3); N0R0 = f(8.0e6); N0RMIN = f(1.0e4)
    NCW = f(300.0e6)

    local = dict(locals())
    return {name: local[name] for name in REAL_NAMES}


def run(ctx):
    import cupy as cp

    if KEY not in ctx.tables:
        ctx.tables[KEY] = cp.zeros(len(REAL_NAMES), dtype=cp.float32)
        ctx.tables[KEY_INT] = cp.zeros(len(INT_NAMES), dtype=cp.int32)
    if KEY_WARN not in ctx.tables:
        ctx.tables[KEY_WARN] = cp.zeros(2, dtype=cp.float32)
        first_warn = True
    else:
        first_warn = False
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, KERNEL)((1, 1, 1), (1, 1, 1), (ctx.tables[KEY], ctx.tables[KEY_INT]))
        if first_warn:
            kernels.get(UNIT, KERNEL_WARN)((1, 1, 1), (1, 1, 1), (ctx.tables[KEY_WARN],))
    return ctx.tables[KEY]


def verify(ctx) -> list[str]:
    """Test helper (synchronises). Returns the constants whose device value differs from the host value."""
    import cupy as cp

    bad = []
    got = cp.asnumpy(ctx.tables[KEY])
    want = host_values()
    for idx, name in enumerate(REAL_NAMES):
        if not (np.float32(got[idx]).tobytes() == want[name].tobytes()):
            bad.append(f"{name}: device {got[idx]!r} host {want[name]!r}")
    got_i = cp.asnumpy(ctx.tables[KEY_INT])
    for idx, name in enumerate(INT_NAMES):
        if int(got_i[idx]) != INTEGER_PARAMETERS[name]:
            bad.append(f"{name}: device {int(got_i[idx])} host {INTEGER_PARAMETERS[name]}")
    return bad


def verify_warn_state_initial(ctx) -> list[str]:
    """Test helper (synchronises). Checks the state vector holds the Fortran initial values (before any WARN4 update)."""
    import cupy as cp

    got = cp.asnumpy(ctx.tables[KEY_WARN])
    bad = []
    for idx, name in enumerate(("Pwarn", "QTwarn")):
        if got[idx].tobytes() != WARN_STATE_INIT[idx].tobytes():
            bad.append(f"{name}: device {got[idx]!r} initial {WARN_STATE_INIT[idx]!r}")
    return bad
