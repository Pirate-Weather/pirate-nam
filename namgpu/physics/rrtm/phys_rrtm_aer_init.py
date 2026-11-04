"""phys-rrtm-aer-init: aer_init of the NMMB radiation aerosol module (NAM nam.v4.2.11, 83df4e13c1).

Source: phys/radiation_aerosols_nmmb.f:445-816 (aer_init, wrt_aerlog, set_spectrum, set_volcaer).
Kernel: namgpu/kernels/phys_rrtm_aer_init.cu (aer_init_flags, aer_init_tables, aer_init_solfwv,
aer_init_eirfwv). The kernel file header carries the line-by-line map and the NOAA notice.

In:   ctx.tables['rad_physparam'] (int32[26]) written by phys-rrtm-rad-initialize. Must exist: rad_initialize_nmmb
      precedes aer_init in the Fortran (PHYSICS_INITIALIZE, then grrad_nmmb.f:405). Missing -> RuntimeError.
Out:  ctx.tables['rrtm_aer_init'], a dict of device arrays:
        "aflags"  int32 [N_AF]: scalar module state and guards, slots AF_* below.
        "solfwv"  float32 [NWVTOT = 57600]   one-wavenumber solar flux (set_spectrum)
        "eirfwv"  float32 [NWVTIR = 4000]    one-wavenumber IR flux at T0C (set_spectrum)
        "nwvns0"  int32   [NWVSOL = 151]     wavenumbers per solar region (DATA table)
        "s0intv"  float32 [NWVSOL = 151]     solar flux per region (DATA table)
        "ivolae"  int32   [12, 4, 10]        set_volcaer allocation (zeroed here)
      solfwv/eirfwv are locals of aer_init in the Fortran, consumed by clim_aerinit. That unit is not
      in this file; it reads them from this dict.
Launches: four kernels on ctx.stream, once per forecast start. No host sync inside run().

Status: the Fortran `stop` paths are recorded in aflags[AF_STATUS]; check_status() reads it and raises.
That read is a host sync, so the caller runs it once at init, after the stream work, never per step.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_aer_init"
KEY = "rrtm_aer_init"
RAD_KEY = "rad_physparam"           # ctx.tables key written by phys-rrtm-rad-initialize

NWVSOL = 151                       # radiation_aerosols_nmmb.f:158
NWVTOT = 57600                     # :160
NWVTIR = 4000                      # :161
N_AF = 10

# aflags slots (match the AF_* defines in the kernel)
AF_NSWBND, AF_NLWBND, AF_NSWLWBD = 0, 1, 2
AF_DO_SPECTRUM, AF_DO_VOLCAER, AF_STATUS = 3, 4, 5
AF_KYRSTR, AF_KYREND, AF_KYRSAV, AF_KMONSAV = 6, 7, 8, 9

# rad_physparam slots used here (phys_rrtm_rad_initialize.py RI_*)
RI_IAERFLG, RI_IAERMDL = 4, 8

STATUS_TEXT = {
    1: "wrt_aerlog: aerosol scheme IAER_MDL outside 0..2 (radiation_aerosols_nmmb.f wrt_aerlog stop)",
    2: "aer_init: aerosol scheme selection stop (iaerflg /= 0,100 with iaermdl /= 0; radiation_aerosols_nmmb.f:563-569)",
}


def _table(ctx):
    import cupy as cp

    if KEY not in ctx.tables:
        ctx.tables[KEY] = {
            "aflags": cp.zeros(N_AF, dtype=np.int32),
            "solfwv": cp.zeros(NWVTOT, dtype=np.float32),
            "eirfwv": cp.zeros(NWVTIR, dtype=np.float32),
            "nwvns0": cp.zeros(NWVSOL, dtype=np.int32),
            "s0intv": cp.zeros(NWVSOL, dtype=np.float32),
            "ivolae": cp.zeros((12, 4, 10), dtype=np.int32),
        }
    return ctx.tables[KEY]


def run(ctx):
    rip = ctx.tables.get(RAD_KEY)
    if rip is None:
        raise RuntimeError(
            "phys-rrtm-aer-init: ctx.tables['rad_physparam'] is missing; run phys-rrtm-rad-initialize first "
            "(rad_initialize_nmmb precedes aer_init, grrad_nmmb.f:405)")
    tbl = _table(ctx)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    k_flags = kernels.get(UNIT, "aer_init_flags")
    k_tabs = kernels.get(UNIT, "aer_init_tables")
    k_sol = kernels.get(UNIT, "aer_init_solfwv")
    k_eir = kernels.get(UNIT, "aer_init_eirfwv")
    with scope:
        k_flags((1, 1, 1), (1, 1, 1), (rip, tbl["aflags"]))
        k_tabs(((NWVSOL + 127) // 128, 1, 1), kernels.BLOCK_COL, (tbl["nwvns0"], tbl["s0intv"]))
        k_sol(((NWVTOT + 127) // 128, 1, 1), kernels.BLOCK_COL, (tbl["solfwv"], tbl["aflags"]))
        k_eir(((NWVTIR + 127) // 128, 1, 1), kernels.BLOCK_COL, (tbl["eirfwv"], tbl["aflags"]))
    return tbl


def check_status(ctx):
    """Host read of the scalar outputs (one sync). Raises on a Fortran stop path. Call at init only."""
    tbl = ctx.tables[KEY]
    a = tbl["aflags"].get()
    status = int(a[AF_STATUS])
    if status:
        raise RuntimeError(f"phys-rrtm-aer-init: {STATUS_TEXT.get(status, 'status ' + str(status))}")
    return {
        "nswbnd": int(a[AF_NSWBND]), "nlwbnd": int(a[AF_NLWBND]), "nswlwbd": int(a[AF_NSWLWBD]),
        "do_spectrum": bool(a[AF_DO_SPECTRUM]), "do_volcaer": bool(a[AF_DO_VOLCAER]),
        "kyrstr": int(a[AF_KYRSTR]), "kyrend": int(a[AF_KYREND]),
        "kyrsav": int(a[AF_KYRSAV]), "kmonsav": int(a[AF_KMONSAV]),
    }
