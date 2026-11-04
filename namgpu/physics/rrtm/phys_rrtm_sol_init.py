"""phys-rrtm-sol-init: solar-constant selection of NMMB sol_init (RRTM group).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_astronomy_nmmb.f:113-259, sol_init_nmmb.
Kernel: namgpu/kernels/phys_rrtm_sol_init.cu (sol_init_nmmb, one thread, one launch).

In:    isolar (physparam isolar, default 0), solar_dir (directory the Fortran inquire() runs in;
       default the working directory, as in NMMB). The table file is checked on the host because
       inquire() is host I/O; it is needed only for isolar 1..4.
Out:   ctx.tables[sol_solc0]   float64[1]  solar constant solc0 (W/m2), kind_phys
       ctx.tables[sol_isolflg] int32[1]    solar constant scheme control flag isolflg
       run() returns (solc0, isolflg, solar_fname) where solar_fname is the table name the Fortran
       would open (host string, for the print/diagnostic path).
Next:  phys-rrtm-sol-update (sol_update_nmmb) reads isolflg and solc0 each radiation call and
       refreshes them from the table. That unit is not run here.
Scratch: the two one-element outputs, allocated once on first call.
Launch: one kernel launch on ctx.stream, no host sync. Runs once at initialisation.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_sol_init"
KERNEL = "sol_init_nmmb"
KEY_SOLC0 = "sol_solc0"
KEY_ISOLFLG = "sol_isolflg"

SOLAR_FILE = "solarconstant_noaa_a0.txt"  # physparam.f:106 default solar_file (26 chars with padding)
# solar_fname(15:26) replacement per isolar (radiation_astronomy_nmmb.f:169, 188, 205, 221)
SOLAR_TABLE = {1: "noaa_a0.txt", 2: "noaa_an.txt", 3: "cmip_an.txt", 4: "cmip_mn.txt"}


def solar_fname(isolar: int) -> str:
    """Table file name the Fortran sets for this isolar (solar_fname(15:26) overwrite)."""
    if isolar in SOLAR_TABLE:
        return SOLAR_FILE[:14] + SOLAR_TABLE[isolar]
    return SOLAR_FILE


def run(ctx, isolar: int = 0, solar_dir: str | None = None):
    import cupy as cp

    if KEY_SOLC0 not in ctx.tables:
        ctx.tables[KEY_SOLC0] = cp.zeros(1, dtype=cp.float64)
        ctx.tables[KEY_ISOLFLG] = cp.zeros(1, dtype=cp.int32)
    solc0 = ctx.tables[KEY_SOLC0]
    isolflg = ctx.tables[KEY_ISOLFLG]

    name = solar_fname(isolar)
    exists = 1
    if isolar in SOLAR_TABLE:
        base = solar_dir if solar_dir is not None else os.getcwd()
        exists = int(os.path.isfile(os.path.join(base, name.rstrip())))

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1), (np.int32(isolar), np.int32(exists), solc0, isolflg))
    return solc0, isolflg, name
