# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-sw-rswinit: RSWINIT, NMMB RRTM shortwave initialisation (transmittance table).

Ported from phys/radsw_main_nmmb.f:1244-1395 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE rswinit.
Kernel: namgpu/kernels/phys_rrtm_sw_rswinit.cu (rswinit_tables, one launch, NTBMX+1 threads).

What the routine does, and where it lives now:
  - exp_tbl(0:NTBMX): ctx.tables["rrtm_sw_exp_tbl"] (NTBMX+1 = 10001 entries). Arithmetic in double
    (kind_phys = 8 in the reference build, see the kernel header), stored float32 for the consumers.
  - heatfac: ctx.tables["rrtm_sw_heatfac"] (float32[1], double arithmetic); k/second unless iswrate == 1.
  - Control flags (iovrsw, isubcsw, icldflg, iswcliq, iswrate) are checked on the host here, with the
    physparam.f defaults when ctx.cfg does not set them: iovrsw 1 (:143), isubcsw 0 (:165),
    icldflg 1 (:141), iswcliq 1 (:53), iswrate 2 (:49). The flags are written to
    ctx.tables["rrtm_sw_ctl"] = int32[4] = [iovrsw, isubcsw, icldflg, iswcliq].
    The Fortran STOPs on a bad flag; the wrapper raises ValueError for the same cases.

In:    ctx.cfg (optional attributes iovrsw, isubcsw, icldflg, iswcliq, iswrate).
Out:   the ctx.tables entries above. Tables are allocated once and reused.
Launch: one kernel launch on ctx.stream, no host sync inside the kernel, no per-step allocation.

Not done here: the Fortran print statements (host diagnostics only) and the me argument (used only by
those prints). Single-card run: no MPI.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_radinit import host_ctl

UNIT = "phys_rrtm_sw_rswinit"
KERNEL = "rswinit_tables"
NTBMX = 10000            # radsw_param.f:113

# physparam.f defaults
IOVRSW_DEFAULT = 1       # :143
ISUBCSW_DEFAULT = 0      # :165
ISWCLIQ_DEFAULT = 1      # :53
ISWRATE_DEFAULT = 2      # :49


def _flags(cfg, icldflg: int) -> tuple[int, int, int, int, int]:
    iovrsw = int(getattr(cfg, "iovrsw", IOVRSW_DEFAULT))
    isubcsw = int(getattr(cfg, "isubcsw", ISUBCSW_DEFAULT))
    icldflg = int(icldflg)   # radinit_nmmb value as seen by rswinit (grrad_nmmb.f:424-428 for icmphys 5)
    iswcliq = int(getattr(cfg, "iswcliq", ISWCLIQ_DEFAULT))
    iswrate = int(getattr(cfg, "iswrate", ISWRATE_DEFAULT))
    # rswinit checks: overlap flag (:1317-1321), sub-column flag (:1334-1357, inside the me == 0
    # block, but the STOP applies to every run that gets there), cloud-flag consistency (:1361-1366).
    if iovrsw < 0 or iovrsw > 2:
        raise ValueError(f"rswinit: cloud overlap flag IOVRSW={iovrsw} out of range 0..2 (Fortran STOPs)")
    if isubcsw not in (0, 1, 2):
        raise ValueError(f"rswinit: sub-column flag ISUBCSW={isubcsw} not 0, 1 or 2 (Fortran STOPs)")
    if (icldflg == 0 and iswcliq != 0) or (icldflg == 1 and iswcliq == 0):
        raise ValueError(f"rswinit: ICLDFLG={icldflg} and ISWCLIQ={iswcliq} inconsistent (Fortran STOPs)")
    return iovrsw, isubcsw, icldflg, iswcliq, iswrate


def _alloc(ctx):
    import cupy as cp

    t = ctx.tables
    if "rrtm_sw_exp_tbl" not in t:
        t["rrtm_sw_exp_tbl"] = cp.zeros(NTBMX + 1, dtype=cp.float32)
        t["rrtm_sw_heatfac"] = cp.zeros(1, dtype=cp.float32)
        t["rrtm_sw_ctl"] = cp.zeros(4, dtype=cp.int32)
    return t


def run(ctx):
    import cupy as cp
    import numpy as np

    iovrsw, isubcsw, icldflg, iswcliq, iswrate = _flags(ctx.cfg, host_ctl(ctx)["icldflg_rad"])
    t = _alloc(ctx)
    t["rrtm_sw_ctl"][:] = cp.asarray(np.array([iovrsw, isubcsw, icldflg, iswcliq], dtype=np.int32))

    k = kernels.get(UNIT, KERNEL)
    block = 256
    grid = (NTBMX + 1 + block - 1) // block
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((grid,), (block,), (t["rrtm_sw_exp_tbl"], t["rrtm_sw_heatfac"], np.int32(iswrate)))
    return t


if __name__ == "__main__":
    # Standalone smoke: one launch, no state needed. Run via tools/check_unit.py for the done rule.
    ctx = SimpleNamespace(cfg=SimpleNamespace(), tables={}, stream=None)
    out = run(ctx)
    print({k: v.shape for k, v in out.items()})
