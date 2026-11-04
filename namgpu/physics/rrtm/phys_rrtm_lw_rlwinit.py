# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# RRTMG radiation: Copyright (c) 2002-2020, Atmospheric and Environmental Research, Inc. (AER).
# BSD-3-Clause; see NOTICE. NCEP's modifications (NMMB RRTM package, NAM nam.v4.2.11) are US
# Government work (NOAA/NCEP/EMC). The WOOF-NAM port of this file is Apache-2.0.
"""phys-rrtm-lw-rlwinit: RLWINIT, NMMB RRTM longwave initialisation (lookup tables).

Ported from phys/radlw_main_nmmb.f:1145-1339 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE rlwinit.
Precision: the kernel is single (CONTRACT kfpt=single). The gate0 NMMB build does not define SINGLE_PREC
(kind_phys = 8); see the kernel header. This is a declared divergence pending the project lead's ruling.
Kernel: namgpu/kernels/phys_rrtm_lw_rlwinit.cu (rlwinit_tables, one launch, NTBL+1 threads).

What the routine does, and where it lives now:
  - semiss0, fluxfac, heatfac: device array ctx.tables["rrtm_lw_consts"] (pival, fluxfac, heatfac) and
    ctx.tables["rrtm_lw_semiss0"].
  - tau_tbl, exp_tbl, tfn_tbl (NTBL+1 = 10001 entries each): ctx.tables["rrtm_lw_tau_tbl"],
    ctx.tables["rrtm_lw_exp_tbl"], ctx.tables["rrtm_lw_tfn_tbl"].
  - Control flags after the Fortran checks (iovrlw, isubclw, icldflg, ilwcliq, ilwrate) are scalars and
    are checked on the host here: ctx.tables["rrtm_lw_ctl"] = int32[4] = [iovrlw, isubclw, icldflg,
    ilwcliq] after the iovrlw = 2 -> 1 adjustment when isubclw == 0 (radlw_main_nmmb.f:1229-1241).
    The Fortran STOPs on a bad flag; here the wrapper raises ValueError for the same cases so a run
    never continues with flags the Fortran would have refused.

In:    ctx.cfg (optional attributes iovrlw, isubclw, icldflg, ilwcliq, ilwrate). Defaults are the
       physparam.f initialisers (save variables): iovrlw 1 (:144), isubclw 0 (:166), icldflg 1 (:141),
       ilwcliq 1 (:75), ilwrate 2 (:71). Nothing in the NMMB source assigns these outside physparam.f and
       the RLWINIT checks, so these are the values the NAM run uses; the NAM namelist (nam-gh/parm) has no
       key for them. NAM sets these through rad_initialize_nmmb; the NAM 12 km configfile overrides
       are not yet wired into NamConfig, so the defaults apply until the grrad-level unit passes them.
Out:   the ctx.tables entries above, plus ctx.tables["rrtm_lw_ctl_host"] (python dict with the adjusted
       iovrlw, isubclw, icldflg, ilwcliq, ilwrate). Consumers read that dict, never ctx.cfg. Tables are allocated once and reused.
Launch: one kernel launch on ctx.stream, no host sync inside the kernel, no per-step allocation.

Not done here: the Fortran print statements (diagnostic, host only), and the me argument (only used for
those prints). Single-card run: no MPI.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_radinit import host_ctl

UNIT = "phys_rrtm_lw_rlwinit"
KERNEL = "rlwinit_tables"
NTBL = 10000
NBANDS = 16
CON_G = 9.80665          # physcons.f:46
CON_CP = 1.0046e+3       # physcons.f:59

# physparam.f defaults
IOVRLW_DEFAULT = 1       # :144
ISUBCLW_DEFAULT = 0      # :166
ILWCLIQ_DEFAULT = 1      # :75
ILWRATE = 2              # :71 (parameter, k/second)


def _flags(cfg, icldflg: int) -> tuple[int, int, int, int, int]:
    iovrlw = int(getattr(cfg, "iovrlw", IOVRLW_DEFAULT))
    isubclw = int(getattr(cfg, "isubclw", ISUBCLW_DEFAULT))
    icldflg = int(icldflg)   # radinit_nmmb value as seen by rlwinit (grrad_nmmb.f:424-428 for icmphys 5)
    ilwcliq = int(getattr(cfg, "ilwcliq", ILWCLIQ_DEFAULT))
    ilwrate = int(getattr(cfg, "ilwrate", ILWRATE))
    # Fortran checks in rlwinit: overlap flag and iovrlw=2 adjustment (radlw_main_nmmb.f:1229-1235),
    # isubclw range (print block, :1250-1262), cloud-flag consistency (after the print block).
    if iovrlw < 0 or iovrlw > 2:
        raise ValueError(f"rlwinit: cloud overlap flag IOVRLW={iovrlw} out of range 0..2 (Fortran STOPs)")
    if isubclw not in (0, 1, 2):
        raise ValueError(f"rlwinit: sub-column flag ISUBCLW={isubclw} not 0, 1 or 2 (Fortran STOPs)")
    if (icldflg == 0 and ilwcliq != 0) or (icldflg == 1 and ilwcliq == 0):
        raise ValueError(f"rlwinit: ICLDFLG={icldflg} and ILWCLIQ={ilwcliq} inconsistent (Fortran STOPs)")
    if iovrlw == 2 and isubclw == 0:
        iovrlw = 1       # radlw_main.f:1235 (IOVRLW=2 needs ISUBCLW>0; Fortran falls back to 1)
    return iovrlw, isubclw, icldflg, ilwcliq, ilwrate


def _alloc(ctx):
    import cupy as cp

    t = ctx.tables
    if "rrtm_lw_tau_tbl" not in t:
        t["rrtm_lw_tau_tbl"] = cp.zeros(NTBL + 1, dtype=cp.float32)
        t["rrtm_lw_exp_tbl"] = cp.zeros(NTBL + 1, dtype=cp.float32)
        t["rrtm_lw_tfn_tbl"] = cp.zeros(NTBL + 1, dtype=cp.float32)
        t["rrtm_lw_semiss0"] = cp.zeros(NBANDS, dtype=cp.float32)
        t["rrtm_lw_consts"] = cp.zeros(3, dtype=cp.float32)
        t["rrtm_lw_ctl"] = cp.zeros(4, dtype=cp.int32)
    return t


def run(ctx):
    import cupy as cp
    import numpy as np

    iovrlw, isubclw, icldflg, ilwcliq, ilwrate = _flags(ctx.cfg, host_ctl(ctx)["icldflg_rad"])
    t = _alloc(ctx)
    t["rrtm_lw_ctl"][:] = cp.asarray(np.array([iovrlw, isubclw, icldflg, ilwcliq], dtype=np.int32))
    # Host copy for the consumers (no device sync): radlw reads these module values, not ctx.cfg.
    t["rrtm_lw_ctl_host"] = {"iovrlw": iovrlw, "isubclw": isubclw, "icldflg": icldflg,
                             "ilwcliq": ilwcliq, "ilwrate": ilwrate}

    k = kernels.get(UNIT, KERNEL)
    block = 256
    grid = (NTBL + 1 + block - 1) // block
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((grid,), (block,), (t["rrtm_lw_tau_tbl"], t["rrtm_lw_exp_tbl"], t["rrtm_lw_tfn_tbl"],
                              t["rrtm_lw_semiss0"], t["rrtm_lw_consts"], np.int32(ilwrate)))
    return t


if __name__ == "__main__":
    # Standalone smoke: one launch, no state needed. Run via tools/check_unit.py for the done rule.
    ctx = SimpleNamespace(cfg=SimpleNamespace(), tables={}, stream=None)
    out = run(ctx)
    print({k: v.shape for k, v in out.items()})
