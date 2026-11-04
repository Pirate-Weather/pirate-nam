"""drv-dfi-dyn-init: DIGITAL_FILTER_DYN_INIT_NMM, set up the TDFI dynamics filter sums.

Ported from nmm/module_DIGITAL_FILTER_NMM.F90:46-203 (NAM nam.v4.2.11, 83df4e13c1).
Kernels: namgpu/kernels/drv_dfi_dyn_init.cu, dfi_dolph_window (the DOLPH call, line 108) and
dfi_dyn_init_zero (the zeroing, lines 188-200). Two launches, no host sync, no per-step work.

run(ctx, ndfistep=None, dyn_vars=None) builds two things:

  ctx.filt, the dyn-init record (kept for the leg bookkeeping that dyn-average resets):
    nstep, kstep, dt, taus, dolph, sum2d, sum3d, sum4d, totalsum, tot_rank_2d/3d/4d,
    name_save_2d/3d/4d.

  ctx.tables["DFI"], the handoff drv-dfi-dyn-sum reads and drv-dfi-dyn-average consumes. It holds the
  same device arrays, under the keys dyn-sum and dyn-average use:
    nstep, kstep (-nstep-1), dolph (True), mean_on (0), dolph_wgts (the DOLPH weights),
    digfil_buf, totalsum, names_2d/3d/4d (the name_save lists), sum_2d/sum_3d/sum_4d,
    num_water, num_tracers.

Sums are zeroed here (Fortran 188-200), so every leg starts from zero.

Inputs that the Fortran gets from its caller (not from this routine):
  ndfistep    the caller passes NDFISTEP(MY_DOMAIN_ID) from the configfile (nsecs_bcktdfi / filt_dt).
              Default here: nsecs_bcktdfi from parm/nam_configfile divided by cfg.filt_dt (75 at 600 s / 8 s).
  dyn_vars    the dynamics filter list, (name, rank) in bundle order. Default: every F and B row of
              parm/nam_filt_vars_norad.txt with its rank from the state tables.
  ctx.tables["indx"]  from drv-tracer-index. Its NUM_TRACERS_MET and NUM_TRACERS_CHEM are passed to the
              sum and average calls in the Fortran caller's order (module_DOMAIN_GRID_COMP.F90
              5152-5155 and 5195-5198: NUM_TRACERS_MET in the NUM_WATER slot, NUM_TRACERS_CHEM in the
              NUM_TRACERS slot). That order is mirrored here, and it is only observable for 4-D fields.

Rank-4 fields: the Fortran allocates array_save_4d with SPEC_MAX (line 179), which
module_DIGITAL_FILTER_NMM.F90 never assigns. This port sizes it as max(NUM_TRACERS, NUM_WATER), the
smallest size that holds every species count the 4-D fields can use, and states that choice here.
The production TDFI list (parm/nam_filt_vars_norad.txt) has no rank-4 row, so the 4-D sums are unused
by production; they are allocated and zeroed, and dyn-sum and dyn-average handle them.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.config import PARM
from namgpu.state import all_specs

UNIT = "drv_dfi_dyn_init"
K_ZERO = "dfi_dyn_init_zero"
K_DOLPH = "dfi_dolph_window"
BLOCK_ZERO = 256
BLOCK_DOLPH = 128


def dyn_filter_vars(path: Path = PARM / "nam_filt_vars_norad.txt"):
    """(name, rank) for every F and B row of the NAM filter table, in table order.

    B rows are "Filter if not restart" in the table header, so a cold start (this unit's default)
    filters them too. R rows are restore-only and are never filtered here.
    """
    specs = all_specs()
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^'([A-Za-z0-9_]+)'\s+[FB]\b", line.strip())
        if m:
            name = m.group(1)
            out.append((name, specs[name].rank))
    return out


def _ndfistep(cfg) -> int:
    """nsecs_bcktdfi / filt_dt: the TDFI backward half window in filter steps."""
    text = (PARM / "nam_configfile").read_text()
    m = re.search(r"^nsecs_bcktdfi:\s*([0-9.]+)", text, re.M)
    if m is None:
        raise ValueError(f"{PARM / 'nam_configfile'}: no nsecs_bcktdfi, cannot set ndfistep")
    return int(round(float(m.group(1)) / float(cfg.filt_dt)))


def _species(ctx):
    """NUM_WATER and NUM_TRACERS in the order the Fortran caller passes them (see the module docstring)."""
    indx = ctx.tables.get("indx")
    if indx is None:
        raise RuntimeError("drv-dfi-dyn-init needs ctx.tables['indx'] from drv-tracer-index "
                           "(NUM_TRACERS_MET, NUM_TRACERS_CHEM, NUM_WATER)")
    return int(indx["NUM_TRACERS_MET"]), int(indx["NUM_TRACERS_CHEM"])


def _buf(prev: dict, key: str, shape, cp):
    """Allocate once; reuse when the shape is unchanged (Fortran allocates only if not associated)."""
    a = prev.get(key)
    if a is None or tuple(a.shape) != tuple(shape):
        a = cp.zeros(shape, dtype=cp.float32)
    return a


def run(ctx, ndfistep=None, dyn_vars=None):
    import cupy as cp

    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    nstep = int(ndfistep) if ndfistep is not None else _ndfistep(ctx.cfg)
    fields = dyn_filter_vars() if dyn_vars is None else list(dyn_vars)
    num_water, num_tracers = _species(ctx)
    spec_max = max(num_tracers, num_water, 1)

    # Lines 102 and 106: dt = float(dt_int) + float(dt_num)/float(dt_den); taus = float(2*ndfistep)*dt.
    # dt is the FILTER clock step (cfg.filt_dt, 8 s), not the model step: the caller passes the filter
    # clock INTEGER_DT/NUMERATOR_DT/IDENOMINATOR_DT (module_DOMAIN_GRID_COMP.F90:5092-5098).
    dt = np.float32(ctx.cfg.filt_dt)
    taus = np.float32(2 * nstep) * dt

    # Lines 134-158: classify the fields by rank.
    name_2d = [n for n, r in fields if r == 2]
    name_3d = [n for n, r in fields if r == 3]
    name_4d = [n for n, r in fields if r == 4]
    n2, n3, n4 = len(name_2d), len(name_3d), len(name_4d)

    prev = ctx.filt if isinstance(ctx.filt, dict) else {}
    sum2d = _buf(prev, "sum2d", (n2, jm, im), cp)
    sum3d = _buf(prev, "sum3d", (n3, lm + 1, jm, im), cp)
    sum4d = _buf(prev, "sum4d", (n4, spec_max, lm, jm, im), cp)
    totalsum = _buf(prev, "totalsum", (1,), cp)
    dolph = cp.zeros(2 * nstep + 1, dtype=cp.float32)
    digfil_buf = cp.zeros((1,), dtype=cp.float32)

    stream = ctx.stream
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        # Line 108: DOLPH(dt, taus, nstep, dolph_wgts).
        nblk = (nstep + 1 + BLOCK_DOLPH - 1) // BLOCK_DOLPH
        kernels.get(UNIT, K_DOLPH)(
            (nblk,), (BLOCK_DOLPH,),
            (dolph, np.int32(nstep), dt, taus))
        # Lines 188-200: array_save_2d = 0, array_save_3d = 0, totalsum = 0.
        nel = max(int(sum2d.size), int(sum3d.size), 1)
        nblk = (nel + BLOCK_ZERO - 1) // BLOCK_ZERO
        kernels.get(UNIT, K_ZERO)(
            (nblk,), (BLOCK_ZERO,),
            (sum2d, np.uint64(sum2d.size), sum3d, np.uint64(sum3d.size), totalsum))
        sum4d.fill(0)                                   # array_save_4d = 0 (rank-4 fields, if any)

    ctx.filt = {
        "nstep": nstep,
        "kstep": -nstep - 1,
        "dt": float(dt),
        "taus": float(taus),
        "dolph": dolph,
        "sum2d": sum2d,
        "sum3d": sum3d,
        "sum4d": sum4d,
        "totalsum": totalsum,
        "tot_rank_2d": n2,
        "tot_rank_3d": n3,
        "tot_rank_4d": n4,
        "name_save_2d": name_2d,
        "name_save_3d": name_3d,
        "name_save_4d": name_4d,
    }
    # The handoff for drv-dfi-dyn-sum (reads it every step) and drv-dfi-dyn-average (reads it once per leg).
    ctx.tables["DFI"] = {
        "nstep": nstep,
        "kstep": -nstep - 1,
        "dolph": True,
        "mean_on": 0,
        "dolph_wgts": dolph,
        "digfil_buf": digfil_buf,
        "totalsum": totalsum,
        "names_2d": name_2d,
        "names_3d": name_3d,
        "names_4d": name_4d,
        "sum_2d": sum2d,
        "sum_3d": sum3d,
        "sum_4d": sum4d,
        "num_water": num_water,
        "num_tracers": num_tracers,
    }
    return ctx.filt
