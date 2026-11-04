"""phys-rrtm-radinit: radinit_nmmb, initialisation of the NMMB RRTM radiation package.

Ported from phys/grrad_nmmb.f:226-451 (NAM nam.v4.2.11, 83df4e13c1), SUBROUTINE radinit_nmmb.
Kernel: namgpu/kernels/phys_rrtm_radinit.cu (radinit_ctl, one thread, one launch).

What the routine does in Fortran:
  1. Sets itsfc = iemsflg/10 and loz1st = (ioznflg == 0)              (lines 340-341).
  2. Prints the control settings and checks the McICA flags on me == 0 (diagnostic only).
  3. Calls in order: sol_init_nmmb (astronomy), aer_init (aerosols; clim_aerinit is called inside it when
     iaerflg is not 0 or 100 and iaermdl is 0), gas_init, sfc_init, cld_init, then rlwinit and rswinit.
     icldflg is forced to 0 for the two radiation inits when icmphys == 5 and restored afterwards (424-446).

This wrapper computes the flags on the device from the physparam state that rad_initialize_nmmb wrote
(ctx.tables['rad_physparam'], built from the NAM values ntoz, ntcw, iems, iaer, np3d in cfg). It reads
the results back once at init into ctx.tables['rrtm_ctl_host'] (a host dict, init only, never per step),
then dispatches the sibling init units in Fortran order. The siblings take their flags from that dict
through host_ctl(ctx), so no flag is a module constant.

In:    ctx.tables['rad_physparam'] (int32[26], phys-rrtm-rad-initialize; must run first).
Out:   ctx.tables['rrtm_ctl'] = int32[7] device array [itsfc, loz1st, icldflg_rad, icldflg_out, iaerflg,
       iaermdl, ioznflg]; ctx.tables['rrtm_ctl_host'] = the same values as a host dict (keys below).
Launch: one kernel launch on ctx.stream, one init-time readback, then the siblings.

Not done here: the me == 0 prints (diagnostic, not device work).
"""
from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path

from namgpu import kernels
from namgpu.physics.rrtm.phys_rrtm_rad_initialize import KEY as RAD_KEY

UNIT = "phys_rrtm_radinit"
KERNEL = "radinit_ctl"
KEY = "rrtm_ctl"
HOST_KEY = "rrtm_ctl_host"
N_CTL = 7
ROOT = Path(__file__).resolve().parents[3]

# Sibling init units in the Fortran call order of radinit_nmmb (grrad_nmmb.f:340-446).
# clim-aerinit is called from aer_init (radiation_aerosols_nmmb.f:551), so it follows aer-init.
DEPS_IN_ORDER = [
    "phys-rrtm-sol-init",
    "phys-rrtm-aer-init",
    "phys-rrtm-clim-aerinit",
    "phys-rrtm-gas-init",
    "phys-rrtm-sfc-init",
    "phys-rrtm-cld-init",
    "phys-rrtm-lw-rlwinit",
    "phys-rrtm-sw-rswinit",
]


def host_ctl(ctx) -> dict:
    """Host copy of the radinit control flags (written by run()). Raises if radinit has not run."""
    h = ctx.tables.get(HOST_KEY)
    if h is None:
        raise RuntimeError("phys-rrtm-radinit: ctx.tables['rrtm_ctl_host'] is missing; radinit_nmmb must run "
                           "first (grrad_nmmb.f:340-446)")
    return h


def _sibling_run(uid: str):
    units = {u["id"]: u for u in json.loads((ROOT / "UNITS.json").read_text())["units"]}
    mod = importlib.import_module(units[uid]["wrapper"])
    return mod.run


def run(ctx, dep_calls: bool = True):
    import cupy as cp

    rip = ctx.tables.get(RAD_KEY)
    if rip is None:
        raise RuntimeError("phys-rrtm-radinit: ctx.tables['rad_physparam'] is missing; run phys-rrtm-rad-initialize "
                           "first (rad_initialize_nmmb precedes radinit_nmmb, drv_physics_initialize)")
    if KEY not in ctx.tables:
        ctx.tables[KEY] = cp.zeros(N_CTL, dtype=cp.int32)
    out = ctx.tables[KEY]
    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k((1, 1, 1), (1, 1, 1), (rip, out))
    if ctx.stream is not None:
        ctx.stream.synchronize()
    c = [int(v) for v in out.get()]   # one readback at init, not on the step path
    ctx.tables[HOST_KEY] = {
        "itsfc": c[0], "loz1st": c[1], "icldflg_rad": c[2], "icldflg_out": c[3],
        "iaerflg": c[4], "iaermdl": c[5], "ioznflg": c[6],
    }
    if dep_calls:
        h = ctx.tables[HOST_KEY]
        for uid in DEPS_IN_ORDER:
            if uid == "phys-rrtm-clim-aerinit":
                # radiation_aerosols_nmmb.f:515 returns for iaerflg 0; :539 skips 100; :551 calls clim_aerinit
                # only for the opac-climatology aerosol model (iaermdl 0).
                if h["iaerflg"] in (0, 100) or h["iaermdl"] != 0:
                    continue
            _sibling_run(uid)(ctx)
    return out
