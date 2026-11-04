"""drv-dfi-tdfi-driver: RUN_DIGITAL_FILTER_NMM, TDFI branch (filter_method 3).

Fortran: nmm/module_NMM_GRID_COMP.F90:3624-5418 (RUN_DIGITAL_FILTER_NMM, method_block 3 at
4730-5382) and nmm/module_DOMAIN_GRID_COMP.F90:4839-5321 (NMM_FILTERING, the per-step hook).
NAM nam.v4.2.11, 83df4e13c1 (clone path src/atmos/nmm/). Kernel file: namgpu/kernels/drv_dfi_tdfi_driver.cu
(no kernel).

Order of work (review fix). NMM_INTEGRATE runs the solver step, then the hook NMM_FILTERING (phase 2),
then advances the clock (module_NMM_INTEGRATE.F90 lines ~1076-1111). The hook therefore sees the state
after the step, labelled with the clock value from before the advance. This wrapper does the same: for
filter call k it runs the solver step first, then the hook work for k.

Hook work for call k, in Fortran order:
  backward leg (clock START - 8k before the hook, dt = -filt_dt):
    k == 0        : phy_init (FIRST_FILTER, once), dyn_init (CURRTIME==STARTTIME), phy_save
                    (backward start only, Fortran line ~5090)
    every k       : dyn_sum (CURRTIME <= STARTTIME)
    k == n_bck-1  : dyn_average (TESTTIME == DFITIME, FILTER_METHOD == 3)
  forward leg (clock START + 8k, dt = +filt_dt):
    k == 0        : dyn_init (CURRTIME==STARTTIME again, the forward leg has its own call)
    every k       : dyn_sum (CURRTIME >= STARTTIME)
    k == n_fwd-1  : dyn_average (unconditional in the forward branch), phy_restore
    phy_save is not called: the forward branch saves physics only for FILTER_METHOD == 1 (line ~5145).

Physics. SOLVER_RUN runs physics only when INTEGER_DT > 0 (module_SOLVER_GRID_COMP.F90:4929). The
backward clock timestep is -FILT_TIMESTEP (RUN_DIGITAL_FILTER_NMM, TIMESTEP_FILTER=-FILT_TIMESTEP,
line ~4183), so the backward leg is dynamics only and the forward leg runs physics. The solver's
IFACT is -1 on the backward leg (module_SOLVER_GRID_COMP.F90:3134-3135), so the signed dt is correct.

Clock. Each leg is 2*nsecs/filt_dt steps. With nsecs_bcktdfi = nsecs_fwdtdfi = 600 s and filt_dt = 8 s
that is 150 steps per leg, a 1200 s window per leg (DFITIME = START -/+ 2*HALFDFIINTVAL).

Binding. This unit owns the schedule, the hook order and the filter-step op. It binds no arrays itself:
the filter state lives in ctx.filt (drv-dfi-dyn-init, drv-dfi-dyn-sum, drv-dfi-dyn-average), the physics
save and restore fields live in ctx.st (drv-dfi-phy-init, drv-dfi-phy-save, drv-dfi-phy-restore).

run(ctx, ops=None) returns the schedule actually executed (dict). No host sync is added here.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from namgpu.dynamics import dyn_vtoa

UNIT = "drv_dfi_tdfi_driver"
KERNEL_FILE = "drv_dfi_tdfi_driver"

# Dependency unit id -> wrapper module, and the entry point each one exports.
DEPS = {
    "phy_init": ("namgpu.steps.drv_dfi_phy_init", "run"),
    "dyn_init": ("namgpu.steps.drv_dfi_dyn_init", "run"),
    "dyn_sum": ("namgpu.steps.drv_dfi_dyn_sum", "run"),
    "dyn_average": ("namgpu.steps.drv_dfi_dyn_average", "run"),
    "phy_save": ("namgpu.steps.drv_dfi_phy_save", "run"),
    "phy_restore": ("namgpu.steps.drv_dfi_phy_restore", "run"),
}


@dataclass(frozen=True)
class Schedule:
    filt_dt: float
    ndfistep: int          # NDFISTEP = nsecs_bcktdfi / filt_dt (line 4792)
    n_bck: int             # backward steps = 2*ndfistep (DFITIME = start - 2*HALF)
    n_fwd: int             # forward steps = 2*nsecs_fwdtdfi / filt_dt


def schedule(cfg) -> Schedule:
    """Filter steps per leg from the configfile (nsecs_bcktdfi, nsecs_fwdtdfi, filt_dt)."""
    if int(cfg.filter_method) != 3:
        raise ValueError(f"drv-dfi-tdfi-driver: filter_method must be 3 (TDFI), got {cfg.filter_method}")
    raw = cfg.raw
    dt = float(cfg.filt_dt)
    if dt <= 0.0:
        raise ValueError(f"drv-dfi-tdfi-driver: filt_dt must be > 0, got {dt}")
    bck = float(raw["nsecs_bcktdfi"])
    fwd = float(raw["nsecs_fwdtdfi"])
    # The Fortran forward leg takes HALFDFIINTVAL (and so its averaging time DFITIME) from the backward
    # count NDFISTEP (line ~5135), while its clock stop comes from nsecs_fwdtdfi (line ~5151). If the two
    # differ the forward average fires at the wrong step, so refuse rather than run that.
    if abs(fwd - bck) > 1e-6:
        raise ValueError(
            f"drv-dfi-tdfi-driver: nsecs_fwdtdfi={fwd} != nsecs_bcktdfi={bck}; the Fortran forward leg "
            "takes its window from the backward count but its clock stop from nsecs_fwdtdfi, so the "
            "forward average would fire at the wrong step")
    ndfistep = int(0.1 + bck / dt)                      # INT(0.1+DFIHR/(S+Sn/Sd))
    if abs(ndfistep * dt - bck) > 1e-6:
        raise ValueError(f"drv-dfi-tdfi-driver: nsecs_bcktdfi={bck} is not a multiple of filt_dt={dt}")
    n_fwd = int(round(2.0 * fwd / dt))
    if abs(n_fwd * dt - 2.0 * fwd) > 1e-6:
        raise ValueError(f"drv-dfi-tdfi-driver: nsecs_fwdtdfi={fwd} is not a multiple of filt_dt={dt}")
    return Schedule(filt_dt=dt, ndfistep=ndfistep, n_bck=2 * ndfistep, n_fwd=n_fwd)


class Ops:
    """The dependency calls. Each callable takes ctx first.

    phy_init(ctx)                        one-time physics save setup (Fortran FIRST_FILTER)
    dyn_init(ctx, ndfistep)
    dyn_sum(ctx, mean_on)
    dyn_average(ctx)
    phy_save(ctx)
    phy_restore(ctx)
    integrate(ctx, direction, dt, hdiff_on, mean_on, ntimestep)   one solver step with the filter flags
    """

    def __init__(self, **fns):
        required = {"phy_init", "dyn_init", "dyn_sum", "dyn_average", "phy_save", "phy_restore", "integrate"}
        missing = required - set(fns)
        if missing:
            raise ValueError(f"drv-dfi-tdfi-driver: Ops missing {sorted(missing)}")
        for k, v in fns.items():
            setattr(self, k, v)


def integrate_filter_step(ctx, direction, dt, hdiff_on, mean_on, ntimestep):
    """One TDFI filter model step: the whole solver plan once, with the filter flags set.

    dt is the signed timestep the solver sees (negative on the backward leg). Driver.step() runs the
    solver plan and advances ctx.ntimestep. This function sets the filter values first and restores every
    model value afterwards, so the forecast clock is unchanged by the filter.

    first_step follows Fortran NTIMESTEP == 0 for the filter call index (the Fortran counter is not traced
    in this wrapper; see STATUS notes). The mean flag is taken from the Fortran call (MEAN_ON = 0 for TDFI).
    """
    if ctx.driver is None or not hasattr(ctx.driver, "step"):
        raise RuntimeError("drv-dfi-tdfi-driver: ctx.driver (a Driver) is required for filter steps")
    if (direction == "Bckward") != (float(dt) < 0.0):
        raise ValueError(f"drv-dfi-tdfi-driver: direction {direction!r} does not match dt sign {dt}")
    if int(mean_on) != 0:
        raise ValueError("drv-dfi-tdfi-driver: MEAN_ON must be 0 for TDFI (filter_method 3)")
    saved = (ctx.ntimestep, ctx.first_step, ctx.dt, ctx.hdiff_on, ctx.physics_on, ctx.filter_active)
    try:
        ctx.dt = float(dt)
        ctx.hdiff_on = bool(hdiff_on)
        ctx.physics_on = float(dt) > 0.0          # SOLVER_RUN physics gate INTEGER_DT > 0 (line 4929)
        ctx.filter_active = True
        ctx.ntimestep = int(ntimestep)
        ctx.first_step = int(ntimestep) == 0
        ctx.driver.step()
    finally:
        (ctx.ntimestep, ctx.first_step, ctx.dt, ctx.hdiff_on, ctx.physics_on, ctx.filter_active) = saved


def default_ops(ctx) -> Ops:
    """Bind the dependency wrappers. Fails naming every missing unit at once."""
    fns, missing = {}, []
    for key, (mod, attr) in DEPS.items():
        try:
            fns[key] = getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError):
            missing.append(mod)
    if missing:
        raise ImportError("drv-dfi-tdfi-driver: dependency wrappers not written yet: " + ", ".join(missing))
    fns["integrate"] = lambda c, direction, dt, hdiff_on, mean_on, k: integrate_filter_step(
        c, direction, dt, hdiff_on, mean_on, k)
    return Ops(**fns)


def run(ctx, ops: Ops | None = None) -> dict:
    """Execute the TDFI filter once on ctx.st: backward leg, then forward leg."""
    ops = ops if ops is not None else default_ops(ctx)
    s = schedule(ctx.cfg)
    # EF4T lifecycle of SOLVER_RUN: the first pass scales int_state%EF4T by IFACT*DTRATIO
    # (backward leg, dt < 0); the direction change flips it at the start of the forward leg; the
    # filter end restores EF4T_SV before the first forecast step (dyn-vtoa reads it).
    # SOLVER_RUN FIRST_PASS / changedir / end_filt: every dt-scaled constant (FAD, FAH, FCP, DDMPU,
    # DDMPV, WPDAR, HDAC*, EF4T) goes through drv-filter-consts (module_SOLVER_GRID_COMP.F90:3134-3342).
    from namgpu.steps import drv_filter_consts  # noqa: PLC0415
    drv_filter_consts.filter_start(ctx, -s.filt_dt)

    # Backward leg. Each call: solver step first (Fortran NMM_INTEGRATE), then the hook for that call.
    for k in range(s.n_bck):
        ops.integrate(ctx, "Bckward", -s.filt_dt, 0, 0, k)
        if k == 0:
            ops.phy_init(ctx)                  # FIRST_FILTER one-time setup, before the first save
            ops.dyn_init(ctx, s.ndfistep)
            ops.phy_save(ctx)
        ops.dyn_sum(ctx, 0)
        if k == s.n_bck - 1:
            ops.dyn_average(ctx)

    drv_filter_consts.direction_change(ctx)
    # changedir WRITE_BC (module_SOLVER_GRID_COMP.F90:3290-3299, IF(.NOT.GLOBAL)): recompute the boundary
    # tendencies from the state at the end of the backward leg toward the interval's target, over
    # TBOCO + DFIHR_BOCO/2 (DFIHR_BOCO = nsecs_bcktdfi for filter_method 3, module_GET_CONFIG.F90:613).
    # The forward leg then keeps these (WRITE_BC_FLAG = 1: no READ_BC at its first step, 3679-3686).
    if isinstance(getattr(ctx, "bc", None), dict):
        from namgpu.boundary import bnd_write_bc  # noqa: PLC0415
        tboco = float(ctx.tables.get("consts", {}).get("tboco", 10800.0))
        bnd_write_bc.run_dict(ctx, site="dfi_start", recomp_tend=True,
                              tboco=tboco + float(ctx.cfg.raw["nsecs_bcktdfi"]) / 2.0)

    # Forward leg. Same pattern; average and physics restore on the last call.
    for k in range(s.n_fwd):
        ops.integrate(ctx, "Forward", s.filt_dt, 1, 0, k)
        if k == 0:
            ops.dyn_init(ctx, s.ndfistep)
        ops.dyn_sum(ctx, 0)
        if k == s.n_fwd - 1:
            ops.dyn_average(ctx)
            ops.phy_restore(ctx)

    drv_filter_consts.filter_end(ctx)
    return {"ndfistep": s.ndfistep, "n_bck": s.n_bck, "n_fwd": s.n_fwd, "filt_dt": s.filt_dt}
