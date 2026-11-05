"""drv-time-loop: the NMMB forecast time loop (NMM_INTEGRATE, NAM nam.v4.2.11, 83df4e13c1).

Fortran: nmm/module_NMM_INTEGRATE.F90:79-1666.  Host-side unit: no kernel, because the loop does
no device arithmetic.  Each step's work is launched by the units in driver.py's plan; this wrapper
owns the order of the run:

  1. TDFI (filter_method 3, nam config): drv_dfi_tdfi_driver.run on its own clock (CLOCK_FILTER,
     GRID_COMP 4730+ calls NMM_INTEGRATE with clock_integrate=CLOCK_FILTER).  The forecast clock
     (CLOCK_INTEGRATE) is not advanced by the filter, so the forecast starts at ntimestep 0 with
     first_step True (the SOLVER_RUN firststep block runs once, on the forecast's first step).
  2. The forecast loop: driver.Driver.run (history 0 h, step, history on the post-advance count).

Not ported (refused by name, not stubbed): the NAM's own restart files, read when the configfile says restart
(Driver.run raises when ctx.restart; woof-nam's own restart sets are namgpu/restart.py, woof-nam run --resume),
TIMESERIES_RUN (NMM_INTEGRATE 844-861, 1140-1147).  The ESMF alarm checks and the CLOCKTIME print
have no numerical effect and are not ported.

run(ctx, hours) returns the number of forecast steps taken.
"""
from __future__ import annotations

UNIT = "drv_time_loop"
KERNEL_FILE = None          # host-side unit, no device arithmetic (see module docstring)


def run(ctx, hours: float, log=print) -> int:
    from namgpu.steps import drv_dfi_tdfi_driver  # noqa: PLC0415
    from namgpu.steps.drv_physics_initialize import in_run_dir  # noqa: PLC0415

    # The forecast runs in the NMMB run directory: gas_update reopens the CO2 file by name in the loop.
    with in_run_dir(ctx):
        return _run(ctx, hours, drv_dfi_tdfi_driver, log)


def _run(ctx, hours, drv_dfi_tdfi_driver, log=print):
    drv = ctx.driver
    if drv is None:
        raise RuntimeError("drv-time-loop: ctx.driver (the Driver) is required")
    ctx.dt = ctx.cfg.dt
    if int(ctx.cfg.filter_method) > 0:
        ctx.filter_active = True
        try:
            drv_dfi_tdfi_driver.run(ctx)
        finally:
            ctx.filter_active = False
        ctx.ntimestep = 0
        ctx.first_step = True
    return drv.run(hours, log=log)
