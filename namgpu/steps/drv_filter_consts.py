"""drv-filter-consts: the dt-scaled grid constants of SOLVER_RUN and their TDFI state machine.

Fortran: nmm/module_SOLVER_GRID_COMP.F90 (NAM nam.v4.2.11, 83df4e13c1)
  FIRST_PASS block 3134-3196: save *_SV, then DDMPV, EF4T *= IFACT*DTRATIO; per row DDMPU *= IFACT,
      FAD, FAH, FCP *= IFACT*DTRATIO, WPDAR *= IFACT; HDACX, HDACY, HDACVX, HDACVY *= IFACT*DTRATIO.
  changedir 3226-3262 (backward -> forward): FIRST_STEP, TP=T, UP=U, VP=V, then every one of the
      above *= -1 (IFACT = -1, no DTRATIO).
  end_filt 3302-3342: TP=T, UP=U, VP=V, restore every *_SV, WPDAR *= +1, FIRST_STEP.
  DTRATIO = ABS(DT/FUND_DT) (2958), IFACT = +1 for INTEGER_DT >= 0 else -1 (3134-3135).

Host-side unit, no kernel: CuPy in-place scalings of a few rows and four 2-D fields, three times per
run (TDFI start, direction change, end). Never on the per-step path.

One array per Fortran int_state constant. The canonical arrays are init-grid-consts' rows in
ctx.tables["grid"] (fad, fah, fcp, ddmpu, wpdar, hdacx, hdacy, hdacvx, hdacvy). bind(ctx) makes every
consumer read those same objects: it aliases ctx.st.fields FAH, HDACX, HDACY, HDACVX, HDACVY and FAD to
them (before this unit, the state's HDAC* fields were never filled, so HDIFF and the CDWDT diffusion
ran with zero coefficients, and each FAH/FAD consumer held its own copy that the filter never scaled).
DDMPV is the host scalar ctx.tables["ddamp_dvd"] (dyn-ddamp-extmode); EF4T is st.scalars["EF4T"]
(dyn-vtoa's ef4t_* functions are called from here, so there is one EF4T path).

Not ported here (named gap, owner: boundary group): the changedir WRITE_BC call (3264-3277,
recompute the boundary tendencies from the state at the end of the backward leg).
"""
from __future__ import annotations

import numpy as np

ROWS_DTRATIO = ("fad", "fah", "fcp")            # *= IFACT*DTRATIO
ROWS_IFACT = ("ddmpu", "wpdar")                 # *= IFACT
FIELDS_DTRATIO = ("hdacx", "hdacy", "hdacvx", "hdacvy")
ST_ALIAS = {"FAH": "fah", "FAD": "fad", "HDACX": "hdacx", "HDACY": "hdacy",
            "HDACVX": "hdacvx", "HDACVY": "hdacvy"}
SV_KEY = "filter_consts_sv"


def _grid(ctx):
    g = ctx.tables.get("grid") if ctx.tables else None
    if g is None:
        raise RuntimeError("drv-filter-consts: ctx.tables['grid'] missing (init-grid-consts must run first)")
    return g


def bind(ctx) -> None:
    """Alias the state copies of the dt-scaled constants to init-grid-consts' arrays (idempotent)."""
    g = _grid(ctx)
    st = ctx.st
    for name, key in ST_ALIAS.items():
        src = g[key]
        cur = st.fields.get(name)
        if cur is src:
            continue
        if cur is not None and tuple(cur.shape) != tuple(src.shape):
            raise RuntimeError(f"drv-filter-consts: st.{name} shape {tuple(cur.shape)} != grid {key} "
                               f"{tuple(src.shape)}")
        st.fields[name] = src
    ctx.tables["filter_consts_bound"] = True


def _ddmpv(ctx):
    from namgpu.dynamics.dyn_ddamp_extmode import ddamp_coefficients  # noqa: PLC0415
    ddamp_coefficients(ctx)                       # creates ctx.tables["ddamp_dvd"] once
    return np.float32(ctx.tables["ddamp_dvd"])


def _scale(ctx, f_rows_dt, f_rows_i, f_fields):
    g = _grid(ctx)
    for k in ROWS_DTRATIO:
        g[k] *= np.float32(f_rows_dt)
    for k in ROWS_IFACT:
        g[k] *= np.float32(f_rows_i)
    for k in FIELDS_DTRATIO:
        g[k] *= np.float32(f_fields)


def _copy_prev(ctx):
    st = ctx.st
    for a, b in (("TP", "T"), ("UP", "U"), ("VP", "V")):
        st.fields[a][...] = st.fields[b]


def filter_start(ctx, dt_step: float, fund_dt: float | None = None) -> None:
    """FIRST_PASS with FILTER_METHOD /= 0: save, then scale for the first filter dt (signed)."""
    from namgpu.dynamics import dyn_vtoa  # noqa: PLC0415
    bind(ctx)
    g = _grid(ctx)
    fund = float(ctx.cfg.dt if fund_dt is None else fund_dt)
    ifact = np.float32(1.0 if dt_step >= 0 else -1.0)
    dtratio = np.float32(abs(float(dt_step) / fund))
    sv = {k: g[k].copy() for k in ROWS_DTRATIO + ("ddmpu",) + FIELDS_DTRATIO}
    sv["ddmpv"] = _ddmpv(ctx)
    sv["ef4t"] = dyn_vtoa.ef4t_filter_start(ctx, dt_step, fund)
    ctx.tables[SV_KEY] = sv
    ctx.tables["ddamp_dvd"] = np.float32(ifact * dtratio * sv["ddmpv"])
    _scale(ctx, ifact * dtratio, ifact, ifact * dtratio)


def direction_change(ctx) -> None:
    """changedir: TP/UP/VP = T/U/V, then every scaled constant *= -1 (IFACT = -1)."""
    from namgpu.dynamics import dyn_vtoa  # noqa: PLC0415
    _copy_prev(ctx)
    m1 = np.float32(-1.0)
    ctx.tables["ddamp_dvd"] = np.float32(m1 * np.float32(ctx.tables["ddamp_dvd"]))
    dyn_vtoa.ef4t_direction_change(ctx)
    _scale(ctx, m1, m1, m1)
    ctx.first_step = True


def filter_end(ctx) -> None:
    """end_filt: TP/UP/VP = T/U/V, restore every saved constant (WPDAR *= +1 is a no-op)."""
    from namgpu.dynamics import dyn_vtoa  # noqa: PLC0415
    _copy_prev(ctx)
    g = _grid(ctx)
    sv = ctx.tables.pop(SV_KEY)
    for k in ROWS_DTRATIO + ("ddmpu",) + FIELDS_DTRATIO:
        g[k][...] = sv[k]
    ctx.tables["ddamp_dvd"] = np.float32(sv["ddmpv"])
    dyn_vtoa.ef4t_filter_end(ctx, sv["ef4t"])
    ctx.first_step = True
