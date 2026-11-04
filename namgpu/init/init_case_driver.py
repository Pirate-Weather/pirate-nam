"""init-case-driver: the whole NAM init chain, in dependency order, with per-stage timings.

Unit: init-case-driver (group init, mode new, no Fortran counterpart). Wrapper:
namgpu/init/init_case_driver.py (UNITS.json "wrapper"). Kernel: namgpu/kernels/init_case_driver.cu
(init_case_nonfinite_count, the done-rule census; the chain's own work is in its stage units).

In:     ctx.cfg, ctx.grid, ctx.st, ctx.stream, ctx.start (cycle datetime), ctx.case_dir (cases/<case>).
Out:    every stage's outputs in ctx.st / ctx.grid / ctx.tables / ctx.bc, as each stage defines them.
        ctx.timings["init"] = {"stages": [{"unit", "host_s", "status"}...], "total_s", "nonfinite"}.
        Stage timing is host wall clock around each stage's run(ctx); init is host-driven, so no
        device sync is added for it. The one device read is the census at the end: one integer per
        float32 array of ctx.st, copied back in a single transfer.

Order: CHAIN is the init sequence. run() checks it against UNITS.json "deps": every dep that is
itself in CHAIN must come earlier. One exception is recorded in ORDER_EXCEPTIONS: init-native-atm
lists init-vcoord-recover as a dep, but init-vcoord-recover reads the PRES field that
init-native-atm fills (see its docstring), so native-atm runs first. drv-physics-initialize owns
the phys-* init units (rrtm, jsfc, myj, noah, bmj, fer, gwd) and runs as one stage.

Not in the chain: init-read-nemsio (skip: inputs are GRIB2), the SOLVER_RUN step units and the
DFI units (drv-dfi-*), which run in the forecast driver rather than in init. init-gfs-decode,
init-gfs-hinterp, init-gfs-vinterp and init-gfs-wind-rotate are libraries called per boundary time by
init-gfs-bocos and per start by gfs_start.py (GFS start), not stages of their own; init-boco-tendencies
is the nest path (UPDATE_BC_TENDS) and the NAM parent never calls it.
The source of the initial state is ctx.init_source / cfg.raw["init_source"] ("native" for archived
cycles, "gfs" for any date; sources.py), staged by the init-grib2-decode stage.

Fails loud: every missing stage wrapper is reported in one message before anything runs
(check_ready); a stage that raises stops the chain and names the unit; a census that finds a NaN or
Inf raises after the chain and names the arrays.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import time
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "init-case-driver"
KERNEL_UNIT = "init_case_driver"
COUNT_KERNEL = "init_case_nonfinite_count"
from namgpu import paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

CHAIN = [
    "init-fetch-case",          # raw/nam, raw/gfs, raw/mrms (skips files already present)
    "init-grib2-decode",        # GRIB2 -> device arrays (gpuwm Rust bridges)
    "init-grid-geo",            # rotated lat-lon geometry, grid.rotated_to_geographic
    "init-gfs-bocos",           # boco.NNNN from GFS every 3 h (init-gfs-decode/hinterp/vinterp/
                                # wind-rotate per boundary time, gfs_bocos.py); sets ctx.boco_dir.
                                # Before drv-consts, which reads tboco from boco.0000 (module_CONTROL)
    "init-grid-consts",         # GRID_CONSTS (dxh, dxv, fcp, fdiv, ...)
    "drv-loop-limits",          # its_b1 / ite_h2 limit constants
    "drv-consts",               # CONSTS
    "drv-spline",               # host spline for the tables
    "drv-exptbl",               # EXPTBL
    "drv-psitbl",               # PSITBL
    "drv-tablet",               # TABLET
    "drv-tablep",               # TABLEP
    "drv-tracer-index",         # INDX_Q, INDX_CW, ...
    "init-native-sfc",          # surface and soil state from bgrd3d00
    "init-native-atm",          # hybrid 3-D fields -> state, fills PRES
    "init-vcoord-recover",      # SG / PDTOP from PRES against PD
    "init-static-missing",      # ALBASE, MXSNAL, IVGTYP, ISLTYP (missing or out-of-range)
    "init-derive-state",        # PINT, FIS, Z, DWDT, OMGALF, TRACERS, UP/VP/TP
    "bnd-boundary-init",        # boundary-zone constants and working arrays
    "init-gwd-orog",            # GWD orographic statistics
    "drv-physics-initialize",   # PHYSICS_INITIALIZE for the selected schemes
]

# (unit, dep) pairs where the UNITS.json dep direction is reversed against the data flow.
ORDER_EXCEPTIONS = {
    ("init-native-atm", "init-vcoord-recover"):
        "vcoord reads PRES that native-atm fills (init_vcoord_recover docstring)",
}


def _units() -> dict:
    return {u["id"]: u for u in json.loads((paths.UNITS_JSON).read_text())["units"]}


def check_order(units: dict, chain=CHAIN) -> None:
    """Every dep of a chain unit that is itself in the chain must come earlier."""
    pos = {u: i for i, u in enumerate(chain)}
    bad = []
    for u in chain:
        for d in units[u].get("deps", []):
            if d in pos and pos[d] > pos[u] and (u, d) not in ORDER_EXCEPTIONS:
                bad.append(f"{u} needs {d}, which runs later")
    if bad:
        raise RuntimeError("init chain order violates UNITS.json deps:\n  " + "\n  ".join(bad))


def check_ready(units: dict, chain) -> None:
    """Import every stage wrapper once; report every missing or skipped stage in one message."""
    bad = []
    for uid in chain:
        if units[uid].get("mode") == "skip":
            bad.append(f"{uid}: mode skip, cannot be a chain stage")
            continue
        try:
            importlib.import_module(units[uid]["wrapper"])
        except Exception as e:  # noqa: BLE001
            bad.append(f"{uid}: {type(e).__name__}: {e}")
    if bad:
        raise RuntimeError("init chain stages not ready:\n  " + "\n  ".join(bad))


def census(ctx) -> dict:
    """Non-finite count per float32 array of ctx.st. Launches only; one device read at the end."""
    import cupy as cp
    names = [n for n, a in ctx.st.fields.items() if a.dtype == cp.float32 and a.size]
    if not names:
        return {}
    counts = cp.zeros((len(names),), dtype=cp.uint32)
    k = kernels.get(KERNEL_UNIT, COUNT_KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    for i, name in enumerate(names):
        arr = ctx.st.fields[name].reshape(-1)
        with scope:
            k(((arr.size + 255) // 256, 1, 1), (256, 1, 1),
              (arr, np.int64(arr.size), counts[i:i + 1]))
    if ctx.stream is not None:
        ctx.stream.synchronize()   # the census stream is non-blocking: wait for it before the read
    host = cp.asnumpy(counts)
    return {n: int(c) for n, c in zip(names, host) if c}


def run(ctx, only=None, log=print) -> dict:
    """Run the init chain (or only the stages named in `only`), then the census.

    Returns the timing record, also stored at ctx.timings["init"]. Raises on a failed stage or on
    any NaN/Inf left in the state.
    """
    units = _units()
    check_order(units, CHAIN)
    chain = [u for u in CHAIN if only is None or u in only]
    check_ready(units, chain)

    stages = []
    t_all = time.perf_counter()
    from namgpu.state import fit_on as _fit_on  # noqa: PLC0415
    for uid in chain:
        mod = importlib.import_module(units[uid]["wrapper"])
        t0 = time.perf_counter()
        if uid == "drv-physics-initialize" and _fit_on():
            # 32 GB fit: the MYJ/BMJ table kernels need a large device local-memory reservation at launch;
            # hand the init temporaries cached in the CuPy pool back to the driver first
            import cupy as _cp  # noqa: PLC0415
            if ctx.stream is not None:
                ctx.stream.synchronize()
            _cp.get_default_memory_pool().free_all_blocks()
        try:
            mod.run(ctx)
        except Exception as e:
            raise RuntimeError(f"{UNIT}: stage {uid} failed: {type(e).__name__}: {e}") from e
        if uid == "init-gwd-orog" and _fit_on():
            # the 30-arcsec terrain (gwd_dem, int16 21600 x 43200 = 1.78 GiB) is read only by init-gwd-orog
            ctx.tables.pop("gwd_dem", None)
        if uid == "init-vcoord-recover":
            from namgpu.init import sources  # noqa: PLC0415
            sources.release_staged(ctx, keep=())     # staged decode inputs are dead from here on
        dt = time.perf_counter() - t0
        stages.append({"unit": uid, "host_s": round(dt, 3), "status": "ok"})
        log(f"{UNIT}: {uid:<26} {dt:9.2f} s")

    from namgpu.state import INIT_ONLY, fit_on  # noqa: PLC0415
    _fit = fit_on()
    if _fit:
        # the 30-arcsec terrain (gwd_dem, int16 21600 x 43200 = 1.78 GiB) is read only by init-gwd-orog
        ctx.tables.pop("gwd_dem", None)
        for _n in INIT_ONLY:                # init-derive-state locals (no step unit reads them)
            ctx.st.fields.pop(_n, None)
    bad = census(ctx)
    # init-only table kernels (phys_myj_tablett 28.8 KB/thread, bmj ttblq 12.3 KB) left ~7 GiB of device
    # local memory reserved for the whole run; give it back before the forecast (kernels.trim_local_memory)
    if _fit:
        from namgpu import kernels as _k  # noqa: PLC0415
        freed = _k.trim_local_memory("init")
        log(f"{UNIT}: local-memory reservation trimmed after init: {freed} GiB freed")
    total = time.perf_counter() - t_all
    rec = {"stages": stages, "total_s": round(total, 3), "nonfinite": bad}
    ctx.timings["init"] = rec
    if bad:
        raise RuntimeError(f"{UNIT}: NaN/Inf after init in {bad}")
    log(f"{UNIT}: done, {len(stages)} stages, {total:.2f} s")
    return rec
