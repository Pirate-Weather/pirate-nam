"""drv-step-plan: the SOLVER_RUN step plan as data, checked against UNITS.json and the Fortran order.

Fortran: nmm/module_SOLVER_GRID_COMP.F90:2453-7168 (NAM nam.v4.2.11, 83df4e13c1), SOLVER_RUN call
order and cadences. The plan itself is held in namgpu/driver.py (FIRST_STEP, DYNAMICS, PHYSICS,
DIAGNOSTICS); this unit does not copy it. It checks it.

check_plan() returns (problems, pending):
- problems (hard, the plan is wrong):
  * a plan unit missing from UNITS.json; a plan entry naming a unit whose mode is "skip";
  * an unknown cadence key (driver.CADENCE_KEYS is the one list; Driver._due handles exactly those);
  * a "first" cadence outside the first-step block;
  * a plan entry kwarg that the unit's run() does not accept (checked by signature when the wrapper
    module imports);
  * a PHYSICS entry out of Fortran order (ORDER below: the SOLVER_RUN physics call sequence);
  * a UNITS.json entry without a wrapper.
- pending (not an error here): a plan unit whose wrapper file is not written yet. Those units are
  still todo. driver.Driver.check_ready() refuses to run until every one exists.

GPU: kernel drv_step_plan_nonfinite_count (kernels/drv_step_plan.cu) counts NaN/Inf on the device.
nonfinite_count() launches it and returns the device counter; it does not synchronise.

run(ctx): the step plan is not itself a plan entry, so run() is only a guard that performs the
host-side check once and caches the result. It raises on problems and never touches the device.
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from namgpu import driver as _driver
from namgpu import kernels

UNIT = "drv_step_plan"
ROOT = Path(__file__).resolve().parents[2]
KNOWN_CADENCES = set(_driver.CADENCE_KEYS)

# Fortran SOLVER_RUN physics call order (non-GFS branch, 4929-6077). Each marker is (unit id, kwargs
# subset). The plan's PHYSICS list must contain them in this order; a missing or out-of-order marker
# is a problem. Source lines are the Fortran call sites.
PHYSICS_ORDER = [
    ("io-max-fields", {}),                               # 5000
    ("drv-update-water", {}),                            # 5202
    ("phys-rad-driver", {}),                             # 5312
    ("drv-bucket-reset", {"blocks": "nclod"}),           # 5367
    ("drv-rdtemp", {}),                                  # 5384
    ("drv-bucket-reset", {"blocks": "post_rdtemp"}),     # 5418-5467
    ("phys-turbl-driver", {}),                           # 5490
    ("drv-h-to-v-tend", {}),                             # 5586
    ("drv-cltend", {"icltend": -1}),                     # 5708
    ("phys-bmj-driver", {}),                             # 5732
    ("phys-fer-driver", {}),                             # 5880
    ("drv-cltend", {"icltend": 0}),                      # 5928
    ("drv-cltend", {"icltend": 1}),                      # 6008
    ("drv-tqadjust", {}),                                # 6022
    ("drv-bucket-reset", {"blocks": "post_turb"}),       # 6348-6384
]

_checked = False


def _units() -> dict:
    return {u["id"]: u for u in json.loads((ROOT / "UNITS.json").read_text())["units"]}


def _matches(entry, uid, want) -> bool:
    if entry[0] != uid:
        return False
    kw = entry[2] if len(entry) > 2 else {}
    return all(kw.get(k) == v for k, v in want.items())


def _check_order(physics) -> list[str]:
    problems = []
    pos = -1
    for uid, want in PHYSICS_ORDER:
        idx = [i for i, e in enumerate(physics) if _matches(e, uid, want)]
        if not idx:
            problems.append(f"PHYSICS: missing {uid} {want or ''} (Fortran order marker)")
            continue
        if idx[0] <= pos:
            problems.append(f"PHYSICS: {uid} {want or ''} out of Fortran order (entry {idx[0]} after {pos})")
        pos = max(pos, idx[0])
    return problems


def _check_kwargs(entry, path, problems, pending) -> None:
    kw = entry[2] if len(entry) > 2 else {}
    if not kw:
        return
    uid = entry[0]
    try:
        mod = importlib.import_module(path)
    except Exception as e:  # noqa: BLE001  (wrapper not importable on this host: check at run time)
        pending.append(f"{uid}: kwargs {sorted(kw)} not checked, wrapper import failed ({type(e).__name__})")
        return
    params = inspect.signature(mod.run).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    for k in kw:
        if k not in params:
            problems.append(f"{uid}: plan passes kwarg {k!r} that {path}.run does not accept")


def check_plan() -> tuple[list[str], list[str]]:
    units = _units()
    problems: list[str] = []
    pending: list[str] = []
    entries = _driver.FIRST_STEP + _driver.DYNAMICS + _driver.PHYSICS + _driver.DIAGNOSTICS
    reported_missing = set()
    for entry in entries:
        uid, key = entry[0], entry[1]   # kwargs ride as a third element
        if key not in KNOWN_CADENCES:
            problems.append(f"{uid}: unknown cadence key {key!r}")
        if key == "first" and (uid, key) not in _driver.FIRST_STEP:
            problems.append(f"{uid}: 'first' cadence outside the first-step block")
        if uid not in units:
            if uid not in reported_missing:
                reported_missing.add(uid)
                problems.append(f"{uid}: not in UNITS.json")
            continue
        u = units[uid]
        if u.get("mode") == "skip":
            problems.append(f"{uid}: plan names a unit with mode 'skip'")
            continue
        mod = u.get("wrapper")
        if not mod:
            problems.append(f"{uid}: UNITS.json has no wrapper")
            continue
        path = ROOT.joinpath(*mod.split(".")).with_suffix(".py")
        if not path.exists():
            pending.append(f"{uid}: wrapper not written yet ({mod})")
            continue
        _check_kwargs(entry, mod, problems, pending)
    problems += _check_order(_driver.PHYSICS)
    return problems, pending


def run(ctx):
    """Host-side guard, done once per process. No device work, no sync."""
    global _checked
    if _checked:
        return
    problems, _pending = check_plan()
    if problems:
        raise RuntimeError("step plan inconsistent:\n  " + "\n  ".join(problems))
    _checked = True


def nonfinite_count(arr):
    """Device counter (uint32, shape (1,)) of NaN/Inf values in a float32 array. No sync."""
    import cupy as cp
    if arr.dtype != np.float32:
        raise TypeError("nonfinite_count expects float32")
    if not arr.flags.c_contiguous:   # reshape(-1) would copy; the scan must read the caller's array
        raise ValueError("nonfinite_count expects a C-contiguous array (no implicit copy)")
    flat = arr.reshape(-1)
    n = int(flat.size)
    counter = cp.zeros(1, dtype=cp.uint32)
    block = 256
    grid = min(max(1, (n + block - 1) // block), 65535)
    k = kernels.get(UNIT, "drv_step_plan_nonfinite_count")
    k((grid, 1, 1), (block, 1, 1), (flat, np.int64(n), counter))
    return counter
