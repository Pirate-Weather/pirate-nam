"""drv-dfi-filt-vars: the DFI filter / restore variable lists from the NAM filter table.

Fortran: nmm/module_DOMAIN_GRID_COMP.F90:5844-6055 (BUILD_FILT_BUNDLE reads the table named
filt_vars.txt; exnam_fcst.sh:638 copies parm/nam_filt_vars_norad.txt to it). NAM nam.v4.2.11,
83df4e13c1.
Kernel:  namgpu/kernels/drv_dfi_filt_vars.cu (dfi_filt_classify, one thread per table entry).

Table format (one entry per line): 'NAME'  F|R|B  'description'. Lines whose first character is
'#' and blank lines are skipped, as the Fortran does.
    F  filter (DFI filter bundle, cold and warm starts)
    R  restore (DFI restore bundle, restored after the filter window)
    B  both: restore when RESTART is true, filter otherwise
    any other code: ignored (Fortran FILT_TYP_INT = -999, no bundle entry)
Every entry, ignored or not, must be a state-table field: the Fortran calls FIND_VAR_INDX on each
parsed line before the F/R/B test, so an unknown name aborts even when its code is unused.
The 2-D and 3-D classes are the state table ranks. Rank 4 is refused: the Fortran R4D branch
(module_DOMAIN_GRID_COMP.F90:5938-5946) passes VARS(N)%R3D to ESMF_FieldCreate, and R3D is not
associated for a 4-D variable. No rank-4 entry exists in the current table.

Reads:   table path (default parm/nam_filt_vars_norad.txt; the NAM copies it to filt_vars.txt);
         restart from cfg.raw["restart"]. A substituted true/false is required: an unsubstituted
         placeholder (the parm value REST) or a missing key raises, it does not default to a cold
         start.
Writes:  ctx.tables["dfi_filt"]  dict:
         "filter_2d", "filter_3d", "restore_2d", "restore_3d": lists of names in table order
         (the Fortran bundle order); "filter_4d": always empty (rank 4 is refused);
         "entries": [(name, action)] for every accepted line (action 0 ignore, 1 filter, 2 restore);
         "bundle_fields": one dict per bundle member, in bundle order:
             name, bundle ("filter"|"restore"), rank, halo (IHALO, JHALO as the Fortran
             ESMF_FieldCreate totalUWidth/totalLWidth), ungridded (None for 2-D, else (1, extent):
             the level extent of the allocated state array, lbound taken as 1), shape (the
             allocated state array shape);
         "restart": bool; "table": str path.
         ctx.tables["DFI_FILT_ACT"] int32 device array [n] written by the kernel (same numbers).
Idempotent for the same (table, restart): built on the first call and reused. A call with a
different table or restart rebuilds the lists and relaunches the kernel. One launch on ctx.stream;
no host sync (the lists come from the host mirror, the device array is for device-side consumers).
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.config import PARM
from namgpu.state import all_specs
from namgpu.steps.drv_loop_limits import IHALO, JHALO

UNIT = "drv_dfi_filt_vars"
KERNEL = "dfi_filt_classify"
DEFAULT_TABLE = PARM / "nam_filt_vars_norad.txt"

CODE = {"F": 1, "R": 2, "B": 3}
ACT_IGNORE, ACT_FILTER, ACT_RESTORE = 0, 1, 2

_TRUE = {"true", ".true.", "t", "yes", "1"}
_FALSE = {"false", ".false.", "f", "no", "0"}

# Fortran: READ(UNIT=STRING,FMT=*) VBL_NAME, CH_FILTREST (list-directed; quotes optional).
_ENTRY = re.compile(r"""^\s*['"]?([^'"\s]+)['"]?\s+(\S+)""")


def read_entries(path) -> list[tuple[str, str]]:
    """Fortran reader: [(name, first character of the code token)] in file order.

    A non-comment, non-blank line with no code token raises. The Fortran READ fails there and
    keeps the previous VBL_NAME/CH_FILTREST, so it re-bundles a stale entry. That is a Fortran
    defect with no valid behaviour to mirror, so the wrapper refuses the table instead.
    """
    out: list[tuple[str, str]] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line[:1] == "#" or line.strip() == "":
            continue                              # Fortran: STRING(1:1)=='#' or blank
        m = _ENTRY.match(line)
        if m is None:
            raise ValueError(f"{path}:{lineno}: no 'NAME CODE' pair in {line!r}. The Fortran READ "
                             "fails here and re-bundles the previous entry; refused.")
        out.append((m.group(1), m.group(2)[:1]))
    return out


def host_actions(codes, restart: bool) -> list[int]:
    """Host mirror of dfi_filt_classify (same numbers as the device array)."""
    out = []
    for c in codes:
        if c == 1:
            out.append(ACT_FILTER)
        elif c == 2:
            out.append(ACT_RESTORE)
        elif c == 3:
            out.append(ACT_RESTORE if restart else ACT_FILTER)
        else:
            out.append(ACT_IGNORE)
    return out


def _restart_flag(ctx) -> bool:
    raw = getattr(ctx.cfg, "raw", None) or {}
    if "restart" not in raw:
        raise ValueError("drv-dfi-filt-vars: configfile has no 'restart' key; refusing to assume "
                         "a cold start")
    v = raw["restart"]
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    raise ValueError(f"drv-dfi-filt-vars: restart value {v!r} is not a substituted true/false "
                     "(the parm placeholder is REST, which exnam_fcst.sh replaces); refusing to "
                     "default to a cold start")


def _classify_names(names, actions, restart: bool, specs) -> dict:
    lists = {"filter_2d": [], "filter_3d": [], "filter_4d": [],
             "restore_2d": [], "restore_3d": []}
    # Fortran FIND_VAR_INDX runs on every parsed line before the F/R/B test: check all names first.
    for name in names:
        if specs.get(name) is None:
            raise ValueError(f"DFI filter variable {name!r} is not in the NAM state tables "
                             "(Fortran FIND_VAR_INDX stops here, module_VARS.F90:475)")
    kept = []
    for name, act in zip(names, actions):
        if act == ACT_IGNORE:
            continue
        rank = specs[name].rank
        if act == ACT_FILTER and rank == 4:
            raise ValueError(f"DFI filter variable {name!r} is rank 4: the Fortran R4D branch "
                             "passes VARS(N)%R3D to ESMF_FieldCreate (module_DOMAIN_GRID_COMP.F90:"
                             "5938-5946), which is not associated for a 4-D variable. Refused.")
        if act == ACT_FILTER:
            key = {2: "filter_2d", 3: "filter_3d"}.get(rank)
        else:
            key = {2: "restore_2d", 3: "restore_3d"}.get(rank)
        if key is None:
            raise ValueError(f"DFI {'filter' if act == ACT_FILTER else 'restore'} variable "
                             f"{name!r} has rank {rank}: the Fortran aborts "
                             "(module_DOMAIN_GRID_COMP.F90 SELECTED FILTER VARIABLE IS NOT ...)")
        lists[key].append(name)
        kept.append((name, act))
    lists["entries"] = kept
    lists["restart"] = bool(restart)
    return lists


def _bundle_fields(lists, st, specs) -> list[dict]:
    """Bundle members with the allocated state extent, rank check against the allocated array."""
    out = []
    for name, act in lists["entries"]:
        if name not in st.fields:
            raise RuntimeError(f"drv-dfi-filt-vars: {name} is not allocated in the state")
        arr = st.fields[name]
        rank = specs[name].rank
        if arr.ndim != rank:
            raise RuntimeError(f"drv-dfi-filt-vars: {name} is allocated rank {arr.ndim} but the "
                               f"state table gives rank {rank}")
        ung = None if rank == 2 else (1, int(arr.shape[0]))
        out.append({"name": name, "bundle": "filter" if act == ACT_FILTER else "restore",
                    "rank": rank, "halo": (IHALO, JHALO), "ungridded": ung,
                    "shape": tuple(int(s) for s in arr.shape)})
    return out


def run(ctx, table=None):
    import cupy as cp

    tables = ctx.tables
    path = Path(table) if table is not None else DEFAULT_TABLE
    restart = _restart_flag(ctx)
    key = (str(path), bool(restart))
    if "dfi_filt" in tables and tables.get("dfi_filt_key") == key:
        return tables["dfi_filt"]

    entries = read_entries(path)
    names = [n for n, _ in entries]
    codes = [CODE.get(c, 0) for _, c in entries]
    actions = host_actions(codes, restart)
    specs = all_specs()
    lists = _classify_names(names, actions, restart, specs)
    lists["table"] = str(path)
    lists["bundle_fields"] = _bundle_fields(lists, ctx.st, specs)

    n = len(codes)
    codes_d = cp.asarray(np.asarray(codes, dtype=np.int32).reshape(-1))
    act_d = cp.zeros((max(n, 1),), dtype=cp.int32)
    if n > 0:
        k_classify = kernels.get(UNIT, KERNEL)
        scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
        with scope:
            k_classify(((n + 255) // 256,), (256,),
                       (codes_d, np.int32(n), np.int32(1 if restart else 0), act_d))

    tables["dfi_filt"] = lists
    tables["dfi_filt_key"] = key
    tables["DFI_FILT_ACT"] = act_d
    return lists
