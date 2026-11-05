"""Every NMMB prognostic and diagnostic array, with shape, units, staggering.

Source of truth: the NAM's own state tables ``parm/nam_solver_state.txt`` (dynamics/solver
internal state) and ``parm/nam_phy_state.txt`` (physics), copied unmodified.  Each table row is
    'NAME'  History  Restart  Owned  Import  eXport  Time_ser  'Description (units)'
and the rank of each block comes from its ``###  ND real|integer`` section header.

Layout rule (CONTRACT.md): Fortran (i, j, l) -> CuPy [l, j, i], float32 (int32 for integer
fields).  L=1 (index 0) is the model top.  2-D fields are [j, i].  4-D fields (TRACERS ...)
are [n, l, j, i].  V-point fields have the H shape; the NE-most row/column is unused.

Shapes not derivable from the table (vertical extent LM+1 or NSOIL, V staggering, boundary
strips) are in the override tables below; a unit that adds an array adds it there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PARM
from .steps.drv_tracer_index import NUM_TRACERS_MET  # Fortran 943 (configfile num_tracers_met is overridden)

V_POINTS = {
    "U", "V", "UP", "VP", "TCU", "TCV", "U10", "V10", "UZ0", "VZ0", "DUDT", "DVDT",
    "UBOT", "VBOT", "U_PHY", "V_PHY", "UPB", "VPB",
}
LEVELS_LMP1 = {"PINT", "W_TOT", "Z", "EXCH_H", "EL_PBL_I", "PSGDT"}
LEVELS_NSOIL = {"STC", "SMC", "SH2O", "SLDPTH", "SOIL1DIN"}
NSOIL = 4

# 1-D vertical arrays that are not [lm] (SG1/SG2/PSG1 are interfaces).
ONE_D_LMP1 = {"SG1", "SG2", "PSG1"}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    rank: int              # 0..4 as declared in the table
    dtype: str             # "float32" | "int32"
    history: bool
    restart: bool
    description: str
    stagger: str           # "H" | "V" | "-"
    source: str            # which table


def _units(desc: str) -> str:
    m = re.search(r"\(([^()]*)\)\s*$", desc)
    return m.group(1) if m else ""


def parse_state_table(path: Path) -> list[FieldSpec]:
    rank, dtype = 0, "int32"
    out: list[FieldSpec] = []
    row = re.compile(r"^'([A-Za-z0-9_]+)'\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+'(.*)'?")
    for line in Path(path).read_text().splitlines():
        hdr = re.match(r"^###\s+(\d)D\s+(integer|real)", line.strip())
        if hdr:
            rank = int(hdr.group(1))
            dtype = "int32" if hdr.group(2) == "integer" else "float32"
            continue
        if line.startswith("######") and "TRACKER" in line:
            break   # tracker variables: not ported (no inline vortex tracker in the parent)
        m = row.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        out.append(FieldSpec(
            name=name, rank=rank, dtype=dtype, history=m.group(2) == "H",
            restart=m.group(3) == "R", description=m.group(8).rstrip("'"),
            stagger=("V" if name in V_POINTS else "H") if rank >= 2 else "-",
            source=Path(path).name))
    return out


def all_specs() -> dict[str, FieldSpec]:
    specs: dict[str, FieldSpec] = {}
    for table in ("nam_solver_state.txt", "nam_phy_state.txt"):
        for s in parse_state_table(PARM / table):
            specs.setdefault(s.name, s)
    return specs


def shape_of(spec: FieldSpec, im: int, jm: int, lm: int, ntracers: int):
    n = spec.name
    if spec.rank == 0:
        return ()
    if spec.rank == 1:
        if n in ONE_D_LMP1:
            return (lm + 1,)
        if n in LEVELS_NSOIL:
            return (NSOIL,)
        return (lm,)
    if spec.rank == 2:
        return (jm, im)
    if spec.rank == 3:
        if n in LEVELS_LMP1:
            return (lm + 1, jm, im)
        if n in LEVELS_NSOIL:
            return (NSOIL, jm, im)
        return (lm, jm, im)
    return (ntracers, lm, jm, im)


def num_tracers_total() -> int:
    """Fortran NUM_TRACERS_TOTAL = met + chem + water for the configured microphysics."""
    from .config import load  # noqa: PLC0415
    from .steps.drv_tracer_index import host_index  # noqa: PLC0415
    cfg = load()
    return int(host_index(cfg.microphysics, cfg.raw.get("num_tracers_chem", 0))["NUM_TRACERS_TOTAL"])


# Step scratch (32 GB fit, 2026-11-04). Arrays the dynamics units add with st.add() that are
# written and read inside ONE step's dynamics group only (Fortran locals of ADV1/ADV2/PGFORCE/DHT/CDZDT/...).
# Holding them as persistent state cost ~11.8 GiB on top of the 12.9 GiB NAM state, all of it idle through
# physics and radiation, so the 12 km domain did not fit a 32 GB card. Proof that nothing reads them across
# steps: NAMGPU_SCRATCH_POISON=1 A/B (every state field identical). Excluded on purpose: RTOP (init writes it),
# T1 (name also used by physics), PDWDT and CDWDT_WW (previous-step correction factor).
STEP_SCRATCH = frozenset({
    "U1D", "V1D", "PFXX1", "PFYX1", "PFNEX1", "PFNWX1", "PFXY1", "PFYY1", "PFNEY1", "PFNWY1",
    "PFX", "PFY", "PFNE", "PFNW", "PDOP", "U2D", "V2D", "ADV1_RCMT", "ADV1_CRT",
    "RCMW", "CRW", "RSTU", "RSTV", "PDOP_UV", "TRACERS_SQRT", "SS1", "ADV2_RCMS", "ADV2_CRS",
    "TDIV", "APEL_3D", "FIM_3D", "DFI_3D", "PCX", "PCY", "PCNE", "PCNW",
    "PGX_3D", "PGY_3D", "PGNE_3D", "PGNW_3D", "TRACERS_TEND",
    # 2026-11-04: HDIFF deformation and slope masks (dyn-hdiff-coef writes, dyn-hdiff-apply and
    # dyn-cdwdt-diffusion read in the same step; cdwdt-diffusion raises if DEF3D is missing) and RTOP (dyn-pgforce-geopot
    # writes it every step before cdzdt/vtoa/vsound read it; the init copy is dropped with INIT_ONLY).
    # Write-before-read checked with the NAMGPU_SCRATCH_POISON=1 A/B.
    "DEF3D", "FMLX_3D", "FMLY_3D", "RTOP",
})
# ctx.tables work arrays of the dynamics group (dyn-hdiff-apply diff/update buffers), dropped with STEP_SCRATCH
# (+ dyn-mono-fix per-row partial sums, float64, rewritten in full on every call). hdiff_coef_* are host scalars
# (no device memory) read once; dropping them cost a device-to-host read every step, so they stay.
STEP_SCRATCH_TABLE_PREFIXES = ("hdiff_", "MONO_WPART_", "MONO_ROWP")
STEP_SCRATCH_TABLE_KEEP = ("hdiff_coef_",)
# History-only device buffers (written in full by io-wind-rotate / io-quasipost-driver / io-history-fields
# at each output time, staged to pinned host memory by io-history on the same stream): ~2.9 GiB.
HISTORY_SCRATCH = frozenset({
    "H_CW_SPEC", "H_FICE_SPEC", "H_FRAIN_SPEC", "H_P", "H_T", "H_QVAPOR", "H_QCLOUD", "H_QRAIN",
    "H_QICE", "H_QSNOW", "H_PH", "UEARTH", "VEARTH", "QP_PRESTV", "QP_PRESZ", "QP_GROUND_MASK",
    # the 2-D history buffers of the same units (io-wind-rotate, io-quasipost-*, io-history-fields)
    "UE10", "VE10", "H_PSFC", "H_HGT", "H_RAINNC", "H_RAINC", "H_T2", "H_COMPOSITE",
    "QP_GROUND_LEVEL", "QP_PMSL", "QP_RELAXMASK", "QP_RELAXWORK", "QP_RELAX_NEXT", "RELAXWORK_NEXT", "MEMBRANE_MSLP",
    "P850Z", "P700Z", "QP_KS850", "QP_KS700", "QP_KS500", "QP_DUMMY1", "QP_DUMMY2",
    "P500U", "P500V", "P700WIND", "P700RV", "P700U", "P700V", "P850WIND", "P850RV", "P850U", "P850V", "M10WIND", "M10RV",
    # the one shared 3-D history stream buffer
    "H_STREAM",
})
# Init-only memory that the forecast held to the end (2026-11-04). Each name is
# written and read only inside the init chain: the GFS start's vertical-interpolation scratch
# (init-gfs-vinterp; the boundary builder gfs_bocos uses its own private State) and the decoded surface
# records (ctx.tables["raw_sfc"], read only by init-native-sfc). 130 MB at 333x333 (1.2 KB per column) and
# 70 MB at 12 km. Dropped at the end of the init chain and again where a forecast starts, because a saved init
# (slab_init) made before this change still carries them.
INIT_LEFTOVER_FIELDS = ("P3D_OUT", "P3DV_IN", "P3DV_OUT", "QTMP2", "CWM_TMP", "FRIMEF_LEV", "PSFC_OUTV", "PDVP")
INIT_LEFTOVER_TABLES = ("raw_sfc",)


def release_init_leftovers(ctx) -> int:
    """Drop the init-only arrays above from ctx.st and ctx.tables. Idempotent. Returns the bytes released."""
    n = 0
    st = getattr(ctx, "st", None)
    if st is not None:
        for k in INIT_LEFTOVER_FIELDS:
            a = st.fields.pop(k, None)
            if a is not None and not any(b is a for b in st.fields.values()):
                n += int(getattr(a, "nbytes", 0))
    tabs = getattr(ctx, "tables", None)
    if isinstance(tabs, dict):
        for k in INIT_LEFTOVER_TABLES:
            v = tabs.pop(k, None)
            if isinstance(v, dict):
                n += sum(int(getattr(x, "nbytes", 0)) for x in v.values())
    return n
# Physics locals released right after the unit that owns them (driver: after that unit's run()).
UNIT_SCRATCH = {
    # MYJ vertical diffusion locals (VDIFH / VDIFV inside MYJPBL, called by phys-turbl-driver)
    "phys-turbl-driver": frozenset({
        "MYJ_TEST_D", "MYJ_TEST_RKM", "MYJ_TEST_RHO", "MYJ_TEST_AKMS_DENS", "MYJ_TEST_LMH", "MYJ_TEST_Z",
        "MYJ_TEST_UK", "MYJ_TEST_VK", "MYJ_TEST_DUDT", "MYJ_TEST_DVDT",
        "THK", "QK", "CWMK", "ZHK", "RHOK", "AKHK", "AKHS_DENS",
        "DELP"}),                     # TURBL local (pressure depth), read by MYJ and GWD inside this unit only
    # ADV1 Adams-Bashforth T (adv1_t_ab_col fills it, the same unit reads it)
    "dyn-adv1-horiz-t": frozenset({"T1"}),
    # BMJ column scratch (CUCNVC)
    "phys-bmj-driver": frozenset({"BMJ_CT_DPRS", "BMJ_SCAL", "BMJ_FLAG"}),
    # GSMDRIVE locals (phys_fer_driver._SCRATCH3)
    "phys-fer-driver": frozenset({"DZ8W", "RHO_PHY", "P_PHY", "PI_PHY", "TH_PHY"}),
}
# Physics driver-level arrays (TURBL / CUCNVC / GSMDRIVE locals that the port keeps on ctx.st), released after
# their LAST user in the PHYSICS plan (Driver._phys_last_use maps helper modules to their driver unit). Radiation
# runs FIRST in the physics group, so while it runs these ~3-4 GiB sat idle from the previous step.
PHYS_SCRATCH = frozenset({
    "TURBL_Z", "RTHBLTEN", "RQBLTEN", "RQCBLTEN", "RQIBLTEN", "DUDT_GWD", "DVDT_GWD", "RTHCUTEN", "RQCUTEN",
    "PHINT", "DZ", "TH", "EXNER", "RR", "PHMID", "U_PHY", "V_PHY", "DUDT", "DVDT",
})
# init-only arrays of init-derive-state (no step unit names them)
INIT_ONLY = ("TOLD", "TADJ", "RTOP")     # RTOP: dyn-pgforce-geopot rewrites it before any reader (STEP_SCRATCH)
# Init stage locals dropped right after their stage, on every memory path (holding them buys no speed):
# {stage: (state names, ctx.tables keys)}
INIT_STAGE_ONLY = {
    # init-gwd-orog accumulators (namgpu/init/init_gwd_orog.py FLOAT_SCRATCH, INT_SCRATCH); no other unit names them
    "init-gwd-orog": (("GWD_SUMTOPO", "GWD_SDX2", "GWD_SDY2", "GWD_SDXY", "GWD_TOPO", "GWD_SUM2", "GWD_SUM4",
                       "GWD_COUNT", "GWD_NUL", "GWD_NLL", "GWD_NUR", "GWD_NLR", "GWD_NUL2", "GWD_NLL2",
                       "GWD_NUR2", "GWD_NLR2", "GWD_CNUL", "GWD_CNLL", "GWD_CNUR", "GWD_CNLR"), ("gwd_dem",)),
    # the decoded surface records (init-grib2-decode / gfs_start); init-native-sfc is their only reader
    "init-native-sfc": ((), ("raw_sfc",)),
}
# TDFI memory, dropped when the filter ends (drv_time_loop), on every memory path: the running sums (DFI, ctx.filt),
# the physics save (DFI_PHY_SAVE_*), the 0 h reflectivity work arrays. Nothing reads them after phy_restore.
TDFI_STATE = ("DFI_PHY_SAVE_2D", "DFI_PHY_SAVE_3D")
TDFI_TABLES = ("DFI", "dfi_refl_curefl", "fer_calmict_dbz", "dfi_phy_save", "dfi_phy_restore")
import os as _os  # noqa: E402
POISON = _os.environ.get("NAMGPU_SCRATCH_POISON", "") == "1"
_POISON_VALUES = (float("nan"), 1.0e30, -1.0e30)

# Memory path. "fit" = the 32 GB fit above (step scratch released after its last user, radiation working set
# dropped after each call, init-only memory returned, RRTM in 8-row chunks): the whole 12 km domain with
# every unit on runs on one 32 GB card. "full" = every array held for the whole run (needs ~58-76 GB, a little
# faster). Same physics, same units on both paths; the released arrays are proven write-before-read
# (NAMGPU_SCRATCH_POISON=1 A/B: every state field identical). NAMGPU_MEMORY_PATH=auto|fit|full (default auto:
# fit when the card has less than FIT_BELOW_GIB). NAMGPU_SCRATCH_ARENA=1/0 is the older spelling of fit/full.
FIT_BELOW_GIB = 64.0
FIT_RRTM_ROWS = 8
_MEMORY_PATH: dict | None = None


def memory_path() -> dict:
    """{"path": "fit"|"full", "how": "auto"|"set", "card_gib": float, "reason": str}, resolved once per
    process on the current CuPy device and cached. Read by the driver, init, radiation and the run receipt."""
    global _MEMORY_PATH
    if _MEMORY_PATH is not None:
        return _MEMORY_PATH
    want = _os.environ.get("NAMGPU_MEMORY_PATH", "").strip().lower()
    old = _os.environ.get("NAMGPU_SCRATCH_ARENA", "").strip()
    if not want and old in ("0", "1"):
        want = "fit" if old == "1" else "full"
    try:
        import cupy as _cp
        card = _cp.cuda.Device().mem_info[1] / 2**30
    except Exception:  # noqa: BLE001  (no CUDA: CPU-only imports and tests)
        card = 0.0
    _MEMORY_PATH = resolve_memory_path(want, card)
    return _MEMORY_PATH


def resolve_memory_path(want: str, card: float) -> dict:
    """Pure part of memory_path(): the setting (auto/fit/full/empty) and the card size in GiB -> the path."""
    want = (want or "").strip().lower()
    if want in ("fit", "full"):
        path, how = want, "set"
        reason = f"NAMGPU_MEMORY_PATH={want}"
    elif want in ("", "auto"):
        path, how = ("fit" if card < FIT_BELOW_GIB else "full"), "auto"
        reason = f"card {card:.1f} GiB {'<' if path == 'fit' else '>='} {FIT_BELOW_GIB:.0f} GiB"
    else:
        raise ValueError(f"NAMGPU_MEMORY_PATH={want!r}: use auto, fit or full")
    return {"path": path, "how": how, "card_gib": round(card, 1), "reason": reason}


def fit_on() -> bool:
    return memory_path()["path"] == "fit"


def _empty_scratch() -> bool:
    # released scratch comes back without the zero fill on the fit path (contents proven irrelevant by the
    # poison A/B); NAMGPU_SCRATCH_EMPTY=0/1 overrides
    v = _os.environ.get("NAMGPU_SCRATCH_EMPTY", "")
    return v == "1" if v in ("0", "1") else fit_on()


class State:
    """Device-resident NMMB state.  ``State(grid)`` allocates every 2-D/3-D/4-D field of the
    two tables on the current CuPy device, zero-filled; scalars and 1-D arrays live in
    ``self.scalars`` / ``self.cols`` (small, uploaded once).  Access fields as attributes
    (``st.T``, ``st.PINT``) or ``st[name]``."""

    def __init__(self, grid, xp=None, ntracers: int | None = None, names=None):
        if xp is None:
            import cupy as xp  # noqa: N813
        if getattr(xp, "__name__", "") == "cupy":
            from namgpu import devmem  # noqa: PLC0415
            devmem.setup()                  # the run's allocator, before the first state array (devmem.py)
        if ntracers is None:
            ntracers = num_tracers_total()
        self.xp = xp
        self.grid = grid
        self.specs = all_specs()
        self.fields: dict[str, object] = {}
        self.scalars: dict[str, float] = {}
        self._poison_k = 0
        for s in self.specs.values():
            if names is not None and s.name not in names:
                continue
            shp = shape_of(s, grid.im, grid.jm, grid.lm, ntracers)
            if s.rank >= 2:
                self.fields[s.name] = xp.zeros(shp, dtype=s.dtype)
            elif s.rank == 1:
                self.fields[s.name] = xp.zeros(shp, dtype=s.dtype)
            else:
                self.scalars[s.name] = 0

    def __getattr__(self, name):
        f = self.__dict__.get("fields", {})
        if name in f:
            return f[name]
        raise AttributeError(name)

    def __getitem__(self, name):
        return self.fields[name]

    def add(self, name, shape, dtype="float32"):
        """Scratch or unit-owned array not in the NAM tables (e.g. Adams-Bashforth history).

        Names in STEP_SCRATCH / HISTORY_SCRATCH are pool temporaries: the driver drops them after the
        dynamics group (release_scratch("dyn")) or after the history write (release_scratch("history")),
        and the next add() hands out a fresh zero-filled array, i.e. every step sees them exactly as the
        first step did. NAMGPU_SCRATCH_POISON=1 fills them with NaN / +1e30 / -1e30 instead (by step
        parity) to prove no unit reads a scratch value it did not write in the same step."""
        if name not in self.fields:
            if POISON and (name in STEP_SCRATCH or name in HISTORY_SCRATCH or name in PHYS_SCRATCH
                           or any(name in v for v in UNIT_SCRATCH.values())):
                a = self.xp.empty(shape, dtype=dtype)
                if a.dtype.kind == "f":
                    a.fill(_POISON_VALUES[self._poison_k % 3])
                else:
                    a.fill(-2147483647)
                self.fields[name] = a
            elif _empty_scratch() and (name in STEP_SCRATCH or name in HISTORY_SCRATCH or name in PHYS_SCRATCH
                            or any(name in v for v in UNIT_SCRATCH.values())):
                # contents proven irrelevant (poison A/B): skip the zero fill
                self.fields[name] = self.xp.empty(shape, dtype=dtype)
            else:
                self.fields[name] = self.xp.zeros(shape, dtype=dtype)
        return self.fields[name]

    def release_scratch(self, which: str) -> int:
        """Drop the step-scratch arrays of `which` ("dyn" or "history") from the state; their memory goes
        back to the CuPy pool (stream-ordered, so later kernels on the same stream may reuse it).
        Returns the bytes released."""
        names = (STEP_SCRATCH if which == "dyn" else HISTORY_SCRATCH if which == "history"
                 else UNIT_SCRATCH.get(which, ()))
        n = 0
        for k in names:
            a = self.fields.pop(k, None)
            if a is not None:
                n += int(a.nbytes)
        if which == "dyn":
            self._poison_k += 1
        return n

    def nbytes(self) -> int:
        # aliases (one array under two names) count once
        return int(sum(a.nbytes for a in {id(a): a for a in self.fields.values()}.values()))

    def randomize(self, seed: int = 0):
        """Full-size random but physically shaped state for the unit 'runs without NaN' check:
        T 200-310 K, Q 0-0.02, winds +-30 m/s, PD 50-100 kPa, PINT built from the grid's
        hybrid coefficients, everything else uniform [0, 1)."""
        xp = self.xp
        rng = xp.random.default_rng(seed)
        for name, a in self.fields.items():
            if a.dtype.kind == "f":
                a[...] = rng.random(a.shape, dtype=xp.float32)
        g = self.grid
        if "T" in self.fields:
            self.T[...] = 200.0 + 110.0 * rng.random(self.T.shape, dtype=xp.float32)
        if "Q" in self.fields:
            self.Q[...] = 0.02 * rng.random(self.Q.shape, dtype=xp.float32)
        for w in ("U", "V"):
            if w in self.fields:
                self.fields[w][...] = 60.0 * rng.random(self.fields[w].shape, dtype=xp.float32) - 30.0
        if "PD" in self.fields:
            self.PD[...] = 50000.0 + 50000.0 * rng.random(self.PD.shape, dtype=xp.float32)
            if "PINT" in self.fields and g.sg1 is not None:
                sg1 = xp.asarray(g.sg1)[:, None, None]
                sg2 = xp.asarray(g.sg2)[:, None, None]
                self.PINT[...] = sg1 * g.pdtop + sg2 * self.PD[None] + g.pt
        return self


def summary() -> str:
    specs = all_specs()
    by = {}
    for s in specs.values():
        by.setdefault((s.source, s.rank), 0)
        by[(s.source, s.rank)] += 1
    return "\n".join(f"{src} rank {r}: {n}" for (src, r), n in sorted(by.items()))


if __name__ == "__main__":
    print(summary())
