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


class State:
    """Device-resident NMMB state.  ``State(grid)`` allocates every 2-D/3-D/4-D field of the
    two tables on the current CuPy device, zero-filled; scalars and 1-D arrays live in
    ``self.scalars`` / ``self.cols`` (small, uploaded once).  Access fields as attributes
    (``st.T``, ``st.PINT``) or ``st[name]``."""

    def __init__(self, grid, xp=None, ntracers: int | None = None, names=None):
        if xp is None:
            import cupy as xp  # noqa: N813
        if ntracers is None:
            ntracers = num_tracers_total()
        self.xp = xp
        self.grid = grid
        self.specs = all_specs()
        self.fields: dict[str, object] = {}
        self.scalars: dict[str, float] = {}
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
        """Scratch or unit-owned array not in the NAM tables (e.g. Adams-Bashforth history)."""
        if name not in self.fields:
            self.fields[name] = self.xp.zeros(shape, dtype=dtype)
        return self.fields[name]

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
