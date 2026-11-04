"""Where WOOF-NAM finds its files.

Shipped inside the package (so a plain `pip install` has them): the NAM parm files (parm/), the
tables and NOAA fix files (data/, data/fix/), the unit list (UNITS.json) and the reused ArWen files
(namgpu/third_party/arwen/).

Your working files (cases/, static/, runs/, scratch/) live under the work root:
  1. $NAMGPU_ROOT when it is set (env.sh and env-box.sh set it to the source checkout);
  2. else the source checkout, when the package is imported from one (python -m ... in a clone,
     or pip install -e .);
  3. else the directory the command was started in.
The work root is fixed when namgpu is first imported, so a later chdir (the radiation init runs
inside the fix directory) never moves it.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
PARM = PKG / "parm"
DATA = PKG / "data"
FIX = DATA / "fix"                 # NOAA fix/am files (radiation, CO2, ozone) and fix/noah tables
UNITS_JSON = PKG / "UNITS.json"
ARWEN = PKG / "third_party" / "arwen"

_START_CWD = Path.cwd()


def checkout_root() -> Path | None:
    """The source checkout this package is imported from, or None for an installed wheel."""
    root = PKG.parent
    if (root / "pyproject.toml").is_file() and (root / "tools").is_dir():
        return root
    return None


def work_root() -> Path:
    env = os.environ.get("NAMGPU_ROOT")
    if env:
        return Path(env)
    return checkout_root() or _START_CWD


def case_dir(case: str | os.PathLike) -> Path:
    """A case given as a folder path (cases/2025-03-14_12z, /data/x/2025-03-14_12z) or as a bare
    name, which is looked up under <work root>/cases/."""
    p = Path(case)
    if p.is_dir() or len(p.parts) > 1:
        return p.resolve()
    return work_root() / "cases" / p


def box_mode() -> bool:
    """Our own shared-card plumbing (card lease, card lock hand-off): on only with WOOF_NAM_BOX=1.
    A stranger's run never touches it."""
    return os.environ.get("WOOF_NAM_BOX", "").strip() not in ("", "0", "false", "no")
