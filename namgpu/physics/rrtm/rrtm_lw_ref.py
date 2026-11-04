"""module_radlw_ref reference gas table shared by the RRTM LW taumol units.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radlw_datatb.f:733-905, module_radlw_ref:
    real (kind=kind_phys), dimension(7,59) :: chi_mls       ! chi_mls(gas, reference level)
    real (kind=kind_phys), dimension(59) :: pref, preflog, tref
The port stores chi_mls level-major: chi[level-1, gas-1] with shape (59, 7), from
namgpu/data/rrtm_lw/ref_chi_mls.npz (tools/extract_rrtm_lw_kgb07.py). preflog and tref come from
namgpu/data/rrtm_lw/ref.npz, which matches the radlw_datatb.f:761-790 DATA statements to float32
rounding (max relative error 5.6e-8). Every consumer goes through load_chi_mls() or
load_preflog_tref(); no stand-in values are used anywhere. The layout
conversion for tables stored gas-major (namgpu/data/rrtm_lw_kgb05.npz is (7, 59)) lives in
normalize_chi_mls() and nowhere else.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "ref_chi_mls.npz"
NPZ_REF = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw" / "ref.npz"
NREF = 59       # reference pressure levels
NGAS_REF = 7    # reference gases


def normalize_chi_mls(chi):
    """Return chi_mls as float32 [59, 7] (level, gas). Accepts [59, 7] or the gas-major [7, 59]."""
    a = np.asarray(chi)
    if a.shape == (NGAS_REF, NREF):
        a = a.T
    if a.shape != (NREF, NGAS_REF):
        raise ValueError(f"chi_mls shape {a.shape} is neither {(NREF, NGAS_REF)} nor {(NGAS_REF, NREF)}")
    return np.ascontiguousarray(a, dtype=np.float32)


@lru_cache(maxsize=1)
def load_preflog_tref():
    """preflog and tref (module_radlw_ref, radlw_datatb.f:761-790), float32 [59] each, read once."""
    with np.load(NPZ_REF) as d:
        pl = np.ascontiguousarray(d["preflog"], dtype=np.float32)
        tr = np.ascontiguousarray(d["tref"], dtype=np.float32)
    if pl.shape != (NREF,) or tr.shape != (NREF,):
        raise ValueError(f"preflog/tref shapes {pl.shape} {tr.shape}, want ({NREF},)")
    pl.setflags(write=False)
    tr.setflags(write=False)
    return pl, tr


@lru_cache(maxsize=1)
def load_chi_mls() -> np.ndarray:
    """The module_radlw_ref chi_mls table, read once. The returned array is read-only and the same object
    on every call, so device copies can be validated by identity."""
    with np.load(NPZ) as d:
        chi = normalize_chi_mls(d["chi_mls"])
    chi.setflags(write=False)
    return chi
