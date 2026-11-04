"""Where the gpuwm Rust bridges live (grib2_inventory, grib2_dump, rw_wrfbatch, libnetcdf_writer.so).

Order: $GPUWM_BRIDGES if set; else the copy inside the installed gpuwm wheel (gpuwm/libexec/bridges,
which `pip install` puts there); else ~/.gpuwm/bridges (older gpuwm installs).
"""
from __future__ import annotations

import os
from pathlib import Path


def bridges_dir() -> Path:
    env = os.environ.get("GPUWM_BRIDGES")
    if env:
        return Path(env)
    try:
        import gpuwm  # noqa: PLC0415
        wheel = Path(gpuwm.__file__).resolve().parent / "libexec" / "bridges"
        if wheel.is_dir():
            return wheel
    except ImportError:
        pass
    return Path.home() / ".gpuwm" / "bridges"
