"""How many CPU cores this process may really use: the CPU affinity set, capped by the cgroup CPU quota.

os.cpu_count() reports the host's cores. A rented container can see all of them and be allowed far fewer: the
container (2026-11-04) shows 96 cores with cgroup cpu.max 2304000/100000 (23 cores). Pools sized by
os.cpu_count() there started 96 grib2_dump processes for one IFS boundary time and starved a forecast running
beside it (its history writer stalled for minutes at 73 % CPU pressure). Every worker pool sizes from usable().
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def usable() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max",):                     # cgroup v2
        try:
            quota, period = Path(path).read_text().split()[:2]
            if quota != "max":
                n = min(n, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    try:                                                          # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            n = min(n, max(1, q // p))
    except (OSError, ValueError):
        pass
    return max(1, n)
