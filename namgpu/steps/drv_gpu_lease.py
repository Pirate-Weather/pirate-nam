"""drv-gpu-lease: take and release a shared-card pool lease around a run (optional).

Internal plumbing for our own shared cards, off by default: the lease is taken only with
WOOF_NAM_BOX=1 (env-box.sh sets it). Without it gpu_lease() is a no-op on every machine.

No Fortran counterpart (UNITS.json mode "new"). Kernel: namgpu/kernels/drv_gpu_lease.cu
(lease_probe, one thread, one launch).

The lease is pool-v3's card lease, ~/gpuwm-work/pool-v3/run.sh (README.txt). A run asks for
a per-card VRAM budget and an estimate; run.sh admits it when the shared 5090 has room. The
lease is held by a small holder process that run.sh launches and that prints
"LEASED <CUDA_VISIBLE_DEVICES>" once admitted, then blocks reading its stdin. Closing stdin
releases the lease, so:
  - the lease covers exactly the `with gpu_lease(...)` block;
  - if the driving process dies (power cut, crash), the pipe closes and the lease is released.
The start and release lines are appended to the OWNER record (README convention), under the
OWNER.lock flock.

In:      ctx.stream (launch only), ctx.tables (scratch flag, allocated once).
Out:     none on the step path. run(ctx) launches lease_probe and returns with no host sync.
Scratch: one float32 device element, allocated on first call.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UNIT = "drv_gpu_lease"
KERNEL = "lease_probe"
# the pool-v3 / gpu-mutex tree of the node (NAMGPU_GPUWM_WORK, or the older NAMGPU_POOL_HOME; default ~/gpuwm-work);
# no user home is written into the package
GPUWM_WORK = Path(os.environ.get("NAMGPU_GPUWM_WORK") or os.environ.get("NAMGPU_POOL_HOME") or Path.home() / "gpuwm-work")
_POOL_HOME = GPUWM_WORK
POOL_RUN = GPUWM_WORK / "pool-v3" / "run.sh"
OWNER = GPUWM_WORK / "gpu-mutex" / "OWNER"
OWNER_LOCK = GPUWM_WORK / "gpu-mutex" / "OWNER.lock"
DEFAULT_LANE = "nam-gpu-drv-gpu-lease"
DEFAULT_VRAM_MIB = 12000
DEFAULT_EST_S = 900

# Holder command run under the lease. Prints one LEASED line, then blocks until stdin closes.
_HOLD = (
    "import os, sys\n"
    "print('LEASED', os.environ.get('CUDA_VISIBLE_DEVICES', ''), flush=True)\n"
    "sys.stdin.read()\n"
)


def pool_argv(lane: str, command, vram_mib: int, est_s: int, cards: int = 1) -> list[str]:
    """argv for a shell-level leased run: run.sh --cards N --vram-mib M --est S LANE -- COMMAND."""
    return [str(POOL_RUN), "--cards", str(cards), "--vram-mib", str(int(vram_mib)),
            "--est", str(int(est_s)), lane, "--", *command]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _owner_append(line: str) -> None:
    """Append one line to the OWNER record under its flock."""
    with open(OWNER_LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(OWNER, "a") as f:
                f.write(line + "\n")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


@contextmanager
def gpu_lease(lane: str = DEFAULT_LANE, vram_mib: int = DEFAULT_VRAM_MIB,
              est_s: int = DEFAULT_EST_S, cards: int = 1, what: str = "nam-gpu run"):
    """Hold a pool lease for the with-block. Yields the CUDA_VISIBLE_DEVICES string granted.

    Blocks until run.sh admits the lease (it queues behind tenants and other leases). Raises
    RuntimeError if the holder ends before it is granted.
    """
    from namgpu.paths import box_mode
    if not box_mode() or os.environ.get("NAMGPU_NO_LEASE") or not POOL_RUN.is_file():
        # Box plumbing off (the default), dedicated card, or no shared-card pool on this machine: the
        # lease is a no-op and the run uses CUDA_VISIBLE_DEVICES as set.
        yield os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return
    holder = [sys.executable, "-u", "-c", _HOLD]
    proc = subprocess.Popen(pool_argv(lane, holder, vram_mib, est_s, cards),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    card = None
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"pool lease {lane!r} ended before it was granted "
                                   f"(rc={proc.wait()})")
            if line.startswith("LEASED"):
                parts = line.split(None, 1)
                card = parts[1].strip() if len(parts) > 1 else ""
                break
        _owner_append(f"{lane} {_utc()} start pid {proc.pid} bounded {max(1, int(est_s) // 60)} min "
                      f"(pool, card has room: {int(vram_mib)} MiB: {what})")
        yield card
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()          # releases the lease
        rc = proc.wait()
        if card is not None:
            _owner_append(f"{lane} {_utc()} release rc {rc}")
        else:
            print(f"gpu_lease {lane!r}: holder exited rc={rc} before the grant", file=sys.stderr)


def run(ctx) -> None:
    """Launch the device probe on ctx.stream. Scratch flag allocated once in ctx.tables."""
    import cupy as cp
    from namgpu import kernels

    flag = ctx.tables.get("lease_flag")
    if flag is None:
        flag = cp.zeros(1, dtype=cp.float32)
        ctx.tables["lease_flag"] = flag
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern((1,), (32,), (flag, np.int32(1)))


def verify(ctx) -> bool:
    """Host check, off the step path: the probe wrote 1.0 on this card. Syncs; not for the loop."""
    flag = ctx.tables.get("lease_flag")
    if flag is None:
        return False
    return float(flag.get()[0]) == 1.0
