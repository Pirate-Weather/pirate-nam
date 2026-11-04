"""Write one history NetCDF per output hour (unit io-history, host-io mode).

Driver entry (namgpu/driver.py DIAGNOSTICS): ``("io-history", "history")``, gated by
``minutes_history = 60`` in the configfile, so ``run`` is called once per output hour.

Input: ``ctx.history_fields`` -- name -> (device array, wrf dims), built by unit
io-history-fields (its wrapper sets it on the ctx).  Output: ``<ctx.out_dir>/wrfout_d01_<valid>``.
Start time: ``ctx.start_time`` (datetime, set by the driver from the run's start date).
Output directory: ``ctx.out_dir`` (set by Driver.run; defaults to nam-gpu/runs/<start>).

No CUDA kernel and no device arithmetic.  The unit does two things per output hour:
  1. the device->host copy of every history field, queued on ctx.stream into pinned host buffers
     (history.stage_history, no host sync on the step path; the writer thread waits on its event);
  2. the NetCDF encode and file write (history.write_payload), on one background worker thread,
     so the single-threaded zlib encode does not stall the stepping.  At most one write is in
     flight: the next hour waits for the previous file.  ``drain(ctx)`` waits for the last one and
     re-raises any write error; Driver.run calls it before the receipt.
Reference for the field list and attributes: io/history.py (module docstring).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from .history import stage_and_write, stage_history

_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io-history")


def drain(ctx) -> None:
    """Wait for the in-flight history write (if any) and re-raise its error."""
    fut = getattr(ctx, "history_future", None)
    if fut is not None:
        ctx.history_future = None
        fut.result()


def run(ctx) -> Path:
    fields = getattr(ctx, "history_fields", None)
    if fields is None:
        raise RuntimeError("io-history: ctx.history_fields not set (unit io-history-fields must run first)")
    start = getattr(ctx, "start_time", None)
    if start is None:
        raise RuntimeError("io-history: ctx.start_time not set (driver must give the run start date)")
    if getattr(ctx, "out_dir", None) is None:
        raise RuntimeError("io-history: ctx.out_dir not set")
    from ..steps import drv_timing_receipt as timing  # noqa: PLC0415
    drain(ctx)                           # previous hour's file must be complete first (pinned buffers reused)
    timing.stamp(ctx, "io-history/d2h")
    valid = start + timedelta(seconds=ctx.ntimestep * ctx.cfg.dt)
    path = Path(ctx.out_dir) / f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"
    # async copies on ctx.stream into pinned host buffers; the writer thread waits on their event, so the
    # step path no longer syncs or copies (history.stage_history; was 2-65 s per hour with the GPU idle)
    staged = stage_history(ctx, fields, start)
    timing.stamp(ctx, "io-history/submit")
    ctx.history_future = _WRITER.submit(stage_and_write, path, staged)
    return path
