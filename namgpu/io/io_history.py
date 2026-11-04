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
    if getattr(ctx, "slab", None) is not None:
        return _run_slab(ctx, fields, start, timing)
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


# ---- several cards (MULTICARD-PLAN.md section 6) ------------------------------------------------------------------
# Each rank queues the async copy of its OWNED rows into pinned buffers on ctx.stream (no host sync on the step path);
# its writer thread then writes them, at their global rows, into one full-domain float32 file per field under
# $NAMGPU_HIST_SHM/<valid>/ (/dev/shm) and drops a done.r<rank> flag. A pool of writer processes started by
# tools/run_forecast.py --cards N (tools/history_merge.py) writes each complete hour as one NetCDF, one hour per
# writer, so no single zlib thread sits on the critical path. Rank 0 adds meta.json and the global XLAT/XLONG.


def _run_slab(ctx, fields, start, timing):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    import cupy as cp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from .history import _pinned  # noqa: PLC0415
    s = ctx.slab
    shm = os.environ.get("NAMGPU_HIST_SHM")
    if not shm:
        raise RuntimeError("io-history: several cards need NAMGPU_HIST_SHM (tools/run_forecast.py --cards N sets it)")
    drain(ctx)
    timing.stamp(ctx, "io-history/d2h")
    valid = start + timedelta(seconds=ctx.ntimestep * ctx.cfg.dt)
    tag = f"{valid:%Y-%m-%d_%H:%M:%S}"
    jm = int(ctx.grid.jm)
    stream = ctx.stream
    staged, keep = [], []
    with stream:
        for name, (arr, dims) in fields.items():
            if arr.shape[-2] != jm:
                raise RuntimeError(f"io-history: {name} shape {arr.shape} has no local row axis of {jm}")
            if isinstance(arr, cp.ndarray):
                src = cp.ascontiguousarray(arr[..., s.own0:s.own0 + s.nown, :], dtype=cp.float32)
                dst = _pinned("slab:" + name, src.shape)
                src.get(stream=stream, out=dst, blocking=False)
                keep.append(src)
            else:
                dst = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)[..., s.own0:s.own0 + s.nown, :])
            staged.append((name, dst, tuple(dims)))
        ev = cp.cuda.Event(block=True, disable_timing=True)
        ev.record(stream)
    meta = None
    if s.rank == 0:
        geo = ctx.hist_geo_global
        meta = {"valid": tag, "start": f"{start:%Y-%m-%d_%H:%M:%S}",
                "xtime_min": float(ctx.ntimestep * ctx.cfg.dt) / 60.0,
                "dims": {"west_east": int(ctx.grid.im), "south_north": int(s.jmg), "bottom_top": int(ctx.grid.lm)},
                "cen_lat": geo["cen_lat"], "cen_lon": geo["cen_lon"],
                "pole_lat": float(ctx.cfg.tph0d), "pole_lon": float(ctx.cfg.tlm0d),
                "dx": geo["dx"], "dy": float(ctx.grid.dyh), "nranks": int(s.nranks),
                "fields": [[n, list(d), list(h.shape[:-2])] for n, h, d in staged]}
    timing.stamp(ctx, "io-history/submit")
    ctx.history_future = _WRITER.submit(_write_slab, Path(shm) / tag, staged, ev, keep, s, meta,
                                        getattr(ctx, "hist_geo_global", None))
    _ = json
    return Path(shm) / tag


def _write_slab(hour, staged, ev, keep, s, meta, geo):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    ev.synchronize()
    del keep
    hour.mkdir(parents=True, exist_ok=True)
    for name, h, _dims in staged:
        if h.ndim == 3:
            h = h[::-1]                      # NMMB l=1 top -> WRF bottom-up, as finish_payload
        full = tuple(h.shape[:-2]) + (int(s.jmg), int(h.shape[-1]))
        path = hour / f"{name}.f32"
        nbytes = int(np.prod(full)) * 4
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size < nbytes:
                os.ftruncate(fd, nbytes)
        finally:
            os.close(fd)
        mm = np.memmap(path, dtype=np.float32, mode="r+", shape=full)
        mm[..., s.jts - 1:s.jte, :] = h
        del mm
    if meta is not None:
        for nm in ("xlat", "xlon"):
            np.ascontiguousarray(geo[nm], dtype=np.float32).tofile(hour / f"{nm}.f32")
        tmp = hour / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, hour / "meta.json")
    (hour / f"done.r{s.rank}").write_text("1")
    return hour
