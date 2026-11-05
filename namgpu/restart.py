"""Restart sets: `woof-nam run --restart-every 1h` writes the whole forecast state every N hours, and
`woof-nam run --resume <restart>` continues the run from one to its original end, bit for bit.

What a set holds: everything the step reads, taken right after that hour's history write. That is every
device and host array reachable from the state (prognostic and physics fields, soil and snow, the precipitation
and max-field accumulators, the radiation state), the boundary arrays and the position in the boundary files,
the tables built at init, and the run's own counters (time step count, radiation step count, first-step and
filter flags, the clock), through the same walker as the saved initial state (namgpu/init/slab_init.py).
meta.json names the run (case, source, end hour, output folder, history settings) and the valid time.
The boundary files of the run are copied once into <restart root>/boco, so a set can be resumed on another
machine that has the case's inputs at a different place or not at all.

Layout:  <restart root>/wrfrst_d01_<valid>/ {a<k>.npy, skeleton.pkl, meta.json, COMPLETE}
         <restart root>/LATEST       name of the newest complete set
         <restart root>/boco/        the run's boundary files
A set is written to <name>.part and renamed when complete, so a run killed while writing leaves the previous
set intact. Only the newest --restart-keep sets are kept (default 1).

Cost on the step path: one stream-ordered device-to-pinned-host copy of the state (no host sync; the copy
runs in order with the step's kernels). The files are written by a background thread while the forecast steps
on; the next set waits for that write. Host memory: one pinned buffer the size of the state, reused.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FORMAT = 1
LATEST = "LATEST"
PREFIX = "wrfrst_d01_"
_ALIGN = 512


def parse_hours(text) -> float:
    """'1h', '90m', '3', '0'/'off' -> hours (0 = off)."""
    s = str(text).strip().lower()
    if s in ("", "0", "off", "none", "no"):
        return 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(h|m|min)?", s)
    if not m:
        raise ValueError(f"not a duration: {text!r} (examples: 1h, 90m, 3)")
    v = float(m.group(1))
    return v / 60.0 if m.group(2) in ("m", "min") else v


def steps_for(hours: float, dt: float) -> int:
    return int(round(hours * 3600.0 / float(dt))) if hours > 0 else 0


def _sets(root: Path) -> list:
    """Complete sets under root, oldest first (by time step count)."""
    out = []
    for d in Path(root).glob(PREFIX + "*"):
        if d.is_dir() and not d.name.endswith(".part") and (d / "COMPLETE").exists():
            try:
                out.append((int(json.loads((d / "meta.json").read_text())["ntimestep"]), d))
            except (OSError, ValueError, KeyError):
                continue
    return [d for _, d in sorted(out)]


def resolve(path) -> Path:
    """A restart set, or a restart root (its newest complete set)."""
    p = Path(path)
    if (p / "COMPLETE").exists() and (p / "meta.json").exists():
        return p
    sets = _sets(p)
    if not sets:
        raise SystemExit(f"--resume {p}: no complete restart set there (a set has COMPLETE and meta.json; "
                         f"woof-nam run --restart-every writes them)")
    return sets[-1]


def read_meta(rset) -> dict:
    meta = json.loads((Path(rset) / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise SystemExit(f"{rset}: restart format {meta.get('format')}, this woof-nam reads {FORMAT}")
    return meta


# Table entries the 32 GB fit path drops between uses and every user rebuilds before reading: the RRTM chunk
# context (rrtm_call.KEY_CHUNK: chunk copies and chunk-local scratch, rebuilt zero-filled by the next radiation call),
# driver-prep's per-call scratch (rrtm_prep:*, not the rrtm_prep:vec:* vectors), the horizontal-diffusion step
# tables (hdiff_*) and the 30-arcsec terrain read only by init-gwd-orog (gwd_dem, 1.78 GiB).
TRANSIENT_TABLES = ("rrtm_call:chunkctx", "gwd_dem")
TRANSIENT_TABLE_PREFIXES = ("rrtm_prep:", "hdiff_")
TRANSIENT_TABLE_KEEP_PREFIXES = ("rrtm_prep:vec:",)
TRANSIENT_ATTRS = ("history_fields",)


def transient(ctx) -> dict:
    """What a set leaves out: exactly what the 32 GB fit path (state.memory_path) has already released at the hour
    boundary where a set is taken (after that hour's history write), so a set holds the same arrays on either memory
    path. The full path (cards of 64 GiB and more) keeps these for the whole run; they are written before they are
    read in every step (the NAMGPU_SCRATCH_POISON=1 A/B, state.py), and a resumed run allocates them again on first
    use exactly as a fresh start does. Defect fixed (2026-11-05, WOOF owner canary run_4c391d9e4ea194ef, full path on
    a 96 GB card): a 12 km set was 36.7 GiB with these in, 15 GiB on the fit path."""
    from namgpu import state as S
    fields = set(S.STEP_SCRATCH) | set(S.HISTORY_SCRATCH) | set(S.INIT_ONLY)
    for names in S.UNIT_SCRATCH.values():
        fields |= set(names)
    drv = getattr(ctx, "driver", None)
    if drv is not None and hasattr(drv, "_phys_last_use"):
        # only the PHYS_SCRATCH names the fit path releases (their last user is a physics driver unit); the
        # others have readers elsewhere and stay in the state on both paths
        if getattr(drv, "_phys_lu", None) is None:
            drv._phys_lu = drv._phys_last_use()
        for names in drv._phys_lu.values():
            fields |= set(names)
    tables = {k for k in ctx.tables
              if k in TRANSIENT_TABLES or (k.startswith(TRANSIENT_TABLE_PREFIXES)
                                          and not k.startswith(TRANSIENT_TABLE_KEEP_PREFIXES))}
    return {"fields": sorted(fields), "tables": sorted(tables), "attrs": list(TRANSIENT_ATTRS)}


class Writer:
    """Writes a restart set every `every_hours` of forecast (after that step's history)."""

    def __init__(self, root, every_hours: float, dt: float, meta: dict, keep: int = 1):
        self.root = Path(root)
        self.every_hours = float(every_hours)
        self.every = steps_for(every_hours, dt)
        self.meta = dict(meta)
        self.keep = max(1, int(keep))
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="restart")
        self._fut = None
        self._arena = None
        self.log = []
        self._boco_done = False

    def due(self, ntimestep: int) -> bool:
        return self.every > 0 and ntimestep > 0 and ntimestep % self.every == 0

    def drain(self) -> None:
        """Wait for the set in flight (if any) and re-raise its error."""
        fut, self._fut = self._fut, None
        if fut is not None:
            fut.result()

    def _buffer(self, nbytes: int) -> np.ndarray:
        import cupy as cp
        if self._arena is None or self._arena.nbytes < nbytes:
            self._arena = None
            mem = cp.cuda.alloc_pinned_memory(max(nbytes, 1))
            self._arena = np.frombuffer(mem, np.uint8, nbytes)
        return self._arena

    def write(self, ctx) -> dict:
        """Stage the state now (stream-ordered copies into pinned memory); write it on the background thread."""
        import cupy as cp
        from namgpu.init import slab_init
        t0 = time.perf_counter()
        self.drain()                                       # previous set is on disk; the pinned buffer is free
        t_wait = time.perf_counter() - t0
        stream = ctx.stream if ctx.stream is not None else cp.cuda.get_current_stream()
        with stream:
            jobs, header, rec = slab_init.plan(ctx, leave_out=transient(ctx))
            sizes = [(-(-a.nbytes // _ALIGN)) * _ALIGN for _, a in jobs]
            buf = self._buffer(sum(sizes))
            staged, keep, off = [], [], 0
            for (name, a), sz in zip(jobs, sizes):
                dst = buf[off:off + a.nbytes].view(a.dtype).reshape(a.shape)
                if isinstance(a, cp.ndarray):
                    a.get(stream=stream, out=dst, blocking=False)
                    keep.append(a)
                else:
                    dst[...] = a
                staged.append((name, dst))
                off += sz
            ev = cp.cuda.Event(block=True, disable_timing=True)
            ev.record(stream)
        valid = ctx.start + timedelta(seconds=int(ctx.ntimestep) * float(ctx.cfg.dt))
        meta = dict(self.meta, format=FORMAT, ntimestep=int(ctx.ntimestep),
                    valid=f"{valid:%Y-%m-%d_%H:%M:%S}",
                    fhr=round(int(ctx.ntimestep) * float(ctx.cfg.dt) / 3600.0, 4),
                    boco_dir=str(getattr(ctx, "boco_dir", "") or ""))
        name = PREFIX + meta["valid"]
        entry = {"set": name, "fhr": meta["fhr"], "gib": round(off / 2**30, 2),
                 "stage_s": round(time.perf_counter() - t0, 3), "waited_for_previous_s": round(t_wait, 3)}
        self.log.append(entry)
        self._fut = self._exec.submit(self._finish, name, staged, keep, ev, header, rec, meta, entry)
        return entry

    def _finish(self, name, staged, keep, ev, header, rec, meta, entry):
        t0 = time.perf_counter()
        ev.synchronize()
        del keep
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._boco_done:
            self._copy_boco(meta.get("boco_dir"))
            self._boco_done = True
        part = self.root / (name + ".part")
        final = self.root / name
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda j: np.save(part / j[0], j[1], allow_pickle=False), staged))
        with open(part / "skeleton.pkl", "wb") as f:
            pickle.dump(header, f, protocol=pickle.HIGHEST_PROTOCOL)
        (part / "save-receipt.json").write_text(json.dumps(rec, indent=1))
        (part / "meta.json").write_text(json.dumps(meta, indent=1))
        (part / "COMPLETE").write_text(f"{datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}\n")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(part, final)
        tmp = self.root / (LATEST + ".tmp")
        tmp.write_text(name + "\n")
        os.replace(tmp, self.root / LATEST)
        for old in _sets(self.root)[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        entry["write_s"] = round(time.perf_counter() - t0, 2)
        entry["path"] = str(final)
        return final

    def _copy_boco(self, boco_dir) -> None:
        """The run's boundary files, once per restart root (hard links where the filesystem allows)."""
        if not boco_dir or not Path(boco_dir).is_dir():
            return
        dst = self.root / "boco"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(boco_dir).iterdir()):
            if not f.is_file() or (dst / f.name).exists():
                continue
            try:
                os.link(f, dst / f.name)
            except OSError:
                shutil.copy2(f, dst / f.name)

    def receipt(self) -> dict:
        latest = self.root / LATEST
        return {"every_hours": self.every_hours, "every_steps": self.every, "root": str(self.root),
                "keep": self.keep, "sets": self.log,
                "latest": str(self.root / latest.read_text().strip()) if latest.exists() else None}


def resume_boco(ctx, rset) -> None:
    """Point ctx.boco_dir at the restart root's copy when the original folder is gone (another machine)."""
    cur = getattr(ctx, "boco_dir", None)
    if cur and Path(cur).is_dir():
        return
    copy = Path(rset).parent / "boco"
    if copy.is_dir():
        ctx.boco_dir = str(copy)
