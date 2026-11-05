"""Unit io-maps: render this runs history files to weather maps with the Rust renderer.

Driver entry (namgpu/driver.py DIAGNOSTICS, after io-history): ``run(ctx)`` is called once per
output hour or once at the end of a run.  It renders the history files it has not rendered yet
with rw_wrfbatch through namgpu/io/maps.py (render law: no matplotlib for weather fields), beside
the NAM 218 and MRMS sheets that io-referee-sheets draws.

Inputs (first one set wins):
  ``ctx.map_files``  list of history paths; set by io-map-regrid when rw_wrfbatch refuses
                     MAP_PROJ 203 and a Lambert companion file is written;
  otherwise every ``wrfout_d01_*`` file in ``ctx.out_dir``.
Only frames not yet rendered in this run (tracked on ``ctx.io_maps_rendered``) are drawn, so each
call costs one frame, not the whole series.  The renderer's series store ``maps/.store`` is kept
across calls so the 6-hour-style products (qpf_1h) can difference against earlier frames.

Done rule (raises instead of passing silently):
  - any renderer FAILED line raises;
  - a SKIPPED line raises unless it is a named expected skip (EXPECTED_SKIPS);
  - every requested product must have a PNG written by this call for every frame it was asked
    to draw, minus its expected skips;
  - the PNG count is taken from the renderer's own written list, never from a folder scan.
Output: ``<ctx.out_dir>/maps/`` (products per frame, renderer layout: case -> domain -> product).

No CUDA kernel: the unit does no device arithmetic (the field copies happen in io-history, and
rendering is host work outside the step loop), so there is no kernel file for this unit.
Reference: io/maps.py (DEFAULT_PRODUCTS, render).
"""
from __future__ import annotations

from pathlib import Path

from .io_history import drain as drain_history
from .maps import DEFAULT_PRODUCTS, render as render_series

# slug -> why a skip of that product is expected, and on which frame.  A qpf_1h
# skip is expected only on the run's first frame (nothing earlier to difference against).
EXPECTED_SKIPS = {
    "qpf_1h": "run's first frame has no earlier frame to difference against",
    "composite_reflectivity": "the 0 h frame precedes the first microphysics call (no REFL_10CM yet)",
    "qpf_6h": "frames of the run's first 6 h have no history frame 6 h earlier to difference against",
}
FIRST_FRAME_SKIPS = {"qpf_1h", "composite_reflectivity"}
WINDOW_SKIPS = {"qpf_6h"}      # expected while the frame is inside the first window (maps.WINDOW_H)


def _candidates(ctx) -> list[Path]:
    # io-history writes the file on a worker thread; the frame must be on disk before it is listed.
    drain_history(ctx)
    files = getattr(ctx, "map_files", None)
    if files:
        return sorted(Path(f) for f in files)
    out_dir = getattr(ctx, "out_dir", None)
    if out_dir is None:
        raise RuntimeError("io-maps: ctx.out_dir not set and ctx.map_files empty")
    return sorted(Path(out_dir).glob("wrfout_d01_*"))


def _rendered(ctx) -> set[str]:
    done = getattr(ctx, "io_maps_rendered", None)
    if done is None:
        done = set()
        ctx.io_maps_rendered = done
    return done


def _drawn(ctx, f: Path, products, first_name: str) -> bool:
    """Every requested product of frame f already has its PNG in <out>/maps (e.g. drawn by an hourly
    watcher during the run), so the post-run pass skips it."""
    from datetime import datetime
    start = getattr(ctx, "start", None)
    if start is None:
        return False
    try:
        valid = datetime.strptime(f.name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
    except ValueError:
        return False
    fhr = int(round((valid - start).total_seconds() / 3600))
    out = Path(ctx.out_dir) / "maps"
    for prod in products:
        if f.name == first_name and prod in FIRST_FRAME_SKIPS:
            continue
        if not (out / prod / f"namgpu_{start:%Y%m%d_%H}z_f{fhr:03d}_{prod}.png").is_file():
            return False
    return True


def history_files(ctx, products: str = DEFAULT_PRODUCTS) -> list[Path]:
    """Frames not yet rendered in this run (nor already on disk), oldest first."""
    done = _rendered(ctx)
    cands = _candidates(ctx)
    first = cands[0].name if cands else ""
    prods = [p.strip() for p in products.split(",") if p.strip()]
    todo = []
    for f in cands:
        if f.name in done:
            continue
        if _drawn(ctx, f, prods, first):
            done.add(f.name)
            continue
        todo.append(f)
    return todo


def _expected_skip(slug: str, reason: str, first_name: str) -> bool:
    if slug not in EXPECTED_SKIPS:
        return False
    if slug in FIRST_FRAME_SKIPS:
        return first_name in reason
    if slug in WINDOW_SKIPS:
        # no earlier frame yet, or a dry window (maps.py skips a window with no rain instead of painting zeros
        # domain-wide; a 3 km nest's first hour is often dry, and the nest door died here at maps, 2026-11-05)
        return "earlier to difference" in reason or "dry window" in reason
    return True


def run(ctx, products: str = DEFAULT_PRODUCTS, frames: str = "all",
        width: int = 1200, height: int = 900) -> Path:
    everything = _candidates(ctx)
    if not everything:
        raise RuntimeError("io-maps: no history files to render (io-history has not written any)")
    files = history_files(ctx, products)
    out = Path(ctx.out_dir) / "maps"
    if not files:
        return out
    files = everything[everything.index(files[0]):]
    mrms = Path(ctx.case_dir) / "raw" / "mrms" if getattr(ctx, "case_dir", None) else None
    extra = ["--mrms", str(mrms)] if mrms is not None and mrms.is_dir() else []
    # every frame so far goes to the renderer for qpf_1h differencing; only the new ones are drawn
    argv = [str(f) for f in everything] + extra + ["--frames", str(len(files))] + ["--out", str(out), "--products", products,
                                      "--width", str(width),
                                      "--height", str(height)]
    result = render_series(argv)
    if result["failures"]:
        raise RuntimeError("io-maps: renderer FAILED: " + "; ".join(result["failures"]))
    first_name = everything[0].name
    expected: list[tuple[str, str]] = []
    unexpected: list[str] = []
    for slug, reason in result["skipped"]:
        if _expected_skip(slug, reason, first_name):
            expected.append((slug, reason))
        else:
            unexpected.append(f"{slug}: {reason}")
    if unexpected:
        raise RuntimeError("io-maps: unexpected SKIPPED products: " + "; ".join(unexpected))
    written = [Path(p) for p in result["written"]]
    bad = [str(p) for p in written if p.suffix != ".png" or not p.is_file()]
    if bad or not written:
        raise RuntimeError(f"io-maps: renderer reported no usable PNG (bad={bad[:3]}, n={len(written)})")
    requested = [p.strip() for p in products.split(",") if p.strip()]
    missing = []
    for prod in requested:
        drawn = sum(1 for p in written if prod in p.as_posix())
        skipped_here = sum(1 for slug, _ in expected if slug == prod)
        if drawn + skipped_here < len(files):
            missing.append(f"{prod} rendered {drawn} of {len(files)} frames")
    if missing:
        raise RuntimeError("io-maps: products missing from this call: " + "; ".join(missing))
    _rendered(ctx).update(f.name for f in files)
    return out
