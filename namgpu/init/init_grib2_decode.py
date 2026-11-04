"""init-grib2-decode: GRIB2 records to device float32 fields in NMMB layout (CONTRACT section 1).

Init stage only, never on the step path. Decode chain:
  1. ``grib2_inventory <file>`` (gpuwm Rust bridge): one TSV row per GRIB2 message with the grid
     template (gdt), nx, ny, scan_mode, resolution_flags, shape_of_earth and the selector columns
     (discipline, category, parameter, level, PDT).
  2. ``grib2_dump <file> <outdir> <index...>`` (gpuwm Rust bridge): decodes every packing on the
     host (simple, complex/spatial differencing, JPEG2000, PNG) and writes ``metadata.tsv`` plus
     one little-endian float64 file per record (nx*ny values, stored scan order).
  3. Each record is read once (host I/O), copied to the device, and ``init_grib2_gather``
     (kernels/init_grib2_decode.cu) places it in the NMMB [j, i] layout with the scan-mode flags
     applied and float32 conversion.

Public:
    inventory(path)                        -> list of dicts (parsed ints, scan = int of scan_mode hex)
    run(ctx, grib_path, selectors, missing_value=None, tmpdir=None, expect_shape=None)
                                           -> {name: cupy float32 [NY, NX]}
        selectors: {name: record index (int) | callable(inventory_row) -> bool}; each must match
        exactly one record. expect_shape=(nx, ny) refuses a record on any other grid.
        ctx may be None; if it has a ``stream`` attribute the uploads, the kernel and the counter
        reads all run on it.
        Binding into the state (ctx.st) is done by the init driver, which owns the names.

Bitmaps and missing values (measured on cases/2025-03-14_12z bgrd3d00.grib2, record 1205 Geopotential
Height: count 796590, finite 793016, 3574 points NaN in the dumped float64 file): the bridge writes
bitmap-masked points as NaN and reports them as ``count - finite`` in metadata.tsv. This unit counts
them on the host and on the device and requires the two to agree. Without a missing_value they are
refused; with one they are replaced by it.

Units: none. The values come out in the units the GRIB2 record carries (K, Pa, m, degrees, ...).
Conversions (for example geopotential height to geopotential, or RH to specific humidity) belong to
the consumer, which must name them where it reads the record. The one current consumer,
init-grid-geo, takes NLAT/ELON in degrees with no conversion.

Geometry: the inventory has no grid geometry for GDT 32769 (lat1, lon1, dx, dy and the pole fields
are zero), so this unit does not verify pole, origin or spacing. That check is init-grid-geo: every
H point of the configured geometry is compared with the record's own NLAT/ELON (tolerance 1e-5 rad).
What this unit does verify: the grid template, shape_of_earth (GDT 32769), the grid shape against
expect_shape, and the scan mode.

Refusals (each names the breakage it prevents):
  - gdt not in {0, 32769}: other grid templates would be placed with a regular-grid layout they do
    not have (a projected grid read as lat-lon puts every point in the wrong cell).
  - scan_mode bit 0x10 set: boustrophedon storage is read as if every row ran the same way, which
    scrambles every other row.
  - scan_mode in the inventory differs from scan_mode in the dump: the two bridge calls disagree on
    the orientation of the same record, so its placement is unknown.
  - GDT 32769 with shape_of_earth other than 6: the grid was measured with that earth shape only.
  - wind components (U/V, discipline 0 category 2 parameters 2 and 3) on GDT 32769 with the
    resolution flag 0x08 set: those winds are grid-relative and would be placed unrotated.
  - record grid differs from expect_shape, or the dump grid or record size differs from the inventory.
  - non-finite values with no missing_value: they would enter the state silently.
  - a host count of non-finite values, the dump's count - finite, and the device counter disagree.
"""
from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "init_grib2_decode"
KERNEL = "init_grib2_gather"
SUPPORTED_GDT = (0, 32769)
REFUSED_SCAN_BITS = 0x10
GDT_NAM_ROTATED = 32769
GDT32769_SHAPE_OF_EARTH = 6      # WMO Code Table 3.2 code 6: sphere of radius 6371229.0 m
GRID_RELATIVE_UV_BIT = 0x08      # WMO Code Table 3.3 bit 5 (value 8): u, v relative to the grid
WIND_COMPONENTS = {(0, 2, 2), (0, 2, 3)}   # (discipline, category, parameter): U and V wind
_INV_REQUIRED = ("index", "discipline", "category", "parameter", "level_type", "level_value", "gdt",
                 "nx", "ny", "scan_mode", "resolution_flags", "shape_of_earth", "bitmap",
                 "reference_time", "forecast_time", "pdt")
_DUMP_REQUIRED = ("index", "nx", "ny", "scan_mode", "count", "finite", "filename")


def _bridge(name: str) -> Path:
    from namgpu.bridges import bridges_dir  # noqa: PLC0415
    root = bridges_dir()
    exe = root / name
    if not exe.is_file():
        raise FileNotFoundError(f"{exe}: set GPUWM_BRIDGES to the gpuwm bridge directory")
    return exe


def _tsv(lines, required, label):
    rows = [ln for ln in lines if ln and not ln.startswith("#")]
    reader = csv.DictReader(rows, delimiter="\t")
    missing = sorted(set(required) - set(reader.fieldnames or []))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    out = list(reader)
    if not out:
        raise ValueError(f"{label}: no records")
    return out


def inventory(path) -> list[dict]:
    proc = subprocess.run([str(_bridge("grib2_inventory")), str(path)],
                          text=True, capture_output=True, check=False)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise ValueError(f"GRIB2 inventory failed for {path}: "
                         f"{detail or 'decoder exited without a diagnostic'}")
    rows = _tsv(proc.stdout.splitlines(), _INV_REQUIRED, f"GRIB2 inventory for {path}")
    for r in rows:
        r["index"] = int(r["index"])
        r["gdt"] = int(r["gdt"])
        r["nx"] = int(r["nx"])
        r["ny"] = int(r["ny"])
        r["scan"] = int(r["scan_mode"], 16)
        r["resflags"] = int(r["resolution_flags"], 16)
        r["shape"] = int(r["shape_of_earth"])
        r["dcp"] = (int(r["discipline"]), int(r["category"]), int(r["parameter"]))
    return rows


def _dump(path, indices, workdir):
    # grib2_dump refuses an existing OUTPUT_DIR, so it gets a fresh child of the work directory.
    outdir = Path(workdir) / "dump"
    cmd = [str(_bridge("grib2_dump")), str(path), str(outdir), *(str(int(i)) for i in indices)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise ValueError(f"GRIB2 dump failed for {path}: {detail or 'decoder exited without a diagnostic'}")
    text = (Path(outdir) / "metadata.tsv").read_text(encoding="utf-8").splitlines()
    by_index = {int(r["index"]): r for r in _tsv(text, _DUMP_REQUIRED, "GRIB2 dump metadata")}
    omitted = [int(i) for i in indices if int(i) not in by_index]
    if omitted:
        raise ValueError(f"GRIB2 dump omitted selected fields {omitted}")
    return by_index, outdir


def _select(rows, selector, name):
    if isinstance(selector, (int, np.integer)):
        hits = [r for r in rows if r["index"] == int(selector)]
    else:
        hits = [r for r in rows if selector(r)]
    if len(hits) != 1:
        raise ValueError(f"{name}: selector matched {len(hits)} GRIB2 records (need exactly one)")
    return hits[0]


def _check_row(name, row, expect_shape):
    """Refusals decided from the inventory alone (before any decode)."""
    if row["gdt"] not in SUPPORTED_GDT:
        raise ValueError(f"{name}: GDT {row['gdt']} is not handled by init-grib2-decode "
                         f"(supported {SUPPORTED_GDT}); its layout would be misplaced")
    if row["scan"] & REFUSED_SCAN_BITS:
        raise ValueError(f"{name}: scan_mode 0x{row['scan']:02x} sets the boustrophedon bit "
                         f"0x10; rows would be read in the wrong direction")
    if row["gdt"] == GDT_NAM_ROTATED:
        if row["shape"] != GDT32769_SHAPE_OF_EARTH:
            raise ValueError(f"{name}: shape_of_earth {row['shape']} on GDT 32769; this unit's "
                             f"rotated-grid handling was measured on code {GDT32769_SHAPE_OF_EARTH}")
        if row["dcp"] in WIND_COMPONENTS and row["resflags"] & GRID_RELATIVE_UV_BIT:
            raise ValueError(f"{name}: resolution flags 0x{row['resflags']:02x} mark this wind as "
                             f"grid-relative; it would be placed unrotated and be wrong")
    if expect_shape is not None and (row["nx"], row["ny"]) != (int(expect_shape[0]), int(expect_shape[1])):
        raise ValueError(f"{name}: record grid {row['nx']}x{row['ny']} differs from the configured "
                         f"{expect_shape[0]}x{expect_shape[1]}")


def _storage_scan(row):
    """Scan mode the gather applies to a record, with the NAM rotated-grid convention.

    GDT 3.32769 (NAM, Arakawa E grid) records are stored south to north in j, as NMMB writes them,
    although the inventory reports scan 0x00 (north to south) for them. Evidence: the NLAT record of
    the bgrd3d00 analysis (2025-03-14 12Z, cases/*/raw/nam) matches the NMMB H-point geometry
    (init-grid-geo) only with j reversed when the 0x00 flag is honoured, to within the packing
    precision of the record. So the j flag is forced to +j for this GDT; the i flag is honoured.
    GFS (GDT 0) keeps the flag as written.
    """
    scan = int(row["scan"])
    if int(row["gdt"]) == GDT_NAM_ROTATED:
        scan |= 0x40
    return scan


def _gather(raw_dev, out, counter, nx, ny, scan, missing_value):
    kern = kernels.get(UNIT, KERNEL)
    kern(kernels.grid_2d(nx, ny), kernels.BLOCK_2D,
         (raw_dev, out, counter, np.int32(nx), np.int32(ny), np.int32(scan),
          np.float32(missing_value)))


def run(ctx, grib_path=None, selectors=None, missing_value=None, tmpdir=None, expect_shape=None):
    import cupy as cp

    if grib_path is None:
        # Chain stage (init-case-driver): stage the case's init source into ctx.st / ctx.tables
        # (namgpu/init/sources.py: one table row per source, native NAM analysis or GFS/HRRR/RRFS).
        from namgpu.init import sources  # noqa: PLC0415
        return sources.stage(ctx)

    inv = inventory(grib_path)
    chosen = {}
    for name, sel in selectors.items():
        row = _select(inv, sel, name)
        _check_row(name, row, expect_shape)
        chosen[name] = row

    stream = getattr(ctx, "stream", None)
    scope = stream if stream is not None else contextlib.nullcontext()
    fill = np.float32(np.nan) if missing_value is None else np.float32(missing_value)
    indices = sorted({r["index"] for r in chosen.values()})
    out, checks = {}, []
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)   # scratch/ is not in a fresh clone (.gitignore)
    with tempfile.TemporaryDirectory(prefix="namgpu-grib2-", dir=tmpdir) as td:
        dumped, dump_dir = _dump(grib_path, indices, td)
        with scope:
            for name, row in chosen.items():
                d = dumped[row["index"]]
                nx, ny = row["nx"], row["ny"]
                if int(d["nx"]) != nx or int(d["ny"]) != ny:
                    raise ValueError(f"{name}: dump grid {d['nx']}x{d['ny']} differs from inventory {nx}x{ny}")
                if int(d["scan_mode"], 16) != row["scan"]:
                    raise ValueError(f"{name}: scan_mode 0x{row['scan']:02x} in the inventory but "
                                     f"{d['scan_mode']} in the dump; the orientation of this record is unknown")
                count, finite = int(d["count"]), int(d["finite"])
                if count != nx * ny:
                    raise ValueError(f"{name}: dump count {count} differs from {nx}x{ny}")
                host = np.fromfile(dump_dir / d["filename"], dtype="<f8")
                if host.size != nx * ny:
                    raise ValueError(f"{name}: record holds {host.size} values, expected {nx * ny}")
                host_nonfinite = int(np.count_nonzero(~np.isfinite(host)))
                if host_nonfinite != count - finite:
                    raise ValueError(f"{name}: {host_nonfinite} non-finite values in the dumped file but "
                                     f"metadata says {count - finite} (count - finite)")
                raw = cp.asarray(host)
                arr = cp.empty((ny, nx), dtype=cp.float32)
                counter = cp.zeros(1, dtype=cp.int32)
                _gather(raw, arr, counter, nx, ny, _storage_scan(row), fill)
                out[name] = arr
                checks.append((name, row["index"], counter, count - finite))
            # One read of the device counters for the whole run (init stage), not one per field.
            counted = [(n, i, int(c.get()[0]), exp) for n, i, c, exp in checks]

    wrong = [(n, i, got, exp) for n, i, got, exp in counted if got != exp]
    if wrong:
        raise ValueError("device non-finite count disagrees with the dump metadata: "
                         + ", ".join(f"{n} (record {i}) device {g} vs {e}" for n, i, g, e in wrong))
    if missing_value is None:
        pending = [(n, i, e) for n, i, _, e in counted if e]
        if pending:
            raise ValueError("non-finite GRIB2 values (bitmap-masked points) with no missing_value given: "
                             + ", ".join(f"{n} (record {i}) {e}" for n, i, e in pending))
    return out


def _orientation_selftest() -> None:
    """Known pattern through every storage order the gather accepts (scan bits 0x80, 0x40, 0x20).

    The truth is NMMB layout, value = 1000*j0 + i0 on a non-square grid, so a transposed or flipped
    placement cannot pass. The storage order is built on the host by a route independent of the
    kernel: the columns are flipped when 0x80 is set, the rows when 0x40 is clear, and the storage
    is raveled column-major when 0x20 is set. The kernel must put every value back at [j0, i0]. Each
    order runs once with all points finite and once with one NaN (which must be replaced and counted).
    """
    import cupy as cp

    nx, ny = 37, 23
    truth = 1000.0 * np.arange(ny)[:, None] + np.arange(nx)[None, :]
    for scan in (0x00, 0x40, 0x80, 0xC0, 0x20, 0x60, 0xA0, 0xE0):
        for nan_at in (None, (3, 5)):
            t = truth.copy()
            if nan_at is not None:
                t[nan_at] = np.nan
            storage = t
            if scan & 0x80:
                storage = storage[:, ::-1]
            if not scan & 0x40:
                storage = storage[::-1, :]
            raw = storage.ravel() if not scan & 0x20 else storage.T.ravel()
            raw_dev = cp.asarray(np.ascontiguousarray(raw, dtype=np.float64))
            out = cp.empty((ny, nx), dtype=cp.float32)
            counter = cp.zeros(1, dtype=cp.int32)
            _gather(raw_dev, out, counter, nx, ny, scan, np.float32(-999.0))
            cp.cuda.Device().synchronize()
            got = cp.asnumpy(out)
            expect = np.where(np.isnan(t), -999.0, t).astype(np.float32)
            n_bad = int(np.count_nonzero(got != expect))
            n_nan = 0 if nan_at is None else 1
            assert n_bad == 0, f"scan 0x{scan:02x} nan_at={nan_at}: {n_bad} points placed wrongly"
            assert int(cp.asnumpy(counter)[0]) == n_nan, f"scan 0x{scan:02x}: counter != {n_nan}"
    print(f"orientation ok: 8 scan orders x (all finite, one NaN), {nx}x{ny} known pattern")


def _selftest(fixture: str | None) -> None:
    import cupy as cp

    _orientation_selftest()

    nx, ny = 954, 835
    rng = np.random.default_rng(1)
    raw = cp.asarray(rng.uniform(200.0, 300.0, nx * ny))
    arr = cp.empty((ny, nx), dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.int32)
    _gather(raw, arr, counter, nx, ny, 0x40, np.float32(np.nan))
    cp.cuda.Device().synchronize()
    assert bool(cp.isfinite(arr).all()), "non-finite output on a full-size random record"
    assert int(counter[0]) == 0, f"counter {int(counter[0])} on finite input"
    print(f"gather ok: {arr.shape} float32, min {float(arr.min()):.3f} max {float(arr.max()):.3f}")

    if fixture:
        inv = inventory(fixture)
        first = inv[0]["index"]
        got = run(None, fixture, {"first": first})["first"]
        assert bool(cp.isfinite(got).all())
        print(f"bridge decode ok: record {first} -> {got.shape} gdt {inv[0]['gdt']} "
              f"scan 0x{inv[0]['scan']:02x}")


if __name__ == "__main__":
    import sys
    fixture = None
    if "--fixture" in sys.argv:
        fixture = sys.argv[sys.argv.index("--fixture") + 1]
    _selftest(fixture)
