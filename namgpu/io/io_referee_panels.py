"""io-referee-panels: writes the per-source panel PNGs that io_referee_sheets composes.

Layout written (the layout io_referee_sheets reads):
    <panels>/<product>/<source>/<YYYYmmddHH>.png
    source in {mrms, nam_gpu, nam_archive}

Sources:
  mrms         MRMS MergedReflectivityQCComposite_00.50 (truth), composite reflectivity only.
  nam_gpu      this port's history files, drawn by the Rust renderer through the wrfout path.
  nam_archive  archived NAM 12 km grid 218 forecast of the same cycle (awphysFF), drawn through
               the Rust renderer's CF path.

Every picture is drawn by the Rust renderer rw_wrfbatch (render law: no matplotlib). GRIB2 is
read on the host with the gpuwm bridges grib2_inventory and grib2_dump (no device work, no step
path). Record placement follows the WMO scan-mode bits: 0x80 set means points run west to east
reversed; 0x40 unset means rows run north to south (flipped to south to north). Bit 0x20
(column-major storage) is refused.

Usage:
    python -m namgpu.io.io_referee_panels mrms --raw <case>/raw/mrms --panels <dir> --hours 2025031412
    python -m namgpu.io.io_referee_panels archive --raw <case>/raw/nam --cycle 2025031412 --panels <dir> --hours ...
    python -m namgpu.io.io_referee_panels woof-nam --history <wrfout ...> --panels <dir> --hours ...

Refusals (each names the breakage it prevents):
  - a GRIB2 record whose reference time or forecast hour differs from the requested cycle/hour:
    the archived panel would show another forecast on the same row as nam-gpu.
  - a scan mode with bit 0x20 set: column-major storage read as row-major scrambles the field.
  - a renderer that writes no PNG for a panel: the sheet would otherwise be composed from nothing.
  - a nam-gpu history without the requested renderer slug stored: the renderer's SKIPPED reason is
    raised, not hidden.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import struct
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu.init.init_grib2_decode import _dump, inventory
from namgpu.io.io_map_regrid import NAM218, lambert_forward, lambert_inverse
from namgpu.io.maps import BOUNDS

UNIT = "io_referee_panels"
REFUSED_SCAN_BITS = 0x20
MRMS_FILL = -90.0  # MRMS marks no echo with -99; anything below -90 dBZ is missing

# Referee product -> renderer slug (nam-gpu wrfout path), CF variable, units, title.
# Slugs verified with rw_wrfbatch --list-products: composite_reflectivity, 2m_temperature and
# qpf_total (windowed: needs the whole-hour frames of a multi-hour history).
# recipe / display_units / unit_factors: the renderer's colour table per product, the same rows
# namgpu/io/maps.py uses (without a recipe the CF renderer put every panel on a 0-1 scale).
PRODUCT_SPECS = {
    "composite_reflectivity": dict(slug="composite_reflectivity", cf_var="REFC", units="dBZ",
                                   title="Composite reflectivity", recipe="composite_reflectivity",
                                   display_units="dBZ", unit_factors={"dBZ": 1.0}),
    "t2": dict(slug="2m_temperature", cf_var="T2", units="degF", title="2 m temperature",
               recipe="2m_temperature", display_units="degF", unit_factors={"degF": 1.0}),
    "precip_total": dict(slug="qpf_total", cf_var="APCP", units="kg m-2",
                         title="Precipitation, total since start", recipe="total_qpf",
                         display_units="in", unit_factors={"kg m-2": 0.0393700787}),
}
# Archived NAM selectors: GRIB2 discipline, category, parameter, level type (+ level value / pdt).
ARCHIVE_SELECT = {
    "composite_reflectivity": dict(disc=0, cat=16, par=196, lt=200),
    "t2": dict(disc=0, cat=0, par=0, lt=103, lv=2),
    "precip_total": dict(disc=0, cat=1, par=8, lt=1, pdt=8),
}
MRMS_PRODUCT = "composite_reflectivity"


def _ints(row: dict, key: str) -> int:
    return int(float(row[key]))


def _sections(grib: Path) -> dict[int, bytes]:
    """Raw sections of the first GRIB2 message in a file (section number -> bytes)."""
    data = grib.read_bytes()
    if data[:4] != b"GRIB":
        raise ValueError(f"{grib.name}: not a GRIB2 file")
    total = int.from_bytes(data[8:16], "big")
    if data[total - 4:total] != b"7777":
        raise ValueError(f"{grib.name}: first message does not end in 7777")
    out, p = {}, 16
    while p < total - 4:
        ln = int.from_bytes(data[p:p + 4], "big")
        out[data[p + 4]] = data[p:p + ln]
        p += ln
    return out


def _png_values(grib: Path, row: dict) -> np.ndarray:
    """Host decode of GRIB2 PNG packing (template 5.41, used by MRMS): Y = (R + 2^E * pixel) / 10^D.

    The gpuwm grib2_dump bridge is built without its PNG codec, so this one template is decoded here
    with PIL; every other packing still goes through the bridge. Bitmaps (section 6) are applied.
    """
    from PIL import Image  # noqa: PLC0415

    sec = _sections(grib)
    s5 = sec[5]
    if int.from_bytes(s5[9:11], "big") != 41:
        raise ValueError(f"{grib.name}: section 5 template {int.from_bytes(s5[9:11], 'big')} is not PNG (5.41)")
    npts = int.from_bytes(s5[5:9], "big")
    if npts != row["nx"] * row["ny"]:
        raise ValueError(f"{grib.name}: {npts} data points, inventory says {row['nx'] * row['ny']}")
    ref = struct.unpack(">f", s5[11:15])[0]
    bscale = int.from_bytes(s5[15:17], "big", signed=True)
    dscale = int.from_bytes(s5[17:19], "big", signed=True)
    mask = None
    s6 = sec.get(6)
    if s6 is not None and s6[5] == 0:
        bits = np.unpackbits(np.frombuffer(s6[6:], dtype=np.uint8))
        mask = bits[:npts].astype(bool)
    elif s6 is not None and s6[5] != 255:
        raise ValueError(f"{grib.name}: bitmap indicator {s6[5]} is not handled")
    pix = np.asarray(Image.open(io.BytesIO(sec[7][5:]))).astype(np.float64).reshape(-1)
    want = int(mask.sum()) if mask is not None else npts
    if pix.size != want:
        raise ValueError(f"{grib.name}: PNG holds {pix.size} pixels, expected {want}")
    vals = (ref + pix * 2.0 ** bscale) / 10.0 ** dscale
    if mask is None:
        return vals
    full = np.full(npts, np.nan)
    full[mask] = vals
    return full


def _grib_values(grib: Path, row: dict, workdir: Path) -> np.ndarray:
    """Flat float64 values of one record in file order (PNG on the host, others via the bridge)."""
    if _ints(row, "drt") == 41:
        return _png_values(grib, row)
    dumped, dd = _fresh_dump(grib, row["index"], workdir)
    rec = dumped[row["index"]]
    host = np.fromfile(dd / rec["filename"], dtype="<f8")
    if host.size != row["nx"] * row["ny"]:
        raise ValueError(f"record holds {host.size} values, expected {row['nx'] * row['ny']}")
    return host


def _place(host: np.ndarray, row: dict) -> np.ndarray:
    """Host float32 [j, i] array from a flat record: j south to north, i west to east."""
    nx, ny, scan = row["nx"], row["ny"], row["scan"]
    if scan & REFUSED_SCAN_BITS:
        raise ValueError(f"scan_mode 0x{scan:02x} sets 0x20 (column-major storage): refused")
    arr = np.asarray(host, dtype=np.float64).reshape(ny, nx)
    if scan & 0x80:
        arr = arr[:, ::-1]
    if not scan & 0x40:
        arr = arr[::-1, :]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _fresh_dump(grib: Path, index: int, workdir: Path):
    """_dump needs an existing work directory and makes its own fresh child 'dump'."""
    workdir.mkdir(parents=True, exist_ok=True)
    return _dump(grib, [index], workdir)


def _renderer():
    from gpuwm import rustwx  # noqa: PLC0415

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise SystemExit(rustwx.renderer_remedy())
    return rustwx, renderer


def _cf_table(path: Path, product: str, spec: dict) -> Path:
    """One-row CF product table, the shape gpuwm's own CF callers write."""
    row = dict(names=[spec["cf_var"]], standard_names=[], product=product, title=spec["title"],
               recipe=spec["recipe"], default_units=spec["units"], display_units=spec["display_units"],
               unit_factors=spec["unit_factors"], contour_level=None, observation_product=None)
    path.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
    return path


def _write_cf(path: Path, valid: datetime, var: str, values: np.ndarray, units: str,
              lat: np.ndarray, lon: np.ndarray, source: str | None = None) -> Path:
    import netCDF4  # noqa: PLC0415

    ny, nx = values.shape
    with netCDF4.Dataset(path, "w") as d:
        if source:
            d.source = source        # the panel title (else the renderer writes "CF diagnostic <file>")
        d.createDimension("time", 1)
        d.createDimension("y", ny)
        d.createDimension("x", nx)
        t = d.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {valid:%Y-%m-%d %H:%M:%S}"
        t.calendar = "gregorian"
        t[:] = [0.0]
        la = d.createVariable("latitude", "f8", ("y", "x"))
        la.units, la.standard_name = "degrees_north", "latitude"
        la[:, :] = lat
        lo = d.createVariable("longitude", "f8", ("y", "x"))
        lo.units, lo.standard_name = "degrees_east", "longitude"
        lo[:, :] = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0   # NAM 218 / MRMS carry 0-360
        v = d.createVariable(var, "f4", ("time", "y", "x"), fill_value=np.float32(np.nan))
        v.units = units
        v.coordinates = "latitude longitude"
        v[0, :, :] = values
    return path


def _cf_render(rustwx, renderer, nc: Path, spec: dict, product: str, valid: datetime,
               table: Path, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="referee-cf-", dir=out_png.parent))
    try:
        rc = rustwx.run_cf_renderer(renderer, nc, variable=spec["cf_var"], out_dir=scratch,
                                    valid=f"{valid:%Y-%m-%d %H:%M}", table=table,
                                    bounds=BOUNDS)   # the same CONUS frame as namgpu/io/maps.py
        pngs = sorted(scratch.rglob("*.png"))
        if rc or len(pngs) != 1:
            raise RuntimeError(f"CF renderer exit {rc}, {len(pngs)} PNG for {product} {valid:%Y%m%d%H}")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pngs[0]), str(out_png))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out_png


def _pick(rows, pred, what: str):
    hits = [r for r in rows if pred(r)]
    if len(hits) != 1:
        raise ValueError(f"{what}: {len(hits)} GRIB2 records match (need exactly one)")
    return hits[0]


# ---------------------------------------------------------------- MRMS (truth)
def mrms_panels(raw: Path, panels: Path, hours: list[datetime]) -> list[Path]:
    rustwx, renderer = _renderer()
    spec = PRODUCT_SPECS[MRMS_PRODUCT]
    made = []
    with tempfile.TemporaryDirectory(prefix="referee-mrms-") as td:
        table = _cf_table(Path(td) / "cf.json", MRMS_PRODUCT, spec)
        for valid in hours:
            grib = raw / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
            if not grib.is_file():
                raise FileNotFoundError(f"missing MRMS file for referee truth: {grib}")
            inv = inventory(grib)
            row = _pick(inv, lambda r: True, f"MRMS {grib.name}")
            if row["gdt"] != 0:
                raise ValueError(f"MRMS GDT {row['gdt']} is not a regular lat-lon grid")
            field = _place(_grib_values(grib, row, Path(td) / f"w{valid:%Y%m%d%H}"), row)
            vals = np.where(field < MRMS_FILL, np.nan, field).astype(np.float32)
            lat0, lon0 = float(row["lat1"]), float(row["lon1"])
            if not -90.0 <= lat0 <= 90.0:
                raise ValueError(f"{grib.name}: lat1 {lat0} is not a latitude")
            if lon0 > 180.0:
                lon0 -= 360.0
            dy, dx = float(row["dy"]), float(row["dx"])
            # Row 0 of the file is the northern row; placed south to north here.
            lat = (lat0 - dy * np.arange(row["ny"]))[::-1]
            lon = lon0 + dx * np.arange(row["nx"])
            LON, LAT = np.meshgrid(lon, lat)
            nc = _write_cf(Path(td) / f"mrms_{valid:%Y%m%d%H}.nc", valid, spec["cf_var"], vals,
                           spec["units"], LAT, LON, source="MRMS observed")
            out = panels / MRMS_PRODUCT / "mrms" / f"{valid:%Y%m%d%H}.png"
            made.append(_cf_render(rustwx, renderer, nc, spec, MRMS_PRODUCT, valid, table, out))
    return made


# ---------------------------------------------------------------- archived NAM 218
def _archive_file(raw: Path, f: int) -> Path:
    full, sub = raw / f"awphys{f:02d}.grib2", raw / f"awphys{f:02d}.sub.grib2"
    if full.is_file():
        return full
    if sub.is_file():
        return sub
    raise FileNotFoundError(f"no archived NAM awphys file for forecast hour {f} in {raw}")


def _nam218_latlon():
    f = NAM218
    x0, y0 = lambert_forward(f, f.lat_ll, f.lon_ll)
    X, Y = np.meshgrid(x0 + f.dx * np.arange(f.nx, dtype=np.float64),
                       y0 + f.dy * np.arange(f.ny, dtype=np.float64))
    return lambert_inverse(f, X, Y)


def archive_panels(raw: Path, cycle: datetime, panels: Path, hours: list[datetime],
                   products=tuple(PRODUCT_SPECS)) -> list[Path]:
    rustwx, renderer = _renderer()
    f_ = NAM218
    LAT, LON = _nam218_latlon()
    made = []
    with tempfile.TemporaryDirectory(prefix="referee-arch-") as td:
        for valid in hours:
            delta = (valid - cycle).total_seconds()
            if delta < 0 or delta % 3600:
                raise ValueError(f"valid {valid} is not a whole forecast hour of cycle {cycle}")
            f = int(delta // 3600)
            grib = _archive_file(raw, f)
            inv = inventory(grib)
            ref = inv[0]["reference_time"]
            if re.sub(r"\D", "", ref)[:10] != f"{cycle:%Y%m%d%H}":
                raise ValueError(f"{grib.name}: reference time {ref!r} is not cycle {cycle:%Y%m%d%H}")
            for product in products:
                spec = PRODUCT_SPECS[product]
                sel = ARCHIVE_SELECT[product]

                def match(r, sel=sel):
                    ok = (_ints(r, "discipline") == sel["disc"] and _ints(r, "category") == sel["cat"]
                          and _ints(r, "parameter") == sel["par"] and _ints(r, "level_type") == sel["lt"])
                    if "lv" in sel:
                        ok = ok and _ints(r, "level_value") == sel["lv"]
                    if "pdt" in sel:
                        ok = ok and _ints(r, "pdt") == sel["pdt"]
                    if sel.get("pdt") == 8:
                        # an accumulation's forecast_time is the start of its window: "total since start"
                        # is the 0-f record (the f-hour file also holds the (f-3)-f bucket)
                        ok = ok and _ints(r, "forecast_time") == 0
                    return ok
                row = _pick(inv, match, f"{product} in {grib.name}")
                if sel.get("pdt") != 8 and _ints(row, "forecast_time") != f:
                    raise ValueError(f"{grib.name}: {product} forecast_time {row['forecast_time']} != {f}")
                if row["gdt"] != 30 or (row["nx"], row["ny"]) != (f_.nx, f_.ny):
                    raise ValueError(f"{grib.name}: grid gdt {row['gdt']} {row['nx']}x{row['ny']} "
                                     f"is not NAM 218 ({f_.nx}x{f_.ny} Lambert)")
                if abs(float(row["lat1"]) - f_.lat_ll) > 1e-3:
                    raise ValueError(f"{grib.name}: lat1 {row['lat1']} differs from the NAM 218 frame")
                vals = _place(_grib_values(grib, row, Path(td) / f"w{valid:%Y%m%d%H}{product}"), row)
                if product == "t2":
                    vals = (vals - np.float32(273.15)) * np.float32(1.8) + np.float32(32.0)   # K -> degF
                nc = _write_cf(Path(td) / f"arch_{product}_{valid:%Y%m%d%H}.nc", valid, spec["cf_var"],
                               vals, spec["units"], LAT, LON,
                               source=f"Archived NAM 12 km (grid 218), init {cycle:%Y-%m-%d %HZ} +{f} h")
                table = _cf_table(Path(td) / f"cf_{product}.json", product, spec)
                out = panels / product / "nam_archive" / f"{valid:%Y%m%d%H}.png"
                made.append(_cf_render(rustwx, renderer, nc, spec, product, valid, table, out))
    return made


# ---------------------------------------------------------------- nam-gpu history
_NAME = re.compile(r"_(\d{8})_(\d{2})z_f(\d{3})_")


def nam_gpu_panels(history: list[Path], panels: Path, hours: list[datetime],
                   products=tuple(PRODUCT_SPECS)) -> list[Path]:
    rustwx, renderer = _renderer()
    made = []
    with tempfile.TemporaryDirectory(prefix="referee-namgpu-") as td:
        store, out = Path(td) / "store", Path(td) / "out"
        store.mkdir()
        out.mkdir()
        slugs = ",".join(PRODUCT_SPECS[p]["slug"] for p in products)
        _written, failures, skipped = rustwx.run_renderer_series(
            renderer, [str(Path(h).resolve()) for h in history], store_root=store, out_dir=out,
            products=slugs, frames="all", source_label="WOOF-NAM 12 km")
        if failures:
            raise RuntimeError(f"renderer FAILED: {failures}")
        by_key = {}
        for p in out.rglob("*.png"):
            m = _NAME.search(p.name)
            if not m:
                continue
            init = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H")
            valid = init + timedelta(hours=int(m.group(3)))
            for product in products:
                if p.name.endswith(f"_{PRODUCT_SPECS[product]['slug']}.png"):
                    by_key[(product, valid)] = p
        for product in products:
            for valid in hours:
                src = by_key.get((product, valid))
                if src is None:
                    why = [s for s in skipped if PRODUCT_SPECS[product]["slug"] in str(s)]
                    raise FileNotFoundError(f"no WOOF-NAM {product} frame for {valid:%Y%m%d%H}"
                                            + (f": {why}" if why else ""))
                dst = panels / product / "nam_gpu" / f"{valid:%Y%m%d%H}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                made.append(dst)
    return made


def _hours(values) -> list[datetime]:
    return [datetime.strptime(h, "%Y%m%d%H") for h in values]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mrms")
    m.add_argument("--raw", type=Path, required=True)
    m.add_argument("--panels", type=Path, required=True)
    m.add_argument("--hours", nargs="+", required=True)
    a = sub.add_parser("archive")
    a.add_argument("--raw", type=Path, required=True)
    a.add_argument("--cycle", required=True)
    a.add_argument("--panels", type=Path, required=True)
    a.add_argument("--hours", nargs="+", required=True)
    g = sub.add_parser("woof-nam", aliases=["nam-gpu"])
    g.add_argument("--history", type=Path, nargs="+", required=True)
    g.add_argument("--panels", type=Path, required=True)
    g.add_argument("--hours", nargs="+", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "mrms":
        out = mrms_panels(args.raw, args.panels, _hours(args.hours))
    elif args.cmd == "archive":
        out = archive_panels(args.raw, datetime.strptime(args.cycle, "%Y%m%d%H"), args.panels,
                             _hours(args.hours))
    else:
        out = nam_gpu_panels(args.history, args.panels, _hours(args.hours))
    for p in out:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
