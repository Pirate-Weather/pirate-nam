"""Referee rows for WOOF-NAM maps on any domain: which truth and which source forecast sit beside a
nest panel, chosen by region and date from table rows (the arbitrary acceptance test: a new referee or
source is a row here, not a code path).

Row kinds (each row names its licence; nothing NC-licensed is read; a row whose licence requires attribution
also carries "credit", the line printed on every panel and sheet that shows it, naming the actual licensor):
  mrms          NOAA MRMS on the noaa-mrms-pds bucket, CONUS only (region row): composite reflectivity
                (nearest file to the valid time), and any product named by the row's mrms_dir key at the
                exact valid hour (MultiSensor QPE 1 h / 6 h Pass 2 for qpf_1h / qpf_6h).
  imerg         NASA GPM IMERG half-hourly precipitation, summed to 1 h / 6 h.
  grib-idx      a GRIB2 file on a public bucket with a wgrib2 .idx (GFS 0.25): byte-range subsets.
  grib-jsonl    a GRIB2 file with an ECMWF jsonl .index (IFS open data): byte-range subsets.

Roles:
  truth         the observed or analysed field at the valid time (MRMS, IMERG, the source's own
                later analysis: GFS f000 or IFS step 0 of the valid cycle).
  source        the forecast of the source the nest was started from, same init, same lead.

Every referee field is read on the host, then sampled onto the nest's own H-point grid (XLAT/XLONG of
the history), so the referee panel and the WOOF-NAM panel are drawn on the same mesh and the same frame
by the same renderer call. Sampling: bilinear for continuous fields (MSLP, 2 m T/Td, 10 m wind),
nearest for precipitation and reflectivity (no value smoothing), periodic in longitude for global grids.
A regular lat-lon grid that crosses the dateline is not handed to rw_wrfbatch directly: its inverse
raster path painted only the negative-longitude half of a 166E-186E frame (2026-11-04, shared-card test
y_dl3/y_dl4), while the curvilinear nest mesh with explicit bounds draws whole.

Host work: fetches in a thread pool (I/O), GRIB decode by the gpuwm grib2_dump bridge in parallel
subprocesses, IMERG reads in a process pool; nothing here waits on a single core.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

UNIT = "referee_rows"

# ---------------------------------------------------------------------------------------------- the table
# products: maps.py product names. "fields" per product: the records a row reads for it.
# GFS .idx selectors are ":NAME:LEVEL:" substrings of the idx line; IFS jsonl selectors are param/levtype.
GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
IFS_AWS = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
IFS_GOOGLE = "https://storage.googleapis.com/ecmwf-open-data"     # official mirror, last ~72 h only
IMERG_FINAL = "https://osdf-director.osg-htc.org/ncar/gdex/d731000/gpm_3imerghh_v07"
MRMS_BASE = "https://noaa-mrms-pds.s3.amazonaws.com"

GFS_FIELDS = {
    "mslp": [":PRMSL:mean sea level:"],
    "t2m": [":TMP:2 m above ground:"],
    "d2m": [":DPT:2 m above ground:"],
    "wind10": [":UGRD:10 m above ground:", ":VGRD:10 m above ground:"],
    "composite_reflectivity": [":REFC:entire atmosphere"],
    "qpf_1h": [":APCP:surface:"],
    "qpf_6h": [":APCP:surface:"],
}
IFS_FIELDS = {
    "mslp": [("msl", "sfc")],
    "t2m": [("2t", "sfc")],
    "d2m": [("2d", "sfc")],
    "wind10": [("10u", "sfc"), ("10v", "sfc")],
    "qpf_1h": [("tp", "sfc")],
    "qpf_6h": [("tp", "sfc")],
}

ROWS = [
    dict(name="mrms", kind="mrms", role="truth", label="MRMS", products=("composite_reflectivity",),
         region=(-130.0, -60.0, 20.0, 55.0), cadence_h=1, dates=("2020-10-14", None),
         licence="NOAA MRMS MergedReflectivityQCComposite_00.50, US government work (public domain), "
                 "AWS NODD noaa-mrms-pds"),
    # MRMS QPE ahead of IMERG over CONUS: radar + gauge, 1 km, hourly at the top of the hour (Pass 2 = the
    # gauge-corrected pass, ~1 h latency). Missing is -3 in the file; 0 is a measured dry hour.
    dict(name="mrms-qpe-01h", kind="mrms", role="truth", label="MRMS QPE 1 h (Pass 2)", products=("qpf_1h",),
         region=(-130.0, -60.0, 20.0, 55.0), cadence_h=1, dates=("2020-10-14", None),
         mrms_dir="MultiSensor_QPE_01H_Pass2_00.00", units="kg m-2",
         licence="NOAA MRMS MultiSensor_QPE_01H_Pass2_00.00, US government work (public domain), "
                 "AWS NODD noaa-mrms-pds"),
    dict(name="mrms-qpe-06h", kind="mrms", role="truth", label="MRMS QPE 6 h (Pass 2)", products=("qpf_6h",),
         region=(-130.0, -60.0, 20.0, 55.0), cadence_h=1, dates=("2020-10-14", None),
         mrms_dir="MultiSensor_QPE_06H_Pass2_00.00", units="kg m-2",
         licence="NOAA MRMS MultiSensor_QPE_06H_Pass2_00.00, US government work (public domain), "
                 "AWS NODD noaa-mrms-pds"),
    dict(name="imerg-final", kind="imerg", role="truth", label="NASA GPM IMERG Final V07B",
         products=("qpf_1h", "qpf_6h"), region=(-180.0, 180.0, -90.0, 90.0), cadence_h=1,
         dates=("2000-06-01", "2025-09-30"), base=IMERG_FINAL,
         licence="NASA/JAXA GPM IMERG Final Run V07B half-hourly (NASA open data, no restrictions), "
                 "mirrored by NCAR GDEX d731000, CC BY 4.0, doi:10.5065/SRTD-0R12",
         # the CC BY 4.0 licensor is the NCAR GDEX mirror (NASA data carry no restrictions), so the map credit
         # names GDEX and its DOI (fix round 2: it read "NASA GPM IMERG Final V07B open data, CC BY 4.0")
         credit="IMERG V07B via NCAR GDEX d731000 (doi:10.5065/SRTD-0R12), CC BY 4.0"),
    dict(name="imerg-late", kind="imerg", role="truth", label="NASA GPM IMERG Late V07B",
         products=("qpf_1h", "qpf_6h"), region=(-180.0, 180.0, -90.0, 90.0), cadence_h=1,
         dates=("2025-10-01", None), auth="urs.earthdata.nasa.gov",
         licence="NASA/JAXA GPM IMERG Late Run (NASA open data, no restrictions; GES DISC needs an Earthdata login)"),
    dict(name="gfs-analysis", kind="grib-idx", role="truth", label="GFS 0.25 analysis",
         products=("mslp", "t2m", "d2m", "wind10"), region=(-180.0, 180.0, -90.0, 90.0), cadence_h=6,
         for_sources=("gfs", "native", "hrrr", "rrfs", "parent"), dates=("2021-01-01", None),
         fields=GFS_FIELDS, lead="analysis",
         url="{base}/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.0p25.f{f:03d}", base=GFS_BASE,
         licence="NOAA NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds"),
    dict(name="ifs-analysis", kind="grib-jsonl", role="truth", label="ECMWF IFS 0.25 step 0",
         products=("mslp", "t2m", "d2m", "wind10"), region=(-180.0, 180.0, -90.0, 90.0), cadence_h=6,
         for_sources=("ifs", "ifs-0p25"), dates=("2024-02-28", None), fields=IFS_FIELDS, lead="analysis",
         url="{base}/{c:%Y%m%d}/{c:%H}z/ifs/0p25/{stream}/{c:%Y%m%d%H}0000-{f}h-{stream}-fc.grib2", base=IFS_AWS,
         mirrors=(IFS_GOOGLE,),
         licence="ECMWF open data, CC BY 4.0 (AWS mirror ecmwf-forecasts)", credit="ECMWF open data, CC BY 4.0"),
    dict(name="gfs-forecast", kind="grib-idx", role="source", label="GFS 0.25",
         products=("composite_reflectivity", "qpf_1h", "qpf_6h", "mslp", "t2m", "d2m", "wind10"),
         region=(-180.0, 180.0, -90.0, 90.0), cadence_h=1, for_sources=("gfs",), dates=("2021-01-01", None),
         fields=GFS_FIELDS, lead="forecast",
         url="{base}/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.0p25.f{f:03d}", base=GFS_BASE,
         licence="NOAA NCEP GFS 0.25, US government work (public domain), AWS NODD noaa-gfs-bdp-pds"),
    dict(name="ifs-forecast", kind="grib-jsonl", role="source", label="ECMWF IFS 0.25",
         products=("qpf_6h", "mslp", "t2m", "d2m", "wind10"), region=(-180.0, 180.0, -90.0, 90.0), cadence_h=3,
         for_sources=("ifs", "ifs-0p25"), dates=("2024-02-28", None), fields=IFS_FIELDS, lead="forecast",
         url="{base}/{c:%Y%m%d}/{c:%H}z/ifs/0p25/{stream}/{c:%Y%m%d%H}0000-{f}h-{stream}-fc.grib2", base=IFS_AWS,
         mirrors=(IFS_GOOGLE,),
         licence="ECMWF open data, CC BY 4.0 (AWS mirror ecmwf-forecasts)", credit="ECMWF open data, CC BY 4.0"),
]


# ---------------------------------------------------------------------------------------------- selection
def _overlaps(bounds, region) -> bool:
    """bounds (W, E, S, N) may run past 180 (dateline frame); region is -180..180."""
    w, e, s, n = bounds
    rw, re_, rs, rn = region
    if n < rs or s > rn:
        return False
    if re_ - rw >= 360.0:
        return True
    for shift in (-360.0, 0.0, 360.0):
        if w <= re_ + shift and e >= rw + shift:
            return True
    return False


def _in_dates(row, when: datetime) -> bool:
    a, b = row.get("dates", (None, None))
    if a and when < datetime.fromisoformat(a):
        return False
    if b and when >= datetime.fromisoformat(b) + timedelta(days=1):
        return False
    return True


def rows_for(bounds, source: str, start: datetime, hours: int) -> dict:
    """{product: {"truth": row|None, "source": row|None, "why": {role: reason}}} for a domain frame.
    A row is picked when its region overlaps the frame, its date range covers the run, and (for rows that
    name for_sources) the run's source is listed. The first matching row per role wins (table order)."""
    from namgpu.io.maps import TABLE  # noqa: PLC0415
    end = start + timedelta(hours=hours)
    out = {}
    for prod in TABLE:
        pick = {"truth": None, "source": None, "why": {}}
        for role in ("truth", "source"):
            reasons = []
            for row in ROWS:
                if row["role"] != role or prod not in row["products"]:
                    continue
                if not _overlaps(bounds, row["region"]):
                    reasons.append(f"{row['label']}: domain outside its region {row['region']}")
                    continue
                if "for_sources" in row and source not in row["for_sources"]:
                    reasons.append(f"{row['label']}: run source is {source}")
                    continue
                if not (_in_dates(row, start) and _in_dates(row, end)):
                    reasons.append(f"{row['label']}: not published for {start:%Y-%m-%d} (rows dates {row['dates']})")
                    continue
                if row.get("auth") and not _have_netrc(row["auth"]):
                    reasons.append(f"{row['label']}: needs a {row['auth']} login in ~/.netrc, none on this host")
                    continue
                pick[role] = row
                break
            if pick[role] is None:
                pick["why"][role] = "; ".join(reasons) or "no row carries this product"
        out[prod] = pick
    return out


def _have_netrc(machine: str) -> bool:
    p = Path.home() / ".netrc"
    return p.is_file() and machine in p.read_text(errors="ignore")


# ---------------------------------------------------------------------------------------------- fetch
def _get(url: str, rng: str | None = None, timeout=180, tries=5) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or k == tries - 1:
                raise
            if e.code in (429, 503):           # S3 SlowDown: back off harder (measured 2026-11-04 on ecmwf-forecasts)
                time.sleep(5.0 * (k + 1))
        except (OSError, urllib.error.URLError):
            if k == tries - 1:
                raise
        time.sleep(2.0 * (k + 1))
    raise RuntimeError("unreachable")


def _tmp(dest: Path) -> Path:
    """A writer-private temporary name beside dest: parallel jobs that need the same file (a 1 h and a 6 h
    window share half hours and GRIB leads) each write their own and rename, so no reader sees a torn file."""
    import threading  # noqa: PLC0415
    return dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part")


def _idx_subset(url: str, dest: Path, selectors: list[str]) -> list[str]:
    """Byte-range subset of a .idx GRIB2 file: the records whose idx line contains a selector, in file order.
    Returns the idx lines written (the record order of dest)."""
    lines = [ln for ln in _get(url + ".idx").decode().splitlines() if ln.strip()]
    offs = [int(ln.split(":")[1]) for ln in lines]
    want = [k for k, ln in enumerate(lines) if any(s in ":" + ":".join(ln.split(":")[3:]) + ":" for s in selectors)]
    if not want:
        raise ValueError(f"{url}: no idx record matches {selectors}")
    tmp = _tmp(dest)
    with open(tmp, "wb") as f:
        for k in want:
            end = offs[k + 1] - 1 if k + 1 < len(offs) else ""
            f.write(_get(url, f"{offs[k]}-{end}"))
    tmp.rename(dest)
    return [lines[k] for k in want]


def _jsonl_subset(url: str, dest: Path, selectors: list[tuple[str, str]]) -> list[str]:
    """Byte-range subset of an ECMWF open-data file by its jsonl .index (param, levtype)."""
    idx = url[: -len(".grib2")] + ".index"
    recs = [json.loads(ln) for ln in _get(idx).decode().splitlines() if ln.strip()]
    want = [r for r in recs if (r.get("param"), r.get("levtype")) in set(selectors)]
    if not want:
        raise ValueError(f"{idx}: no record matches {selectors}")
    want.sort(key=lambda r: int(r["_offset"]))
    tmp = _tmp(dest)
    with open(tmp, "wb") as f:
        for r in want:
            o, n = int(r["_offset"]), int(r["_length"])
            f.write(_get(url, f"{o}-{o + n - 1}"))
    tmp.rename(dest)
    return [f"{r['param']}:{r['levtype']}:{r.get('step')}" for r in want]


def _grib_fields(path: Path, labels: list[str], workdir: Path):
    """Decode every record of a subset file with the gpuwm bridges: [(label, lat1d, lon1d, values[j, i])],
    rows south to north, columns from lon1 eastward (scan bits applied, as io_referee_panels._place)."""
    from namgpu.init.init_grib2_decode import _dump, inventory  # noqa: PLC0415
    from namgpu.io.io_referee_panels import _place  # noqa: PLC0415
    inv = inventory(path)
    if len(inv) != len(labels):
        raise ValueError(f"{path.name}: {len(inv)} records decoded, {len(labels)} requested")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    dumped, dd = _dump(path, [r["index"] for r in inv], workdir)
    out = []
    for row, label in zip(inv, labels):
        if row["gdt"] != 0:
            raise ValueError(f"{path.name}: GDT {row['gdt']} is not a regular lat-lon grid")
        if row["scan"] & 0x80:
            # _place reverses such rows to west-to-east, but lon1 is then the eastern edge: refused rather
            # than drawn mirrored
            raise ValueError(f"{path.name}: scan mode 0x{row['scan']:02x} runs east to west: not handled")
        rec = dumped[row["index"]]
        host = np.fromfile(dd / rec["filename"], dtype="<f8")
        vals = _place(host, row)
        nx, ny = row["nx"], row["ny"]
        lat1, lon1 = float(row["lat1"]), float(row["lon1"]) % 360.0
        dy, dx = abs(float(row["dy"])), float(row["dx"])
        lat_n = lat1 if row["scan"] & 0x40 == 0 else lat1 + dy * (ny - 1)    # 0x40 unset: rows run north to south
        lat = lat_n - dy * np.arange(ny)[::-1]
        lon = lon1 + dx * np.arange(nx)
        out.append((label, lat, lon, vals.astype(np.float64)))
    return out


# ---------------------------------------------------------------------------------------------- IMERG
def _imerg_names(start_half: datetime):
    s = start_half
    e = s + timedelta(minutes=29, seconds=59)
    mmmm = s.hour * 60 + s.minute
    return f"{s:%Y/%m/%d}/3B-HHR.MS.MRG.3IMERG.{s:%Y%m%d}-S{s:%H%M%S}-E{e:%H%M%S}.{mmmm:04d}.V07B.HDF5"


def _imerg_read(path: str):
    """(lat[1800], lon[3600], rate[lat, lon] mm/h) of one IMERG half hour (HDF5 read through netCDF4)."""
    import netCDF4  # noqa: PLC0415
    with netCDF4.Dataset(path) as nc:
        g = nc.groups["Grid"]
        p = np.asarray(g.variables["precipitation"][0], dtype=np.float64)     # (lon, lat)
        lat = np.asarray(g.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(g.variables["lon"][:], dtype=np.float64)
    p = np.where(p < 0, np.nan, p).T                                         # (lat, lon)
    return lat, lon, p


# ---------------------------------------------------------------------------------------------- sampling
def sample_regular(lat_ax, lon_ax, vals, tlat, tlon, method="bilinear", periodic=None):
    """Sample vals[lat, lon] on regular axes at target points (tlat, tlon arrays). lat_ax ascending.
    lon_ax ascending with constant step; periodic (default: nx steps span 360 to within half a cell) wraps the
    longitude. The steps come from the whole span, not the first pair: IMERG stores its axes as float32
    (-179.94999695, -179.85000610), so lon[1] - lon[0] = 0.0999908, 3600 steps came to 359.967, the periodic test
    failed and every nest point between 179.95E and 179.95W was NaN (the blank strip at 180 on every Gabrielle
    IMERG panel, fix round 2), and the cell pick drifted by up to 0.33 cell across the axis."""
    nx = lon_ax.size
    dlat = (lat_ax[-1] - lat_ax[0]) / (lat_ax.size - 1)
    dlon = (lon_ax[-1] - lon_ax[0]) / (nx - 1)
    if periodic is None:
        periodic = abs(dlon * nx - 360.0) < 0.5 * abs(dlon)
    fx = ((tlon - lon_ax[0]) % 360.0) / dlon
    fy = (tlat - lat_ax[0]) / dlat
    ok = (fy >= 0) & (fy <= lat_ax.size - 1)
    if not periodic:
        ok &= fx <= nx - 1
    fy = np.clip(fy, 0, lat_ax.size - 1)
    if method == "nearest":
        ix = np.rint(fx).astype(np.int64) % nx if periodic else np.clip(np.rint(fx).astype(np.int64), 0, nx - 1)
        iy = np.rint(fy).astype(np.int64)
        out = vals[iy, ix]
    else:
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.minimum(np.floor(fy).astype(np.int64), lat_ax.size - 2)
        wx, wy = fx - x0, fy - y0
        if periodic:
            x0 %= nx
            x1 = (x0 + 1) % nx
        else:
            x0 = np.clip(x0, 0, nx - 1)
            x1 = np.clip(x0 + 1, 0, nx - 1)
        y1 = y0 + 1
        out = ((1 - wx) * (1 - wy) * vals[y0, x0] + wx * (1 - wy) * vals[y0, x1]
               + (1 - wx) * wy * vals[y1, x0] + wx * wy * vals[y1, x1])
    return np.where(ok, out, np.nan)


def _ascending(lat, vals):
    if lat[0] > lat[-1]:
        return lat[::-1], vals[::-1]
    return lat, vals


# ---------------------------------------------------------------------------------------------- products
_ACC = re.compile(r":(\d+)-(\d+) (hour|day) acc")


def _apcp_total(recs, f: int):
    """The run-total APCP record (0 .. f hours) of one decoded GFS file, or None. wgrib2 labels it
    "0-N hour acc fcst", or "0-D day acc fcst" when N is a whole number of days (f024: "0-1 day")."""
    for rec in recs:
        m = _ACC.search(rec[0])
        if m:
            k = 24 if m.group(3) == "day" else 1
            if int(m.group(1)) == 0 and int(m.group(2)) * k == f:
                return rec
    return None


def _k2f(k):
    return (k - 273.15) * 1.8 + 32.0


class Fetcher:
    """Fetches and samples one row's fields for a set of valid times onto one nest grid. Files are cached
    under `cache` (one folder per row), so the sheets of later leads and later runs reuse them."""

    def __init__(self, cache: Path, tlat: np.ndarray, tlon: np.ndarray, log=print):
        self.cache = Path(cache)
        self.tlat, self.tlon = tlat, tlon
        self.log = log
        self.bytes = 0

    # -- GRIB rows
    def _grib_file(self, row, cycle: datetime, f: int, product: str):
        kind = row["kind"]
        # ECMWF open data: 06/18Z were the scda stream until 2026-05-12 06Z, oper since (sources.py ifs-0p25 eras;
        # scda URLs 404 for later cycles, which dropped every 06Z/18Z truth panel)
        stream = "oper" if cycle.hour in (0, 12) or cycle >= datetime(2026, 5, 12, 6) else "scda"
        sel = row["fields"][product]
        tag = re.sub(r"[^A-Za-z0-9]+", "_", "+".join(s if isinstance(s, str) else "_".join(s) for s in sel)).strip("_")
        dest = self.cache / row["name"] / f"{cycle:%Y%m%d%H}_f{f:03d}_{tag}.grib2"
        dest.parent.mkdir(parents=True, exist_ok=True)
        meta = dest.with_suffix(".json")
        if dest.is_file() and meta.is_file():
            return dest, json.loads(meta.read_text())
        err = None
        bases = (row["base"], *row.get("mirrors", ()))
        if row.get("mirrors") and datetime.utcnow() - cycle < timedelta(hours=60):
            bases = (*row["mirrors"], row["base"])   # inside the Google mirror's ~72 h window: it is not throttled
        for base in bases:
            url = row["url"].format(base=base, c=cycle, f=f, stream=stream)
            try:
                labels = _idx_subset(url, dest, sel) if kind == "grib-idx" else _jsonl_subset(url, dest, sel)
                break
            except (OSError, urllib.error.URLError) as e:
                err = e
        else:
            raise err
        t = _tmp(meta)
        t.write_text(json.dumps(labels))
        t.rename(meta)
        self.bytes += dest.stat().st_size
        return dest, labels

    def _grib(self, row, cycle, f, product):
        path, labels = self._grib_file(row, cycle, f, product)
        with tempfile.TemporaryDirectory(prefix="ref-grib-", dir=path.parent) as td:
            recs = _grib_fields(path, labels, Path(td))
        return recs

    def _sample_recs(self, recs, method):
        outs = []
        for _label, lat, lon, vals in recs:
            lat, vals = _ascending(lat, vals)
            outs.append(sample_regular(lat, lon, vals, self.tlat, self.tlon, method))
        return outs

    def grib_product(self, row, product: str, start: datetime, valid: datetime):
        """(values on the nest grid in maps.py units, CF units) or raises with the reason."""
        lead_h = int(round((valid - start).total_seconds() / 3600))
        if row["lead"] == "analysis":
            if valid.hour % row["cadence_h"]:
                raise LookupError(f"{row['label']}: no analysis at {valid:%HZ} (cycles every {row['cadence_h']} h)")
            cycle, f = valid, 0
        else:
            cycle, f = start, lead_h
        ifs = row["kind"] == "grib-jsonl"
        if product in ("qpf_1h", "qpf_6h"):
            n = 1 if product == "qpf_1h" else 6
            if f < n:
                raise LookupError(f"{row['label']}: lead {f} h is shorter than the {n} h window")
            if ifs:   # tp accumulates from step 0 (m)
                a = self._sample_recs(self._grib(row, cycle, f, product), "nearest")[0]
                b = self._sample_recs(self._grib(row, cycle, f - n, product), "nearest")[0] if f - n > 0 else 0.0
                return np.maximum((a - b) * 1000.0, 0.0), "kg m-2"
            # GFS 0.25 files carry two APCP records: the bucket (6*floor((N-1)/6) .. N, reset every 6 h) and the run
            # total (0 .. N). Any window is total(f) - total(f - n). Reading only the bucket refused 35 of 43 6 h
            # windows per run as "spans a bucket reset" (fix round 2); the bucket rule stays only for a file that
            # has no run-total record.
            recs = self._grib(row, cycle, f, product)
            tot = _apcp_total(recs, f)
            prev = None
            if tot is not None and f - n > 0:
                prev = _apcp_total(self._grib(row, cycle, f - n, product), f - n)
            if tot is not None and (f - n == 0 or prev is not None):
                a = self._sample_recs([tot], "nearest")[0]
                q = a if f - n == 0 else a - self._sample_recs([prev], "nearest")[0]
                return np.maximum(q, 0.0), "kg m-2"
            a = self._sample_recs(recs, "nearest")[0]
            b0 = 6 * ((f - 1) // 6)
            if f - n == b0:
                q = a
            elif f - n > b0:
                q = a - self._sample_recs(self._grib(row, cycle, f - n, product), "nearest")[0]
            else:
                raise LookupError(f"{row['label']}: a {n} h window ending at f{f:03d} spans a bucket reset and the "
                                  f"file has no 0-{f} h run-total record")
            return np.maximum(q, 0.0), "kg m-2"
        recs = self._grib(row, cycle, f, product)
        method = "nearest" if product == "composite_reflectivity" else "bilinear"
        vals = self._sample_recs(recs, method)
        if product == "mslp":
            return vals[0] / 100.0, "hPa"
        if product in ("t2m", "d2m"):
            return _k2f(vals[0]), "degF"
        if product == "wind10":
            return np.hypot(vals[0], vals[1]), "m s-1"
        if product == "composite_reflectivity":
            return np.maximum(vals[0], -30.0), "dBZ"
        raise LookupError(f"{row['label']}: no recipe for {product}")

    # -- IMERG
    def imerg_product(self, row, product: str, valid: datetime):
        n = 1 if product == "qpf_1h" else 6
        halves = [valid - timedelta(minutes=30 * k) for k in range(1, 2 * n + 1)]
        paths = []
        jobs = []
        for s in halves:
            rel = _imerg_names(s)
            dest = self.cache / row["name"] / rel.rsplit("/", 1)[-1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            paths.append(dest)
            if not (dest.is_file() and dest.stat().st_size > 0):
                jobs.append((f"{row['base']}/{rel}", dest))

        def dl(job):
            url, dest = job
            data = _get(url, timeout=300)
            tmp = _tmp(dest)
            tmp.write_bytes(data)
            tmp.rename(dest)
            return len(data)
        if jobs:
            with ThreadPoolExecutor(max_workers=min(12, len(jobs))) as ex:
                self.bytes += sum(ex.map(dl, jobs))
        total = None
        for p in paths:                    # each read is ~0.2 s; the caller runs leads in parallel processes
            lat, lon, rate = _imerg_read(str(p))
            mm = rate * 0.5                # mm/h over a half hour
            total = mm if total is None else total + mm
        vals = sample_regular(lat, lon, total, self.tlat, self.tlon, "nearest")
        return vals, "kg m-2"

    # -- MRMS
    def _mrms_exact(self, row, valid: datetime, d: Path) -> Path:
        """The row's MRMS product at the exact valid hour (QPE files are stamped HH:00:00)."""
        import gzip  # noqa: PLC0415
        name = f"MRMS_{row['mrms_dir']}_{valid:%Y%m%d-%H}0000.grib2"
        g = d / name
        if not g.is_file():
            try:
                data = _get(f"{MRMS_BASE}/CONUS/{row['mrms_dir']}/{valid:%Y%m%d}/{name}.gz")
            except urllib.error.HTTPError as e:
                raise LookupError(f"MRMS {row['mrms_dir']}: no file for {valid:%Y-%m-%d %H}Z (HTTP {e.code})")
            self.bytes += len(data)
            tmp = _tmp(g)
            tmp.write_bytes(gzip.decompress(data))
            tmp.rename(g)
        return g

    def mrms_product(self, row, valid: datetime):
        from namgpu.init.fetch_case import _mrms_hour  # noqa: PLC0415
        from namgpu.init.init_grib2_decode import inventory  # noqa: PLC0415
        from namgpu.io.io_referee_panels import MRMS_FILL, _grib_values, _place  # noqa: PLC0415
        d = self.cache / row["name"]
        d.mkdir(parents=True, exist_ok=True)
        if row.get("mrms_dir"):
            g = self._mrms_exact(row, valid, d)
        else:
            msg = _mrms_hour(valid, d)
            g = d / f"mrms_refc_{valid:%Y%m%d%H}.grib2"
            if not g.is_file():
                raise LookupError(f"MRMS: {msg}")
        inv = inventory(g)
        r = inv[0]
        with tempfile.TemporaryDirectory(prefix="ref-mrms-", dir=d) as td:
            field = _place(_grib_values(g, r, Path(td)), r).astype(np.float64)
        # reflectivity marks no echo with -99 (MRMS_FILL); QPE marks missing with -3 and a dry hour with 0
        field = np.where(field < (0.0 if row.get("mrms_dir") else MRMS_FILL), np.nan, field)
        dy, dx = float(r["dy"]), float(r["dx"])
        lat0, lon0 = float(r["lat1"]), float(r["lon1"]) % 360.0
        lat = (lat0 - dy * np.arange(r["ny"]))[::-1]
        lon = lon0 + dx * np.arange(r["nx"])
        vals = sample_regular(lat, lon, field, self.tlat, self.tlon, "nearest", periodic=False)
        if row.get("mrms_dir"):
            return vals, row.get("units", "kg m-2")
        return np.maximum(vals, -30.0), "dBZ"

    def product(self, row, product: str, start: datetime, valid: datetime):
        if row["kind"] in ("grib-idx", "grib-jsonl"):
            return self.grib_product(row, product, start, valid)
        if row["kind"] == "imerg":
            return self.imerg_product(row, product, valid)
        if row["kind"] == "mrms":
            return self.mrms_product(row, valid)
        raise LookupError(f"row kind {row['kind']}")


def licences(rows) -> list[str]:
    seen, out = set(), []
    for r in rows:
        if r and r["name"] not in seen:
            seen.add(r["name"])
            out.append(f"{r['label']}: {r['licence']}")
    return out
