"""History output: one NetCDF file per output hour, shaped like a WRF wrfout so the Rust
renderer (rw_wrfbatch via gpuwm.rustwx) can draw it.

Route: gpuwm's Rust NetCDF writer (``~/.gpuwm/bridges/libnetcdf_writer.so``) once unit
io-history-rust binds it; until then netCDF4 (host write).

Split for the step path: ``prepare_history`` does the only device->host copy (one per field, once
per output hour, ordered after ctx.stream) and builds the attributes; ``write_payload`` does the
NetCDF encode and file write and touches no device memory, so io-history runs it on a background
worker while the stepping continues.  ``write_history`` is the synchronous form (checks, tools).

Fields written (names as WRF where WRF has them, values from NMMB state):
  XLAT, XLONG (H points, degrees), HGT (FIS/g), PSFC (PINT bottom), T2 (2 m temperature from the TSHLTR
  theta), Q2 (QSHLTR), MSLP (Pa, NMMB QUASIPOST membrane MSLP),
  U10, V10 (earth-relative, rotated from grid-relative by io-wind-rotate), RAINNC/RAINC (ACPREC
  minus CUPREC, CUPREC, mm), T (theta - 300), QVAPOR, QCLOUD, QRAIN, QICE, QSNOW (mixing ratios),
  PH (g*Z on the interfaces), XTIME (minutes), and the 3-D winds U, V (earth-relative at mass
  points, not WRF's staggered U/V) and W for soundings.
  PRES is the full mid-layer pressure (Pa). It is not WRF's perturbation P, and the file has no PB
  or PHB, so a reader must take PRES as the pressure itself.
  REFL_10CM and COMPOSITE (column maximum of REFL_10CM) are written when the Ferrier microphysics
  has produced REFL_10CM (phys-fer-calmict); otherwise they are absent.
WRF dims: Time, south_north, west_east, bottom_top (bottom-up: NMMB l is flipped on write).
Global attrs: MAP_PROJ=203 (rotated lat-lon, as WRF-NMM); CEN_LAT/CEN_LON = geographic lat/lon at
the domain centre; POLE_LAT/POLE_LON = rotated-pole coordinates (tph0d, tlm0d); DX/DY in metres
(nominal, DX taken at the centre row because the E-W spacing varies with latitude on the rotated
grid); WEST-EAST/SOUTH-NORTH/BOTTOM-TOP_GRID_DIMENSION = dimension + 1 (WRF staggered convention);
START_DATE, SIMULATION_START_DATE.  If rw_wrfbatch refuses MAP_PROJ 203, unit io-map-regrid writes
a companion file on a Lambert grid (NAM 218 frame) for the maps.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DIMS2 = ("south_north", "west_east")
# Only model-level fields are flipped (NMMB l=1 top -> WRF bottom-up). Soil layers (soil_layers_stag, the parent
# hand-off set) stay top-down, as WRF writes TSLB/SMOIS.
LEVEL_DIMS = ("bottom_top", "bottom_top_stag")


def history_title(res_km: float | None, parent: bool) -> str:
    """Global TITLE of a history: WOOF-NAM, the grid length (maps.resolution_km rule: DY / 1000.8 m, halves) and the
    role. Every history used to say "nam-gpu NMMB 12 km parent history", 3 km nests included (review round 3, #5)."""
    res = f"{res_km:g} km " if res_km else ""
    role = "parent " if parent else ("nest " if res_km and res_km < 12 else "")
    return f"WOOF-NAM {res}{role}history"


def _res_from_dy(dy_m: float) -> float:
    return round(float(dy_m) / 1000.8 * 2.0) / 2.0


def _handoff(ctx):
    """Parent grid attributes when the configfile turns history_handoff on, else None (file unchanged)."""
    from namgpu.io import io_history_fields as ihf  # noqa: PLC0415
    return ihf.handoff_attrs(ctx) if ihf.handoff_on(ctx.cfg) else None


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def prepare_history(ctx, fields: dict, start: datetime) -> dict:
    """Device->host copy of every field plus the attributes.  Returns a host-only payload."""
    g, cfg = ctx.grid, ctx.cfg
    jm, im = int(g.jm), int(g.im)
    valid = start + timedelta(seconds=ctx.ntimestep * cfg.dt)
    arrays = []
    from .io_history_fields import device_array  # noqa: PLC0415
    for name, (arr, dims) in fields.items():
        h = _host(device_array(arr))
        if h.ndim == 3 and dims[0] in LEVEL_DIMS:
            h = h[::-1]          # NMMB l=1 top -> WRF bottom-up
        arrays.append((name, np.ascontiguousarray(h, dtype=np.float32), tuple(dims)))
    centre = (jm // 2, im // 2)
    return {
        "valid": valid,
        "start": start,
        "xlat": np.degrees(_host(g.glat_h)).astype(np.float32),
        "xlon": np.degrees(_host(g.glon_h)).astype(np.float32),
        "xtime_min": float(ctx.ntimestep * cfg.dt) / 60.0,
        "dims": {"west_east": im, "south_north": jm, "bottom_top": int(g.lm)},
        "cen_lat": float(np.degrees(g.glat_h[centre])),
        "cen_lon": float(np.degrees(g.glon_h[centre])),
        "pole_lat": float(cfg.tph0d),
        "pole_lon": float(cfg.tlm0d),
        "dx": float(np.asarray(g.dxh)[jm // 2]),
        "dy": float(g.dyh),
        "fields": arrays,
        **({"handoff": ho} if (ho := _handoff(ctx)) is not None else {}),
    }


# ---- step-path form: async device -> pinned host, finished on the writer thread -------------------
# prepare_history() above does a synchronous pageable .get() per field on the step path. Measured on the
# box (dynbnd 1150Z, stamps io-history/d2h): 1.9, 2.8, 7.1 s per output hour with the GPU idle, and 27-65 s
# per hour in the 48 h run r1008 (the "gap after drv-tqadjust", 24-32 % of device time). stage_history()
# instead queues every copy on ctx.stream into pinned host buffers (reused hour to hour, so drain() of the
# previous write must run first, as io-history does) and records an event; finish_payload() waits on that
# event on the writer thread and builds the same payload prepare_history() builds.
_PINNED: dict = {}


def _pinned(key: str, shape: tuple) -> np.ndarray:
    import cupy as cp  # noqa: PLC0415
    a = _PINNED.get(key)
    if a is None or a.shape != tuple(shape):
        n = int(np.prod(shape))
        mem = cp.cuda.alloc_pinned_memory(max(n, 1) * 4)
        a = np.frombuffer(mem, np.float32, n).reshape(shape)
        _PINNED[key] = a
    return a


def stage_history(ctx, fields: dict, start: datetime) -> dict:
    """Queue the device->host copies on ctx.stream (no host sync). Returns the staged payload."""
    import cupy as cp  # noqa: PLC0415
    g, cfg = ctx.grid, ctx.cfg
    jm, im = int(g.jm), int(g.im)
    stream = ctx.stream if ctx.stream is not None else cp.cuda.get_current_stream()
    keep, staged = [], []
    from .io_history_fields import device_array  # noqa: PLC0415
    with stream:
        for name, (arr, dims) in fields.items():
            arr = device_array(arr)           # StreamField: computed into H_STREAM now, copied out before the next
            if isinstance(arr, cp.ndarray):
                src = arr if (arr.dtype == cp.float32 and arr.flags.c_contiguous) else                     cp.ascontiguousarray(arr, dtype=cp.float32)
                dst = _pinned(name, src.shape)
                src.get(stream=stream, out=dst, blocking=False)
                keep.append(src)
            else:
                dst = np.asarray(arr, dtype=np.float32)
            staged.append((name, dst, tuple(dims)))
        geo = ctx.tables.get("history_geo")
        if geo is None:
            xlat = np.degrees(_host(g.glat_h)).astype(np.float32)
            xlon = np.degrees(_host(g.glon_h)).astype(np.float32)
            cen = (jm // 2, im // 2)       # static: host values computed once per run (first history)
            geo = ctx.tables["history_geo"] = (xlat, xlon, float(np.degrees(g.glat_h[cen])),
                                               float(np.degrees(g.glon_h[cen])))
        ev = cp.cuda.Event(block=True, disable_timing=True)
        ev.record(stream)
    xlat, xlon, cen_lat, cen_lon = geo
    return {
        "event": ev, "keep": keep, "staged": staged,
        "valid": start + timedelta(seconds=ctx.ntimestep * cfg.dt),
        "start": start, "xlat": xlat, "xlon": xlon,
        "xtime_min": float(ctx.ntimestep * cfg.dt) / 60.0,
        "dims": {"west_east": im, "south_north": jm, "bottom_top": int(g.lm)},
        "cen_lat": cen_lat, "cen_lon": cen_lon,
        "pole_lat": float(cfg.tph0d), "pole_lon": float(cfg.tlm0d),
        "dx": float(np.asarray(g.dxh)[jm // 2]), "dy": float(g.dyh),
        **({"handoff": ho} if (ho := _handoff(ctx)) is not None else {}),
    }


def finish_payload(staged: dict) -> dict:
    """Writer thread: wait for the staged copies, then the host payload prepare_history() builds."""
    staged["event"].synchronize()
    staged["keep"] = None
    arrays = []
    for name, h, dims in staged.pop("staged"):
        if h.ndim == 3 and dims[0] in LEVEL_DIMS:
            h = h[::-1]          # NMMB l=1 top -> WRF bottom-up
        arrays.append((name, np.ascontiguousarray(h, dtype=np.float32), dims))
    out = {k: v for k, v in staged.items() if k not in ("event", "keep")}
    out["fields"] = arrays
    return out


def stage_and_write(path: Path, staged: dict) -> Path:
    return write_payload(path, finish_payload(staged))


def write_payload(path: Path, payload: dict) -> Path:
    """NetCDF encode and write.  Host data only; safe on a worker thread."""
    import netCDF4  # noqa: PLC0415

    valid, start = payload["valid"], payload["start"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = payload["dims"]
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("south_north", d["south_north"])
        nc.createDimension("west_east", d["west_east"])
        nc.createDimension("bottom_top", d["bottom_top"])
        nc.createDimension("bottom_top_stag", d["bottom_top"] + 1)
        soil = {f[1].shape[0] for f in payload["fields"] if f[2][0] == "soil_layers_stag"}
        if soil:                                   # parent hand-off set only (history_handoff)
            nc.createDimension("soil_layers_stag", soil.pop())
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = netCDF4.stringtochar(np.array([valid.strftime("%Y-%m-%d_%H:%M:%S")], "S19"))
        xt = nc.createVariable("XTIME", "f4", ("Time",))
        xt[0] = payload["xtime_min"]
        for name, arr in (("XLAT", payload["xlat"]), ("XLONG", payload["xlon"])):
            v = nc.createVariable(name, "f4", ("Time", "south_north", "west_east"), zlib=True, complevel=1,
                                  shuffle=True)
            v[0] = arr
        for name, arr, dims in payload["fields"]:
            # lossless: zlib level 1 with byte shuffle (2.6x on a full 12 km hour; levels 4-9 and zstd/blosc
            # measured no better than 3 % smaller, README "Output size")
            v = nc.createVariable(name, "f4", ("Time",) + dims, zlib=True, complevel=1, shuffle=True)
            v[0] = arr
        nc.MAP_PROJ = 203
        nc.CEN_LAT = payload["cen_lat"]
        nc.CEN_LON = payload["cen_lon"]
        nc.POLE_LAT = payload["pole_lat"]
        nc.POLE_LON = payload["pole_lon"]
        nc.DX = payload["dx"]
        nc.DY = payload["dy"]
        nc.WEST_EAST_GRID_DIMENSION = d["west_east"] + 1
        nc.SOUTH_NORTH_GRID_DIMENSION = d["south_north"] + 1
        nc.BOTTOM_TOP_GRID_DIMENSION = d["bottom_top"] + 1
        nc.START_DATE = start.strftime("%Y-%m-%d_%H:%M:%S")
        nc.SIMULATION_START_DATE = nc.START_DATE
        nc.TITLE = history_title(_res_from_dy(payload["dy"]), bool(payload.get("handoff")))
        for k, v in (payload.get("handoff") or {}).items():
            setattr(nc, k, v)                      # parent grid keys (io_history_fields.handoff_attrs)
    return path


def write_history(path: Path, ctx, fields: dict, start: datetime) -> Path:
    """Synchronous form: copy, then write.  fields: name -> (device array, wrf dims tuple)."""
    return write_payload(path, prepare_history(ctx, fields, start))
