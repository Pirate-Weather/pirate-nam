"""io-map-regrid: bilinear regrid of NMMB H-point fields onto a Lambert conformal map frame
(NAM 218 by default) on the GPU, for weather maps when rw_wrfbatch refuses MAP_PROJ 203.

Unit: io-map-regrid (new; no Fortran counterpart).  Kernel: namgpu/kernels/io_map_regrid.cu
(io_map_regrid_bilinear, one launch per field, no host sync).

Source: NMMB rotated lat-lon B grid, H points, arrays [j, i] (2-D) or [l, j, i] (3-D, l=1 model
top).  V-point fields must be averaged to H or rotated first (io-wind-rotate); this unit regrids
what it is given.
Target: spherical Lambert conformal, true latitude 25 N, LOV 265 E (-95), dx = dy = 12190 m,
614 x 428 (LambertFrame defaults; NAM 218 as in the NAM clone NPS rd_grib2.F90 grid table).
Outside the NMMB domain the output is FILL = -9999 (finite, so the NaN/Inf check still holds).

Tables: target lat/lon, source index (i0, j0), bilinear fractions (wx, wy) and the validity mask
are built once per frame with vectorised numpy (about 1.5 M points; one-time setup, not on the
step path) and kept on the device in ctx.tables.

Run:   run(ctx)                       2-D H fields of ctx.st -> ctx.st["MAP_<name>"] [ny, nx]
       run(ctx, fields={...})         any [j,i] or [l,j,i] device arrays -> dict of outputs
       write_companion(path, ctx, outputs, start)   NetCDF on the Lambert frame, WRF-style names
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "io_map_regrid"
KERNEL = "io_map_regrid_bilinear"
FILL = np.float32(-9999.0)
DEG = np.pi / 180.0

# NMMB V-point fields (CONTRACT.md section 1: U, V and their 10 m / tendency / tracer-flux
# companions sit on V(i,j), the NE corner of H(i,j)). Regridding them as H points misplaces them
# by a half cell (about 6 km at 12 km), so they are never regridded directly.
V_POINT_FIELDS = re.compile(
    r"^(U|V|TCU|TCV|UP|VP|DUDT|DVDT|U10|V10|U10MAX|V10MAX)$")


@dataclass(frozen=True)
class LambertFrame:
    """Spherical Lambert conformal map frame.  Defaults are NAM 218 (12 km CONUS)."""
    name: str = "NAM218"
    nx: int = 614
    ny: int = 428
    dx: float = 12190.58         # metres (NPS gribcode grid 218: 12.19058 km)
    dy: float = 12190.58
    lat1: float = 25.0           # true latitudes (equal: single cone parallel)
    lat2: float = 25.0
    lov: float = -95.0           # longitude of orientation, degrees east (265 E)
    lat_ll: float = 12.190       # lower-left grid point
    lon_ll: float = -133.459     # 226.541 E
    radius: float = 6371200.0    # metres


NAM218 = LambertFrame()
# NCEP grid 218 exactly as NOAA's awphys GRIB2 encodes it (section 3: shape of earth 6, a sphere of radius
# 6,371,229 m; Dx = Dy = 12191 m; first point 12.19 N 226.541 E). `woof-nam products` writes on this frame so a
# NAM ingest finds every point where NOAA's file puts it.
NCEP218 = LambertFrame(name="NCEP218", dx=12191.0, dy=12191.0, radius=6371229.0)


# ---- Lambert conformal (spherical) -------------------------------------------------------
def _lambert_consts(f: LambertFrame):
    l1, l2 = f.lat1 * DEG, f.lat2 * DEG
    if abs(l1 - l2) < 1e-9:
        n = np.sin(l1)
    else:
        n = (np.log(np.cos(l1)) - np.log(np.cos(l2))) / (
            np.log(np.tan(np.pi / 4 + l2 / 2)) - np.log(np.tan(np.pi / 4 + l1 / 2)))
    F = np.cos(l1) * np.tan(np.pi / 4 + l1 / 2) ** n / n
    return n, F


def _rho(f, n, F, lat_rad):
    return f.radius * F / np.tan(np.pi / 4 + lat_rad / 2) ** n


def lambert_forward(f: LambertFrame, lat_deg, lon_deg):
    """Geographic (deg) -> projected (x, y) metres from the pole-centred origin."""
    n, F = _lambert_consts(f)
    rho0 = _rho(f, n, F, f.lat1 * DEG)
    rho = _rho(f, n, F, np.asarray(lat_deg, np.float64) * DEG)
    theta = n * (np.asarray(lon_deg, np.float64) - f.lov) * DEG
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def lambert_inverse(f: LambertFrame, x, y):
    """Projected (x, y) metres -> geographic (lat, lon) degrees."""
    n, F = _lambert_consts(f)
    rho0 = _rho(f, n, F, f.lat1 * DEG)
    dy = rho0 - np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    rho = np.hypot(x, dy)
    theta = np.arctan2(x, dy)
    lon = f.lov + np.degrees(theta / n)
    t = (f.radius * F / rho) ** (1.0 / n)          # tan(pi/4 + lat/2)
    lat = np.degrees(2.0 * np.arctan(t) - np.pi / 2)
    return lat, lon


# ---- geographic <-> NMMB rotated coordinates (inverse of namgpu.grid.rotated_to_geographic) ----
def geo_to_rotated(tph0, tlm0, lat_rad, lon_rad):
    """Geographic (rad) -> rotated (tlm, tph) rad for the rotation centred on (tlm0, tph0).
    Forward map (grid.rotated_to_geographic): sin(lat) = sin(tph0) cos(tph) cos(tlm) + cos(tph0) sin(tph)."""
    cphi = np.cos(lat_rad)
    lr = lon_rad - tlm0
    gx, gy, gz = cphi * np.cos(lr), cphi * np.sin(lr), np.sin(lat_rad)
    s0, c0 = np.sin(tph0), np.cos(tph0)
    rx = c0 * gx + s0 * gz
    ry = gy
    rz = -s0 * gx + c0 * gz
    return np.arctan2(ry, rx), np.arcsin(np.clip(rz, -1.0, 1.0))


def build_tables(g, f: LambertFrame = NAM218):
    """Host tables for the target frame: source i0, j0, wx, wy, valid, and target lat/lon (f32)."""
    x0, y0 = lambert_forward(f, f.lat_ll, f.lon_ll)
    xs = x0 + f.dx * np.arange(f.nx, dtype=np.float64)
    ys = y0 + f.dy * np.arange(f.ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)                            # [ny, nx]
    lat, lon = lambert_inverse(f, X, Y)
    tlm, tph = geo_to_rotated(g.tph0, g.tlm0, lat * DEG, lon * DEG)
    xi = ((tlm - g.wb + np.pi) % (2.0 * np.pi) - np.pi) / g.dlm    # fractional H column index
    yj = (tph - g.sb) / g.dph                                       # fractional H row index
    valid = (xi >= 0.0) & (xi <= g.im - 1) & (yj >= 0.0) & (yj <= g.jm - 1)
    i0 = np.clip(np.floor(xi), 0, g.im - 2).astype(np.int32)
    j0 = np.clip(np.floor(yj), 0, g.jm - 2).astype(np.int32)
    wx = np.where(valid, xi - i0, 0.0).astype(np.float32)
    wy = np.where(valid, yj - j0, 0.0).astype(np.float32)
    return {"i0": i0, "j0": j0, "wx": wx, "wy": wy,
            "valid": valid.astype(np.uint8),
            "lat": lat.astype(np.float32), "lon": lon.astype(np.float32)}


def _tables(ctx, f: LambertFrame):
    import cupy as cp  # noqa: PLC0415
    key = f"io-map-regrid:{f.name}"
    if key not in ctx.tables:
        host = build_tables(ctx.grid, f)
        ctx.tables[key] = {k: cp.asarray(v) for k, v in host.items()}
    return ctx.tables[key]


def regrid_field(ctx, src, f: LambertFrame = NAM218, out=None):
    """src: device float32 [jm, im] or [lm, jm, im] (NMMB H layout).  Returns [ny, nx] or [lm, ny, nx]."""
    import cupy as cp  # noqa: PLC0415
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    two_d = src.ndim == 2
    src3 = cp.ascontiguousarray(src.reshape(1, jm, im) if two_d else src, dtype=cp.float32)
    lm = int(src3.shape[0])
    t = _tables(ctx, f)
    if out is None:
        out = cp.empty((lm, f.ny, f.nx), dtype=cp.float32)
    kern = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(kernels.grid_2d(f.nx, f.ny, lm), kernels.BLOCK_2D, (
            src3, out, t["i0"], t["j0"], t["wx"], t["wy"], t["valid"],
            np.int32(f.nx), np.int32(f.ny), np.int32(im), np.int32(jm), np.int32(lm), FILL))
    return out[0] if two_d else out


def run(ctx, fields=None, f: LambertFrame = NAM218):
    """Regrid the given fields (default: every 2-D H-point field in ctx.st).  Each output is also
    kept in ctx.st as MAP_<name> (added on the first call, reused after).  Returns name -> array."""
    g = ctx.grid
    im, jm = int(g.im), int(g.jm)
    if fields is None:
        fields = {n: a for n, a in ctx.st.fields.items()
                  if a.ndim == 2 and a.shape == (jm, im) and not n.startswith("MAP_")
                  and not V_POINT_FIELDS.match(n)}
    else:
        bad = sorted(n for n in fields if V_POINT_FIELDS.match(n))
        if bad:
            raise ValueError(
                f"io-map-regrid refuses V-point fields {bad}: they sit on V(i,j), not H(i,j). "
                "Average them to H (io-wind-rotate / module_H_TO_V inverse) before regridding.")
    outs = {}
    for name, arr in fields.items():
        shape = (f.ny, f.nx) if arr.ndim == 2 else (int(arr.shape[0]), f.ny, f.nx)
        key = f"MAP_{name}"
        if key in ctx.st.fields and tuple(ctx.st.fields[key].shape) != tuple(shape):
            raise ValueError(
                f"stale {key} has shape {tuple(ctx.st.fields[key].shape)}, need {shape}; "
                "drop it from ctx.st before regridding")
        dst = ctx.st.add(key, shape)
        outs[name] = regrid_field(ctx, arr, f, out=dst.reshape((-1, f.ny, f.nx)))
    return outs


def write_companion(path, ctx, outputs, start: datetime, f: LambertFrame = NAM218,
                    valid_offset_s: float = 0.0):
    """Write regridded fields as a WRF-style NetCDF on the Lambert frame (MAP_PROJ=1) so
    rw_wrfbatch can draw it.  outputs: name -> device array from run() or regrid_field()."""
    import netCDF4  # noqa: PLC0415
    from namgpu.io.history import _host  # noqa: PLC0415

    t = _tables(ctx, f)
    valid = start + timedelta(seconds=valid_offset_s)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("south_north", f.ny)
        nc.createDimension("west_east", f.nx)
        tv = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tv[0, :] = netCDF4.stringtochar(np.array([valid.strftime("%Y-%m-%d_%H:%M:%S")], "S19"))
        for name, arr in (("XLAT", _host(t["lat"])), ("XLONG", _host(t["lon"]))):
            v = nc.createVariable(name, "f4", ("Time", "south_north", "west_east"), zlib=True)
            v[0] = arr
        for name, arr in outputs.items():
            h = _host(arr)
            if h.ndim == 3:
                if "bottom_top" not in nc.dimensions:
                    nc.createDimension("bottom_top", h.shape[0])
                dims = ("Time", "bottom_top", "south_north", "west_east")
                h = h[::-1]                                # NMMB l=1 top -> WRF bottom-up
            else:
                dims = ("Time", "south_north", "west_east")
            v = nc.createVariable(name, "f4", dims, zlib=True, complevel=1, fill_value=FILL)
            v[0] = h
        nc.MAP_PROJ = 1
        nc.TRUELAT1 = float(f.lat1)
        nc.TRUELAT2 = float(f.lat2)
        nc.STAND_LON = float(f.lov)
        # Domain centre (grid point at nx//2, ny//2) and pole, as WRF writes them; rw_wrfbatch
        # uses these for placement.
        cen_lat = float(t["lat"][f.ny // 2, f.nx // 2])
        cen_lon = float(t["lon"][f.ny // 2, f.nx // 2])
        nc.CEN_LAT = cen_lat
        nc.CEN_LON = cen_lon
        nc.MOAD_CEN_LAT = cen_lat
        nc.POLE_LAT = 90.0
        nc.POLE_LON = 0.0
        nc.DX = float(f.dx)
        nc.DY = float(f.dy)
        nc.START_DATE = start.strftime("%Y-%m-%d_%H:%M:%S")
        nc.SIMULATION_START_DATE = nc.START_DATE
        nc.TITLE = f"WOOF-NAM NMMB 12 km parent regridded to {f.name} Lambert (io-map-regrid)"
    return path
