"""init-gfs-decode: GFS 0.25 deg boundary fields, decoded and sanitised on the device.

Unit: init-gfs-decode (group init, mode "new": no NAM Fortran counterpart, written from
CONTRACT.md section 5 and the GFS 0.25 deg pgrb2 record set init-fetch-case fetches).
Kernel:  namgpu/kernels/init_gfs_decode.cu (gfs_decode_sanitize, gfs_fill_missing, gfs_decode_pl).

Inputs (one boundary time at a time; the GRIB2 decode is init-grib2-decode, which writes each
record south to north, west to east, as device arrays):
  raw["pl"]    [10, nlev, nlat, nlon] float32   PL_VARS order: HGT TMP SPFH RH UGRD VGRD CLMR ICMR RWMR SNMR
  raw["plev"]  [nlev] float32, Pa               the isobaric levels of the file, from
                                                levels_from_inventory(); strictly decreasing
                                                (index 0 = lowest level, metgrid order)
  raw["sfc"]   [2, nlat, nlon] float32          SFC_VARS order: PRES (Pa), HGT (gpm) at the surface
  raw["valid"] optional key (hour or datetime) recorded on the result
GFS 0.25 deg global size: nlat = 721, nlon = 1440. The level count is not fixed here: it is the
number of TMP isobaric records in the file (41 in the 2025-03-14 12Z f006 file, 0.01 .. 1000 hPa).

Outputs (GfsDecoded, device arrays, same order as the input: lat south to north, lon 0..359.75):
  pl, pl_mask   [10, nlev, nlat, nlon] float32 / uint8
  sfc, sfc_mask [2, nlat, nlon] float32 / uint8
Mask bits (see the kernel header): 1 PRESENT, 2 BELOW (plev > surface pressure), 4 FILLED
(missing point replaced by the mean of its present neighbours), 8 PS_UNKNOWN (no surface
pressure for the point, so BELOW is not set there).
Every stored value is finite. A point with bit 1 and bit 4 both clear is missing and has no
present neighbour; its stored value is 0.0 and consumers must not use it.
SPFH, RH and the condensate species are clamped to >= 0. HGT is geopotential height in gpm, as
GFS writes it; init-gfs-vinterp reads GHT_IN in metres (the same unit), so no conversion is done here.

In place: the input device arrays are overwritten (pl and sfc are sanitised and filled where they
sit). Passing host arrays copies them to the device first, so the caller's host arrays are untouched.
No host sync inside the kernels. The only host work is the small upload of plev and the nonneg
flags (cached once per process).

Refusals (each names the breakage it prevents):
  - no fallback data: run() needs the decoded record set. A synthetic substitute would let a forecast
    run on fake boundary data without an error.
  - plev not one per level, or not strictly decreasing: the level set would be read in the wrong
    order and the vertical interpolation would run upside down.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np

from namgpu import kernels
from namgpu.kernels import BLOCK_2D, grid_2d

UNIT = "init-gfs-decode"
KSTEM = "init_gfs_decode"          # kernels/init_gfs_decode.cu
PL_VARS = ("HGT", "TMP", "SPFH", "RH", "UGRD", "VGRD", "CLMR", "ICMR", "RWMR", "SNMR")
NONNEG = tuple(1 if v in ("SPFH", "RH", "CLMR", "ICMR", "RWMR", "SNMR") else 0 for v in PL_VARS)
SFC_VARS = ("PRES", "HGT")
NLAT_FULL, NLON_FULL = 721, 1440
MASK_PRESENT, MASK_BELOW, MASK_FILLED, MASK_PS_UNKNOWN = 1, 2, 4, 8

_consts: dict = {}


@dataclass
class GfsDecoded:
    pl: object
    pl_mask: object
    sfc: object
    sfc_mask: object
    plev: object
    valid: object = None
    names: tuple = field(default=PL_VARS)


def levels_from_inventory(rows) -> np.ndarray:
    """Isobaric levels (Pa) of the TMP records in an init-grib2-decode inventory, high to low pressure."""
    lv = {float(r["level_value"]) for r in rows
          if str(r["level_type"]) == "100" and str(r["discipline"]) == "0"
          and str(r["category"]) == "0" and str(r["parameter"]) == "0"}
    if not lv:
        raise ValueError(f"{UNIT}: the inventory has no isobaric TMP records")
    return np.asarray(sorted(lv, reverse=True), np.float64)


def _const_device(key, host, dtype):
    """Upload a small constant vector once per process (H2D, outside the step path)."""
    import cupy as cp
    if key not in _consts:
        _consts[key] = cp.asarray(np.ascontiguousarray(host, dtype=dtype))
    return _consts[key]


def _stream_ctx(stream):
    return stream if stream is not None else contextlib.nullcontext()


def decode(pl, plev_pa, sfc, valid=None, stream=None) -> GfsDecoded:
    """Sanitise and fill one GFS boundary time in place on the device. Returns GfsDecoded."""
    import cupy as cp
    pl = cp.ascontiguousarray(cp.asarray(pl, dtype=cp.float32))
    sfc = cp.ascontiguousarray(cp.asarray(sfc, dtype=cp.float32))
    if pl.ndim != 4 or pl.shape[0] != len(PL_VARS):
        raise ValueError(f"{UNIT}: pl must be [{len(PL_VARS)}, nlev, nlat, nlon], got {pl.shape}")
    nlev, nlat, nlon = int(pl.shape[1]), int(pl.shape[2]), int(pl.shape[3])
    if sfc.shape != (len(SFC_VARS), nlat, nlon):
        raise ValueError(f"{UNIT}: sfc must be [{len(SFC_VARS)}, {nlat}, {nlon}], got {sfc.shape}")
    plev_host = np.ascontiguousarray(plev_pa, dtype=np.float64)
    if plev_host.shape != (nlev,):
        raise ValueError(f"{UNIT}: plev must have one entry per pl level ({nlev}), got {plev_host.shape}")
    if nlev > 1 and not bool(np.all(np.diff(plev_host) < 0.0)):
        raise ValueError(f"{UNIT}: plev must be strictly decreasing (index 0 = lowest level, metgrid "
                         "order); reorder the inventory selection")
    plev = cp.asarray(plev_host.astype(np.float32))
    nonneg = _const_device("nonneg", np.asarray(NONNEG, np.int32), np.int32)
    sfc_mask = cp.zeros(sfc.shape, dtype=cp.uint8)
    pl_mask = cp.zeros(pl.shape, dtype=cp.uint8)
    i32 = np.int32
    nsfc, nslab = len(SFC_VARS), len(PL_VARS) * nlev
    with _stream_ctx(stream):
        kernels.get(KSTEM, "gfs_decode_sanitize")(
            grid_2d(nlon, nlat, nsfc), BLOCK_2D,
            (sfc, sfc_mask, i32(nsfc), i32(nlat), i32(nlon)))
        kernels.get(KSTEM, "gfs_fill_missing")(
            grid_2d(nlon, nlat, nsfc), BLOCK_2D,
            (sfc, sfc_mask, i32(nsfc), i32(nlat), i32(nlon)))
        kernels.get(KSTEM, "gfs_decode_pl")(
            grid_2d(nlon, nlat, nlev), BLOCK_2D,
            (sfc[0], sfc_mask[0], plev, nonneg, pl, pl_mask,
             i32(len(PL_VARS)), i32(nlev), i32(nlat), i32(nlon)))
        kernels.get(KSTEM, "gfs_fill_missing")(
            grid_2d(nlon, nlat, nslab), BLOCK_2D,
            (pl, pl_mask, i32(nslab), i32(nlat), i32(nlon)))
    return GfsDecoded(pl=pl, pl_mask=pl_mask, sfc=sfc, sfc_mask=sfc_mask, plev=plev, valid=valid)


def run(ctx=None, raw=None):
    """Driver entry point. raw is the decoded record set (pl, plev, sfc, valid). No fallback data."""
    if raw is None:
        raise ValueError(f"{UNIT}: run needs the decoded GFS record set (pl, plev, sfc); there is no "
                         "synthetic fallback, so a forecast cannot run on fake boundary data")
    stream = getattr(ctx, "stream", None)
    out = decode(raw["pl"], raw["plev"], raw["sfc"], valid=raw.get("valid"), stream=stream)
    if ctx is not None:
        ctx.tables["gfs_decoded"] = out      # only the latest time is kept (VRAM, see note)
    return out
