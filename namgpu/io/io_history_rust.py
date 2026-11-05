"""Unit io-history-rust: write the history NetCDF through gpuwm's Rust classic-NetCDF writer.

Same file as unit io-history (io/history.py, shaped like a WRF wrfout, dims Time / south_north /
west_east / bottom_top), written by libnetcdf_writer.so instead of netCDF4-python. The C ABI is
the one gpuwm's ctypes seam speaks (gpuwm/io/nc_writer_bridge.py, NCWRITE_ABI = 1); this module
binds the same entry points itself so WOOF-NAM does not import the gpuwm package.

Library: env NAMGPU_NCWRITE_LIB if set, else $GPUWM_BRIDGES/libnetcdf_writer.so (the gpuwm bridges copy)
(built from tools/rustwx/crates/netcdf-writer with cargo build --release --locked --offline
-p netcdf-writer --target-dir build/ncwrite). A library reporting another ABI is refused.

Driver contract (same as io-history): run(ctx) reads ctx.history_fields (name -> (device array,
wrf dims tuple)), ctx.start_time, ctx.out_dir, ctx.ntimestep, ctx.cfg, ctx.grid, ctx.stream;
returns the written Path.

Device work: per 3-D field one io_history_flip_levels launch (namgpu/kernels/io_history_rust.cu)
puts the levels in WRF bottom-up order in a staging buffer; then one device->host copy per field,
once per output hour. Nothing per step.

Global attributes follow io/history.py. MAP_PROJ is written as int32 (netCDF4-python writes an
int64 for a Python int); the value is the same, the type differs.

Intentional change, compression: io/history.py writes zlib (complevel 1) on every field and on
XLAT/XLONG. This unit does not. gpuwm_ncwrite_schema_new accepts only the classic formats (CDF-1,
CDF-2, CDF-5), and classic NetCDF has no per-variable filters, so the Rust writer cannot express
zlib. Every field is written uncompressed (about 191 MB per 3-D field at 954 x 835 x 60 float32).
Values are exact either way. Restoring compression needs a NetCDF-4 (HDF5) path in the writer, not a
flag here.
"""
from __future__ import annotations

import ctypes
import os
from datetime import timedelta
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.bridges import bridges_dir

UNIT = "io_history_rust"
KERNEL = "io_history_flip_levels"
NCWRITE_ABI = 1
DEFAULT_LIB = bridges_dir() / "libnetcdf_writer.so"
FORMAT_CDF2 = 2

NC_CHAR, NC_INT, NC_FLOAT, NC_DOUBLE = 2, 4, 5, 6
_NUMPY_TO_NC = {("S", 1): NC_CHAR, ("i", 4): NC_INT, ("f", 4): NC_FLOAT, ("f", 8): NC_DOUBLE}
_LIB = None
_FLIP = None                       # the flip kernel handle, looked up once
_STAGE: dict[tuple, object] = {}   # device staging buffers, one per field shape, allocated once


class NcWriteError(RuntimeError):
    pass


def _lib():
    """Load and bind the Rust writer once; refuse a library with another ABI."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = Path(os.environ.get("NAMGPU_NCWRITE_LIB", DEFAULT_LIB))
    if not path.is_file():
        raise FileNotFoundError(f"io-history-rust: Rust NetCDF writer not found at {path}")
    lib = ctypes.CDLL(str(path))
    lib.gpuwm_ncwrite_abi_version.argtypes = []
    lib.gpuwm_ncwrite_abi_version.restype = ctypes.c_uint32
    observed = int(lib.gpuwm_ncwrite_abi_version())
    if observed != NCWRITE_ABI:
        raise NcWriteError(f"{path} speaks ABI {observed}, io-history-rust needs {NCWRITE_ABI}")
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    void, size = ctypes.c_void_p, ctypes.c_size_t
    sigs = (
        ("gpuwm_ncwrite_last_error", [u8p, size], size),
        ("gpuwm_ncwrite_schema_new", [ctypes.c_uint32], void),
        ("gpuwm_ncwrite_schema_free", [void], None),
        ("gpuwm_ncwrite_def_dim", [void, ctypes.c_char_p, size, ctypes.c_uint64, ctypes.c_int32], ctypes.c_int64),
        ("gpuwm_ncwrite_def_var", [void, ctypes.c_char_p, size, ctypes.c_uint32, size, u64p], ctypes.c_int64),
        ("gpuwm_ncwrite_put_att", [void, ctypes.c_int64, ctypes.c_char_p, size, ctypes.c_uint32, size, ctypes.c_char_p], ctypes.c_int32),
        ("gpuwm_ncwrite_create", [void, ctypes.c_char_p, size], void),
        ("gpuwm_ncwrite_write_record", [void, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p, size], ctypes.c_int32),
        ("gpuwm_ncwrite_finish", [void], ctypes.c_int32),
        ("gpuwm_ncwrite_abort", [void], None),
    )
    for name, args, res in sigs:
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    _LIB = lib
    return lib


def _err(lib) -> str:
    n = int(lib.gpuwm_ncwrite_last_error(None, 0))
    if n == 0:
        return "the Rust writer reported a failure with no message"
    buf = (ctypes.c_uint8 * n)()
    lib.gpuwm_ncwrite_last_error(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)), n)
    return bytes(buf).decode("utf-8", "replace")


def _check(lib, status: int) -> None:
    if status != 0:
        raise NcWriteError(_err(lib))


def _put_att(lib, handle, varid: int, name: str, value) -> None:
    raw = name.encode()
    if isinstance(value, str):
        payload = value.encode()
        nc_type, n = NC_CHAR, len(payload)
    else:
        arr = np.atleast_1d(np.asarray(value))
        nc_type = _NUMPY_TO_NC[(arr.dtype.kind, arr.dtype.itemsize)]
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("="), copy=False))
        payload, n = arr.tobytes(), int(arr.size)
    _check(lib, lib.gpuwm_ncwrite_put_att(handle, varid, raw, len(raw), nc_type, n, payload))


class _NullScope:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _flip_to_host(ctx, arr, nl: int, jm: int, im: int) -> np.ndarray:
    """Device NMMB [nl, jm, im] -> host WRF bottom-up float32 slab: one kernel, one copy."""
    global _FLIP
    import cupy as cp

    if _FLIP is None:
        _FLIP = kernels.get(UNIT, KERNEL)
    key = (nl, jm, im)
    stage = _STAGE.get(key)
    if stage is None:
        stage = cp.empty((nl, jm, im), dtype=cp.float32)
        _STAGE[key] = stage
    scope = ctx.stream if ctx.stream is not None else _NullScope()
    with scope:
        _FLIP(kernels.grid_2d(im, jm, nl), kernels.BLOCK_2D,
              (np.int32(im), np.int32(jm), np.int32(nl), arr, stage))
        return stage.get()


def _host(a) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def write_history_rust(path: Path, ctx, fields: dict, start) -> Path:
    """Write one history file. fields: name -> (device array, wrf dims tuple)."""
    lib = _lib()
    g, cfg = ctx.grid, ctx.cfg
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    valid = start + timedelta(seconds=ctx.ntimestep * cfg.dt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = lib.gpuwm_ncwrite_schema_new(FORMAT_CDF2)
    if not schema:
        raise NcWriteError(_err(lib))
    writer = None
    try:
        def dim(name: str, n: int, unlimited: int = 0) -> int:
            raw = name.encode()
            d = int(lib.gpuwm_ncwrite_def_dim(schema, raw, len(raw), int(n), unlimited))
            _check(lib, -1 if d < 0 else 0)
            return d

        def def_var(name: str, nc_type: int, dids) -> int:
            raw = name.encode()
            ids = (ctypes.c_uint64 * len(dids))(*dids)
            vid = int(lib.gpuwm_ncwrite_def_var(schema, raw, len(raw), nc_type, len(dids), ids))
            _check(lib, -1 if vid < 0 else 0)
            return vid

        d_time = dim("Time", 0, 1)
        d_str = dim("DateStrLen", 19)
        dimids = {
            "south_north": dim("south_north", jm),
            "west_east": dim("west_east", im),
            "bottom_top": dim("bottom_top", lm),
            "bottom_top_stag": dim("bottom_top_stag", lm + 1),
        }
        d_sn, d_we = dimids["south_north"], dimids["west_east"]

        vid_times = def_var("Times", NC_CHAR, (d_time, d_str))
        vid_xlat = def_var("XLAT", NC_FLOAT, (d_time, d_sn, d_we))
        vid_xlon = def_var("XLONG", NC_FLOAT, (d_time, d_sn, d_we))
        field_vars = []
        extent = {"bottom_top": lm, "bottom_top_stag": lm + 1, "south_north": jm, "west_east": im}
        for name, (arr, dims) in fields.items():
            if np.dtype(arr.dtype) != np.float32:
                raise NcWriteError(f"io-history-rust: field {name} is {arr.dtype}, the writer takes float32")
            if any(d not in extent for d in dims):
                raise NcWriteError(f"io-history-rust: field {name} declares unknown dims {dims}")
            want = tuple(extent[d] for d in dims)
            if tuple(arr.shape) != want:
                raise NcWriteError(f"io-history-rust: field {name} has shape {tuple(arr.shape)}, "
                                   f"WRF dims {dims} need {want}")
            dids = (d_time,) + tuple(dimids[d] for d in dims)
            field_vars.append((def_var(name, NC_FLOAT, dids), name, arr))

        _put_att(lib, schema, -1, "MAP_PROJ", np.int32(203))
        _put_att(lib, schema, -1, "CEN_LAT", np.float64(cfg.tph0d))
        _put_att(lib, schema, -1, "CEN_LON", np.float64(cfg.tlm0d))
        _put_att(lib, schema, -1, "DX", np.float64(cfg.dlmd))
        _put_att(lib, schema, -1, "DY", np.float64(cfg.dphd))
        stamp = start.strftime("%Y-%m-%d_%H:%M:%S")
        _put_att(lib, schema, -1, "START_DATE", stamp)
        _put_att(lib, schema, -1, "SIMULATION_START_DATE", stamp)
        from namgpu.io import io_history_fields as _ihf  # noqa: PLC0415
        from namgpu.io.history import history_title  # noqa: PLC0415
        # cfg.dphd = 0.009 deg x res (namgpu/nest/domain.py nest_grid)
        _put_att(lib, schema, -1, "TITLE", history_title(round(cfg.dphd / 0.009 * 2.0) / 2.0, _ihf.handoff_on(cfg)))

        raw_path = str(path).encode()
        writer = lib.gpuwm_ncwrite_create(schema, raw_path, len(raw_path))
        schema = None   # the Rust side owns the schema from create on
        if not writer:
            raise NcWriteError(_err(lib))

        def record(vid: int, nc_type: int, payload: bytes) -> None:
            _check(lib, lib.gpuwm_ncwrite_write_record(writer, 0, vid, nc_type, payload, len(payload)))

        stampb = valid.strftime("%Y-%m-%d_%H:%M:%S").encode()
        record(vid_times, NC_CHAR, stampb)
        record(vid_xlat, NC_FLOAT, np.ascontiguousarray(np.degrees(_host(g.glat_h)), dtype=np.float32).tobytes())
        record(vid_xlon, NC_FLOAT, np.ascontiguousarray(np.degrees(_host(g.glon_h)), dtype=np.float32).tobytes())
        from .io_history_fields import device_array  # noqa: PLC0415
        for vid, name, arr in field_vars:
            arr = device_array(arr)
            if arr.ndim == 3:
                slab = _flip_to_host(ctx, arr, arr.shape[0], arr.shape[1], arr.shape[2])
            else:
                slab = np.ascontiguousarray(_host(arr), dtype=np.float32)
            record(vid, NC_FLOAT, slab.tobytes())
        _check(lib, lib.gpuwm_ncwrite_finish(writer))
        writer = None
    finally:
        if writer:
            lib.gpuwm_ncwrite_abort(writer)
        if schema:
            lib.gpuwm_ncwrite_schema_free(schema)
    return path


def run(ctx) -> Path:
    fields = getattr(ctx, "history_fields", None)
    if fields is None:
        raise RuntimeError("io-history-rust: ctx.history_fields not set (unit io-history-fields must run first)")
    start = getattr(ctx, "start_time", None)
    if start is None:
        raise RuntimeError("io-history-rust: ctx.start_time not set (driver must give the run start date)")
    if getattr(ctx, "out_dir", None) is None:
        raise RuntimeError("io-history-rust: ctx.out_dir not set")
    if ctx.stream is not None:
        ctx.stream.synchronize()
    valid = start + timedelta(seconds=ctx.ntimestep * ctx.cfg.dt)
    path = Path(ctx.out_dir) / f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"
    return write_history_rust(path, ctx, fields, start)
