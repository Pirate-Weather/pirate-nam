"""init-gfs-hinterp: horizontal interpolation of GFS fields onto the NMMB boundary strips.

Unit: init-gfs-hinterp (group init, mode "new": the rule is the METGRID table, not a Fortran routine).
Kernel:  namgpu/kernels/init_gfs_hinterp.cu (init_gfs_hinterp_strips), one launch per field and grid
         class (H or V), one thread per strip point, no host sync, no allocation after the first call
         (strip buffers are cached per (slot, NSLAB, IM, JM, width)).
Rule:    nam.METGRID.TBL.bocos. interp_option=four_pt+average_4pt for the fields the NAM boundary input
         uses (PSFC, RH, TT, UU, VV, GHT). four_pt is bilinear on the four nearest GFS points and is
         missing if any of the four is missing; then average_4pt is the mean of the present points; if none
         is present the point is missing and gets fill_missing (0.0), counted in the "missing" output.
         See the kernel header for the full rule. The reach of average_4pt on NAM 12 km from GFS 0.25 deg
         is not checked against a WPS run (no WPS source in the tree); the table lists it unconditionally,
         so it is implemented unconditionally.

Time and level binding:
    src, mask  [NT, NL, NJS, NIS] with NT = TIME_LEVELS = 2 (the two boundary times read_bc reads).
               NL = 1 for 2-D fields, the isobaric level count for 3-D GFS fields (init-grib2-decode, 41 at
               0.01 .. 1000 hPa in the 2025-03-14 12Z file). NL is the SOURCE level count. It is not LM.
    Out:       dict s, n [NT*NL, W, IM], w, e [NT*NL, JM, W], missing int32 [1], nt, nl.
               Slab index = t * NL + l. The arrays are shared scratch per slot: copy them before the next
               call with the same slot (run() does).
    The model levels (LM) exist only after init-gfs-vinterp (todo). require_read_bc_levels() refuses to
    hand isobaric strips to read_bc.

Source contract (checked, not assumed):
    grid.lat_order must be "south_to_north" and grid.dlat > 0: source row j is lat0 + j*dlat. init-grib2-decode
    and init-gfs-decode write rows south to north. A north-to-south array is refused, not re-sorted.

Wind order: earth-relative U and V are interpolated here (the components, not the grid-relative winds),
then rotated to grid-relative at V points by init-gfs-wind-rotate. Rotating first is not used.

Grid: default is GFS 0.25 degree (lat -90..90, 721 rows; lon 0..359.75, 1440 columns, periodic).

run(ctx) acts only when ctx.gfs is staged. It does not fabricate input. It stores copies of the H and V strips
for every staged field in ctx.tables["gfs_hinterp_strips"][name] = {"H": ..., "V": ...} (the consumer
wiring). init-gfs-vinterp reads full-grid columns from ctx.tables["gfs_hinterp"], built by
full_domain_inputs() from the decoded set (the full-grid kernel init_gfs_hinterp_full). It returns the same dict.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np

from namgpu import kernels

UNIT = "init_gfs_hinterp"
KERNEL = "init_gfs_hinterp_strips"
BLOCK = 256             # threads per block (tuning knob, kept here)
LNSH = 5                # H-point boundary rows/cols (configfile lnsh)
LNSV = 5                # V-point boundary rows/cols (configfile lnsv)
TIME_LEVELS = 2         # boundary times in read_bc (n3 = 1,2 for 2-D; n4 = 1,2 for 3-D)
LAT_ORDER = "south_to_north"
GFS_QUARTER_DEG = dict(lat0=-90.0, lon0=0.0, dlat=0.25, dlon=0.25, lon_periodic=True)

_scratch: dict = {}


@dataclass
class Grid:
    lat0: float
    lon0: float
    dlat: float
    dlon: float
    lon_periodic: bool = True
    lat_order: str = LAT_ORDER

    def __post_init__(self):
        if self.lat_order != LAT_ORDER or not (self.dlat > 0.0):
            raise ValueError(
                f"init-gfs-hinterp: grid lat_order={self.lat_order!r} dlat={self.dlat}; the source must be "
                f"{LAT_ORDER!r} with dlat > 0 (row j = lat0 + j*dlat). A north-to-south GFS array would be "
                "silently misregistered, so it is refused.")


@dataclass
class GfsField:
    src: object             # cupy float32 [TIME_LEVELS, NL, NJS, NIS]
    mask: object            # cupy uint8, same shape (decode bits: 1 PRESENT, 4 FILLED are valid)


@dataclass
class GfsSource:
    grid: Grid
    fields: dict = field(default_factory=dict)   # name -> GfsField (any GFS field to interpolate)
    u: GfsField | None = None   # earth-relative eastward wind (init-gfs-wind-rotate)
    v: GfsField | None = None   # earth-relative northward wind (init-gfs-wind-rotate)


def _buffers(slot: str, nslab: int, im: int, jm: int, w: int):
    import cupy as cp
    key = (slot, nslab, im, jm, w)
    buf = _scratch.get(key)
    if buf is None:
        buf = {
            "s": cp.empty((nslab, w, im), dtype=cp.float32),
            "n": cp.empty((nslab, w, im), dtype=cp.float32),
            "w": cp.empty((nslab, jm, w), dtype=cp.float32),
            "e": cp.empty((nslab, jm, w), dtype=cp.float32),
            "missing": cp.zeros((1,), dtype=cp.int32),
        }
        _scratch[key] = buf
    return buf


def interp_strips(src, mask, glat, glon, *, width: int, grid: Grid | dict = GFS_QUARTER_DEG,
                  slot: str = "H", stream=None) -> dict:
    """Interpolate one GFS field (both boundary times) onto the four strips of one grid class, one launch."""
    import cupy as cp

    if not isinstance(grid, Grid):
        grid = Grid(**grid)
    if src.ndim != 4 or src.dtype != cp.float32:
        raise ValueError(f"init-gfs-hinterp: src shape {src.shape} dtype {src.dtype}, want 4-D float32 "
                         f"[{TIME_LEVELS}, NL, NJS, NIS]")
    if int(src.shape[0]) != TIME_LEVELS:
        raise ValueError(f"init-gfs-hinterp: src has {src.shape[0]} time levels, read_bc needs {TIME_LEVELS}")
    if tuple(mask.shape) != tuple(src.shape) or mask.dtype != cp.uint8:
        raise ValueError(f"init-gfs-hinterp: mask {mask.shape} {mask.dtype}, want {src.shape} uint8 "
                         "(a source without its decode mask is refused, not treated as all-present)")
    nt, nl, njs, nis = (int(x) for x in src.shape)
    nslab = nt * nl
    jm, im = (int(x) for x in glat.shape)
    for name, a in (("glat", glat), ("glon", glon)):
        if tuple(a.shape) != (jm, im) or a.dtype != cp.float32:
            raise ValueError(f"init-gfs-hinterp: {name} {a.shape} {a.dtype}, want ({jm}, {im}) float32")
    if width < 1 or 2 * width > min(im, jm):
        raise ValueError(f"init-gfs-hinterp: width {width} does not fit a {im} x {jm} grid")

    buf = _buffers(slot, nslab, im, jm, width)
    buf["missing"].fill(0)
    npts = max(width * im, jm * width)
    launch = ((npts + BLOCK - 1) // BLOCK, 4, nslab)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(launch, (BLOCK, 1, 1), (
            src, mask, glat, glon, buf["s"], buf["n"], buf["w"], buf["e"], buf["missing"],
            np.int32(im), np.int32(jm), np.int32(width), np.int32(nslab),
            np.int32(nis), np.int32(njs), np.int32(1 if grid.lon_periodic else 0),
            np.float32(grid.lat0), np.float32(grid.lon0), np.float32(grid.dlat), np.float32(grid.dlon),
        ))
    return {**buf, "nt": nt, "nl": nl}


def require_read_bc_levels(strips: dict, lm: int) -> None:
    """read_bc takes model-level strips. Isobaric strips (NL != LM) are refused until init-gfs-vinterp exists."""
    if int(strips["nl"]) != int(lm):
        raise RuntimeError(
            f"init-gfs-hinterp: strips have {strips['nl']} levels, read_bc needs {lm} model levels. "
            "Isobaric strips must go through init-gfs-vinterp (todo) first; refusing to feed read_bc.")


def _copy(strips: dict) -> dict:
    out = {k: strips[k].copy() for k in ("s", "n", "w", "e", "missing")}
    out["nt"], out["nl"] = strips["nt"], strips["nl"]
    return out


def run(ctx):
    """Driver entry. Interpolates every staged GFS field onto the H and V strips (both boundary times)."""
    gfs = getattr(ctx, "gfs", None)
    if gfs is None:
        return None
    st = ctx.st
    stream = ctx.stream
    result = {}
    for name, fld in gfs.fields.items():
        h = interp_strips(fld.src, fld.mask, st["GLAT"], st["GLON"], width=LNSH, grid=gfs.grid,
                          slot=f"H:{name}", stream=stream)
        v = interp_strips(fld.src, fld.mask, st["VLAT"], st["VLON"], width=LNSV, grid=gfs.grid,
                          slot=f"V:{name}", stream=stream)
        result[name] = {"H": _copy(h), "V": _copy(v)}
    if result:
        ctx.tables.setdefault("gfs_hinterp_strips", {}).update(result)
    return result


# ---------------------------------------------------------------------------
# Full-domain producer for init-gfs-vinterp (ctx.tables["gfs_hinterp"]).
# Same METGRID rule as the strips (kernel init_gfs_hinterp_full, one thread per point and level).
# ---------------------------------------------------------------------------
# index of each decoded pressure-level variable (init-gfs-decode PL_VARS order)
PL_IDX = dict(HGT=0, TMP=1, SPFH=2, RH=3, UGRD=4, VGRD=5, CLMR=6, ICMR=7, RWMR=8, SNMR=9)
FULL_KERNEL = "init_gfs_hinterp_full"
FULL_BLOCK = (16, 16, 1)


def interp_full(src, mask, glat, glon, *, grid: Grid | dict = GFS_QUARTER_DEG, stream=None):
    """One isobaric variable [NL, NJS, NIS] (one boundary time) onto every point of an H or V grid [JM, IM].

    Returns (out [NL, JM, IM] float32, nmiss device int32 [1]). nmiss counts points with no present
    neighbour (they are 0.0); it is not read on the host here.
    """
    import cupy as cp

    if not isinstance(grid, Grid):
        grid = Grid(**grid)
    if src.ndim != 3 or src.dtype != cp.float32:
        raise ValueError(f"init-gfs-hinterp: full-domain src shape {src.shape} dtype {src.dtype}, want 3-D float32")
    if tuple(mask.shape) != tuple(src.shape) or mask.dtype != cp.uint8:
        raise ValueError(f"init-gfs-hinterp: full-domain mask {mask.shape} {mask.dtype}, want {src.shape} uint8")
    nl, njs, nis = (int(x) for x in src.shape)
    jm, im = (int(x) for x in glat.shape)
    out = cp.empty((nl, jm, im), dtype=cp.float32)
    nmiss = cp.zeros((1,), dtype=cp.int32)
    bx, by, _ = FULL_BLOCK
    launch = ((im + bx - 1) // bx, (jm + by - 1) // by, nl)
    kern = kernels.get(UNIT, FULL_KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    with scope:
        kern(launch, FULL_BLOCK, (
            src, mask, glat, glon, out,
            np.int32(im), np.int32(jm), np.int32(nl), np.int32(nis), np.int32(njs),
            np.int32(1 if grid.lon_periodic else 0),
            np.float32(grid.lat0), np.float32(grid.lon0), np.float32(grid.dlat), np.float32(grid.dlon),
            nmiss,
        ))
    return out, nmiss


def full_domain_inputs(ctx) -> dict:
    """Full-domain GFS columns for init-gfs-vinterp, from the decoded set (ctx.tables["gfs_decoded"]).

    Keys (device arrays, [NL, JM, IM], index 0 = lowest level): PRES_IN, GHT_IN, TT_IN, RH_IN, UU_IN,
    VV_IN, SNMR_IN, CLWMR_IN, RWMR_IN, CICE_IN, FRIMEF_IN.
      PRES_IN   the isobaric level pressure (Pa), constant over the grid (isobaric GFS).
      GHT_IN    geopotential height (m) on H points; UU_IN, VV_IN earth-relative winds on V points.
      FRIMEF_IN zeros: GFS carries no rime-factor field and the Fortran rimef_input is never read
                (dead output), so zero is the value the consumer can see.
    Refuses (no fallback data) when no decoded set is present.
    """
    import cupy as cp

    dec = ctx.tables.get("gfs_decoded")
    if dec is None:
        raise RuntimeError(
            "init-gfs-vinterp: no decoded GFS set (ctx.tables['gfs_decoded'] from init-gfs-decode). "
            "Refusing: there is no random or fallback GFS column set.")
    st = ctx.st
    gfs_grid = Grid(**GFS_QUARTER_DEG)
    nl = int(dec.pl.shape[1])
    jm, im = (int(x) for x in st["GLAT"].shape)
    h_pts = (st["GLAT"], st["GLON"])
    v_pts = (st["VLAT"], st["VLON"])
    stream = getattr(ctx, "stream", None)

    def field(name, pts):
        idx = PL_IDX[name]
        out, _ = interp_full(dec.pl[idx], dec.pl_mask[idx], pts[0], pts[1], grid=gfs_grid, stream=stream)
        return out

    pres = cp.ascontiguousarray(cp.broadcast_to(dec.plev.astype(cp.float32)[:, None, None], (nl, jm, im)))
    return {
        "PRES_IN": pres,
        "GHT_IN": field("HGT", h_pts),
        "TT_IN": field("TMP", h_pts),
        "RH_IN": field("RH", h_pts),
        "UU_IN": field("UGRD", v_pts),
        "VV_IN": field("VGRD", v_pts),
        "SNMR_IN": field("SNMR", h_pts),
        "CLWMR_IN": field("CLMR", h_pts),
        "RWMR_IN": field("RWMR", h_pts),
        "CICE_IN": field("ICMR", h_pts),
        "FRIMEF_IN": cp.zeros((nl, jm, im), dtype=cp.float32),
    }
