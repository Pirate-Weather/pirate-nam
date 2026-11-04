"""phys-rrtm-aer-property: AER_PROPERTY + RADCLIMAER, the sw/lw aerosol optics of the NMMB radiation module.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/radiation_aerosols_nmmb.f:2377-3107. NOAA US government
work, no licence file; the NOAA module header is kept as the notice in the kernel file
(namgpu/kernels/phys_rrtm_aer_property.cu).

Kernel: namgpu/kernels/phys_rrtm_aer_property.cu, entry phys_rrtm_aer_property_column, one thread per
column (kernels.grid_cols), one launch on ctx.stream, no host sync, no allocation after the first call.

Inputs (CONTRACT names, device float32 unless noted; Fortran A(i,j,l) -> [l, j, i]):
  prsi    [LM+1, JM, IM]  interface pressure, mb (prsi(i,k))
  dz      [LM,   JM, IM]  layer thickness, m
  hz      [LM+1, JM, IM]  level height, m
  rhlay   [LM,   JM, IM]  layer mean relative humidity
  alon    [JM, IM]        longitude, degrees 0..360 (caller adds 360 to negatives)
  alat    [JM, IM]        latitude, degrees, clamped to -90..90 by the caller
  slmsk   [JM, IM]        sea/land mask (0 sea, 1 land, 2 sea-ice)
  prsl, prslk, tvly, tracer are Fortran arguments aer_property never reads; they are not bound.
Tables (ctx.tables["rrtm_aer"]; built by phys-rrtm-aer-init and phys-rrtm-clim-aerinit, read-only here;
the Fortran index order is reversed in memory):
  haer, prsref, sigref  [NAE=7, NDM=5]
  rhlev                 [8]
  extrhi, scarhi, ssarhi, asyrhi  [NSWLWBD=30, NCM1=6]
  extrhd, scarhd, ssarhd, asyrhd  [NSWLWBD=30, NCM2=4, NRHLEV=8]
  extstra               [30]
  cmixg                 [JMXAE=37, IMXAE=72, NXC=5]  float32
  denng                 [JMXAE=37, IMXAE=72, 2]      float32
  idxcg                 [JMXAE=37, IMXAE=72, NXC=5]  int32
  kprfg                 [JMXAE=37, IMXAE=72]         int32
Outputs (scratch, allocated once by new_outputs, written every call when the flag is set):
  aerosw [3, NBDSW=14, LM, JM, IM]   Fortran aerosw(i,k,m,f): f = 0 tau, 1 ssa, 2 asy
  aerolw [3, NBDLW=16, LM, JM, IM]

Scalars: ivflip (physparam.f:164, default 1 = input surface to top; the index meaning is the Fortran
one, so the caller must hand over arrays in that order or pass ivflip = 0), laersw, laerlw,
nswbnd = 14, nlwbnd = 16 (lalw1bd false), nswlwbd = nswbnd + nlwbnd = 30.

run(ctx) is the done-rule entry. The done-rule run ignores ctx's fields and builds a full-size random
state (954 x 835 x 60) with plausible tables (every cmixg/idxcg valid, so no column has zero extinction),
launches the kernel once and returns the outputs. The production binding (rhlay, dz, hz from the NMMB
state) is the caller's job in phys-rrtm-grrad-aerosols; this unit does not reach into ctx.st.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_aer_property"
KERNEL = "phys_rrtm_aer_property_column"
TABLE_KEY = "rrtm_aer"

NDM, NAE = 5, 7
IMXAE, JMXAE = 72, 37
NXC, NRHLEV = 5, 8
NCM1, NCM2 = 6, 4
NSWLWBD = 30
NBDSW, NBDLW = 14, 16
NF_AESW, NF_AELW = 3, 3
LM_MAX = 64

_RHLEV = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99], dtype=np.float32)


def new_outputs(im: int, jm: int, lm: int):
    """aerosw / aerolw scratch, allocated once and reused."""
    import cupy as cp

    return {
        "aerosw": cp.zeros((NF_AESW, NBDSW, lm, jm, im), dtype=cp.float32),
        "aerolw": cp.zeros((NF_AELW, NBDLW, lm, jm, im), dtype=cp.float32),
    }


def bands(laswflg: bool = True, lalwflg: bool = True, lalw1bd: bool = False):
    """Band counts as aer_init sets them (radiation_aerosols_nmmb.f:521-537):
    NSWBND = NBDSW if laswflg else 0; NLWBND = 1 if lalw1bd else NBDLW if lalwflg else 0;
    NSWLWBD = NSWBND + NLWBND."""
    nswbnd = NBDSW if laswflg else 0
    nlwbnd = (1 if lalw1bd else NBDLW) if lalwflg else 0
    return nswbnd, nlwbnd, nswbnd + nlwbnd


def _check_inputs(arrs, lm: int, ivflip: int):
    """The refusals the Fortran makes with stop, which the kernel would otherwise answer silently:
    alon outside [0, 360] (lab_do_IMXAE), alat outside [-90, 90] (lab_do_JMXAE), and the reference
    surface pressure not above 100 mb (prsi(i,1) for ivflip 1, prsi(i,NLP1) for ivflip 0).
    One host read of five reductions per launch, not on a forecast step."""
    import cupy as cp

    ref = arrs["prsi"][0] if ivflip == 1 else arrs["prsi"][lm]
    st = cp.asnumpy(cp.stack([arrs["alon"].min(), arrs["alon"].max(),
                              arrs["alat"].min(), arrs["alat"].max(), ref.min()]))
    lo_lon, hi_lon, lo_lat, hi_lat, p_sfc = (float(x) for x in st)
    if not (0.0 <= lo_lon and hi_lon <= 360.0):
        raise ValueError(f"{UNIT}: alon outside [0, 360] (min {lo_lon}, max {hi_lon}); the Fortran "
                         "stops here. The caller adds 360 to negative longitudes.")
    if not (-90.0 <= lo_lat and hi_lat <= 90.0):
        raise ValueError(f"{UNIT}: alat outside [-90, 90] (min {lo_lat}, max {hi_lat}); the Fortran stops here.")
    if not (p_sfc > 100.0):
        raise ValueError(f"{UNIT}: surface pressure min {p_sfc} mb is not above 100; the Fortran stops "
                         "on an unrealistic surface pressure.")


def launch(im, jm, lm, tbl, arrs, outs, *, laersw=True, laerlw=True, ivflip=1,
           laswflg=True, lalwflg=True, lalw1bd=False, stream=None):
    """One launch over every column. arrs: prsi, dz, hz, rhlay, alon, alat, slmsk (device float32).
    tbl: the tables listed in the module docstring. outs: from new_outputs.
    ivflip: 1 = arrays surface first (physparam default, what the Fortran caller builds);
    0 = arrays top first (the CONTRACT orientation). laswflg, lalwflg, lalw1bd: the physparam flags;
    the band counts NSWBND, NLWBND, NSWLWBD are derived from them exactly as aer_init does."""
    if lm > LM_MAX:
        raise ValueError(f"{UNIT}: lm={lm} exceeds the kernel scratch bound {LM_MAX}")
    if ivflip not in (0, 1):
        raise ValueError(f"{UNIT}: ivflip={ivflip}; physparam ivflip is 0 (toa to sfc) or 1 (sfc to toa)")
    nswbnd, nlwbnd, nswlwbd = bands(laswflg, lalwflg, lalw1bd)
    if laersw and nswlwbd < NBDSW:
        raise ValueError(f"{UNIT}: laersw with NSWLWBD={nswlwbd} < NBDSW={NBDSW}: the Fortran copies "
                         f"{NBDSW} sw bands from an {nswlwbd}-band array. Pass laersw=False.")
    if laerlw and nlwbnd == 0:
        raise ValueError(f"{UNIT}: laerlw with lalwflg off: the Fortran copies {NBDLW} lw bands from an "
                         "array with no lw bands. Pass laerlw=False.")
    _check_inputs(arrs, lm, ivflip)
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    i32 = np.int32
    with scope:
        kern(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            arrs["prsi"], arrs["dz"], arrs["hz"], arrs["rhlay"],
            arrs["alon"], arrs["alat"], arrs["slmsk"],
            tbl["haer"], tbl["prsref"], tbl["sigref"], tbl["rhlev"],
            tbl["extrhi"], tbl["scarhi"], tbl["ssarhi"], tbl["asyrhi"],
            tbl["extrhd"], tbl["scarhd"], tbl["ssarhd"], tbl["asyrhd"],
            tbl["extstra"], tbl["cmixg"], tbl["denng"], tbl["idxcg"], tbl["kprfg"],
            outs["aerosw"], outs["aerolw"],
            i32(im), i32(jm), i32(lm), i32(int(laersw)), i32(int(laerlw)), i32(ivflip),
            i32(nswbnd), i32(nlwbnd), i32(nswlwbd),
        ))
    return outs


BUNDLE_KEY = "rrtm_aer"
_CLIM_KEYS = ("haer", "prsref", "sigref", "extrhi", "scarhi", "ssarhi", "asyrhi",
              "extrhd", "scarhd", "ssarhd", "asyrhd", "extstra")


def bind_tables(tables):
    """The aer_property table bundle from the init units: clim_aerinit (haer .. extstra, top-level keys of
    ctx.tables), the module DATA rhlev, and the monthly climatology of aer_update (aer_idxcg, aer_cmixg,
    aer_denng, aer_kprfg; cmixg and denng are kind_phys there and float32 here, cast every call because
    aer_update rewrites them on a month change; 37 x 72 x 5 values). Missing tables raise by name."""
    import cupy as cp

    b = tables.get(BUNDLE_KEY)
    if b is None:
        b = {}
        for k in _CLIM_KEYS:
            if k not in tables:
                raise RuntimeError(f"{UNIT}: ctx.tables[{k!r}] is missing (phys-rrtm-clim-aerinit writes it at init)")
            b[k] = cp.ascontiguousarray(tables[k], dtype=cp.float32)
        b["rhlev"] = cp.asarray(_RHLEV)
        tables[BUNDLE_KEY] = b
    for k, dt in (("idxcg", cp.int32), ("cmixg", cp.float32), ("denng", cp.float32), ("kprfg", cp.int32)):
        src = tables.get("aer_" + k)
        if src is None:
            raise RuntimeError(f"{UNIT}: ctx.tables['aer_{k}'] is missing (phys-rrtm-aer-update writes it; "
                               "radupdate calls it on the first step and at each month change)")
        b[k] = cp.ascontiguousarray(src, dtype=dt)
    return b


def run_call(*, prsi, dz, hz, rhlay, alon, alat, slmsk, laersw, laerlw, imax, nlay, nlp1, ivflip,
             aerosw, aerolw, stream=None, tables=None, prsl=None, prslk=None, tvly=None, tracer=None,
             laswflg=True, lalwflg=True, lalw1bd=False):
    """AER_PROPERTY as setaer calls it (radiation_aerosols_nmmb.f:2045): column arrays [.., IMAX] (one
    launch over IMAX columns, jm = 1), tables from bind_tables(tables). prsl, prslk, tvly, tracer are Fortran
    arguments aer_property never reads. The input range check (a host read) runs on the first call only."""
    if tables is None:
        raise RuntimeError(f"{UNIT}: run_call needs tables= (ctx.tables) to bind the aerosol optics tables")
    tbl = bind_tables(tables)
    arrs = {"prsi": prsi, "dz": dz, "hz": hz, "rhlay": rhlay, "alon": alon, "alat": alat, "slmsk": slmsk}
    if not tables.get(BUNDLE_KEY + ":checked"):
        _check_inputs(arrs, nlay, ivflip)
        tables[BUNDLE_KEY + ":checked"] = True
    nswbnd, nlwbnd, nswlwbd = bands(laswflg, lalwflg, lalw1bd)
    if nlay > LM_MAX:
        raise ValueError(f"{UNIT}: nlay={nlay} exceeds the kernel scratch bound {LM_MAX}")
    kern = kernels.get(UNIT, KERNEL)
    scope = stream if stream is not None else contextlib.nullcontext()
    i32 = np.int32
    with scope:
        kern(kernels.grid_cols(imax, 1), kernels.BLOCK_COL, (
            prsi, dz, hz, rhlay, alon, alat, slmsk,
            tbl["haer"], tbl["prsref"], tbl["sigref"], tbl["rhlev"],
            tbl["extrhi"], tbl["scarhi"], tbl["ssarhi"], tbl["asyrhi"],
            tbl["extrhd"], tbl["scarhd"], tbl["ssarhd"], tbl["asyrhd"],
            tbl["extstra"], tbl["cmixg"], tbl["denng"], tbl["idxcg"], tbl["kprfg"],
            aerosw, aerolw,
            i32(imax), i32(1), i32(nlay), i32(int(laersw)), i32(int(laerlw)), i32(ivflip),
            i32(nswbnd), i32(nlwbnd), i32(nswlwbd),
        ))
    return {"aerosw": aerosw, "aerolw": aerolw}


def _random_tables(seed: int):
    """Plausible climatology for the done-rule run. Values are in range, not a real aerosol set."""
    import cupy as cp

    rng = np.random.default_rng(seed + 1)
    f32 = np.float32

    def u(lo, hi, shape):
        return rng.uniform(lo, hi, shape).astype(f32)

    haer = u(1.0, 8.0, (NAE, NDM))                       # km, [kp][idm]
    prsref = np.sort(u(100.0, 1000.0, (NAE, NDM)), axis=1)[:, ::-1].copy()
    sigref = np.sort(u(0.05, 1.0, (NAE, NDM)), axis=1)[:, ::-1].copy()

    extrhi = u(0.2, 2.0, (NSWLWBD, NCM1))
    ssarhi = u(0.5, 0.99, (NSWLWBD, NCM1))
    asyrhi = u(0.5, 0.9, (NSWLWBD, NCM1))
    scarhi = (extrhi * ssarhi).astype(f32)

    extrhd = u(0.2, 4.0, (NSWLWBD, NCM2, NRHLEV))
    ssarhd = u(0.5, 0.99, (NSWLWBD, NCM2, NRHLEV))
    asyrhd = u(0.5, 0.9, (NSWLWBD, NCM2, NRHLEV))
    scarhd = (extrhd * ssarhd).astype(f32)

    extstra = u(1.0e-4, 1.0e-3, (NSWLWBD,))

    cmixg = u(1.0e-8, 1.0e-6, (JMXAE, IMXAE, NXC))
    denng = u(1.0e6, 1.0e8, (JMXAE, IMXAE, 2))
    idxcg = rng.integers(1, 11, (JMXAE, IMXAE, NXC)).astype(np.int32)   # 1..NCM, never 0
    kprfg = rng.integers(1, NAE + 1, (JMXAE, IMXAE)).astype(np.int32)   # 1..7

    host = {
        "haer": haer, "prsref": prsref, "sigref": sigref, "rhlev": _RHLEV,
        "extrhi": extrhi, "scarhi": scarhi, "ssarhi": ssarhi, "asyrhi": asyrhi,
        "extrhd": extrhd, "scarhd": scarhd, "ssarhd": ssarhd, "asyrhd": asyrhd,
        "extstra": extstra, "cmixg": cmixg, "denng": denng, "idxcg": idxcg, "kprfg": kprfg,
    }
    return {k: cp.asarray(v) for k, v in host.items()}


def _random_inputs(im: int, jm: int, lm: int, seed: int):
    """Full-size column inputs on the device, surface-first (ivflip = 1)."""
    import cupy as cp

    cp.random.seed(seed)
    lmp = lm + 1
    base = cp.linspace(1010.0, 2.0, lmp, dtype=cp.float32)[:, None, None]
    jitter = 1.0 + 0.01 * (cp.random.random((1, jm, im), dtype=cp.float32) - 0.5)
    prsi = (base * jitter).astype(cp.float32)                       # decreasing with level
    dz = (50.0 + 400.0 * cp.random.random((lm, jm, im), dtype=cp.float32)).astype(cp.float32)
    hz = cp.concatenate([cp.zeros((1, jm, im), dtype=cp.float32), cp.cumsum(dz, axis=0)], axis=0)
    rhlay = cp.random.random((lm, jm, im), dtype=cp.float32)
    alon = (360.0 * cp.random.random((jm, im), dtype=cp.float32)).astype(cp.float32)
    alat = (-90.0 + 180.0 * cp.random.random((jm, im), dtype=cp.float32)).astype(cp.float32)
    slmsk = cp.floor(3.0 * cp.random.random((jm, im), dtype=cp.float32)).astype(cp.float32)
    return {"prsi": prsi, "dz": dz, "hz": hz, "rhlay": rhlay,
            "alon": alon, "alat": alat, "slmsk": slmsk}


def _slab(full, j0: int, j1: int):
    import cupy as cp

    out = {}
    for k, v in full.items():
        part = v[:, j0:j1, :] if v.ndim == 3 else v[j0:j1, :]
        out[k] = cp.ascontiguousarray(part)
    return out


def _summarise(summary, outs, jm: int):
    import cupy as cp

    for name, a in outs.items():
        rec = summary.setdefault(name, {"nonfinite": 0, "min": float("inf"),
                                        "max": float("-inf"), "shape": None})
        rec["nonfinite"] += int((~cp.isfinite(a)).sum())
        rec["min"] = min(rec["min"], float(a.min()))
        rec["max"] = max(rec["max"], float(a.max()))
        rec["shape"] = (a.shape[0], a.shape[1], a.shape[2], jm, a.shape[4])


def run(ctx=None, im: int = 954, jm: int = 835, lm: int = 60, seed: int = 0, slabs: int = 5):
    """Done-rule entry: production arm over the full-size random state, plus two checks on slab 0.

    Production arm: every slab with the production flags (laswflg, lalwflg on; lalw1bd off) and ivflip 1
    on surface-first input. The outputs are about 17 GB at full size, so the domain runs as `slabs` row
    slabs of IM = 954 columns each. Columns are independent.
    Mirror arm (slab 0): the same state flipped to top-first with ivflip 0. Each layer sees the same
    interface pressures, heights and thicknesses, and the smoothing sweeps are mirrored, so the outputs
    must equal the production outputs layer-reversed. Max abs difference is returned.
    Flag-off arm (slab 0): laswflg off, so NSWBND = 0 and NSWLWBD = 16. Finiteness is checked.
    """
    import cupy as cp

    if jm % slabs:
        raise ValueError(f"{UNIT}: jm={jm} does not split into {slabs} slabs")
    rows = jm // slabs
    tbl = _random_tables(seed)
    full = _random_inputs(im, jm, lm, seed)
    summary = {}
    prod0 = arrs0 = None
    for s in range(slabs):
        arrs = _slab(full, s * rows, (s + 1) * rows)
        outs = new_outputs(im, rows, lm)
        launch(im, rows, lm, tbl, arrs, outs, laersw=True, laerlw=True, ivflip=1)
        cp.cuda.Stream.null.synchronize()
        _summarise(summary, outs, jm)
        if s == 0:
            prod0, arrs0 = outs, arrs
        else:
            del outs, arrs
    results = {"production": summary}

    top = {k: (cp.ascontiguousarray(arrs0[k][::-1]) if k in ("prsi", "dz", "hz", "rhlay") else arrs0[k])
           for k in arrs0}
    outs_m = new_outputs(im, rows, lm)
    launch(im, rows, lm, tbl, top, outs_m, laersw=True, laerlw=True, ivflip=0)
    cp.cuda.Stream.null.synchronize()
    mirror = {}
    for name in prod0:
        mirror[name] = float(cp.abs(prod0[name][:, :, ::-1] - outs_m[name]).max())
    results["mirror_ivflip0_max_abs_diff"] = mirror
    del outs_m, top, prod0

    outs_f = new_outputs(im, rows, lm)
    launch(im, rows, lm, tbl, arrs0, outs_f, laersw=True, laerlw=True, ivflip=1, laswflg=False)
    cp.cuda.Stream.null.synchronize()
    off = {}
    _summarise(off, outs_f, jm)
    results["laswflg_off_slab0"] = off
    return results
