"""init-native-sfc: surface and soil initial state from the NAM native analysis into ctx.st.

Unit: init-native-sfc (group init, mode new, no Fortran counterpart). Kernel:
namgpu/kernels/init_native_sfc.cu (kernel init_native_sfc), one launch, one thread per column.

Inputs (device arrays, names fixed by the GRIB decode unit init-grib2-decode):
    ctx.tables["raw_sfc"]          dict of cupy arrays, shapes in the kernel header:
        TSOIL [4, JM, IM] K          SOILW [4, JM, IM] volumetric      TMP_SFC [JM, IM] K
        WTMP [JM, IM] K              ICEC [JM, IM]                     LAND [JM, IM] (1 = land)
        TSOIL_3M [JM, IM] K          SNOD [JM, IM] m                   WEASD [JM, IM] kg m-2
        ALBDO [JM, IM] percent       VEG [JM, IM] percent              SFCR [JM, IM] m
        SNOWC [JM, IM]               CNWAT [JM, IM] kg m-2
        VGTYP [JM, IM] int32         SOTYP [JM, IM] int32
    ctx.tables["soil_hydraulics"]  dict with "bexp", "psisat" (m), "smcmax": cupy float32 [30],
                                   from init-static-missing (soil class table).
Outputs (ctx.st, NMMB names, CONTRACT layout):
    STC, SMC, SH2O [4, JM, IM]; TSKIN, SST, SICE, SM, TG, SI (mm), SNO (mm), SNOWC, ALBEDO, VEGFRC, Z0,
    Z0BASE, CMC [JM, IM] float32; IVGTYP, ISLTYP [JM, IM] int32.
Conversions and the frozen-fraction formula are documented in the kernel header.
Run the self-test (random plausible inputs, full size, NaN/Inf check):
    source env.sh && python -m namgpu.init.init_native_sfc
"""
from __future__ import annotations

import contextlib

from namgpu import kernels

UNIT = "init-native-sfc"
KERNEL = "init_native_sfc"
KFILE = "init_native_sfc"   # namgpu/kernels/<KFILE>.cu (underscored unit id)
NCLASS = 30
NSOIL = 4
BLOCK = 128

RAW_KEYS = ("TSOIL", "SOILW", "TMP_SFC", "ICEC", "LAND", "TSOIL_3M", "SNOD", "WEASD",
            "ALBDO", "VEG", "SFCR", "CNWAT", "VGTYP", "SOTYP")
OPT_KEYS = ("SNOWC",)   # NAM's snowc record overrides the 0.98/0 rule when present
OUT_FLOAT_2D = ("TSKIN", "SST", "SICE", "SM", "TG", "SI", "SNO", "SNOWC", "ALBEDO", "VEGFRC",
                "Z0", "Z0BASE", "CMC")
OUT_SOIL = ("STC", "SMC", "SH2O")


def _check_raw(raw, im, jm):
    missing = [k for k in RAW_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{UNIT}: ctx.tables['raw_sfc'] lacks {missing} (init-grib2-decode)")
    for k in RAW_KEYS + tuple(o for o in OPT_KEYS if o in raw):
        shp = tuple(raw[k].shape)
        want = (NSOIL, jm, im) if k in ("TSOIL", "SOILW") else (jm, im)
        if shp != want:
            raise ValueError(f"{UNIT}: raw {k} shape {shp}, want {want}")


def run(ctx):
    import cupy as cp
    import numpy as np

    st = ctx.st
    im, jm = int(ctx.grid.im), int(ctx.grid.jm)
    raw = ctx.tables["raw_sfc"]
    soil = ctx.tables["soil_hydraulics"]
    _check_raw(raw, im, jm)

    bexp = cp.ascontiguousarray(soil["bexp"], dtype=cp.float32)
    psisat = cp.ascontiguousarray(soil["psisat"], dtype=cp.float32)
    smcmax = cp.ascontiguousarray(soil["smcmax"], dtype=cp.float32)
    if bexp.shape != (NCLASS,) or psisat.shape != (NCLASS,) or smcmax.shape != (NCLASS,):
        raise ValueError(f"{UNIT}: soil_hydraulics tables must have {NCLASS} classes")

    f = st.fields
    if "ALBEDO" not in f:
        st.add("ALBEDO", (jm, im))
        f = st.fields
    if "SNOWC" in raw:
        snowc_arr, has_snowc = raw["SNOWC"], 1
    else:
        snowc_arr, has_snowc = cp.zeros((jm, im), dtype=cp.float32), 0
    has_snowc_d = cp.asarray([has_snowc], dtype=cp.int32)
    k = kernels.get(KFILE, KERNEL)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    grid, block = kernels.grid_cols(im, jm), (BLOCK,)
    with scope:
        k(grid, block, (
            raw["TSOIL"], raw["SOILW"], raw["TMP_SFC"], raw["ICEC"], raw["LAND"],
            raw["TSOIL_3M"], raw["SNOD"], raw["WEASD"], raw["ALBDO"], raw["VEG"], raw["SFCR"],
            snowc_arr, has_snowc_d, raw["CNWAT"], raw["VGTYP"], raw["SOTYP"],
            bexp, psisat, smcmax,
            f["STC"], f["SMC"], f["SH2O"],
            f["TSKIN"], f["SST"], f["SICE"], f["SM"], f["TG"], f["SI"], f["SNO"], f["SNOWC"],
            f["ALBEDO"], f["VEGFRC"], f["Z0"], f["Z0BASE"], f["CMC"],
            f["IVGTYP"], f["ISLTYP"],
            np.int32(im), np.int32(jm),
        ))
    return st


def _selftest():
    """Full-size random but plausible state (954 x 835 x 60 grid, the real NAM parent size)."""
    import cupy as cp
    import numpy as np
    from types import SimpleNamespace

    from namgpu import state as statemod

    im, jm, lm = 954, 835, 60
    grid = SimpleNamespace(im=im, jm=jm, lm=lm)
    st = statemod.State(grid)
    rng = cp.random.default_rng(7)
    raw = {
        "TSOIL": 250.0 + 60.0 * rng.random((NSOIL, jm, im), dtype=cp.float32),
        "SOILW": 0.02 + 0.5 * rng.random((NSOIL, jm, im), dtype=cp.float32),
        "TMP_SFC": 250.0 + 60.0 * rng.random((jm, im), dtype=cp.float32),
        "ICEC": rng.random((jm, im), dtype=cp.float32),
        "LAND": (rng.random((jm, im), dtype=cp.float32) > 0.5).astype(cp.float32),
        "TSOIL_3M": 260.0 + 30.0 * rng.random((jm, im), dtype=cp.float32),
        "SNOD": 0.5 * rng.random((jm, im), dtype=cp.float32),
        "WEASD": 200.0 * rng.random((jm, im), dtype=cp.float32),
        "ALBDO": 100.0 * rng.random((jm, im), dtype=cp.float32),
        "VEG": 100.0 * rng.random((jm, im), dtype=cp.float32),
        "SFCR": 0.01 + rng.random((jm, im), dtype=cp.float32),
        "SNOWC": rng.random((jm, im), dtype=cp.float32),
        "CNWAT": 0.1 * rng.random((jm, im), dtype=cp.float32),
        "VGTYP": rng.integers(1, 21, (jm, im), dtype=cp.int32),
        "SOTYP": rng.integers(1, 13, (jm, im), dtype=cp.int32),
    }
    # Test-only soil classes (the real table comes from init-static-missing).
    soil = {
        "bexp": (2.5 + 10.0 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
        "psisat": (0.03 + 0.7 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
        "smcmax": (0.33 + 0.25 * cp.linspace(0, 1, NCLASS, dtype=cp.float32)),
    }
    ctx = SimpleNamespace(st=st, grid=grid, stream=cp.cuda.Stream(non_blocking=True),
                          tables={"raw_sfc": raw, "soil_hydraulics": soil})
    run(ctx)
    ctx.stream.synchronize()
    bad = []
    for name in OUT_SOIL + OUT_FLOAT_2D:
        a = st.fields[name]
        if not bool(cp.isfinite(a).all()):
            bad.append(name)
    frozen = float(cp.mean(st.fields["SMC"] - st.fields["SH2O"]))
    print(f"{UNIT}: fields {len(OUT_SOIL) + len(OUT_FLOAT_2D) + 2}, "
          f"NaN/Inf in {bad or 'none'}, mean frozen fraction {frozen:.4f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
