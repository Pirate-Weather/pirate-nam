"""phys-rrtm-lw-taugb07: LW band 7 optical depths and Planck fractions of the NMMB RRTM longwave.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1) phys/radlw_main_nmmb.f:4687-4942, subroutine taugb07.
The header block of that file is the notice in namgpu/kernels/phys_rrtm_lw_taugb07.cu (US government work).

Kernel: namgpu/kernels/phys_rrtm_lw_taugb07.cu, phys_rrtm_lw_taugb07. One thread per column,
the layer loops inside the thread. One launch per column chunk on ctx.stream, no host sync, no print.
Scratch outputs are allocated once per chunk shape and reused.

Columns: the Fortran works one column at a time (taumol is called inside `lab_do_iplon`), so every
per-layer array here is per column: [ncol, nlay] with ncol = IM*JM (col = j0*IM + i0) and nlay = LM.
The domain is processed in column chunks (chunk_cols, default 2**17) so the VRAM peak is set by the
chunk, not by the 796,590-column domain. Each column is independent, so chunking changes no value.

Inputs (CuPy arrays, the taumol state for band 7; names are the Fortran names, shapes above):
    colamt_h2o [ncol, nlay] float32   colamt(k,1)
    colamt_co2 [ncol, nlay] float32   colamt(k,2)
    colamt_o3  [ncol, nlay] float32   colamt(k,3)
    coldry     [ncol, nlay] float32
    rfrate_h2oo3 [ncol, nlay, 2] float32   rfrate(k,2,n), n = 1, 2 (setcoef, radlw_main_nmmb.f:2080-2081)
    jp, jt, jt1, indself, indfor, indminor  [ncol, nlay] int32   (jp, jt, jt1 are 1-based, as Fortran)
    selffac, selffrac, forfac, forfrac, minorfrac, fac00, fac01, fac10, fac11  [ncol, nlay] float32
    laytrop    [ncol] int32      number of lower-atmosphere layers (Fortran laytrop, 1-based count)
Tables (ctx.tables["rrtm_lw_kgb07"], uploaded once from namgpu/data/rrtm_lw/kgb07.npz, the table that
the Fortran reads through `use module_radlw_kgb07`):
    absa [595, 12] (585 Fortran rows + zero pad), absb [236, 12] (235 + zero pad),
    selfref [10, 12], forref [4, 12], fracrefa [9, 12], fracrefb [12],
    ka_mco2 [19, 9, 12], kb_mco2 [19, 12]
    chi_mls [59, 7] (ctx.tables["rrtm_lw_chi_mls"], from namgpu/data/rrtm_lw/ref_chi_mls.npz, module_radlw_ref)
Outputs (band-local, the assembler places them at the Fortran offset NS07 = 76 of the 140 g-points):
    taug7  [chunk, nlay, 12] float32   taug(ns07+ig, k), ig = 1..12 at [..., ig-1]
    fracs7 [chunk, nlay, 12] float32   fracs(ns07+ig, k)

Vertical order (declared, CONTRACT section 1): the RRTM internal order of radlw_main_nmmb.f with
ivflip = 0, index 0 = the LOWEST layer (surface side). Fortran k = 1 is the surface layer and laytrop counts
lower layers up from it, so this unit takes and returns its per-layer arrays in that order and does not flip
them. setcoef (phys-rrtm-lw-setcoef) must supply the inputs bottom-up; lwrad_prep reverses the NMMB arrays.

Tables are read with the same index arithmetic as the Fortran (ind1 uses jp, not jp-1, as taugb07 does).
Table padding (declared deviation): at the edge of its own index range the Fortran reads past the end of
its tables, which is undefined memory there. Lower layers reach id 594 of absa (585 Fortran rows) at
jp = 12, jt1 = 5, js1 = 8; upper layers reach id 236 of absb (235 rows) at jp = 58, jt1 = 5. absa is
zero-padded to ABSA_ROWS = 595 rows and absb to ABSB_ROWS = 236 rows, so those reads return 0. Every index
that lies inside the Fortran tables is unchanged.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_lw_taugb07"
KERNEL = "phys_rrtm_lw_taugb07"
DATA = Path(__file__).resolve().parents[2] / "data" / "rrtm_lw"
KGB07_FILE = DATA / "kgb07.npz"
CHI_FILE = DATA / "ref_chi_mls.npz"
KEY_TABLE = "rrtm_lw_kgb07"
KEY_CHI = "rrtm_lw_chi_mls"
NG07 = 12
NLAY_DEFAULT = 60
CHUNK_COLS = 1 << 17
TABLE_NAMES = ("absa", "absb", "selfref", "forref", "fracrefa", "fracrefb", "ka_mco2", "kb_mco2")
INPUT_NAMES = (
    "colamt_h2o", "colamt_co2", "colamt_o3", "coldry", "rfrate_h2oo3",
    "jp", "jt", "jt1", "indself", "indfor", "indminor",
    "selffac", "selffrac", "forfac", "forfrac", "minorfrac",
    "fac00", "fac01", "fac10", "fac11", "laytrop",
)


ABSA_ROWS = 595   # Fortran absa has 585 rows; lower-layer ids reach 594 (module docstring)
ABSB_ROWS = 236   # Fortran absb has 235 rows; upper-layer ids reach 236 (module docstring)


def _pad_rows(a, rows):
    """Zero-pad a [n, 12] float32 table to `rows` rows (declared deviation, module docstring)."""
    assert a.shape[0] <= rows, (a.shape, rows)
    out = np.zeros((rows, a.shape[1]), dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


def tables(ctx):
    """Upload the band-7 k-tables and chi_mls once; cached in ctx.tables."""
    import cupy as cp

    t = ctx.tables
    if KEY_TABLE not in t:
        z = np.load(KGB07_FILE)
        host = {k: z[k] for k in TABLE_NAMES}
        host["absa"] = _pad_rows(host["absa"], ABSA_ROWS)
        host["absb"] = _pad_rows(host["absb"], ABSB_ROWS)
        t[KEY_TABLE] = {k: cp.asarray(v) for k, v in host.items()}
    if KEY_CHI not in t:
        t[KEY_CHI] = cp.asarray(np.load(CHI_FILE)["chi_mls"])
    return t[KEY_TABLE], t[KEY_CHI]


def _scratch(ctx, nc, nlay):
    import cupy as cp

    key = "rrtm_lw_taugb07_out"
    out = ctx.tables.get(key)
    if out is None or out[0].shape != (nc, nlay, NG07):
        out = (cp.empty((nc, nlay, NG07), dtype=cp.float32),
               cp.empty((nc, nlay, NG07), dtype=cp.float32))
        ctx.tables[key] = out
    return out


def _random_inputs(nc, nlay, seed):
    """Random-but-plausible band-7 state on the device, indices inside the table bounds.

    Lower layers (k <= laytrop): jp in 1..12. setcoef gives laytrop layers jp <= 12 (laytrop test
    plog > 4.56 and jp = max(1, min(58, int(36 - 5*(plog + 0.04)))), radlw_main_nmmb.f:2006 and 2050-2061);
    jp = 12 with jt1 = 5 is the edge that reaches absa row 594. jt/jt1 in 1..5, inds in 1..9, indf in 1..3,
    indm in 1..18.
    Upper layers: jp in 13..58 (the setcoef clamp at 58; jp = 58 with jt1 = 5 reaches absb row 236),
    jt/jt1 in 1..5, indm in 1..18.
    Column amounts are positive so every specmult ratio is finite.
    """
    import cupy as cp

    rs = cp.random.RandomState(seed)
    cell = (nc, nlay)
    lt = rs.randint(1, nlay, size=(nc,), dtype=cp.int32)            # laytrop in 1..nlay-1
    k = cp.arange(1, nlay + 1, dtype=cp.int32)[None, :]
    lower = k <= lt[:, None]

    def ri(lo, hi):
        return rs.randint(lo, hi + 1, size=cell, dtype=cp.int32)

    def uf(lo, hi, shape=cell):
        return rs.uniform(lo, hi, size=shape).astype(cp.float32)

    return dict(
        colamt_h2o=uf(1e-3, 1.0), colamt_co2=uf(1e-4, 1e-1), colamt_o3=uf(1e-4, 1.0),
        coldry=uf(1e18, 1e21),
        rfrate_h2oo3=uf(0.2, 5.0, (nc, nlay, 2)),
        jp=cp.where(lower, ri(1, 12), ri(13, 58)).astype(cp.int32),
        jt=ri(1, 5), jt1=ri(1, 5),
        indself=ri(1, 9), indfor=ri(1, 3), indminor=ri(1, 18),
        selffac=uf(0.0, 1.0), selffrac=uf(0.0, 1.0), forfac=uf(0.0, 1.0), forfrac=uf(0.0, 1.0),
        minorfrac=uf(0.0, 1.0), fac00=uf(0.0, 1.0), fac01=uf(0.0, 1.0), fac10=uf(0.0, 1.0), fac11=uf(0.0, 1.0),
        laytrop=lt,
    )


def run_chunk(ctx, inputs, nlay: int):
    """Launch the kernel on one chunk of columns. Returns (taug7, fracs7) views of the scratch."""
    nc = int(inputs["laytrop"].shape[0])
    tab, chi = tables(ctx)
    taug7, fracs7 = _scratch(ctx, nc, nlay)

    k = kernels.get(UNIT, KERNEL)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    args = [inputs[n] for n in INPUT_NAMES]
    args += [chi, tab["absa"], tab["absb"], tab["selfref"], tab["forref"], tab["fracrefa"],
             tab["fracrefb"], tab["ka_mco2"], tab["kb_mco2"], taug7, fracs7,
             np.int32(nc), np.int32(nlay)]
    block = kernels.BLOCK_COL
    grid = ((nc + block[0] - 1) // block[0], 1, 1)
    with scope:
        k(grid, block, tuple(args))
    return taug7, fracs7


def run(ctx, inputs, nlay: int = NLAY_DEFAULT):
    """Single launch on one chunk of already-built inputs (the taumol call site)."""
    return run_chunk(ctx, inputs, nlay)


def run_domain(ctx, ncol: int, nlay: int = NLAY_DEFAULT, chunk_cols: int = CHUNK_COLS, seed: int = 1):
    """Run every column of the domain in chunks, yielding (c0, c1, taug7, fracs7) per chunk.

    Random inputs stand in for the taumol state here (the standalone test); the caller of the real
    pipeline builds each chunk's inputs from setcoef/taumol and calls run() instead.
    """
    for ci, c0 in enumerate(range(0, ncol, chunk_cols)):
        c1 = min(c0 + chunk_cols, ncol)
        inputs = _random_inputs(c1 - c0, nlay, seed + ci)
        taug7, fracs7 = run_chunk(ctx, inputs, nlay)
        yield c0, c1, taug7, fracs7


if __name__ == "__main__":
    # Standalone check at the full CONUS parent size: 954 x 835 columns, LM = 60 layers.
    import time
    from types import SimpleNamespace

    import cupy as cp

    im, jm, lm = 954, 835, NLAY_DEFAULT
    ncol = im * jm
    ctx = SimpleNamespace(tables={}, stream=None)
    t0 = time.perf_counter()
    kernels.module(UNIT).compile()
    print(f"compiled {UNIT} (NVRTC) in {time.perf_counter() - t0:.2f} s", flush=True)
    nbad = {"taug7": 0, "fracs7": 0}
    lo = {"taug7": np.inf, "fracs7": np.inf}
    hi = {"taug7": -np.inf, "fracs7": -np.inf}
    ncov = 0
    t1 = time.perf_counter()
    for c0, c1, taug7, fracs7 in run_domain(ctx, ncol, lm, seed=1):
        cp.cuda.Stream.null.synchronize()
        for name, a in (("taug7", taug7), ("fracs7", fracs7)):
            nbad[name] += int((~cp.isfinite(a)).sum())
            lo[name] = min(lo[name], float(a.min()))
            hi[name] = max(hi[name], float(a.max()))
        ncov += c1 - c0
    t2 = time.perf_counter()
    print(f"columns covered {ncov} of {ncol} (IM={im}, JM={jm}, LM={lm})")
    for name in ("taug7", "fracs7"):
        print(f"{name}: shape per chunk (*,{lm},{NG07}) min {lo[name]:.4e} max {hi[name]:.4e} nonfinite {nbad[name]}")
        assert nbad[name] == 0, f"{name} has {nbad[name]} NaN/Inf values"
    print(f"domain run {t2 - t1:.2f} s")
