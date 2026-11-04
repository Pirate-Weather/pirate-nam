"""phys-fer-col-rain-sedim: rain evaporation, accretion, freezing, fall-out and column updates.

Ported from phys/module_MP_FER_HIRES.F90:1588-2159 (NAM nam.v4.2.11, 83df4e13c1), the tail of the
level loop of SUBROUTINE EGCP01COLUMN_hr. Kernel: namgpu/kernels/phys_fer_col_rain_sedim.cu
(phys_fer_col_rain_sedim), one launch per chunk of levels, grid_cols(IM, JM), no host sync.

This unit is a column recurrence: its level loop reads and writes the column carry (ARAIN, ASNOW,
VRAIN1, VSNOW1, TIMLT, STRAT, DRZL, and the carry2 locals), so the levels of one column must be
processed in order. The chunked launch (lev0, nchunk) gives the same result as one launch over all
levels, because the carry is read at the start of each launch and written at the end.

Reads (state names; per-level arrays are [nchunk, JM, IM], chunk-local level index 0 = global level lev0;
level 0 is the model top):
    FCR_IN_<NAME> for NAME in INPUT_NAMES     per-level values from the upstream units of this column loop.
                                              FCR_IN_INDEXRMIN is the per-level INDEXRmin after the ice
                                              section, i.e. phys-fer-col-ice-props output slot 25
                                              (FER_OUT_INDEXRMIN in ctx.tables["fer_col_ice_out"]). The
                                              satcheck FER_SC_INDEXRMIN is not a per-level producer.
    FER_SC_CLEAR  int32 [LM, JM, IM]          CLEAR after the satcheck section (1 = CYCLE big_loop), read
                                              chunk-sliced by lev0.
    FER_LSFC      int32 [JM, IM]              LSFC per column (phys-fer-col-setup)
    FER_COL_CARRY float32 [9, JM, IM]         carry rows 0..8 (phys-fer-col-setup); rows 0,1,2,3,4,6,7 are
                                              read and written here, rows 5 and 8 are left as they are
    ctx.tables["fer"]                         massr, vrain, accrr, ventr1, ventr2, massi (index m - 50)
                                              and scalars (ABFR, CBFR, CIACR, CRACW, RQR_DRMIN, RQR_DRMAX,
                                              CRAIN, CN0R0, RFMX1, DTPH) from phys-fer-init
Writes (state names, [nchunk, JM, IM] per level):
    FCR_OUT_<NAME> for T_col, Q_col, WC_col, QI_col, QR_col, QW_col, RimeF_col, NR_col, NS_col, DBZ_col
    and the 1-D process arrays pcond1d ... RFlag1d (same names as phys-fer-col-setup). A CLEAR level writes
    none of them (Fortran CYCLE big_loop leaves them unchanged).
    FCR_OUT_ARAIN_COL, FCR_OUT_ASNOW_COL      ARAIN and ASNOW on entry to each level, written on every level:
                                              the ice unit's ARAIN_COL / ASNOW_COL per-level inputs.
    FER_COL_CARRY2 float32 [6, JM, IM]        carry2: 0 Nrain, 1 Zrain, 2 Nsnow, 3 Zsnow, 4 QInew, 5 ASNOWnew.
                                              The first chunk of each column (lev0 == 0) starts it at 0.
    FER_COL_CARRY                             updated in place (rows above)

Status: NOT a finished port. The per-level inputs ARAIN_COL/ASNOW_COL (from this unit) and INDEXRmin (from
the ice unit) feed each other level by level, so the level loops of the upstream units must be interleaved
with this unit in one fused driver. That driver is not built in the tree yet (see STATUS.md). Until it is,
the unit is verified only on synthetic inputs (selftest below).

VRAM: 49 float inputs + 1 int CLEAR + 34 outputs = 84 per-level arrays. At full size (835 x 954 x 4 B =
3.19 MB each) that is 268 MB per level, 16.1 GB for all 60 levels; a chunk of 20 levels needs 5.4 GB.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_fer_col_rain_sedim"
KERNEL = "phys_fer_col_rain_sedim"
BLOCK = 128

# Per-level inputs, in the order of the kernel's in_* arguments. Names are the Fortran names.
INPUT_NAMES = (
    "THICK", "RHO", "RRHO", "TK", "TK2", "TC", "TCC", "WV", "QSW", "QW", "QI", "QR",
    "PCOND", "PIDEP", "PIEVP", "PICND", "PIMLT", "PIACW", "PIACWI", "PRAUT", "PRACW", "PREVP",
    "PIACR", "PINIT", "PRLOSS", "QTRAIN", "TOT_RAIN", "VRAIN2", "INDEXR", "INDEXRMIN", "N0R",
    "GAMMAR", "DIFFUS", "DYNVIS", "THERM_COND", "QLICE", "NLICE", "NSMICE", "VSNOW", "VCI",
    "FLIMASS", "INDEXS", "TOT_ICE", "RIMEF1", "RCW", "BLDTRH", "DWVR", "ICE_LOGICAL",
    "RAIN_LOGICAL",
)

# Per-level outputs, in the order of the kernel's out_* / 1-D arguments.
OUTPUT_NAMES = (
    "T_col", "Q_col", "WC_col", "QI_col", "QR_col", "QW_col", "RimeF_col", "NR_col", "NS_col",
    "DBZ_col", "pcond1d", "pevap1d", "pidep1d", "pisub1d", "piacw1d", "piacwi1d", "piacwr1d",
    "piacr1d", "picnd1d", "pievp1d", "pimlt1d", "praut1d", "pracw1d", "prevp1d", "vsnow1d",
    "vrain11d", "vrain21d", "vci1d", "NSmICE1d", "INDEXS1d", "INDEXR1d", "RFlag1d", "ARAIN_COL", "ASNOW_COL",
)

CARRY_NAME = "FER_COL_CARRY"
CARRY2_NAME = "FER_COL_CARRY2"
NCARRY = 9
NCARRY2 = 6
LSFC_NAME = "FER_LSFC"
CLEAR_NAME = "FER_SC_CLEAR"
# Scalars in the kernel's argument order (fer init scalars): DTPH, CIACR, CRACW, ABFR, CBFR, RFMX1,
# CRAIN, CN0R0, RQR_DRMIN, RQR_DRMAX.
SCALARS = ("DTPH", "CIACR", "CRACW", "ABFR", "CBFR", "RFMX1", "CRAIN", "CN0R0", "RQR_DRMIN", "RQR_DRMAX")


def in_name(name: str) -> str:
    return f"FCR_IN_{name}"


def out_name(name: str) -> str:
    return f"FCR_OUT_{name}"


def run(ctx, lev0: int = 0, nchunk: int | None = None):
    """Launch the unit for levels lev0 .. lev0+nchunk-1 of every column (default: all levels).

    Input and output arrays are the chunk-local [nchunk, JM, IM] fields named FCR_IN_* and FCR_OUT_*.
    The carry arrays are the full-column FER_COL_CARRY/FER_COL_CARRY2 and persist across launches.
    """
    import cupy as cp

    st = ctx.st
    g = ctx.grid
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if nchunk is None:
        nchunk = lm - lev0
    shape3 = (nchunk, jm, im)

    fer = ctx.tables.get("fer")
    if fer is None:
        raise RuntimeError("phys-fer-col-rain-sedim: ctx.tables['fer'] is missing (phys-fer-init first)")
    sc = fer["scalars"]
    scal = [np.float32(sc[k]) for k in SCALARS]

    lsfc = st.fields[LSFC_NAME]
    carry = st.fields[CARRY_NAME]
    carry2 = st.add(CARRY2_NAME, (NCARRY2, jm, im))
    ins = [st.fields[in_name(n)] for n in INPUT_NAMES]
    if CLEAR_NAME not in st.fields:
        raise RuntimeError("phys-fer-col-rain-sedim: FER_SC_CLEAR is missing (phys-fer-col-satcheck first)")
    clr = st.fields[CLEAR_NAME][lev0:lev0 + nchunk]
    if tuple(clr.shape) != shape3:
        raise RuntimeError(f"phys-fer-col-rain-sedim: FER_SC_CLEAR slice {clr.shape} != {shape3}")
    for a in ins:
        if tuple(a.shape) != shape3:
            raise RuntimeError(f"phys-fer-col-rain-sedim: input shape {a.shape} != {shape3}")
    outs = [st.add(out_name(n), shape3) for n in OUTPUT_NAMES]

    k = kernels.get(UNIT, KERNEL)
    grid = ((im * jm + BLOCK - 1) // BLOCK, 1, 1)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k(grid, (BLOCK, 1, 1),
          (np.int32(im), np.int32(jm), np.int32(lm),
           *scal,
           lsfc,
           np.int32(lev0), np.int32(nchunk),
           *ins, clr,
           fer["massr"], fer["vrain"], fer["accrr"], fer["ventr1"], fer["ventr2"], fer["massi"],
           carry, carry2,
           *outs))
    return dict(zip(OUTPUT_NAMES, outs))


def plausible_inputs(st, im: int, jm: int, lm: int, lev0: int, nchunk: int, seed: int):
    """Fill the FCR_IN_* chunk arrays with physically plausible values (device-side, no host loop).

    Used by the self-test only. Ranges are chosen to stay inside the Fortran's valid domain; they are
    not measured atmospheric profiles, so the test shows finiteness, not skill.
    """
    import cupy as cp

    rng = cp.random.default_rng(seed + 1000 * lev0)
    shape = (nchunk, jm, im)

    def u():
        return rng.random(shape, dtype=cp.float32)

    def set_(name, arr):
        st.fields[in_name(name)] = cp.asarray(arr, dtype=cp.float32)

    tk = 230.0 + 60.0 * u()
    rho = 0.6 + 0.5 * u()
    tc = tk - 273.15
    set_("THICK", 50.0 + 400.0 * u())
    set_("RHO", rho)
    set_("RRHO", 1.0 / rho)
    set_("TK", tk)
    set_("TK2", 1.0 / (tk * tk))
    set_("TC", tc)
    set_("TCC", tc)
    set_("WV", 0.001 + 0.015 * u())
    set_("QSW", 0.001 + 0.02 * u())
    set_("QW", 1.0E-4 * u())
    set_("QI", 1.0E-4 * u())
    set_("QR", 1.0E-4 * u())
    for n in ("PCOND", "PIDEP", "DWVR"):
        set_(n, 2.0E-5 * (u() - 0.5))
    for n in ("PIEVP", "PICND", "PIACW", "PIACWI", "PRAUT", "PRACW", "PIACR", "PINIT", "PIMLT"):
        set_(n, 1.0E-6 * u())
    set_("PREVP", -1.0E-6 * u())
    set_("PRLOSS", -1.0E-4 * u())
    set_("QTRAIN", 1.0E-4 * u())
    set_("TOT_RAIN", 5.0E-3 * u())
    set_("VRAIN2", 1.0 + 9.0 * u())
    set_("INDEXR", cp.floor(50.0 + 950.0 * u()))
    set_("INDEXRMIN", cp.floor(50.0 + 450.0 * u()))
    set_("N0R", 1.0E6 + 1.0E7 * u())
    set_("GAMMAR", 1.0 + 0.2 * u())
    set_("DIFFUS", 2.0E-5 + 1.0E-5 * u())
    set_("DYNVIS", 1.7E-5 + 0.1E-5 * u())
    set_("THERM_COND", 0.024 + 0.002 * u())
    set_("QLICE", 1.0E-4 * u())
    set_("NLICE", 1.0E3 + 1.0E4 * u())
    set_("NSMICE", 1.0E3 * u())
    set_("VSNOW", 0.5 + u())
    set_("VCI", 0.3 + 0.5 * u())
    set_("FLIMASS", u())
    set_("INDEXS", cp.floor(50.0 + 950.0 * u()))
    set_("TOT_ICE", 1.0E-3 * u())
    set_("RIMEF1", 1.0 + 4.0 * u())
    set_("RCW", 1.0E-5 + 1.0E-5 * u())
    set_("BLDTRH", 1.0 + 9.0 * u())
    set_("ICE_LOGICAL", (u() > 0.2).astype(cp.float32))
    set_("RAIN_LOGICAL", (u() > 0.1).astype(cp.float32))


def _selftest():
    """Full-size selftest: 954 x 835 x 60 levels, chunks of 20 levels, real FER tables, two full passes.

    One check per review defect (no sweeps):
      1. every output is finite;
      2. CLEAR levels (random 30 percent; RHO = RRHO = 0 there, as the satcheck writes them) write no output:
         the outputs are pre-filled with a sentinel and must keep it on CLEAR levels;
      3. carry2 is per step: pass 2 restarts the column carry at zero, carry2 is left as pass 1 ended, and
         level 0 of every column must reproduce pass 1.
    """
    import types

    import cupy as cp

    from namgpu import config, grid as gridmod
    from namgpu.physics.fer_hires import phys_fer_init

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)

    class _Stub:
        def __init__(self):
            self.fields = {}

        def add(self, name, shape, dtype="float32"):
            if name not in self.fields:
                self.fields[name] = cp.zeros(shape, dtype=dtype)
            return self.fields[name]

    st = _Stub()
    ctx = types.SimpleNamespace(cfg=cfg, grid=g, st=st, stream=None, tables={}, dt=cfg.dt)
    dtph = np.float32(np.float32(cfg.nprecip) * np.float32(cfg.dt))
    ctx.tables["fer"] = phys_fer_init._build(ctx, dtph, False)

    lsfc = st.add(LSFC_NAME, (jm, im), "int32")
    lsfc[...] = cp.asarray(np.full((jm, im), lm, dtype=np.int32))
    carry = st.add(CARRY_NAME, (NCARRY, jm, im))     # zero: no ARAIN/ASNOW/VRAIN1/VSNOW1/TIMLT, STRAT/DRZL off
    clear_full = st.add(CLEAR_NAME, (lm, jm, im), "int32")
    clear_full[...] = (cp.random.default_rng(11).random((lm, jm, im), dtype=cp.float32) < 0.3).astype(cp.int32)
    check_l0 = ("T_col", "NR_col", "DBZ_col", "QI_col", "WC_col")

    chunk = 20
    sentinel = np.float32(-7.0)
    ref = {}
    n_nonfinite = n_clear_written = n_l0_diff = 0
    for pass_ in (1, 2):
        if pass_ == 2:
            carry[...] = 0.0      # the column carry returns to its step-start state; carry2 does not
        for lev0 in range(0, lm, chunk):
            nchunk = min(chunk, lm - lev0)
            plausible_inputs(st, im, jm, lm, lev0, nchunk, seed=7)
            clr = clear_full[lev0:lev0 + nchunk]
            for n in ("RHO", "RRHO"):
                st.fields[in_name(n)] *= (1 - clr).astype(cp.float32)
            for n in OUTPUT_NAMES:
                st.add(out_name(n), (nchunk, jm, im)).fill(sentinel)
            outs = run(ctx, lev0=lev0, nchunk=nchunk)
            cp.cuda.Stream.null.synchronize()
            clr_b = clr.astype(bool)
            for name, a in outs.items():
                if not bool(cp.isfinite(a).all()):
                    n_nonfinite += 1
                    print(f"pass {pass_} lev0={lev0}: NaN/Inf in {name}")
                if name not in ("ARAIN_COL", "ASNOW_COL"):
                    n_clear_written += int(cp.count_nonzero(cp.where(clr_b, a != sentinel, False)))
            if lev0 == 0:
                if pass_ == 1:
                    ref = {nm: outs[nm][0].copy() for nm in check_l0}
                else:
                    for nm in check_l0:
                        n_l0_diff += int(cp.count_nonzero(outs[nm][0] != ref[nm]))
    print(f"phys-fer-col-rain-sedim selftest: {lm} levels, {im}x{jm}, non-finite arrays = {n_nonfinite}, "
          f"CLEAR-level writes = {n_clear_written}, pass-2 level-0 differences = {n_l0_diff}")
    return n_nonfinite + n_clear_written + n_l0_diff


if __name__ == "__main__":
    raise SystemExit(1 if _selftest() else 0)
