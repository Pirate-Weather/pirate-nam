"""phys-fer-init: FERRIER_INIT_hr, the once-per-run Ferrier-Aligo microphysics setup.

Fortran: phys/module_MP_FER_HIRES.F90:2391-2744 (FERRIER_INIT_hr), with MY_GROWTH_RATES_NMM_hr
(2746-2785), GPVS_hr (2791-2836), FPVS0 (2894-2911), FPVSX (2915-2971) and FPVSX0 (2974-2989);
nam.v4.2.11, 83df4e13c1. Called from module_SOLVER_GRID_COMP.F90:10997 with GSMDT = DT_MICRO =
NPRECIP*DT (10993), DELX/DELY (unused here: DX feeds only the commented RHgrd line).
Kernel: namgpu/kernels/phys_fer_init.cu (phys_fer_init_gpvs, _sdens, _scalars, _fill).
Growth table: phys-fer-growth-rates (namgpu/physics/fer_hires/phys_fer_growth_rates.py), run here.

Reads : ctx.cfg.nprecip, ctx.cfg.dt (the step), ctx.grid (im, jm, lm), ctx.st.F_ICE / F_RAIN /
        F_RIMEF (cold-start fill), the restart flag (run(ctx, restart=...) from PHYSICS_INITIALIZE;
        when not given, ctx.restart, default False), and the table file
        namgpu/data/ferrier/ETAMPNEW_DATA.expanded_rain (sha256 checked; the same bytes NAM's
        native-initialization ships; NAM's gfortran build reads it with -fconvert=big-endian,
        configure.nems.Linux.gnu:39).
Writes: ctx.tables["fer"], one dict, built once per (GSMDT, restart) and never rebuilt per step:
    dtph        float, GSMDT in s (50.0 at NPRECIP=2, DT=25)
    restart     bool, the restart flag the dict was built for
    tbpvs       float32 [NX=7501] device, Fortran TBPVS(1:NX)
    tbpvs0      float32 [7501]    device, Fortran TBPVS0(1:NX)
    tbpvs_state  float32 [7501]   device, Fortran TBPVS_STATE(1:NX) (restart-writer copy). A copy of
                                  tbpvs on a cold start (2730-2731 inside IF(.NOT.RESTART)); zeros on
                                  a restart, where the restart reader supplies it (not ported yet).
    tbpvs0_state float32 [7501]   device, Fortran TBPVS0_STATE(1:NX), same rule as tbpvs_state.
    growth      float32 [35]      device, MY_GROWTH_NMM(1:35): the array phys-fer-growth-rates
                                  writes (growth_unit.run); this unit does not compute it.
    sdens       float32 [951]     device, SDENS(50:1000), index k = I - 50
    massr, massi, ventr1, ventr2, accrr, vrain, rrate, venti1, venti2, accri, vsnowi
                float32 [951]     device, Fortran tables (50:1000), index k = M - 50.
                massr is MASSR (MASSRout in the Fortran), massi is MASSI (MASSIout).
    vel_rf      float32 [41, 8]   device, Fortran VEL_RF(2:9, 0:40): vel_rf[n, i-2]
    mp_restart  float32 [43]      device, MP_RESTART_STATE(1:43) in Fortran order. Written on a cold
                                  start only (2719); zeros on a restart (restart reader supplies it).
    sc          float32 [39]      device, scalar slots (SCALAR_SLOTS order)
    scalars     dict name -> float, one host readback at init (not per step)
Fill    : on a cold start, F_ICE = 0, F_RAIN = 0, F_RIMEF = 1 over the tile ITS:ITE, JTS:JTE,
          KTS:KTE (2481-2483); the tile is the domain, 1..IM, 1..JM, 1..LM (single process, the
          convention of bnd_bocoh_update: its=1, ite=im).
Launch  : five kernels on ctx.stream (or the null stream), no host sync except the one init readback.

Host file I/O: the table parse is numpy big-endian (Fortran unformatted records, 4-byte markers),
the same layout WRF's build reads (ArWen parses it the same way). Whether NAM's own build reads the
records big-endian is not verified (no -convert flag in the clone's makefile for this build), so the
arrays are checked for finite, positive, monotone-where-expected values before use.
"""
from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import numpy as np

from namgpu import kernels
from namgpu.physics.fer_hires import phys_fer_growth_rates as growth_unit

UNIT = "phys_fer_init"
TABLE_NAME = "ETAMPNEW_DATA.expanded_rain"
TABLE_SHA256 = "28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13"

MDMIN, MDMAX = 50, 1000
NTAB = MDMAX - MDMIN + 1          # 951: MDRmin:MDRmax == MDImin:MDImax
NX = 7501                         # NX (module, line 148)
NRIME = 40                        # Nrime
NMY = 35                          # MY_T1:MY_T2
NMP = 43                          # MP_RESTART_STATE words

#: The twelve records FERRIER_INIT_hr READs (2972-2983), in order.
RECORDS = (("VENTR1", NTAB), ("VENTR2", NTAB), ("ACCRR", NTAB),
           ("MASSR", NTAB), ("VRAIN", NTAB), ("RRATE", NTAB),
           ("VENTI1", NTAB), ("VENTI2", NTAB), ("ACCRI", NTAB),
           ("MASSI", NTAB), ("VSNOWI", NTAB), ("VEL_RF", 8 * (NRIME + 1)))
#: The 35-word MY_600 record at the end of the file is validated and not read (the READ is
#: commented out at 2984).
TRAILING_WORDS = 35

#: Scalar slots, same order as the enum in kernels/phys_fer_init.cu.
SCALAR_SLOTS = (
    "C1XPVS", "C2XPVS", "C1XPVS0", "C2XPVS0", "PI", "ABFR", "CBFR", "CIACW", "CIACR",
    "CRACW", "BRAUT", "RQR_DRMIN", "RQR_DRMAX", "RR_DRMIN", "RR_DR1", "RR_DR2", "RR_DR3",
    "RR_DR4", "RR_DR5", "RR_DRMAX", "C_NR", "CRAIN", "C_N0R0", "CN0R0", "CN0R_DMRMIN",
    "CN0R_DMRMAX", "ESW0", "RFMAX", "RFMX1", "BETA6", "ARAUT", "QAUT0", "ARCW", "RH_NGC",
    "RQHAIL", "BVHAIL", "AVHAIL", "RH_NGT", "DTPH",
)
NSC = len(SCALAR_SLOTS)           # 39


class FerrierTableError(ValueError):
    """The Ferrier table is missing, modified, or not the shape FERRIER_INIT_hr reads."""


def table_path() -> Path:
    """The packaged copy: namgpu/data/ferrier/ETAMPNEW_DATA.expanded_rain."""
    return Path(__file__).resolve().parents[2] / "data" / "ferrier" / TABLE_NAME


def load_host_tables(path: Path | None = None) -> dict[str, np.ndarray]:
    """Parse the big-endian Fortran unformatted records into float32 arrays (host, once)."""
    path = path or table_path()
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != TABLE_SHA256:
        raise FerrierTableError(f"{path} is not the pinned table (sha256 mismatch)")
    out: dict[str, np.ndarray] = {}
    pos = 0

    def record(name: str, words: int) -> np.ndarray:
        nonlocal pos
        if pos + 4 > len(blob):
            raise FerrierTableError(f"{TABLE_NAME} ends before record {name}")
        head = int.from_bytes(blob[pos:pos + 4], "big")
        if head != 4 * words:
            raise FerrierTableError(
                f"{TABLE_NAME} record {name} holds {head} bytes; FERRIER_INIT_hr reads {4 * words}")
        body = blob[pos + 4:pos + 4 + head]
        tail = int.from_bytes(blob[pos + 4 + head:pos + 8 + head], "big")
        if len(body) != head or tail != head:
            raise FerrierTableError(f"{TABLE_NAME} record {name} is truncated or markers disagree")
        pos += 8 + head
        return np.frombuffer(body, dtype=">f4").astype(np.float32)

    for name, words in RECORDS:
        out[name] = record(name, words)
    record("MY_600 (unread)", TRAILING_WORDS)
    if pos != len(blob):
        raise FerrierTableError(f"{TABLE_NAME} has {len(blob) - pos} bytes after its last record")
    for name in ("MASSR", "MASSI", "RRATE"):
        arr = out[name]
        if not (np.all(np.isfinite(arr)) and np.all(arr > 0.0)):
            raise FerrierTableError(f"{TABLE_NAME} record {name} has non-positive or non-finite words")
    return out


def _build(ctx, dtph: np.float32, restart: bool) -> dict:
    import cupy as cp

    # MY_GROWTH_NMM from the growth-rates unit (same DTPH); bound as fer["growth"] below.
    growth = growth_unit.run(ctx, dtph)
    host = load_host_tables()
    dev = {}
    for name in ("VENTR1", "VENTR2", "ACCRR", "MASSR", "VRAIN", "RRATE",
                 "VENTI1", "VENTI2", "ACCRI", "MASSI", "VSNOWI"):
        dev[name.lower()] = cp.asarray(host[name])
    # VEL_RF(2:9, 0:40), Fortran column-major: flat index (i-2) + 8*n -> shape (41, 8) in C order.
    dev["vel_rf"] = cp.asarray(host["VEL_RF"].reshape((NRIME + 1, 8)))

    tbpvs = cp.zeros(NX, dtype=cp.float32)
    tbpvs0 = cp.zeros(NX, dtype=cp.float32)
    tbpvs_state = cp.zeros(NX, dtype=cp.float32)
    tbpvs0_state = cp.zeros(NX, dtype=cp.float32)
    sdens = cp.zeros(NTAB, dtype=cp.float32)
    sc = cp.zeros(NSC, dtype=cp.float32)
    mp = cp.zeros(NMP, dtype=cp.float32)

    k_gpvs = kernels.get(UNIT, "phys_fer_init_gpvs")
    k_sdens = kernels.get(UNIT, "phys_fer_init_sdens")
    k_scal = kernels.get(UNIT, "phys_fer_init_scalars")
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        k_gpvs(((NX + 127) // 128,), (128,), (tbpvs, tbpvs0))
        k_sdens(((NTAB + 127) // 128,), (128,), (dev["massi"], sdens))
        k_scal((1,), (1,), (dev["rrate"], dev["massr"], tbpvs0, growth, sc, mp, dtph,
                            np.int32(0 if restart else 1)))
        if not restart:
            # Fortran 2730-2731 (inside IF(.NOT.RESTART)): the restart-writer copies of the tables.
            cp.copyto(tbpvs_state, tbpvs)
            cp.copyto(tbpvs0_state, tbpvs0)

    vals = cp.asnumpy(sc)                 # one readback, at init only
    scalars = {name: float(v) for name, v in zip(SCALAR_SLOTS, vals)}
    fer = {
        "dtph": float(dtph), "restart": bool(restart),
        "tbpvs": tbpvs, "tbpvs0": tbpvs0, "tbpvs_state": tbpvs_state, "tbpvs0_state": tbpvs0_state,
        "growth": growth, "sdens": sdens,
        "sc": sc, "mp_restart": mp, "scalars": scalars,
    }
    fer.update(dev)
    return fer


def run(ctx, *, restart: bool | None = None):
    """Build the Ferrier tables once per (GSMDT, restart), then fill F_ICE/F_RAIN/F_RIMEF on a cold start.

    restart: Fortran RESTART as PHYSICS_INITIALIZE passes it (int_state%RESTART). None means
    ctx.restart, default False.
    """
    import cupy as cp

    if restart is None:
        restart = bool(getattr(ctx, "restart", False))
    restart = bool(restart)
    dtph = np.float32(np.float32(ctx.cfg.nprecip) * np.float32(ctx.cfg.dt))   # DT_MICRO = NPRECIP*DT
    tables = ctx.tables
    fer = tables.get("fer")
    if fer is None or fer["dtph"] != float(dtph) or fer["restart"] != restart:
        tables["fer"] = fer = _build(ctx, dtph, restart)

    kept = ctx.tables.get("native_warm_kept") or ()
    if not restart and not {"F_ICE", "F_RAIN", "F_RIMEF"} <= set(kept):
        # (a native NAM start read F_ICE/F_RAIN/F_RIMEF from the analysis, as the operational forecast read them from
        # its restart-format input: no cold fill over them. Breakage it fixes: the fill set them to 0/0/1, so every
        # analysis drop started as cloud water: rain 0 vs 134, snow 0 vs 193 at 0 h; first-hour grid-scale rain
        # 69-91 % of the NAM's, NAM start study 2026-11-05.)
        st = ctx.st
        im, jm, lm = int(ctx.grid.im), int(ctx.grid.jm), int(ctx.grid.lm)
        for name in ("F_ICE", "F_RAIN", "F_RIMEF"):
            if tuple(getattr(st, name).shape) != (lm, jm, im):
                raise ValueError(f"{name} shape {getattr(st, name).shape} is not (lm, jm, im) = {(lm, jm, im)}")
        # Tile = domain (its=1, ite=im, jts=1, jte=jm, kts=1, kte=lm), as FERRIER_INIT_hr is called
        # with the single-process tile bounds (module_SOLVER_GRID_COMP.F90:11006-11007).
        k_fill = kernels.get(UNIT, "phys_fer_init_fill")
        n = lm * jm * im
        scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
        with scope:
            k_fill(((n + 255) // 256,), (256,),
                   (st.F_ICE, st.F_RAIN, st.F_RIMEF, np.int32(im), np.int32(jm),
                    np.int32(1), np.int32(im), np.int32(1), np.int32(jm), np.int32(1), np.int32(lm)))
    return fer
