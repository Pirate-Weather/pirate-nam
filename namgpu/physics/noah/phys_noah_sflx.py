"""phys-noah-sflx: SFLX, Noah LSM column physics with the NCEP v4 changes (NAM v4.2.11).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:1550-2278. NOAA US
government work; the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_sflx.cu (phys_noah_sflx, one thread per (i, j) column, one launch).
Mode copy-arwen: the kernel body is the ArWen Noah kernel (Apache-2.0, header kept in the .cu).

The kernel runs the per-column driver prep and the SFLX subtree (PENMAN, CANRES, NOPAC, SNOPAC,
SRT/SSTEP/SHFLX/HSTEP, TDFCND/WDFCND, CSNOW, SNFRAC, ALCALC, SNOWPACK, REDPRM) in one launch. No
host sync, no per-step allocation beyond first-call scratch, everything on the current stream.

Tables: the constant kernel inputs (vegetation table (lucats, 15), soil table (slcats, 10), general
table (16,), soil-layer thicknesses (4,)) are passed in. run() without ctx builds plausible synthetic
tables (the NAM NCEP fix tables are not available yet, see STATUS.md).

run(ctx): binds the NMMB state-table fields by their CONTRACT names where the state table has them
(SMC, STC, SH2O, CMC, TSKIN, SNO, SNOWC, ALBEDO, ALBASE, EPSR, MXSNAL); every other kernel field is
scratch allocated once through ctx.st.add. Inputs with no state-table field raise, naming them.

Without ctx: full-size 954 x 835 random but plausible state, one launch, asserts every float output is
finite and returns the output dict.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_sflx"
KERNEL = "phys_noah_sflx"
BLOCK = (128, 1, 1)
NSOIL = 4
NVEGC = 15
NSOILC = 10
NGEN = 16
ISURBAN = 13
ISICE = 15

# Ordered 2-D float32 fields, exactly the kernel signature order from ivgtyp/isltyp.
_F2D = ("psfc", "sfcprs", "sfctmp", "qv1", "qgh", "dz8w1", "glw",
        "swdown", "rainbl", "sr", "chs", "cqs2", "chs2", "rib",
        "vegfra", "shdmin", "shdmax", "tmn", "xland", "xice", "snoalb",
        "embck",
        "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc",
        "canwat", "snow", "snowc", "snowh",
        "albedo", "albbck", "emiss", "znt", "z0", "snotime", "lai",
        "smstav", "smstot",
        "sfcrunoff", "udrunoff", "acsnow", "acsnom", "snopcx", "potevp",
        "noahres", "reslin", "chklowq")
# Ordered 3-D float32 fields [NSOIL, JM, IM].
_F3D = ("smois", "tslb", "sh2o", "smcrel")
# Outputs the kernel writes (float). chs2 is read-write, the rest are written.
_OUT_FLOAT = ("chs2", "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc", "canwat", "snow", "snowc",
              "snowh", "albedo", "albbck", "emiss", "znt", "z0", "snotime", "lai", "smstav",
              "smstot", "sfcrunoff", "udrunoff", "acsnow", "acsnom", "snopcx", "potevp",
              "noahres", "reslin", "chklowq", "smois", "tslb", "sh2o", "smcrel")

# kernel field name -> NMMB state-table name (parm/nam_solver_state.txt)
_STATE_NAME = {"smois": "SMC", "tslb": "STC", "sh2o": "SH2O", "canwat": "CMC",
               "tsk": "TSKIN", "snow": "SNO", "snowc": "SNOWC", "albedo": "ALBEDO",
               "albbck": "ALBASE", "emiss": "EPSR", "snoalb": "MXSNAL"}

_tables: dict = {}


def _grid(im: int, jm: int):
    return ((im * jm + BLOCK[0] - 1) // BLOCK[0],)


def synthetic_tables(lucats: int = 20, slcats: int = 9, seed: int = 0):
    """Plausible NOAH tables (veg (lucats,15), soil (slcats,10), gen (16,), dzs (4,)) for the smoke run."""
    import cupy as cp

    rng = np.random.default_rng(seed)
    veg = np.zeros((lucats, NVEGC), np.float32)
    veg[:, 0] = 3.0                                              # nroot
    veg[:, 1] = rng.uniform(100.0, 250.0, lucats)              # rsmin
    veg[:, 2] = rng.uniform(30.0, 100.0, lucats)               # rgl
    veg[:, 3] = rng.uniform(36.0, 45.0, lucats)                # hs
    veg[:, 4] = rng.uniform(0.02, 0.08, lucats)                # snup
    veg[:, 5] = rng.uniform(0.5, 1.0, lucats)                  # laimin
    veg[:, 6] = rng.uniform(3.0, 6.0, lucats)                  # laimax
    veg[:, 7] = 0.90                                           # emissmin
    veg[:, 8] = 0.97                                           # emissmax
    veg[:, 9] = rng.uniform(0.10, 0.20, lucats)                # albedomin
    veg[:, 10] = rng.uniform(0.20, 0.30, lucats)               # albedomax
    veg[:, 11] = rng.uniform(0.01, 0.10, lucats)               # z0min
    veg[:, 12] = rng.uniform(0.5, 1.5, lucats)                 # z0max
    veg[:, 13] = rng.uniform(0.3, 0.9, lucats)                 # shdtbl
    veg[:, 14] = rng.uniform(0.3, 0.5, lucats)                 # maxalb
    soil = np.zeros((slcats, NSOILC), np.float32)
    soil[:, 0] = rng.uniform(2.0, 12.0, slcats)                # bexp
    soil[:, 1] = rng.uniform(0.02, 0.10, slcats)               # smcdry
    soil[:, 2] = rng.uniform(0.10, 0.50, slcats)               # f1
    soil[:, 3] = rng.uniform(0.30, 0.50, slcats)               # smcmax
    soil[:, 4] = rng.uniform(0.20, 0.30, slcats)               # smcref
    soil[:, 5] = rng.uniform(0.05, 0.80, slcats)               # psisat
    soil[:, 6] = rng.uniform(1e-6, 1e-4, slcats)               # dksat
    soil[:, 7] = rng.uniform(1e-6, 1e-4, slcats)               # dwsat
    soil[:, 8] = rng.uniform(0.01, 0.10, slcats)               # smcwlt
    soil[:, 9] = rng.uniform(0.10, 0.90, slcats)               # quartz
    gen = np.zeros(NGEN, np.float32)
    gen[0] = 298.0      # TOPT
    gen[1] = 0.5e-3    # CMCMAX
    gen[2] = 0.5       # CFACTR
    gen[3] = 5000.0    # RSMAX
    gen[4] = -2.0      # SBETA
    gen[5] = 2.0       # FXEXP
    gen[6] = 2.0e6     # CSOIL
    gen[7] = 2.6       # SALP
    gen[8] = 2.0e-6    # REFDK
    gen[9] = 3.0       # REFKDT
    gen[10] = 0.15     # FRZK
    gen[11] = -8.0     # ZBOT
    gen[12] = 0.5      # LVCOEF
    gen[13] = 0.1      # SLOPE
    gen[14] = 16.0     # BARE
    gen[15] = 14.0     # NATURAL
    dzs = np.array([0.1, 0.3, 0.6, 1.0], np.float32)
    return (cp.asarray(veg.ravel()), cp.asarray(soil.ravel()), cp.asarray(gen),
            cp.asarray(dzs), lucats, slcats)


def launch(dev: dict, tables, dt: float, *, isurban: int = ISURBAN, isice: int = ISICE,
           xice_threshold: float = 0.5, itimestep: int = 2, frpcpn: int = 0,
           usemonalb: int = 0, rdlai2d: int = 0, opt_thcnd: int = 1) -> None:
    """One launch of phys_noah_sflx over the full (JM, IM) column set. dev maps names to CuPy arrays."""
    import cupy as cp

    vegtbl, soiltbl, genp, dzs4, lucats, slcats = tables
    ny, nx = dev["tsk"].shape
    args = [dev["ivgtyp"], dev["isltyp"]]
    for name in _F2D:
        a = dev[name]
        if a.shape != (ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected ({ny},{nx}) float32, got {a.shape} {a.dtype}")
        args.append(a)
    for name in _F3D:
        a = dev[name]
        if a.shape != (NSOIL, ny, nx) or a.dtype != cp.float32:
            raise ValueError(f"{name}: expected ({NSOIL},{ny},{nx}) float32, got {a.shape} {a.dtype}")
        args.append(a)
    args.append(dev["ebal"])
    args += [vegtbl, soiltbl, genp, dzs4,
             np.float32(dt), np.int32(lucats), np.int32(slcats),
             np.int32(isurban), np.int32(isice), np.float32(xice_threshold),
             np.int32(itimestep), np.int32(frpcpn), np.int32(usemonalb),
             np.int32(rdlai2d), np.int32(opt_thcnd), np.int32(ny), np.int32(nx)]
    kern = kernels.get(UNIT, KERNEL)
    kern(_grid(nx, ny), BLOCK, tuple(args))


def _random_state(ny: int, nx: int, lucats: int, slcats: int, seed: int = 1):
    import cupy as cp

    rng = np.random.default_rng(seed)

    def f2(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(ny, nx)).astype(np.float32))

    def f3(lo, hi):
        return cp.asarray(rng.uniform(lo, hi, size=(NSOIL, ny, nx)).astype(np.float32))

    dev = {
        "ivgtyp": cp.asarray(rng.choice([v for v in range(1, lucats + 1) if v != ISICE],
                                        size=(ny, nx)).astype(np.int32)),
        "isltyp": cp.asarray(rng.integers(1, slcats + 1, size=(ny, nx)).astype(np.int32)),
        "ebal": cp.zeros((ny, nx), dtype=cp.int32),
        "psfc": f2(80000.0, 101000.0), "sfcprs": f2(80000.0, 101000.0),
        "sfctmp": f2(250.0, 310.0), "qv1": f2(0.001, 0.02), "qgh": f2(0.001, 0.02),
        "dz8w1": f2(20.0, 200.0), "glw": f2(250.0, 450.0), "swdown": f2(0.0, 1000.0),
        "rainbl": f2(0.0, 1e-4), "sr": f2(0.0, 1.0), "chs": f2(0.001, 0.02),
        "cqs2": f2(0.001, 0.02), "chs2": cp.zeros((ny, nx), cp.float32), "rib": f2(-0.1, 0.1),
        "vegfra": f2(0.0, 100.0), "shdmin": f2(0.0, 20.0), "shdmax": f2(50.0, 100.0),
        "tmn": f2(270.0, 290.0), "xland": cp.ones((ny, nx), cp.float32),
        "xice": cp.zeros((ny, nx), cp.float32), "snoalb": f2(0.6, 0.9),
        "embck": cp.full((ny, nx), 0.95, cp.float32), "tsk": f2(250.0, 310.0),
        "hfx": cp.zeros((ny, nx), cp.float32), "qfx": cp.zeros((ny, nx), cp.float32),
        "lh": cp.zeros((ny, nx), cp.float32), "grdflx": cp.zeros((ny, nx), cp.float32),
        "qsfc": cp.zeros((ny, nx), cp.float32), "canwat": f2(0.0, 1e-3),
        "snow": f2(0.0, 100.0), "snowc": f2(0.0, 1.0), "snowh": f2(0.0, 0.5),
        "albedo": f2(0.1, 0.4), "albbck": f2(0.1, 0.4), "emiss": f2(0.9, 0.98),
        "znt": f2(0.01, 0.5), "z0": f2(0.01, 0.5), "snotime": cp.zeros((ny, nx), cp.float32),
        "lai": f2(0.0, 5.0), "smstav": f2(0.0, 1.0), "smstot": f2(0.0, 1.0),
        "sfcrunoff": cp.zeros((ny, nx), cp.float32), "udrunoff": cp.zeros((ny, nx), cp.float32),
        "acsnow": cp.zeros((ny, nx), cp.float32), "acsnom": cp.zeros((ny, nx), cp.float32),
        "snopcx": cp.zeros((ny, nx), cp.float32), "potevp": cp.zeros((ny, nx), cp.float32),
        "noahres": cp.zeros((ny, nx), cp.float32), "reslin": cp.zeros((ny, nx), cp.float32),
        "chklowq": cp.zeros((ny, nx), cp.float32),
        "smois": f3(0.05, 0.4), "tslb": f3(260.0, 300.0), "smcrel": f3(0.0, 1.0),
    }
    dev["sh2o"] = dev["smois"] * 0.8
    return dev


def run(ctx=None, *, dt: float = 50.0):
    """Done-rule entry. Without ctx: one launch on a full-size 954 x 835 random state; asserts every
    float output is finite and returns {name: array}. With ctx: binds ctx.st by CONTRACT names."""
    import cupy as cp

    if ctx is None:
        ny, nx = 835, 954
        tables = synthetic_tables()
        dev = _random_state(ny, nx, tables[4], tables[5])
        launch(dev, tables, dt)
        cp.cuda.Stream.null.synchronize()
        outs = {}
        for name in _OUT_FLOAT:
            a = dev[name]
            if not bool(cp.isfinite(a).all()):
                bad = int((~cp.isfinite(a)).sum())
                raise FloatingPointError(f"{name}: {bad} non-finite values")
            outs[name] = a
        return outs

    st = ctx.st
    ny, nx = int(ctx.grid.jm), int(ctx.grid.im)
    if "sfn_tables" not in _tables:
        _tables["sfn_tables"] = synthetic_tables()
    tables = _tables["sfn_tables"]
    dev = {}
    missing = []
    for name in _F2D + _F3D + ("ivgtyp", "isltyp", "ebal"):
        sname = _STATE_NAME.get(name, name.upper())
        if sname in st.fields:
            dev[name] = st.fields[sname]
        elif name in _OUT_FLOAT or name in ("ebal",):
            shape = (NSOIL, ny, nx) if name in _F3D else (ny, nx)
            dtype = "int32" if name in ("ebal", "ivgtyp", "isltyp") else "float32"
            st.add(name, shape, dtype)
            dev[name] = st.fields[name]
        else:
            missing.append(name)
    if missing:
        raise NotImplementedError(f"phys_noah_sflx: no NMMB state-table field for inputs {missing}")
    launch(dev, tables, dt)
    return {n: dev[n] for n in _OUT_FLOAT}
