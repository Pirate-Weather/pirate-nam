"""phys-rrtm-driver-prep: NMMB RRTM driver inputs from the NMMB state (module_RA_RRTM.F90:98-878).

NOAA notice: module_RA_RRTM.F90 (NCEP/EMC NAM nam.v4.2.11, 83df4e13c1), US government work.
Ported to WOOF-NAM from that line range. Kernel: namgpu/kernels/phys_rrtm_driver_prep.cu
(rrtm_prep_p8w, then rrtm_prep_columns). Two launches, no host sync, no per-step allocation.

What it does, in Fortran order:
  1. The first nest: P8W (LM+1 interfaces) and P_PHY, from the last J row (the Fortran keeps the
     J = JTE values; see the kernel header, note 1).
  2. Zeroing of the 2-D cloud outputs CFRACH/CFRACL/CFRACM/CZMEAN/SIGT4, CLDCFR/MTOP/MBOT,
     CUTOP/CUBOT (from HTOP/HBOT), TAUTOTAL, CCMID, CSMID, and CLDFRA when CLD_FRACTION == 0.
  3. ICWP_Test (old NAM/GFDL cloud inputs, only when ICWP == -1, i.e. icmphys == 5) and GFDL_Conv
     (convective cloud fractions, only when CNCLD, i.e. ICWP == -1).
  4. Layering of cloud layers (KTOP/KBTM/CAMT/KCLD) and NCLDS, then the per-layer cloud
     properties (TAUTOTAL, CLDFRA).
  5. The scalars of the routine's prologue that the column loop (unit phys-rrtm-driver-call)
     reads: FHSWR, FHLWR, DTSW, DTLW, LSSWR, LSLWR, CONVPRATE, INDX_CW, FHR and SOLHR.
     They are host arithmetic on the control inputs, computed once per call, passed by value.

Inputs (ctx.st, device): PD [JM,IM], DSG2, SGML2 [LM], T, Q, QC, QS, CUPPT, HTOP, HBOT [..],
CLDFRA [LM,JM,IM] (INOUT), CFRACH/CFRACL/CFRACM/CZMEAN/SIGT4 [JM,IM] (OUT).
Inputs (grid, host): pt, pdsg1, psgml1 (uploaded once to ctx.tables).
Inputs (ctx.tables): "rrtm_axsd" = float32[310], the Gaussian table of RRTM_INIT
(module_RA_RRTM.F90:1480-1486, unit phys-rrtm-init). Required; the wrapper refuses to run without it.
Outputs: ctx.tables["rrtm_prep:out"] (dict of the scratch arrays the column loop reads) and the
returned dict of scalars. Scratch arrays are allocated once and reused (CONTRACT section 1).

Configuration (nothing silently defaulted to NAM values; run() reads ctx.cfg, see run()):
  nrads, nradl      <- ctx.cfg.nrads / ctx.cfg.nradl (parm/nam_configfile)
  icmphys           <- ctx.cfg.np3d (parm/nam_configfile np3d)
  icldflg           <- ctx.cfg.raw["icldflg"] if present, else 1 (physparam.f:141 default; the NAM
                       configfile has no icldflg key)
  oper              <- ctx.cfg.raw["oper"] (parm/nam_configfile oper)
  cld_fraction      <- ctx.cfg.raw["cldfraction"]: "default" -> 0, "thompson" -> 1; others refused
  cucld             <- ctx.cfg.raw["cucld"] if present, else False (physparam default; not in configfile)
  f_ni              <- ctx.cfg.microphysics == "thompson" (F_NI index only exists for Thompson)
  JDAT              <- ctx.tables["JDAT"] (required; no default, Fortran 863 reads JDAT(5), JDAT(6))
NAM 12 km runs as: dt = 25 s, nrads = nradl = 48, icmphys = 3, icldflg = 1, cld_fraction = 0,
oper = True, cucld = False, f_ni = False.

Precision: the scalars FHSWR, FHLWR, DTSW, DTLW, FHR and SOLHR are computed in float32, as the
Fortran REAL arithmetic does, and then returned as the float32 value.

Tile: kernel 1 hard-codes J = JM (Fortran JTE). run() requires the grid to be one full-domain tile.

Not ported (recorded): Fortran 430-435, the ICSDSW/ICSDLW zeroing (documented unused in the Fortran).
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_driver_prep"
K_P8W = "rrtm_prep_p8w"
K_COLS = "rrtm_prep_columns"
BLOCK_P8W = 128
NXSD = 310
CUPRATE = np.float32(24.0 * 1000.0)       # module_RA_RRTM.F90 CUPRATE = 24.*1000.
OUT_KEY = "rrtm_prep:out"
AXSD_KEY = "rrtm_axsd"


def gausin_table() -> np.ndarray:
    """AXSD(I) = GAUSIN(I*DXSD), I = 1..NXSD, DXSD = 0.01, NXSD = 310 (module_RA_RRTM.F90:1424-1517)."""
    import math
    rsqr = 1.0 / math.sqrt(2.0 * math.pi)
    out = np.zeros(NXSD, dtype=np.float32)
    for i in range(1, NXSD + 1):
        xsd = np.float32(i) * np.float32(0.01)
        a1 = xsd * rsqr
        a2 = math.exp(-0.5 * float(xsd) ** 2)
        b1, b2, s, rn = 1.0, 1.0, 1.0, 1.0
        while b2 > 1e-3:
            rn += 1.0
            b2 = float(xsd) ** 2 / (2.0 * rn - 1.0)
            b3 = b1 * b2
            s += b3
            b1 = b3
        out[i - 1] = a1 * a2 * s
    return out


def control(*, dt: float, nrads: int, nradl: int, icmphys: int, icldflg: int, oper: bool,
            cucld: bool, f_ni: bool, ntimestep: int, jdat) -> dict:
    """Scalars of the RRTM prologue (module_RA_RRTM.F90:98-878, the non-array part).

    ICWP = icldflg, and -1 when icmphys == 5. CNCLD (module variable, initially .TRUE.) becomes
    .FALSE. on the first call with ICWP /= -1, so for a fixed configuration CNCLD == (ICWP == -1).
    NTSPH = NINT(3600./DT); NRADPP = MIN(NRADS,NRADL); CLSTP = 1.0*NRADPP/NTSPH;
    CONVPRATE = CUPRATE/CLSTP, or 1000./CLSTP when ICWP > 0 and CUCLD.
    """
    f32 = np.float32
    icwp = -1 if icmphys == 5 else icldflg
    cnc = icwp == -1
    ntsph = int(math.floor(3600.0 / dt + 0.5))            # Fortran NINT (half away from zero)
    nradpp = min(nrads, nradl)
    clstp = f32(1.0 * nradpp) / f32(ntsph)
    convprate = CUPRATE / clstp
    if icwp > 0 and cucld:
        convprate = f32(1000.0) / clstp
    if jdat is None or len(jdat) < 6:
        raise RuntimeError("phys-rrtm-driver-prep: JDAT(1:6) is required (Fortran reads JDAT(5), JDAT(6))")
    ihr = int(jdat[4])                                    # JDAT(5), hour
    imin = int(jdat[5])                                   # JDAT(6), minute
    fhr = f32(ihr) + f32(imin) / f32(60.0)                # REAL arithmetic, as the Fortran
    dt32 = f32(dt)
    dtsw = f32(nrads) * dt32
    dtlw = f32(nradl) * dt32
    return {
        "ICWP": icwp,
        "CNCLD": cnc,
        "OPER": bool(oper),
        "CUCLD": bool(cucld),
        "CONVPRATE": float(convprate),
        "FHSWR": float(dtsw / f32(3600.0)),
        "FHLWR": float(dtlw / f32(3600.0)),
        "DTSW": float(dtsw),
        "DTLW": float(dtlw),
        "LSSWR": ntimestep % nrads == 0,
        "LSLWR": ntimestep % nradl == 0,
        "FHR": float(fhr),
        "SOLHR": float(np.fmod(fhr, f32(24.0))),
        "INDX_CW": 4 if f_ni else 3,                      # Thompson (F_NI) index, else 3
        "NTSPH": ntsph,
        "NRADPP": nradpp,
    }


def _scratch(ctx, name: str, shape, dtype):
    import cupy as cp

    key = "rrtm_prep:" + name
    a = ctx.tables.get(key)
    if a is None or a.shape != tuple(shape) or a.dtype != np.dtype(dtype):
        a = cp.zeros(shape, dtype=dtype)
        ctx.tables[key] = a
    return a


def _vec(ctx, name: str, host):
    import cupy as cp

    key = "rrtm_prep:vec:" + name
    a = ctx.tables.get(key)
    if a is None:
        a = cp.asarray(np.ascontiguousarray(np.asarray(host), dtype=np.float32))
        ctx.tables[key] = a
    return a


def _host(x):
    return x() if callable(x) else x


_CLD_FRACTION = {"default": 0, "thompson": 1}


def _raw(cfg, key):
    raw = cfg.raw
    if key not in raw:
        raise KeyError(f"phys-rrtm-driver-prep: configfile key {key!r} is missing (ctx.cfg.raw)")
    return raw[key]


def run(ctx, *, dt: float | None = None, nrads: int | None = None, nradl: int | None = None,
        icmphys: int | None = None, icldflg: int | None = None, cld_fraction: int | None = None,
        oper: bool | None = None, cucld: bool | None = None, f_ni: bool | None = None,
        jdat=None, ntimestep: int | None = None) -> dict:
    """Launch the two prep kernels once for the whole domain (one thread per column).

    Every control value comes from ctx.cfg (the parm/nam_configfile record) unless an explicit
    keyword overrides it; see the module docstring for the key of each value. JDAT is required.
    """
    import cupy as cp  # noqa: F401  (device arrays are cupy; import keeps the failure local)

    g = ctx.grid
    st = ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    jte = int(getattr(g, "jte", jm))
    if getattr(ctx, "slab", None) is not None:
        jte = jm          # several cards: the last rank's row JM is the domain's JTE, broadcast below
    if jte != jm:
        raise RuntimeError(f"phys-rrtm-driver-prep: kernel 1 assumes the Fortran JTE == JM "
                           f"(one full-domain tile); grid has jte={jte}, jm={jm}")
    cfg = ctx.cfg
    if cfg is None:
        raise RuntimeError("phys-rrtm-driver-prep needs ctx.cfg (the parm/nam_configfile record)")
    dt = float(ctx.dt if dt is None else dt)
    nt = int(ctx.ntimestep if ntimestep is None else ntimestep)
    nrads = int(cfg.nrads if nrads is None else nrads)
    nradl = int(cfg.nradl if nradl is None else nradl)
    icmphys = int(cfg.np3d if icmphys is None else icmphys)
    icldflg = int(cfg.raw.get("icldflg", 1) if icldflg is None else icldflg)
    if cld_fraction is None:
        key = str(_raw(cfg, "cldfraction")).lower()
        if key not in _CLD_FRACTION:
            raise ValueError(f"phys-rrtm-driver-prep: cldfraction {key!r} is not supported "
                             f"(known: {sorted(_CLD_FRACTION)})")
        cld_fraction = _CLD_FRACTION[key]
    oper = bool(_raw(cfg, "oper") if oper is None else oper)
    cucld = bool(cfg.raw.get("cucld", False) if cucld is None else cucld)
    if f_ni is None:
        f_ni = str(cfg.microphysics).lower() == "thompson"
    if jdat is None:
        jdat = ctx.tables.get("JDAT")
        if jdat is None:
            raise RuntimeError("phys-rrtm-driver-prep needs JDAT (ctx.tables['JDAT'] or jdat=); "
                               "Fortran reads JDAT(5) and JDAT(6) with no default")
    axsd = ctx.tables.get(AXSD_KEY)
    if axsd is None:
        # RRTM_INIT builds this once (module_RA_RRTM.F90:1480-1486) from constants only, so the
        # first call produces it here (run lead 0925Z: no unit produced it, radiation could not run).
        import cupy as _cp
        axsd = _cp.asarray(gausin_table())
        ctx.tables[AXSD_KEY] = axsd
    assert tuple(axsd.shape) == (NXSD,), axsd.shape

    ctl = control(dt=dt, nrads=nrads, nradl=nradl, icmphys=icmphys, icldflg=icldflg, oper=oper,
                  cucld=cucld, f_ni=f_ni, ntimestep=nt, jdat=jdat)

    # Scratch (allocated once, reused). Layouts are CONTRACT section 1: [level, j, i] and [j, i].
    p8w = _scratch(ctx, "P8W", (lm + 1, im), np.float32)
    p_phy = _scratch(ctx, "P_PHY", (lm, im), np.float32)
    csmid = _scratch(ctx, "CSMID", (lm, jm, im), np.float32)
    ccmid = _scratch(ctx, "CCMID", (lm, jm, im), np.float32)
    tautotal = _scratch(ctx, "TAUTOTAL", (lm, jm, im), np.float32)
    cutop = _scratch(ctx, "CUTOP", (jm, im), np.float32)
    cubot = _scratch(ctx, "CUBOT", (jm, im), np.float32)
    ktop = _scratch(ctx, "KTOP", (lm + 1, jm, im), np.int32)
    kbtm = _scratch(ctx, "KBTM", (lm + 1, jm, im), np.int32)
    camt = _scratch(ctx, "CAMT", (lm + 1, jm, im), np.float32)
    nclds = _scratch(ctx, "NCLDS", (jm, im), np.int32)
    kcld = _scratch(ctx, "KCLD", (jm, im), np.int32)
    cldcfr = _scratch(ctx, "CLDCFR", (3, jm, im), np.float32)
    mtop = _scratch(ctx, "MTOP", (3, jm, im), np.int32)
    mbot = _scratch(ctx, "MBOT", (3, jm, im), np.int32)

    pdsg1 = _vec(ctx, "PDSG1", _host(g.pdsg1))
    psgml1 = _vec(ctx, "PSGML1", _host(g.psgml1))

    # FPVS(T) is only read on the non-OPER branch; oper: true in the configfile, so it is skipped.
    if oper:
        fpvs = st["T"]
    else:
        from namgpu.physics.fer_hires import phys_fer_fpvs
        fpvs = phys_fer_fpvs.run(ctx, st["T"])[0]

    k8 = kernels.get(UNIT, K_P8W)
    kc = kernels.get(UNIT, K_COLS)
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        k8(((im + BLOCK_P8W - 1) // BLOCK_P8W, 1, 1), (BLOCK_P8W, 1, 1), (
            np.int32(im), np.int32(jm), np.int32(lm), np.float32(g.pt),
            st["PD"], pdsg1, st["DSG2"], st["SGML2"], psgml1, p8w, p_phy,
        ))
        # Several cards: kernel 1 used each slab's last memory row; the domain's row J = JMG lives on the last
        # rank (its memory ends at the domain edge), so its P8W/P_PHY go to every rank and all ranks use the
        # one-card values (MULTICARD-PLAN.md section 5). One card: comm is None, nothing runs.
        from namgpu import parallel as _par
        _comm = _par.get(ctx)
        if _comm is not None:
            _root = _comm.nranks - 1
            _comm.bcast(p8w, _root, ctx.stream)
            _comm.bcast(p_phy, _root, ctx.stream)
        kc(kernels.grid_cols(im, jm), kernels.BLOCK_COL, (
            np.int32(im), np.int32(jm), np.int32(lm),
            np.int32(1 if ctl["ICWP"] == -1 else 0),
            np.int32(1 if ctl["CNCLD"] else 0),
            np.int32(1 if oper else 0),
            np.int32(1 if cld_fraction == 0 else 0),
            np.float32(ctl["CONVPRATE"]),
            st["CLDFRA"], st["T"], st["Q"], st["QC"], st["QS"],
            st["CUPPT"], st["HTOP"], st["HBOT"], p8w, p_phy, fpvs, axsd,
            st["CFRACH"], st["CFRACL"], st["CFRACM"], st["CZMEAN"], st["SIGT4"],
            csmid, ccmid, tautotal, cutop, cubot, ktop, kbtm, camt, nclds, kcld,
            cldcfr, mtop, mbot,
        ))

    ctx.tables[OUT_KEY] = {
        "CSMID": csmid, "CCMID": ccmid, "TAUTOTAL": tautotal, "CUTOP": cutop, "CUBOT": cubot,
        "KTOP": ktop, "KBTM": kbtm, "CAMT": camt, "NCLDS": nclds, "KCLD": kcld,
        "CLDCFR": cldcfr, "MTOP": mtop, "MBOT": mbot, "P8W": p8w, "P_PHY": p_phy,
        "scalars": ctl,
    }
    return ctl
