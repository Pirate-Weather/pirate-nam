"""phys-noah-shflx: SHFLX, soil heat flux and soil temperature update (Noah LSM).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_LS_NOAHLSM.F90:3729-3796. NOAA US
government work; the module NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_noah_shflx.cu (phys_noah_shflx, one thread per (i, j), one launch).
SHFLX calls HRT (or HRTICE when ICE == 1), HSTEP and ROSR12; their Fortran (and SNKSRC, FRH2O,
TBND, TDFCND, TMPAVG, which HRT calls) is ported as device functions inside the same .cu file.

Two call sites, both bound from the model state (no per-column constants):
  site "sflx"   SFLX (module_LS_NOAHLSM.F90:3352-3372, the caller side, call at :886). YY and ZZ1
                are SFLX own values. T1 is the state skin temperature TSKIN, updated in place.
  site "snopac" SNOPAC (module_LS_NOAHLSM.F90:4356). ZZ1 = 1.0 and YY = STC(1) - 0.5*SSOIL*ZSOIL(1)
                *ZZ1/DF1, computed here from the caller SSOIL. T11 is a dummy copy of T1, so
                TSKIN is not written by this site. Its SSOIL output goes to SSOIL1 (not used).

Caller contract (ctx.sflx, per-column [JM, IM]; the caller computes these, as SFLX does):
  ICE   int32 sea-ice flag.
  DF1   thermal diffusivity AFTER the caller modifications: TDFCND(...), then
        IF (VEGTYP == 1) DF1 = 3.24 (module_LS_NOAHLSM.F90:3354-3356; the caller tests VEGTYP == 1
        while HRT tests the urban index IW, an inconsistency in the Fortran caller that is kept),
        then DF1 = DF1*EXP(SBETA*SHDFAC). SHFLX uses DF1 as given.
  YY, ZZ1 (site sflx only) as SFLX computes them (module_LS_NOAHLSM.F90:3368-3371).
  SSOIL (site snopac only) the caller SSOIL before the SNOPAC call.

Bound from the NMMB state table: STC, SMC, SH2O (INOUT), TSKIN (T1, INOUT, sflx site), TBOT, ISLTYP,
IVGTYP (VEGTYP), SLDPTH (ZSOIL(1) = -SLDPTH(1), cumulative). The soil parameters SMCMAX, PSISAT, BEXP,
QUARTZ come from the soil table at ISLTYP (SOILPARM), with the urban override of SFLX
(module_LS_NOAHLSM.F90:1852-1866, VEGTYP == IW). CSOIL and ZBOT are GENPARM scalars.
F1 = ALOG10(PSISAT)+BEXP*ALOG10(SMCMAX)+2 (SFLX :1867). HRT never reads F1, so the value is
inert in SHFLX; it is bound with the same formula for every cell.

Loop limits: NOAHLSM is called with ITS_B1..ITE_B1 and JTS_B1..JTE_B1 (module_TURBULENCE.F90:899-900),
so the kernel updates only the interior (0-based rows and columns 1..JM-2 and 1..IM-2) and leaves the
boundary untouched. The sflx kernel (phys_noah_sflx.cu) does not apply this limit yet (named follow-up).

Outputs: STC, SH2O, T1 (site sflx: the state TSKIN) and SSOIL (state SSOIL for site sflx, SSOIL1 for
snopac). Scratch is allocated once through ctx.st.add. Nothing is read back to the host on the step
path; the soil table is bound once through the parm-table cache.

run(ctx): with ctx, binds the state as above and raises naming any field that is missing. Without ctx,
run() builds a full-size 954 x 835 synthetic column set (synthetic soil table, test values only) and
asserts every output is finite.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels

UNIT = "phys_noah_shflx"
KERNEL = "phys_noah_shflx"
BLOCK = (32, 8, 1)
NSOIL = 4
NSLTYPE = 30          # module_LS_NOAHLSM.F90:162 (soil table length)
DT_LSM = 50.0         # DTPHS: the LSM step (NPHS = 2, dt 25 s)
IVEGSRC = 1           # IGBP vegetation (CONTRACT section 5)

# Urban override of SFLX, module_LS_NOAHLSM.F90:1852-1867 (VEGTYP == IW)
URBAN_SMCMAX = 0.45
URBAN_SMCWLT = 0.40
URBAN_SMLOW = 0.4978
URBAN_BEXP = 17.0

SITES = ("sflx", "snopac")
SITE_KEYS = {
    "sflx": ("ICE", "DF1", "YY", "ZZ1"),
    "snopac": ("ICE", "DF1", "SSOIL"),
}
STATE_KEYS = ("STC", "SMC", "SH2O", "TBOT", "ISLTYP", "IVGTYP", "SLDPTH", "TSKIN")


def _grid(im: int, jm: int):
    bx, by, _ = BLOCK
    return ((im + bx - 1) // bx, (jm + by - 1) // by, 1)


def _interior(im: int, jm: int):
    """0-based inclusive interior: ITS_B1 = MAX(ITS, IDS+1), ITE_B1 = MIN(ITE, IDE-1), and the JTS/JTE
    pair, on the full halo-free domain (module_MY_DOMAIN_SPECS.F90:146-158)."""
    return (1, im - 2, 1, jm - 2)


def iw_for(ivegsrc: int) -> int:
    """IW (module_LS_NOAHLSM.F90:629-630): 13 for IGBP (IVEGSRC == 1), otherwise 1."""
    return 13 if ivegsrc == 1 else 1


def shflx(stc, smc, sh2o, zsoil, smcmax, t1, ssoil, yy, zz1, tbot, zbot, psisat, bexp, f1,
          df1, quartz, csoil, ice, vegtyp, dt=DT_LSM, ivegsrc=IVEGSRC, interior=None):
    """STC, SH2O (INOUT), T1 (INOUT), SSOIL (OUT). stc, smc, sh2o are [NSOIL, JM, IM] float32."""
    if stc.shape[0] != NSOIL:
        raise ValueError(f"phys-noah-shflx: STC has {stc.shape[0]} layers, kernel NSOIL = {NSOIL}")
    jm, im = int(stc.shape[1]), int(stc.shape[2])
    i_lo, i_hi, j_lo, j_hi = _interior(im, jm) if interior is None else interior
    k = kernels.get(UNIT, KERNEL)
    k(_grid(im, jm), BLOCK,
      (stc, smc, sh2o, zsoil, smcmax, t1, ssoil, yy, zz1, tbot, zbot, psisat, bexp, f1, df1,
       quartz, csoil, ice, vegtyp, np.float32(dt), np.int32(ivegsrc), np.int32(im), np.int32(jm),
       np.int32(i_lo), np.int32(i_hi), np.int32(j_lo), np.int32(j_hi)))
    return stc, sh2o, t1, ssoil


def _soil_columns(isltyp, ivgtyp, soil, csoil_scalar, zbot_scalar, iw):
    """Per-column SFLX soil parameters at ISLTYP (1-based), with the urban override for VEGTYP == IW."""
    import cupy as cp

    f32 = cp.float32
    jm, im = isltyp.shape
    idx = cp.clip(isltyp.astype(cp.int32) - 1, 0, NSLTYPE - 1)
    urban = ivgtyp == iw
    smcmax = cp.where(urban, f32(URBAN_SMCMAX), cp.take(soil["soil.MAXSMC"], idx)).astype(f32)
    bexp = cp.where(urban, f32(URBAN_BEXP), cp.take(soil["soil.BB"], idx)).astype(f32)
    urban_psisat = np.float32(200.0) / np.power(
        np.float32(URBAN_SMCWLT) / (np.float32(URBAN_SMCMAX) * (np.float32(1.0) - np.float32(URBAN_SMLOW))),
        np.float32(-URBAN_BEXP))
    psisat = cp.where(urban, f32(urban_psisat), cp.take(soil["soil.SATPSI"], idx)).astype(f32)
    f1 = (cp.log10(psisat) + bexp * cp.log10(smcmax) + f32(2.0)).astype(f32)
    quartz = cp.take(soil["soil.QTZ"], idx).astype(f32)
    csoil = cp.full((jm, im), csoil_scalar, dtype=f32)
    zbot = cp.full((jm, im), zbot_scalar, dtype=f32)
    return dict(smcmax=smcmax, bexp=bexp, psisat=psisat, f1=f1, quartz=quartz, csoil=csoil, zbot=zbot)


def _advance(site, fields, sf, soil, csoil, zbot, ivegsrc, jm, im, add):
    """One SHFLX call for a site. fields: the state (name -> array); sf: the caller per-column dict;
    soil: device soil table; add(name, shape) returns a unit-owned scratch array."""
    import cupy as cp

    if site not in SITES:
        raise ValueError(f"phys-noah-shflx: unknown site {site!r}; use one of {SITES}")
    missing = [k for k in STATE_KEYS if k not in fields]
    missing += [f"sflx[{k}]" for k in SITE_KEYS[site] if k not in sf]
    if missing:
        raise ValueError(f"phys-noah-shflx ({site}): not bound: " + ", ".join(missing))
    if sf["ICE"].dtype != cp.int32:
        raise ValueError(f"phys-noah-shflx ({site}): ICE must be int32, got {sf['ICE'].dtype}")
    for k in SITE_KEYS[site]:
        if k != "ICE" and tuple(sf[k].shape) != (jm, im):
            raise ValueError(f"phys-noah-shflx ({site}): {k} shape {tuple(sf[k].shape)} != {(jm, im)}")
    if tuple(fields["SLDPTH"].shape) != (NSOIL,):
        raise ValueError(f"phys-noah-shflx: SLDPTH shape {tuple(fields['SLDPTH'].shape)} != ({NSOIL},)")

    iw = iw_for(ivegsrc)
    cols = _soil_columns(fields["ISLTYP"], fields["IVGTYP"], soil, csoil, zbot, iw)
    zsoil = -cp.cumsum(cp.asarray(fields["SLDPTH"], dtype=cp.float32))
    stc = fields["STC"]
    df1 = sf["DF1"]
    if site == "sflx":
        t1 = fields["TSKIN"]
        yy = sf["YY"]
        zz1 = sf["ZZ1"]
        ssoil = add("SSOIL", (jm, im))
    else:
        t1 = add("T11", (jm, im))
        t1[...] = fields["TSKIN"]
        zz1 = add("ZZ1_SNOPAC", (jm, im))
        zz1[...] = 1.0
        yy = stc[0] - 0.5 * sf["SSOIL"] * zsoil[0] * zz1 / df1
        ssoil = add("SSOIL1", (jm, im))
    shflx(stc, fields["SMC"], fields["SH2O"], zsoil, cols["smcmax"], t1, ssoil, yy, zz1,
          fields["TBOT"], cols["zbot"], cols["psisat"], cols["bexp"], cols["f1"], df1,
          cols["quartz"], cols["csoil"], sf["ICE"], fields["IVGTYP"], DT_LSM, ivegsrc)
    return {"STC": stc, "SH2O": fields["SH2O"], "T1": t1, "SSOIL": ssoil}


def _ctx_run(ctx, site):
    from namgpu.physics.noah import phys_noah_parm_tables as pt

    st = ctx.st
    jm, im = int(ctx.grid.jm), int(ctx.grid.im)
    sf = getattr(ctx, "sflx", None) or {}
    soil = pt.run(ctx)
    host = ctx.tables[pt.KEY_HOST]
    ivegsrc = int(ctx.cfg.raw["ivegsrc"])
    return _advance(site, st.fields, sf, soil, float(host.gen["CSOIL"]), float(host.gen["ZBOT"]),
                    ivegsrc, jm, im, st.add)


def _synthetic_soil():
    import cupy as cp

    k = np.arange(NSLTYPE, dtype=np.float32)
    return {
        "soil.BB": cp.asarray(4.0 + 0.1 * k),
        "soil.MAXSMC": cp.asarray(0.43 + 0.001 * k),
        "soil.SATPSI": cp.asarray(0.1 + 0.02 * k),
        "soil.QTZ": cp.asarray(0.05 + 0.02 * k),
    }


def _synthetic_run(site, seed=1):
    """Full-size 954 x 835 synthetic column set (test values only, not NAM tables)."""
    import cupy as cp

    jm, im = 835, 954
    rng = np.random.default_rng(seed)

    def f(lo, hi, shape=(jm, im)):
        return cp.asarray(rng.uniform(lo, hi, size=shape).astype(np.float32))

    smc = f(0.2, 0.4, (NSOIL, jm, im))
    fields = {
        "STC": f(250.0, 285.0, (NSOIL, jm, im)),
        "SMC": smc,
        "SH2O": cp.minimum(f(0.0, 0.2, (NSOIL, jm, im)), smc),
        "TBOT": f(270.0, 280.0),
        "ISLTYP": cp.asarray(rng.integers(1, 10, size=(jm, im)).astype(np.int32)),
        "IVGTYP": cp.asarray(rng.choice([1, 2, 4, 7, 10, 13, 15, 16], size=(jm, im)).astype(np.int32)),
        "SLDPTH": cp.asarray(np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)),
        "TSKIN": f(260.0, 300.0),
    }
    sf = {
        "ICE": cp.asarray((rng.uniform(size=(jm, im)) < 0.05).astype(np.int32)),
        "DF1": f(0.5, 3.0),
        "YY": f(270.0, 290.0),
        "ZZ1": f(0.5, 1.5),
        "SSOIL": f(-50.0, 50.0),
    }

    def add(name, shape):
        if name not in fields:
            fields[name] = cp.zeros(shape, dtype=cp.float32)
        return fields[name]

    outs = _advance(site, fields, sf, _synthetic_soil(), 1.4e6, -2.0, IVEGSRC, jm, im, add)
    cp.cuda.Stream.null.synchronize()
    for name in ("STC", "SH2O", "T1", "SSOIL"):
        a = outs[name]
        if not bool(cp.isfinite(a).all()):
            raise FloatingPointError(f"phys-noah-shflx ({site}): {name} has "
                                     f"{int((~cp.isfinite(a)).sum())} non-finite values")
    return outs


def run(ctx=None, site: str = "sflx"):
    """Done-rule entry. With ctx: binds the state and the caller ctx.sflx for the site and returns
    {STC, SH2O, T1, SSOIL}. Without ctx: the full-size synthetic check for the site."""
    if ctx is None:
        return _synthetic_run(site)
    return _ctx_run(ctx, site)
