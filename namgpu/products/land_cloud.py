"""awphys carriers for the soil, land, snow, surface flux, radiation and cloud records (record table group D).

Each carrier is a function of one WOOF-NAM history file on the model grid (the NMMB H points), computed the way
NCEP's post computes the record from the NMMB state (UPP / nam_post: SURFCE.f, FIXED.f, CLDRAD.f, CALDRG.f,
CANRES.f, ETCALC.f, SNFRAC.f; the formula of each is beside it), then regridded to grid 218. Soil, land, snow,
class, table and cloud-base carriers go to grid 218 by the nearest model point: NOAA's own awphys records of these
are nearest-point values (only 0 and 1 in LAND and ICEC, exactly 1 soil moisture and 273.16 K soil temperature at
every water point, only table values in the soil parameter records), and a bilinear blend would mix the water
sentinels into every coastal point. Fluxes, radiation and cloud cover are continuous and go bilinear, as the
hand-off does for every other field.

A carrier whose history inputs are missing is None: its record is left out of the file and named in the run index,
never filled. Inputs are read under their NMMB state names (io_history_fields.PRODUCT_CARRIERS_2D) or, for a
history written with the nest hand-off set, under the wrfout names that set uses (TSLB, SMOIS, SH2O, SNOW, SNOWH,
CANWAT, TMN, LANDMASK, SEAICE).

Frame names: NAM_<record>. Soil layers (NAM_TSOIL, NAM_SOILW, NAM_SOILL) are [4, ny, nx] on soil_layers_stag,
layer 0 the top (0-10 cm, then 10-40, 40-100, 100-200 cm); the table rows read them as carrier:NAM_TSOIL[k].
"""
from __future__ import annotations

import numpy as np

SOIL_DIMS = ("soil_layers_stag", "south_north", "west_east")
NSOIL_DZ = np.array([0.1, 0.3, 0.6, 1.0], np.float32)      # Noah layer thicknesses (m), NMMB SLDPTH
G = np.float32(9.8060226)                                  # NMMB module_CONSTANTS g (io_history_fields PH)
CAPPA = np.float32(287.04 / 1004.6)                        # NMMB R_D / CP (history T is theta - 300)
SMALL = 1.0e-6                                             # UPP params SMALL
QCLDMIN = 1.0e-6                                           # UPP params QCLDmin (the NAM-era value)
PQ0, A2, A3, A4 = 379.90516, 17.2693882, 273.16, 35.86     # UPP params (saturation specific humidity)
C2K = 273.15
STBOL = 5.67e-8                                            # CLDRAD.f FACTRL constant

# UPP's own soil tables (STATSGO classes 1-19): CANRES.f SMREF and SMWLT, ETCALC.f SMDRY and SMMAX
SMREF = np.array([0.236, 0.283, 0.312, 0.360, 0.360, 0.329, 0.315, 0.387, 0.382, 0.338, 0.404, 0.403, 0.348,
                  0.000, 0.133, 0.283, 0.403, 0.133, 0.236], np.float32)
SMWLT = np.array([0.023, 0.028, 0.047, 0.084, 0.084, 0.066, 0.069, 0.120, 0.103, 0.100, 0.126, 0.135, 0.069,
                  0.000, 0.012, 0.028, 0.135, 0.012, 0.023], np.float32)
SMDRY = SMWLT                                              # ETCALC.f SMDRY holds the same values as SMWLT
SMMAX = np.array([0.395, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406, 0.468, 0.457, 0.464,
                  0.000, 0.200, 0.421, 0.457, 0.200, 0.395], np.float32)
# CANRES.f IROOT for IGBP vegetation (ivegsrc 1: NAM's MODIS-IGBP 20 classes, 17 = water)
IROOT_IGBP = np.array([4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 0, 3, 3, 2], np.float32)
# SNFRAC.f SNUP (m) per vegetation class, SALP 4
SNUP = np.array([0.080, 0.080, 0.080, 0.080, 0.080, 0.020, 0.020, 0.060, 0.040, 0.020, 0.010, 0.020, 0.020, 0.020,
                 0.013, 0.013, 0.010, 0.020, 0.020, 0.020], np.float32)
SALP = 4.0


def _get(s, *names):
    """The first of `names` the history holds, Time level 0, float32; None if it holds none."""
    for n in names:
        if n in s.variables:
            return np.asarray(s[n][0], dtype=np.float32)
    return None


def _sm(s):
    """NMMB SM (1 = water), from SM or the hand-off LANDMASK (= 1 - SM)."""
    sm = _get(s, "SM")
    if sm is None:
        lm = _get(s, "LANDMASK")
        sm = None if lm is None else np.float32(1.0) - lm
    return sm


def _sice(s):
    return _get(s, "SICE", "SEAICE")


def _class_index(c, n):
    """Class numbers (float) -> 0-based table index, and the mask of numbers inside 1..n."""
    k = np.rint(c).astype(np.int64)
    ok = (k >= 1) & (k <= n)
    return np.where(ok, k - 1, 0), ok


# ------------------------------------------------------------------------------------------------ soil and land

def tsoil(s):
    """SURFCE.f soil temperature per Noah layer: STC as the state holds it (273.16 K over water)."""
    return _get(s, "TSLB", "STC")


def soilw(s):
    """SURFCE.f volumetric soil moisture per layer: SMC (1 over water and sea ice)."""
    return _get(s, "SMOIS", "SMC")


def soill(s):
    """SURFCE.f liquid volumetric soil moisture per layer: SH2O."""
    return _get(s, "SH2O")


def tsoil_3m(s):
    """SURFCE.f bottom soil temperature: TG, the NMMB deep soil temperature, written at 3 m."""
    return _get(s, "TG", "TMN")


def soilm(s):
    """SURFCE.f total soil moisture 0-2 m (kg m-2): SMSTOT; 1000 on open water (SM > SMALL and SICE < SMALL).
    With no SMSTOT in the history, Noah's own SMSTOT (phys_noah_driver: SOILM * 1000, SOILM the column sum of SMC
    times the layer thickness; 0 on sea ice) is rebuilt from SMOIS, the same state at the same time."""
    sm, sice = _sm(s), _sice(s)
    if sm is None or sice is None:
        return None
    tot = _get(s, "SMSTOT")
    if tot is None:
        smc = soilw(s)
        if smc is None:
            return None
        tot = np.float32(1000.0) * np.tensordot(NSOIL_DZ[: smc.shape[0]], smc, axes=1).astype(np.float32)
        tot = np.where(sice > 0.5, np.float32(0.0), tot)
    return np.where((sm > SMALL) & (sice < SMALL), np.float32(1000.0), tot).astype(np.float32)


def mstav(s):
    """SURFCE.f soil moisture availability (%): SMSTAV * 100 (Noah's root-zone availability)."""
    a = _get(s, "SMSTAV")
    return None if a is None else (a * np.float32(100.0)).astype(np.float32)


def cnwat(s):
    """SURFCE.f plant canopy surface water (kg m-2): CMC * 1000 (CMC in m); the hand-off CANWAT is already that."""
    c = _get(s, "CMC")
    if c is not None:
        return (c * np.float32(1000.0)).astype(np.float32)
    return _get(s, "CANWAT")


def weasd(s):
    """SURFCE.f snow water equivalent (kg m-2): SNO."""
    return _get(s, "SNO", "SNOW")


def snod(s):
    """SURFCE.f snow depth (m): SI * 0.001 (SI in mm); the hand-off SNOWH is already that."""
    si = _get(s, "SI")
    if si is not None:
        return (si * np.float32(0.001)).astype(np.float32)
    return _get(s, "SNOWH")


def snowc(s):
    """SURFCE.f percent snow cover: SNFRAC(SNO, IVGTYP) * 100, IVGTYP 0 read as 7 and classes outside 1-20 as 1
    (SNFRAC.f). SNO goes in as kg m-2 against SNUP in m, exactly as the post passes it."""
    sno, vt = weasd(s), _get(s, "IVGTYP")
    if sno is None or vt is None:
        return None
    iv = np.rint(vt).astype(np.int64)
    iv = np.where(iv == 0, 7, iv)
    iv = np.where((iv < 1) | (iv > 20), 1, iv)
    snup = SNUP[iv - 1]
    r = sno / snup
    cov = np.where(sno < snup, 1.0 - (np.exp(-SALP * r) - r * np.exp(-SALP)), 1.0)
    return (np.clip(cov, 0.0, 1.0) * 100.0).astype(np.float32)


def icec(s):
    """FIXED.f sea ice mask: SICE."""
    return _sice(s)


def land(s):
    """FIXED.f land-sea mask: 1 - SM, and 0 where SICE > 0.1."""
    sm, sice = _sm(s), _sice(s)
    if sm is None or sice is None:
        return None
    return np.where(sice > 0.1, np.float32(0.0), np.float32(1.0) - sm).astype(np.float32)


def _class_on_land(s, name):
    """FLOAT(class) as the post writes it from the NMMB state. NAM's own state holds 0 at every sea and sea-ice
    point (its archived SOTYP/VGTYP are 0 wherever its LAND is 0); WOOF-NAM's init fills those points with land
    classes for the land model (init-static-missing), so they are written 0 where LAND is 0, as NAM's are."""
    c, lm = _get(s, name), land(s)
    if c is None or lm is None:
        return None
    return np.where(lm > 0.5, c, np.float32(0.0)).astype(np.float32)


def sotyp(s):
    """SURFCE.f soil type: FLOAT(ISLTYP), 0 off land."""
    return _class_on_land(s, "ISLTYP")


def vgtyp(s):
    """SURFCE.f vegetation type: FLOAT(IVGTYP), 0 off land."""
    return _class_on_land(s, "IVGTYP")


def _land_noice(s):
    sm, sice = _sm(s), _sice(s)
    if sm is None or sice is None:
        return None
    return (np.abs(sm) < 1.0e-5) & (np.abs(sice) < 1.0e-5)


def _soil_table(s, table, need_veg):
    """A UPP soil-table value at ISLTYP on land with no sea ice (and IVGTYP != 17 for the CANRES set), else 0."""
    ok, soil = _land_noice(s), _get(s, "ISLTYP")
    if ok is None or soil is None:
        return None
    if need_veg:
        vt = _get(s, "IVGTYP")
        if vt is None:
            return None
        ok = ok & (np.rint(vt) != 17)
    k, inside = _class_index(soil, table.size)
    return np.where(ok & inside, table[k], np.float32(0.0)).astype(np.float32)


def smdry(s):
    """SURFCE.f/ETCALC.f direct evaporation cease soil moisture: SMDRY(ISLTYP) on land with no ice, else 0."""
    return _soil_table(s, SMDRY, False)


def poros(s):
    """SURFCE.f/ETCALC.f soil porosity: SMMAX(ISLTYP) on land with no ice, else 0."""
    return _soil_table(s, SMMAX, False)


def wilt(s):
    """SURFCE.f/CANRES.f wilting point: SMWLT(ISLTYP) on land with no ice and IVGTYP != 17, else 0."""
    return _soil_table(s, SMWLT, True)


def smref(s):
    """SURFCE.f/CANRES.f transpiration stress-onset soil moisture: SMREF(ISLTYP), masked as the wilting point."""
    return _soil_table(s, SMREF, True)


def rlyrs(s):
    """SURFCE.f/CANRES.f number of root layers: IROOT(IVGTYP) (IGBP), on land with no ice and IVGTYP != 17,
    else 0."""
    ok, vt = _land_noice(s), _get(s, "IVGTYP")
    if ok is None or vt is None:
        return None
    ok = ok & (np.rint(vt) != 17)
    k, inside = _class_index(vt, IROOT_IGBP.size)
    return np.where(ok & inside, IROOT_IGBP[k], np.float32(0.0)).astype(np.float32)


def veg(s):
    """SURFCE.f green vegetation fraction (%): VEGFRC * 100."""
    v = _get(s, "VEGFRC")
    return None if v is None else (v * np.float32(100.0)).astype(np.float32)


def sfcr(s):
    """SURFCE.f roughness length (m): Z0."""
    return _get(s, "Z0")


def fricv(s):
    """SURFCE.f friction velocity (m s-1): USTAR."""
    return _get(s, "USTAR")


def cd(s):
    """SURFCE.f/CALDRG.f surface drag coefficient: USTAR^2 over the squared mean wind between the lowest layer and
    z0 (B grid, threshold 1e-6), computed on the device from the state (io_history_fields DRAGCO)."""
    return _get(s, "DRAGCO")


# ------------------------------------------------------------------------------------------------ fluxes, radiation

def shtfl(s):
    """SURFCE.f instantaneous sensible heat flux (W m-2, upward positive): -TWBS."""
    t = _get(s, "TWBS")
    return None if t is None else (-t).astype(np.float32)


def lhtfl(s):
    """SURFCE.f instantaneous latent heat flux (W m-2, upward positive): -QWBS."""
    q = _get(s, "QWBS")
    return None if q is None else (-q).astype(np.float32)


def albdo(s):
    """FIXED.f surface albedo (%): ALBEDO * 100."""
    a = _get(s, "ALBEDO")
    return None if a is None else (a * np.float32(100.0)).astype(np.float32)


def _factrs(s):
    czen, czmean = _get(s, "CZEN"), _get(s, "CZMEAN")
    if czen is None or czmean is None:
        return None
    return np.where(czmean > 1.0e-6, czen / np.where(czmean > 1.0e-6, czmean, 1.0), 0.0).astype(np.float32)


def dswrf(s):
    """CLDRAD.f current incoming SW at the surface: RSWIN * CZEN / CZMEAN (0 where CZMEAN <= 1e-6)."""
    r, f = _get(s, "RSWIN"), _factrs(s)
    return None if r is None or f is None else (r * f).astype(np.float32)


def uswrf(s):
    """CLDRAD.f current outgoing SW at the surface: RSWOUT * CZEN / CZMEAN."""
    r, f = _get(s, "RSWOUT"), _factrs(s)
    return None if r is None or f is None else (r * f).astype(np.float32)


def _t_lowest(s):
    """Temperature of the lowest model layer (K), from the history's theta - 300 and mid-layer pressure."""
    if "T" not in s.variables or "PRES" not in s.variables:
        return None
    th = np.asarray(s["T"][0, 0], dtype=np.float32) + np.float32(300.0)
    p = np.asarray(s["PRES"][0, 0], dtype=np.float32)
    return (th * (p * np.float32(1.0e-5)) ** CAPPA).astype(np.float32)


def dlwrf(s):
    """CLDRAD.f current incoming LW at the surface: RLWIN * 5.67e-8 T_lowest^4 / SIGT4 (0 where SIGT4 <= 0)."""
    r, sig, t = _get(s, "RLWIN"), _get(s, "SIGT4"), _t_lowest(s)
    if r is None or sig is None or t is None:
        return None
    t4 = np.float32(STBOL) * t * t * t * t
    f = np.where(sig > 0.0, t4 / np.where(sig > 0.0, sig, 1.0), 0.0)
    return (r * f).astype(np.float32)


def ulwrf(s):
    """CLDRAD.f current outgoing LW at the surface: RADOT."""
    return _get(s, "RADOT")


# ------------------------------------------------------------------------------------------------ clouds

def _box_mean(a, r):
    """Mean over the (2r+1)^2 box around every point, truncated at the edges (CLDRAD.f nmmb_clds1); a [j, i]."""
    jm, im = a.shape
    c = np.zeros((jm + 1, im + 1), np.float64)
    c[1:, 1:] = np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1)
    j = np.arange(jm)
    i = np.arange(im)
    j0, j1 = np.maximum(j - r, 0), np.minimum(j + r, jm - 1) + 1
    i0, i1 = np.maximum(i - r, 0), np.minimum(i + r, im - 1) + 1
    tot = (c[j1[:, None], i1[None, :]] - c[j0[:, None], i1[None, :]]
           - c[j1[:, None], i0[None, :]] + c[j0[:, None], i0[None, :]])
    return tot / ((j1 - j0)[:, None] * (i1 - i0)[None, :])


def tcdc(s):
    """CLDRAD.f total cloud cover (%), NMMB B grid: TCLD, the column maximum of the radiation cloud fraction CLDFRA
    averaged over a box of half width NINT(14259 m / DY) (the area of a 10 mi radius), times 100."""
    if "CLDFRA" not in s.variables:
        return None
    cf = np.asarray(s["CLDFRA"][0], dtype=np.float32)
    dy = float(getattr(s, "DY", 0.0) or 0.0)               # the history writes DY in m (DYH); older files in degrees
    dy_m = dy if dy > 1.0 else dy * 111200.0                 # UPP: DY_m = DYVAL (1e-6 deg) * 0.1112
    r = max(int(round(14259.0 / dy_m)), 0) if dy_m > 0 else 1
    tcld = np.zeros(cf.shape[-2:], np.float64)
    for k in range(cf.shape[0]):
        np.maximum(tcld, _box_mean(cf[k], r), out=tcld)
    return (tcld * 100.0).astype(np.float32)


def cloud_base(s):
    """CLDRAD.f cloud base height (m above sea level), NMMB: the mid-layer height of the lowest layer that is
    grid-scale cloud (cloud water + cloud ice >= QCLDmin, or snow >= QCLDmin at RH >= 98 % at or above the PBL top)
    or the BMJ convective cloud base (CNVBOT, kept only where the convective cloud is more than one layer deep),
    whichever is lower; -5000 with no cloud. The saturation test runs only at the layers holding snow."""
    need = ("T", "PRES", "PH", "QVAPOR", "QCLOUD", "QICE", "QSNOW", "PBLH", "CNVBOT", "CNVTOP")
    if any(n not in s.variables for n in need):
        return None
    w = np.asarray(s["QVAPOR"][0], dtype=np.float32)                      # [nz, j, i], k = 0 the lowest layer
    inv = np.float32(1.0) / (np.float32(1.0) + w)                          # history mixing ratios -> specific
    cloudy = (np.asarray(s["QCLOUD"][0], dtype=np.float32) + np.asarray(s["QICE"][0], dtype=np.float32)) * inv
    cloudy = cloudy >= QCLDMIN
    qqs = np.asarray(s["QSNOW"][0], dtype=np.float32) * inv
    ph = np.asarray(s["PH"][0], dtype=np.float32)
    nz = w.shape[0]
    snow = np.nonzero((qqs >= QCLDMIN) & ~cloudy)                          # only these can change the answer
    if snow[0].size:
        th = np.asarray(s["T"][0], dtype=np.float32)[snow] + np.float32(300.0)
        p = np.asarray(s["PRES"][0], dtype=np.float32)[snow]
        tk = th * (p * np.float32(1.0e-5)) ** CAPPA
        q = (w[snow] * inv[snow]).astype(np.float64)
        warm = tk >= C2K
        qsat = np.where(warm, PQ0 / p * np.exp(A2 * (tk - A3) / (tk - A4)),
                        PQ0 / p * np.exp(21.8745584 * (tk - A3) / (tk - 7.66)))
        kk, jj, ii = snow
        zm = np.float32(0.5) * (ph[kk, jj, ii] + ph[kk + 1, jj, ii]) / G
        zpbl = _get(s, "PBLH")[jj, ii] + ph[0, jj, ii] / G
        hit = (q / qsat >= 0.98) & (zm >= zpbl)
        cloudy[kk[hit], jj[hit], ii[hit]] = True
    anyc = cloudy.any(axis=0)
    kbase = np.argmax(cloudy, axis=0)                                      # lowest cloudy layer, bottom-up
    ibot_gr = np.where(anyc, nz - kbase, 0)                                # UPP L (1 = top layer)
    # BMJ: the post reads CNVBOT/CNVTOP (bottom-up layer numbers) as HBOT = LM - CNVBOT + 1, HTOP likewise
    ib = np.rint(nz - _get(s, "CNVBOT") + 1.0).astype(np.int64)
    it = np.rint(nz - _get(s, "CNVTOP") + 1.0).astype(np.int64)
    ibot_cu = np.where(ib - it <= 1, 0, ib)
    ibot = np.maximum(ibot_gr, ibot_cu)
    okb = (ibot > 0) & (ibot <= nz)
    k = np.clip(nz - ibot, 0, nz - 1)
    zb = np.take_along_axis(ph, k[None], axis=0)[0]
    zt = np.take_along_axis(ph, (k + 1)[None], axis=0)[0]
    return np.where(okb, np.float32(0.5) * (zb + zt) / G, np.float32(-5000.0)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the carrier set

# frame name -> (function of the history Dataset, regrid to grid 218: "nearest" | "bilinear")
CARRIERS = {
    "NAM_TSOIL": (tsoil, "nearest"), "NAM_SOILW": (soilw, "nearest"), "NAM_SOILL": (soill, "nearest"),
    "NAM_TSOIL3M": (tsoil_3m, "nearest"), "NAM_MSTAV": (mstav, "nearest"), "NAM_SOILM": (soilm, "nearest"),
    "NAM_CNWAT": (cnwat, "nearest"), "NAM_WEASD": (weasd, "nearest"), "NAM_SNOWC": (snowc, "nearest"),
    "NAM_SNOD": (snod, "nearest"), "NAM_SMDRY": (smdry, "nearest"), "NAM_POROS": (poros, "nearest"),
    "NAM_SFCR": (sfcr, "nearest"), "NAM_FRICV": (fricv, "bilinear"), "NAM_CD": (cd, "nearest"),
    "NAM_VEG": (veg, "nearest"), "NAM_VGTYP": (vgtyp, "nearest"), "NAM_SOTYP": (sotyp, "nearest"),
    "NAM_RLYRS": (rlyrs, "nearest"), "NAM_WILT": (wilt, "nearest"), "NAM_SMREF": (smref, "nearest"),
    "NAM_CLDBASE": (cloud_base, "nearest"), "NAM_LAND": (land, "nearest"), "NAM_ICEC": (icec, "nearest"),
    "NAM_ALBDO": (albdo, "bilinear"), "NAM_SHTFL": (shtfl, "bilinear"), "NAM_LHTFL": (lhtfl, "bilinear"),
    "NAM_TCDC": (tcdc, "bilinear"), "NAM_DSWRF": (dswrf, "bilinear"), "NAM_DLWRF": (dlwrf, "bilinear"),
    "NAM_USWRF": (uswrf, "bilinear"), "NAM_ULWRF": (ulwrf, "bilinear"),
}
SOIL = ("NAM_TSOIL", "NAM_SOILW", "NAM_SOILL")


def compute(history: str, name: str):
    """Worker side: (native-grid array or None, regrid mode) for one carrier of one history file."""
    import netCDF4
    fn, mode = CARRIERS[name]
    with netCDF4.Dataset(history) as s:
        return fn(s), mode
