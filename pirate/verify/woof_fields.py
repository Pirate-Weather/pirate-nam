"""Turn one WOOF-NAM wrfout file into the native-grid fields compare_series.py scores.

WOOF-NAM writes model levels (bottom first, T = theta - 300, PH = g*z on interfaces, winds already earth-relative
on V points), not UPP output, so pressure-level fields are made here. Interpolation follows UPP's habit: linear in
ln(p) for T and height, and RH over water. Below-ground levels are left NaN, and the comparison drops them.
The remaining error this adds is measured by woof_floor.py.
Usage: woof_fields.py WRFOUT OUT.npz
"""
import sys, numpy as np, netCDF4
RD, CP, G = 287.04, 1004.6, 9.80665
KAPPA = RD / CP
EPS = 0.622
LEVELS_PA = {"850": 85000.0, "700": 70000.0, "500": 50000.0, "250": 25000.0}

def to_level(field, lnp, target):
    """Linear in ln(p). lnp falls with index, because the file is bottom first."""
    above = np.argmax(lnp < target, axis=0)
    below = np.clip(above - 1, 0, None)
    jj, ii = np.indices(above.shape)
    p_low, p_up = lnp[below, jj, ii], lnp[above, jj, ii]
    weight = (target - p_up) / np.where(p_low == p_up, np.nan, p_low - p_up)
    out = field[above, jj, ii] + weight * (field[below, jj, ii] - field[above, jj, ii])
    out[(above == 0) | (lnp[0] < target)] = np.nan  # the level is below ground
    return out

def es_water(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))

def dewpoint(q, p):
    e = q * p / (EPS + (1 - EPS) * q)
    x = np.log(np.maximum(e, 1e-3) / 611.2)
    return 273.15 + 243.5 * x / (17.67 - x)

d = netCDF4.Dataset(sys.argv[1]); d.set_auto_mask(False)
pres = d["PRES"][0].astype("f8"); lnp = np.log(pres)
temp = (d["T"][0] + 300.0) * (pres / 1e5) ** KAPPA
# History QVAPOR is a mixing ratio; the RH and PWAT formulas below take specific humidity.
qv = d["QVAPOR"][0]; qv = qv / (1.0 + qv)
ph = d["PH"][0] / G
# NMMB mid-level pressure is the arithmetic mean of its interfaces, so rebuild the interfaces upward from PSFC and
# interpolate height on them, as UPP's MDL2P does. Using mid-level means instead biases 250 mb height low.
pint = [d["PSFC"][0].astype("f8")]
for k in range(pres.shape[0]):
    pint.append(2 * pres[k] - pint[-1])
lnpint = np.log(np.maximum(np.stack(pint), 1.0))
uwind, vwind = d["U"][0], d["V"][0]
out = {"PRMSL": d["MSLP"][0], "TMP2m": d["T2"][0], "DPT2m": dewpoint(d["Q2"][0], d["PSFC"][0]), "REFC": d["COMPOSITE"][0],
       "APCP_total": d["RAINNC"][0] + d["RAINC"][0], "WIND10m": np.hypot(d["U10"][0], d["V10"][0])}
out["TMP850"] = to_level(temp, lnp, np.log(LEVELS_PA["850"]))
out["TMP500"] = to_level(temp, lnp, np.log(LEVELS_PA["500"]))
for lev in ("850", "500", "250"):
    out["HGT" + lev] = to_level(ph, lnpint, np.log(LEVELS_PA[lev]))
for lev in ("850", "250"):
    out["WIND" + lev] = np.hypot(to_level(uwind, lnp, np.log(LEVELS_PA[lev])), to_level(vwind, lnp, np.log(LEVELS_PA[lev])))
t700 = to_level(temp, lnp, np.log(LEVELS_PA["700"])); q700 = to_level(qv, lnp, np.log(LEVELS_PA["700"]))
qsat = EPS * es_water(t700) / (LEVELS_PA["700"] - (1 - EPS) * es_water(t700))
out["RH700"] = 100.0 * np.clip(q700 / qsat, 0, 1)
np.savez_compressed(sys.argv[2], **{k: np.asarray(v, "f4") for k, v in out.items()}, XLAT=d["XLAT"][0], XLONG=d["XLONG"][0])
