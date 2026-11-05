"""One WOOF-NAM wrfout -> the published fields, as float32 on the NAM H grid. Runs on the GPU machine.
Winds are earth-relative at H points. Height is interpolated on rebuilt interfaces, as UPP's MDL2P does.
Usage: product_fields.py WRFOUT OUT.npz
"""
import sys, numpy as np, netCDF4
RD, CP, G = 287.04, 1004.6, 9.80665
KAPPA, EPS = RD / CP, 0.622
LEVELS_HPA = (1000, 925, 850, 700, 500, 300, 250)

def to_level(field, lnp, target):
    above = np.argmax(lnp < target, axis=0); below = np.clip(above - 1, 0, None); jj, ii = np.indices(above.shape)
    p_low, p_up = lnp[below, jj, ii], lnp[above, jj, ii]
    weight = (target - p_up) / np.where(p_low == p_up, np.nan, p_low - p_up)
    out = field[above, jj, ii] + weight * (field[below, jj, ii] - field[above, jj, ii])
    out[(above == 0) | (lnp[0] < target)] = np.nan   # below ground: left missing, not extrapolated
    return out
def es_water(t): return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
def dewpoint(q, p):
    e = q * p / (EPS + (1 - EPS) * q); x = np.log(np.maximum(e, 1e-3) / 611.2)
    return 273.15 + 243.5 * x / (17.67 - x)

d = netCDF4.Dataset(sys.argv[1]); d.set_auto_mask(False)
pres = d["PRES"][0].astype("f8"); lnp = np.log(pres)
temp = (d["T"][0] + 300.0) * (pres / 1e5) ** KAPPA; qv = d["QVAPOR"][0]; qv = qv / (1.0 + qv); ph = d["PH"][0] / G; u, v = d["U"][0], d["V"][0]
# qv above: history QVAPOR is a mixing ratio, the RH and PWAT formulas here take specific humidity.
pint = [d["PSFC"][0].astype("f8")]
for k in range(pres.shape[0]): pint.append(2 * pres[k] - pint[-1])
pint = np.stack(pint); lnpint = np.log(np.maximum(pint, 1.0))
out = {"mslp": d["MSLP"][0], "t2m": d["T2"][0], "d2m": dewpoint(d["Q2"][0], d["PSFC"][0]), "refc": d["COMPOSITE"][0],
       "apcp_total": d["RAINNC"][0] + d["RAINC"][0], "u10": d["U10"][0], "v10": d["V10"][0], "psfc": d["PSFC"][0],
       "pwat": (qv * (pint[:-1] - pint[1:])).sum(axis=0) / G}
for lev in LEVELS_HPA:
    target = np.log(lev * 100.0)
    t = to_level(temp, lnp, target); q = to_level(qv, lnp, target)
    qsat = EPS * es_water(t) / (lev * 100.0 - (1 - EPS) * es_water(t))
    out[f"t{lev}"] = t; out[f"u{lev}"] = to_level(u, lnp, target); out[f"v{lev}"] = to_level(v, lnp, target)
    out[f"z{lev}"] = to_level(ph, lnpint, target); out[f"rh{lev}"] = 100.0 * np.clip(q / qsat, 0, 1)
np.savez_compressed(sys.argv[2], **{k: np.asarray(a, "f4") for k, a in out.items()})
