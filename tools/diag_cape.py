"""Warm-sector CAPE census: our history vs the archived NAM forecast (grid 218), same valid time.
python tools/diag_cape.py --hist runs/.../wrfout_d01_2025-03-14_18:00:00 --nam cases/.../awphys06.grib2
"""
import argparse, os, subprocess, shutil, sys
from pathlib import Path
import numpy as np, netCDF4

import gpuwm
B = Path(os.path.dirname(gpuwm.__file__)) / "libexec/bridges"
TMP = Path("/work/nam-gpu/scratch/physconv/tmp")


def inventory(path):
    lines = subprocess.run([B / "grib2_inventory", path], capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [l for l in lines if not l.startswith("#")]
    head = rows[0].split("\t")
    return [dict(zip(head, r.split("\t"))) for r in rows[1:]]


def dump(path, idxs, tag):
    dd = TMP / tag
    if dd.exists():
        shutil.rmtree(dd)
    subprocess.run([B / "grib2_dump", path, dd, *map(str, idxs)], check=True, capture_output=True)
    meta = [l.split("\t") for l in (dd / "metadata.tsv").read_text().splitlines()]
    h = meta[0]; out = {}
    for m in meta[1:]:
        r = dict(zip(h, m))
        out[int(r["index"])] = np.fromfile(dd / r["filename"], "<f8").reshape(int(r["ny"]), int(r["nx"]))
    shutil.rmtree(dd)
    return out


def lcc218(nx, ny):
    R = 6371229.0; phi0 = np.radians(25.0); lov = np.radians(265.0 - 360.0)
    n = np.sin(phi0); F = np.cos(phi0) * np.tan(np.pi / 4 + phi0 / 2) ** n / n
    rho = lambda phi: R * F / np.tan(np.pi / 4 + phi / 2) ** n
    rho0 = rho(phi0)
    la1, lo1 = np.radians(12.19), np.radians(226.541 - 360.0)
    r1 = rho(la1)
    x0 = r1 * np.sin(n * (lo1 - lov)); y0 = rho0 - r1 * np.cos(n * (lo1 - lov))
    x = x0 + 12191.0 * np.arange(nx)[None, :]; y = y0 + 12191.0 * np.arange(ny)[:, None]
    yy = rho0 - y
    rr = np.hypot(x, yy)
    lat = 2 * np.arctan((R * F / rr) ** (1 / n)) - np.pi / 2
    lon = lov + np.arctan2(x, yy) / n
    return np.degrees(lat), np.degrees(lon)


EPS = 0.622


def esat(T):  # Pa, Bolton
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def qsat(T, p):
    e = esat(T)
    return EPS * e / (p - (1 - EPS) * e)


def thetae(T, p, q):  # Bolton 1980, q specific humidity
    r = q / (1 - q)
    e = np.maximum(p * r / (EPS + r), 1e-3)
    TL = 2840.0 / (3.5 * np.log(T) - np.log(e / 100.0) - 4.805) + 55.0
    return T * (1e5 / p) ** (0.2854 * (1 - 0.28 * r)) * np.exp((3376.0 / TL - 2.54) * r * (1 + 0.81 * r))


def cape_cols(T, p, q, z):
    """T,p,q,z [L, N] bottom-first. Most-unstable parcel in lowest 180 hPa. MUCAPE, CIN, parcel level."""
    L, N = T.shape
    the = thetae(T, p, q)
    psfc = p[0]
    ok = p >= psfc[None] - 18000.0
    th_m = np.where(ok, the, -1)
    k0 = np.argmax(th_m, axis=0)
    idx = np.arange(N)
    thp = th_m[k0, idx]; pp0 = p[k0, idx]; Tp0 = T[k0, idx]; qp0 = q[k0, idx]
    thd = Tp0 * (1e5 / pp0) ** 0.2854
    cape = np.zeros(N); cin = np.zeros(N)
    Tv = lambda TT, qq: TT * (1 + 0.608 * qq)
    for k in range(L):
        above = k >= k0
        pk = p[k]
        Tdry = thd * (pk / 1e5) ** 0.2854
        Tm = Tdry.copy()
        for _ in range(25):
            f = thetae(Tm, pk, qsat(Tm, pk)) - thp
            f2 = thetae(Tm + 0.1, pk, qsat(Tm + 0.1, pk)) - thp
            den = np.where(np.abs(f2 - f) < 1e-6, 1e-6, f2 - f)
            Tm = np.clip(Tm - f * 0.1 / den, 150.0, 350.0)
        sat = qsat(Tdry, pk) <= qp0
        Tpk = np.where(sat, Tm, Tdry)
        qpk = np.where(sat, qsat(Tpk, pk), qp0)
        b = 9.81 * (Tv(Tpk, qpk) - Tv(T[k], q[k])) / Tv(T[k], q[k])
        if k + 1 < L:
            dz = z[k + 1] - z[k]
            cape += np.where(above & (b > 0), b * dz, 0)
            cin += np.where(above & (b < 0) & (cape == 0), b * dz, 0)
    return cape, cin, k0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist"); ap.add_argument("--nam", default="")
    ap.add_argument("--pts", default="33,-94;35,-92;37,-91;34,-97;38,-94;31,-92")
    ap.add_argument("--levels", default="0,1,2,3,5,8,10,14,18,22,26,30,35,40")
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    d = netCDF4.Dataset(a.hist)
    lat = np.asarray(d["XLAT"][0]); lon = np.asarray(d["XLONG"][0])
    T = np.asarray(d["T"][0]); P = np.asarray(d["PRES"][0]); Q = np.asarray(d["QVAPOR"][0]); PH = np.asarray(d["PH"][0])
    print("T range", float(T.min()), float(T.max()), "P lev0", float(P[0].mean()), "lev-1", float(P[-1].mean()),
          "PH0", float(PH[0].mean()), "PH-1", float(PH[-1].mean()), "PH units", getattr(d["PH"], "units", "?"))
    print("T attrs", getattr(d["T"], "units", "?"), getattr(d["T"], "description", "?"))
    if T.min() < 100:   # WRF convention: perturbation potential temperature (theta - 300)
        T = (T + 300.0) * (P / 1e5) ** (287.04 / 1004.6)
    if P[0].mean() < P[-1].mean():
        T = T[::-1]; P = P[::-1]; Q = Q[::-1]; PH = PH[::-1]
    zmid = 0.5 * (PH[:-1] + PH[1:])
    if np.nanmean(zmid[-1]) > 1e5:
        zmid = zmid / 9.81
    box = (lat > 30) & (lat < 40) & (lon > -98) & (lon < -86)
    jj, ii = np.nonzero(box)
    jj, ii = jj[::3], ii[::3]
    f8 = lambda x: x.astype(np.float64)
    cape, cin, k0 = cape_cols(f8(T[:, jj, ii]), f8(P[:, jj, ii]), f8(Q[:, jj, ii]), f8(zmid[:, jj, ii]))
    print(f"OURS box 30-40N 98-86W: n={len(cape)} MUCAPE mean {cape.mean():.0f} p90 {np.percentile(cape,90):.0f} "
          f"max {cape.max():.0f}; frac>500 {np.mean(cape>500):.2f} frac>1000 {np.mean(cape>1000):.2f}")
    levs_print = [int(x) for x in a.levels.split(",")]
    for s in a.pts.split(";"):
        la, lo = map(float, s.split(","))
        j, i = np.unravel_index(np.argmin((lat - la) ** 2 + ((lon - lo) * np.cos(np.radians(la))) ** 2), lat.shape)
        c, ci, kk = cape_cols(f8(T[:, j:j+1, i]), f8(P[:, j:j+1, i]), f8(Q[:, j:j+1, i]), f8(zmid[:, j:j+1, i]))
        print(f"OURS pt {la},{lo} (j{j} i{i}) MUCAPE {c[0]:.0f} CIN {ci[0]:.0f} parcel lev {kk[0]} psfc {P[0,j,i]/100:.0f}")
        for k in levs_print:
            p_ = P[k, j, i]; t_ = T[k, j, i]; q_ = Q[k, j, i]
            print(f"   k{k:2d} p {p_/100:7.1f} T {t_-273.15:6.1f}C q {q_*1e3:5.2f} RH {q_/qsat(t_,p_):4.2f} z {zmid[k,j,i]:7.0f}")

    if a.nam:
        inv = inventory(a.nam)

        def find(d_, c_, p_, lt, lv=None):
            for r in inv:
                if (int(r["discipline"]), int(r["category"]), int(r["parameter"]), int(r["level_type"])) == (d_, c_, p_, lt) \
                   and (lv is None or float(r["level_value"]) == lv):
                    return int(r["index"])
            return None
        levs = [100000, 97500, 95000, 92500, 90000, 85000, 80000, 75000, 70000, 60000, 50000, 40000, 30000, 25000]
        want = {"cape": find(0, 7, 6, 1), "mucape": find(0, 7, 6, 108, 18000), "psfc": find(0, 3, 0, 1),
                "t2": find(0, 0, 0, 103, 2), "q2": find(0, 1, 0, 103, 2)}
        for L in levs:
            want[f"t{L}"] = find(0, 0, 0, 100, L); want[f"q{L}"] = find(0, 1, 0, 100, L); want[f"rh{L}"] = find(0, 1, 1, 100, L)
        flds = dump(a.nam, sorted(set(v for v in want.values() if v is not None)), "nam")
        g = lambda k, j, i: (flds[want[k]][j, i] if want.get(k) is not None else np.nan)
        ny, nx = next(iter(flds.values())).shape
        nlat, nlon = lcc218(nx, ny)
        nlon = np.where(nlon > 180, nlon - 360, nlon)
        nb = (nlat > 30) & (nlat < 40) & (nlon > -98) & (nlon < -86)
        for k in ("cape", "mucape"):
            if want[k] is not None:
                f = flds[want[k]][nb]
                print(f"NAM {k} box: mean {f.mean():.0f} p90 {np.percentile(f,90):.0f} max {f.max():.0f} "
                      f"frac>500 {np.mean(f>500):.2f} frac>1000 {np.mean(f>1000):.2f}")
        for s in a.pts.split(";"):
            la, lo = map(float, s.split(","))
            j, i = np.unravel_index(np.argmin((nlat - la) ** 2 + ((nlon - lo) * np.cos(np.radians(la))) ** 2), nlat.shape)
            print(f"NAM pt {la},{lo} sbcape {g('cape',j,i):.0f} mucape {g('mucape',j,i):.0f} psfc {g('psfc',j,i)/100:.0f} "
                  f"T2 {g('t2',j,i)-273.15:.1f}C q2 {g('q2',j,i)*1e3:.2f}")
            for L in levs:
                print(f"   p {L/100:6.0f} T {g(f't{L}',j,i)-273.15:6.1f}C q {g(f'q{L}',j,i)*1e3:5.2f} RH {g(f'rh{L}',j,i):5.1f}")


if __name__ == "__main__":
    main()
