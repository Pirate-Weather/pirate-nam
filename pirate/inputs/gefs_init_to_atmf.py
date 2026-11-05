"""Build an approximate GDAS EnKF from GEFS init files, in the netCDF atmf layout GSI's ensemble reader expects.

GEFS v12 starts from 30 of the 80 GDAS EnKF 6-h forecasts, so its init state at time T stands in for
gdas.t(T-6)z.atmf006 from the real EnKF, whose atmf files are not published. Files are written to the
path getbest_EnKF_linkfiles.py searches, so production's script finds them unchanged.

Usage: gefs_init_to_atmf.py YYYYMMDDHH [procs]   (HH = GEFS init time; output goes under the cycle 6 h earlier)
"""
import os, sys, datetime as dt, subprocess, numpy as np, netCDF4 as nc
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree

GEFS_S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
ROOT = "/nam/com/gfs/v16.3"; SCRATCH = "/nam/ens_tmp"
N_MEMBERS = 30                # p01-p30; c00 is the control, not an EnKF member.
NLON, NLAT = 1536, 768        # Same Gaussian grid as the operational C384 EnKF history files.
N_NEIGHBORS = 4
TRACERS = {"sphum": "spfh", "o3mr": "o3mr", "liq_wat": "clwmr", "ice_wat": "icmr"}

def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

def gaussian_grid():
    # N to S, longitudes from 0, matching FV3 write-component output.
    x, _ = np.polynomial.legendre.leggauss(NLAT)
    return np.degrees(np.arcsin(x))[::-1], np.arange(NLON) * 360.0 / NLON

def download(init, mem):
    d = f"{SCRATCH}/{init}/{mem}"; os.makedirs(d, exist_ok=True)
    url = f"{GEFS_S3}/gefs.{init[:8]}/{init[8:]}/atmos/init/{mem}"
    for n in ["gfs_ctrl.nc"] + [f"gfs_data.tile{t}.nc" for t in range(1, 7)]:
        if not os.path.exists(f"{d}/{n}"):
            subprocess.run(["curl", "-sf", "--retry", "4", "--retry-delay", "20", "-m", "1800", "-o", f"{d}/{n}.part", f"{url}/{n}"], check=True)
            os.rename(f"{d}/{n}.part", f"{d}/{n}")
    return d

def weights(d):
    lat, lon = [], []
    for t in range(1, 7):
        with nc.Dataset(f"{d}/gfs_data.tile{t}.nc") as f:
            lat.append(f["geolat"][:].ravel().filled()); lon.append(f["geolon"][:].ravel().filled())
    tree = cKDTree(xyz(np.concatenate(lat), np.concatenate(lon)))
    glat, glon = gaussian_grid(); lo2, la2 = np.meshgrid(glon, glat)
    dist, idx = tree.query(xyz(la2.ravel(), lo2.ravel()), k=N_NEIGHBORS)
    w = 1.0 / np.maximum(dist, 1e-9); w /= w.sum(1, keepdims=True)
    return idx.astype("i4"), w.astype("f4")

def regrid(cells, idx, w):
    return (cells[idx] * w).sum(1).reshape(NLAT, NLON).astype("f4")

def create(path, ctrl_ak, ctrl_bk, cycle, nlev):
    glat, glon = gaussian_grid()
    f = nc.Dataset(path, "w", format="NETCDF4")
    for name, n in [("time", 1), ("pfull", nlev), ("phalf", nlev + 1), ("grid_yt", NLAT), ("grid_xt", NLON)]:
        f.createDimension(name, n)
    f.createVariable("grid_xt", "f8", ("grid_xt",))[:] = glon
    f.createVariable("grid_yt", "f8", ("grid_yt",))[:] = glat
    ph = (ctrl_ak + ctrl_bk * 101325.0) / 100.0
    f.createVariable("phalf", "f4", ("phalf",))[:] = ph
    f.createVariable("pfull", "f4", ("pfull",))[:] = 0.5 * (ph[1:] + ph[:-1])
    t = f.createVariable("time", "f8", ("time",)); t.units = f"hours since {cycle[:4]}-{cycle[4:6]}-{cycle[6:8]} {cycle[8:]}:00:00"; t[:] = 6.0
    f.setncattr("ak", ctrl_ak); f.setncattr("bk", ctrl_bk)
    f.setncattr("source", "Approximation of gdas atmf006, regridded from GEFS init; not an NCEP EnKF product")
    return f

def var3(f, name, level_data=None):
    return f.createVariable(name, "f4", ("time", "pfull", "grid_yt", "grid_xt"), zlib=True, complevel=1, chunksizes=(1, 1, NLAT, NLON))

def convert(a):
    init, cycle, mem, n, idx, w = a
    d = download(init, mem)
    out = f"{ROOT}/enkfgdas.{cycle[:8]}/{cycle[8:]}/atmos/mem{n:03d}"; os.makedirs(out, exist_ok=True)
    path = f"{out}/gdas.t{cycle[8:]}z.atmf006.nc"
    if os.path.exists(path): return path
    with nc.Dataset(f"{d}/gfs_ctrl.nc") as c: ak, bk = c["vcoord"][0].filled(), c["vcoord"][1].filled()
    tiles = [nc.Dataset(f"{d}/gfs_data.tile{t}.nc") for t in range(1, 7)]
    nlev = tiles[0].dimensions["lev"].size
    def cells(fn): return np.concatenate([np.asarray(fn(t)).ravel() for t in tiles])
    f = create(path + ".part", ak, bk, cycle, nlev)
    # Layers run top-down in both formats, so k maps straight across.
    for src, dst in list(TRACERS.items()) + [("t", "tmp")]:
        v = var3(f, dst)
        for k in range(nlev): v[0, k] = regrid(cells(lambda t: t[src][k]), idx, w)
    # chgres writes earth-relative winds on the cell edges; average the four edges to the center.
    vu, vv = var3(f, "ugrd"), var3(f, "vgrd")
    for k in range(nlev):
        vu[0, k] = regrid(cells(lambda t: 0.25 * (t["u_s"][k, :-1, :] + t["u_s"][k, 1:, :] + t["u_w"][k, :, :-1] + t["u_w"][k, :, 1:])), idx, w)
        vv[0, k] = regrid(cells(lambda t: 0.25 * (t["v_s"][k, :-1, :] + t["v_s"][k, 1:, :] + t["v_w"][k, :, :-1] + t["v_w"][k, :, 1:])), idx, w)
    for dst, fn in [("pressfc", lambda t: t["ps"][:]), ("hgtsfc", lambda t: t["zh"][-1])]:
        f.createVariable(dst, "f4", ("time", "grid_yt", "grid_xt"), zlib=True, complevel=1)[0] = regrid(cells(fn), idx, w)
    f.close(); [t.close() for t in tiles]; os.rename(path + ".part", path)
    subprocess.run(["rm", "-rf", d], check=True)
    return path

def ensmean(paths, out, ak, bk, cycle):
    with nc.Dataset(paths[0]) as p: nlev = p.dimensions["pfull"].size
    f = create(out + ".part", ak, bk, cycle, nlev)
    src = [nc.Dataset(p) for p in paths]
    for name in ["tmp", "spfh", "o3mr", "clwmr", "icmr", "ugrd", "vgrd"]:
        v = var3(f, name)
        for k in range(nlev): v[0, k] = np.mean([s[name][0, k] for s in src], axis=0)
    for name in ["pressfc", "hgtsfc"]:
        f.createVariable(name, "f4", ("time", "grid_yt", "grid_xt"), zlib=True, complevel=1)[0] = np.mean([s[name][0] for s in src], axis=0)
    f.close(); [s.close() for s in src]; os.rename(out + ".part", out)

if __name__ == "__main__":
    init = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    t = dt.datetime.strptime(init, "%Y%m%d%H") - dt.timedelta(hours=6); cycle = t.strftime("%Y%m%d%H")
    d0 = download(init, "p01"); idx, w = weights(d0)
    with nc.Dataset(f"{d0}/gfs_ctrl.nc") as c: ak, bk = c["vcoord"][0].filled(), c["vcoord"][1].filled()
    print("weights ready", flush=True)
    with ProcessPoolExecutor(procs) as ex:
        paths = list(ex.map(convert, [(init, cycle, f"p{n:02d}", n, idx, w) for n in range(1, N_MEMBERS + 1)]))
    print(len(paths), "members", flush=True)
    mean = f"{ROOT}/enkfgdas.{cycle[:8]}/{cycle[8:]}/atmos/gdas.t{cycle[8:]}z.atmf006.ensmean.nc"
    ensmean(paths, mean, ak, bk, cycle)
    print("ENS_DONE", cycle, flush=True)
