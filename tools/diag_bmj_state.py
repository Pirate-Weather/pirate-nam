"""Run the production BMJ kernel (phys_bmj_bmjdrv_column, tables from phys-bmj-init) on a REAL forecast
state rebuilt from a history file, whole domain, and census what it decides in a box.
Diagnosis tool (which part of BMJ rejects the warm sector); the evidence is a forecast run.
python tools/diag_bmj_state.py --hist scratch/.../wrfout_d01_2025-03-14_18:00:00 [--box 30,40,-98,-86]
"""
import argparse, sys, types
from pathlib import Path
import numpy as np, netCDF4, cupy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from namgpu import kernels  # noqa: E402
from namgpu.physics.bmj import phys_bmj_init, phys_bmj_bmjdrv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hist"); ap.add_argument("--box", default="30,40,-98,-86")
ap.add_argument("--dt", type=float, default=25.0)
a = ap.parse_args()

d = netCDF4.Dataset(a.hist)
lat = np.asarray(d["XLAT"][0]); lon = np.asarray(d["XLONG"][0])
P = np.asarray(d["PRES"][0], np.float32); TH = np.asarray(d["T"][0], np.float32) + 300.0
R = np.asarray(d["QVAPOR"][0], np.float32); PS = np.asarray(d["PSFC"][0], np.float32)
U = np.asarray(d["U"][0], np.float32); V = np.asarray(d["V"][0], np.float32)
if P[0].mean() > P[-1].mean():          # history is bottom-first; the state is top-first (L=1 top)
    P, TH, R, U, V = (x[::-1].copy() for x in (P, TH, R, U, V))
LM, JM, IM = P.shape
CAPPA = np.float32(287.04 / 1004.6)
EX = (P / np.float32(1e5)) ** CAPPA
T = TH * EX
Q = R / (1 + R)
PHINT = np.empty((LM + 1, JM, IM), np.float32)
PHINT[1:LM] = 0.5 * (P[:-1] + P[1:])
PHINT[LM] = PS
PHINT[0] = np.maximum(P[0] - (PHINT[1] - P[0]), 1.0)

st = types.SimpleNamespace(fields={})
def add(name, shape, dtype="float32"):
    if name not in st.fields:
        st.fields[name] = cp.zeros(shape, dtype=dtype)
    return st.fields[name]
st.add = add
st.__getitem__ = lambda self, k: st.fields[k]
class S(dict):
    def add(self, name, shape, dtype="float32"):
        if name not in self:
            self[name] = cp.zeros(shape, dtype=dtype)
        return self[name]
    @property
    def fields(self):
        return self
st = S()
for n, x in (("PHINT", PHINT), ("PHMID", P), ("EXNER", EX), ("Q", Q), ("T", T), ("U_PHY", U), ("V_PHY", V), ("TH", TH)):
    st[n] = cp.asarray(np.ascontiguousarray(x))
st["XLAND"] = cp.ones((JM, IM), cp.float32)          # land everywhere (box is land)
st["KPBL"] = cp.full((JM, IM), LM, cp.int32)
for n in ("RAINCV", "CUTOP", "CUBOT"):
    st.add(n, (JM, IM))
for n in ("DUDT_PHY", "DVDT_PHY", "RTHCUTEN", "RQCUTEN"):
    st.add(n, (LM, JM, IM))
grid = types.SimpleNamespace(im=IM, jm=JM, lm=LM, dxh=np.full(JM, 12000.0, np.float32), dyh=12000.0)
cfg = types.SimpleNamespace(nprecip=2, dt=a.dt, raw={"fres": 0.75, "fr": 1.0, "fsl": 0.85, "fss": 0.85,
                            "entrain": False, "newall": False, "newswap": True, "newupup": False, "nodeep": False})
ctx = types.SimpleNamespace(cfg=cfg, grid=grid, st=st, stream=None, tables={}, ntimestep=0)
phys_bmj_init.run(ctx)
phys_bmj_bmjdrv.run(ctx)
cp.cuda.Device().synchronize()
rain = st["RAINCV"].get(); top = st["CUTOP"].get(); bot = st["CUBOT"].get(); dtdt = st["RTHCUTEN"].get()
la0, la1, lo0, lo1 = map(float, a.box.split(","))
m = (lat > la0) & (lat < la1) & (lon > lo0) & (lon < lo1)
act = np.abs(dtdt).max(axis=0) > 0
print(f"box {a.box}: n {m.sum()} | columns with any T tendency {int((act & m).sum())} | RAINCV>0 {int(((rain > 0) & m).sum())} "
      f"max {rain[m].max():.4f} mm/step | CUBOT<999 {int(((bot < 900) & m).sum())}")
print(f"domain: active {int(act.sum())} RAINCV>0 {int((rain > 0).sum())} max {rain.max():.4f}")
cb = bot[m & act]; ct = top[m & act]
if cb.size:
    print("  active CUBOT (levels from bottom) p10/50/90", np.percentile(cb, [10, 50, 90]), "CUTOP", np.percentile(ct, [10, 50, 90]))
np.savez("/work/nam-gpu/scratch/physconv/bmj_state_out.npz", rain=rain, top=top, bot=bot, act=act)
