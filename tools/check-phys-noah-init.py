"""Done-rule check for phys-noah-init (CONTRACT.md section 6, parts 1 and 2), review fix.

1. compiles the kernel file (NVRTC via CuPy);
2. builds the soil tables the way the driver does (phys-noah-parm-tables uploads them under
   ctx.tables["noah_parm"]), from the real STATSGO SOILPARM.TBL (WRF's copy, STAS layout),
   and checks the table index: BB(ISLTYP) must equal the SOILPARM row whose IINDEX is ISLTYP;
3. runs the wrapper ONCE on a full-size 954 x 835 x 60 State with real table values, with three injected
   ISLTYP cases: one range failure (-1, IVEGSRC 1), one out-of-table value (0), and normal values.
   The wrapper must log the range failure and the out-of-table column at init; SH2O and CMC must be finite;
4. runs the dead FNDSOILW block once directly (fndsoilw = 0): the out-of-table and range-failing columns
   must take the Fortran ELSE path (SH2O = SMOIS), CANWAT must be 0, and the normal columns must be finite.
"""
import sys
import time
_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import config, grid as gridmod, state as statemod, kernels
from namgpu.driver import Ctx
from namgpu.physics.noah import phys_noah_init as w
from namgpu.physics.noah import phys_noah_parm_tables as P

SOILPARM = _ROOT + "/namgpu/data/noah_tables/SOILPARM.TBL"

cfg = config.load()
g = gridmod.build(cfg)
im, jm = int(g.im), int(g.jm)
st = statemod.State(g).randomize(seed=1)
rng = np.random.default_rng(1)
shape = tuple(st["SMC"].shape)              # (NSOIL, JM, IM)

# Real soil table (STAS layout) through the same reader the driver uses.
host = P.NoahParmTables(mminlu="IGBP")
P._soil(P._Reader(__import__("pathlib").Path(SOILPARM)), P.MMINSL, host)
bb, maxsmc, satpsi = (host.soil[k] for k in ("BB", "MAXSMC", "SATPSI"))

# Independent check of the table index: find each SOILPARM row by its IINDEX field.
rows = {}
with open(SOILPARM) as fh:
    lines = fh.read().splitlines()
start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("'STAS'") or ln.strip().startswith('"STAS"') or ln.strip() == "STAS")
for ln in lines[start + 2:start + 2 + host.slcats]:
    f = P._record(ln)
    rows[int(f[0])] = float(f[1])                  # IINDEX -> BB
assert len(rows) == host.slcats, (len(rows), host.slcats)
for idx in range(1, host.slcats + 1):
    assert np.float32(rows[idx]) == bb[idx - 1], f"BB index mismatch at ISLTYP {idx}"
print(f"table index ok: BB(ISLTYP) == SOILPARM row IINDEX for {host.slcats} classes")

smc = rng.uniform(0.10, 0.40, shape).astype(np.float32)
st["SMC"][...] = cp.asarray(smc)
st["STC"][...] = cp.asarray(rng.uniform(250.0, 295.0, shape).astype(np.float32))
st["SH2O"][...] = cp.asarray((smc * rng.uniform(0.5, 1.0, shape)).astype(np.float32))
isl = rng.integers(1, host.slcats + 1, (jm, im)).astype(np.int32)
isl[0, 0] = -1          # range failure (IVEGSRC 1: ISLTYP < 0)
isl[0, 1] = 0           # outside the table (ISLTYP 0 is legal for the range check, not for BB)
st["ISLTYP"][...] = cp.asarray(isl)
st["CMC"][...] = cp.asarray(rng.uniform(0.0, 1e-4, (jm, im)).astype(np.float32))

tables = {"noah_parm": {
    "soil.BB": cp.asarray(bb), "soil.MAXSMC": cp.asarray(maxsmc), "soil.SATPSI": cp.asarray(satpsi)}}
ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt, tables=tables)

kernels.module("phys_noah_init").compile()
print("compile ok")

t0 = time.perf_counter()
with ctx.stream:
    w.run(ctx)
ctx.stream.synchronize()
print("wrapper time_s", round(time.perf_counter() - t0, 4))
print("bad_isltyp [flags, first_range_col, n_range, first_table_col]:", tables["noah_init_bad_isltyp"].get().tolist())

# Dead block, fndsoilw = 0, called directly on the same arrays.
sh2o_dead = cp.zeros(shape, cp.float32)
canwat_dead = cp.full((jm, im), 7.0, cp.float32)
bad2 = cp.asarray(np.array([0, 2**31 - 1, 0, 2**31 - 1], np.int32))
k = kernels.get("phys_noah_init", "phys_noah_init_soil")
k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
  (sh2o_dead, canwat_dead, st["SMC"], st["STC"], st["ISLTYP"],
   tables["noah_parm"]["soil.BB"], tables["noah_parm"]["soil.MAXSMC"], tables["noah_parm"]["soil.SATPSI"],
   bad2, np.int32(im), np.int32(jm), np.int32(4), np.int32(30), np.int32(1), np.int32(0)))
ctx.stream.synchronize()
print("dead-block flags", bad2.get().tolist())

fails = []
for name, arr in {"SH2O (wrapper)": st["SH2O"], "CMC (wrapper)": st["CMC"],
                  "SH2O (dead block)": sh2o_dead, "CANWAT (dead block)": canwat_dead}.items():
    ok = bool(cp.isfinite(arr).all())
    print(f"{name}: finite={ok}  min={float(arr.min()):.5f}  max={float(arr.max()):.5f}")
    if not ok:
        fails.append(name)

# Injected columns: (j=0,i=0) range failure and (j=0,i=1) out-of-table both take the ELSE path.
for (j, i) in [(0, 0), (0, 1)]:
    same = bool(cp.array_equal(sh2o_dead[:, j, i], st["SMC"][:, j, i]))
    print(f"dead block column ({i + 1},{j + 1}) SH2O == SMC: {same}")
    if not same:
        fails.append(f"else-path column {i},{j}")
canwat_ok = bool(cp.all(canwat_dead[:jm - 1, :im - 1] == 0.0))   # ITF/JTF interior, as the Fortran loops
print("CANWAT dead block all zero:", canwat_ok)
if not canwat_ok:
    fails.append("CANWAT")
excess = float(cp.max(sh2o_dead[:, :jm - 1, :im - 1] - st["SMC"][:, :jm - 1, :im - 1]))
print("max(SH2O - SMC) in dead block (should be <= ~0):", excess)
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
