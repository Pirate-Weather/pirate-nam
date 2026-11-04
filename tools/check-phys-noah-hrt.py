"""Done-rule check for phys-noah-hrt (CONTRACT.md section 6), with the review fixes.

1. compiles the kernels this unit uses (phys_noah_hrt, phys_noah_hrtice, phys_noah_tbnd, phys_noah_tmpavg);
2. no-ctx full-size 954 x 835 x 4 run: no NaN/Inf in RHSTS, AI, BI, CI, SH2O;
3. ctx path, full size, production bindings: a stand-in ctx with the state and SFLX-produced fields,
   the real parm tables (GENPARM ZBOT_DATA = -8.0), ISLTYP gathers, SLDPTH-derived ZSOIL, ICE routing.
   Asserts the bindings equal their sources, sea-ice columns leave SH2O untouched, and the outputs are finite.
Run under the pool lease: ~/gpuwm-work/pool-v3/run.sh --cards 1 --vram-mib 4000 --est 300 ...
"""
import sys, time
from types import SimpleNamespace

_ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import numpy as np
import cupy as cp
from namgpu import kernels
from namgpu.physics.noah import phys_noah_hrt as w
from namgpu.physics.noah import phys_noah_parm_tables as parm

for unit in ("phys_noah_hrt", "phys_noah_hrtice", "phys_noah_tbnd", "phys_noah_tmpavg"):
    kernels.module(unit).compile()
print("compile ok: phys_noah_hrt, phys_noah_hrtice, phys_noah_tbnd, phys_noah_tmpavg")

# ---- 2. no-ctx full-size run
t0 = time.perf_counter()
out = w.run()
cp.cuda.Stream.null.synchronize()
print("no-ctx wrapper time_s", round(time.perf_counter() - t0, 4))
for name, arr in out.items():
    a = arr.get()
    assert np.isfinite(a).all(), f"NaN/Inf in {name} (no-ctx)"
    print("no-ctx", name, tuple(a.shape), "finite")

# ---- 3. ctx path, full size, production bindings
JM, IM = 835, 954
rng = np.random.default_rng(7)
ice = (rng.uniform(size=(JM, IM)) < 0.3).astype(np.int32)
isltyp = rng.integers(1, 13, size=(JM, IM)).astype(np.int32)
smcmax_ref = None
stc = rng.uniform(262.0, 292.0, size=(4, JM, IM)).astype(np.float32)
smc = (rng.uniform(0.05, 0.45, size=(4, JM, IM))).astype(np.float32)
sh2o = (smc * rng.uniform(0.5, 1.0, size=(4, JM, IM))).astype(np.float32)
sfctmp = rng.uniform(260.0, 300.0, size=(JM, IM)).astype(np.float32)
tbot = np.where(ice == 1, np.float32(271.16), rng.uniform(270.0, 290.0, size=(JM, IM))).astype(np.float32)
sflx = {
    "ICE": cp.asarray(ice),
    "VEGTYP": cp.asarray(rng.integers(1, 21, size=(JM, IM)).astype(np.int32)),
    "YY": cp.asarray(rng.uniform(265.0, 295.0, size=(JM, IM)).astype(np.float32)),
    "ZZ1": cp.asarray(rng.uniform(1.0, 1.5, size=(JM, IM)).astype(np.float32)),
    "DF1": cp.asarray(rng.uniform(0.5, 3.0, size=(JM, IM)).astype(np.float32)),
    "SFCTMP": cp.asarray(sfctmp),
    "TH2": cp.asarray(rng.uniform(260.0, 300.0, size=(JM, IM)).astype(np.float32)),
    "FDOWN": cp.asarray(rng.uniform(200.0, 400.0, size=(JM, IM)).astype(np.float32)),
    "EMISSI": cp.asarray(np.full((JM, IM), 0.95, dtype=np.float32)),
    "T24": cp.asarray((sfctmp ** 4).astype(np.float32)),
    "RCH": cp.asarray(rng.uniform(0.002, 0.006, size=(JM, IM)).astype(np.float32)),
    "RR": cp.asarray(rng.uniform(1.0, 2.0, size=(JM, IM)).astype(np.float32)),
    "BETA": cp.asarray(rng.uniform(0.0, 1.0, size=(JM, IM)).astype(np.float32)),
    "EPSCA": cp.asarray(rng.uniform(-5.0, 5.0, size=(JM, IM)).astype(np.float32)),
}
sldpth = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)   # DZSOIL (NOAH_LSM_INIT)
st = SimpleNamespace(fields={
    "STC": cp.asarray(stc), "SMC": cp.asarray(smc), "SH2O": cp.asarray(sh2o),
    "ISLTYP": cp.asarray(isltyp), "SLDPTH": cp.asarray(sldpth), "TBOT": cp.asarray(tbot),
})
ctx = SimpleNamespace(st=st, tables={}, sflx=sflx, cfg=SimpleNamespace(raw={"ivegsrc": 1}))
sh2o_in = sh2o.copy()

# parm tables: the real GENPARM/SOILPARM from namgpu/data/noah_tables (ZBOT_DATA must be -8.0)
parm.run(ctx)
host = ctx.tables[parm.KEY_HOST]
assert float(host.gen["ZBOT_DATA"]) == -8.0, host.gen["ZBOT_DATA"]
print("GENPARM ZBOT_DATA", float(host.gen["ZBOT_DATA"]), "CSOIL_DATA", float(host.gen["CSOIL_DATA"]))

t0 = time.perf_counter()
res = w.run(ctx)
cp.cuda.Stream.null.synchronize()
print("ctx wrapper time_s", round(time.perf_counter() - t0, 4))

sc = w._scratch_arrays(JM, IM)
# bindings equal their sources
idx = isltyp - 1
host_tabs = {}
for k, v in parm.run(ctx).items():
    if k.startswith("soil."):
        host_tabs[k.split(".", 1)[1]] = v.get() if hasattr(v, "get") else np.asarray(v)
assert np.array_equal(sc["SMCMAX"].get(), host_tabs["MAXSMC"][idx]), "SMCMAX binding"
assert np.array_equal(sc["PSISAT"].get(), host_tabs["SATPSI"][idx]), "PSISAT binding"
assert np.array_equal(sc["BEXP"].get(), host_tabs["BB"][idx]), "BEXP binding"
assert np.array_equal(sc["QUARTZ"].get(), host_tabs["QTZ"][idx]), "QUARTZ binding"
assert np.all(sc["ZBOT"].get() == np.float32(-8.0)), "ZBOT binding"
assert np.all(sc["CSOIL"].get() == np.float32(host.gen["CSOIL_DATA"])), "CSOIL binding"
assert np.array_equal(sc["ZSOIL"].get(), np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float32)), "ZSOIL"
print("bindings: SMCMAX, PSISAT, BEXP, QUARTZ, ZBOT=-8, CSOIL, ZSOIL verified")

# sea-ice routing: HRT must leave SH2O on ICE == 1 columns alone (SFLX 3762-3764)
sh2o_out = res["SH2O"].get()
assert np.array_equal(sh2o_out[:, ice == 1], sh2o_in[:, ice == 1]), "SH2O changed on sea-ice columns"
assert not np.array_equal(sh2o_out[:, ice == 0], sh2o_in[:, ice == 0]), "SH2O not updated on land columns"
print("ICE routing: SH2O unchanged on", int((ice == 1).sum()), "sea-ice columns")

for name in ("RHSTS", "AI", "BI", "CI", "SH2O"):
    a = res[name].get()
    assert np.isfinite(a).all(), f"NaN/Inf in {name} (ctx)"
    print("ctx", name, tuple(a.shape), "finite")
print("PASS")
