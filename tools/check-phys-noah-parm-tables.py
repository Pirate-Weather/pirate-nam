"""Check phys-noah-parm-tables and its consumer phys-noah-redprm (the review's exact issues, no sweep).

1. List-directed read semantics (gfortran rules): a row split across records is read; extra fields are
   dropped; NROTBL '4.0' is refused (INTEGER); D exponents parse; strtof rounding; A4 mismatch refused;
   a repeat count is refused.
2. Soil and general sections parse from the in-tree WRF files with the Fortran READ sequence and the
   values they carry (the WRF files are a stand-in for the NAM fix; this checks the reader, not the NAM values).
3. No fallback: redprm without table bindings and without NAM_NOAH_FIX_DIR raises.
4. Device binding: parm.run binds ctx.tables["noah_parm"]; redprm consumes that same object; one
   full-size 954 x 835 launch on the GPU; every output finite.

The VEGPARM.TBL written for step 4 is a FIXTURE in the NAM column layout (values generated here, not NAM
fix data). It exercises the wiring and the NaN/Inf check only. Its VEG values are not evidence of NAM tables.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.physics.noah import phys_noah_parm_tables as P  # noqa: E402

IN_TREE = ROOT / "namgpu" / "data" / "noah_tables"
OUT = ROOT / "runs" / "check-phys-noah-parm-tables"
OUT.mkdir(parents=True, exist_ok=True)


def expect_refusal(fn, needle):
    try:
        fn()
    except P.NoahParmTableError as e:
        assert needle in str(e), (needle, str(e))
        return str(e)
    raise AssertionError(f"no refusal, wanted {needle!r}")


# ---- 1. list-directed read semantics -------------------------------------------------------------
r = P._Reader(Path(__file__))            # reader only, file is not parsed here
r.lines = ["1 2", "3,4,5", "6 7 8 9 10 11 12 13 14", "tail"]
r.pos = 0
assert r.read(4) == ["1", "2", "3", "4"], r.read  # first read takes 4 items, rest of record 2 dropped
r.pos = 0
assert r.read(11)[-1] == "11"                  # spans three records, then rest of record 3 dropped
assert P._strtof("1.0D-3") == np.float32(1.0e-3)
assert P._strtof("0.1") == np.float32(0.1)
expect_refusal(lambda: P._int("4.0"), "bad integer")
expect_refusal(lambda: P._strtof("1.0x"), "bad real")
try:
    P._tokens("3*1.0")
    raise AssertionError("repeat count accepted")
except P.NoahParmTableError as e:
    assert "repeat count" in str(e)
print("list-directed reads ok (split rows, dropped tail, D exponent, strtof, integer refusal, repeat refusal)")

# ---- 2. soil and general on the in-tree files ----------------------------------------------------
out = P.NoahParmTables(mminlu="IGBP")
P._soil(P._Reader(IN_TREE / "SOILPARM.TBL"), P.MMINSL, out)
P._gen(P._Reader(IN_TREE / "GENPARM.TBL"), out)
out.assert_finite()
assert out.slcats == 19, out.slcats
assert abs(out.soil["BB"][0] - 2.79) < 1e-6 and abs(out.soil["SATDK"][0] - 4.66e-5) < 1e-10
assert out.gen["SBETA_DATA"] == np.float32(-2.0) and out.gen["CSOIL_DATA"] == np.float32(2.0e6)
assert out.gen["SLOPE_DATA"][8] == np.float32(0.0) and out.slpcats == 9
print("soil+gen ok: slcats=%d slpcats=%d SBETA=%g CSOIL=%g" % (
    out.slcats, out.slpcats, out.gen["SBETA_DATA"], out.gen["CSOIL_DATA"]))

expect_refusal(lambda: P.parse_tables(IN_TREE, ivegsrc=1), "Mismatched vegetation tables")

# ---- fixture dir for steps 3-4 (soil/gen copied from the in-tree files; VEG is a labelled FIXTURE) --
FIX = OUT / "fixture-fix-dir"
FIX.mkdir(parents=True, exist_ok=True)
for name in ("SOILPARM.TBL", "GENPARM.TBL"):
    (FIX / name).write_text((IN_TREE / name).read_text())
rng = np.random.default_rng(7)
veg = ["FIXTURE VEGPARM (NAM column layout, generated; not NAM fix data)", "IGBP", "20 1"]
for lc in range(1, 21):
    row = [lc, round(float(rng.uniform(0.1, 0.3)), 3), round(float(rng.uniform(0.01, 1.0)), 3),
           round(float(rng.uniform(0.0, 1.0)), 3), 1 + (lc % 4), round(float(rng.uniform(30, 300)), 1),
           round(float(rng.uniform(30, 100)), 1), round(float(rng.uniform(30, 60)), 1),
           round(float(rng.uniform(0.02, 0.1)), 3), round(float(rng.uniform(0, 6)), 2),
           round(float(rng.uniform(0.3, 0.9)), 2)]
    if lc == 1:      # row 1 split over two records, with an extra trailing field (list-directed continuation)
        veg.append(", ".join(str(x) for x in row[:6]))
        veg.append(" ".join(str(x) for x in row[6:]) + " 99.0")
    else:
        veg.append(", ".join(str(x) for x in row))
veg += ["label", "0.0", "label", "0.5", "label", "0.5", "label", "5000.0", "label", "19"]
(FIX / "VEGPARM.TBL").write_text("\n".join(veg) + "\n")
host = P.parse_tables(FIX, 1)
assert host.lucats == 20 and host.bare == 19
assert host.veg["NROTBL"][0] == 2 and host.veg["NROTBL"][18] == 4  # row 1 reassembled across records
print("VEG fixture parsed through the list-directed reader (row 1 split across records)")

# ---- 3. no fallback -------------------------------------------------------------------------------
os.environ.pop(P.ENV_DIR, None)
ctx0 = types.SimpleNamespace(tables={}, cfg=types.SimpleNamespace(raw={"ivegsrc": 1}), grid=None, st=None)
expect_refusal(lambda: P.run(ctx0), "set NAM_NOAH_FIX_DIR")

# ---- 4. device binding and one full-size launch ---------------------------------------------------
import cupy as cp  # noqa: E402

from namgpu.physics.noah import phys_noah_redprm as R  # noqa: E402

JM, IM = 835, 954


class StubState:
    def __init__(self):
        self.fields = {}
        self.xp = cp

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


st = StubState()
st.fields["VEGTYP"] = cp.asarray(rng.integers(1, 21, size=(JM, IM)).astype(np.int32))
st.fields["SOILTYP"] = cp.asarray(rng.integers(1, 17, size=(JM, IM)).astype(np.int32))
st.fields["SLOPETYP"] = cp.asarray(rng.integers(1, 10, size=(JM, IM)).astype(np.int32))
# caller inputs (greenness, background roughness and albedo), set by the init in NAM
st.fields["SHDFAC"] = cp.asarray(rng.uniform(0.0, 1.0, size=(JM, IM)).astype(np.float32))
st.fields["Z0BRD"] = cp.asarray(rng.uniform(0.01, 0.5, size=(JM, IM)).astype(np.float32))
st.fields["ALBBRD"] = cp.asarray(rng.uniform(0.05, 0.4, size=(JM, IM)).astype(np.float32))
ctx = types.SimpleNamespace(tables={}, cfg=types.SimpleNamespace(raw={"ivegsrc": 1}),
                            grid=types.SimpleNamespace(jm=JM, im=IM), st=st)
os.environ[P.ENV_DIR] = str(FIX)
dev = P.run(ctx)
bound = ctx.tables[P.KEY_DEV]
assert dev is bound and "veg.ALBTBL" in bound and "soil.BB" in bound and "gen.SLOPE_DATA" in bound
outs = R.run(ctx)

# SLDPTH is the run's layer thickness (state, from DZS at NOAH_LSM_INIT). ZSOIL and RTDIS must follow it,
# not the module default. Rerun with a different SLDPTH and check RTDIS(k) = -SLDPTH(k) / ZSOIL(NROOT).
sd_alt = np.array([0.05, 0.25, 0.5, 1.0], dtype=np.float32)
st.fields["SLDPTH"][...] = cp.asarray(sd_alt)
outs = R.run(ctx)
_, zs_alt = R._sldpth_zsoil(cp, tuple(float(x) for x in sd_alt))
nr = outs["NROOT"]
for k in range(P.NSOIL):
    want = cp.where(nr > k, -cp.float32(sd_alt[k]) / zs_alt[nr - 1], cp.float32(0.0))
    assert bool(cp.allclose(outs["RTDIS"][k], want, rtol=1e-6, atol=0.0)), ("RTDIS not bound to SLDPTH", k)
st.fields["SLDPTH"][...] = cp.asarray(np.asarray(R.SLDPTH_DEFAULT, dtype=np.float32))
outs = R.run(ctx)

# a class index outside its table range is refused (the kernel counts it), not clamped into a result
vt_save = st.fields["VEGTYP"][0, 0].item()
st.fields["VEGTYP"][0, 0] = 0
try:
    R.run(ctx)
    raise AssertionError("out-of-range VEGTYP was not refused")
except P.NoahParmTableError as e:
    assert "refused" in str(e), str(e)
st.fields["VEGTYP"][0, 0] = vt_save
outs = R.run(ctx)
assert ctx.tables[P.KEY_DEV] is bound, "redprm must not re-bind or re-parse"
# FRZX = FRZK * (SMCMAX / SMCREF) * ... (module_LS_NOAHLSM.F90:3612). Soil class 14 (WATER) has
# REFSMC = 0.0 in the table, so the Fortran formula itself gives +inf there; water points never use FRZX.
# Any NaN is a defect; an inf is allowed only at SOILTYP 14 and only in FRZX.
water = cp.asarray(st.fields["SOILTYP"]) == 14
nan_names, inf_names = [], []
for name, a in outs.items():
    if a.dtype.kind != "f":
        continue
    if bool(cp.isnan(a).any()):
        nan_names.append(name)
    notfin = ~cp.isfinite(a)
    if bool(notfin.any()):
        if name == "FRZX" and not bool((notfin & ~water).any()):
            inf_names.append(name)
        else:
            nan_names.append(name + "(inf)")
assert not nan_names, nan_names
print("non-finite only where the Fortran formula divides by REFSMC=0 (SOILTYP 14): %s" % inf_names)
nroot = outs["NROOT"]
assert int(nroot.min()) >= 1 and int(nroot.max()) <= P.NSOIL
cp.cuda.Device().synchronize()
print("redprm full-size %dx%d: %d outputs, no NaN; NROOT %d..%d; tables bound once, consumed"
      % (IM, JM, len(outs), int(nroot.min()), int(nroot.max())))
print("PASS check-phys-noah-parm-tables")
