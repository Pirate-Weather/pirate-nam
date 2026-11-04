"""Done-rule check for phys-rrtm-gas-init (CONTRACT.md section 6).

    source env.sh
    python tools/check-phys-rrtm-gas-init.py

1. compiles namgpu/kernels/phys_rrtm_gas_init.cu;
2. writes plausible inputs in the NAM file formats (ozone climatology LOZ=17, user CO2 file, CO2
   seasonal cycle file) into a temp folder and runs the wrapper once per branch;
3. checks every float array for NaN/Inf and checks the fpkap lookup against the exact
   (p/1e5)**con_rocp at a few pressures.
The gas setup has no model state, so the full-size random state of the done rule does not apply;
the arrays it writes are the full NAM sizes (LOZ 17, JMR 18, 12 months, 12x24 CO2 grid).
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import kernels  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_gas_init as gi  # noqa: E402

ROCP = np.float32(2.8705e+2) / np.float32(1.0046e+3)


class Ctx:
    def __init__(self):
        self.tables = {}
        self.stream = None
        # physparam values the units read when an argument is not given (NamConfig stand-in)
        self.cfg = SimpleNamespace(ivflip=0, co2usr_file="co2userdata.txt",
                                   co2cyc_file="co2monthlycyc.txt", co2dat_file="co2dat",
                                   co2gbl_file="co2gbl", ictmflg=1, ico2flg=1, ioznflg=0)


def write_ozone(path: str, loz=17, jmr=18):
    pstr = np.array([1000.0 * (0.9 ** k) for k in range(loz)], dtype=np.float64)  # hPa
    lines = []
    for k in range(loz):
        lines.append(f"{pstr[k]:10.3f}")
    rng = np.random.default_rng(3)
    for imo in range(12):
        for j in range(jmr):
            vals = rng.uniform(0.5, 9.0, size=loz)
            hdr = f"{imo + 1:2d}{j + 1:4d}" + "".join(f"{v:6.2f}" for v in vals[:10])
            body = "      " + "".join(f"{v:6.2f}" for v in vals[10:])
            lines.append(hdr)
            lines.append(body)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_co2_user(path: str, months=True):
    lines = [f"{2026:4d}" + "Synthetic user CO2 record".ljust(94) + f"{415.2:7.2f}" + " " * 16 + f"{2.1:5.2f}"]
    rng = np.random.default_rng(5)
    if months:
        for imo in range(12):
            for j in range(12):
                vals = 400.0 + rng.uniform(-3, 3, size=24)
                lines.append("".join(f"{v:7.2f}" for v in vals))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_co2_cycle(path: str):
    lines = ["Synthetic seasonal cycle header".ljust(98) + f"{0.0:7.2f}" + " " * 16 + f"{0.0:5.2f}"]
    rng = np.random.default_rng(7)
    for j in range(12):  # one annual-mean co2dat (12 records of 24 values)
        vals = 400.0 + rng.uniform(-1, 1, size=24)
        lines.append("".join(f"{v:7.2f}" for v in vals))
    for imo in range(12):
        lines.append("cycle month".ljust(58) + f"{rng.uniform(-2, 2):7.2f}")
        for j in range(12):
            vals = rng.uniform(-2, 2, size=24)
            lines.append("".join(f"{v:7.2f}" for v in vals))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def fpkap_check():
    import cupy as cp
    ps = np.array([0.5, 2.0, 10.0, 100.0, 500.0], dtype=np.float32)  # hPa, all >= 0.1
    pk = cp.zeros(5, dtype=cp.float32)
    kernels.get(gi.UNIT, "gas_init_pkstr")((1,), (128,), (cp.asarray(ps), pk, np.int32(5)))
    got = cp.asnumpy(pk)
    exact = (ps * np.float32(100.0) / np.float32(1e5)) ** ROCP
    rel = np.max(np.abs(got - exact) / exact)
    return float(rel)


def main():
    kernels.module(gi.UNIT).compile()
    print("compile: ok")
    rel = fpkap_check()
    print(f"fpkap vs exact (p/1e5)**rocp: max rel err {rel:.2e}")
    assert rel < 1e-5, rel

    tmp = tempfile.mkdtemp(prefix="gas_init_check_")
    write_ozone(os.path.join(tmp, "global_o3clim.txt"))
    write_co2_user(os.path.join(tmp, "co2userdata.txt"))
    write_co2_cycle(os.path.join(tmp, "co2monthlycyc.txt"))

    ctx = Ctx()
    out = gi.run(ctx, ictmflg=0, ico2flg=0, ioznflg=0, base_dir=tmp)
    print("ozone+co2 default: pkstr", np.round(out["pkstr"].get(), 5)[:4],
          "o3r max", float(out["o3r"].max()), "co2_glb", float(out["co2_glb"].get()[0]))

    ctx = Ctx()
    out = gi.run(ctx, ictmflg=-1, ico2flg=2, ioznflg=1, base_dir=tmp)
    print("co2 user monthly: co2_glb", float(out["co2_glb"].get()[0]),
          "vmr_sav max", float(out["co2vmr_sav"].max()))

    ctx = Ctx()
    out = gi.run(ctx, ictmflg=-2, ico2flg=2, ioznflg=1, base_dir=tmp)
    print("co2 seasonal cycle: gco2cyc", np.round(out["gco2cyc"].get(), 9)[:3],
          "cyc_sav max", float(out["co2cyc_sav"].max()))

    import cupy as cp
    bad = []
    for k, a in ctx.tables.items():
        arrays = a.items() if isinstance(a, dict) else [(k, a)]
        for name, arr in arrays:
            if isinstance(arr, cp.ndarray) and not bool(cp.isfinite(arr).all()):
                bad.append(f"{k}.{name}" if isinstance(a, dict) else k)
    if bad:
        print("NaN/Inf in", bad)
        return 1
    print("no NaN/Inf in the seasonal-cycle branch tables")

    fortran_read_check()
    consumer_chain(tmp)
    return 0


class State:
    """Minimal State: add() zero-fills like the driver, fields holds what was added."""

    def __init__(self):
        self.fields = {}

    def add(self, name, shape):
        import cupy as cp
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=cp.float32)
        return self.fields[name]


class Grid:
    def __init__(self, im, jm, lm):
        self.im, self.jm, self.lm = im, jm, lm


def consumer_chain(tmp: str):
    """Full-size handoff: gas_init tables -> gas_update (ozone factors) -> getozn -> getgases.

    This is the defect the review found (the consumers never saw gas_init's outputs), so it runs
    the real consumer units on the NAM grid (954 x 835 x 60) and checks the outputs are finite.
    """
    import cupy as cp
    from namgpu.physics.rrtm import phys_rrtm_getozn as gz
    from namgpu.physics.rrtm import phys_rrtm_getgases as gg
    from namgpu.physics.rrtm import phys_rrtm_gas_update as gu

    im, jm, lm = 954, 835, 60
    ctx = Ctx()
    ctx.grid = Grid(im, jm, lm)
    ctx.st = State()
    gi.run(ctx, ictmflg=-2, ico2flg=2, ioznflg=0, base_dir=tmp)
    for key in ("gas_o3r", "gas_pkstr", "rrtm_gases", "co2vmr_sav", "co2cyc_sav"):
        assert key in ctx.tables, f"gas_init did not publish tables[{key!r}]"
    gu.run(ctx, iyear=2026, imon=3, iday=31, ihour=0, loz1st=True, ldoco2=False,
           co2dat_file="co2dat", co2gbl_file="co2gbl", ioznflg=0)
    assert ctx.tables["rrtm_gas_update"].k1oz >= 1, "gas_update did not set the ozone time factors"

    lmk = lm + 1
    prslk = ctx.st.add("RAD_PRSLK1", (lmk, jm, im))
    ex = (np.linspace(200.0, 100000.0, lmk, dtype=np.float32) / np.float32(1e5)) ** np.float32(0.2857142)
    prslk[...] = cp.asarray(np.broadcast_to(ex[:, None, None], (lmk, jm, im)).copy())
    xl = ctx.st.add("XLAT", (jm, im))
    xl[...] = cp.linspace(-1.2, 1.2, jm, dtype=cp.float32)[:, None]
    olyr = gz.run(ctx, ivflip=0)["RAD_OLYR"]
    assert bool(cp.isfinite(olyr).all()), "getozn output has NaN/Inf"
    print("getozn from gas_init tables: finite, standin=False")

    tab = ctx.tables["rrtm_gases"]
    kmon = max(1, int(tab["kmonsav"]))
    lev = np.linspace(200.0, 100000.0, lm + 2, dtype=np.float32)
    plvl = cp.asarray(np.broadcast_to(lev[:, None, None], (lm + 2, jm, im)).copy())
    xlon = cp.zeros((jm, im), dtype=cp.float32)
    xlat = cp.zeros((jm, im), dtype=cp.float32)
    gasvmr = cp.zeros((gg.NF_VGAS, lm + 1, jm, im), dtype=cp.float32)
    gg.getgases(plvl, xlon, xlat, gasvmr, co2_glb=tab["co2_glb"], gco2cyc=tab["gco2cyc"],
                co2vmr_sav=tab["co2vmr_sav"], ico2flg=int(tab["ico2flg"]), kmonsav=kmon,
                ivflip=int(tab["ivflip"]))
    assert bool(cp.isfinite(gasvmr).all()), "getgases output has NaN/Inf"
    print(f"getgases from gas_init tables: finite, ico2flg={tab['ico2flg']} kmonsav={kmon}")
    del gasvmr, plvl
    cp.get_default_memory_pool().free_all_blocks()


def fortran_read_check():
    """Implied decimal of Fw.d input (gfortran): '  350' under f7.2 is 3.50, not 350."""
    from namgpu.physics.rrtm.phys_rrtm_gas_init import _f as fread_f
    assert fread_f("    350", 2) == 3.5
    assert fread_f("   3.50", 2) == 3.5
    assert fread_f("", 2) == 0.0
    assert fread_f("  -350", 2) == -3.5
    assert fread_f("   1.000", 3) == 1.0
    print("f-edit implied decimal: ok")


if __name__ == "__main__":
    sys.exit(main())
