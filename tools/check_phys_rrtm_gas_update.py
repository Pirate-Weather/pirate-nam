"""Done-rule check for phys-rrtm-gas-update (CONTRACT section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python tools/check_phys_rrtm_gas_update.py

Compiles the kernel and runs the wrapper once per branch that writes the CO2 field (monthly,
annual mean, historical), on synthetic co2 files written to a temporary folder, under a pool lease.
The state here is the 24 x 12 x 12 CO2 grid and scalars; no domain-size state is involved.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import kernels  # noqa: E402
from namgpu.physics.rrtm import phys_rrtm_gas_update as unit  # noqa: E402
from namgpu.steps.drv_gpu_lease import gpu_lease  # noqa: E402

TEMPLATE = "monthly_co2_data__0000.txt"   # 18 characters, then the year at characters 19:22
HIST = "history_co2.txt"


class Ctx:
    def __init__(self):
        self.tables = {}
        self.stream = None


def write_monthly(year: int, rng) -> None:
    cline = "synthetic monthly CO2 check".ljust(94)
    lines = [f"{year:4d}{cline}{400.0:7.2f}" + " " * 16 + f"{2.0:5.2f}"]
    for _ in range(12):
        for _ in range(12):
            vals = np.round(rng.uniform(380.0, 420.0, unit.IMXCO2), 2)
            lines.append("".join(f"{v:7.2f}" for v in vals))
    with open(unit.co2_file_name(TEMPLATE, year), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_hist() -> None:
    lines = [f"{1950:4d}    {1955:4d}" + "synthetic historical".ljust(48)]
    for y, g1, g2 in [(1955, 320.5, 321.5), (1954, 319.0, 320.0), (1953, 317.25, 318.0),
                      (1952, 316.0, 316.5), (1951, 315.5, 316.0), (1950, 314.0, 315.0)]:
        lines.append(f"{y} {g1:.2f} {g2:.2f}")
    with open(HIST, "w") as f:
        f.write("\n".join(lines) + "\n")


def check(name, ctx) -> None:
    vmr = unit.gases_table(ctx)["co2vmr_sav"].get()
    st = ctx.tables[unit.KEY]
    assert np.isfinite(vmr).all(), f"{name}: non-finite co2vmr_sav"
    assert np.isfinite(st.co2_glb) and np.isfinite(st.facoz), f"{name}: non-finite scalars"
    print(f"{name}: co2vmr min {vmr.min():.4e} max {vmr.max():.4e} co2_glb {float(st.co2_glb):.4e} "
          f"facoz {float(st.facoz):.4f}")


def main() -> None:
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp(prefix="gasupd-")
    os.chdir(tmp)
    write_monthly(2024, rng)
    write_hist()
    with gpu_lease(lane="nam-gpu-phys-rrtm-gas-update", vram_mib=1000, est_s=600,
                   what="phys-rrtm-gas-update compile and three branch runs"):
        import cupy as cp
        kernels.get(unit.UNIT, unit.KERNEL)      # NVRTC compile, on the leased card

        # 1. monthly data (ictmflg 1 -> idyr = iyear = 2024, not historical), kernel mode 0.
        ctx = Ctx()
        unit.run(ctx, iyear=2024, imon=7, iday=10, ihour=0, loz1st=True, ldoco2=True,
                 co2dat_file=TEMPLATE, co2gbl_file=HIST, ioznflg=0, ico2flg=2, ictmflg=1)
        check("monthly", ctx)

        # 2. annual mean plus seasonal cycle (ictmflg -2), kernel mode 1.
        ctx = Ctx()
        unit.gases_table(ctx)["co2cyc_sav"] = cp.asarray(
            rng.uniform(-3e-6, 3e-6, (12, unit.JMXCO2, unit.IMXCO2)).astype(np.float32))
        unit.run(ctx, iyear=2024, imon=7, iday=10, ihour=0, loz1st=True, ldoco2=True,
                 co2dat_file=TEMPLATE, co2gbl_file=HIST, ioznflg=0, ico2flg=2, ictmflg=-2)
        check("annual", ctx)

        # 3. historical semi-yearly global means (idyr 1953 < 1957), kernel mode 2.
        ctx = Ctx()
        unit.run(ctx, iyear=1953, imon=7, iday=10, ihour=0, loz1st=True, ldoco2=True,
                 co2dat_file=TEMPLATE, co2gbl_file=HIST, ioznflg=0, ico2flg=2, ictmflg=19531)
        check("historical", ctx)
        vmr = unit.gases_table(ctx)["co2vmr_sav"].get()
        assert abs(float(vmr[0, 0, 0]) - 317.25e-6) < 1e-10, "historical months 1-6 take g1"
        assert abs(float(vmr[6, 0, 0]) - 318.00e-6) < 1e-10, "historical months 7-12 take g2"
    print("CHECK PASS: phys-rrtm-gas-update compiled and ran on all three branches")


if __name__ == "__main__":
    main()
