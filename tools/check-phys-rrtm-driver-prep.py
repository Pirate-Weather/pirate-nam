"""Done-rule check for phys-rrtm-driver-prep (CONTRACT section 6, parts 1 and 2).

Compiles the kernel file and runs the wrapper ONCE at full size (954 x 835 x 60) with the NAM
12 km flags (icmphys 3, icldflg 1, cld_fraction 0, oper true). Then checks every float output
for NaN/Inf. The AXSD table is the RRTM_INIT GAUSIN table (module_RA_RRTM.F90:1480-1517),
evaluated here in float64 for the test input.

    source env.sh
    python tools/check-phys-rrtm-driver-prep.py
"""
from __future__ import annotations

import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEEDED = {
    "PD", "DSG2", "SGML2", "T", "Q", "QC", "QS", "CUPPT", "HTOP", "HBOT", "CLDFRA",
    "CFRACH", "CFRACL", "CFRACM", "CZMEAN", "SIGT4",
}


def gausin_table() -> np.ndarray:
    """AXSD(I) = GAUSIN(I*DXSD), I = 1..NXSD, DXSD = 0.01, NXSD = 310 (module_RA_RRTM.F90:1424-1517)."""
    rsqr = 1.0 / math.sqrt(2.0 * math.pi)
    out = np.zeros(310, dtype=np.float32)
    for i in range(1, 311):
        xsd = np.float32(i) * np.float32(0.01)
        a1 = xsd * rsqr
        a2 = math.exp(-0.5 * float(xsd) ** 2)
        rn, b1, b2, b3, s = 1.0, 1.0, 1.0, 1.0, 1.0
        while b2 > 1e-3:
            rn += 1.0
            b2 = float(xsd) ** 2 / (2.0 * rn - 1.0)
            b3 = b1 * b2
            s += b3
            b1 = b3
        out[i - 1] = a1 * a2 * s
    return out


def main() -> int:
    import cupy as cp

    from namgpu import config, grid as gridmod, kernels, state as statemod
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_driver_prep as pr

    cfg = config.load()
    g = gridmod.build(cfg)
    st = statemod.State(g, names=NEEDED).randomize(seed=1)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=25.0,
              ntimestep=0, start=datetime.datetime(2023, 3, 31, 12))
    ctx.tables["JDAT"] = [2023, 3, 31, 0, 12, 0, 0, 0]
    ctx.tables["rrtm_axsd"] = cp.asarray(gausin_table())
    print("state fields:", len(st.fields), "bytes:", round(st.nbytes() / 2**30, 2), "GiB", flush=True)

    kernels.module(pr.UNIT).compile()
    print("compiled:", pr.UNIT, flush=True)

    t0 = time.perf_counter()
    with ctx.stream:
        ctl = pr.run(ctx)
    ctx.stream.synchronize()
    dt = time.perf_counter() - t0
    print("wrapper ran in", round(dt, 3), "s; scalars:", ctl, flush=True)

    bad = []
    for name, a in st.fields.items():
        if a.dtype.kind == "f" and name in NEEDED and not bool(cp.isfinite(a).all()):
            bad.append(name)
    out = ctx.tables[pr.OUT_KEY]
    for name, a in out.items():
        if name == "scalars":
            continue
        if a.dtype.kind == "f" and not bool(cp.isfinite(a).all()):
            bad.append("out:" + name)
    print("float outputs with NaN/Inf:", bad or "none", flush=True)
    nclds = out["NCLDS"]
    print("NCLDS min/max:", int(nclds.min()), int(nclds.max()), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
