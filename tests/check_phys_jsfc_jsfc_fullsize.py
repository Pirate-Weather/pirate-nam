"""Full-size check for phys-jsfc-jsfc (NAM CONUS grid 954 x 835 x 60): one run, then
  - every B1-tile column of every output is finite and was written (not the -999 sentinel),
  - the outer ring (columns outside ITS_B1..ITE_B1 / JTS_B1..JTE_B1) is untouched: outputs keep the
    sentinel and INOUT state equals its pre-run value.
Run from the repository root (source env.sh first): python tests/check_phys_jsfc_jsfc_fullsize.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from namgpu.physics.jsfc import phys_jsfc_init, phys_jsfc_jsfc, phys_jsfc_sfcdif  # noqa: E402

IM, JM, LM = 954, 835, 60
SENT = np.float32(-999.0)


class _St:
    def __init__(self):
        self.fields = {}

    def __getitem__(self, name):
        return self.fields[name]

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def build(seed=3):
    rng = np.random.default_rng(seed)
    st = _St()
    f = st.fields

    def f2(lo, hi):
        return cp.asarray((lo + (hi - lo) * rng.random((JM, IM))).astype(np.float32))

    def f3(lo, hi):
        return cp.asarray((lo + (hi - lo) * rng.random((LM, JM, IM))).astype(np.float32))

    pint = np.linspace(2.0e4, 1.02e5, LM + 1, dtype=np.float32)[:, None, None] * np.ones((LM + 1, JM, IM), np.float32)
    f["PHINT"] = cp.asarray(pint)
    f["PHMID"] = cp.asarray(0.5 * (pint[:-1] + pint[1:]))
    f["SFCZ"] = f2(0.0, 200.0)
    f["TSFC"] = f2(270.0, 290.0)
    f["XLAND"] = cp.asarray(rng.integers(1, 3, size=(JM, IM)).astype(np.float32))
    f["MAVAIL"] = cp.ones((JM, IM), dtype=cp.float32)
    f["Z0BASE"] = f2(0.05, 0.15)
    f["VEGFRC"] = f2(0.0, 1.0)
    f["SNOWC"] = cp.zeros((JM, IM), dtype=cp.float32)
    f["EPSQ2"] = cp.full((LM,), 0.02, dtype=cp.float32)
    f["DZ"] = f3(50.0, 450.0)
    f["TH"] = f3(290.0, 300.0)
    f["T"] = f3(270.0, 280.0)
    f["Q"] = f3(1.0e-3, 6.0e-3)
    f["QC"] = f3(0.0, 1.0e-5)
    f["U_PHY"] = f3(-10.0, 10.0)
    f["V_PHY"] = f3(-10.0, 10.0)
    f["Q2"] = f3(0.0, 0.5)
    f["THZ0"] = f2(275.0, 290.0)
    f["QSH"] = f2(5.0e-3, 6.0e-3)
    f["UZ0"] = cp.zeros((JM, IM), dtype=cp.float32)
    f["VZ0"] = cp.zeros((JM, IM), dtype=cp.float32)
    f["QZ0"] = f2(5.0e-3, 6.0e-3)
    f["USTAR"] = f2(0.2, 0.3)
    f["Z0"] = f2(0.01, 0.1)
    f["AKHS"] = f2(0.01, 0.02)
    f["AKMS"] = f2(0.01, 0.02)
    f["PBLH"] = f2(300.0, 1500.0)
    f["RIB"] = f2(-0.1, 0.1)
    f["RMOL"] = cp.zeros((JM, IM), dtype=cp.float32)
    return st


def main():
    st = build()
    ctx = SimpleNamespace(st=st, grid=SimpleNamespace(im=IM, jm=JM, lm=LM), tables={}, stream=None)
    ctx.tables["NMMB_LIMITS"] = phys_jsfc_sfcdif._full_domain_limits(IM, JM)
    phys_jsfc_init.run(ctx)                              # psim/psih tables, scal, USTAR cold start
    f = st.fields

    inout = [n for n in phys_jsfc_jsfc.INOUT_STATE.values()]
    outs = [n for n in list(phys_jsfc_jsfc.SFCDIF_OUT_STATE.values()) + list(phys_jsfc_jsfc.POST_OUT_STATE.values()) if n not in inout]
    before = {n: cp.asnumpy(st.add(n, (JM, IM))).copy() for n in inout}
    for n in outs:
        st.add(n, (JM, IM))
        f[n].fill(SENT)

    res = phys_jsfc_jsfc.run(ctx, ntsd=1)
    cp.cuda.Stream.null.synchronize()

    tile = np.zeros((JM, IM), dtype=bool)
    tile[1:-1, 1:-1] = True                              # 1-based 2..IM-1, 2..JM-1
    bad = {}
    ring_bad = {}
    for n in outs:
        a = cp.asnumpy(res[n])
        bad[n] = int(np.count_nonzero(~np.isfinite(a[tile]))) + int(np.count_nonzero(a[tile] == SENT))
        ring_bad[n] = int(np.count_nonzero(a[~tile] != SENT))
    for n in inout:
        ring_bad[n] = int(np.count_nonzero(cp.asnumpy(res[n])[~tile] != before[n][~tile]))
        bad[n] = int(np.count_nonzero(~np.isfinite(cp.asnumpy(res[n])[tile])))

    print("tile columns:", int(tile.sum()), "of", IM * JM)
    print("non-finite or unwritten in tile, by field:", {k: v for k, v in bad.items() if v} or "none")
    print("changed outside tile, by field:", {k: v for k, v in ring_bad.items() if v} or "none")
    ok = not any(bad.values()) and not any(ring_bad.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
