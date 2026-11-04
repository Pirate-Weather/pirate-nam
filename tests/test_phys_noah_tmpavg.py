"""Check phys-noah-tmpavg: the eight TMPAVG branches against a float64 scalar transcription of
module_LS_NOAHLSM.F90:5095-5197, and the SFLX interface-temperature contract (TUP/TDN).

Run: python tests/test_phys_noah_tmpavg.py   (needs a GPU; prints PASS or FAIL)
"""
from __future__ import annotations

import sys

import cupy as cp
import numpy as np

from namgpu.physics.noah import phys_noah_tmpavg as m

T0 = 273.15


def tmpavg_reference(tup, tm, tdn, zsoil, k):
    """Literal float64 transcription of TMPAVG (K is 1-based)."""
    dz = -zsoil[0] if k == 1 else zsoil[k - 2] - zsoil[k - 1]
    dzh = dz * 0.5
    if tup < T0:
        if tm < T0:
            if tdn < T0:
                return (tup + 2.0 * tm + tdn) / 4.0
            x0 = (T0 - tm) * dzh / (tdn - tm)
            return 0.5 * (tup * dzh + tm * (dzh + x0) + T0 * (2.0 * dzh - x0)) / dz
        if tdn < T0:
            xup = (T0 - tup) * dzh / (tm - tup)
            xdn = dzh - (T0 - tm) * dzh / (tdn - tm)
            return 0.5 * (tup * xup + T0 * (2.0 * dz - xup - xdn) + tdn * xdn) / dz
        xup = (T0 - tup) * dzh / (tm - tup)
        return 0.5 * (tup * xup + T0 * (2.0 * dz - xup)) / dz
    if tm < T0:
        if tdn < T0:
            xup = dzh - (T0 - tup) * dzh / (tm - tup)
            return 0.5 * (T0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh) / dz
        xup = dzh - (T0 - tup) * dzh / (tm - tup)
        xdn = (T0 - tm) * dzh / (tdn - tm)
        return 0.5 * (T0 * (2.0 * dz - xup - xdn) + tm * (xup + xdn)) / dz
    if tdn < T0:
        xdn = dzh - (T0 - tm) * dzh / (tdn - tm)
        return (T0 * (dz - xdn) + 0.5 * (T0 + tdn) * xdn) / dz
    return (tup + 2.0 * tm + tdn) / 4.0


# One hand-picked (TUP, TM, TDN) per branch, in the order of the Fortran IF tree:
CASES = [
    (265.0, 266.0, 267.0),   # TUP, TM, TDN < T0
    (265.0, 266.0, 275.0),   # TUP, TM < T0, TDN >= T0
    (265.0, 275.0, 266.0),   # TUP < T0, TM >= T0, TDN < T0
    (265.0, 275.0, 278.0),   # TUP < T0, TM >= T0, TDN >= T0
    (278.0, 266.0, 265.0),   # TUP >= T0, TM < T0, TDN < T0
    (278.0, 266.0, 275.0),   # TUP >= T0, TM < T0, TDN >= T0
    (278.0, 278.0, 265.0),   # TUP >= T0, TM >= T0, TDN < T0
    (278.0, 278.0, 279.0),   # TUP, TM, TDN >= T0
]


def check_branches() -> bool:
    im = len(CASES)
    zsoil = m.zsoil_vector(0)
    tup = np.zeros((m.NSOIL, 1, im), dtype=np.float32)
    tm = np.zeros_like(tup)
    tdn = np.zeros_like(tup)
    for i, (a, b, c) in enumerate(CASES):
        tup[:, 0, i], tm[:, 0, i], tdn[:, 0, i] = a, b, c
    out = cp.zeros_like(cp.asarray(tm))
    m.tmpavg(out, cp.asarray(tup), cp.asarray(tm), cp.asarray(tdn), cp.asarray(zsoil))
    got = cp.asnumpy(out)
    worst = 0.0
    for k in range(1, m.NSOIL + 1):
        for i, (a, b, c) in enumerate(CASES):
            ref = tmpavg_reference(a, b, c, zsoil.astype(np.float64), k)
            worst = max(worst, abs(float(got[k - 1, 0, i]) - ref))
    print(f"branch check: worst |kernel - float64 reference| = {worst:.3e} K over 8 branches x 4 layers")
    return worst < 1e-3


def check_interface_contract() -> bool:
    """TUP[0] = TSURF, TUP[k] = TDN[k-1] (SFLX TBK), TDN[k] = TBND bottom interface of layer k."""
    rng = np.random.default_rng(3)
    jm, im = 2, 3
    stc = cp.asarray(rng.uniform(255.0, 290.0, size=(m.NSOIL, jm, im)).astype(np.float32))
    tsurf = cp.asarray(rng.uniform(255.0, 290.0, size=(jm, im)).astype(np.float32))
    tbot = cp.asarray(rng.uniform(265.0, 285.0, size=(jm, im)).astype(np.float32))
    zs = cp.asarray(m.zsoil_vector(0))
    tup = cp.zeros_like(stc)
    tdn = cp.zeros_like(stc)
    tb = cp.zeros_like(stc)
    m.interface_temps(stc, tsurf, tbot, zs, m.ZBOT_TEST, tup, tdn, tb)
    ok_top = bool(cp.allclose(tup[0], tsurf))
    ok_chain = bool(cp.array_equal(tup[1:], tdn[:-1]))
    # TBND for K = 1: (TU, TB) = (STC1, STC2), ZUP = 0, ZB = ZSOIL(2)
    s = cp.asnumpy(stc).astype(np.float64)
    z = m.zsoil_vector(0).astype(np.float64)
    tb1 = s[0] + (s[1] - s[0]) * (0.0 - z[0]) / (0.0 - z[1])
    ok_tbnd = bool(np.allclose(cp.asnumpy(tdn[0]), tb1, atol=1e-3))
    print(f"interface contract: TUP[0]=TSURF {ok_top}, TUP[k]=TDN[k-1] {ok_chain}, TDN[1]=TBND {ok_tbnd}")
    return ok_top and ok_chain and ok_tbnd


def main() -> int:
    ok = check_branches() and check_interface_contract()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
