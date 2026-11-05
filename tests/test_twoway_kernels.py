"""Concurrent / two-way nest kernels (kernels/nest_bc.cu, kernels/nest_2way.cu) against a plain-Python reference of the
NMMB routines (module_PARENT_CHILD_CPL_COMP.F90 PARENT_UPDATE_CHILD_PSFC, PARENT_UPDATE_CHILD_BNDRY,
PRESSURE_ON_NEST_BNDRY_V, PARENT_2WAY_UPDATE; module_NESTING.F90 GENERATE_2WAY_DATA, CHILD_2WAY_BOOKKEEPING, SPLINE),
on small shapes. The host tables (interp factors, update region) are checked on the CPU alone; the kernel tests need
a CUDA card and skip without one."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from namgpu.nest import twoway as tw

R_D, R_V = 287.04, 461.6
P608 = R_V / R_D - 1.0
HYPER_A = 10000.0 * 0.05 / 0.95


# ------------------------------------------------------------------------------------------- reference (Fortran)

def spline_ref(xold, yold, xnew):
    """module_NESTING.F90 SPLINE with Y2 = 0 on entry (1-based copy of the Fortran)."""
    nold = len(xold)
    X = [None] + list(map(float, xold))
    Y = [None] + list(map(float, yold))
    Y2 = [None] + [0.0] * nold
    P = [None] * (nold + 1)
    Q = [None] * (nold + 1)
    dxl = X[2] - X[1]; dxr = X[3] - X[2]
    dydxl = (Y[2] - Y[1]) / dxl; dydxr = (Y[3] - Y[2]) / dxr
    rtdxc = 0.5 / (dxl + dxr)
    P[1] = rtdxc * (6. * (dydxr - dydxl) - dxl * Y2[1]); Q[1] = -rtdxc * dxr
    for k in range(3, nold):
        dxl = dxr; dydxl = dydxr
        dxr = X[k + 1] - X[k]; dydxr = (Y[k + 1] - Y[k]) / dxr
        dxc = dxl + dxr
        den = 1. / (dxl * Q[k - 2] + dxc + dxc)
        P[k - 1] = den * (6. * (dydxr - dydxl) - dxl * P[k - 2]); Q[k - 1] = -den * dxr
    for k in range(nold - 1, 1, -1):
        Y2[k] = P[k - 1] + Q[k - 1] * Y2[k + 1]
    out = []
    for xk in xnew:
        kold = next((k2 - 1 for k2 in range(2, nold + 1) if X[k2] > xk), None)
        if kold is None:
            out.append(Y[nold]); continue
        k = kold
        dx = X[k + 1] - X[k]; rdx = 1. / dx
        ak = 0.1666667 * rdx * (Y2[k + 1] - Y2[k]); bk = 0.5 * Y2[k]
        ck = rdx * (Y[k + 1] - Y[k]) - 0.1666667 * dx * (Y2[k + 1] + Y2[k] + Y2[k])
        x = xk - X[k]
        out.append(ak * x ** 3 + bk * x * x + ck * x + Y[k])
    return np.array(out)


def vcoord(lm):
    eta = np.linspace(0.0, 1.0, lm + 1) ** 1.3
    sg1 = np.where(eta < 0.3, eta / 0.3, 1.0).astype(np.float32)
    sg2 = np.where(eta < 0.3, 0.0, (eta - 0.3) / 0.7).astype(np.float32)
    pt, pdtop = 200.0, 30000.0 - 200.0
    psgml1 = (0.5 * (sg1[1:] + sg1[:-1]) * pdtop + pt).astype(np.float32)
    sgml2 = (0.5 * (sg2[1:] + sg2[:-1])).astype(np.float32)
    return sg1, sg2, psgml1, sgml2, pt, pdtop


def bil(a, iw, js, w):
    return w[0] * a[js, iw] + w[1] * a[js, iw + 1] + w[2] * a[js + 1, iw] + w[3] * a[js + 1, iw + 1]


def psfc_ref(P, fis_c, iw, js, w, vc):
    sg1, sg2, _, _, pt, pdtop = vc
    lm = P["T"].shape[0]
    pint = np.zeros(lm + 1); phi = np.zeros(lm + 1)
    pint[lm] = bil(P["PD"] + pt, iw, js, w)
    lpb = math.log(pint[lm])
    phi[lm] = bil(P["FIS"], iw, js, w)
    for l in range(lm - 1, -1, -1):
        pint[l] = bil(sg2[l] * P["PD"] + (pdtop * sg1[l] + pt), iw, js, w)
        t = bil(P["T"][l], iw, js, w); q = bil(P["Q"][l], iw, js, w)
        lpt = math.log(pint[l])
        phi[l] = phi[l + 1] + R_D * t * (1 + P608 * q) * (lpb - lpt)
        lpb = lpt
    if fis_c < phi[lm]:
        c1 = (pint[lm] - pint[lm - 1]) / (phi[lm] - phi[lm - 1])
        c2 = (c1 - (pint[lm - 1] - pint[lm - 2]) / (phi[lm - 1] - phi[lm - 2])) / (phi[lm] - phi[lm - 2])
        d = fis_c - phi[lm]
        ps = c2 * d * d + c1 * d + pint[lm]
    else:
        for l in range(lm - 1, -1, -1):
            if fis_c < phi[l]:
                lp1 = math.log(pint[l + 1])
                ps = math.exp(lp1 + (lp1 - math.log(pint[l])) / (phi[l + 1] - phi[l]) * (fis_c - phi[l + 1]))
                break
    return ps - pt


def column_ref(pd_corners, cols, w, pdb, vc, tfix):
    """PARENT_UPDATE_CHILD_BNDRY for one point: pd_corners [4] (parent PD at the 4 corners), cols [4, lm]."""
    sg1, sg2, psgml1, sgml2, pt, pdtop = vc
    lm = cols.shape[1]
    pint = np.zeros(lm + 1); pmid = np.zeros(lm)
    pint[lm] = float(np.dot(w, pd_corners + pt))
    for l in range(lm - 1, -1, -1):
        hi = float(np.dot(w, sg2[l] * pd_corners + (sg1[l] * pdtop + pt)))
        pmid[l] = 0.5 * (hi + pint[l + 1]); pint[l] = hi
    col = w @ cols
    if tfix and col[lm - 1] < col[lm - 2]:
        dp1, dp2, dp3 = pint[lm] - pint[lm - 1], pint[lm - 1] - pint[lm - 2], pint[lm - 2] - pint[lm - 3]
        p1, p2, p3 = col[lm - 1] * dp1, col[lm - 2] * dp2, col[lm - 3] * dp3
        col[lm - 1] = (p1 + p2 + p3) / (dp1 + dp2 + dp3); col[lm - 2] = (p2 + p3) / (dp2 + dp3)
    pch = psgml1 + sgml2 * pdb
    pin, vin = list(pmid), list(col)
    if pch[lm - 1] > pin[lm - 1]:
        delp = pch[lm - 1] - pin[lm - 1]
        c1 = (vin[lm - 1] - vin[lm - 2]) / (pin[lm - 1] - pin[lm - 2])
        vin.append(vin[lm - 1] + c1 * delp * HYPER_A / (delp + HYPER_A)); pin.append(pch[lm - 1])
    return spline_ref(pin, vin, pch)


# ------------------------------------------------------------------------------------------- host tables (CPU)

def test_interp_factors_h_points_land_on_parent_points():
    # nest H point I = 1 + 4k sits on parent H point I_PARENT_START + k: weight 1 on SW
    iw, js, w = tw.interp_factors(10, 7, 4, np.array([0, 4, 8, 2]), np.array([0, 0, 4, 2]), False)
    assert list(iw[:3]) == [9, 10, 11] and list(js[:3]) == [6, 6, 7]
    assert np.allclose(w[:3], [[1, 0, 0, 0]] * 3)
    assert np.allclose(w[3], [0.25, 0.25, 0.25, 0.25])          # half way in both directions


def test_interp_factors_v_points():
    # nest V point (1, 1) is at nest H coords (1.5, 1.5) = parent H coords I_PS + 0.125; parent V index space is the
    # H index minus 0.5: I_PS - 0.375 -> SW parent V index I_PS - 1, weight 0.625 east (and north)
    iw, js, w = tw.interp_factors(10, 7, 4, np.array([0]), np.array([0]), True)
    assert (iw[0], js[0]) == (8, 5)                              # 0-based: parent V (9, 6) 1-based
    assert np.allclose(w[0], [0.375 * 0.375, 0.625 * 0.375, 0.375 * 0.625, 0.625 * 0.625], atol=1e-6)


def test_update_region_even_ratio():
    # nest 25 x 21 (6 x 5 parent cells), ratio 4, N_BLEND 5: start = I_PS + 6/4 = I_PS + 1.5 -> I1 = I_PS + 2;
    # end = I_PS + 6 - 1.5 -> I2 = I_PS + 4
    pi, pj, reg = tw.update_points(10, 7, 4, 25, 21, 5, 40, 40)
    assert reg == (12, 14, 9, 10)
    assert len(pi) == 3 * 2 and (pi[0], pj[0]) == (12, 9) and (pi[1], pj[1]) == (13, 9)


def test_strip_points_layout():
    ic, jc, sd, off, lstr = tw.strip_points(9, 7, 2, False)
    assert len(ic) == 2 * (2 * 9 + 2 * 7)
    s = (sd == 1) & (ic == 3) & (jc == 6)                         # N strip, outermost row is s = 1
    assert off[s][0] == 1 * 9 + 3
    ic, jc, sd, off, lstr = tw.strip_points(9, 7, 2, True)
    assert jc[sd == 1].min() == 7 - 1 - 2 and ic[sd == 3].max() == 9 - 2 and ic[sd == 0].max() == 9 - 2


def test_nest_grid_associated_ian(tmp_path):
    # the proof layout: 12 km parent 3000 x 2600 km at 24N 83W, 3 km nest 1900 x 1400 km on the parent's pole
    from namgpu import config
    from namgpu.nest import domain
    pg = domain.nest_grid(24.0, -83.0, 3000, 12, 2600)
    ng = domain.nest_grid_associated(pg, 24.0, -83.0, 1900, 3, 1400)
    assert (ng["im"], ng["jm"], ng["parent_child_space_ratio"]) == (545, 465, 4)
    assert (ng["tph0d"], ng["tlm0d"]) == (pg["tph0d"], pg["tlm0d"])
    assert abs(ng["wbd"] - (pg["wbd"] + (ng["i_parent_start"] - 1) * pg["dlmd"])) < 1e-6
    domain.write_configfile(tmp_path / "cf", ng, "n", tmp_path / "s.npz")
    c = config.load(tmp_path / "cf")
    assert abs(c.dlmd - 0.0315) < 1e-9 and abs(c.dphd - 0.027) < 1e-9
    assert int(c.raw["i_parent_start"]) == ng["i_parent_start"]
    with pytest.raises(SystemExit, match="does not sit"):
        domain.nest_grid_associated(pg, 24.0, -83.0, 2900, 3, 1400)


def test_spline_ref_reproduces_a_cubic_interior():
    x = np.linspace(100., 1000., 12)
    y = np.sin(x / 300.)
    xn = np.linspace(150., 950., 7)
    assert np.allclose(spline_ref(x, y, xn), np.sin(xn / 300.), atol=2e-3)


# ------------------------------------------------------------------------------------------- kernels (CUDA)

cp = pytest.importorskip("cupy")
try:
    _HAVE_CARD = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    _HAVE_CARD = False
needs_card = pytest.mark.skipif(not _HAVE_CARD, reason="no CUDA card")


class _St(dict):
    def __init__(self, fields, scalars):
        super().__init__(fields)
        self.scalars = scalars
        self.fields = self


def _case(seed=0, lm=10, ratio=4, nxp=6, nyp=5, lnsh=2, lnsv=2):
    rng = np.random.default_rng(seed)
    vc = vcoord(lm)
    sg1, sg2, psgml1, sgml2, pt, pdtop = vc
    imp = jmp = 16
    i_ps, j_ps = 5, 6
    imc, jmc = nxp * ratio + 1, nyp * ratio + 1
    y, x = np.mgrid[0:jmp, 0:imp]
    P = {"PD": (96000. + 800 * np.sin(x / 3.) + 500 * np.cos(y / 4.)).astype(np.float32),
         "FIS": (9.81 * (300 + 250 * np.sin(x / 2.5) * np.cos(y / 3.))).clip(0).astype(np.float32)}
    tl = np.linspace(215., 300., lm).astype(np.float32)
    P["T"] = (tl[:, None, None] + rng.normal(0, 1.0, (lm, jmp, imp))).astype(np.float32)
    P["T"][-1] -= 3.0 * (x % 2)                                    # some surface inversions for the T fix
    P["Q"] = (np.linspace(1e-5, 0.015, lm)[:, None, None] * (1 + 0.1 * rng.random((lm, jmp, imp)))).astype(np.float32)
    P["CW"] = (1e-4 * rng.random((lm, jmp, imp))).astype(np.float32)
    P["U"] = rng.normal(0, 10, (lm, jmp, imp)).astype(np.float32)
    P["V"] = rng.normal(0, 10, (lm, jmp, imp)).astype(np.float32)
    yc, xc = np.mgrid[0:jmc, 0:imc]
    C = {"FIS": (9.81 * (200 + 600 * rng.random((jmc, imc)))).astype(np.float32),
         "PD": (95500. + 300 * rng.random((jmc, imc))).astype(np.float32)}
    for k in ("T", "Q", "CW", "U", "V"):
        C[k] = (rng.normal(0, 1, (lm, jmc, imc)) + (tl[:, None, None] if k == "T" else 0)).astype(np.float32)
    for st in (P, C):
        st.update(SG1=sg1, SG2=sg2, PSGML1=psgml1, SGML2=sgml2)
    bc = {}
    for name, nl, wdt in (("PD", 1, lnsh), ("T", lm, lnsh), ("Q", lm, lnsh), ("CW", lm, lnsh), ("U", lm, lnsv),
                          ("V", lm, lnsv)):
        for s in "SN":
            bc[f"{s}_{name}"] = cp.asarray(rng.normal(0, 1, (2, nl, wdt, imc)).astype(np.float32).squeeze(1)
                                           if nl == 1 else rng.normal(0, 1, (2, nl, wdt, imc)).astype(np.float32))
        for s in "WE":
            bc[f"{s}_{name}"] = cp.asarray(rng.normal(0, 1, (2, nl, jmc, wdt)).astype(np.float32).squeeze(1)
                                           if nl == 1 else rng.normal(0, 1, (2, nl, jmc, wdt)).astype(np.float32))
    mk = lambda d, im, jm: types.SimpleNamespace(  # noqa: E731
        st=_St({k: cp.asarray(v) for k, v in d.items()}, {"PDTOP": 0.0, "PT": 0.0}),   # as on the GFS-start path
        grid=types.SimpleNamespace(im=im, jm=jm, lm=lm, pt=pt, pdtop=pdtop),
        cfg=types.SimpleNamespace(lnsh=lnsh, lnsv=lnsv, dt=6.25), bc=None, stream=None)
    cp_, cc = mk(P, imp, jmp), mk(C, imc, jmc)
    cc.bc = bc
    return cp_, cc, P, C, vc, (i_ps, j_ps, ratio)


@needs_card
def test_boundary_kernels_match_reference():
    ctx_p, ctx_c, P, C, vc, (i_ps, j_ps, ratio) = _case()
    old = {k: cp.asnumpy(v) for k, v in ctx_c.bc.items()}
    cpl = tw.Coupler(ctx_p, ctx_c, ratio, i_ps, j_ps, two_way=False)
    cpl.boundary()
    cp.cuda.Device().synchronize()
    recip = 1.0 / (6.25 * ratio)
    pdb = cp.asnumpy(cpl.pdb)
    # PSFC on every ring point
    ri, rj = cp.asnumpy(cpl.ring[0]), cp.asnumpy(cpl.ring[1])
    iw, js, w = tw.interp_factors(i_ps, j_ps, ratio, ri, rj, False)
    for k in range(0, len(ri), 7):
        ref = psfc_ref(P, float(C["FIS"][rj[k], ri[k]]), iw[k], js[k], w[k], vc)
        assert abs(pdb[rj[k], ri[k]] - ref) < 0.5 + 2e-6 * abs(ref), (k, pdb[rj[k], ri[k]], ref)
    lm = P["T"].shape[0]
    # H strips: PD, T, Q, CW tendencies
    hi, hj, hs, ho, hl = tw.strip_points(ctx_c.grid.im, ctx_c.grid.jm, 2, False)
    iw, js, w = tw.interp_factors(i_ps, j_ps, ratio, hi, hj, False)
    new = {k: cp.asnumpy(v) for k, v in ctx_c.bc.items()}
    for k in range(0, len(hi), 5):
        s = "SNWE"[hs[k]]
        a_old, a_new = old[f"{s}_PD"].reshape(2, -1), new[f"{s}_PD"].reshape(2, -1)
        assert np.isclose(a_new[1, ho[k]], (pdb[hj[k], hi[k]] - a_old[0, ho[k]]) * recip, rtol=1e-5, atol=1e-3)
        corners = np.array([P["PD"][js[k], iw[k]], P["PD"][js[k], iw[k] + 1], P["PD"][js[k] + 1, iw[k]],
                            P["PD"][js[k] + 1, iw[k] + 1]], np.float64)
        for var in ("T", "Q", "CW"):
            cols = np.stack([P[var][:, js[k], iw[k]], P[var][:, js[k], iw[k] + 1], P[var][:, js[k] + 1, iw[k]],
                             P[var][:, js[k] + 1, iw[k] + 1]]).astype(np.float64)
            out = column_ref(corners, cols, w[k].astype(np.float64), float(pdb[hj[k], hi[k]]), vc, var == "T")
            ao, an = old[f"{s}_{var}"].reshape(2, lm, -1), new[f"{s}_{var}"].reshape(2, lm, -1)
            want = (out - ao[0, :, ho[k]]) * recip
            scale = {"T": 1.0, "Q": 1e-4, "CW": 1e-6}[var]
            assert np.allclose(an[1, :, ho[k]], want, rtol=1e-3, atol=2e-3 * scale / 1.0), (var, k)
    # V strips: U, V tendencies (parent PD on V points and the nest PDB on V points are 4-point means)
    vi, vj, vs, vo, vl = tw.strip_points(ctx_c.grid.im, ctx_c.grid.jm, 2, True)
    iw, js, w = tw.interp_factors(i_ps, j_ps, ratio, vi, vj, True)
    pdv = 0.25 * (P["PD"][:-1, :-1] + P["PD"][:-1, 1:] + P["PD"][1:, :-1] + P["PD"][1:, 1:])
    for k in range(0, len(vi), 5):
        s = "SNWE"[vs[k]]
        corners = np.array([pdv[js[k], iw[k]], pdv[js[k], iw[k] + 1], pdv[js[k] + 1, iw[k]], pdv[js[k] + 1, iw[k] + 1]])
        pdbv = 0.25 * (pdb[vj[k], vi[k]] + pdb[vj[k], vi[k] + 1] + pdb[vj[k] + 1, vi[k]] + pdb[vj[k] + 1, vi[k] + 1])
        for var in ("U", "V"):
            cols = np.stack([P[var][:, js[k], iw[k]], P[var][:, js[k], iw[k] + 1], P[var][:, js[k] + 1, iw[k]],
                             P[var][:, js[k] + 1, iw[k] + 1]]).astype(np.float64)
            out = column_ref(corners.astype(np.float64), cols, w[k].astype(np.float64), float(pdbv), vc, False)
            ao, an = old[f"{s}_{var}"].reshape(2, lm, -1), new[f"{s}_{var}"].reshape(2, lm, -1)
            assert np.allclose(an[1, :, vo[k]], (out - ao[0, :, vo[k]]) * recip, rtol=1e-3, atol=2e-3), (var, k)
    # the V padding column/row is never written
    for s, sl in (("S", (1, slice(None), slice(None), -1)), ("W", (1, slice(None), -1, slice(None)))):
        assert np.array_equal(new[f"{s}_U"][sl], old[f"{s}_U"][sl])


def _update_ref(P, C, vc, pi, pj, ich, jch, ratio, wgt):
    sg1, sg2, _, _, pt, pdtop = vc
    lm = P["T"].shape[0]
    out = {k: P[k].astype(np.float64).copy() for k in ("T", "Q", "CW", "U", "V")}
    for n in range(len(pi)):
        i, j, ic, jc = pi[n] - 1, pj[n] - 1, ich[n], jch[n]
        fis_c = C["FIS"][jc - 1:jc + 2, ic - 1:ic + 2].mean()
        pd_c = C["PD"][jc - 1:jc + 2, ic - 1:ic + 2].mean()
        for k in ("T", "Q", "CW", "U", "V"):
            if k in ("T", "Q", "CW"):
                col = C[k][:, jc - 1:jc + 2, ic - 1:ic + 2].reshape(lm, -1).mean(axis=1).astype(np.float64)
            else:
                icv, jcv = ic + ratio // 2, jc + ratio // 2
                col = C[k][:, jcv - 1:jcv + 1, icv - 1:icv + 1].reshape(lm, -1).mean(axis=1).astype(np.float64)
            if fis_c > 1 or P["FIS"][j, i] > 1:
                pic = sg2 * pd_c + sg1 * pdtop + pt
                pip = sg2 * P["PD"][j, i] + sg1 * pdtop + pt
                pmc = list(0.5 * (pic[:-1] + pic[1:])); pmp = 0.5 * (pip[:-1] + pip[1:])
                vx = list(col)
                if pmp[lm - 1] > pmc[lm - 1]:
                    pmc.append(pmp[lm - 1])
                    p0, p1, p2 = col[lm - 1] * pmc[lm - 1], col[lm - 2] * pmc[lm - 2], col[lm - 3] * pmc[lm - 3]
                    vx[lm - 1] = (p0 + p1) / (pmc[lm - 1] + pmc[lm - 2])
                    vx[lm - 2] = (p0 + p1 + p2) / (pmc[lm - 1] + pmc[lm - 2] + pmc[lm - 3])
                    a, b = 0.5 * (pmc[lm - 2] + pmc[lm - 1]), (pmc[lm - 3] + pmc[lm - 2] + pmc[lm - 1]) / 3.
                    pmc[lm - 1], pmc[lm - 2] = a, b
                    delp = pmp[lm - 1] - pmc[lm - 1]
                    c1 = (vx[lm - 1] - vx[lm - 2]) / (pmc[lm - 1] - pmc[lm - 2])
                    vx.append(vx[lm - 1] + c1 * delp * HYPER_A / (delp + HYPER_A))
                col = spline_ref(pmc, vx, pmp)
            out[k][:, j, i] = col * wgt + out[k][:, j, i] * (1 - wgt)
    return out


@needs_card
@pytest.mark.parametrize("wgt", [1.0, 0.5])
def test_two_way_kernels_match_reference(wgt):
    ctx_p, ctx_c, P, C, vc, (i_ps, j_ps, ratio) = _case(seed=3, nxp=10, nyp=9)
    cpl = tw.Coupler(ctx_p, ctx_c, ratio, i_ps, j_ps, two_way=True, wgt=wgt)
    assert cpl.npts > 0
    cpl.generate()
    cpl.update()
    cp.cuda.Device().synchronize()
    pi, pj, _ = tw.update_points(i_ps, j_ps, ratio, ctx_c.grid.im, ctx_c.grid.jm, 2, 16, 16)
    ich, jch = (pi - i_ps) * ratio, (pj - j_ps) * ratio
    ref = _update_ref(P, C, vc, pi, pj, ich, jch, ratio, wgt)
    for k in ("T", "Q", "CW", "U", "V"):
        got = cp.asnumpy(ctx_p.st[k])
        assert np.allclose(got, ref[k], rtol=1e-4, atol=2e-4), k
        mask = np.zeros(got.shape[1:], bool)
        mask[pj - 1, pi - 1] = True
        assert np.array_equal(got[:, ~mask], P[k][:, ~mask])     # outside the update region nothing changes
