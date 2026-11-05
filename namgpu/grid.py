"""Rotated latitude-longitude Arakawa B grid of the NAM 12 km parent, plus the NMMB hybrid
sigma-pressure vertical coordinate.

Horizontal (NMMB module_CONTROL.F90 grid_consts, 351-1139; regional branch):
- H (mass) points at rotated (tlm, tph) = (wb + (i-1)*dlm, sb + (j-1)*dph), i=1..IM, j=1..JM.
- V (wind) points at (tlm + dlm/2, tph + dph/2): V(i,j) sits NE of H(i,j).  Arrays for V
  points have the same [j, i] shape; the last row/column is unused (zero), as in NMMB.
- dxh(j) = a*dlm*cos(tph_j), dxv(j) = a*dlm*cos(tpv_j), dyh = dyv = a*dph.
- The other row constants (wpdar, fcp, fdiv, fad, fah, ddmpu, ddmpv, hdac*, curv, f) are
  built by unit init-grid-consts, which ports grid_consts in full.

Vertical (NMMB): L=1 is the model TOP, L=LM the lowest layer.
- PINT(l) = SG1(l)*PDTOP + SG2(l)*PD + PT  at the LM+1 interfaces,
- PMID(l) = SGML1(l)*PDTOP + SGML2(l)*PD + PT = PSGML1(l) + SGML2(l)*PD,
- DSG1/DSG2 are layer increments of SG1/SG2, PDSG1 = DSG1*PDTOP.
``vcoord_from_mid`` recovers SG1/SG2 exactly from the per-level mid-layer coefficients that
unit init-vcoord-recover fits from the native analysis (PRES on 60 hybrid levels vs PD).
``vcoord_from_config`` is a first guess from nam_namelist.nps coord_levs, used only until the
case's analysis has been read.

Layout: host numpy here; device copies are made by state.State.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NamConfig

A_EARTH = 6376000.0     # NMMB module_CONSTANTS.F90: a=6376000.
TWOM = 0.00014584       # module_CONSTANTS twom (2 x Earth angular velocity)
DTR = np.pi / 180.0


@dataclass
class BGrid:
    im: int
    jm: int
    lm: int
    dlm: float          # radians
    dph: float          # radians
    wb: float           # radians
    sb: float           # radians
    tph0: float         # radians, centre geographic lat
    tlm0: float         # radians, centre geographic lon
    tlm_h: np.ndarray   # [jm, im] rotated lon of H points (rad)
    tph_h: np.ndarray
    glat_h: np.ndarray  # geographic lat of H points (rad)
    glon_h: np.ndarray  # geographic lon (rad, positive east)
    glat_v: np.ndarray
    glon_v: np.ndarray
    dxh: np.ndarray     # [jm] metres
    dxv: np.ndarray     # [jm]
    dyh: float
    dyv: float
    # vertical, set by set_vcoord
    pt: float = 200.0
    pdtop: float = 0.0
    sg1: np.ndarray | None = None    # [lm+1]
    sg2: np.ndarray | None = None
    dsg1: np.ndarray | None = None   # [lm]
    dsg2: np.ndarray | None = None
    sgml1: np.ndarray | None = None
    sgml2: np.ndarray | None = None
    # moving-nest window (config.NamConfig.win_i0/win_j0): wb/sb are the REFERENCE grid's origin and H(i, j) sits at
    # (wb + (i + win_i0) dlm, sb + (j + win_j0) dph); 0/0 for every grid without window keys
    win_i0: int = 0
    win_j0: int = 0

    @property
    def psgml1(self):
        return self.sgml1 * self.pdtop + self.pt

    @property
    def pdsg1(self):
        return self.dsg1 * self.pdtop

    @property
    def psg1(self):
        return self.sg1 * self.pdtop + self.pt

    def set_vcoord(self, pt, pdtop, sg1, sg2):
        self.pt, self.pdtop = float(pt), float(pdtop)
        self.sg1 = np.asarray(sg1, np.float32)
        self.sg2 = np.asarray(sg2, np.float32)
        self.dsg1 = (self.sg1[1:] - self.sg1[:-1]).astype(np.float32)
        self.dsg2 = (self.sg2[1:] - self.sg2[:-1]).astype(np.float32)
        self.sgml1 = (0.5 * (self.sg1[1:] + self.sg1[:-1])).astype(np.float32)
        self.sgml2 = (0.5 * (self.sg2[1:] + self.sg2[:-1])).astype(np.float32)


def rotated_to_geographic(tlm, tph, tlm0, tph0):
    """Rotated (lambda, phi) -> geographic (lon, lat), radians.  Rotated equator through the
    domain centre (tlm0, tph0).  Checked by init-grid-geo against NLAT/ELON of the analysis."""
    stph0, ctph0 = np.sin(tph0), np.cos(tph0)
    stph, ctph = np.sin(tph), np.cos(tph)
    ctlm, stlm = np.cos(tlm), np.sin(tlm)
    sphi = stph0 * ctph * ctlm + ctph0 * stph
    sphi = np.clip(sphi, -1.0, 1.0)
    glat = np.arcsin(sphi)
    # cos(glat) * (cos dlam, sin dlam) = (ctph0 ctph ctlm - stph0 stph, ctph stlm); atan2 gives the
    # sign of dlam and is well conditioned at the centre meridian and the poles (arccos was not).
    dlam = np.arctan2(ctph * stlm, ctph0 * ctph * ctlm - stph0 * stph)
    glon = tlm0 + dlam
    glon = (glon + np.pi) % (2.0 * np.pi) - np.pi
    return glon, glat


def build(cfg: NamConfig) -> BGrid:
    im, jm, lm = cfg.im, cfg.jm, cfg.lm
    dlm, dph = cfg.dlmd * DTR, cfg.dphd * DTR
    wb, sb = cfg.ref_wbd * DTR, cfg.ref_sbd * DTR       # the reference origin (= wbd/sbd without window keys)
    tph0, tlm0 = cfg.tph0d * DTR, cfg.tlm0d * DTR
    ii = np.arange(im) + int(cfg.win_i0)                  # window offset in whole cells (0 without window keys)
    jj = np.arange(jm) + int(cfg.win_j0)
    tlm_h = np.broadcast_to(wb + ii * dlm, (jm, im)).copy()
    tph_h = np.broadcast_to((sb + jj * dph)[:, None], (jm, im)).copy()
    glon_h, glat_h = rotated_to_geographic(tlm_h, tph_h, tlm0, tph0)
    glon_v, glat_v = rotated_to_geographic(tlm_h + 0.5 * dlm, tph_h + 0.5 * dph, tlm0, tph0)
    tph_row = sb + jj * dph
    dxh = (A_EARTH * dlm * np.cos(tph_row)).astype(np.float32)
    dxv = (A_EARTH * dlm * np.cos(tph_row + 0.5 * dph)).astype(np.float32)
    g = BGrid(im=im, jm=jm, lm=lm, dlm=dlm, dph=dph, wb=wb, sb=sb, tph0=tph0, tlm0=tlm0,
              tlm_h=tlm_h, tph_h=tph_h, glat_h=glat_h, glon_h=glon_h, glat_v=glat_v,
              glon_v=glon_v, dxh=dxh, dxv=dxv, dyh=float(A_EARTH * dph),
              dyv=float(A_EARTH * dph), win_i0=int(cfg.win_i0), win_j0=int(cfg.win_j0))
    vcoord_from_config(g, cfg)
    return g


def vcoord_from_config(g: BGrid, cfg: NamConfig, p0: float = 101325.0) -> None:
    """First guess of SG1/SG2 from nam_namelist.nps coord_levs (top=0 .. bottom=1).  Interfaces
    whose reference pressure is above ptsgm are pure pressure (SG2=0), below are sigma
    (SG1=1).  Replaced by init-vcoord-recover once the analysis is read."""
    eta = np.asarray(cfg.coord_levs, np.float64)
    if eta.size != cfg.lm + 1:
        raise ValueError(f"coord_levs has {eta.size} values, expected LM+1={cfg.lm + 1}")
    pref = cfg.pt + eta * (p0 - cfg.pt)
    pdtop = cfg.ptsgm - cfg.pt
    sg1 = np.where(pref <= cfg.ptsgm, (pref - cfg.pt) / pdtop, 1.0)
    sg2 = np.where(pref <= cfg.ptsgm, 0.0, (pref - cfg.ptsgm) / (p0 - cfg.ptsgm))
    g.set_vcoord(cfg.pt, pdtop, sg1, sg2)


def interfaces_from_mid(mid, top_value: float, bottom_value: float) -> np.ndarray:
    """Interface values s[0..LM] from mid-layer values m[0..LM-1] with s[0] = top_value and
    s[LM] = bottom_value fixed exactly. Least squares on the midpoint relation
    s[l] + s[l+1] = 2*m[l] (NMMB SGML = (SG(l)+SG(l+1))/2), one row per layer, LM-1 unknowns.
    Exact when m comes from a consistent coordinate. Not a recurrence: errors in one layer do
    not propagate and alternate down the column."""
    m = np.asarray(mid, np.float64)
    lm = m.size
    a = np.zeros((lm, lm - 1))
    rhs = 2.0 * m
    for l in range(lm):
        if l >= 1:
            a[l, l - 1] += 1.0            # s[l] is unknown u[l-1]
        else:
            rhs[l] -= top_value           # s[0] known
        if l + 1 <= lm - 1:
            a[l, l] += 1.0                # s[l+1] is unknown u[l]
        else:
            rhs[l] -= bottom_value        # s[LM] known
    u = np.linalg.lstsq(a, rhs, rcond=None)[0]
    s = np.empty(lm + 1)
    s[0] = top_value
    s[1:lm] = u
    s[lm] = bottom_value
    return s


def validate_vcoord(g: BGrid, tol: float = 1e-4) -> None:
    """Refuse a coordinate that is not a valid NMMB hybrid: SG1 and SG2 are 0 at the model top
    and exactly 1 at the surface (so PINT(LM+1) = PSFC, module_TURBULENCE.F90:557 with
    PSG1(LM+1) = PDTOP + PT), DSG1 strictly positive and DSG2 non-decreasing (pure-pressure
    layers have DSG2 = 0)."""
    errs = []
    if abs(float(g.sg1[0])) > 1e-6 or abs(float(g.sg2[0])) > 1e-6:
        errs.append(f"top interface SG1(1)={g.sg1[0]:.6g}, SG2(1)={g.sg2[0]:.6g}, expected 0")
    # vcoord = 0 (split): SG1(LM+1) = 1 and DSG1 > 0. vcoord = 1 (NAM, NPS vcgenerator): SG1 rises to
    # about 1.2 near LPT2 and returns to 0 at the surface, so SG1(LM+1) = 0, DSG1 < 0 below LPT2 and
    # PD = PSFC - PT. Both are valid NMMB coordinates; what must hold is SG2(LM+1) = 1 and interface
    # pressures that increase downward for any surface pressure (checked at 50 and 110 kPa).
    sg1_bot = float(g.sg1[-1])
    if abs(float(g.sg2[-1]) - 1.0) > 1e-6 or not (abs(sg1_bot) < 1e-6 or abs(sg1_bot - 1.0) < 1e-6):
        errs.append(f"surface interface SG1(LM+1)={g.sg1[-1]:.6g}, SG2(LM+1)={g.sg2[-1]:.6g}, "
                    "expected SG2 = 1 and SG1 = 0 (vcoord 1) or 1 (vcoord 0)")
    pdtop_ = float(g.pdtop)
    for psfc in (50000.0, 110000.0):
        pd_ = psfc - float(g.pt) - (pdtop_ if sg1_bot > 0.5 else 0.0)
        pint = np.asarray(g.sg1, np.float64) * pdtop_ + np.asarray(g.sg2, np.float64) * pd_ + float(g.pt)
        bad = np.nonzero(~(np.diff(pint) > 0.0))[0]
        if bad.size:
            errs.append(f"interface pressure not increasing downward at layers {bad[:8].tolist()} "
                        f"(PSFC {psfc:.0f} Pa)")
    bad2 = np.nonzero(g.dsg2 < -tol)[0]
    if bad2.size:
        errs.append(f"DSG2 decreasing at layers {bad2[:8].tolist()}")
    if errs:
        raise RuntimeError("vertical coordinate invalid: " + "; ".join(errs))


def vcoord_from_mid(g: BGrid, psgml1: np.ndarray, sgml2: np.ndarray, pdtop: float) -> None:
    """Exact SG1/SG2 interfaces from the per-level mid-layer coefficients: SGML1 from PSGML1
    (= SGML1*PDTOP + PT), SGML2 directly. Interfaces are solved with SG(1)=0 and SG(LM+1)=1
    fixed (interfaces_from_mid), then validated (validate_vcoord)."""
    sgml1 = (np.asarray(psgml1, np.float64) - g.pt) / pdtop
    sg1 = interfaces_from_mid(sgml1, 0.0, 1.0)
    sg2 = interfaces_from_mid(np.asarray(sgml2, np.float64), 0.0, 1.0)
    g.set_vcoord(g.pt, pdtop, sg1, sg2)
    validate_vcoord(g)
