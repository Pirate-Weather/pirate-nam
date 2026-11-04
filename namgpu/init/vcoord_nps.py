"""The NAM hybrid vertical coordinate exactly as NPS builds it (vcoord = 1, the NAM namelist).

Ported from sorc/nam_nps.fd/sorc/NMMB_init/NPS/nemsinterp/src/module_vinterp_routines.F90
(nam.v4.2.11, 83df4e13c1): define_nmm_vertical_coord (1144-1330, VCOORD_OPT = 1 branch) ->
vcgenerator (3741-3949), with the levels taken from the namelist (coord_levs(1) = 0.0 and
vcoord /= 0, so auto_define is false at line 362 and sgm = coord_levs as given, top = 0).
NOAA NPS code, US government work; this header is the notice.

Single precision (the Fortran is real(kind=4)), evaluated in the Fortran order. Init only, 61 values
on the host. The result is the header dict init-vcoord-recover consumes (ctx.tables["vcoord_header"]):
PT, PDTOP, LPT2, SG1, SG2, DSG1, DSG2, SGML1, SGML2.

Note on the shape (why grid.validate_vcoord had to change, 2026-11-04): with vcoord = 1 the
coordinate is a smooth hybrid, not the vcoord = 0 split. SG1 rises 0 -> 1 down to LPT2 and then
falls back to 0 at the surface (SG1(LM+1) = 0, DSG1 < 0 below LPT2), and SG2 rises 0 -> 1 below
LPT2. So PINT(LM+1) = SG2*PD + SG1*PDTOP + PT = PD + PT, i.e. PD = PSFC - PT, which is what
init-native-atm and init-gfs-vinterp already use for vcoord = 1.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def vcgenerator(lm: int, pt: float, ptsgm: float, sgm_in) -> dict:
    sgm = np.array(sgm_in, dtype=np.float32).copy()
    if sgm.size != lm + 1:
        raise ValueError(f"vcoord-nps: {sgm.size} coord_levs, expected LM+1 = {lm + 1}")
    if sgm[0] != 0.0 or abs(float(sgm[-1]) - 1.0) > 1e-6:
        raise ValueError("vcoord-nps: coord_levs must run 0 (top) .. 1 (surface) (auto_define false branch)")
    pt, ptsgm = F(pt), F(ptsgm)
    pbot = F(101300.0)                                 # reference mslp (3851)
    lpt2 = 0
    for l in range(1, lm + 2):                         # do l=1,lm+1
        pl = F(sgm[l - 1] * (pbot - pt) + pt)
        if pl < ptsgm:
            lpt2 = l
    if lpt2 > 0:
        pt2 = F(sgm[lpt2 - 1] * (F(101300.0) - pt) + pt)
    else:
        pt2 = pt
    pdtop = F(pt2 - pt)
    a = np.zeros(lm + 1, np.float32)
    b = np.zeros(lm + 1, np.float32)
    for l in range(1, lpt2 + 1):
        a[l - 1] = F(sgm[l - 1] / sgm[lpt2 - 1] * pdtop + pt) if sgm[lpt2 - 1] != 0 else pt
        b[l - 1] = F(0.0)
    lsal = 0
    rp, rs, es = F(2.2), F(1.2), F(9.0)
    for l in range(lpt2 + 1, lm + 1 - lsal + 1):       # do l=lpt2+1,lm+1-lsal
        bsg = F((sgm[l - 1] - sgm[lpt2 - 1]) / (F(1.0) - sgm[lpt2 - 1]))
        rsg = F(rp + (rs - rp) / F(np.arctan(es)) * F(np.arctan(F(es * bsg))))
        b[l - 1] = F(np.power(bsg, rsg)) if bsg > 0 else F(0.0)
        a[l - 1] = F((sgm[l - 1] - b[l - 1]) * (pbot - pt) + pt)
    for l in range(lm + 1 - lsal + 1, lm + 2):         # SAL layers (none: lsal = 0)
        b[l - 1] = F((sgm[l - 1] - sgm[lpt2 - 1]) / (F(1.0) - sgm[lpt2 - 1]))
        a[l - 1] = F((sgm[l - 1] - b[l - 1]) * (pbot - pt) + pt)
    sg1 = ((a - pt) / pdtop).astype(np.float32)
    sg2 = b.copy()
    dsg1 = (sg1[1:] - sg1[:-1]).astype(np.float32)
    dsg2 = (sg2[1:] - sg2[:-1]).astype(np.float32)
    sgml1 = ((sg1[:-1] + sg1[1:]) * F(0.5)).astype(np.float32)
    sgml2 = ((sg2[:-1] + sg2[1:]) * F(0.5)).astype(np.float32)
    return {"PT": float(pt), "PDTOP": float(pdtop), "LPT2": int(lpt2), "SG1": sg1, "SG2": sg2,
            "DSG1": dsg1, "DSG2": dsg2, "SGML1": sgml1, "SGML2": sgml2,
            # SGM: the reference sigma levels as given (auto_define false, vcgenerator 3838-3845 leaves
            # them as read), LM+1 interfaces top first. NPS writes this array as the header SGM that
            # module_INIT_READ_BIN.F90:752 reads and RRTM_INIT uses as SFULL.
            "SGM": sgm.astype(np.float32),
            "source": "NPS vcgenerator (module_vinterp_routines.F90:3741-3949), namelist coord_levs"}


def header_from_config(cfg) -> dict:
    vc = int(getattr(cfg, "vcoord", cfg.raw.get("vcoord", 1)) if hasattr(cfg, "raw") else cfg.vcoord)
    if vc != 1:
        raise NotImplementedError(f"vcoord-nps: vcoord={vc}; only the NAM vcoord = 1 generator is ported")
    return vcgenerator(int(cfg.lm), float(cfg.pt), float(cfg.ptsgm), cfg.coord_levs)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from namgpu import config
    h = header_from_config(config.load())
    print("PT", h["PT"], "PDTOP", h["PDTOP"], "LPT2", h["LPT2"])
    for l in range(0, len(h["SG1"])):
        p = h["SG1"][l] * h["PDTOP"] + h["SG2"][l] * (101300.0 - h["PT"]) + h["PT"]
        print(l + 1, f"{h['SG1'][l]:.6f} {h['SG2'][l]:.6f}  pref {p:9.1f}")
