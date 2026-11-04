"""drv-dfi-refl-restore: the 0 h radar reflectivity recomputed from the state the digital filter leaves behind.

Why: after TDFI the filtered dynamics (dyn_average) and the restored physics fields (drv-dfi-phy-restore: CW,
F_ICE, F_RAIN, F_RIMEF and the other R-flagged fields of parm/nam_filt_vars_norad.txt) are the 0 h state, but
REFL_10CM is not in the restore table, so it kept the filter's last FER_HIRES values: 1,037 cells over 40 dBZ
at f00 in smoke24 (2026-11-04 11:25Z) that the 0 h condensate does not hold. The no-DFI start writes no
reflectivity at all, so the 0 h map had to be either dropped or made consistent; this makes it consistent.

How: CALMICT (phys-fer-calmict, module_DIAGNOSE.F90:1589-1798, the Ferrier reflectivity diagnostic the NAM post
uses for REFC) on the 0 h state: P = mid-layer pressure SGML2*PD + PSGML1 (the same expression as the Ferrier
driver prep, phys_fer_driver.cu), T, Q, C = CW, FI = F_ICE, FR = F_RAIN, FS = F_RIMEF. CUREFL (the convective
term) is zero: no convection scheme has run on the 0 h state. The next microphysics call (FER_HIRES, step 0 of
the forecast) overwrites REFL_10CM as before, so only the 0 h history changes.
Tables: Ferrier MASSI and the rain-size scalars from phys-fer-init's ctx.tables["fer"]; TBPVS and its
C1XPVS/C2XPVS from phys-fer-gpvs (one 4-float readback, once per run).
Called by drv-dfi-tdfi-driver right after drv-dfi-phy-restore, on ctx.stream.
"""
from __future__ import annotations

import contextlib

import numpy as np

UNIT = "drv_dfi_refl_restore"


def run(ctx):
    import cupy as cp

    from namgpu.physics.fer_hires import phys_fer_calmict, phys_fer_gpvs, phys_fer_init

    st = ctx.st
    if "REFL_10CM" not in st.fields:
        return None
    g = ctx.grid
    lm = int(g.lm)
    fer = ctx.tables.get("fer")
    if fer is None:
        phys_fer_init.run(ctx)
        fer = ctx.tables["fer"]
    if "fer_gpvs_tbpvs" not in ctx.tables:
        phys_fer_gpvs.run(ctx)
    if "fer_scalars" not in ctx.tables:
        coef = cp.asnumpy(ctx.tables["fer_gpvs_coef"])        # once per run, at the end of the filter
        sc = dict(fer["scalars"])
        sc["C1XPVS"], sc["C2XPVS"] = float(coef[0]), float(coef[1])
        ctx.tables["fer_scalars"] = sc
        ctx.tables["fer_massi"] = fer["massi"]
    from namgpu.physics.driver.phys_fer_driver import _vcoord
    psgml1 = _vcoord(ctx, "psgml1")
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        sgml2 = st["SGML2"].reshape(lm, 1, 1)
        p = sgml2 * st["PD"][None, :, :] + psgml1.reshape(lm, 1, 1)
        p = p.astype(cp.float32, copy=False)
        curefl = ctx.tables.get("dfi_refl_curefl")
        if curefl is None or curefl.shape != st["PD"].shape:
            curefl = cp.zeros(st["PD"].shape, dtype=cp.float32)
            ctx.tables["dfi_refl_curefl"] = curefl
        dbz = phys_fer_calmict.run(ctx, p, st["T"], st["Q"], st["CW"], st["F_ICE"], st["F_RAIN"],
                                   st["F_RIMEF"], curefl)
        before = float(st["REFL_10CM"].max())        # one readback, once per run (the end of the filter)
        cp.copyto(st["REFL_10CM"], dbz)
        after = float(dbz.max())
        n40 = int((dbz.max(axis=0) > 40.0).sum())
    print(f"drv-dfi-refl-restore: 0 h REFL_10CM recomputed (CALMICT): max {before:.1f} -> {after:.1f} dBZ, "
          f"{n40} columns over 40 dBZ", flush=True)
    return {"refl": "recomputed from the 0 h state (CALMICT, CUREFL 0)", "max_before": before, "max_after": after}
