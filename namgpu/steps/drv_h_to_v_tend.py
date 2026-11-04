"""drv-h-to-v-tend: H_TO_V_TEND, add the H-point wind tendencies to the V-point winds.

Fortran: nmm/module_H_TO_V.F90:128-224 (H_TO_V_TEND_3D, DELTA_T = DT*NSTEPS in single precision),
called from SOLVER_RUN at two sites, one per branch:
  branch "nphs" : module_SOLVER_GRID_COMP.F90:5586 (DUDT -> U) and :5588 (DVDT -> V). Inside
                  turbulence: IF(CALL_TURBULENCE) (5475-5664), CALL_TURBULENCE =
                  MOD(NTIMESTEP,NPHS)==0 (5175). NSTEPS = NPHS. Runs after TURBL, which writes
                  DUDT/DVDT (zeroed at module_TURBULENCE.F90:451, GWD added at 1311).
  branch "bmj"  : module_SOLVER_GRID_COMP.F90:5827 (DUDT -> U) and :5830 (DVDT -> V). Inside
                  wind: IF(CONVECTION==sas .or. bmj) (5813-5863), inside convection:
                  IF(CALL_PRECIP .AND. CONVECTION/=none) (5722-5867), CALL_PRECIP =
                  MOD(NTIMESTEP,NPRECIP)==0 (5176). NSTEPS = NPRECIP. Runs after CUCNVC (5732),
                  which writes DUDT/DVDT through its DUDT_PHY/DVDT_PHY arguments (5768). The
                  phys-bmj-driver glue aliases those names to the same arrays (see that unit).
Scope exclusion: module_SOLVER_GRID_COMP.F90:6998 and :7000 (H_TO_V_TEND with NPHS, inside
gfs_physics, 6401-7059, fed by 6958) are NOT ported. They run only when gfs is true, and
parm/nam_configfile sets gfs: false. The wrapper refuses if the config asks for gfs.
The sas branch (5813, convection sas) is not ported and refuses by name.

Reads : DUDT, DVDT [LM,JM,IM] (H points; unit-owned, added to ctx.st by st.add on first call).
Writes: U, V [LM,JM,IM] (V points) in place.
Scalars: DELTA_T = DT*NSTEPS, single precision, formed on the host exactly as the Fortran
(NSTEPS = NPHS for "nphs", NPRECIP for "bmj"). IM, JM, LM from ctx.grid.
The driver gates each call on its cadence (namgpu/driver.py: "nphs" and "nprecip" keys); this
wrapper does not repeat the gate.

Two launches per branch (one per Fortran call: U then V). No host sync, no per-step allocation.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "drv_h_to_v_tend"
KERNEL = "h_to_v_tend_3d"
BRANCHES = ("nphs", "bmj")


def run(ctx, branch: str = "nphs"):
    if branch not in BRANCHES:
        raise ValueError(f"drv-h-to-v-tend: branch must be one of {BRANCHES}, got {branch!r}")
    st = ctx.st
    g = ctx.grid
    cfg = ctx.cfg
    if cfg.raw.get("gfs", False):
        raise RuntimeError("drv-h-to-v-tend: gfs physics is configured. Its H_TO_V_TEND call "
                           "(module_SOLVER_GRID_COMP.F90:6998-7000) is a scope exclusion and is "
                           "not ported.")
    if branch == "bmj":
        if cfg.convection == "sas":
            raise NotImplementedError("drv-h-to-v-tend: convection=sas wind tendency is not "
                                      "ported (module_SOLVER_GRID_COMP.F90:5813-5830).")
        if cfg.convection != "bmj":
            return      # Fortran wind block is not entered for other convection schemes
        nsteps = int(cfg.nprecip)
    else:
        nsteps = int(cfg.nphs)
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if nsteps <= 0:
        raise RuntimeError(f"drv-h-to-v-tend: NSTEPS ({branch}) must be > 0, got {nsteps}")
    dt = np.float32(cfg.dt)
    delta_t = np.float32(dt * np.float32(nsteps))
    # DUDT/DVDT are not in parm/nam_*_state.txt. They are the H-point tendencies that TURBL
    # (nphs branch) and CUCNVC (bmj branch) write. Unit-owned, same [LM,JM,IM] shape as U.
    dudt = st.add("DUDT", (lm, jm, im))
    dvdt = st.add("DVDT", (lm, jm, im))

    kern = kernels.get(UNIT, KERNEL)
    grid = kernels.grid_2d(im, jm, lm)
    block = kernels.BLOCK_2D
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()
    with scope:
        kern(grid, block, (np.int32(im), np.int32(jm), np.int32(lm),
                           delta_t, dudt, st.U))
        kern(grid, block, (np.int32(im), np.int32(jm), np.int32(lm),
                           delta_t, dvdt, st.V))
