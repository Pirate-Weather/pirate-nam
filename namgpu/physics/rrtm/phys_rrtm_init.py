"""phys-rrtm-init: RRTM_INIT + GAUSIN, the RRTM driver init (module_RA_RRTM.F90:1424-1517, nam.v4.2.11).

Kernel: namgpu/kernels/phys_rrtm_init.cu (phys_rrtm_init_gauss, phys_rrtm_init_ltop, phys_rrtm_init_emiss).

Caller: module_SOLVER_GRID_COMP.F90:10824 (GFDL/RRTM branch, :10265-10269 and :10614-10629 for the flip).
The caller builds:
  SFULL(L) = int_state%SGM(L), L = 1..LM+1            (:10265, SGM read from the input file,
                                                      module_INIT_READ_BIN.F90:752 and
                                                      module_INIT_READ_NEMSIO.F90:192)
  SMID(L) = (SFULL(L)+SFULL(L+1))*0.5, L = 1..LM      (:10269)
  SMID_FLIP(LM+1-K) = SMID(K)                          (:10624), the array this unit reads as SHALF
  PT_CB = int_state%PT*1.0E-3                          (:10230)

Reads:  ctx.grid.pt (Pa), ctx.grid.im/jm/lm, and the NAM header value SGM from
        ctx.tables["vcoord_header"]["SGM"] (LM+1 values, SFULL in the Fortran).
Writes:
  ctx.st["EMISS"]          float32 [JM, IM] = 1.0 (added here, the RRTM EMISS array).
  ctx.tables["rrtm_init"], a dict of device arrays and host ints:
      "axsd"   float32 [NXSD=310]  AXSD, the Gaussian area table
      "rsqr"   float32 [1]         RSQR = 1/sqrt(2*pi)
      "ltop"   int32   [3]         LTOP(1..3), 1-based level indices
      "ptopc"  float32 [4]         PTOPC(1..4), Pa
      "nxsd"   310

SFULL binding. SHALF, and therefore every LTOP(1..3) value, is a function of SGM. nam-gpu does not
carry SGM on the live path yet (no reader supplies ctx.tables["vcoord_header"]; see init_vcoord_recover.py).
There is no stand-in: grid.sg2 is a different quantity (the interfaces the grid was built from, not the
NAM SGM header), so substituting it would give wrong cloud-layer limits. The unit refuses until the init
carries SGM. Refusal names the breakage: LTOP and PTOPC feed the RRTM diagnostic cloud layers.

Domain. The Fortran EMISS loop runs over the tile, DO J=JTS,JTE / DO I=ITS,ITE (:1475-1479 of RRTM_INIT,
caller limits from int_state, :509-517 of the caller). nam-gpu runs one card with one tile covering the
whole domain, so ITS=1, ITE=IDE=IM, JTS=1, JTE=JDE=JM. The single-tile equality ITE==IDE is the NMMB
convention (module_DOMAIN_GRID_COMP.F90:3770 tests ITE==IDE for the eastern boundary task), and
init_derive_state.py sets IDE=IM, JDE=JM for the single-card tile. The kernel covers i0<IM, j0<JM.

Launches: four kernel launches on ctx.stream, no host sync, no per-step allocation (the tables are
allocated once). The init runs once per forecast start, off the step path.
"""
from __future__ import annotations

import contextlib

import numpy as np

from namgpu import kernels

UNIT = "phys_rrtm_init"
NXSD = 310                       # NXSD = XSDmax/DXSD = 3.1/.01 (module_RA_RRTM.F90:44-45)
PT_TO_CB = np.float32(1.0e-3)    # PT_CB = int_state%PT*1.0E-3 (module_SOLVER_GRID_COMP.F90:10230)
VCOORD_KEY = "vcoord_header"     # the NAM input header values, see init_vcoord_recover.py


def sfull_from_header(ctx, lm: int) -> np.ndarray:
    """SFULL(1..LM+1) = the NAM header SGM (int_state%SGM, module_INIT_READ_BIN.F90:752). Refuses if absent."""
    hdr = ctx.tables.get(VCOORD_KEY) if ctx.tables else None
    if hdr is None or "SGM" not in hdr:
        raise RuntimeError(
            "phys-rrtm-init: SFULL is the NAM header SGM (module_INIT_READ_BIN.F90:752); "
            "ctx.tables['vcoord_header']['SGM'] is absent. Without it SHALF and LTOP(1..3) would be wrong, "
            "and grid.sg2 is not a stand-in for SGM. The init unit must carry SGM.")
    v = np.asarray(hdr["SGM"], np.float64).ravel()
    if v.size != lm + 1:
        raise RuntimeError(f"phys-rrtm-init: header SGM has {v.size} values, need LM+1={lm + 1}")
    return v.astype(np.float32)


def smid_flip(sfull: np.ndarray, lm: int) -> np.ndarray:
    """SMID_FLIP as the caller builds it. sfull has LM+1 values, SFULL(1..LM+1) (1-based in the Fortran).

    SMID(L) = (SFULL(L)+SFULL(L+1))*0.5 for L = 1..LM; SMID_FLIP(LM+1-K) = SMID(K).
    Returns float32 [LM], with index 0 = SMID_FLIP(1) = SMID(LM).
    """
    f = np.asarray(sfull, dtype=np.float32)
    if f.size != lm + 1:
        raise RuntimeError(f"phys-rrtm-init: SFULL has {f.size} values, need LM+1={lm + 1}")
    smid = (f[:lm] + f[1:lm + 1]) * np.float32(0.5)
    return np.ascontiguousarray(smid[::-1])


def run(ctx):
    import cupy as cp

    g, st = ctx.grid, ctx.st
    im, jm, lm = int(g.im), int(g.jm), int(g.lm)
    if g.pt is None:
        raise RuntimeError("phys-rrtm-init: ctx.grid.pt is not set")
    sfull = sfull_from_header(ctx, lm)                    # refuses before any write when SGM is absent

    if ctx.tables is None:
        ctx.tables = {}
    scope = ctx.stream if ctx.stream is not None else contextlib.nullcontext()

    f32, i32 = np.float32, np.int32
    tbl = ctx.tables.get("rrtm_init")
    if tbl is None:
        tbl = {
            "axsd": cp.zeros(NXSD, dtype=f32),
            "rsqr": cp.zeros(1, dtype=f32),
            "ltop": cp.zeros(3, dtype=i32),
            "ptopc": cp.zeros(4, dtype=f32),
            "nxsd": NXSD,
        }
        ctx.tables["rrtm_init"] = tbl

    emiss = st.add("EMISS", (jm, im), dtype="float32")
    # PT_CB = int_state%PT*1.0E-3, single precision (PT in Pa, so the value is in centibars).
    pptop = f32(f32(g.pt) * PT_TO_CB)
    shalf_d = cp.asarray(smid_flip(sfull, lm))           # SMID_FLIP(1..LM) on the device

    k_emiss = kernels.get(UNIT, "phys_rrtm_init_emiss")
    k_gauss = kernels.get(UNIT, "phys_rrtm_init_gauss")
    k_ltop = kernels.get(UNIT, "phys_rrtm_init_ltop")
    with scope:
        tbl["ltop"].fill(0)          # LTOP(1..3)=0 before the loop (:1452-1454)
        k_emiss(kernels.grid_2d(im, jm, 1), kernels.BLOCK_2D, (emiss, i32(im), i32(jm)))
        k_gauss(((NXSD + 127) // 128, 1, 1), kernels.BLOCK_COL, (tbl["axsd"], tbl["rsqr"]))
        k_ltop(((lm + 127) // 128, 1, 1), kernels.BLOCK_COL,
               (shalf_d, i32(lm), pptop, tbl["ltop"], tbl["ptopc"]))
    return tbl
