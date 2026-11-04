"""phys-gwd-col-wave: GWD_col part 2 (low-level wind, base stress, Ri-limited stress profile,
top de-acceleration limit).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), nmm/module_GWD.F90:898-1132. NOAA US government work;
the module's NOAA header is kept as the notice in the kernel file.
Kernel: namgpu/kernels/phys_gwd_col_wave.cu (phys_gwd_col_wave, one thread per active column,
one launch, grid_cols(npt, 1)).

Precision: double (kind_phys = real*8), the same as phys-gwd-col-blocking, which feeds this unit.
Vertical order: GWD_col is bottom-up, index 0 is the surface (Fortran K = 1). PRSI is in centibars.

Inputs (Fortran (i,j,l) -> CuPy [l, j, i], C order, float64 unless noted):
  ipt        int32 [npt]        full-domain cell index p = j*IM+i of each active point (IPT of the source)
  full-domain [JM, IM]:         HPRIME, OC (read); DUsfc, DVsfc (INOUT, zeroed by the blocking part);
                                UBAR, VBAR, ROLL, BNV2BAR, OA, CLX (blocking outputs);
                                KREF, IDXZB (int32, Fortran 1-based level)
  full-domain [LM, JM, IM]:     U1, V1, DEL (read); A, B (INOUT tendencies);
                                RO, BNV2, RI_N, DB (blocking outputs)
  full-domain [LM+1, JM, IM]:   PRSI (read)
  scratch [LM+1, npt]:          TAUP, TAUD, VELCO (compact, allocated once per npt, reused every call)
Every per-point input is read at p = ipt[c]; there is no compaction step. The blocking unit's
outputs feed this unit directly (run() below does exactly that).
Scalars: DTPHS (DELTIM = NPHS*DT, phys-gwd-driver), CLEFF (device scalar from phys-gwd-init),
FACTOP, RLOLEV (cb, the units PRSI is compared in).

Output: DUsfc, DVsfc (cell layout, ROGN applied), A, B (LM levels updated in place).
No host sync, no per-step allocation after the first call, everything on the current stream.

run(ctx) is the done-rule entry point. It builds a full-size 954 x 835 x 60 GWD_col batch with
phys-gwd-col-blocking's random-input generator, runs the blocking unit on it, and feeds its outputs
straight into this unit. OC is not produced by the blocking unit (the driver supplies it), so it is
synthetic here.
"""
from __future__ import annotations

import numpy as np

from namgpu import kernels
from namgpu.physics.gwd import phys_gwd_col_blocking as blk

UNIT = "phys_gwd_col_wave"
KERNEL = "phys_gwd_col_wave"
LM = 60
IM = 954
JM = 835
DTPHS_DEFAULT = 50.0      # DTPHS = NPHS * DT = 2 * 25 s, as phys-gwd-driver.py sets it for the NAM step
CLEFF_DEFAULT = 1.0e-5    # GWD_init: CLEFFAMP * 1e-5 * sqrt(DPHD/DPHD0), CLEFFAMP 1.0, DPHD = DPHD0
FACTOP_DEFAULT = 0.5      # configfile factop
RLOLEV_DEFAULT = 50.0     # configfile rlolev, centibars (500 hPa), the units PRSI is in

_scratch: dict = {}


def gwd_col_wave(ipt, npt, np_cells, lm, dtphs, cleff, factop, rlolev,
                 dusfc, dvsfc, hprime, oc, u1, v1, del_, prsi, a, b,
                 ubar, vbar, roll, bnv2bar, oa, clx, kref, idxzb,
                 ro, bnv2, ri_n, db, taup, taud, velco):
    """One launch over the npt active columns. Arrays are CuPy device arrays (see module doc)."""
    k = kernels.get(UNIT, KERNEL)
    k(kernels.grid_cols(int(npt), 1), kernels.BLOCK_COL,
      (ipt, np.int32(npt), np.int32(np_cells), np.int32(lm),
       np.float64(dtphs), np.float64(cleff), np.float64(factop), np.float64(rlolev),
       dusfc, dvsfc, hprime, oc, u1, v1, del_, prsi, a, b,
       ubar, vbar, roll, bnv2bar, oa, clx, kref, idxzb,
       ro, bnv2, ri_n, db, taup, taud, velco))
    return dusfc, dvsfc, a, b


def _scratch_arrays(npt: int):
    import cupy as cp

    if npt not in _scratch:
        _scratch[npt] = dict(
            taup=cp.zeros((LM + 1, npt), dtype=cp.float64),
            taud=cp.zeros((LM + 1, npt), dtype=cp.float64),
            velco=cp.zeros((LM + 1, npt), dtype=cp.float64),
        )
    return _scratch[npt]


def run(ctx=None):
    """Done-rule entry: blocking unit on a full-size 954 x 835 x 60 batch, then this unit on its outputs."""
    import cupy as cp

    jm, im, lm = JM, IM, LM
    npc = jm * im

    inp = blk._random_inputs(jm, im, lm)
    bout = blk._outputs(jm, im, lm)
    blk.gwd_col_blocking(inp, bout, im=im, jm=jm, lm=lm)

    rs = cp.random.RandomState(2)
    oc = rs.uniform(0.1, 1.0, size=(jm, im)).astype(cp.float64)
    a = cp.zeros((lm, jm, im), dtype=cp.float64)
    b = cp.zeros((lm, jm, im), dtype=cp.float64)
    ipt = cp.arange(npc, dtype=cp.int32)
    npt = npc

    s = _scratch_arrays(npt)
    gwd_col_wave(ipt, npt, npc, lm, DTPHS_DEFAULT, CLEFF_DEFAULT, FACTOP_DEFAULT, RLOLEV_DEFAULT,
                 bout["DUsfc"], bout["DVsfc"], inp["HPRIME"], oc, inp["U1"], inp["V1"],
                 inp["DEL"], inp["PRSI"], a, b,
                 bout["UBAR"], bout["VBAR"], bout["ROLL"], bout["BNV2BAR"], bout["OA"], bout["CLX"],
                 bout["KREF"], bout["IDXZB"],
                 bout["RO"], bout["BNV2"], bout["RI_N"], bout["DB"],
                 s["taup"], s["taud"], s["velco"])
    return {"DUSFC": bout["DUsfc"], "DVSFC": bout["DVsfc"], "A": a, "B": b}
