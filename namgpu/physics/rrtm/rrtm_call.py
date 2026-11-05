"""rrtm-call: SUBROUTINE RRTM of module_RA_RRTM.F90 as one chain of the RRTM units, in Fortran order.

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), phys/module_RA_RRTM.F90:98-1418 (RRTM) and
phys/grrad_nmmb.f:660-2210 (grrad_nmmb). NOAA US government work; each unit's kernel keeps the
NOAA header as its notice. This module has no kernel of its own: it is the call order.

Called by phys-rad-driver (module_RADIATION.F90:586-625) on radiation steps as run(ctx, args),
args = the Fortran RRTM argument dict built by phys_rad_driver._rrtm_args.

Order (one radiation call):
  module_RA_RRTM.F90
    98-878     phys-rrtm-driver-prep     control scalars and P8W/P_PHY (whole domain: they come from the
                                         J = JTE row, see its header)
  then per row chunk (rrtm_chunk: rows in chunks of ROWS, columns independent):
    98-878     phys-rrtm-driver-prep     cloud layering, TAUTOTAL, CUTOP/CUBOT (kernel 2, per column, on
                                         the chunk's rows: no whole-domain scratch)
    879-881    phys-rrtm-dayparts        COSZMN over the SW interval, sunlit runs per row
    879-1112   phys-rrtm-driver-call     grrad inputs per column (prsi, prsl, gt, gq, gr1, surface)
  grrad_nmmb.f
    660-1396   phys-rrtm-grrad-setup     plvl/plyr/tlyr/prslk1/rhly/qstl, GR1 tracer, cloud fractions
    1397-1407  phys-rrtm-getozn          climatological ozone (ntoz == 0)
    1409-1414  phys-rrtm-coszmn          coszen_loc, coszdg
    1419-1427  phys-rrtm-getgases        gas volume mixing ratios
    1428-1539  phys-rrtm-grrad-gases     tlvl, tvly, qlyr, tsfa
    1540-1576  phys-rrtm-grrad-aerosols  setaer -> faersw, faerlw
    1556-1794  phys-rrtm-grrad-clouds    clw, progcld2 (np3d 3), IALB=2 albedo
    1818-1962  phys-rrtm-grrad-sw        setalb, swrad          (lsswr)
    1963-2201  phys-rrtm-grrad-lw        setemis, lwrad, fluxr  (lslwr inside; fluxr/coszen always)
  module_RA_RRTM.F90
    1113-1418  phys-rrtm-driver-post     RSWTT, RLWTT, RSWIN, RLWIN, ..., cloud fractions
  and the chunk's owned rows of every state field it touched go back to the State.

lsswr = MOD(NTIMESTEP, NRADS) == 0, lslwr = MOD(NTIMESTEP, NRADL) == 0 (module_RA_RRTM.F90:842-846),
NTIMESTEP = NTIMESTEP_RAD. dtsw = NRADS*DT, dtlw = NRADL*DT.

The runtime physparam flags are the ones rad_initialize_nmmb.f:134-183 sets from the NAM solver
values (module_SOLVER_GRID_COMP.F90:10650-10700): ioznflg = NTOZ = 0, ico2flg = ICO2 = 1,
ivflip = IFLIP = 0 (namgpu/config.py carries those runtime values).

Memory: ROWS rows per chunk (default 8, owned 6 per chunk over JM = 835). The chain's scratch is held for
one chunk only (about 1/10 of the whole-domain need); see rrtm_chunk for why the whole domain does
not fit. Chunk scratch is allocated on the first call and reused (one chunk shape).
"""
from __future__ import annotations

import os

from namgpu.physics.rrtm import (
    phys_rrtm_coszmn as coszmn,
    phys_rrtm_dayparts as dayparts,
    phys_rrtm_driver_call as driver_call,
    phys_rrtm_driver_post as driver_post,
    phys_rrtm_driver_prep as driver_prep,
    phys_rrtm_getgases as getgases,
    phys_rrtm_getozn as getozn,
    phys_rrtm_grrad_aerosols as grrad_aerosols,
    phys_rrtm_grrad_clouds as grrad_clouds,
    phys_rrtm_grrad_gases as grrad_gases,
    phys_rrtm_grrad_lw as grrad_lw,
    phys_rrtm_grrad_setup as grrad_setup,
    phys_rrtm_grrad_sw as grrad_sw,
    phys_rrtm_progcld2 as progcld2,
)
from namgpu.physics.rrtm import rrtm_chunk

# NAM iflip = 0 (module_SOLVER_GRID_COMP.F90:10714): input from TOA to surface.
IVFLIP_NAM = 0
ROWS = int(os.environ.get("NAMGPU_RRTM_ROWS", "32"))   # 32 rows: about 6 GB of chain scratch; 8 rows was launch-bound
KEY_CHUNK = "rrtm_call:chunkctx"
# driver-call 2-D outputs that are GRRAD actual arguments (module_RA_RRTM.F90 CALL GRRAD_NMMB).
GRRAD_SFC_ARGS = ("XLON", "XLAT", "SINLAT", "COSLAT", "SLMSK", "TSEA", "TISFC", "FICE", "ZORL",
                  "SNWDPH", "SNCOVR", "SNOALB", "HPRIME_V", "SFCALBEDO", "SMX", "CV", "CVB", "CVT",
                  "FLGMIN_L")


def _chunk_ctx(ctx, rows):
    cc = ctx.tables.get(KEY_CHUNK)
    if cc is None or cc.w != rows:
        cc = rrtm_chunk.ChunkCtx(ctx, rows)
        ctx.tables[KEY_CHUNK] = cc
    return cc


def run(ctx, args: dict, *, stamp=None, rows: int | None = None, sw: bool = True) -> None:
    """sw=False is a test-harness knob only (tools/run_rrtm_real.py --lw-only): it skips grrad-sw and says
    so in ctx.tables['rrtm_call_flags']; the forecast path always runs SW on SW steps."""
    cfg = ctx.cfg
    nts = int(args["NTIMESTEP"])
    dt = float(args["DT"])
    nrads, nradl = int(args["NRADS"]), int(args["NRADL"])
    lsswr = nts % nrads == 0
    lslwr = nts % nradl == 0
    dtsw, dtlw = nrads * dt, nradl * dt
    ntoz = int(cfg.ntoz)
    ivflip = int(getattr(cfg, "ivflip", IVFLIP_NAM))
    ctx.tables["rrtm_call_flags"] = {"lsswr": lsswr, "lslwr": lslwr, "ntimestep": nts,
                                     "solcon": float(args["SOLCON"]), "sw_run": bool(sw and lsswr)}

    def _s(name):
        if stamp is not None:
            stamp(name)

    from namgpu.state import fit_on as _fit_on  # noqa: PLC0415
    if _fit_on():
        # 32 GB fit: hand the pool's cached free blocks back before the radiation working set is allocated.
        # They are what the 47 steps since the last call left cached (dynamics and physics temporaries, in
        # their sizes); the chunk working set's large blocks cannot be cut from them, so without this they
        # sit reserved beside it at the radiation peak (333x333: 0.66 GiB reserved over the 2.88 GiB used).
        # The call already frees everything at its end; values do not depend on placement.
        import cupy as _cpf
        if ctx.stream is not None:
            ctx.stream.synchronize()
        _cpf.get_default_memory_pool().free_all_blocks()
    _s("driver-prep")
    driver_prep.run(ctx, ntimestep=nts, jdat=args["JDAT"], dt=dt, nrads=nrads, nradl=nradl, columns=False)
    prep = ctx.tables[driver_prep.PRE_KEY]

    # P_PHY (layer pressure) for this call, full domain, before the chunk loop: driver-call reads it
    # ahead of grrad-setup in each chunk (module_RA_RRTM.F90:486-487 sets it before the column work).
    # Same formula grrad-setup writes; without this the first call has no P_PHY and later calls read
    # the previous call's PD (run lead 0935Z: KeyError 'P_PHY' at step 0). This is the module_RA_RRTM.F90:486-487 prologue
    # on the whole domain, the same step phys-rrtm-driver-prep does per row.
    import cupy as _cp
    _g, _st = ctx.grid, ctx.st
    _lm, _jm, _im = int(_g.lm), int(_g.jm), int(_g.im)
    _ppy = _st.add("P_PHY", (_lm, _jm, _im))
    _ppy[...] = (_cp.asarray(_st["SGML2"]).reshape(_lm, 1, 1) * _st["PD"][None]
                 + _cp.asarray(_g.psgml1).reshape(_lm, 1, 1))

    jm = int(ctx.grid.jm)
    from namgpu.state import FIT_RRTM_ROWS, fit_on  # noqa: PLC0415
    _fit = fit_on()
    if not rows:
        # NAMGPU_RRTM_ROWS wins; otherwise 32 rows on the full path, 8 on the 32 GB fit (16 rows ran out of memory at the 5th radiation call)
        rows = ROWS if ("NAMGPU_RRTM_ROWS" in os.environ or not _fit) else FIT_RRTM_ROWS
    rows = min(int(rows), jm)
    cc = _chunk_ctx(ctx, rows)
    if _fit and rrtm_chunk.workspace(cc) is None:
        # one g-point workspace shared in time by swrad's chunk scratch and the LW taug/fracs pair (see
        # rrtm_chunk.WS_KEY): max of the two, not their sum
        import numpy as _np
        from namgpu.physics.rrtm import lw_core as _lwc
        from namgpu.physics.rrtm import phys_rrtm_sw_swrad as _swr
        _nlay = _lm + _swr.LTP
        _need = max(_lwc.pair_floats(_nlay, rows, _im),
                    _swr.chunk_scratch_floats(_swr.sw_ncap(rows * _im), _nlay, _nlay + 1))
        cc.tables[rrtm_chunk.WS_KEY] = _cp.empty(_need, dtype=_np.float32)
    _s("grrad-chunks")
    for (w0, w1, a, b) in rrtm_chunk.windows(jm, rows):
        cc.begin(w0, w1)
        driver_prep.run_columns(cc, prep)
        dayparts.run(cc)
        dc = driver_call.run(cc, ntoz=ntoz, nrads=nrads, nradl=nradl, dt=dt)
        # The GRRAD actual arguments (module_RA_RRTM.F90:1013-1046: XLON, XLAT, SLMSK, TSEA, ...) are the
        # driver-call outputs; the grrad units read them from the chunk state by those names.
        for _n in GRRAD_SFC_ARGS:
            cc.st.local[_n] = dc[_n]
        grrad_setup.run(cc, dtsw=dtsw, dtlw=dtlw, ivflip=ivflip, ntoz=ntoz)
        if ntoz <= 0:
            getozn.run(cc, ivflip=ivflip)
        coszmn.run(cc)
        getgases.run(cc, ico2flg=int(cfg.ico2flg), ivflip=ivflip)
        grrad_gases.run(cc, ivflip=ivflip)
        grrad_aerosols.run(cc, lsswr=lsswr, lslwr=lslwr)
        grrad_clouds.run(cc, dep=progcld2.dep)
        if lsswr and sw:
            grrad_sw.run(cc)
        grrad_lw.run(cc, sw_done=True)
        driver_post.run(cc, ntimestep=nts)
        cc.st.writeback(a, b)
    if _fit and os.environ.get("NAMGPU_RRTM_KEEP_CHUNK", "") != "1":
        # 32 GB fit: the chunk context holds the whole RRTM working set (chunk copies, chunk-local
        # scratch and tables: swrad chunk_scratch alone is 2 x 0.9 GB). Held between radiation calls it sat
        # idle through 47 of every 48 steps and pushed the step peak past a 32 GB card. Drop it; the next
        # call rebuilds it exactly as the first call does (zero-filled, one shape).
        _cc = ctx.tables.pop(KEY_CHUNK, None)
        if _cc is not None:
            _cc.release()                    # its arrays back to the pool now, before devmem.release below
        # driver-prep's per-call P8W/P_PHY and record are rebuilt at the start of every call (its column
        # scratch is chunk-local and went with the chunk context); the small rrtm_prep:vec:* vectors stay
        for _k in [k for k in ctx.tables if k.startswith("rrtm_prep:") and not k.startswith("rrtm_prep:vec:")]:
            del ctx.tables[_k]
        # give the freed radiation working set back whole, so the next 47 steps and the next call start from
        # unsplit blocks (pool fragmentation: the 4th call OOMed at ROWS=16 with 27.7 GiB in the pool)
        from namgpu import devmem
        devmem.release(ctx.stream)
    _s("rrtm-done")
