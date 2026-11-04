"""drv-physics-initialize: PHYSICS_INITIALIZE for the NAM 12 km parent (non-GFS branch).

NOAA / NCEP NAM (nam.v4.2.11, 83df4e13c1), nmm/module_SOLVER_GRID_COMP.F90:9936-11062, SUBROUTINE
PHYSICS_INITIALIZE, ELSE branch of IF(GFS) (the active 12 km path: longwave, shortwave = rrtm;
turbulence, sfc_layer = myj; land_surface = noah; convection = bmj; microphysics = fer_hires;
gwdflg true). Kernel: namgpu/kernels/drv_physics_initialize.cu (physinit_select, physinit_gpkap,
physinit_lowlyr).

Call order (Fortran order, each sub-unit is its own wrapper; this unit only dispatches):
  1. Scheme selection, 10768-10789: rrtm + thompson needs np3d == 8 (else refused), ICICE 4 for thompson,
     ICLIQ 0 for np3d == 5. Done in physinit_select; the result is read back once (init only).
  2. GPKAP, 10638-10639 (funcphys gpkap): tables/gpkap, from physinit_gpkap.
  3. GPVS_HR, 10641: phys-fer-gpvs (its table is the same one FERRIER_INIT_HR builds at 2509).
  4. rad_initialize_nmmb, 10797: phys-rrtm-rad-initialize (RadInitArgs carry the selection result), then
     radinit_nmmb through phys-rrtm-radinit (dep_calls=True runs its seven siblings in order).
  5. RRTM_INIT, 10824: phys-rrtm-init.
  6. ALLOCATE(LOWLYR), 10811, zeroed by physinit_lowlyr (placeholder, filled by TURBULENCE).
  7. JSFC_INIT, 10864: phys-jsfc-init (sfc_layer myj).
  8. MYJPBL_INIT, 10895: phys-myj-init (turbulence myj).
  9. NOAH_LSM_INIT, 10914: phys-noah-parm-tables then phys-noah-init (land_surface noah).
 10. BMJ_INIT, 10941-10946: CU_PHYSICS = 2, phys-bmj-init (convection bmj).
 11. FERRIER_INIT_HR, 10992-11006: DT_MICRO = NPRECIP*DT. DELX/DELY are unused by phys-fer-init (see its
     docstring); phys-fer-init (microphysics fer_hires).
 12. GWD_init, 11030-11041: DTPHS = DT*NPHS, phys-gwd-init.

Anything other than the active 12 km scheme set is refused by name: the Fortran selects a different
branch there, and this unit does not run it.

Reads:  ctx.cfg: shortwave, longwave, turbulence, sfc_layer, land_surface, convection, microphysics,
        np3d, nprecip, dt, nphs; ctx.restart (default False); ctx.grid (im, jm, lm).
Writes: ctx.tables["gpkap"]      dict: tbpkap float32 [NXPKAP=11001], c1xpkap, c2xpkap, nxpkap
        ctx.tables["cu_physics"] = 2   (int_state%CU_PHYSICS for bmj)
        ctx.tables["phys_select"] = dict of the selection result (icice_sw, icice_lw, iclq_sw, iclq_lw)
        ctx.tables["fer_hires_caller"] = DT_MICRO, DELX, DELY as PHYSICS_INITIALIZE computes them (10992-10994)
        ctx.st["LOWLYR"]         int32 [JM, IM] = 0
        plus every table the sub-units write.
Launch: small kernels on ctx.stream; the one readback is the 5-int selection result (init, not per step).
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "drv_physics_initialize"
NXPKAP = 11001
# NAM_NOAH_FIX_DIR (the NCEP NAM fix tables) when set, else namgpu/data/fix/noah built by tools/build_noah_fix.py
# (IGBP-first tables in the NMMB layout; named gap until the NCEP NAM fix VEGPARM.TBL is found).
from namgpu import paths  # noqa: E402

NOAH_TBL_DIR = Path(os.environ.get("NAM_NOAH_FIX_DIR") or paths.FIX / "noah")

ACTIVE = {
    "shortwave": "rrtm", "longwave": "rrtm", "turbulence": "myj", "sfc_layer": "myj",
    "land_surface": "noah", "convection": "bmj", "microphysics": "fer_hires",
}


class PhysicsInitRefusal(RuntimeError):
    """A scheme choice the active 12 km path does not run; the message names the Fortran branch."""


def _check_scheme(cfg) -> None:
    for key, want in ACTIVE.items():
        got = str(getattr(cfg, key)).strip()
        if got != want:
            raise PhysicsInitRefusal(
                f"drv-physics-initialize: {key}={got!r}; this unit ports the NAM 12 km branch "
                f"({key}={want!r}) of PHYSICS_INITIALIZE. Other branches are not ported and are refused.")


def _select(ctx, cp) -> dict:
    """physinit_select on the device; one readback of five ints at init."""
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs

    cfg = ctx.cfg
    d = RadInitArgs()   # Fortran defaults set by the caller (10646-10722)
    params = np.array([
        int(str(cfg.shortwave).strip() == "rrtm"),
        int(str(cfg.microphysics).strip() == "thompson"),
        int(cfg.np3d),
        int(d.icice_sw), int(d.icice_lw), int(d.icliq_sw), int(d.icliq_lw),
    ], dtype=np.int32)
    p_dev = cp.asarray(params)
    sel = cp.zeros(5, dtype=cp.int32)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, "physinit_select")((1,), (1,), (p_dev, sel))
    out = cp.asnumpy(sel)
    if int(out[0]) != 0:
        raise PhysicsInitRefusal(
            f"drv-physics-initialize: shortwave rrtm with microphysics thompson needs np3d=8, got np3d={cfg.np3d}; "
            "NMMB_FINALIZE (module_SOLVER_GRID_COMP.F90:10771)")
    return {"icice_sw": int(out[1]), "icice_lw": int(out[2]),
            "iclq_sw": int(out[3]), "iclq_lw": int(out[4])}


def _gpkap(ctx, cp) -> dict:
    tables = ctx.tables
    if "gpkap" in tables:
        return tables["gpkap"]
    tbl = cp.zeros(NXPKAP, dtype=cp.float32)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()
    with scope:
        kernels.get(UNIT, "physinit_gpkap")(((NXPKAP + 255) // 256,), (256,), (tbl, np.int32(NXPKAP)))
    # xinc = 110000/11000 = 10, c1 = 1 - xmin/xinc, c2 = 1/xinc (funcphys.f:2213-2214), single precision.
    xinc = np.float32(110000.0) / np.float32(NXPKAP - 1)
    entry = {"tbpkap": tbl, "nxpkap": NXPKAP,
             "c1xpkap": float(np.float32(1.0) - np.float32(0.0) / xinc),
             "c2xpkap": float(np.float32(1.0) / xinc)}
    tables["gpkap"] = entry
    return entry


FIX_DIR = paths.FIX   # NCEP fix/am files, shipped in the package (tools/fetch_fix.sh refreshes them)


def run(ctx, *, restart: bool | None = None, noah_tbl_dir: str | Path | None = None):
    """Run PHYSICS_INITIALIZE for the NAM 12 km parent. Returns ctx.tables.

    The Fortran opens the radiation fix files (global_o3clim.txt, co2historicaldata_YYYY.txt,
    sfc_emissivity_idx.txt, solarconstant_*.txt) by name in the run directory, where exnam_fcst.sh
    links them. Our run directory for those files is ctx.fix_dir, else $NAMGPU_FIX_DIR, else
    namgpu/data/fix (shipped in the package); init runs with that directory as the working directory and
    restores the caller's afterwards."""
    with in_run_dir(ctx):
        return _run(ctx, restart=restart, noah_tbl_dir=noah_tbl_dir)


def run_dir(ctx) -> Path:
    """The directory the Fortran would run in for its fix-file opens: ctx.fix_dir, $NAMGPU_FIX_DIR, namgpu/data/fix."""
    fix = Path(getattr(ctx, "fix_dir", None) or os.environ.get("NAMGPU_FIX_DIR") or FIX_DIR)
    if not (fix / "global_o3clim.txt").is_file():
        raise FileNotFoundError(f"drv-physics-initialize: no NCEP fix files in {fix}; the package ships them in "
                                "namgpu/data/fix (reinstall woof-nam, or point NAMGPU_FIX_DIR at a folder "
                                "filled by tools/fetch_fix.sh)")
    return fix


@contextlib.contextmanager
def in_run_dir(ctx):
    """Run with the fix directory as the working directory (the NMMB run directory, where the radiation
    init and gas_update open global_o3clim.txt and co2historicaldata_YYYY.txt by name), then restore it.
    drv-time-loop wraps the whole forecast in it, because gas_update reopens the CO2 file in the loop."""
    fix = run_dir(ctx)
    cwd = os.getcwd()
    os.chdir(fix)
    try:
        yield fix
    finally:
        os.chdir(cwd)


def _run(ctx, *, restart, noah_tbl_dir):
    import cupy as cp

    from namgpu.physics.bmj import phys_bmj_init
    from namgpu.physics.fer_hires import phys_fer_gpvs, phys_fer_init
    from namgpu.physics.gwd import phys_gwd_init
    from namgpu.physics.jsfc import phys_jsfc_init
    from namgpu.physics.myj import phys_myj_init
    from namgpu.physics.noah import phys_noah_init, phys_noah_parm_tables
    from namgpu.physics.rrtm import phys_rrtm_init, phys_rrtm_radinit
    from namgpu.physics.rrtm import phys_rrtm_rad_initialize
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs

    cfg, g, st = ctx.cfg, ctx.grid, ctx.st
    if restart is None:
        restart = bool(getattr(ctx, "restart", False))
    if ctx.tables is None:
        ctx.tables = {}
    _check_scheme(cfg)
    im, jm = int(g.im), int(g.jm)
    scope = ctx.stream if getattr(ctx, "stream", None) is not None else contextlib.nullcontext()

    # 1-2. selection (10768-10789), GPKAP (10638-10639)
    sel = _select(ctx, cp)
    ctx.tables["phys_select"] = sel
    _gpkap(ctx, cp)

    # 3. GPVS_HR (10641)
    phys_fer_gpvs.run(ctx)

    # 4. rad_initialize_nmmb (10797) + radinit_nmmb (through its seven siblings)
    args = RadInitArgs(
        num_p3d=int(cfg.np3d),
        iaer=int(cfg.iaer), iems=int(cfg.iems),
        ntcw=int(cfg.ntcw), ntoz=int(cfg.ntoz),
        icice_sw=sel["icice_sw"], icice_lw=sel["icice_lw"],
        icliq_sw=sel["iclq_sw"], icliq_lw=sel["iclq_lw"],
    )
    ctx.tables["rad_init_args"] = args     # the record phys-rrtm-cld-init reads (ivflip, icmphys, icldflg, iovr)
    phys_rrtm_rad_initialize.run(ctx, args)
    # sfc_init (grrad_nmmb.f:413): the emissivity-type table sfc_idxems that phys-rrtm-setemis reads
    # when IEMS selects the index scheme. Host file read at init only; the table is required then.
    from namgpu.physics.rrtm import phys_rrtm_sfc_init
    phys_rrtm_sfc_init.run(ctx, mode=0, ialb=args.ialb, iems=args.iems)
    phys_rrtm_radinit.run(ctx, dep_calls=True)

    # 5. RRTM_INIT (10824)
    phys_rrtm_init.run(ctx)

    # 6. ALLOCATE(LOWLYR) (10811), zeroed
    lowlyr = st.add("LOWLYR", (jm, im), dtype="int32")
    with scope:
        kernels.get(UNIT, "physinit_lowlyr")(
            ((im + 31) // 32, (jm + 7) // 8, 1), (32, 8, 1), (lowlyr, np.int32(im), np.int32(jm)))

    # 7. JSFC_INIT (10864), sfc_layer myj
    phys_jsfc_init.run(ctx, restart=restart)

    # 8. MYJPBL_INIT (10895), turbulence myj
    phys_myj_init.run(ctx, restart=restart)

    # 9. NOAH_LSM_INIT (10914), land_surface noah: parameter tables first
    phys_noah_parm_tables.run(ctx, tbl_dir=noah_tbl_dir or NOAH_TBL_DIR,
                              ivegsrc=int(cfg.raw["ivegsrc"]))
    phys_noah_init.run(ctx, restart=restart)

    # 10. BMJ_INIT (10941-10946), convection bmj
    ctx.tables["cu_physics"] = 2
    phys_bmj_init.run(ctx, restart=restart)

    # 11. FERRIER_INIT_HR (10992-11006), microphysics fer_hires. The caller's DT_MICRO, DELX, DELY are
    # recorded here (single precision, Fortran operation order); phys-fer-init reads DT_MICRO itself.
    f32 = np.float32
    dt_micro = f32(cfg.nprecip) * f32(cfg.dt)
    delx = (f32(-2.0) * f32(cfg.wbd)) * f32(111.3) / f32(im)   # DX at rotated equator (km)
    dely = (f32(-2.0) * f32(cfg.sbd)) * f32(111.3) / f32(jm)   # DY at rotated equator (km)
    ctx.tables["fer_hires_caller"] = {"dt_micro": float(dt_micro), "delx": float(delx), "dely": float(dely)}
    phys_fer_init.run(ctx, restart=restart)

    # 12. GWD_init (11030-11041): DTPHS = DT*NPHS
    dtphs = float(np.float32(cfg.dt) * np.float32(cfg.nphs))
    phys_gwd_init.run(ctx, dtphs=dtphs)

    return ctx.tables
