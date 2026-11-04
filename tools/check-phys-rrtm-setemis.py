"""Done-rule check for phys-rrtm-setemis (review fix: SICE mask, emissivity table wiring, no stand-ins).

    source ~/nam-gpu/env.sh
    python tools/check-phys-rrtm-setemis.py

Runs the kernel once on the full 954 x 835 grid. Inputs are synthetic and deterministic (labelled):
  SM and SICE: a random sea/land mask, and a random sea-ice block (SICE 0.8 where chosen, else 0).
  SI = 0 and SNOWC = 0, so the snow term does not fire and the expected values are the emsref entries.
  The emissivity table is a synthetic sfc_emissivity_idx.txt of all 3s, loaded through the real
  phys-rrtm-sfc-init run (IEMS = 1), so land points read it through the kernel (emsref(3) = 0.94).
  STDH is nonzero random (the refusal check below covers the all-zero case).
Expected: ice (SICE > 0.5) = 0.96 whatever SM says (emsref(7)), sea = 0.97, land = 0.94. All finite.
Refusals: STDH all zero and phys-rrtm-sfc-init not run (sfc_flags missing) must raise.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _MiniState:
    """Only the fields this unit touches (a full State allocates every table field)."""

    def __init__(self, cp):
        self.fields = {}
        self._cp = cp

    def add(self, name, shape, dtype="float32"):
        arr = self._cp.zeros(shape, dtype=dtype)
        self.fields[name] = arr
        return arr


def _expect_raises(fn, label):
    try:
        fn()
    except RuntimeError as e:
        print(f"refusal ok: {label}: {str(e)[:120]}")
        return True
    print(f"refusal MISSING: {label}")
    return False


def _write_table(path: Path):
    """Line 1 is the a80 tag; then 810 records of 80 one-digit integers, all 3."""
    rec = "3" * 80
    path.write_text("tag\n" + "\n".join([rec] * 810) + "\n")


def main():
    import cupy as cp
    import tempfile
    from namgpu import config, grid as gridmod, kernels
    from namgpu.driver import Ctx
    from namgpu.physics.rrtm import phys_rrtm_setemis as unit
    from namgpu.physics.rrtm import phys_rrtm_sfc_init as sfc
    from namgpu.physics.rrtm import phys_rrtm_rad_initialize as radinit
    from namgpu.physics.rrtm.phys_rrtm_rad_initialize import RadInitArgs

    cfg = config.load()
    g = gridmod.build(cfg)
    im, jm = int(g.im), int(g.jm)
    shp = (jm, im)
    rng = cp.random.default_rng(11)
    stream = cp.cuda.Stream(non_blocking=True)
    scratch = Path(tempfile.mkdtemp(dir=ROOT / "scratch", prefix="setemis-check-"))
    try:
        _write_table(scratch / sfc.SEMIS_FILE)

        def make_ctx(stdh_zero=False, with_sfc_init=True):
            st = _MiniState(cp)
            f = st.fields
            f["GLON"] = st.add("GLON", shp)
            f["GLON"][...] = rng.random(shp, dtype=cp.float32) * 2.0 * 3.14159265
            f["GLAT"] = st.add("GLAT", shp)
            f["GLAT"][...] = (rng.random(shp, dtype=cp.float32) - 0.5) * 3.0
            f["SM"] = st.add("SM", shp)
            f["SM"][...] = cp.floor(2.0 * rng.random(shp, dtype=cp.float32))
            f["SICE"] = st.add("SICE", shp)
            f["SICE"][...] = cp.where(rng.random(shp, dtype=cp.float32) < 0.1, 0.8, 0.0).astype(cp.float32)
            st.add("SI", shp)
            st.add("SNOWC", shp)
            f["Z0"] = st.add("Z0", shp)
            f["Z0"][...] = 0.05 + 0.5 * rng.random(shp, dtype=cp.float32)
            f["TSKIN"] = st.add("TSKIN", shp)
            f["TSKIN"][...] = 250.0 + 50.0 * rng.random(shp, dtype=cp.float32)
            f["STDH"] = st.add("STDH", shp)
            if not stdh_zero:
                f["STDH"][...] = 500.0 * rng.random(shp, dtype=cp.float32)
            ctx = Ctx(cfg=cfg, grid=g, st=st, stream=stream, dt=cfg.dt)
            if with_sfc_init:
                args = RadInitArgs(iems=1)
                radinit.run(ctx, args)
                ctx.tables["rad_init_args"] = args
                sfc.run(ctx, mode=0, ialb=args.ialb, iems=args.iems, ems_dir=str(scratch))
            return ctx

        kernels.module(unit.KERNEL).compile()

        ctx = make_ctx()
        with stream:
            out = unit.run(ctx)["sfcemis"]
        stream.synchronize()
        ice = ctx.st.fields["SICE"] > 0.5
        sm = ctx.st.fields["SM"]
        land = (~ice) & (sm == 0)
        sea = (~ice) & (sm == 1)
        finite = bool(cp.isfinite(out).all())
        err_ice = float(cp.abs(out[ice] - 0.96).max()) if bool(ice.any()) else -1.0
        err_land = float(cp.abs(out[land] - 0.94).max()) if bool(land.any()) else -1.0
        err_sea = float(cp.abs(out[sea] - 0.97).max()) if bool(sea.any()) else -1.0
        print(f"grid {im}x{jm}: finite={finite} ice={int(ice.sum())} land={int(land.sum())} sea={int(sea.sum())}")
        print(f"max |err|: ice vs 0.96 = {err_ice:.2e}, land vs 0.94 = {err_land:.2e}, sea vs 0.97 = {err_sea:.2e}")
        ok = finite and max(err_ice, err_land, err_sea) <= 1e-6

        ok &= _expect_raises(lambda: unit.run(make_ctx(stdh_zero=True)), "STDH all zero")
        ok &= _expect_raises(lambda: unit.run(make_ctx(with_sfc_init=False)), "phys-rrtm-sfc-init not run")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
