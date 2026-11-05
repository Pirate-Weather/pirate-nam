"""CPU checks of the installed package's front door (no GPU, no network).

Run: python tests/test_cli.py   (or pytest tests/test_cli.py). CI runs it against the built wheel.
  - every `woof-nam` subcommand answers --help with exit 0;
  - everything a run reads ships inside the package (parm, UNITS.json, NOAA fix files, Noah tables,
    ArWen noah.py, kernels), so a wheel install needs no source checkout;
  - the box plumbing is off unless WOOF_NAM_BOX=1: gpu_lease() never starts the pool runner.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import cli, paths  # noqa: E402

SUBCOMMANDS = [[], ["fetch"], ["static"], ["init"], ["run"], ["maps"], ["side-by-side"],
               ["side-by-side", "mrms"], ["side-by-side", "archive"], ["side-by-side", "woof-nam"],
               ["side-by-side", "sheets"], ["products"], ["compare"], ["columns"], ["licence-check"]]


def _help(args):
    buf = io.StringIO()
    argv0 = sys.argv[0]
    try:
        with contextlib.redirect_stdout(buf):
            try:
                rc = cli.main(args + ["--help"])
            except SystemExit as e:
                rc = e.code or 0
    finally:
        sys.argv[0] = argv0
    return rc, buf.getvalue()


def test_help():
    for args in SUBCOMMANDS:
        rc, out = _help(args)
        assert rc == 0, (args, rc)
        assert "usage: woof-nam" in out, (args, out[:200])


def test_unknown_command():
    with contextlib.redirect_stderr(io.StringIO()):
        assert cli.main(["no-such-command"]) == 2


def test_package_data():
    need = [paths.PARM / "nam_configfile", paths.PARM / "nam_climaeropac_global.txt", paths.UNITS_JSON,
            paths.FIX / "global_o3clim.txt", paths.FIX / "co2historicaldata_2025.txt",
            paths.FIX / "sfc_emissivity_idx.txt", paths.FIX / "noah" / "VEGPARM.TBL",
            paths.DATA / "noah_tables" / "SOILPARM.TBL", paths.ARWEN / "core" / "noah.py",
            paths.ARWEN / "LICENSE", paths.PKG / "kernels" / "drv_gpu_lease.cu"]
    missing = [str(p) for p in need if not p.is_file()]
    assert not missing, missing
    from namgpu import config
    assert config.PARM == paths.PARM


def test_box_plumbing_off_by_default():
    from namgpu.steps import drv_gpu_lease as L
    old = os.environ.pop("WOOF_NAM_BOX", None)
    try:
        assert not paths.box_mode()
        real = L.POOL_RUN
        L.POOL_RUN = Path(sys.executable)          # a runner that "exists": the lease must still not start
        try:
            with L.gpu_lease("test", 1, 1) as card:
                assert card == os.environ.get("CUDA_VISIBLE_DEVICES", "")
        finally:
            L.POOL_RUN = real
        os.environ["WOOF_NAM_BOX"] = "1"
        assert paths.box_mode()
    finally:
        os.environ.pop("WOOF_NAM_BOX", None)
        if old is not None:
            os.environ["WOOF_NAM_BOX"] = old


def test_domains_table():
    """Every domain row names a configfile that ships, starts from known sources and selects its grid."""
    from namgpu import config, domains
    old = os.environ.pop(config.CONFIG_ENV, None)
    try:
        assert domains.get(None)["name"] == "conus12"
        for n in domains.names():
            row = domains.get(n)
            assert Path(row["configfile_path"]).is_file(), row
            assert set(row["sources"]) <= {"native", "gfs"} and row["static"] in domains.table()["static_kinds"]
            domains.select(n)
            assert domains.current()["name"] == n
        domains.select("conus4")
        cfg = config.load()
        assert (cfg.im, cfg.jm, cfg.lm) == (1371, 1101, 60) and config.static_npz(cfg) is not None
        domains.select("conus5")
        cfg = config.load()
        assert (cfg.im, cfg.jm, cfg.lm) == (1097, 881, 60) and config.static_npz(cfg).name == "nam5km_static.npz"
        domains.select("conus12")
        cfg = config.load()
        assert (cfg.im, cfg.jm) == (954, 835) and config.static_npz(cfg) is None
        try:
            domains.select("conus4", "native")          # the archived analysis is on the 12 km grid only
            raise AssertionError("conus4 took a native start")
        except SystemExit:
            pass
    finally:
        os.environ.pop(config.CONFIG_ENV, None)
        if old is not None:
            os.environ[config.CONFIG_ENV] = old


def test_static_soil_set():
    """The geog static reads the permissive soil set unless NAMGPU_GEOG_SOIL=wrf (an opt-in that names its hold),
    and finds each dataset across a partial NAMGPU_GEOG that links topo into the full tree."""
    import tempfile

    from namgpu.init import static_build as sb
    keys = ("NAMGPU_GEOG", "WPS_GEOG", "NAMGPU_GEOG_SOIL")
    old = {k: os.environ.pop(k, None) for k in keys}
    try:
        with tempfile.TemporaryDirectory() as td:
            full = Path(td) / "WPS_GEOG"
            for rel in ("topo_gmted2010_30s", "soilgrids/texture_top", "soiltemp_era5_1deg", "soiltype_top_30s",
                        "soiltemp_1deg"):
                (full / rel).mkdir(parents=True)
                (full / rel / "index").write_text("type=categorical\n")
            part = Path(td) / "nam-geog"
            part.mkdir()
            (part / "topo_gmted2010_30s").symlink_to(full / "topo_gmted2010_30s")
            os.environ["NAMGPU_GEOG"] = str(part)
            roots = sb.geog_roots()
            assert roots[:2] == [part, full], roots
            assert sb.soil_set() == "permissive"
            assert sb.dataset(roots, sb.SOIL_SETS["permissive"]["soil"]) == full / "soilgrids/texture_top"
            assert sb.dataset(roots, "topo_gmted2010_30s") == part / "topo_gmted2010_30s"
            os.environ["NAMGPU_GEOG_SOIL"] = "wrf"
            assert sb.dataset(roots, sb.SOIL_SETS[sb.soil_set()]["soiltemp"]) == full / "soiltemp_1deg"
            for k in ("terrain", "landuse", "greenfrac", "albedo", "maxsnowalb"):
                (full / sb.DATASETS[k]["dataset"]).mkdir(parents=True, exist_ok=True)
                (full / sb.DATASETS[k]["dataset"] / "index").write_text("type=continuous\n")
            ds = sb.resolve_datasets(roots, soil="wrf")
            assert ds["soil_top"]["path"] == str(full / "soiltype_top_30s") and "FAO" in ds["soil_top"]["source_and_licence"]
            try:
                sb.resolve_datasets(roots, {"soil_top": "soiltype_top_30s"})
                raise AssertionError("held dataset accepted without the opt-in soil set")
            except sb.LicenceHold:
                pass
            assert sb.resolve_datasets(roots)["soil_top"]["path"] == str(full / "soilgrids/texture_top")
            os.environ["NAMGPU_GEOG_SOIL"] = "statsgo"
            try:
                sb.soil_set()
                raise AssertionError("unknown soil set accepted")
            except SystemExit:
                pass
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_column_view_faces():
    """The column store's WRF view: faces average back to the mass values the reader destaggers to."""
    import numpy as np

    from namgpu.io.columns import _faces
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for axis in (-1, -2, -3):
        assert _faces(x, axis).shape[axis % 3] == x.shape[axis % 3] + 1
    f = _faces(x, -1)
    # the reader's mass value i is (f[i] + f[i + 1]) / 2: a 1-2-1 average inside, the value itself at the ends
    back = 0.5 * (f[..., :-1] + f[..., 1:])
    assert np.allclose(back[..., 1:-1], 0.25 * x[..., :-2] + 0.5 * x[..., 1:-1] + 0.25 * x[..., 2:])


def test_fetch_window():
    """fetch --hours N plans exactly the GFS boundary times an N-hour run reads (0.1.0 fetched 48 h inside init)."""
    from datetime import datetime

    from namgpu.domains import archive_cycle, bc_hours
    from namgpu.init import fetch_case
    assert [bc_hours(h) for h in (1, 6, 7, 24, 48, 84)] == [6, 6, 9, 24, 48, 84]
    cyc = datetime(2026, 11, 4, 12)

    def gfs(hours, **kw):
        jobs, _ = fetch_case.plan(cyc, hours, Path("raw"), source="gfs", **kw)
        return sorted(str(j[2].name) for j in jobs if "gfs" in str(j[2]))
    assert gfs(6) == ["gfs.t06z.f006.grib2", "gfs.t06z.f009.grib2", "gfs.t06z.f012.grib2",
                      "gfs.t12z.pgrb2.0p25.f000"]     # the start itself (FETCH_ROWS gfs-f000)
    assert gfs(1) == gfs(6)
    assert len(gfs(24)) == 10
    # --inputs-only (init_fetch_case): only the init and boco rows, no referee or truth files
    rows = tuple(r for r in fetch_case.FETCH_ROWS if r["role"] not in ("referee", "truth"))
    only, _ = fetch_case.plan(datetime(2025, 3, 14, 12), 24, Path("raw"), source="native", rows=rows)
    assert [j[2].name for j in only if "nam" in str(j[2])] == ["bgrd3d00.grib2"]
    assert not [j for j in only if "mrms" in str(j[2])]
    assert archive_cycle(cyc) == datetime(2025, 11, 4) and archive_cycle(datetime(2026, 11, 2, 12)) == datetime(2026, 11, 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("test_cli: PASS (namgpu from %s)" % paths.PKG)
