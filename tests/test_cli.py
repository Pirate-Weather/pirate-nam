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
               ["side-by-side", "sheets"], ["licence-check"]]


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("test_cli: PASS (namgpu from %s)" % paths.PKG)
