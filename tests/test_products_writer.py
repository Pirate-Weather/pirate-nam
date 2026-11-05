"""`woof-nam products` finds its GRIB2 writer on a plain pip install, or says in one line how to get it.

Origin: recast-woof-nam 0.1.1 from PyPI (2026-11-05): `woof-nam products` failed every hour with FileNotFoundError
".../gpuwm/libexec/bridges/rw_rrfs_upp ... not in a gpuwm release yet", after regridding each hour first, because
the writer it calls ships in no gpuwm release. The fix stages woof-nam's own pinned writer (namgpu/exporter.py).
CPU only, no network: the "download" is a file:// folder. Run: python tests/test_products_writer.py
(WOOF_NAM_TEST_INSTALLED=1 runs it against an installed package).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(ROOT))

ENV_KEYS = ("WOOF_NAM_RW_RRFS_UPP", "GPUWM_BRIDGES", "WOOF_NAM_HOME", "WOOF_NAM_BRIDGE_URL", "WOOF_NAM_NO_FETCH")


def _clean_env(tmp: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(HOME=str(tmp / "home"), WOOF_NAM_HOME=str(tmp / "home" / ".woof-nam"),
               GPUWM_BRIDGES=str(tmp / "empty-bridges"), WOOF_NAM_BRIDGE_URL=(tmp / "no-assets").as_uri(),
               NAMGPU_ROOT=str(tmp / "work"))
    env.update(extra)
    for d in ("home", "empty-bridges", "no-assets", "work"):
        (tmp / d).mkdir(exist_ok=True)
    if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_missing_writer_is_one_line():
    """No writer and none to fetch: rc 2, one line naming the writer and `woof-nam fetch-bridges`, no traceback,
    and no hour is read first (0.1.1: rc 1, a FileNotFoundError per hour after its regrid)."""
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        run = tmp / "work" / "runs" / "r"
        env = _clean_env(tmp)
        run.mkdir(parents=True)
        (run / "wrfout_d01_2025-03-14_12:00:00").write_bytes(b"")     # never opened: the check comes first
        p = subprocess.run([sys.executable, "-c", "import sys; from namgpu import cli; sys.exit(cli.main(sys.argv[1:]))",
                            "products", "--run", str(run)], env=env, cwd=tmp, capture_output=True, text=True,
                           timeout=120)
        err = p.stderr.strip()
        assert p.returncode == 2, (p.returncode, p.stdout[-800:], err[-800:])
        assert "Traceback" not in err and "Traceback" not in p.stdout, err[-800:]
        lines = err.splitlines()                  # the one refusal, after the download notice it tried first
        assert [ln for ln in lines if not ln.startswith("woof-nam: fetching rw_rrfs_upp")] == lines[-1:], err
        assert lines[-1].startswith("woof-nam products: ") and "rw_rrfs_upp" in lines[-1], err
        assert "woof-nam fetch-bridges" in lines[-1], err
        assert not (run / "products" / "index.json").exists()


def _fake_asset(tmp: Path, body: bytes):
    from namgpu import exporter
    assets = tmp / "assets"
    assets.mkdir(exist_ok=True)
    (assets / exporter.PINS["linux-x86_64"]["asset"]).write_bytes(body)
    return assets, {"linux-x86_64": dict(exporter.PINS["linux-x86_64"], bytes=len(body),
                                         sha256=hashlib.sha256(body).hexdigest())}


def test_fetch_stages_pinned_copy():
    """The asset is downloaded, checked against the pins, installed executable, and found from then on."""
    from namgpu import exporter
    if exporter.platform_key() is None:
        print("  skipped: no prebuilt writer for this platform")
        return
    body = b"#!/bin/sh\necho fake writer\n"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assets, pins = _fake_asset(tmp, body)
        saved_env, saved_pins = dict(os.environ), exporter.PINS
        try:
            os.environ.clear()
            os.environ.update(_clean_env(tmp, WOOF_NAM_BRIDGE_URL=assets.as_uri()))
            exporter.PINS = pins
            assert exporter.find() is None
            p = exporter.ensure()
            assert p == exporter.staged_path() and p.read_bytes() == body and os.access(p, os.X_OK), p
            assert exporter.find() == p
            assert subprocess.run([str(p)], capture_output=True, text=True).stdout.strip() == "fake writer"
            # GPUWM_BRIDGES with a writer in it, then WOOF_NAM_RW_RRFS_UPP, take precedence
            (tmp / "empty-bridges" / "rw_rrfs_upp").write_bytes(b"x")
            assert exporter.find() == tmp / "empty-bridges" / "rw_rrfs_upp"
            os.environ["WOOF_NAM_RW_RRFS_UPP"] = str(tmp / "mine")
            assert exporter.find() == tmp / "mine"
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
            exporter.PINS = saved_pins


def test_wrong_bytes_refused():
    """A file that is not the pinned writer is deleted, never installed, and the refusal is one line."""
    from namgpu import exporter
    if exporter.platform_key() is None:
        return
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        assets, pins = _fake_asset(tmp, b"#!/bin/sh\necho fake writer\n")
        (assets / pins["linux-x86_64"]["asset"]).write_bytes(b"#!/bin/sh\necho tampered\n")
        saved_env, saved_pins = dict(os.environ), exporter.PINS
        try:
            os.environ.clear()
            os.environ.update(_clean_env(tmp, WOOF_NAM_BRIDGE_URL=assets.as_uri()))
            exporter.PINS = pins
            try:
                exporter.ensure()
                raise AssertionError("a file that fails the pin was installed")
            except exporter.ExporterMissing as e:
                msg = str(e)
            assert "not the rw_rrfs_upp this release pins" in msg and "woof-nam fetch-bridges" in msg, msg
            assert "\n" not in msg
            staged = exporter.staged_path()
            assert not staged.exists() and not any(staged.parent.glob(".rw_rrfs_upp.*")), list(staged.parent.iterdir())
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
            exporter.PINS = saved_pins


def test_fetch_bridges_help():
    from namgpu import cli
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cli.main(["fetch-bridges", "--help"])
    except SystemExit as e:
        assert not e.code, e.code
    assert "rw_rrfs_upp" in buf.getvalue() and "SHA-256" in buf.getvalue(), buf.getvalue()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: ok")
    print("test_products_writer: PASS")
