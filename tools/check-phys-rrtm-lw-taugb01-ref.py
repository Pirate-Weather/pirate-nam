"""Value check for phys-rrtm-lw-taugb01 against the NAM Fortran TAUGB01 body.

Builds the same FULL-SIZE state as tools/check-phys-rrtm-lw-taugb01.py, runs the GPU wrapper on all
954 x 835 x 60 columns, then runs the NAM TAUGB01 reference (tools/ref_taugb01/build/ref_taugb01, built by
tools/ref_taugb01/build.sh from the NAM clone) on a seeded random sample of columns and compares taug and fracs.
The kernel is one thread per column with no cross-column terms, so a column sample checks the same code path.

    source env.sh && python tools/check-phys-rrtm-lw-taugb01-ref.py [--ncol 4096]
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tools" / "ref_taugb01" / "build" / "ref_taugb01"
NG01 = 10
TOL = 1.0e-5  # max |gpu - ref| / max|ref| per array (float32 kernel vs float64 reference)


def _load_check():
    spec = importlib.util.spec_from_file_location("chk_taugb01", ROOT / "tools" / "check-phys-rrtm-lw-taugb01.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=4096)
    args = ap.parse_args(argv)

    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb01 as glue

    chk = _load_check()
    kernels.module(glue.UNIT).compile()
    st, _ = chk.build_state(cp)
    ctx = SimpleNamespace(cfg=None, tables={}, st=st, stream=cp.cuda.Stream(non_blocking=True), grid=None)
    t0 = time.perf_counter()
    with ctx.stream:
        glue.run(ctx)
    ctx.stream.synchronize()
    print(f"gpu wrapper full-size run: {time.perf_counter() - t0:.2f} s")

    lm, jm, im = st["pavel"].shape
    ncol = jm * im
    rng = np.random.default_rng(7)
    cols = np.sort(rng.choice(ncol, size=min(args.ncol, ncol), replace=False))
    n = cols.size

    def lay(name):  # [N, LM] host copy of a layer field for the sampled columns
        return cp.asnumpy(st[name].reshape(lm, ncol)[:, cols]).T

    with tempfile.TemporaryDirectory(prefix="taugb01-ref-", dir=ROOT / "runs" if (ROOT / "runs").is_dir() else None) as d:
        d = Path(d)
        for name in ("pavel", "colbrd", "scaleminorn2", "selffac", "selffrac", "forfac", "forfrac",
                     "minorfrac", "fac00", "fac01", "fac10", "fac11"):
            lay(name).astype(np.float64).tofile(d / f"{name}.f64")
        for name in ("indself", "indfor", "indminor", "jp", "jt", "jt1"):
            lay(name).astype(np.int32).tofile(d / f"{name}.i32")
        cp.asnumpy(st["laytrop"].reshape(ncol)[cols]).astype(np.int32).tofile(d / "laytrop.i32")
        # colamt [6, LM, JM, IM] -> [N, 6, LM] so the Fortran colamt(LM, 6, N) memory order matches
        cam = cp.asnumpy(st["colamt"].reshape(6, lm, ncol)[:, :, cols])  # [6, LM, N]
        np.ascontiguousarray(np.transpose(cam, (2, 0, 1))).astype(np.float64).tofile(d / "colamt.f64")

        t1 = time.perf_counter()
        subprocess.run([str(REF), str(d), str(n)], check=True)
        print(f"fortran reference on {n} columns: {time.perf_counter() - t1:.2f} s")
        ref_taug = np.fromfile(d / "taug_ref.f64", dtype=np.float64).reshape(n, lm, NG01)
        ref_fracs = np.fromfile(d / "fracs_ref.f64", dtype=np.float64).reshape(n, lm, NG01)

    gpu_taug = np.transpose(cp.asnumpy(st["taug_b01"].reshape(lm, NG01, ncol)[:, :, cols]), (2, 0, 1))
    gpu_fracs = np.transpose(cp.asnumpy(st["fracs_b01"].reshape(lm, NG01, ncol)[:, :, cols]), (2, 0, 1))

    ok = True
    for label, gpu, ref in (("taug_b01", gpu_taug, ref_taug), ("fracs_b01", gpu_fracs, ref_fracs)):
        scale = float(np.max(np.abs(ref)))
        err = np.abs(gpu.astype(np.float64) - ref)
        rel = float(err.max() / scale)
        ok = ok and rel < TOL and bool(np.isfinite(gpu).all())
        print(f"{label}: max|ref|={scale:.6e} max|gpu-ref|={float(err.max()):.6e} "
              f"max rel={rel:.3e} (tol {TOL:g}) columns={n} values={ref.size}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
