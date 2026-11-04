"""Reference test for phys-fer-get-indexr (GET_INDEXR, module_MP_FER_HIRES.F90:2297-2380).

Referee: tools/fer-get-indexr-ref/ref_get_indexr.F90 holds GET_INDEXR copied verbatim from the NAM source
(nam.v4.2.11, 83df4e13c1) and is compiled with gfortran -O2. This script:
  1. asserts the MDR* integer constants re-derived from the Fortran REAL parameters;
  2. writes one full-size RR grid (LM x JM x IM = 60 x 835 x 954) plus the seven breakpoints to a file;
  3. runs the Fortran referee and the CUDA kernel on identical inputs and asserts exact equality;
  4. asserts no NaN-type garbage (outputs in [50, 1000]) and that the wrapper refuses bad buffers.
Numerics: fast numerics (--fmad=true, no fast math). Measured on set B: 18 of 47.8 M values differ by one
count at INT(x+.5) boundaries; the same 18 differ with --fmad=false, so the residue is the power function
(CUDA powf vs glibc powf), not FMA. The unit is therefore NOT claimed bitwise; the test asserts exact
equality on set A and a one-count, <=1e-6 residue on set B.
The breakpoints are caller inputs to GET_INDEXR. The production RR_DR* values need RRATE from
ETAMPNEW_DATA.expanded_rain, which is not in the NAM clone, so two monotone breakpoint sets are replayed.
"""
import os
import subprocess
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import cupy as cp  # noqa: E402
from namgpu import kernels  # noqa: E402
from namgpu.physics.fer_hires import phys_fer_get_indexr as w  # noqa: E402

LM, JM, IM = 60, 835, 954
REF_DIR = os.path.join(REPO, "tools", "fer-get-indexr-ref")
REF_BIN = os.path.join(REF_DIR, "ref_get_indexr")
SCRATCH = os.path.join(REPO, "scratch", "fer-get-indexr")

EXPECT = {"MDRmin": 50, "MDR1": 100, "MDR2": 200, "MDR3": 320, "MDR4": 450, "MDR5": 670, "MDRmax": 1000}
DMR = {"MDRmin": 0.05e-3, "MDR1": 0.1e-3, "MDR2": 0.2e-3, "MDR3": 0.32e-3,
       "MDR4": 0.45e-3, "MDR5": 0.67e-3, "MDRmax": 1.0e-3}

THRESH_SETS = {
    "A": [1e-3, 5e-3, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1],
    "B": [2e-4, 1e-3, 4e-3, 9e-3, 2e-2, 6e-2, 3e-1],
}


def check_constants():
    for name, dmr in DMR.items():
        # Fortran: XMRx = 1.e6*DMRx is REAL (single); MDRx = XMRx is INTEGER (truncated).
        got = int(np.float32(np.float32(1.0e6) * np.float32(dmr)))
        assert got == EXPECT[name], (name, got, EXPECT[name])
    print("constants ok", EXPECT)


def build_inputs(thr, rng):
    n = LM * JM * IM
    rr = np.exp(rng.uniform(np.log(1e-9), np.log(10.0), n)).astype(np.float32)
    # Exact breakpoints and their float32 neighbours, at random positions (the <= tests decide these).
    pos = rng.choice(n, size=len(thr) * 3 * 40, replace=False)
    vals = []
    for t in thr:
        t32 = np.float32(t)
        vals += [t32, np.nextafter(t32, np.float32(0)), np.nextafter(t32, np.float32(1))] * 40
    rr[pos] = np.asarray(vals, dtype=np.float32)
    # Special values: zero, negative, +inf, NaN.
    special = np.array([0.0, -1e-3, np.inf, np.nan], dtype=np.float32)
    rr[rng.choice(n, size=400, replace=False)] = np.tile(special, 100)
    return rr


def run_fortran(rr, thr, tag):
    os.makedirs(SCRATCH, exist_ok=True)
    fin = os.path.join(SCRATCH, f"in-{tag}.bin")
    fout = os.path.join(SCRATCH, f"out-{tag}.bin")
    with open(fin, "wb") as f:
        np.array([rr.size], dtype=np.int32).tofile(f)
        np.asarray(thr, dtype=np.float32).tofile(f)
        rr.astype(np.float32).tofile(f)
    subprocess.run([REF_BIN, fin, fout], check=True)
    return np.fromfile(fout, dtype=np.int32)


def run_port(rr, thr):
    ctx = types.SimpleNamespace(st={}, tables={}, stream=None)
    out = w.run(ctx, RR=cp.asarray(rr.reshape(LM, JM, IM)), thresh=cp.asarray(np.asarray(thr, dtype=np.float32)))
    cp.cuda.Device().synchronize()
    return cp.asnumpy(out).ravel()


def check_guards(thr):
    ctx = types.SimpleNamespace(st={}, tables={}, stream=None)
    thr_dev = cp.asarray(np.asarray(thr, dtype=np.float32))
    base = cp.zeros((LM, JM, IM), dtype=cp.float32)
    for label, call, exc in [
        ("float64 RR", lambda: w.run(ctx, RR=base.astype(cp.float64), thresh=thr_dev), TypeError),
        ("non-contiguous RR", lambda: w.run(ctx, RR=base[:, :, ::2], thresh=thr_dev), ValueError),
        ("bad out shape", lambda: w.run(ctx, RR=base, thresh=thr_dev,
                                        out=cp.zeros((LM, JM, IM - 1), dtype=cp.int32)), ValueError),
    ]:
        try:
            call()
        except exc:
            print("guard refused:", label)
        else:
            raise AssertionError(f"guard did not refuse: {label}")


def main():
    check_constants()
    kernels.get(w.UNIT, w.KERNEL)
    print("compiled", w.UNIT, w.KERNEL)
    rng = np.random.default_rng(7)
    for tag, thr in THRESH_SETS.items():
        rr = build_inputs(thr, rng)
        ref = run_fortran(rr, thr, tag)
        got = run_port(rr, thr)
        assert ref.size == got.size == rr.size
        bad = np.flatnonzero(got != ref)
        print(f"set {tag}: n={rr.size} mismatches={bad.size} "
              f"range=[{int(got.min())},{int(got.max())}] ref_range=[{int(ref.min())},{int(ref.max())}]")
        for i in bad[:10]:
            print(f"  rr={rr[i]!r} fortran={int(ref[i])} cuda={int(got[i])}")
        if tag == "A":
            assert bad.size == 0, f"set A: {bad.size} differing values"
        else:
            assert np.all(np.abs(got[bad].astype(np.int64) - ref[bad].astype(np.int64)) == 1)
            assert bad.size <= 1e-6 * rr.size, f"set {tag}: {bad.size} differing values"
        assert int(got.min()) >= 50 and int(got.max()) <= 1000
    check_guards(THRESH_SETS["A"])
    print("PASS")


if __name__ == "__main__":
    main()
