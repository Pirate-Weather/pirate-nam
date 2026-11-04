"""Done-rule check for phys-rrtm-lw-rtrnmc (CONTRACT section 6), review fix.

    source ~/nam-gpu/env.sh
    python tools/check-phys-rrtm-lw-rtrnmc.py

1. compiles the kernel with NVRTC;
2. tables: build_tables() equals a scalar, loop-by-loop transcription of rlwinit
   (radlw_main_nmmb.f:1313-1339) element for element;
3. heatfac: the bound value for ilwrate=2 (physparam.f:71) equals con_g*1e-2/con_cp in single;
4. ngb: a 0-based band table is refused (no silent zero fluxes);
5. contiguity: a non-contiguous view chunk gives exactly the output of a contiguous copy;
6. full domain 954 x 835 x 60, default flags (lhlw0 = lhlwb = False), random plausible inputs,
   chunked, one launch per chunk: every output finite, and htrcl and htrb are zero when their
   flags are off.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# module_radlw_parameters NGn values as USEd by radlw_main_nmmb.f (radlw_param.f:107-113)
NG = [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2]
NB, NGPT, NTBL = 16, 140, 10000
IM, JM, LM = 954, 835, 60
CHUNK = 8192   # the card is shared; 32768 ran out of memory at 2.3 GB of a 12 GB free card


def fortran_tables():
    """Scalar transcription of rlwinit (radlw_main_nmmb.f:1313-1339), one element at a time."""
    import numpy as np
    f = np.float32
    bpade = f(1.0 / 0.278)
    expeps = f(1.0e-20)
    explimit = f(87.0)
    tau = np.zeros(NTBL + 1, f)
    ex = np.zeros(NTBL + 1, f)
    tfn = np.zeros(NTBL + 1, f)
    ex[0] = f(1.0)
    tau[NTBL] = f(1.0e10)
    ex[NTBL] = expeps
    tfn[NTBL] = f(1.0)
    for i in range(1, NTBL):
        t = f(i) / f(NTBL - i)
        tau[i] = bpade * t
        if tau[i] >= explimit:
            ex[i] = expeps
        else:
            ex[i] = f(np.exp(-tau[i]))
        if tau[i] < f(0.06):
            tfn[i] = tau[i] / f(6.0)
        else:
            tfn[i] = f(1.0) - f(2.0) * ((f(1.0) / tau[i]) - (ex[i] / (f(1.0) - ex[i])))
    return tau, ex, tfn


def check_tables(w):
    import numpy as np
    got = w.build_tables()
    ref = fortran_tables()
    bad = {}
    for name, a, b in zip(("tau_tbl", "exp_tbl", "tfn_tbl"), (got["tau_tbl"], got["exp_tbl"], got["tfn_tbl"]), ref):
        n_diff = int(np.count_nonzero(a != b))
        max_ulp = float(np.max(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))))
        bad[name] = (n_diff, max_ulp)
        print(f"tables {name}: {n_diff} of {a.size} differ from the scalar rlwinit transcription, max ulp {max_ulp:.0f}")
    return all(v[0] == 0 for v in bad.values())


def check_heatfac(w):
    import numpy as np
    hf = w.heatfac_for(2)
    expect = float(np.float32(np.float32(9.80665) * np.float32(1.0e-2)) / np.float32(1.0046e3))
    print(f"heatfac ilwrate=2 (k/s): {hf!r}, expected {expect!r}, k/day: {w.heatfac_for(1)!r}")
    return hf == expect


def check_ngb_refused(w):
    import cupy as cp
    import numpy as np
    bad = cp.asarray(np.repeat(np.arange(0, NB, dtype=np.int32), NG))  # 0-based: must be refused
    try:
        w._check_ngb(cp, bad)
    except ValueError as e:
        print("ngb 0-based: refused:", str(e)[:90])
        return True
    print("ngb 0-based: NOT refused")
    return False


def make_inputs(cp, nc, tables, ngb, rng, view_of=None):
    """Random plausible inputs; with view_of, every array is a non-contiguous last-axis slice."""
    def take(shape):
        return rng(shape)
    return {
        "semiss": take((NB, nc)), "secdif": cp.full((NB, nc), 1.66, dtype=cp.float32),
        "delp": cp.asarray(rng((LM, nc), 0.5, 30.0)), "cldfmc": rng((LM, NGPT, nc), 0.0, 1.0),
        "taucld": rng((LM, NB, nc), 0.0, 2.0), "tautot": rng((LM, NGPT, nc), 0.0, 5.0),
        "pklay": rng((LM + 1, NB, nc), 20.0, 400.0), "pklev": rng((LM + 1, NB, nc), 20.0, 400.0),
        "fracs": rng((LM, NGPT, nc), 0.0, 1.0), "ngb": ngb, **tables,
    }


def main():
    import cupy as cp
    import numpy as np
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_rtrnmc as w

    kernels.module("phys_rrtm_lw_rtrnmc").compile()
    print("compile: ok")

    ok = check_tables(w)
    ok &= check_heatfac(w)
    ok &= check_ngb_refused(w)

    tb = {k: cp.asarray(v) for k, v in w.build_tables().items()}
    ngb = cp.asarray(np.repeat(np.arange(1, NB + 1, dtype=np.int32), NG))
    assert ngb.size == NGPT
    cp.random.seed(1)

    def rng(shape, lo=0.0, hi=1.0):
        return (cp.random.random(shape, dtype=cp.float32) * (hi - lo) + lo).astype(cp.float32)

    ctx = type("Ctx", (), {"stream": None})()

    # 5. contiguity: a last-axis view chunk must give the same output as its contiguous copy.
    ncb, c0, c1 = 4096, 1024, 2048
    big = make_inputs(cp, ncb, tb, ngb, rng)
    view_in = {k: (v[..., c0:c1] if v.ndim >= 1 and v.shape[-1] == ncb and k not in ("ngb",) and k not in tb
                   else v) for k, v in big.items()}
    copy_in = {k: (cp.ascontiguousarray(v) if (k not in ("ngb",) and k not in tb) else v)
               for k, v in view_in.items()}
    assert not view_in["cldfmc"].flags.c_contiguous
    nc = c1 - c0

    def outs(n):
        return {"totuflux": cp.zeros((LM + 1, n), cp.float32), "totdflux": cp.zeros((LM + 1, n), cp.float32),
                "totuclfl": cp.zeros((LM + 1, n), cp.float32), "totdclfl": cp.zeros((LM + 1, n), cp.float32),
                "htr": cp.zeros((LM, n), cp.float32), "htrcl": cp.zeros((LM, n), cp.float32),
                "htrb": cp.zeros((NB, LM, n), cp.float32)}
    base = outs(ncb)
    o_view = {k: v[..., c0:c1] for k, v in base.items()}     # non-contiguous output views
    o_copy = outs(nc)
    assert not o_view["htrb"].flags.c_contiguous
    w.run(ctx, inputs=view_in, outputs=o_view, nlay=LM, lhlw0=True, lhlwb=True)
    w.run(ctx, inputs=copy_in, outputs=o_copy, nlay=LM, lhlw0=True, lhlwb=True)
    cp.cuda.Stream.null.synchronize()
    same = all(bool(cp.array_equal(o_view[k], o_copy[k])) for k in o_view)
    print(f"contiguity: view chunk == contiguous copy on all outputs: {same}")
    ok &= same
    del big, view_in, copy_in, o_view, o_copy

    # 6. full domain, default flags, one launch per chunk.
    ncol_all = IM * JM
    bad = []
    n_chunks = 0
    t0 = time.perf_counter()
    for c in range(0, ncol_all, CHUNK):
        n = min(CHUNK, ncol_all - c)
        inp = make_inputs(cp, n, tb, ngb, rng)
        out = outs(n)
        w.run(ctx, inputs=inp, outputs=out, nlay=LM)
        n_chunks += 1
        for name, arr in out.items():
            if not bool(cp.isfinite(arr).all()):
                bad.append((c, name))
        if not bool((out["htrcl"] == 0).all()) or not bool((out["htrb"] == 0).all()):
            bad.append((c, "flag-off zeros"))
        del inp, out
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0
    if bad:
        print("NaN/Inf or nonzero flag-off outputs:", bad[:10])
        return 1
    print(f"full domain: {ncol_all} columns in {n_chunks} chunks, default flags, no NaN/Inf, "
          f"htrcl and htrb zero, {dt:.2f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
