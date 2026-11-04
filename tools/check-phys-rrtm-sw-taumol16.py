"""Done-rule check for phys-rrtm-sw-taumol16 (CONTRACT.md section 6, parts 1 and 2).

    source env.sh && python tools/check-phys-rrtm-sw-taumol16.py

Compiles the kernel and runs both entry points at full NAM 12 km size (954 x 835 x 60) on random but
plausible inputs:
  1. the full-domain path (one launch, taug and taur planes 0..5);
  2. the driver chunk path (run_chunk over every column, SW_NDAY = all columns live, chunks of 4096).
Checks that the planes this unit writes hold no NaN or Inf, that the setcoef index ranges used by the
self and foreign continuum reads hold (indself 1..9, indfor 1..2), and that the first chunk of the
chunk path equals the full-domain path bitwise (same kernel body, same arithmetic order).
Memory: about 6 GB for the full-domain arrays and outputs; the chunk path adds one chunk at a time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_sw_taumol16 as u

    im, jm, nlay = 954, 835, 60
    ncol = jm * im
    kernels.module(u.UNIT).compile()
    rng = cp.random.default_rng(16)
    f = lambda: rng.random((nlay, jm, im), dtype=cp.float32)
    ints = lambda lo, hi: rng.integers(lo, hi + 1, size=(nlay, jm, im), dtype=cp.int32)

    laytrop = rng.integers(1, nlay + 1, size=(jm, im), dtype=cp.int32)
    inp = {
        "colamt": rng.random((6, nlay, jm, im), dtype=cp.float32) * cp.float32(1.0e20),
        "colmol": f() * cp.float32(1.0e25),
        "laytrop": laytrop,
        "jp": ints(1, 58), "jt": ints(1, 5), "jt1": ints(1, 5),
        "indself": ints(1, 9), "indfor": ints(1, 2),
        "fac00": f(), "fac01": f(), "fac10": f(), "fac11": f(),
        "selffac": f(), "selffrac": f(), "forfac": f(), "forfrac": f(),
    }
    assert int(inp["indself"].min()) >= 1 and int(inp["indself"].max()) <= 9
    assert int(inp["indfor"].min()) >= 1 and int(inp["indfor"].max()) <= 2
    tab = u.load_tables()

    # 1. full-domain path
    out = u.alloc_outputs(nlay, jm, im)
    t0 = time.perf_counter()
    u.taumol16(inp, out, tab)
    cp.cuda.Device().synchronize()
    dt_full = time.perf_counter() - t0
    bad = 0
    for name in ("taug", "taur"):
        n_bad = int((~cp.isfinite(out[name])).sum())
        bad += n_bad
        print(f"full {name}: shape {tuple(out[name].shape)} nonfinite {n_bad}")
    print(f"full launch + sync {dt_full:.3f} s, minval(laytrop) {int(cp.min(laytrop))}")

    # 2. driver chunk path over every column, all columns live
    ctx = SimpleNamespace(tables={}, st={"SW_NDAY": cp.asarray([ncol], dtype=cp.int32)}, stream=None)
    colamt2 = inp["colamt"].reshape(6, nlay, ncol)
    flat2 = {k: inp[k].reshape(nlay, ncol) for k in
             ("colmol", "jp", "jt", "jt1", "indself", "indfor", "fac00", "fac01", "fac10", "fac11",
              "selffac", "selffrac", "forfac", "forfrac")}
    lt_flat = laytrop.reshape(ncol)
    nc = 4096
    t0 = time.perf_counter()
    first = None
    for c0 in range(0, ncol, nc):
        c1 = min(c0 + nc, ncol)
        n = c1 - c0
        ch = {k: cp.ascontiguousarray(v[:, c0:c1].T) for k, v in flat2.items()}
        ch["colamt"] = cp.ascontiguousarray(colamt2[:, :, c0:c1].transpose(2, 1, 0))
        ch["laytrop"] = cp.ascontiguousarray(lt_flat[c0:c1])
        ch["taug"] = cp.zeros((n, nlay, u.NGPTSW), dtype=cp.float32)
        ch["taur"] = cp.zeros((n, nlay, u.NGPTSW), dtype=cp.float32)
        u.run_chunk(ctx, ch, c0, n)
        cp.cuda.Device().synchronize()
        for name in ("taug", "taur"):
            nb = int((~cp.isfinite(ch[name][:, :, :u.NG16])).sum())
            bad += nb
        if first is None:
            first = (c0, n, ch["taug"][:, :, :u.NG16].transpose(2, 1, 0).copy(),
                     ch["taur"][:, :, :u.NG16].transpose(2, 1, 0).copy())
    dt_chunk = time.perf_counter() - t0
    print(f"chunk path over {ncol} columns in chunks of {nc}: {dt_chunk:.3f} s")

    c0, n, tg0, tr0 = first
    same_g = bool(cp.array_equal(tg0, out["taug"].reshape(u.NG16, nlay, ncol)[:, :, c0:c0 + n]))
    same_r = bool(cp.array_equal(tr0, out["taur"].reshape(u.NG16, nlay, ncol)[:, :, c0:c0 + n]))
    print(f"first chunk equals full domain: taug {same_g}, taur {same_r}")
    print(f"nonfinite total {bad}")
    if bad or not (same_g and same_r):
        raise SystemExit(1)
    print("PASS phys-rrtm-sw-taumol16: full and chunk paths compiled, full-size, no NaN/Inf, chunk == full")


if __name__ == "__main__":
    main()
