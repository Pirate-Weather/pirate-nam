"""Done-rule check for phys-rrtm-lw-taugb10 (CONTRACT section 6, parts 1 and 2).

    source ~/nam-gpu/env.sh
    python -I tools/check-phys-rrtm-lw-taugb10.py [--vram-chunk-rows 42]

1. compiles the kernel with NVRTC;
2. runs the wrapper over the whole 954 x 835 x 60 domain, in column chunks of JM rows (the output of
   one full-domain launch is about 2.7 GB per array, too big for a shared card; the kernel is
   column-independent, so chunking changes no value), on random but index-valid inputs and the
   real band-10 k-tables from namgpu/data/rrtm_lw/kgb10.npz;
3. checks every output for NaN/Inf, the band-10 g-points for positivity and for the Planck fractions
   of the reference levels.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IM, JM, LM = 954, 835, 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-chunk-rows", type=int, default=42)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from namgpu import kernels
    from namgpu.physics.rrtm import phys_rrtm_lw_taugb10 as w

    kernels.module("phys_rrtm_lw_taugb10").compile()
    print("compile: ok")

    tab = w.load_tables()
    cp.random.seed(1)
    fra = cp.asarray(np.asarray(tab["fracrefa"].get(), dtype=np.float32))
    frb = cp.asarray(np.asarray(tab["fracrefb"].get(), dtype=np.float32))
    lay_idx = cp.arange(LM, dtype=cp.int32)[:, None, None]

    t0 = time.perf_counter()
    n_bad = 0
    band_min = np.inf
    band_max = -np.inf
    n_chunks = 0
    for j_start in range(0, JM, args.vram_chunk_rows):
        jm = min(args.vram_chunk_rows, JM - j_start)
        n_chunks += 1
        u = lambda shape: cp.random.random(shape, dtype=cp.float32)  # noqa: E731
        laytrop = (20 + cp.random.randint(0, 26, (jm, IM), dtype=cp.int32)).astype(cp.int32)
        mask = lay_idx < laytrop[None]                       # lower layers
        jp_low = 1 + cp.random.randint(0, 12, (LM, jm, IM), dtype=cp.int32)    # 1..12
        jp_up = 13 + cp.random.randint(0, 46, (LM, jm, IM), dtype=cp.int32)    # 13..58
        inp = {
            "laytrop": laytrop,
            "jp": cp.where(mask, jp_low, jp_up).astype(cp.int32),
            "jt": 1 + cp.random.randint(0, 5, (LM, jm, IM), dtype=cp.int32),
            "jt1": 1 + cp.random.randint(0, 5, (LM, jm, IM), dtype=cp.int32),
            "indself": 1 + cp.random.randint(0, 9, (LM, jm, IM), dtype=cp.int32),
            "indfor": 1 + cp.random.randint(0, 3, (LM, jm, IM), dtype=cp.int32),
            "fac00": u((LM, jm, IM)), "fac01": u((LM, jm, IM)),
            "fac10": u((LM, jm, IM)), "fac11": u((LM, jm, IM)),
            "selffac": u((LM, jm, IM)) * cp.float32(1.0e-2),
            "selffrac": u((LM, jm, IM)), "forfac": u((LM, jm, IM)) * cp.float32(1.0e-2),
            "forfrac": u((LM, jm, IM)),
            "colamt": cp.float32(1.0e18) + u((6, LM, jm, IM)) * cp.float32(1.0e24),
        }
        out = w.alloc_outputs(LM, jm, IM)
        w.taugb10(inp, tab, out)
        cp.cuda.Stream.null.synchronize()

        for name in ("taug", "fracs"):
            if not bool(cp.isfinite(out[name]).all()):
                n_bad += 1
                print(f"NaN/Inf in {name}, chunk starting at row {j_start}")
        band = out["taug"][:, 108:114]
        band_min = min(band_min, float(band.min()))
        band_max = max(band_max, float(band.max()))
        if not bool((band > 0).all()):
            n_bad += 1
            print(f"non-positive band-10 optical depth, chunk starting at row {j_start}")
        fr = out["fracs"][:, 108:114]
        exp_a = cp.broadcast_to(fra[None, None, None, :], (LM, jm, IM, 6)).transpose(0, 3, 1, 2)
        exp_b = cp.broadcast_to(frb[None, None, None, :], (LM, jm, IM, 6)).transpose(0, 3, 1, 2)
        lower = (lay_idx < laytrop[None])[:, None]
        exp = cp.where(lower, exp_a, exp_b)
        if not bool(cp.array_equal(fr, exp)):
            n_bad += 1
            print(f"fracs band-10 mismatch, chunk starting at row {j_start}")
        del out, inp

    elapsed = time.perf_counter() - t0
    print(f"domain {IM} x {JM} x {LM} in {n_chunks} chunks of {args.vram_chunk_rows} rows, {elapsed:.1f} s")
    print(f"band-10 taug range over the domain: {band_min:.4e} .. {band_max:.4e}")
    if n_bad:
        raise SystemExit(f"FAIL: {n_bad} checks failed")
    print("PASS: no NaN/Inf in taug or fracs; band-10 taug > 0; band-10 fracs equal the reference fractions")


if __name__ == "__main__":
    main()
