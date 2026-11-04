"""E1 multicard checks of the exact pieces (MULTICARD-PLAN.md sections 3, 5, 10).

  python tests/check_multicard_e1.py one            # one card: MONO xsums old kernel (a55d11e atomics) twice and the
                                                    # new deterministic kernel twice; tcs old == new
  python tests/check_multicard_e1.py two [N]        # N ranks (default 2) on CUDA_VISIBLE_DEVICES list NAM_CARDS:
                                                    # halo exchange, MONO sums + rescale vs one card, bcast, allreduce

Domain: the 12 km centre band shape 954 x 279 x 60, 4 tracer slots, synthetic but limiter-active fields (the same seed
on every rank, so every rank builds the same global arrays and slices its slab).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IM, JMG, LM, NTR = 954, 279, 60, 4
KSS, KSE, IDTADT = 2, 4, 2


def global_fields(cp):
    rng = cp.random.default_rng(1234)
    s = (rng.random((NTR, LM, JMG, IM), dtype=cp.float32) * 1e-3).astype(cp.float32)
    s1 = cp.sqrt(s) + (rng.random(s.shape, dtype=cp.float32) - 0.5).astype(cp.float32) * 1e-3
    tcs = ((rng.random(s.shape, dtype=cp.float32) - 0.5) * 2e-3).astype(cp.float32)
    pd = (50000.0 + 50000.0 * rng.random((JMG, IM), dtype=cp.float32)).astype(cp.float32)
    dsg2 = cp.linspace(0.001, 0.03, LM, dtype=cp.float32)
    pdsg1 = np.linspace(10.0, 1.0, LM).astype(np.float32)
    dxh = (12000.0 * np.cos(np.linspace(-0.3, 0.3, JMG))).astype(np.float32)
    return s, s1.astype(cp.float32), tcs, pd, dsg2, pdsg1, dxh


class _St:
    def __init__(self, cp, fields):
        self.fields = fields
        self.cp = cp

    def __getitem__(self, k):
        return self.fields[k]

    def __getattr__(self, k):
        f = self.__dict__.get("fields", {})
        if k in f:
            return f[k]
        raise AttributeError(k)

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = self.cp.zeros(shape, dtype=dtype)
        return self.fields[name]


def make_ctx(cp, s, s1, tcs, pd, dsg2, pdsg1, dxh, jm, slab=None, comm=None):
    grid = SimpleNamespace(im=IM, jm=jm, lm=LM, pdsg1=pdsg1, dsg2=cp.asnumpy(dsg2), dxh=dxh, dyh=np.float32(12000.0))
    st = _St(cp, {"TRACERS": s, "TRACERS_SQRT": s1, "TRACERS_TEND": tcs, "PD": pd, "DSG2": dsg2})
    ctx = SimpleNamespace(st=st, grid=grid, cfg=SimpleNamespace(raw={"use_allreduce": True}), tables={},
                          stream=cp.cuda.Stream(non_blocking=True), slab=slab, comm=comm)
    dare = cp.asarray(dxh.astype(np.float32) * np.float32(12000.0))
    ctx.tables["grid"] = {"dare": dare}
    return ctx


def mono_new(cp, ctx):
    from namgpu.dynamics import dyn_mono_fix, dyn_mono_global_sum
    with ctx.stream:
        x = dyn_mono_fix.run(ctx, kss=KSS, kse=KSE, idtadt=IDTADT)
        xs = x.copy()
        dyn_mono_global_sum.run(ctx, kss=KSS, kse=KSE)
    ctx.stream.synchronize()
    return xs


def mono_old(cp, s, s1, tcs, pd, dsg2, pdsg1, dxh):
    src = subprocess.run(["git", "-C", str(ROOT), "show", "a55d11e:namgpu/kernels/dyn_mono_fix.cu"],
                         capture_output=True, text=True, check=True).stdout
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=true", f"-I{ROOT / 'namgpu' / 'kernels'}"))
    k = mod.get_function("dyn_mono_fix")
    xs = cp.zeros((2 * NTR,), dtype=cp.float64)
    dare = cp.asarray(dxh * np.float32(12000.0))
    grid = ((IM + 31) // 32, (JMG + 7) // 8, LM)
    k(grid, (32, 8, 1), (np.int32(IM), np.int32(JMG), np.int32(LM), np.int32(IDTADT), np.int32(KSS), np.int32(KSE),
                         np.int32(NTR), np.int32(1), s, s1, tcs, pd, dare, dsg2, cp.asarray(pdsg1), xs))
    cp.cuda.Device().synchronize()
    return xs


def one():
    import cupy as cp
    res = {}
    base = global_fields(cp)
    olds, news, tcs_old, tcs_new = [], [], None, None
    for _ in range(4):
        s, s1, tcs, pd, dsg2, pdsg1, dxh = [a.copy() if hasattr(a, "copy") else a for a in base]
        olds.append(cp.asnumpy(mono_old(cp, s, s1, tcs, pd, dsg2, pdsg1, dxh)))
        tcs_old = cp.asnumpy(tcs)
    for _ in range(4):
        s, s1, tcs, pd, dsg2, pdsg1, dxh = [a.copy() if hasattr(a, "copy") else a for a in base]
        ctx = make_ctx(cp, s, s1, tcs, pd, dsg2, pdsg1, dxh, JMG)
        from namgpu.dynamics import dyn_mono_fix
        with ctx.stream:
            x = dyn_mono_fix.run(ctx, kss=KSS, kse=KSE, idtadt=IDTADT)
        ctx.stream.synchronize()
        news.append(cp.asnumpy(x))
        tcs_new = cp.asnumpy(tcs)
    old_bits = {o.tobytes() for o in olds}
    new_bits = {n.tobytes() for n in news}
    res["old_atomic_distinct_results_of_4"] = len(old_bits)
    res["new_distinct_results_of_4"] = len(new_bits)
    res["old_vs_new_max_rel"] = float(np.max(np.abs(olds[0] - news[0]) / np.maximum(np.abs(olds[0]), 1e-300)))
    res["xsums_new"] = news[0][2 * (KSS - 1):2 * KSE].tolist()
    res["xsums_old_spread"] = [float(np.ptp([o[e] for o in olds])) for e in range(2 * (KSS - 1), 2 * KSE)]
    res["tcs_old_eq_new"] = bool(np.array_equal(tcs_old.view(np.uint32), tcs_new.view(np.uint32)))
    print(res, flush=True)
    ok = res["new_distinct_results_of_4"] == 1 and res["tcs_old_eq_new"] and res["old_vs_new_max_rel"] < 1e-10
    print("ONE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def two_rank(rank, nranks):
    import cupy as cp

    from namgpu import slab as slabmod
    from namgpu.parallel import Comm
    from namgpu.steps import drv_halo_exch as halo
    comm = Comm(rank, nranks, os.environ["NAMGPU_NCCL_ID"])
    sl = slabmod.Slab.for_rank(rank, nranks, JMG)
    out = {"rank": rank, "rows": (sl.jts, sl.jte), "mem": (sl.jms, sl.jme)}
    base = global_fields(cp)
    s, s1, tcs, pd, dsg2, pdsg1, dxh = base
    # one-card reference on this card
    ref_ctx = make_ctx(cp, s.copy(), s1.copy(), tcs.copy(), pd.copy(), dsg2, pdsg1, dxh, JMG)
    x_ref = cp.asnumpy(mono_new(cp, ref_ctx))
    tcs_ref = ref_ctx.st["TRACERS_TEND"]
    # slab copies
    rows = slice(sl.jms - 1, sl.jme)
    loc = lambda a: cp.ascontiguousarray(a[..., rows, :])  # noqa: E731
    ctx = make_ctx(cp, loc(s), loc(s1), loc(tcs), loc(pd), dsg2, pdsg1, dxh, sl.jm, slab=sl, comm=comm)
    ctx.tables["grid"] = {"dare": cp.asarray(dxh[rows] * np.float32(12000.0))}
    x = cp.asnumpy(mono_new(cp, ctx))
    out["mono_xsums_bitwise"] = bool(np.array_equal(x.view(np.uint64), x_ref.view(np.uint64)))
    own = slice(sl.own0, sl.own0 + sl.nown)
    tl = ctx.st["TRACERS_TEND"][..., own, :]
    tr = tcs_ref[..., sl.jts - 1:sl.jte, :]
    out["mono_tcs_owned_bitwise"] = bool(cp.array_equal(tl.view(cp.uint32), tr.view(cp.uint32)))
    # halo exchange: zero the halo rows of a copy of global data, exchange, compare with the global slice
    f3 = cp.ascontiguousarray(base[0][1][..., rows, :])            # [LM, jm, IM] float32
    i2 = cp.arange(JMG * IM, dtype=cp.int32).reshape(JMG, IM)[rows].copy()
    g3, g2 = f3.copy(), i2.copy()
    for a in (f3, i2):
        a[..., :sl.own0, :] = 0
        a[..., sl.own0 + sl.nown:, :] = 0
    hctx = SimpleNamespace(slab=sl, comm=comm, grid=SimpleNamespace(im=IM, jm=sl.jm, lm=LM), tables={},
                           stream=cp.cuda.Stream(non_blocking=True))
    halo.exchange(hctx, [("F3", f3), ("I2", i2)], "TEST")
    hctx.stream.synchronize()
    out["halo_f3_bitwise"] = bool(cp.array_equal(f3.view(cp.uint32), g3.view(cp.uint32)))
    out["halo_i2_equal"] = bool(cp.array_equal(i2, g2))
    # bcast from the last rank and an int allreduce
    b = cp.full((5,), float(rank), dtype=cp.float32)
    comm.bcast(b, nranks - 1, hctx.stream)
    m = cp.array([rank + 1], dtype=cp.int32)
    comm.allreduce(m, m, "max", hctx.stream)
    hctx.stream.synchronize()
    out["bcast_ok"] = bool(cp.all(b == nranks - 1))
    out["allreduce_max_ok"] = int(m.get()[0]) == nranks
    ok = all(v for k, v in out.items() if k.endswith(("bitwise", "equal", "_ok")))
    out["PASS"] = ok
    print(out, flush=True)
    comm.barrier(hctx.stream)
    return 0 if ok else 1


def two(nranks):
    cards = [c for c in os.environ.get("NAM_CARDS", "").split(",") if c] or [str(i) for i in range(nranks)]
    idf = Path("/tmp") / f"e1-nccl-{os.getpid()}.id"
    procs = []
    for r in range(nranks):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[r], NAMGPU_RANK=str(r), NAMGPU_NRANKS=str(nranks),
                   NAMGPU_NCCL_ID=str(idf))
        procs.append(subprocess.Popen([sys.executable, __file__, "rank"], env=env))
    codes = [p.wait() for p in procs]
    idf.unlink(missing_ok=True)
    print("TWO", "PASS" if all(c == 0 for c in codes) else f"FAIL {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "one":
        sys.exit(one())
    if mode == "two":
        sys.exit(two(int(sys.argv[2]) if len(sys.argv) > 2 else 2))
    if mode == "rank":
        sys.exit(two_rank(int(os.environ["NAMGPU_RANK"]), int(os.environ["NAMGPU_NRANKS"])))
