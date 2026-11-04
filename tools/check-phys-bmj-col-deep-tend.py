"""Done-rule check for unit phys-bmj-col-deep-tend (CONTRACT.md section 6, parts 1 and 2).

    source env.sh
    python tools/check-phys-bmj-col-deep-tend.py

Compiles namgpu/kernels/phys_bmj_col_deep_tend.cu and runs the wrapper through the full itrefi loop
(ITREFI_MAX passes) on a FULL-SIZE random State (954 x 835 x 60). The per-column bookkeeping
(ltop, lb, l0, pkt, rdp0t) and the reference profiles normally come from deep-ref / deep-iter, which
are not wired yet, so they are plausible synthetic values. The check is NaN/Inf on the outputs, plus a
refusal test: a missing required input (BMJ_LB) must raise before any kernel launch.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cupy as cp
    from namgpu import config, grid as gridmod
    from namgpu.driver import Ctx
    from namgpu.physics.bmj import phys_bmj_col_deep_tend as unit

    cfg = config.load()
    g = gridmod.build(cfg)
    IM, JM, LM = g.im, g.jm, g.lm
    rng = cp.random.default_rng(1)

    # Only the fields this unit reads or writes are allocated (full 954 x 835 x 60 shapes). The
    # full 48-field State needs about 10.6 GiB, more than the free VRAM on the shared card.
    class _Fields:
        def __init__(self):
            self.fields = {}

        def add(self, name, shape, dtype="float32"):
            if name not in self.fields:
                self.fields[name] = cp.zeros(shape, dtype=dtype)

        def __getitem__(self, name):
            return self.fields[name]

    st = _Fields()
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True), dt=cfg.dt)

    # Inputs stand in for what the upstream units leave in ctx.st. The wrapper only adds its outputs.
    for name in unit.LEVEL_IN:
        st.add(name, (LM, JM, IM))
    for name, dt in unit.COLUMN_IN.items():
        st.add(name, (JM, IM), dtype=dt)
    for name in ("U_PHY", "V_PHY"):
        st.add(name, (LM, JM, IM))
    st.add("CLDEFI", (JM, IM))

    f = st.fields
    lvl = (LM, JM, IM)
    f["BMJ_TK"][...] = rng.uniform(230.0, 300.0, lvl, dtype=cp.float32)
    f["BMJ_QK"][...] = rng.uniform(1e-4, 0.02, lvl, dtype=cp.float32)
    f["BMJ_DPRS"][...] = rng.uniform(500.0, 3000.0, lvl, dtype=cp.float32)
    f["BMJ_EL"][...] = rng.uniform(0.5, 1.5, lvl, dtype=cp.float32)
    f["BMJ_TREFK"][...] = f["BMJ_TK"] + rng.uniform(-2.0, 2.0, lvl, dtype=cp.float32)
    f["BMJ_QREFK"][...] = f["BMJ_QK"] * rng.uniform(0.8, 1.2, lvl, dtype=cp.float32)
    f["BMJ_PK"][...] = rng.uniform(0.3, 1.0, lvl, dtype=cp.float32)
    f["U_PHY"][...] = rng.uniform(-20.0, 20.0, lvl, dtype=cp.float32)
    f["V_PHY"][...] = rng.uniform(-20.0, 20.0, lvl, dtype=cp.float32)
    f["BMJ_SM"][...] = rng.uniform(0.0, 1.0, (JM, IM), dtype=cp.float32)
    f["BMJ_PKT"][...] = rng.uniform(0.3, 0.6, (JM, IM), dtype=cp.float32)
    f["BMJ_RDP0T"][...] = rng.uniform(2.0, 5.0, (JM, IM), dtype=cp.float32)
    f["BMJ_LTOP"][...] = cp.floor(rng.uniform(5.0, 16.0, (JM, IM), dtype=cp.float32)).astype(cp.int32)
    f["BMJ_LB"][...] = cp.floor(rng.uniform(45.0, 59.0, (JM, IM), dtype=cp.float32)).astype(cp.int32)
    f["BMJ_L0"][...] = cp.floor(rng.uniform(20.0, 41.0, (JM, IM), dtype=cp.float32)).astype(cp.int32)
    f["CLDEFI"][...] = 0.5

    # Refusal: a missing required input must raise before the kernel runs.
    lb_saved = st.fields.pop("BMJ_LB")
    try:
        unit.run(ctx, itrefi_pass=1)
    except RuntimeError as e:
        print("refusal ok:", e)
    else:
        print("phys-bmj-col-deep-tend: FAIL missing BMJ_LB did not raise")
        return 1
    finally:
        st.fields["BMJ_LB"] = lb_saved

    bad = False
    with ctx.stream:
        for p in range(1, unit.ITREFI_MAX + 1):
            outs = unit.run(ctx, itrefi_pass=p)
    ctx.stream.synchronize()
    for name, arr in zip(("BMJ_DTDT", "BMJ_DQDT"), outs):
        ok = bool(cp.isfinite(arr).all())
        bad = bad or (not ok)
        print(f"{name}: {'NaN/Inf' if not ok else 'finite'} shape {tuple(arr.shape)} "
              f"min {float(arr.min()):.4e} max {float(arr.max()):.4e}")
    for name in ("BMJ_EFI", "BMJ_SUMDP", "BMJ_DENTPY", "BMJ_PRECK", "BMJ_FEFI", "BMJ_PCPCOL", "CLDEFI"):
        ok = bool(cp.isfinite(f[name]).all())
        bad = bad or (not ok)
        print(f"{name}: {'NaN/Inf' if not ok else 'finite'}")
    print("phys-bmj-col-deep-tend:", "FAIL NaN/Inf" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
