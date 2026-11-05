"""History reflectivity as NAM's post writes it (io_history_fields._refl_post, kernel hist_refl_post): the microphysics
REFL_10CM plus the BMJ convective-rain term Z = 300 R**1.4 below the convective cloud top, tapered by 20 dBZ between
the freezing level and the cloud top; COMPOSITE is the column maximum. Checked on synthetic columns against a plain
numpy copy of the NAM start study's offline check (bottom-up levels, as the history file stores them). Needs a CUDA
card; skips without one."""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001
    pytest.skip("no CUDA card", allow_module_level=True)

from namgpu import kernels  # noqa: E402

LM, JM, IM = 60, 3, 4
DT, NPHS = 25.0, 2
RDTPHS = 3.6e6 / (DT * NPHS)


def post_ref(refl_bu, t_bu, z_bu, cprate, cnvtop):
    """The study's offline formula (why-stronger-refl/tools/whyrefl.py post_refc) on [LM, J, I] bottom-up arrays."""
    lm = refl_bu.shape[0]
    warm = t_bu >= 273.15
    ktop = lm - 1 - np.argmax(warm[::-1], axis=0)
    zfrz = np.where(warm.any(0), np.take_along_axis(z_bu, ktop[None], 0)[0], z_bu[0])
    cuprate = RDTPHS * cprate
    has = (cuprate > 0) & (cnvtop >= 1)
    curefl_s = np.where(has, 300.0 * np.maximum(cuprate, 0) ** 1.4, 0.0)
    kct = np.clip(cnvtop.astype(int) - 1, 0, lm - 1)
    zct = np.take_along_axis(z_bu, kct[None], 0)[0]
    curefl_i = -2.0 / np.maximum(1000.0, zct - zfrz)
    k = np.arange(lm)[:, None, None]
    inside = has[None] & (k <= kct[None])
    delz = z_bu - zfrz[None]
    fctr = np.where(delz <= 0, 1.0, 10.0 ** (curefl_i[None] * delz))
    curefl = np.where(inside, fctr * curefl_s[None], 0.0)
    dbz = 10.0 * np.log10(np.maximum(0.01, 10.0 ** (0.1 * refl_bu) + curefl))
    return np.where(inside, np.maximum(-20.0, dbz), refl_bu)


def columns():
    rng = np.random.default_rng(7)
    z_bu = np.cumsum(np.full((LM, JM, IM), 250.0), axis=0) + 100.0          # 350 m .. 15 km
    t_bu = 300.0 - 0.0065 * z_bu                                             # freezing level near 4.3 km
    refl_bu = np.full((LM, JM, IM), -20.0)
    refl_bu[:20] = rng.uniform(-20.0, 35.0, (20, JM, IM))                    # microphysics echo low down
    cprate = np.zeros((JM, IM))
    cnvtop = np.zeros((JM, IM))
    # convective columns: 4 mm/h to cloud top level 40 (10 km); 15 mm/h to level 30; 1 mm/h to level 5 (below freezing)
    cprate[0, 0], cnvtop[0, 0] = 4.0 / RDTPHS, 40
    cprate[1, 2], cnvtop[1, 2] = 15.0 / RDTPHS, 30
    cprate[2, 3], cnvtop[2, 3] = 1.0 / RDTPHS, 5
    cprate[2, 1], cnvtop[2, 1] = 2.0 / RDTPHS, 0                             # rain with no cloud top: no term
    t_bu[:, 1, 2] = 260.0                                                    # a column with no warm level
    return [a.astype(np.float32) for a in (refl_bu, t_bu, z_bu, cprate, cnvtop)]


def run_kernel(refl_bu, t_bu, z_bu, cprate, cnvtop, write3=1):
    flip = lambda a: cp.asarray(np.ascontiguousarray(a[::-1]))               # state levels run top-down
    refl, t, z = flip(refl_bu), flip(t_bu), flip(z_bu)
    out3 = cp.zeros((LM, JM, IM), cp.float32)
    comp = cp.zeros((JM, IM), cp.float32)
    k = kernels.get("io_history_fields", "hist_refl_post")
    k(kernels.grid_cols(IM, JM), kernels.BLOCK_COL,
      (np.int32(IM), np.int32(JM), np.int32(LM), refl, t, z, cp.asarray(cprate), cp.asarray(cnvtop),
       np.float32(RDTPHS), np.int32(write3), out3, comp))
    return cp.asnumpy(out3)[::-1], cp.asnumpy(comp)


def test_matches_offline_formula():
    refl, t, z, cpr, top = columns()
    out3, comp = run_kernel(refl, t, z, cpr, top)
    ref = post_ref(refl.astype(np.float64), t.astype(np.float64), z.astype(np.float64), cpr.astype(np.float64), top)
    np.testing.assert_allclose(out3, ref, atol=2e-3)
    np.testing.assert_allclose(comp, ref.max(0), atol=2e-3)
    # the 4 mm/h column: 300 * 4**1.4 = 2,089 mm6 m-3, about 33.2 dBZ, at and below the freezing level
    assert out3[:10, 0, 0].min() >= 33.1
    # 20 dB weaker at the cloud top than at the freezing level (where the microphysics echo is gone)
    zf = (300.0 - 273.15) / 0.0065
    kf = int(np.argmin(np.abs(z[:, 0, 0] - zf)))
    assert out3[39, 0, 0] < out3[kf, 0, 0] - 15.0
    # above the cloud top nothing changes
    assert np.array_equal(out3[40:, 0, 0], refl[40:, 0, 0])


def test_no_convective_rain_is_byte_identical():
    refl, t, z, cpr, top = columns()
    out3, comp = run_kernel(refl, t, z, cpr, top)
    quiet = (cpr <= 0) | (top < 1)
    assert np.array_equal(out3[:, quiet], refl[:, quiet])
    assert np.array_equal(comp[quiet], refl.max(0)[quiet])


def test_composite_only_mode_matches():
    refl, t, z, cpr, top = columns()
    _, comp_full = run_kernel(refl, t, z, cpr, top, write3=1)
    _, comp_only = run_kernel(refl, t, z, cpr, top, write3=0)
    assert np.array_equal(comp_full, comp_only)
