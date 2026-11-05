"""woof-nam doctor: what this machine has for WOOF-NAM, and what its card can run.

Checks, each one line: the card (nvidia-smi: name, memory, driver), CuPy and NVRTC (a small kernel compiled and run),
the WPS_GEOG terrain and land data a new grid's static build reads, the map renderer, the GRIB2 writer of
`woof-nam products`, and what fits on the card by the same memory estimate the `nest` and `run` doors refuse with
(nest/domain.py vram_estimate): the 12 km CONUS grid and the largest square 3 km nest.
Exit 0 when the card and CuPy work (missing data and writers are listed with the command that fetches them),
1 when there is no usable card or CuPy cannot compile.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

CONUS12 = {"im": 954, "jm": 835}


def _line(ok: str, what: str, detail: str) -> None:
    print(f"  {ok:<7} {what:<14} {detail}", flush=True)


def card() -> dict | None:
    from namgpu.nest import domain
    idx = domain._visible_index()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader,nounits", "-i", idx], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()
        name, total, free, drv = (x.strip() for x in out[0].split(","))
        return {"index": idx, "name": name, "total_gib": float(total) / 1024, "free_gib": float(free) / 1024,
                "driver": drv}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cupy_check() -> tuple[bool, str]:
    try:
        import cupy as cp
        from cupy.cuda import nvrtc
        k = cp.RawKernel(r'extern "C" __global__ void twice(float* x) { x[threadIdx.x] *= 2.0f; }', "twice")
        x = cp.arange(32, dtype=cp.float32)
        k((1,), (32,), (x,))
        ok = float(x[31].get()) == 62.0
        nv = nvrtc.getVersion()
        rt = cp.cuda.runtime.runtimeGetVersion()
        return ok, (f"CuPy {cp.__version__}, CUDA runtime {rt // 1000}.{rt % 1000 // 10}, NVRTC {nv[0]}.{nv[1]}: "
                    f"{'a test kernel compiled and ran' if ok else 'the test kernel gave a wrong answer'}")
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


def terrain() -> tuple[bool, str]:
    from namgpu.init import static_build as sb
    roots = sb.geog_roots()
    try:
        rows = sb.resolve_datasets(roots, soil=sb.soil_set())
    except Exception as e:  # noqa: BLE001
        return False, (f"{str(e).splitlines()[0]} (new nest grids build their static fields from it; "
                       f"`gpuwm fetch-geog --root ~/WPS_GEOG`)")
    return True, f"{len(rows)} WPS_GEOG datasets under {', '.join(str(r) for r in roots)}"


def renderer() -> tuple[bool, str]:
    from namgpu.io import maps
    try:
        return True, str(maps.renderer())
    except SystemExit as e:
        return False, f"{e} (maps and sheets need it; the nest door refuses without it)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def writer() -> tuple[bool, str]:
    from namgpu import exporter
    p = exporter.find()
    if p is not None:
        return True, str(p)
    return False, exporter.missing_message() + " (only `woof-nam products` needs it; it fetches it on first use)"


def what_fits(c: dict) -> list[str]:
    from namgpu.nest import domain
    # one nvidia-smi reading of the card for the whole search (the estimate asks for its name and size per call)
    name, total = domain.card_name(c["index"]), domain.card_total_gib(c["index"])
    keep = domain.card_name, domain.card_total_gib
    domain.card_name, domain.card_total_gib = (lambda index=None: name), (lambda index=None: total)
    try:
        return _what_fits(c)
    finally:
        domain.card_name, domain.card_total_gib = keep


def _what_fits(c: dict) -> list[str]:
    from namgpu import state as statemod
    from namgpu.nest import domain
    out = []
    free = c["free_gib"]
    # the run door's check (run_forecast.memory_check): the fit path's working set, the smallest any path has
    e = domain.vram_estimate(CONUS12, statemod.FIT_RRTM_ROWS, 1, c["index"], path="fit")
    verdict = "yes" if e["total_gib"] <= free else "no"
    out.append(f"12 km CONUS (woof-nam run, 954 x 835): {verdict}, needs about {e['total_gib']} GiB, "
               f"{free:.2f} GiB free")
    for n in (100, 333):
        pf = domain.preflight({"im": n, "jm": n}, free=free, index=c["index"])
        out.append(f"3 km nest {n} x {n} (woof-nam nest): " + (
            f"yes, radiation in {pf['pick']['rrtm_rows']}-row chunks, estimate {pf['pick']['total_gib']} GiB"
            if pf["ok"] else f"no, needs about {pf['tried'][-1]['total_gib']} GiB + {domain.VRAM['margin_gib']}"))
    best = None
    for n in range(50, 3001, 10):
        pf = domain.preflight({"im": n, "jm": n}, free=free, index=c["index"])
        if not pf["ok"]:
            break
        best = (n, pf["pick"])
    if best:
        n, p = best
        out.append(f"largest square nest the door admits now: about {n} x {n} points ({n * 3} km at 3 km), "
                   f"radiation in {p['rrtm_rows']}-row chunks, estimate {p['total_gib']} GiB")
    else:
        out.append("largest square nest the door admits now: none (less than 50 x 50 points fits)")
    return out


def main(argv=None) -> int:
    from namgpu import __version__
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1],
                                 formatter_class=HelpFormatter)
    ap.parse_args(argv)
    print(f"woof-nam {__version__} (Python {sys.version.split()[0]})", flush=True)
    bad = 0
    c = card()
    if c is None:
        _line("MISSING", "card", "nvidia-smi gave no NVIDIA card (WOOF-NAM needs one, with a driver for CUDA 12 or 13)")
        bad += 1
    else:
        from namgpu.nest import domain
        sms = domain.sm_threads(c["index"]) // 1536
        known = any(k in c["name"] for k, _ in domain.CARD_SMS)
        _line("ok", "card", f"{c['name']} (card {c['index']}), {c['total_gib']:.1f} GiB, {c['free_gib']:.2f} GiB free, "
                            f"driver {c['driver']}, {sms} SMs{'' if known else ' (not in the SM table: counted as 170)'}")
    ok, msg = cupy_check()
    _line("ok" if ok else "FAILED", "cupy/nvrtc", msg)
    bad += 0 if ok else 1
    for what, fn in (("terrain data", terrain), ("map renderer", renderer), ("GRIB2 writer", writer)):
        ok, msg = fn()
        _line("ok" if ok else "missing", what, msg)
    if c is not None:
        print("what this card can run now (the doors' memory estimate against the free memory):", flush=True)
        for line in what_fits(c):
            print(f"  {line}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
