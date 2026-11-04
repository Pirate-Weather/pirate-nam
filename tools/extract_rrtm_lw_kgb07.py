"""Extract the RRTM LW band-7 k-tables and the module_radlw_ref chi_mls table.

Outputs (NAM nam.v4.2.11, 83df4e13c1, phys/):
  namgpu/data/rrtm_lw/kgb07.npz      module_radlw_kgb07 (radlw_datatb.f:21919-24540), the table
                                     that taugb07 (radlw_main_nmmb.f:4687-4942) actually reads through
                                     `use module_radlw_kgb07`. Arrays keep C order of the Fortran
                                     dims reversed: Fortran A(ig, m) -> npz a[m-1, ig-1].
                                     The NMMB-module table (module_radlw_kgb07_nmmb) is not read or extracted.
  namgpu/data/rrtm_lw/ref_chi_mls.npz  module_radlw_ref chi_mls(7,59) (radlw_datatb.f:733-905),
                                     stored as chi_mls[p-1, m-1].

Method: gfortran compiles the NAM source files unmodified (with a stub machine/physparam module for
kind_phys = 4) and small dumper programs write the arrays as Fortran stream binaries, so every
DATA statement is evaluated by the compiler. Nothing is retyped by hand.

Usage: python3 tools/extract_rrtm_lw_kgb07.py <NMMB_SRC>/src/atmos/phys
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PHYS = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parents[1] / "namgpu" / "data" / "rrtm_lw"

STUBS = """module machine
  implicit none
  integer, parameter :: kind_phys = 4
end module machine
module physparam
  use machine, only : kind_phys
  implicit none
end module physparam
"""

DUMP_GFS = """program dump_gfs
  use module_radlw_parameters, only : NG07
  use module_radlw_kgb07
  implicit none
  integer :: u
  open(newunit=u, file="gfs_kgb07.bin", access="stream", form="unformatted", status="replace")
  write(u) forref
  write(u) absa
  write(u) absb
  write(u) selfref
  write(u) fracrefa
  write(u) fracrefb
  write(u) ka_mco2
  write(u) kb_mco2
  close(u)
end program dump_gfs
"""

DUMP_REF = """program dump_ref
  use module_radlw_ref, only : chi_mls
  implicit none
  integer :: u
  open(newunit=u, file="ref_chi_mls.bin", access="stream", form="unformatted", status="replace")
  write(u) chi_mls
  close(u)
end program dump_ref
"""


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="kgb07-") as td:
        w = Path(td)
        (w / "stubs.f90").write_text(STUBS)
        for name, text in (("dump_gfs.f90", DUMP_GFS), ("dump_ref.f90", DUMP_REF)):
            (w / name).write_text(text)
        gf = ["gfortran", "-O0", "-J", str(w), "-I", str(w)]
        run(gf + ["-c", "stubs.f90", "-o", "stubs.o"], w)
        run(gf + ["-c", str(PHYS / "radlw_param.f"), "-o", "radlw_param.o"], w)
        run(gf + ["-c", str(PHYS / "radlw_datatb.f"), "-o", "radlw_datatb.o"], w)
        run(gf + ["dump_gfs.f90", "radlw_datatb.o", "radlw_param.o", "stubs.o", "-o", "dump_gfs"], w)
        run(gf + ["dump_ref.f90", "radlw_datatb.o", "radlw_param.o", "stubs.o", "-o", "dump_ref"], w)
        for exe in ("dump_gfs", "dump_ref"):
            run([str(w / exe)], w)

        # module_radlw_kgb07 (NG07=12, MFR07=4, MSA07=585, MSB07=235, MSF07=10, MAF07=9, MMC07=19)
        ng = 12
        raw = np.fromfile(w / "gfs_kgb07.bin", dtype=np.float32)
        off = 0
        def take(shape):
            nonlocal off
            n = int(np.prod(shape))
            a = raw[off:off + n].reshape(shape, order="F")
            off += n
            return a
        forref_f = take((ng, 4))
        absa_f = take((ng, 585))
        absb_f = take((ng, 235))
        selfref_f = take((ng, 10))
        fracrefa_f = take((ng, 9))
        fracrefb_f = take((ng,))
        ka_f = take((ng, 9, 19))
        kb_f = take((ng, 19))
        assert off == raw.size, (off, raw.size)
        kgb07 = {
            "absa": np.ascontiguousarray(absa_f.T),          # [585, 12]  absa[id-1, ig-1]
            "absb": np.ascontiguousarray(absb_f.T),          # [235, 12]
            "selfref": np.ascontiguousarray(selfref_f.T),    # [10, 12]
            "forref": np.ascontiguousarray(forref_f.T),      # [4, 12]
            "fracrefa": np.ascontiguousarray(fracrefa_f.T),  # [9, 12]
            "fracrefb": np.ascontiguousarray(fracrefb_f),    # [12]
            "ka_mco2": np.ascontiguousarray(np.transpose(ka_f, (2, 1, 0))),  # [19, 9, 12]
            "kb_mco2": np.ascontiguousarray(kb_f.T),         # [19, 12]
        }

        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "kgb07.npz", **kgb07)

        # module_radlw_ref chi_mls(7, 59)
        raw = np.fromfile(w / "ref_chi_mls.bin", dtype=np.float32)
        chi = raw.reshape((7, 59), order="F")                 # chi[m-1, p-1]
        np.savez(OUT / "ref_chi_mls.npz", chi_mls=np.ascontiguousarray(chi.T))  # [59, 7]

    print("wrote", OUT / "kgb07.npz", "and", OUT / "ref_chi_mls.npz")


if __name__ == "__main__":
    main()
