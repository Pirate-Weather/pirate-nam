#!/bin/bash
# Build the NAM TAUGB01 reference driver. The routine body is taken verbatim from the NAM clone
# (radlw_main_nmmb.f, subroutine taugb01, nam.v4.2.11 83df4e13c1) and wrapped as taugb01_host so the
# driver can call it per column. Host inputs become dummy arguments; the Fortran statements are unchanged.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAM=${NAM:-$NMMB_SRC/phys}
OUT=${OUT:-$HERE/build}
mkdir -p "$OUT"
SRC=$NAM/radlw_main_nmmb.f
S=$(grep -n "^ *subroutine taugb01 *$" "$SRC" | head -1 | cut -d: -f1)
L=$(awk -v s="$S" 'NR>s && /!  ---  locals:/ {print NR; exit}' "$SRC")
E=$(awk -v s="$S" 'NR>s && /end subroutine taugb01/ {print NR; exit}' "$SRC")
cat > "$OUT/taugb01_host.f90" <<F90
subroutine taugb01_host(nlay, laytrop, pavel, colamt, colbrd, scaleminorn2, selffac, selffrac, indself, &
     forfac, forfrac, indfor, minorfrac, indminor, fac00, fac01, fac10, fac11, jp, jt, jt1, taug, fracs)
  use physparam, only: kind_phys
  use module_radlw_parameters, only: ng01
  use module_radlw_kgb01
  implicit none
  real(kind_phys), parameter :: f_one = 1.0_kind_phys, f_zero = 0.0_kind_phys
  integer, dimension(16) :: nspa, nspb
  data nspa / 1, 1, 9, 9, 9, 1, 9, 1, 9, 1, 1, 9, 9, 1, 9, 9 /
  data nspb / 1, 1, 5, 5, 5, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0 /
  integer, intent(in) :: nlay, laytrop
  real(kind_phys), intent(in) :: pavel(nlay), colamt(nlay,6), colbrd(nlay), scaleminorn2(nlay), &
       selffac(nlay), selffrac(nlay), forfac(nlay), forfrac(nlay), minorfrac(nlay), &
       fac00(nlay), fac01(nlay), fac10(nlay), fac11(nlay)
  integer, intent(in) :: indself(nlay), indfor(nlay), indminor(nlay), jp(nlay), jt(nlay), jt1(nlay)
  real(kind_phys), intent(out) :: taug(10,nlay), fracs(10,nlay)
F90
sed -n "${L},${E}p" "$SRC" | sed 's/end subroutine taugb01/end subroutine taugb01_host/' >> "$OUT/taugb01_host.f90"
cd "$OUT"
gfortran -O2 -ffp-contract=off -ffree-line-length-none -I"$NAM" -c "$NAM/physparam.f" -o physparam.o 2>/dev/null || true
gfortran -O2 -ffp-contract=off -ffree-line-length-none -I"$NAM" -c "$NAM/radlw_param.f" -o radlw_param.o 2>/dev/null || true
gfortran -O2 -ffp-contract=off -ffree-line-length-none -I"$NAM" -c taugb01_host.f90 -o taugb01_host.o
gfortran -O2 -ffp-contract=off -ffree-line-length-none -I"$NAM" -I. -c "$HERE/main.f90" -o main.o
gfortran -o ref_taugb01 main.o taugb01_host.o "$NAM/radlw_datatb.o" "$NAM/radlw_param.o" "$NAM/physparam.o"
echo "built $OUT/ref_taugb01 from $SRC lines $L-$E"
