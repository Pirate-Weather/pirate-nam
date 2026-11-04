! Host scope for the real NAM taumol22 subroutine (radsw_main_nmmb.f:4085-4201). In NAM the subroutine is
! contained in the radsw main module, so colmol, colamt, fac00.., id0/id1, taug/taur and the chunk size CHK
! are host-associated. This module supplies those names with the same shapes; the subroutine text is used
! unchanged. oneminus and f_one match radsw_main_nmmb.f:291-299 (eps = 1.0e-6, f_one = 1.0).
module hostvars
  use machine, only : kind_taum
  implicit none
  integer, parameter :: CHK = 8, NLAY = 60, NGPT = 112
  real (kind=kind_taum), parameter :: eps = 1.0e-6
  real (kind=kind_taum), parameter :: oneminus = 1.0 - eps
  real (kind=kind_taum), parameter :: f_one = 1.0
  real (kind=kind_taum) :: colmol(CHK,NLAY), colamt(CHK,NLAY,6)
  real (kind=kind_taum) :: fac00(CHK,NLAY), fac01(CHK,NLAY), fac10(CHK,NLAY), fac11(CHK,NLAY)
  real (kind=kind_taum) :: selffac(CHK,NLAY), selffrac(CHK,NLAY), forfac(CHK,NLAY), forfrac(CHK,NLAY)
  real (kind=kind_taum) :: taug(CHK,NLAY,NGPT), taur(CHK,NLAY,NGPT)
  integer :: laytrop(CHK), indself(CHK,NLAY), indfor(CHK,NLAY), id0(CHK,NLAY,22), id1(CHK,NLAY,22)
end module hostvars
