! Reference driver for phys-rrtm-lw-taugb01. Reads per-column TAUGB01 inputs written by
! tools/check-phys-rrtm-lw-taugb01-ref.py (float64 for real fields, int32 for indices, stream binary),
! calls taugb01_host (the NAM TAUGB01 body, see build.sh) once per column, writes taug and fracs.
program ref_taugb01
  use physparam, only: kind_phys
  implicit none
  integer, parameter :: LM = 60, NG = 10, MAXGAS = 6
  character(len=1024) :: dir, fn
  integer :: n, c
  integer, allocatable :: laytrop(:), indself(:,:), indfor(:,:), indminor(:,:), jp(:,:), jt(:,:), jt1(:,:)
  real(kind_phys), allocatable :: pavel(:,:), colamt(:,:,:), colbrd(:,:), scaleminorn2(:,:), selffac(:,:), &
       selffrac(:,:), forfac(:,:), forfrac(:,:), minorfrac(:,:), fac00(:,:), fac01(:,:), fac10(:,:), fac11(:,:)
  real(kind_phys), allocatable :: taug(:,:,:), fracs(:,:,:), buf(:)
  real(kind_phys) :: tg(NG,LM), fr(NG,LM)
  integer :: unit

  call get_command_argument(1, dir)
  call get_command_argument(2, fn)
  read(fn, *) n
  allocate(laytrop(n), pavel(LM,n), colamt(LM,MAXGAS,n), colbrd(LM,n), scaleminorn2(LM,n), selffac(LM,n), &
       selffrac(LM,n), forfac(LM,n), forfrac(LM,n), minorfrac(LM,n), fac00(LM,n), fac01(LM,n), fac10(LM,n), &
       fac11(LM,n), indself(LM,n), indfor(LM,n), indminor(LM,n), jp(LM,n), jt(LM,n), jt1(LM,n))
  allocate(taug(NG,LM,n), fracs(NG,LM,n), buf(LM*MAXGAS*n))

  call read_i(trim(dir)//'/laytrop.i32', laytrop, n)
  call read_r(trim(dir)//'/pavel.f64', pavel, LM*n)
  call read_r(trim(dir)//'/colbrd.f64', colbrd, LM*n)
  call read_r(trim(dir)//'/scaleminorn2.f64', scaleminorn2, LM*n)
  call read_r(trim(dir)//'/selffac.f64', selffac, LM*n)
  call read_r(trim(dir)//'/selffrac.f64', selffrac, LM*n)
  call read_r(trim(dir)//'/forfac.f64', forfac, LM*n)
  call read_r(trim(dir)//'/forfrac.f64', forfrac, LM*n)
  call read_r(trim(dir)//'/minorfrac.f64', minorfrac, LM*n)
  call read_r(trim(dir)//'/fac00.f64', fac00, LM*n)
  call read_r(trim(dir)//'/fac01.f64', fac01, LM*n)
  call read_r(trim(dir)//'/fac10.f64', fac10, LM*n)
  call read_r(trim(dir)//'/fac11.f64', fac11, LM*n)
  call read_i(trim(dir)//'/indself.i32', indself, LM*n)
  call read_i(trim(dir)//'/indfor.i32', indfor, LM*n)
  call read_i(trim(dir)//'/indminor.i32', indminor, LM*n)
  call read_i(trim(dir)//'/jp.i32', jp, LM*n)
  call read_i(trim(dir)//'/jt.i32', jt, LM*n)
  call read_i(trim(dir)//'/jt1.i32', jt1, LM*n)
  call read_r(trim(dir)//'/colamt.f64', colamt, LM*MAXGAS*n)

  do c = 1, n
    call taugb01_host(LM, laytrop(c), pavel(:,c), colamt(:,:,c), colbrd(:,c), scaleminorn2(:,c), &
         selffac(:,c), selffrac(:,c), indself(:,c), forfac(:,c), forfrac(:,c), indfor(:,c), &
         minorfrac(:,c), indminor(:,c), fac00(:,c), fac01(:,c), fac10(:,c), fac11(:,c), &
         jp(:,c), jt(:,c), jt1(:,c), tg, fr)
    taug(:,:,c) = tg
    fracs(:,:,c) = fr
  end do

  open(newunit=unit, file=trim(dir)//'/taug_ref.f64', access='stream', form='unformatted', status='replace')
  write(unit) taug
  close(unit)
  open(newunit=unit, file=trim(dir)//'/fracs_ref.f64', access='stream', form='unformatted', status='replace')
  write(unit) fracs
  close(unit)
  print '(a,i0,a)', 'ref_taugb01: wrote ', n, ' columns'

contains
  subroutine read_r(path, arr, cnt)
    character(len=*), intent(in) :: path
    integer, intent(in) :: cnt
    real(kind_phys), intent(inout) :: arr(cnt)
    integer :: u
    open(newunit=u, file=path, access='stream', form='unformatted', status='old')
    read(u) arr
    close(u)
  end subroutine read_r

  subroutine read_i(path, arr, cnt)
    character(len=*), intent(in) :: path
    integer, intent(in) :: cnt
    integer, intent(inout) :: arr(cnt)
    integer :: u
    open(newunit=u, file=path, access='stream', form='unformatted', status='old')
    read(u) arr
    close(u)
  end subroutine read_i
end program ref_taugb01
