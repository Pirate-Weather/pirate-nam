! Driver for the real NAM taumol22 subroutine. Reads G groups of CHK columns (the Fortran chunk of 8 columns,
! so the chunk-minimum behaviour at k == laytrop is the reference's own), computes the band-22 id0/id1 with the
! taumol dispatcher's WHERE blocks (radsw_main_nmmb.f:3249-3256, nspa(22) = 9, nspb(22) = 1 from the DATA at
! lines 308-309), calls taumol22, and writes taug/taur planes NS22+1..NS22+2 for every group.
!
! usage: harness <inputs_dir> <output.f32> <G>
! inputs (stream, Fortran order, first index fastest), one file per field:
!   real4  colmol, fac00, fac01, fac10, fac11, selffac, selffrac, forfac, forfrac   each (CHK,NLAY,G)
!          colamt (CHK,NLAY,6,G)
!   int4   indself, indfor, jp, jt, jt1  each (CHK,NLAY,G);  laytrop (CHK,G)
! output (stream): taug(CHK,NLAY,2,G), then taur(CHK,NLAY,2,G), real4.
program harness
  use hostvars
  use module_radsw_parameters_nmmb, only : NS22
  use taumol22_mod, only : taumol22
  implicit none
  integer :: G, gi, i, k, n, p, nspa22, nspb22, jpv, jtv, jt1v, nn
  character(len=512) :: indir, outfile, gstr
  real (kind=4), allocatable :: f_colmol(:), f_colamt(:), f_fac00(:), f_fac01(:), f_fac10(:), f_fac11(:)
  real (kind=4), allocatable :: f_selffac(:), f_selffrac(:), f_forfac(:), f_forfrac(:)
  integer (kind=4), allocatable :: f_laytrop(:), f_indself(:), f_indfor(:), f_jp(:), f_jt(:), f_jt1(:)
  real (kind=4), allocatable :: out_buf(:)
  real (kind=4) :: tmp

  call get_command_argument(1, indir)
  call get_command_argument(2, outfile)
  call get_command_argument(3, gstr)
  read(gstr, *) G
  nspa22 = 9
  nspb22 = 1
  nn = CHK*NLAY*G

  allocate(f_colmol(nn), f_colamt(nn*6), f_fac00(nn), f_fac01(nn), f_fac10(nn), f_fac11(nn))
  allocate(f_selffac(nn), f_selffrac(nn), f_forfac(nn), f_forfrac(nn))
  allocate(f_laytrop(CHK*G), f_indself(nn), f_indfor(nn), f_jp(nn), f_jt(nn), f_jt1(nn))
  allocate(out_buf(2*CHK*NLAY*2*G))

  call rd4(trim(indir)//'/colmol.f32',  f_colmol,  nn)
  call rd4(trim(indir)//'/colamt.f32',  f_colamt,  nn*6)
  call rd4(trim(indir)//'/fac00.f32',   f_fac00,   nn)
  call rd4(trim(indir)//'/fac01.f32',   f_fac01,   nn)
  call rd4(trim(indir)//'/fac10.f32',   f_fac10,   nn)
  call rd4(trim(indir)//'/fac11.f32',   f_fac11,   nn)
  call rd4(trim(indir)//'/selffac.f32', f_selffac, nn)
  call rd4(trim(indir)//'/selffrac.f32', f_selffrac, nn)
  call rd4(trim(indir)//'/forfac.f32',  f_forfac,  nn)
  call rd4(trim(indir)//'/forfrac.f32', f_forfrac, nn)
  call ri4(trim(indir)//'/laytrop.i32', f_laytrop, CHK*G)
  call ri4(trim(indir)//'/indself.i32', f_indself, nn)
  call ri4(trim(indir)//'/indfor.i32',  f_indfor,  nn)
  call ri4(trim(indir)//'/jp.i32',      f_jp,      nn)
  call ri4(trim(indir)//'/jt.i32',      f_jt,      nn)
  call ri4(trim(indir)//'/jt1.i32',     f_jt1,     nn)

  do gi = 1, G
    do i = 1, CHK
      laytrop(i) = f_laytrop(i + CHK*(gi-1))
    end do
    do k = 1, NLAY
      do i = 1, CHK
        p = i + CHK*((k-1) + NLAY*(gi-1))
        colmol(i,k)   = f_colmol(p)
        fac00(i,k)    = f_fac00(p)
        fac01(i,k)    = f_fac01(p)
        fac10(i,k)    = f_fac10(p)
        fac11(i,k)    = f_fac11(p)
        selffac(i,k)  = f_selffac(p)
        selffrac(i,k) = f_selffrac(p)
        forfac(i,k)   = f_forfac(p)
        forfrac(i,k)  = f_forfrac(p)
        indself(i,k)  = f_indself(p)
        indfor(i,k)   = f_indfor(p)
        jpv = f_jp(p); jtv = f_jt(p); jt1v = f_jt1(p)
        ! taumol dispatcher, band 22 (radsw_main_nmmb.f:3249-3256)
        if (k <= laytrop(i)) then
          id0(i,k,22) = ((jpv-1)*5 + (jtv-1))  * nspa22
          id1(i,k,22) = ( jpv*5    + (jt1v-1)) * nspa22
        else
          id0(i,k,22) = ((jpv-13)*5 + (jtv-1))  * nspb22
          id1(i,k,22) = ((jpv-12)*5 + (jt1v-1)) * nspb22
        end if
        do n = 1, 6
          colamt(i,k,n) = f_colamt(i + CHK*((k-1) + NLAY*((n-1) + 6*(gi-1))))
        end do
      end do
    end do

    call taumol22

    do k = 1, NLAY
      do i = 1, CHK
        do p = 1, 2
          out_buf(i + CHK*((k-1) + NLAY*((p-1) + 2*(gi-1)))) = real(taug(i,k,NS22+p), 4)
        end do
      end do
    end do
    call wr_group_taur(gi)
  end do

  call wr4(trim(outfile), out_buf)
contains
  subroutine wr_group_taur(gi)
    integer, intent(in) :: gi
    integer :: i, k, p, base
    base = CHK*NLAY*2*G
    do k = 1, NLAY
      do i = 1, CHK
        do p = 1, 2
          out_buf(base + i + CHK*((k-1) + NLAY*((p-1) + 2*(gi-1)))) = real(taur(i,k,NS22+p), 4)
        end do
      end do
    end do
  end subroutine wr_group_taur

  subroutine rd4(name, a, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real (kind=4), allocatable, intent(inout) :: a(:)
    integer :: u
    open(newunit=u, file=name, access='stream', form='unformatted', status='old', action='read')
    read(u) a(1:n)
    close(u)
  end subroutine rd4

  subroutine ri4(name, a, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    integer (kind=4), allocatable, intent(inout) :: a(:)
    integer :: u
    open(newunit=u, file=name, access='stream', form='unformatted', status='old', action='read')
    read(u) a(1:n)
    close(u)
  end subroutine ri4

  subroutine wr4(name, a)
    character(len=*), intent(in) :: name
    real (kind=4), intent(in) :: a(:)
    integer :: u
    open(newunit=u, file=name, access='stream', form='unformatted', status='replace')
    write(u) a
    close(u)
  end subroutine wr4
end program harness
