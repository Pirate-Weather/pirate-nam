! Move whole-domain records between a NEMSIO NMMB file and float32 files, so WOOF-NAM can stand in for the Fortran forecast.
!   nemsio_fields list  FILE          prints every record name, level type and level
!   nemsio_fields dump  FILE DIR NAMES   writes DIR/<name>__<levtyp>.bin, level L at offset (L-1)*im*jm*4
!   nemsio_fields apply FILE DIR      overwrites every record that has a matching DIR/<name>__<levtyp>.bin
! Level types have their spaces replaced by underscores in the file name. NAMES is comma separated.
program nemsio_fields
  use nemsio_module
  implicit none
  type(nemsio_gfile) :: g
  character(len=8) :: mode
  character(len=512) :: fname, dir
  character(len=4096) :: names
  character(len=16), allocatable :: recname(:), reclevtyp(:)
  integer, allocatable :: reclev(:)
  real(4), allocatable :: a(:)
  integer :: iret, im, jm, nrec, i, u, n, total
  integer(8) :: pos
  logical :: there
  character(len=600) :: path
  call get_command_argument(1, mode); call get_command_argument(2, fname); call get_command_argument(3, dir); call get_command_argument(4, names)
  call nemsio_init(iret=iret)
  call nemsio_open(g, trim(fname), merge('RDWR', 'READ', trim(mode) == 'apply'), iret=iret)
  if (iret /= 0) error stop 'open failed'
  call nemsio_getfilehead(g, iret=iret, dimx=im, dimy=jm, nrec=nrec)
  allocate(recname(nrec), reclevtyp(nrec), reclev(nrec), a(im*jm))
  call nemsio_getfilehead(g, iret=iret, recname=recname, reclevtyp=reclevtyp, reclev=reclev)
  print *, 'dims', im, jm, 'nrec', nrec
  n = 0; total = im*jm
  do i = 1, nrec
    if (trim(mode) == 'list') then
      print '(a16,1x,a16,i4)', recname(i), reclevtyp(i), reclev(i); cycle
    end if
    path = trim(dir)//'/'//trim(recname(i))//'__'//underscored(reclevtyp(i))//'.bin'
    pos = int(reclev(i) - 1, 8)*total*4 + 1
    if (trim(mode) == 'dump') then
      if (index(','//trim(names)//',', ','//trim(recname(i))//',') == 0) cycle
      call nemsio_readrecv(g, trim(recname(i)), trim(reclevtyp(i)), reclev(i), a, iret=iret)
      if (iret /= 0) error stop 'read failed'
      open(newunit=u, file=trim(path), access='stream', form='unformatted', status='unknown', convert='native')
      write(u, pos=pos) a; close(u); n = n + 1
    else if (trim(mode) == 'apply') then
      inquire(file=trim(path), exist=there); if (.not. there) cycle
      open(newunit=u, file=trim(path), access='stream', form='unformatted', status='old', convert='native')
      read(u, pos=pos, iostat=iret) a; close(u)
      ! A short file must not leave the template's stale record as GSI's guess. A missing file is normal: that record isn't exported.
      if (iret /= 0) error stop 'short field file'
      call nemsio_writerecv(g, trim(recname(i)), trim(reclevtyp(i)), reclev(i), a, iret=iret)
      if (iret /= 0) error stop 'write failed'
      n = n + 1
    end if
  end do
  print *, 'records', n
  call nemsio_close(g, iret=iret); call nemsio_finalize()
contains
  function underscored(s) result(r)
    character(len=*), intent(in) :: s
    character(len=len_trim(s)) :: r
    integer :: k
    r = s(1:len_trim(s))
    do k = 1, len(r); if (r(k:k) == ' ') r(k:k) = '_'; end do
  end function
end program
