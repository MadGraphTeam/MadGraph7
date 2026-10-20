! Integer bit check of Rust-formatted records through GNU REAL16 reading.
program real16_read
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  integer,parameter :: qp=selected_real_kind(33,4931)
  integer :: unit,ios,index,seen
  integer(int64) :: expected(2),actual(2),again(2)
  real(qp) :: value,roundtrip
  character(len=4096) :: path
  character(len=128) :: line
  character(len=64) :: text,packed
  character(len=16) :: low,high
  call get_command_argument(1,path)
  if(len_trim(path)==0) error stop 'record path required'
  if(storage_size(value)/=128.or.digits(value)/=113) error stop 'binary128 required'
  actual=transfer(1.0_qp,actual)
  if(actual(1)/=0_int64.or.actual(2)/=int(z'3FFF000000000000',int64)) error stop 'word layout'
  open(newunit=unit,file=trim(path),status='old',action='read')
  seen=0
  do
    read(unit,'(A)',iostat=ios) line
    if(ios<0) exit
    if(ios/=0) error stop 'header read'
    read(line,*) index,low,high
    seen=seen+1
    if(index/=seen.or.seen>14) error stop 'record order'
    read(low,'(Z16)') expected(1)
    read(high,'(Z16)') expected(2)
    read(unit,'(A)') text
    read(text,*,iostat=ios) value
    if(ios/=0) error stop 'REAL16 read failed'
    actual=transfer(value,actual)
    if(any(actual/=expected)) then
      write(*,'(A,I0,1X,4(Z16.16,1X))') 'MISMATCH ',index,expected,actual
      error stop 'REAL16 input word mismatch'
    endif
    ! Same write/read formats as installed mninjars.pack/unpack.
    write(packed,'(ES64.39E4)') value
    read(packed,*,iostat=ios) roundtrip
    if(ios/=0) error stop 'module-format read failed'
    again=transfer(roundtrip,again)
    if(any(again/=expected)) error stop 'module-format word mismatch'
    write(*,'(A,I0,1X,2(Z16.16,1X))') 'EXACT ',index,actual
  enddo
  close(unit)
  if(seen/=14) error stop 'missing records'
  write(*,*) 'COMPLETE',seen
end program
