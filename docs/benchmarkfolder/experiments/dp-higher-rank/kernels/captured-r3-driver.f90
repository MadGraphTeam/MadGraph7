program captured
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env,only:int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: a(4,0:69,4),b(4,0:4,4),expected(4,0:69,4),actual(4,0:69,4),checksum
integer(int64) :: abits(size(a)*2),bbits(size(b)*2),oldbits(size(actual)*2),newbits(size(actual)*2),tick0,tick1,rate
integer :: r,nc,nv,i,method,iteration
logical :: flags(size(ieee_all)),halts(size(ieee_all)),oldflags(size(ieee_all)),newflags(size(ieee_all)),entry,cold_entry,warm_entry
character(512) :: path,arg
call get_command_argument(1,path)
call get_command_argument(2,arg)
read(arg,*) method
open(10,file=trim(path),status='old')
read(10,*) r,nc,nv
if(r/=3.or.nc/=70.or.nv/=5) stop 10
read(10,*) flags
read(10,*) halts
if(any(halts)) stop 11
do i=1,size(abits)
 read(10,'(Z16)') abits(i)
enddo
do i=1,size(bbits)
 read(10,'(Z16)') bbits(i)
enddo
close(10)
a=reshape(transfer(abits,a),shape(a))
b=reshape(transfer(bbits,b),shape(b))
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
call ieee_set_flag(ieee_all,flags)
call TEST_UPDATE_WL_3_1(a,4,b,4,4,expected)
call ieee_get_flag(ieee_all,oldflags)
call ieee_set_flag(ieee_all,flags)
call ieee_get_flag(ieee_inexact,cold_entry)
call TEST_DP_TERMINAL_WL_3_1(a,4,b,4,4,actual,cold_entry)
call ieee_get_flag(ieee_all,newflags)
if(any(oldflags.neqv.newflags)) stop 12
oldbits=transfer(expected,oldbits)
newbits=transfer(actual,newbits)
if(.not.cold_entry.and.any(oldbits/=newbits)) stop 13
do i=1,4
 if(any(transfer(expected(i,:,i),oldbits,140)/=transfer(actual(i,:,i),newbits,140))) stop 14
enddo
if(any(transfer(a,abits)/=abits).or.any(transfer(b,bbits)/=bbits)) stop 15
call ieee_set_flag(ieee_all,flags)
do iteration=1,100
 if(method==0) then
  call TEST_UPDATE_WL_3_1(a,4,b,4,4,actual)
 else
  call ieee_get_flag(ieee_inexact,entry)
  call TEST_DP_TERMINAL_WL_3_1(a,4,b,4,4,actual,entry)
 endif
enddo
call ieee_get_flag(ieee_inexact,warm_entry)
checksum=cmplx(0.0_8,0.0_8,8)
call system_clock(tick0,rate)
do iteration=1,500
 if(method==0) then
  call TEST_UPDATE_WL_3_1(a,4,b,4,4,actual)
 else
  call ieee_get_flag(ieee_inexact,entry)
  call TEST_DP_TERMINAL_WL_3_1(a,4,b,4,4,actual,entry)
 endif
 checksum=checksum+actual(1,0,1)
enddo
call system_clock(tick1)
write(*,'(A,2(1X,L1),3(1X,ES24.16))') 'CAPTURE_PASS',cold_entry,warm_entry,real(tick1-tick0,8)/real(rate,8),real(checksum,8),aimag(checksum)
end program
