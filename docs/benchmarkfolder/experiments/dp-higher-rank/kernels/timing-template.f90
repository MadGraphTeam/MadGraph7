program timing
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: a(4,0:209,4),b(4,0:4,4),out(4,0:209,4),checksum
logical :: entry
integer :: i,j,k,method,iteration
integer,parameter :: iterations=500
integer*8 :: tick0,tick1,rate
character(12) :: arg
call get_command_argument(1,arg)
read(arg,*) method
call ieee_set_halting_mode(ieee_all,.false.)
do k=1,4
 do j=0,209
  do i=1,4
   a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,8)/7,real(mod(37*i+11*j+7*k,31)-15,8)/13,8)
  enddo
 enddo
 do j=0,4
  do i=1,4
   b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,8)/3,real(mod(11*i+17*j+13*k,23)-11,8)/11,8)
  enddo
 enddo
enddo
call ieee_set_flag(ieee_inexact,.true.)
do iteration=1,100
 if(method==0) then
  call TEST_UPDATE_WL_RANK_ID_1(a,4,b,4,4,out)
 else
  call ieee_get_flag(ieee_inexact,entry)
  call TEST_DP_TERMINAL_WL_RANK_ID_1(a,4,b,4,4,out,entry)
 endif
enddo
checksum=cmplx(0.0_8,0.0_8,8)
call system_clock(tick0,rate)
do iteration=1,iterations
 if(method==0) then
  call TEST_UPDATE_WL_RANK_ID_1(a,4,b,4,4,out)
 else
  call ieee_get_flag(ieee_inexact,entry)
  call TEST_DP_TERMINAL_WL_RANK_ID_1(a,4,b,4,4,out,entry)
 endif
 checksum=checksum+out(1,0,1)
enddo
call system_clock(tick1)
write(*,'(A,1X,I0,3(1X,ES24.16))') 'TIMING',iterations,real(tick1-tick0,8)/real(rate,8),real(checksum,8),aimag(checksum)
end program
