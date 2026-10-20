
program ownership_stress
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),expected(4,0:209,4),actual(4,0:209,4)
integer(8) saved_a(4*4*210*4),saved_b(4*4*15*4),bits(4*4*210*4)
integer i,j,k,lc,trial,checks,fast,fallback
common/SHARED_TRACE/fast,fallback
logical old_flags(size(ieee_all)),new_flags(size(ieee_all))
call ieee_set_rounding_mode(ieee_nearest)
call ieee_set_halting_mode(ieee_all,.false.)
checks=0
fast=0
fallback=0
do trial=1,100
 do k=1,4
  do j=0,209
   do i=1,4
    a(i,j,k)=cmplx(real(mod(7*i+11*j+13*k+trial,37)-18,16)/7,&
                  real(mod(11*i+5*j+3*k+trial,31)-15,16)/11,16)
    if(mod(i+3*j+7*k+trial,3)==0) a(i,j,k)=0
    if(mod(i+j+k+trial,11)==0) a(i,j,k)=cmplx(-0.0_16,0.0_16,16)
   enddo
  enddo
  do j=0,14
   do i=1,4
    b(i,j,k)=cmplx(real(i+3*j+k+trial,16)/13,real(2*i+j-3*k-trial,16)/17,16)
    if(mod(i+j+k+trial,4)==0) b(i,j,k)=0
    if(mod(i+j+k+trial,9)==0) b(i,j,k)=cmplx(0.0_16,-0.0_16,16)
   enddo
  enddo
 enddo
 b(1,4,1)=b(1,1,1)
b(4,4,4)=b(4,1,4)
b(1,4,2)=b(1,1,2)
b(4,4,3)=b(4,1,3)
 saved_a=transfer(a,saved_a)
 saved_b=transfer(b,saved_b)
 do lc=0,4
  expected=cmplx(5.0_16,7.0_16,16)
  actual=cmplx(-7.0_16,-5.0_16,16)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call TEST_TERMINAL_WL_4_1(a,lc,b,4,4,expected,.true.)
  call ieee_get_flag(ieee_all,old_flags)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call TEST_TERMINAL_WL_4_1_SHARED(a,lc,b,4,4,actual,.true.)
  call ieee_get_flag(ieee_all,new_flags)
  if(any(transfer(expected,bits)/=transfer(actual,bits))) stop 1
  if(any(old_flags.neqv.new_flags)) stop 2
  if(any(transfer(a,saved_a)/=saved_a).or.any(transfer(b,saved_b)/=saved_b)) stop 3
  checks=checks+1
 enddo
enddo
if(checks/=500.or.fast/=500.or.fallback/=0) stop 4
write(*,*) 'PASS',checks,fast,fallback
end program
