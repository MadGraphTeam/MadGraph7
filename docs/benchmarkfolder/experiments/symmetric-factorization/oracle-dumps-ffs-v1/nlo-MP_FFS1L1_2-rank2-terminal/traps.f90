program traps
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer flag,checks,fast,fallback
common/FACTOR_TRACE/fast,fallback
logical raised(size(ieee_all))
checks=0
do flag=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(flag))) cycle
 call ieee_set_halting_mode(ieee_all,.false.)
 a=0
 b=0
 fast=0
 fallback=0
 call ieee_set_flag(ieee_all,.false.)
 call ieee_set_halting_mode(ieee_all(flag),.true.)
 call TEST_TERMINAL_WL_2_1_FACTOR(a,4,b,4,4,out,.true.)
 call ieee_set_halting_mode(ieee_all,.false.)
 call ieee_get_flag(ieee_all,raised)
 if(fast/=0.or.fallback/=1.or.any(raised).or.any(out/=0)) stop 1
 checks=checks+1
enddo
if(checks/=size(ieee_all)) stop 2
write(*,*) 'PASS',checks
end program
