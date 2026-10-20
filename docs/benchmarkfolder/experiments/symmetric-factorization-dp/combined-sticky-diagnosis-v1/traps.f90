program boundary_traps
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use, intrinsic :: iso_fortran_env,only:output_unit
implicit none
complex*16 a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer flag,backend,scenario,fast,fallback,i,initial_mask
common/FACTOR_TRACE/fast,fallback
logical masks(size(ieee_all)),observed(size(ieee_all)),raised(size(ieee_all)),initial_flags(size(ieee_all))
character(32) arg
call get_command_argument(1,arg)
read(arg,*) backend
call get_command_argument(2,arg)
read(arg,*) scenario
call get_command_argument(3,arg)
read(arg,*) flag
call get_command_argument(4,arg)
read(arg,*) initial_mask
do i=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(i))) stop 1
enddo
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
a=0
b=0
select case(scenario)
case(0)
 a(1,0,1)=cmplx(2.0_8,0.0_8,8)
 b(1,0,1)=cmplx(4.0_8,0.0_8,8)
case(1)
 a(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),a(1,0,1))
 b(1,0,1)=1
case(2)
 a(1,0,1)=1
 b(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),b(1,0,1))
case(3)
 a(1,0,1)=cmplx(huge(1.0_8),0.0_8,8)
 b(1,0,1)=4
case(4)
 a(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
 b(1,0,1)=cmplx(0.5_8,0.0_8,8)
case(5)
 a(1,0,1)=cmplx(1.0_8+epsilon(1.0_8),0.0_8,8)
 b(1,0,1)=a(1,0,1)
case default
 stop 2
end select
do i=1,size(ieee_all)
 initial_flags(i)=btest(initial_mask,i-1)
enddo
call ieee_set_flag(ieee_all,initial_flags)
call ieee_get_flag(ieee_all,raised)
write(*,*) "AFTER_SET_FLAGS",initial_flags,raised
masks=.false.
select case(scenario)
case(0)
 masks(flag)=.true.
case(1,2)
 call ieee_set_halting_mode(ieee_invalid,.true.)
case(3)
 call ieee_set_halting_mode(ieee_overflow,.true.)
case(4)
 call ieee_set_halting_mode(ieee_underflow,.true.)
case(5)
 call ieee_set_halting_mode(ieee_inexact,.true.)
end select
if(scenario==0) call ieee_set_halting_mode(ieee_all,masks)
call ieee_get_flag(ieee_all,raised)
write(*,*) "AFTER_ENABLE",initial_flags,raised
call ieee_get_halting_mode(ieee_all,masks)
call ieee_get_flag(ieee_all,raised)
write(*,*) "AFTER_GET_MASK",initial_flags,raised
fast=0
fallback=0
write(output_unit,*) 'ENTER',backend,scenario,flag
flush(output_unit)
if(backend==0) then
 call TEST_UPDATE_WL_2_1(a,4,b,4,4,out)
else
 call TEST_UPDATE_WL_2_1_FACTOR(a,4,b,4,4,out,.true.)
endif
call ieee_get_halting_mode(ieee_all,observed)
call ieee_get_flag(ieee_all,raised)
call ieee_set_halting_mode(ieee_all,.false.)
if(any(masks.neqv.observed)) stop 3
if(scenario/=0) stop 4
write(*,*) "AFTER_CALL",initial_flags,raised
if(any(raised.neqv.initial_flags)) stop 5
if(out(1,0,1)/=cmplx(8.0_8,0.0_8,8)) stop 6
if(backend==1.and.(fast/=0.or.fallback/=1)) stop 7
write(*,*) 'PASS',backend,scenario,flag
end program
