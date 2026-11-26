program trap_probe
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: a(4,0:14,4), b(4,0:4,4), out(4,0:14,4)
logical :: entry, fast, flags(size(ieee_all))
type(ieee_flag_type) :: selected
integer :: mode, method, cut
character(20) :: text
call get_command_argument(1,text)
read(text,*) mode
call get_command_argument(2,text)
read(text,*) method
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
a=cmplx(1.0_8,0.0_8,kind=8)
b=cmplx(1.0_8,0.0_8,kind=8)
out=cmplx(-123.25_8,456.125_8,kind=8)
cut=4
select case(mode)
case(1,8)
  selected=ieee_invalid
  a(1,0,1)=cmplx(ieee_value(0.0_8,ieee_signaling_nan),0.0_8,kind=8)
  if(mode==8) cut=0
case(2)
  selected=ieee_overflow
  a=cmplx(huge(1.0_8),0.0_8,kind=8)
  b=cmplx(2.0_8,0.0_8,kind=8)
case(3)
  selected=ieee_underflow
  a=cmplx(transfer(1_int64,0.0_8),0.0_8,kind=8)
  b=cmplx(0.25_8,0.0_8,kind=8)
case(4)
  selected=ieee_inexact
  a=cmplx(0.1_8,0.1_8,kind=8)
  b=cmplx(0.1_8,0.1_8,kind=8)
case(5)
  selected=ieee_invalid
case(6)
  selected=ieee_divide_by_zero
case(7)
  selected=ieee_inexact
case default
  stop 10
end select
call ieee_set_flag(ieee_all,.false.)
call ieee_set_flag(ieee_inexact,.true.)
call ieee_get_flag(ieee_inexact,entry)
call ieee_set_halting_mode(selected,.true.)
fast=.false.
if(method==0) then
  call DP_DENSE(a,cut,b,4,4,out)
else
  call DP_TERMINAL(a,cut,b,4,4,out,entry,fast)
endif
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_get_flag(ieee_all,flags)
if(fast) stop 11
write(*,*) 'RETURN',mode,flags
end program
