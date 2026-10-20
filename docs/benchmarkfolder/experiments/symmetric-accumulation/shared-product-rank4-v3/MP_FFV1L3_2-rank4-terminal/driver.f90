
program exact_sparse
use iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex(16) :: a(4,0:209,4),b(4,0:14,4),expected(4,0:209,4),actual(4,0:209,4)
complex(16) :: sentinel
integer(int64) :: old_bits(4*4*210*4),new_bits(4*4*210*4),saved_a(4*4*210*4)
integer(int64) :: b_bits(4*4*15*4),saved_b(4*4*15*4)
real(16) :: scale,inf,nan,snan,small
logical :: old_flags(size(ieee_all)),new_flags(size(ieee_all))
type(ieee_round_type) :: modes(4)
integer :: i,j,k,ls,ins,outs,trial,rounding,checks,fast,fallback,flagstate
common/SHARED_TRACE/fast,fallback
modes=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
sentinel=cmplx(-123.25_16,456.125_16,kind=16)
inf=ieee_value(0.0_16,ieee_positive_inf)
nan=ieee_value(0.0_16,ieee_quiet_nan)
snan=ieee_value(0.0_16,ieee_signaling_nan)
small=nearest(0.0_16,1.0_16)
checks=0
fast=0
fallback=0
do flagstate=0,1
do rounding=1,4
  call ieee_set_rounding_mode(modes(rounding))
  do trial=1,12
    do k=1,4
      do j=0,209
        do i=1,4
          scale=1.0_16
          if(trial==3.and.mod(j,3)==0) scale=1.e2000_16
          if(trial==3.and.mod(j,3)==1) scale=1.e-2000_16
          a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind=16)/7, &
            real(mod(37*i+11*j+7*k,31)-15,kind=16)/13,kind=16)*scale
          if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==4) a(i,j,k)=0
          if(trial==5) a(i,j,k)=cmplx((-1.0_16)**i,(-1.0_16)**(i+1),kind=16)
          if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_16,0.0_16,kind=16)
          if(trial==7) a(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==7.and.mod(i+j+k,5)==0) a(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==8.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==9) a(i,j,k)=cmplx(small,-small,kind=16)
          if(trial==10.and.mod(i+j+k,2)==0) a(i,j,k)=0
          if(trial==10.and.mod(i+j+k,2)==1) a(i,j,k)=cmplx(huge(scale),-huge(scale),kind=16)
          if(trial==11) a(i,j,k)=cmplx(snan,0.0_16,kind=16)
          if(trial==12) a(i,j,k)=0
        enddo
      enddo
      do j=0,14
        do i=1,4
          b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind=16)/3, &
            real(mod(11*i+17*j+13*k,23)-11,kind=16)/11,kind=16)
          if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
          if(trial==4.or.trial==7) b(i,j,k)=0
          if(trial==5) b(i,j,k)=cmplx(1.0_16,-1.0_16,kind=16)
          if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==8.and.mod(i+j+k,5)==0) b(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==9) b(i,j,k)=cmplx(0.25_16,-0.25_16,kind=16)
          if(trial==10.and.mod(i+j+k,2)==1) b(i,j,k)=0
          if(trial==11) b(i,j,k)=0
          if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)
        enddo
      enddo
    enddo
if(mod(trial,2)==1) then
b(2,4,2)=b(2,1,2)
b(2,4,1)=b(2,1,1)
b(2,4,4)=b(2,1,4)
b(2,4,3)=b(2,1,3)
b(3,4,1)=b(3,1,1)
b(3,4,2)=b(3,1,2)
b(3,4,3)=b(3,1,3)
b(3,4,4)=b(3,1,4)
endif
    saved_a=transfer(a,saved_a)
    saved_b=transfer(b,saved_b)
    do ls=0,4
      do ins=0,4
        do outs=0,4

expected=sentinel
actual=sentinel
call ieee_set_flag(ieee_all,.false.)
if(flagstate==1) call ieee_set_flag(ieee_inexact,.true.)
call TEST_TERMINAL_WL_4_1(a,ls,b,ins,outs,expected,flagstate==1)
call ieee_get_flag(ieee_all,old_flags)
call ieee_set_flag(ieee_all,.false.)
if(flagstate==1) call ieee_set_flag(ieee_inexact,.true.)
call TEST_TERMINAL_WL_4_1_SHARED(a,ls,b,ins,outs,actual,flagstate==1)
call ieee_get_flag(ieee_all,new_flags)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', 1,1,ls,ins,outs,trial,rounding
  stop 1
endif
if(any(old_flags.neqv.new_flags)) then
  write(*,*) 'FLAGS MISMATCH', 1,1,ls,ins,outs,trial,rounding
  write(*,*) old_flags,new_flags
  stop 2
endif
if(any(transfer(a,old_bits)/=saved_a).or.any(transfer(b,b_bits)/=saved_b)) stop 3
checks=checks+1

        enddo
      enddo
    enddo
  enddo
enddo
enddo
call ieee_set_rounding_mode(ieee_nearest)
if(checks/=12000) stop 4
if(fast==0.or.fallback==0) stop 8
write(*,*) 'PASS',checks,fast,fallback
end program
