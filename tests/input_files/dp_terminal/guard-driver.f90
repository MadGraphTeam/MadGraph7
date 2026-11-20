
program exact_sparse
use iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use ML5_0_POLYNOMIAL_CONSTANTS
implicit none
complex(8) :: a(4,0:14,4),b(4,0:4,4),expected(4,0:14,4),actual(4,0:14,4)
complex(8) :: sentinel
integer(int64) :: old_bits(4*4*15*2),new_bits(4*4*15*2),saved_a(4*4*15*2)
integer(int64) :: b_bits(4*4*5*2),saved_b(4*4*5*2)
real(8) :: scale,inf,nan,snan,small
logical :: old_flags(size(ieee_all)),new_flags(size(ieee_all))
type(ieee_round_type) :: modes(4)
integer :: i,j,k,ls,ins,outs,trial,rounding,checks,flagstate,fast_count,fallback_count,f
logical :: entry,took_fast
type(ieee_round_type) :: after_rounding
modes=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
sentinel=cmplx(-123.25_8,456.125_8,kind=8)
inf=ieee_value(0.0_8,ieee_positive_inf)
nan=ieee_value(0.0_8,ieee_quiet_nan)
snan=ieee_value(0.0_8,ieee_signaling_nan)
small=nearest(0.0_8,1.0_8)
checks=0
fast_count=0
fallback_count=0
do flagstate=0,2**size(ieee_all)-1
do rounding=1,4
  call ieee_set_rounding_mode(modes(rounding))
  do trial=1,24
    do k=1,4
      do j=0,14
        do i=1,4
          scale=1.0_8
          if(trial==3.and.mod(j,3)==0) scale=1.e200_8
          if(trial==3.and.mod(j,3)==1) scale=1.e-200_8
          a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind=8)/7, &
            real(mod(37*i+11*j+7*k,31)-15,kind=8)/13,kind=8)*scale
          if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==4) a(i,j,k)=0
          if(trial==5) a(i,j,k)=cmplx((-1.0_8)**i,(-1.0_8)**(i+1),kind=8)
          if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_8,0.0_8,kind=8)
          if(trial==7) a(i,j,k)=cmplx(0.0_8,-0.0_8,kind=8)
          if(trial==7.and.mod(i+j+k,5)==0) a(i,j,k)=cmplx(inf,nan,kind=8)
          if(trial==8.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==9) a(i,j,k)=cmplx(small,-small,kind=8)
          if(trial==10.and.mod(i+j+k,2)==0) a(i,j,k)=0
          if(trial==10.and.mod(i+j+k,2)==1) a(i,j,k)=cmplx(huge(scale),-huge(scale),kind=8)
          if(trial==11) a(i,j,k)=cmplx(snan,0.0_8,kind=8)
          if(trial==12) a(i,j,k)=0
          if(trial==13) a(i,j,k)=cmplx(2.0_8**400,-2.0_8**400,kind=8)
          if(trial==14) a(i,j,k)=cmplx(2.0_8**(-400),2.0_8**(-400),kind=8)
          if(trial==15) a(i,j,k)=cmplx(2.0_8**401,0.0_8,kind=8)
          if(trial==16) a(i,j,k)=cmplx(2.0_8**(-401),0.0_8,kind=8)
          if(trial==17) a(i,j,k)=cmplx(2.0_8**400,2.0_8**400,kind=8)
          if(trial==18) a(i,j,k)=cmplx(nearest(2.0_8**401,-1.0_8), &
            -nearest(2.0_8**401,-1.0_8),kind=8)
          if(trial==19) a(i,j,k)=cmplx(nearest(2.0_8**(-400),1.0_8),2.0_8**(-400),kind=8)
          if(trial==20) a(i,j,k)=cmplx(nearest(2.0_8**(-400),-1.0_8),0.0_8,kind=8)
          if(trial==21) a(i,j,k)=cmplx(-0.0_8,0.0_8,kind=8)
          if(trial==22.and.i==4) a(i,j,k)=cmplx(small,0.0_8,kind=8)
          if(trial==23.and.i==4) a(i,j,k)=cmplx(inf,0.0_8,kind=8)
          if(trial==24.and.i==4) a(i,j,k)=cmplx(snan,0.0_8,kind=8)
        enddo
      enddo
      do j=0,4
        do i=1,4
          b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind=8)/3, &
            real(mod(11*i+17*j+13*k,23)-11,kind=8)/11,kind=8)
          if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
          if(trial==4.or.trial==7) b(i,j,k)=0
          if(trial==5) b(i,j,k)=cmplx(1.0_8,-1.0_8,kind=8)
          if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_8,-0.0_8,kind=8)
          if(trial==8.and.mod(i+j+k,5)==0) b(i,j,k)=cmplx(inf,nan,kind=8)
          if(trial==9) b(i,j,k)=cmplx(0.25_8,-0.25_8,kind=8)
          if(trial==10.and.mod(i+j+k,2)==1) b(i,j,k)=0
          if(trial==11) b(i,j,k)=0
          if(trial==12) b(i,j,k)=cmplx(snan,0.0_8,kind=8)
          if(trial==13) b(i,j,k)=cmplx(2.0_8**400,2.0_8**400,kind=8)
          if(trial==14.or.trial==17) b(i,j,k)=cmplx(2.0_8**(-400),-2.0_8**(-400),kind=8)
          if(trial==15.or.trial==16.or.trial==20) b(i,j,k)=cmplx(1.0_8,0.0_8,kind=8)
          if(trial==18) b(i,j,k)=cmplx(nearest(2.0_8**401,-1.0_8), &
            nearest(2.0_8**401,-1.0_8),kind=8)
          if(trial==19) b(i,j,k)=cmplx(2.0_8**(-400),-2.0_8**(-400),kind=8)
        enddo
      enddo
    enddo
    saved_a=transfer(a,saved_a)
    saved_b=transfer(b,saved_b)
    do ls=0,4
      do ins=0,4
        do outs=0,4

expected=sentinel
actual=sentinel
call ieee_set_flag(ieee_all,.false.)
do f=1,size(ieee_all)
if(btest(flagstate,f-1)) call ieee_set_flag(ieee_all(f),.true.)
enddo
call DP_DENSE(a,ls,b,ins,outs,expected)
call ieee_get_flag(ieee_all,old_flags)
call ieee_set_flag(ieee_all,.false.)
do f=1,size(ieee_all)
if(btest(flagstate,f-1)) call ieee_set_flag(ieee_all(f),.true.)
enddo
call ieee_get_flag(ieee_inexact,entry)
call DP_TERMINAL(a,ls,b,ins,outs,actual,entry,took_fast)
call ieee_get_flag(ieee_all,new_flags)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(took_fast) then
  fast_count=fast_count+1
  if(.not.entry.or.rounding/=1.or.ls/=4.or.ins/=4.or.outs/=4) stop 9
  do i=1,4
    if(any(transfer(expected(i,:,i),old_bits,15*2)/= &
      transfer(actual(i,:,i),new_bits,15*2))) then
      write(*,*) 'CONSUMED BITS',trial,rounding,flagstate,i
      stop 6
    endif
  enddo
else
  fallback_count=fallback_count+1
endif
if(.not.took_fast.and.any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', 1,1,ls,ins,outs,trial,rounding
  stop 1
endif
if(any(old_flags.neqv.new_flags)) then
  write(*,*) 'FLAGS MISMATCH', 1,1,ls,ins,outs,trial,rounding
  write(*,*) old_flags,new_flags
  stop 2
endif
if(any(transfer(a,old_bits)/=saved_a).or.any(transfer(b,b_bits)/=saved_b)) stop 3
call ieee_get_rounding_mode(after_rounding)
if(after_rounding/=modes(rounding)) stop 8
checks=checks+1

        enddo
      enddo
    enddo
  enddo
enddo
enddo
call ieee_set_rounding_mode(ieee_nearest)
if(checks/=12000*2**size(ieee_all)) stop 4
if(fast_count==0.or.fallback_count==0) stop 7
write(*,*) 'PASS',checks,fast_count,fallback_count
end program
