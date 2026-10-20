program kernel
use,intrinsic::ieee_arithmetic
use,intrinsic::ieee_exceptions
implicit none
complex*16 A(4,0:209,4),B(4,0:14,4),out(4,0:209,4),reference(4,0:209,4,0:1)
integer*8 bits(2),abits(2*4*210*4),bbits(2*4*15*4),outbits(2*4*210*4),refbits(2*4*210*4)
integer*8 wall0,wall1,rate
integer arm,it,pair,position,j,l,i,ios,expected,zero_changes
integer,parameter::schedule(4,8)=reshape((/0,1,2,3,3,2,1,0,1,2,3,0,0,3,2,1,2,3,0,1,1,0,3,2,3,0,1,2,2,1,0,3/),(/4,8/))
real*8 t0,t1,cpu,wall,checksum
logical flags(size(ieee_all)),baseline_flags(size(ieee_all))
character(8) tag
character(16) hex1,hex2
character(4096) filename
call get_command_argument(1,filename)
if(storage_size(1.0_8)/=64.or.digits(1.0_8)/=53) stop 1
call ieee_set_rounding_mode(ieee_nearest)
do i=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(i))) stop 2
 call ieee_set_halting_mode(ieee_all(i),.false.)
enddo
A=0
B=0
reference=0
open(77,file=trim(filename),status='old',action='read')
do
 read(77,*,iostat=ios) tag,j,l,i,hex1,hex2
 if(ios<0) exit
 if(ios/=0) stop 3
 read(hex1,'(Z16)') bits(1)
 read(hex2,'(Z16)') bits(2)
 select case(trim(tag))
 case('A')
  A(j,l,i)=transfer(bits,A(j,l,i))
 case('B')
  B(j,l,i)=transfer(bits,B(j,l,i))
 case('OLD')
  reference(j,l,i,0)=transfer(bits,out(j,l,i))
 case('NEW')
  reference(j,l,i,1)=transfer(bits,out(j,l,i))
 case default
  stop 4
 end select
enddo
close(77)
abits=transfer(A,abits)
bbits=transfer(B,bbits)
call system_clock(count_rate=rate)
do arm=0,3
 call ieee_set_flag(ieee_all,.false.)
 call ieee_set_flag(ieee_inexact,.true.)
 select case(arm)
case(0)
 call TEST_UPDATE_WL_4_1(A,4,B,4,4,out)
case(1)
 call TEST_UPDATE_WL_4_1_GUARDED(A,4,B,4,4,out,.true.)
case(2)
 call TEST_UPDATE_WL_4_1_FACTOR(A,4,B,4,4,out,.true.)
case(3)
 call TEST_UPDATE_WL_4_1_AZERO(A,4,B,4,4,out,.true.)
end select
 call ieee_get_flag(ieee_all,flags)
 if(arm==0) baseline_flags=flags
 if(any(flags.neqv.baseline_flags)) stop 5
 call verify_output()
 write(*,*) 'VALIDATED',arm,zero_changes
enddo
do arm=0,3
 do it=1,100
  select case(arm)
case(0)
 call TEST_UPDATE_WL_4_1(A,4,B,4,4,out)
case(1)
 call TEST_UPDATE_WL_4_1_GUARDED(A,4,B,4,4,out,.true.)
case(2)
 call TEST_UPDATE_WL_4_1_FACTOR(A,4,B,4,4,out,.true.)
case(3)
 call TEST_UPDATE_WL_4_1_AZERO(A,4,B,4,4,out,.true.)
end select
 enddo
enddo
do pair=1,21
 do position=1,4
  arm=schedule(position,mod(pair-1,8)+1)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call system_clock(wall0)
  call cpu_time(t0)
  do it=1,5000
   select case(arm)
case(0)
 call TEST_UPDATE_WL_4_1(A,4,B,4,4,out)
case(1)
 call TEST_UPDATE_WL_4_1_GUARDED(A,4,B,4,4,out,.true.)
case(2)
 call TEST_UPDATE_WL_4_1_FACTOR(A,4,B,4,4,out,.true.)
case(3)
 call TEST_UPDATE_WL_4_1_AZERO(A,4,B,4,4,out,.true.)
end select
  enddo
  call cpu_time(t1)
  call system_clock(wall1)
  cpu=t1-t0
  wall=real(wall1-wall0,8)/real(rate,8)
  call ieee_get_flag(ieee_all,flags)
  if(any(flags.neqv.baseline_flags)) stop 6
  call verify_output()
  checksum=real(sum(out),8)
  write(*,'(A,3(1X,I0),3(1X,ES25.17))') 'SAMPLE',pair,position,arm,cpu,wall,checksum
 enddo
enddo
write(*,*) 'PASS', 21, 5000
contains
subroutine verify_output()
 expected=merge(1,0,arm==2)
 outbits=transfer(out,outbits)
 refbits=transfer(reference(:,:,:,expected),refbits)
 zero_changes=count(outbits/=refbits)
 if(arm/=3.and.zero_changes/=0) stop 7
 if(arm==3.and.any(out/=reference(:,:,:,expected))) stop 8
 if(any(abits/=transfer(A,abits)).or.any(bbits/=transfer(B,bbits))) stop 9
end subroutine
end program
