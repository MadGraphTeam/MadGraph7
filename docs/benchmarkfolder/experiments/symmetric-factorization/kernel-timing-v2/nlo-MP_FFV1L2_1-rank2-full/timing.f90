program exact_oracle_dump
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
integer, parameter :: NC=15, NO=35, MAXC=210, MAXB=15
complex*32 :: A(4,0:MAXC-1,4), B(4,0:MAXB-1,4)
complex*32 :: oldout(4,0:MAXC-1,4), newout(4,0:MAXC-1,4)
complex*32 :: WV(4), PMOM(0:3), COUP,COUP1,COUP2,CI
real*16 :: MASS
integer*8 :: saved_a(4*4*MAXC*4),saved_b(4*4*MAXB*4),one(2)
integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag,pair,order,which,it
complex*32 measured(4,0:MAXC-1,4)
integer*8 oldbits(4*4*MAXC*4),newbits(4*4*MAXC*4)
real*8 t0,t1,times(0:1),checksums(0:1)
common/FACTOR_TRACE/fast,fallback
logical :: entry,oldflags(size(ieee_all)),newflags(size(ieee_all)),samea,sameb
if(storage_size(1.0_16)/=128.or.digits(1.0_16)/=113.or.maxexponent(1.0_16)/=16384) stop 1
one=transfer(1.0_16,one)
hi=merge(1,2,one(1)/=0_8)
if(one(hi)/=ishft(16383_8,48).or.one(3-hi)/=0_8) stop 2
call ieee_set_rounding_mode(ieee_nearest)
do i=1,size(ieee_all)
  if(.not.ieee_support_halting(ieee_all(i))) stop 3
  call ieee_set_halting_mode(ieee_all(i),.false.)
enddo

open(73,file='initial-values.dat',status='replace')
do trial=1,2
  CI=cmplx(0.0_16,1.0_16,16)
  MASS=1.5_16
  COUP=cmplx(1.125_16+scale(1.0_16,-108),-0.1875_16,16)
  COUP1=cmplx(0.875_16,-0.3125_16+scale(1.0_16,-109),16)
  COUP2=cmplx(-0.625_16,0.4375_16,16)
  WV(1)=cmplx(1.0_16+scale(1.0_16,-109),2.0_16,16)
  WV(2)=cmplx(-0.75_16,0.625_16,16)
  WV(3)=cmplx(0.4375_16,-0.28125_16,16)
  WV(4)=cmplx(1.375_16,0.8125_16,16)
  PMOM(0)=cmplx(1.625_16,0.0625_16,16)
  PMOM(1)=cmplx(-0.875_16,0.03125_16,16)
  PMOM(2)=cmplx(0.3125_16,-0.125_16,16)
  PMOM(3)=cmplx(0.5625_16,0.1875_16,16)
  A=cmplx(9.125_16,-0.25_16,16)
  B=cmplx(-17.5_16,0.25_16,16)
B ( 1 , 0 , 1 ) = COUP * ( - 1Q0 ) * ( PMOM ( 0 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( &
 & PMOM ( 1 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 2 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) ) &
 & + PMOM ( 3 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) ) ) )
B ( 2 , 0 , 1 ) = COUP * ( PMOM ( 0 ) * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 &
 & ) * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( WV ( 1 ) - WV ( 4 ) ) + PMOM ( 3 &
 & ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 3 , 0 , 1 ) = COUP * MASS * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 4 , 0 , 1 ) = COUP * - MASS * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 1 , 1 , 1 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 2 , 1 , 1 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 3 , 1 , 1 ) = 0Q0
B ( 4 , 1 , 1 ) = 0Q0
B ( 1 , 2 , 1 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 2 , 2 , 1 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 3 , 2 , 1 ) = 0Q0
B ( 4 , 2 , 1 ) = 0Q0
B ( 1 , 3 , 1 ) = COUP * ( - 1Q0 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 2 , 3 , 1 ) = COUP * ( WV ( 1 ) - WV ( 4 ) )
B ( 3 , 3 , 1 ) = 0Q0
B ( 4 , 3 , 1 ) = 0Q0
B ( 1 , 4 , 1 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 2 , 4 , 1 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 3 , 4 , 1 ) = 0Q0
B ( 4 , 4 , 1 ) = 0Q0
B ( 1 , 0 , 2 ) = COUP * ( PMOM ( 0 ) * ( - CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 ) * ( + CI * &
 & ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( - 1Q0 ) * ( WV ( 1 ) + WV ( 4 ) ) + PMOM ( 3 ) * ( - CI &
 & * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 2 , 0 , 2 ) = COUP * ( - 1Q0 ) * ( PMOM ( 0 ) * ( - 1Q0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + &
 & ( PMOM ( 1 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 2 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) ) &
 & + PMOM ( 3 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) ) ) )
B ( 3 , 0 , 2 ) = COUP * - MASS * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 4 , 0 , 2 ) = COUP * MASS * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 1 , 1 , 2 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 2 , 1 , 2 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 3 , 1 , 2 ) = 0Q0
B ( 4 , 1 , 2 ) = 0Q0
B ( 1 , 2 , 2 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 2 , 2 , 2 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 3 , 2 , 2 ) = 0Q0
B ( 4 , 2 , 2 ) = 0Q0
B ( 1 , 3 , 2 ) = COUP * ( - WV ( 1 ) - WV ( 4 ) )
B ( 2 , 3 , 2 ) = COUP * ( - 1Q0 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 3 , 3 , 2 ) = 0Q0
B ( 4 , 3 , 2 ) = 0Q0
B ( 1 , 4 , 2 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 2 , 4 , 2 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 3 , 4 , 2 ) = 0Q0
B ( 4 , 4 , 2 ) = 0Q0
B ( 1 , 0 , 3 ) = COUP * MASS * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 2 , 0 , 3 ) = COUP * MASS * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 3 , 0 , 3 ) = COUP * ( - 1Q0 ) * ( PMOM ( 0 ) * ( - 1Q0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + &
 & ( PMOM ( 1 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 2 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) &
 & ) + PMOM ( 3 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) ) ) )
B ( 4 , 0 , 3 ) = COUP * ( PMOM ( 0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 ) * ( - 1Q0 &
 & ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( - 1Q0 ) * ( WV ( 1 ) + WV ( 4 ) ) + PMOM ( &
 & 3 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 1 , 1 , 3 ) = 0Q0
B ( 2 , 1 , 3 ) = 0Q0
B ( 3 , 1 , 3 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 4 , 1 , 3 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 1 , 2 , 3 ) = 0Q0
B ( 2 , 2 , 3 ) = 0Q0
B ( 3 , 2 , 3 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 4 , 2 , 3 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 1 , 3 , 3 ) = 0Q0
B ( 2 , 3 , 3 ) = 0Q0
B ( 3 , 3 , 3 ) = COUP * ( - 1Q0 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 4 , 3 , 3 ) = COUP * ( - WV ( 1 ) - WV ( 4 ) )
B ( 1 , 4 , 3 ) = 0Q0
B ( 2 , 4 , 3 ) = 0Q0
B ( 3 , 4 , 3 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 4 , 4 , 3 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 1 , 0 , 4 ) = COUP * MASS * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 2 , 0 , 4 ) = COUP * MASS * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 3 , 0 , 4 ) = COUP * ( PMOM ( 0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 1 ) * ( - CI * &
 & ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( WV ( 1 ) - WV ( 4 ) ) + PMOM ( 3 ) * ( - CI * &
 & ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 4 , 0 , 4 ) = COUP * ( - 1Q0 ) * ( PMOM ( 0 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( &
 & PMOM ( 1 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 2 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) ) + &
 & PMOM ( 3 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) ) ) )
B ( 1 , 1 , 4 ) = 0Q0
B ( 2 , 1 , 4 ) = 0Q0
B ( 3 , 1 , 4 ) = COUP * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 4 , 1 , 4 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 1 , 2 , 4 ) = 0Q0
B ( 2 , 2 , 4 ) = 0Q0
B ( 3 , 2 , 4 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 4 , 2 , 4 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 1 , 3 , 4 ) = 0Q0
B ( 2 , 3 , 4 ) = 0Q0
B ( 3 , 3 , 4 ) = COUP * ( WV ( 1 ) - WV ( 4 ) )
B ( 4 , 3 , 4 ) = COUP * ( - 1Q0 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 1 , 4 , 4 ) = 0Q0
B ( 2 , 4 , 4 ) = 0Q0
B ( 3 , 4 , 4 ) = COUP * ( - 1Q0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 4 , 4 , 4 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
  if(trial==9.or.trial==10) then
if(trial==9) then
  B(1,1,1)=cmplx(real(3,16)/8,0.0_16,16)
else
  B(1,1,1)=cmplx(0.0_16,real(1,16)/8,16)
endif
call alter_zero_sign(B(1,1,1),mod(2,2)==0)
if(trial==9) then
  B(1,1,2)=cmplx(real(4,16)/8,0.0_16,16)
else
  B(1,1,2)=cmplx(0.0_16,real(0,16)/8,16)
endif
call alter_zero_sign(B(1,1,2),mod(3,2)==0)
if(trial==9) then
  B(2,1,1)=cmplx(real(5,16)/8,0.0_16,16)
else
  B(2,1,1)=cmplx(0.0_16,real(3,16)/8,16)
endif
call alter_zero_sign(B(2,1,1),mod(3,2)==0)
if(trial==9) then
  B(2,1,2)=cmplx(real(6,16)/8,0.0_16,16)
else
  B(2,1,2)=cmplx(0.0_16,real(2,16)/8,16)
endif
call alter_zero_sign(B(2,1,2),mod(4,2)==0)
if(trial==9) then
  B(3,1,3)=cmplx(real(9,16)/8,0.0_16,16)
else
  B(3,1,3)=cmplx(0.0_16,real(3,16)/8,16)
endif
call alter_zero_sign(B(3,1,3),mod(6,2)==0)
if(trial==9) then
  B(3,1,4)=cmplx(real(10,16)/8,0.0_16,16)
else
  B(3,1,4)=cmplx(0.0_16,real(2,16)/8,16)
endif
call alter_zero_sign(B(3,1,4),mod(7,2)==0)
if(trial==9) then
  B(4,1,3)=cmplx(real(11,16)/8,0.0_16,16)
else
  B(4,1,3)=cmplx(0.0_16,real(5,16)/8,16)
endif
call alter_zero_sign(B(4,1,3),mod(7,2)==0)
if(trial==9) then
  B(4,1,4)=cmplx(real(12,16)/8,0.0_16,16)
else
  B(4,1,4)=cmplx(0.0_16,real(4,16)/8,16)
endif
call alter_zero_sign(B(4,1,4),mod(8,2)==0)
  endif
  if(trial/=7) then
B ( 1 , 2 , 1 ) = B ( 2 , 1 , 1 )
B ( 1 , 2 , 2 ) = B ( 2 , 1 , 2 )
B ( 1 , 2 , 3 ) = B ( 2 , 1 , 3 )
B ( 1 , 2 , 4 ) = B ( 2 , 1 , 4 )
B ( 2 , 2 , 1 ) = B ( 1 , 1 , 1 )
B ( 2 , 2 , 2 ) = B ( 1 , 1 , 2 )
B ( 2 , 2 , 3 ) = B ( 1 , 1 , 3 )
B ( 2 , 2 , 4 ) = B ( 1 , 1 , 4 )
B ( 3 , 2 , 1 ) = ( - B ( 4 , 1 , 1 ) )
B ( 3 , 2 , 2 ) = ( - B ( 4 , 1 , 2 ) )
B ( 3 , 2 , 3 ) = ( - B ( 4 , 1 , 3 ) )
B ( 3 , 2 , 4 ) = ( - B ( 4 , 1 , 4 ) )
B ( 4 , 2 , 1 ) = ( - B ( 3 , 1 , 1 ) )
B ( 4 , 2 , 2 ) = ( - B ( 3 , 1 , 2 ) )
B ( 4 , 2 , 3 ) = ( - B ( 3 , 1 , 3 ) )
B ( 4 , 2 , 4 ) = ( - B ( 3 , 1 , 4 ) )
B ( 1 , 3 , 1 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 1 ) ) , REAL ( B ( 2 , 1 , 1 ) , 16 ) , 16 )
B ( 1 , 3 , 2 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 2 ) ) , REAL ( B ( 2 , 1 , 2 ) , 16 ) , 16 )
B ( 1 , 3 , 3 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 3 ) ) , REAL ( B ( 2 , 1 , 3 ) , 16 ) , 16 )
B ( 1 , 3 , 4 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 4 ) ) , REAL ( B ( 2 , 1 , 4 ) , 16 ) , 16 )
B ( 2 , 3 , 1 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 1 ) ) , - REAL ( B ( 1 , 1 , 1 ) , 16 ) , 16 )
B ( 2 , 3 , 2 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 2 ) ) , - REAL ( B ( 1 , 1 , 2 ) , 16 ) , 16 )
B ( 2 , 3 , 3 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 3 ) ) , - REAL ( B ( 1 , 1 , 3 ) , 16 ) , 16 )
B ( 2 , 3 , 4 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 4 ) ) , - REAL ( B ( 1 , 1 , 4 ) , 16 ) , 16 )
B ( 3 , 3 , 1 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 1 ) ) , - REAL ( B ( 4 , 1 , 1 ) , 16 ) , 16 )
B ( 3 , 3 , 2 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 2 ) ) , - REAL ( B ( 4 , 1 , 2 ) , 16 ) , 16 )
B ( 3 , 3 , 3 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 3 ) ) , - REAL ( B ( 4 , 1 , 3 ) , 16 ) , 16 )
B ( 3 , 3 , 4 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 4 ) ) , - REAL ( B ( 4 , 1 , 4 ) , 16 ) , 16 )
B ( 4 , 3 , 1 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 1 ) ) , REAL ( B ( 3 , 1 , 1 ) , 16 ) , 16 )
B ( 4 , 3 , 2 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 2 ) ) , REAL ( B ( 3 , 1 , 2 ) , 16 ) , 16 )
B ( 4 , 3 , 3 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 3 ) ) , REAL ( B ( 3 , 1 , 3 ) , 16 ) , 16 )
B ( 4 , 3 , 4 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 4 ) ) , REAL ( B ( 3 , 1 , 4 ) , 16 ) , 16 )
B ( 1 , 4 , 1 ) = B ( 1 , 1 , 1 )
B ( 1 , 4 , 2 ) = B ( 1 , 1 , 2 )
B ( 1 , 4 , 3 ) = B ( 1 , 1 , 3 )
B ( 1 , 4 , 4 ) = B ( 1 , 1 , 4 )
B ( 2 , 4 , 1 ) = ( - B ( 2 , 1 , 1 ) )
B ( 2 , 4 , 2 ) = ( - B ( 2 , 1 , 2 ) )
B ( 2 , 4 , 3 ) = ( - B ( 2 , 1 , 3 ) )
B ( 2 , 4 , 4 ) = ( - B ( 2 , 1 , 4 ) )
B ( 3 , 4 , 1 ) = ( - B ( 3 , 1 , 1 ) )
B ( 3 , 4 , 2 ) = ( - B ( 3 , 1 , 2 ) )
B ( 3 , 4 , 3 ) = ( - B ( 3 , 1 , 3 ) )
B ( 3 , 4 , 4 ) = ( - B ( 3 , 1 , 4 ) )
B ( 4 , 4 , 1 ) = B ( 4 , 1 , 1 )
B ( 4 , 4 , 2 ) = B ( 4 , 1 , 2 )
B ( 4 , 4 , 3 ) = B ( 4 , 1 , 3 )
B ( 4 , 4 , 4 ) = B ( 4 , 1 , 4 )
  endif
  if(trial==9.or.trial==10) then
    do j=1,4
      do m=1,4
        do k=1,4
          call alter_zero_sign(B(j,m,k),mod(j+m+k,2)==0)
        enddo
      enddo
    enddo
  endif
  do i=1,4
    do l=0,NC-1
      do k=1,4
        A(k,l,i)=cmplx(real(mod(17*k+11*l+7*i,31)-15,16)/16, &
                      real(mod(13*k+5*l+19*i,29)-14,16)/32,16)
        A(k,l,i)=A(k,l,i)*(1.0_16+scale(1.0_16,-109))
        if(trial==2.and.mod(l+k+i,3)/=0) A(k,l,i)=cmplx(0.0_16,-0.0_16,16)
      enddo
    enddo
  enddo
  if(trial==3) then
A(:,0:NC-1,:)=cmplx(0.0_16,0.0_16,16)
do i=1,4
  do l=0,NC-1
    A(1,l,i)=B(1,1,2)*scale(1.0_16,mod(l+i,7)-3)
    A(2,l,i)=-B(1,1,1)*scale(1.0_16,mod(l+i,7)-3)
    A(2,l,i)=A(2,l,i)+cmplx(scale(real(mod(l,3)+1,16),-108),0.0_16,16)
  enddo
enddo
  endif
  if(trial==4.or.trial==5) then
    n=merge(900,-900,trial==4)
    A(:,0:NC-1,:)=A(:,0:NC-1,:)*scale(1.0_16,n)
    B(:,0:4,:)=B(:,0:4,:)*scale(1.0_16,n)
  endif
  if(trial==6) then
    do l=0,NC-1
      n=merge(900,-900,mod(l,2)==0)
      A(:,l,:)=A(:,l,:)*scale(1.0_16,n)
    enddo
  endif

  entry=.true.
  call ieee_set_flag(ieee_inexact,.true.)
  call TEST_UPDATE_WL_2_1(A,4,B,4,4,oldout)
  call TEST_UPDATE_WL_2_1_FACTOR(A,4,B,4,4,newout,entry)
  oldbits=transfer(oldout,oldbits)
  newbits=transfer(newout,newbits)
  write(73,*) 'TRIAL',trial
  do i=1,4
    do l=0,NC-1
      do k=1,4
        call dump_complex('A',k,l,i,A(k,l,i))
      enddo
    enddo
  enddo
  do k=1,4
    do m=0,4
      do j=1,4
        call dump_complex('B',j,m,k,B(j,m,k))
      enddo
    enddo
  enddo
  do i=1,4
    do n=0,NO-1
      do j=1,4
        call dump_complex('OLD',j,n,i,oldout(j,n,i))
        call dump_complex('NEW',j,n,i,newout(j,n,i))
      enddo
    enddo
  enddo
  do pair=1,7
    do order=0,1
      which=mod(order+pair,2)
      call ieee_set_flag(ieee_inexact,.true.)
      call cpu_time(t0)
      if(which==0) then
        do it=1,300
          call TEST_UPDATE_WL_2_1(A,4,B,4,4,measured)
        enddo
      else
        do it=1,300
          call TEST_UPDATE_WL_2_1_FACTOR(A,4,B,4,4,measured,entry)
        enddo
      endif
      call cpu_time(t1)
      times(which)=t1-t0
      if(which==0) then
        if(any(transfer(measured,oldbits)/=oldbits)) stop 10
      else
        if(any(transfer(measured,newbits)/=newbits)) stop 11
      endif
      checksums(which)=real(sum(measured),8)
    enddo
    write(*,*) trial,pair,times,checksums
  enddo
enddo
close(73)
contains
subroutine dump_complex(tag,x,y,z,value)
  character(*),intent(in)::tag
  integer,intent(in)::x,y,z
  complex*32,intent(in)::value
  integer*8::re(2),im(2)
  re=transfer(real(value,16),re)
  im=transfer(aimag(value),im)
  write(73,'(A,3(1X,I0),4(1X,Z16.16))') tag,x,y,z,re(hi),re(3-hi),im(hi),im(3-hi)
end subroutine
subroutine alter_zero_sign(value,negative)
  complex*32,intent(inout)::value
  logical,intent(in)::negative
  integer*8::re(2),im(2)
  re=transfer(real(value,16),re)
  im=transfer(aimag(value),im)
  if(real(value,16)==0) then
    re=0_8
    if(negative) re(hi)=ibset(re(hi),63)
  endif
  if(aimag(value)==0) then
    im=0_8
    if(.not.negative) im(hi)=ibset(im(hi),63)
  endif
  value=cmplx(transfer(re,0.0_16),transfer(im,0.0_16),16)
end subroutine
end program
