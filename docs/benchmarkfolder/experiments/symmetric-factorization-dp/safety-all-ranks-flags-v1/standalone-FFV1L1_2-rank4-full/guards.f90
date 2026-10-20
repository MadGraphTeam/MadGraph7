program exact_oracle_dump
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
integer, parameter :: NC=70, NO=126, MAXC=210, MAXB=15
complex*16 :: A(4,0:MAXC-1,4), B(4,0:MAXB-1,4)
complex*16 :: oldout(4,0:MAXC-1,4), newout(4,0:MAXC-1,4)
complex*16 :: WV(4), PMOM(0:3), COUP,COUP1,COUP2,CI
real*8 :: MASS
integer*8 :: saved_a(2*4*MAXC*4),saved_b(2*4*MAXB*4),one
integer :: trial,i,j,k,l,m,n,fast,fallback,diag,test,ins,outs,mask,flag
integer*8 oldbits(2*4*MAXC*4),newbits(2*4*MAXC*4)
logical initialflags(size(ieee_all)),halting(size(ieee_all))
type(ieee_round_type) observed_rounding,initial_rounding
common/FACTOR_TRACE/fast,fallback
logical :: entry,oldflags(size(ieee_all)),newflags(size(ieee_all)),samea,sameb
if(storage_size(1.0_8)/=64.or.digits(1.0_8)/=53.or.maxexponent(1.0_8)/=1024) stop 1
one=transfer(1.0_8,one)
if(one/=ishft(1023_8,52)) stop 2
call ieee_set_rounding_mode(ieee_nearest)
do i=1,size(ieee_all)
  if(.not.ieee_support_halting(ieee_all(i))) stop 3
  call ieee_set_halting_mode(ieee_all(i),.false.)
enddo

do test=1,20
  trial=1
  call ieee_set_rounding_mode(ieee_nearest)
  CI=cmplx(0.0_8,1.0_8,8)
  MASS=1.5_8
  COUP=cmplx(1.125_8+scale(1.0_8,-48),-0.1875_8,8)
  COUP1=cmplx(0.875_8,-0.3125_8+scale(1.0_8,-49),8)
  COUP2=cmplx(-0.625_8,0.4375_8,8)
  WV(1)=cmplx(1.0_8+scale(1.0_8,-49),2.0_8,8)
  WV(2)=cmplx(-0.75_8,0.625_8,8)
  WV(3)=cmplx(0.4375_8,-0.28125_8,8)
  WV(4)=cmplx(1.375_8,0.8125_8,8)
  PMOM(0)=cmplx(1.625_8,0.0625_8,8)
  PMOM(1)=cmplx(-0.875_8,0.03125_8,8)
  PMOM(2)=cmplx(0.3125_8,-0.125_8,8)
  PMOM(3)=cmplx(0.5625_8,0.1875_8,8)
  A=cmplx(9.125_8,-0.25_8,8)
  B=cmplx(-17.5_8,0.25_8,8)
B ( 1 , 0 , 1 ) = COUP * ( PMOM ( 0 ) * ( - 1D0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 1 &
 & ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 2 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) ) + PMOM ( 3 &
 & ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) ) ) )
B ( 2 , 0 , 1 ) = COUP * ( PMOM ( 0 ) * ( - CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 ) * ( + CI * &
 & ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( - 1D0 ) * ( WV ( 1 ) + WV ( 4 ) ) + PMOM ( 3 ) * ( - CI &
 & * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 3 , 0 , 1 ) = COUP * MASS * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 4 , 0 , 1 ) = COUP * MASS * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 1 , 1 , 1 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 2 , 1 , 1 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 3 , 1 , 1 ) = 0D0
B ( 4 , 1 , 1 ) = 0D0
B ( 1 , 2 , 1 ) = COUP * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 2 , 2 , 1 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 3 , 2 , 1 ) = 0D0
B ( 4 , 2 , 1 ) = 0D0
B ( 1 , 3 , 1 ) = COUP * ( WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 2 , 3 , 1 ) = COUP * ( - WV ( 1 ) - WV ( 4 ) )
B ( 3 , 3 , 1 ) = 0D0
B ( 4 , 3 , 1 ) = 0D0
B ( 1 , 4 , 1 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 2 , 4 , 1 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 3 , 4 , 1 ) = 0D0
B ( 4 , 4 , 1 ) = 0D0
B ( 1 , 0 , 2 ) = COUP * ( PMOM ( 0 ) * ( - 1D0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 &
 & ) * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( WV ( 1 ) - WV ( 4 ) ) + PMOM ( 3 &
 & ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 2 , 0 , 2 ) = COUP * ( PMOM ( 0 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( PMOM ( 1 ) * &
 & ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 2 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) ) + PMOM ( 3 ) &
 & * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) ) ) )
B ( 3 , 0 , 2 ) = COUP * MASS * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 4 , 0 , 2 ) = COUP * MASS * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 1 , 1 , 2 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 2 , 1 , 2 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 3 , 1 , 2 ) = 0D0
B ( 4 , 1 , 2 ) = 0D0
B ( 1 , 2 , 2 ) = COUP * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 2 , 2 , 2 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 3 , 2 , 2 ) = 0D0
B ( 4 , 2 , 2 ) = 0D0
B ( 1 , 3 , 2 ) = COUP * ( WV ( 1 ) - WV ( 4 ) )
B ( 2 , 3 , 2 ) = COUP * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 3 , 3 , 2 ) = 0D0
B ( 4 , 3 , 2 ) = 0D0
B ( 1 , 4 , 2 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 2 , 4 , 2 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 3 , 4 , 2 ) = 0D0
B ( 4 , 4 , 2 ) = 0D0
B ( 1 , 0 , 3 ) = COUP * MASS * ( + CI * ( WV ( 1 ) ) - CI * ( WV ( 4 ) ) )
B ( 2 , 0 , 3 ) = COUP * - MASS * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 3 , 0 , 3 ) = COUP * ( PMOM ( 0 ) * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( PMOM ( 1 ) * &
 & ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 2 ) * ( WV ( 2 ) + CI * ( WV ( 3 ) ) ) + PMOM ( 3 ) * &
 & ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) ) ) )
B ( 4 , 0 , 3 ) = COUP * ( PMOM ( 0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) ) + ( PMOM ( 1 ) * ( - CI * &
 & ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( WV ( 1 ) - WV ( 4 ) ) + PMOM ( 3 ) * ( - CI * &
 & ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 1 , 1 , 3 ) = 0D0
B ( 2 , 1 , 3 ) = 0D0
B ( 3 , 1 , 3 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 4 , 1 , 3 ) = COUP * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 1 , 2 , 3 ) = 0D0
B ( 2 , 2 , 3 ) = 0D0
B ( 3 , 2 , 3 ) = COUP * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 4 , 2 , 3 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 1 , 3 , 3 ) = 0D0
B ( 2 , 3 , 3 ) = 0D0
B ( 3 , 3 , 3 ) = COUP * ( WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 4 , 3 , 3 ) = COUP * ( WV ( 1 ) - WV ( 4 ) )
B ( 1 , 4 , 3 ) = 0D0
B ( 2 , 4 , 3 ) = 0D0
B ( 3 , 4 , 3 ) = COUP * ( - CI * ( WV ( 1 ) ) + CI * ( WV ( 4 ) ) )
B ( 4 , 4 , 3 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 2 ) ) - WV ( 3 ) )
B ( 1 , 0 , 4 ) = COUP * - MASS * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 2 , 0 , 4 ) = COUP * MASS * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 3 , 0 , 4 ) = COUP * ( PMOM ( 0 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 1 ) * ( - 1D0 &
 & ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 2 ) * ( - 1D0 ) * ( WV ( 1 ) + WV ( 4 ) ) + PMOM ( &
 & 3 ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) ) ) )
B ( 4 , 0 , 4 ) = COUP * ( PMOM ( 0 ) * ( - 1D0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) + ( PMOM ( 1 &
 & ) * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) ) + ( PMOM ( 2 ) * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) ) + PMOM ( &
 & 3 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) ) ) ) )
B ( 1 , 1 , 4 ) = 0D0
B ( 2 , 1 , 4 ) = 0D0
B ( 3 , 1 , 4 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 4 , 1 , 4 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 1 , 2 , 4 ) = 0D0
B ( 2 , 2 , 4 ) = 0D0
B ( 3 , 2 , 4 ) = COUP * ( - 1D0 ) * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
B ( 4 , 2 , 4 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 1 , 3 , 4 ) = 0D0
B ( 2 , 3 , 4 ) = 0D0
B ( 3 , 3 , 4 ) = COUP * ( - WV ( 1 ) - WV ( 4 ) )
B ( 4 , 3 , 4 ) = COUP * ( - WV ( 2 ) + CI * ( WV ( 3 ) ) )
B ( 1 , 4 , 4 ) = 0D0
B ( 2 , 4 , 4 ) = 0D0
B ( 3 , 4 , 4 ) = COUP * ( + CI * ( WV ( 2 ) ) + WV ( 3 ) )
B ( 4 , 4 , 4 ) = COUP * ( + CI * ( WV ( 1 ) + WV ( 4 ) ) )
  if(trial==9.or.trial==10) then
if(trial==9) then
  B(1,1,1)=cmplx(real(3,8)/8,0.0_8,8)
else
  B(1,1,1)=cmplx(0.0_8,real(1,8)/8,8)
endif
call alter_zero_sign(B(1,1,1),mod(2,2)==0)
if(trial==9) then
  B(1,1,2)=cmplx(real(4,8)/8,0.0_8,8)
else
  B(1,1,2)=cmplx(0.0_8,real(0,8)/8,8)
endif
call alter_zero_sign(B(1,1,2),mod(3,2)==0)
if(trial==9) then
  B(2,1,1)=cmplx(real(5,8)/8,0.0_8,8)
else
  B(2,1,1)=cmplx(0.0_8,real(3,8)/8,8)
endif
call alter_zero_sign(B(2,1,1),mod(3,2)==0)
if(trial==9) then
  B(2,1,2)=cmplx(real(6,8)/8,0.0_8,8)
else
  B(2,1,2)=cmplx(0.0_8,real(2,8)/8,8)
endif
call alter_zero_sign(B(2,1,2),mod(4,2)==0)
if(trial==9) then
  B(3,1,3)=cmplx(real(9,8)/8,0.0_8,8)
else
  B(3,1,3)=cmplx(0.0_8,real(3,8)/8,8)
endif
call alter_zero_sign(B(3,1,3),mod(6,2)==0)
if(trial==9) then
  B(3,1,4)=cmplx(real(10,8)/8,0.0_8,8)
else
  B(3,1,4)=cmplx(0.0_8,real(2,8)/8,8)
endif
call alter_zero_sign(B(3,1,4),mod(7,2)==0)
if(trial==9) then
  B(4,1,3)=cmplx(real(11,8)/8,0.0_8,8)
else
  B(4,1,3)=cmplx(0.0_8,real(5,8)/8,8)
endif
call alter_zero_sign(B(4,1,3),mod(7,2)==0)
if(trial==9) then
  B(4,1,4)=cmplx(real(12,8)/8,0.0_8,8)
else
  B(4,1,4)=cmplx(0.0_8,real(4,8)/8,8)
endif
call alter_zero_sign(B(4,1,4),mod(8,2)==0)
  endif
  if(trial/=7) then
B ( 1 , 2 , 1 ) = ( - B ( 2 , 1 , 1 ) )
B ( 1 , 2 , 2 ) = ( - B ( 2 , 1 , 2 ) )
B ( 1 , 2 , 3 ) = ( - B ( 2 , 1 , 3 ) )
B ( 1 , 2 , 4 ) = ( - B ( 2 , 1 , 4 ) )
B ( 2 , 2 , 1 ) = ( - B ( 1 , 1 , 1 ) )
B ( 2 , 2 , 2 ) = ( - B ( 1 , 1 , 2 ) )
B ( 2 , 2 , 3 ) = ( - B ( 1 , 1 , 3 ) )
B ( 2 , 2 , 4 ) = ( - B ( 1 , 1 , 4 ) )
B ( 3 , 2 , 1 ) = B ( 4 , 1 , 1 )
B ( 3 , 2 , 2 ) = B ( 4 , 1 , 2 )
B ( 3 , 2 , 3 ) = B ( 4 , 1 , 3 )
B ( 3 , 2 , 4 ) = B ( 4 , 1 , 4 )
B ( 4 , 2 , 1 ) = B ( 3 , 1 , 1 )
B ( 4 , 2 , 2 ) = B ( 3 , 1 , 2 )
B ( 4 , 2 , 3 ) = B ( 3 , 1 , 3 )
B ( 4 , 2 , 4 ) = B ( 3 , 1 , 4 )
B ( 1 , 3 , 1 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 1 ) ) , REAL ( B ( 2 , 1 , 1 ) , 8 ) , 8 )
B ( 1 , 3 , 2 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 2 ) ) , REAL ( B ( 2 , 1 , 2 ) , 8 ) , 8 )
B ( 1 , 3 , 3 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 3 ) ) , REAL ( B ( 2 , 1 , 3 ) , 8 ) , 8 )
B ( 1 , 3 , 4 ) = CMPLX ( - AIMAG ( B ( 2 , 1 , 4 ) ) , REAL ( B ( 2 , 1 , 4 ) , 8 ) , 8 )
B ( 2 , 3 , 1 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 1 ) ) , - REAL ( B ( 1 , 1 , 1 ) , 8 ) , 8 )
B ( 2 , 3 , 2 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 2 ) ) , - REAL ( B ( 1 , 1 , 2 ) , 8 ) , 8 )
B ( 2 , 3 , 3 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 3 ) ) , - REAL ( B ( 1 , 1 , 3 ) , 8 ) , 8 )
B ( 2 , 3 , 4 ) = CMPLX ( AIMAG ( B ( 1 , 1 , 4 ) ) , - REAL ( B ( 1 , 1 , 4 ) , 8 ) , 8 )
B ( 3 , 3 , 1 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 1 ) ) , - REAL ( B ( 4 , 1 , 1 ) , 8 ) , 8 )
B ( 3 , 3 , 2 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 2 ) ) , - REAL ( B ( 4 , 1 , 2 ) , 8 ) , 8 )
B ( 3 , 3 , 3 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 3 ) ) , - REAL ( B ( 4 , 1 , 3 ) , 8 ) , 8 )
B ( 3 , 3 , 4 ) = CMPLX ( AIMAG ( B ( 4 , 1 , 4 ) ) , - REAL ( B ( 4 , 1 , 4 ) , 8 ) , 8 )
B ( 4 , 3 , 1 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 1 ) ) , REAL ( B ( 3 , 1 , 1 ) , 8 ) , 8 )
B ( 4 , 3 , 2 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 2 ) ) , REAL ( B ( 3 , 1 , 2 ) , 8 ) , 8 )
B ( 4 , 3 , 3 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 3 ) ) , REAL ( B ( 3 , 1 , 3 ) , 8 ) , 8 )
B ( 4 , 3 , 4 ) = CMPLX ( - AIMAG ( B ( 3 , 1 , 4 ) ) , REAL ( B ( 3 , 1 , 4 ) , 8 ) , 8 )
B ( 1 , 4 , 1 ) = ( - B ( 1 , 1 , 1 ) )
B ( 1 , 4 , 2 ) = ( - B ( 1 , 1 , 2 ) )
B ( 1 , 4 , 3 ) = ( - B ( 1 , 1 , 3 ) )
B ( 1 , 4 , 4 ) = ( - B ( 1 , 1 , 4 ) )
B ( 2 , 4 , 1 ) = B ( 2 , 1 , 1 )
B ( 2 , 4 , 2 ) = B ( 2 , 1 , 2 )
B ( 2 , 4 , 3 ) = B ( 2 , 1 , 3 )
B ( 2 , 4 , 4 ) = B ( 2 , 1 , 4 )
B ( 3 , 4 , 1 ) = B ( 3 , 1 , 1 )
B ( 3 , 4 , 2 ) = B ( 3 , 1 , 2 )
B ( 3 , 4 , 3 ) = B ( 3 , 1 , 3 )
B ( 3 , 4 , 4 ) = B ( 3 , 1 , 4 )
B ( 4 , 4 , 1 ) = ( - B ( 4 , 1 , 1 ) )
B ( 4 , 4 , 2 ) = ( - B ( 4 , 1 , 2 ) )
B ( 4 , 4 , 3 ) = ( - B ( 4 , 1 , 3 ) )
B ( 4 , 4 , 4 ) = ( - B ( 4 , 1 , 4 ) )
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
        A(k,l,i)=cmplx(real(mod(17*k+11*l+7*i,31)-15,8)/16, &
                      real(mod(13*k+5*l+19*i,29)-14,8)/32,8)
        A(k,l,i)=A(k,l,i)*(1.0_8+scale(1.0_8,-49))
        if(trial==2.and.mod(l+k+i,3)/=0) A(k,l,i)=cmplx(0.0_8,-0.0_8,8)
      enddo
    enddo
  enddo
  if(trial==3) then
A(:,0:NC-1,:)=cmplx(0.0_8,0.0_8,8)
do i=1,4
  do l=0,NC-1
    A(1,l,i)=B(1,1,2)*scale(1.0_8,mod(l+i,7)-3)
    A(2,l,i)=-B(1,1,1)*scale(1.0_8,mod(l+i,7)-3)
    A(2,l,i)=A(2,l,i)+cmplx(scale(real(mod(l,3)+1,8),-48),0.0_8,8)
  enddo
enddo
  endif
  if(trial==4.or.trial==5) then
    n=merge(180,-180,trial==4)
    A(:,0:NC-1,:)=A(:,0:NC-1,:)*scale(1.0_8,n)
    B(:,0:4,:)=B(:,0:4,:)*scale(1.0_8,n)
  endif
  if(trial==6) then
    do l=0,NC-1
      n=merge(180,-180,mod(l,2)==0)
      A(:,l,:)=A(:,l,:)*scale(1.0_8,n)
    enddo
  endif

  ins=4
  outs=4
  entry=.true.
  select case(test)
  case(1)
    call ieee_set_rounding_mode(ieee_up)
  case(2)
    call ieee_set_rounding_mode(ieee_down)
  case(3)
    call ieee_set_rounding_mode(ieee_to_zero)
  case(4)
    A(1,0,1)=cmplx(scale(1.0_8,201),0.0_8,8)
  case(5)
    A(1,0,1)=cmplx(scale(1.0_8,-201),0.0_8,8)
  case(6)
    A(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
  case(7)
    A(1,0,1)=cmplx(ieee_value(0.0_8,ieee_positive_inf),0.0_8,8)
  case(8)
    A(1,0,1)=cmplx(ieee_value(0.0_8,ieee_quiet_nan),0.0_8,8)
  case(9)
    A(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),A(1,0,1))
  case(10)
    B(1,0,1)=cmplx(scale(1.0_8,201),0.0_8,8)
  case(11)
    B(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
  case(12)
    B(1,0,1)=cmplx(ieee_value(0.0_8,ieee_positive_inf),0.0_8,8)
  case(13)
    B(1,0,1)=cmplx(ieee_value(0.0_8,ieee_quiet_nan),0.0_8,8)
  case(14)
    B(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),B(1,0,1))
  case(15)
    B(1,2,1)=B(1,2,1)+cmplx(1.0_8,0.0_8,8)
  case(16)
    ins=3
  case(17)
    outs=3
  case(18)
    ins=0
  case(19)
    outs=0
  case(20)
    B(1,0,1)=cmplx(scale(1.0_8,-201),0.0_8,8)
  end select
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call ieee_get_flag(ieee_all,initialflags)

  saved_a=transfer(A,saved_a)
  saved_b=transfer(B,saved_b)
  call ieee_get_rounding_mode(initial_rounding)
  call ieee_set_flag(ieee_all,initialflags)
  call TEST_UPDATE_WL_4_1(A,4,B,ins,outs,oldout)
  call ieee_get_flag(ieee_all,oldflags)
  if(any(saved_a/=transfer(A,saved_a)).or.any(saved_b/=transfer(B,saved_b))) stop 13
  call ieee_get_rounding_mode(observed_rounding)
  if(observed_rounding/=initial_rounding) stop 14
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,initialflags)
  call TEST_UPDATE_WL_4_1_FACTOR(A,4,B,ins,outs,newout,entry)
  call ieee_get_flag(ieee_all,newflags)
  if(any(oldflags.neqv.newflags)) then
    write(*,*) 'FLAGS',test,oldflags,newflags
    stop 11
  endif
  if(any(initialflags.and..not.newflags)) stop 15
  call ieee_get_rounding_mode(observed_rounding)
  if(observed_rounding/=initial_rounding) stop 16
  call ieee_get_halting_mode(ieee_all,halting)
  if(any(halting)) stop 17
  if(any(saved_a/=transfer(A,saved_a)).or.any(saved_b/=transfer(B,saved_b))) stop 18
  if(fast/=0.or.fallback/=1) stop 12
  if(fallback==1) then
    oldbits=transfer(oldout,oldbits)
    newbits=transfer(newout,newbits)
    if(any(oldbits/=newbits)) then
      write(*,*) 'BITS',test
      stop 10
    endif
  endif
  write(*,*) 'PASS',test,fast,fallback

enddo
call ieee_set_rounding_mode(ieee_nearest)
contains
subroutine dump_complex(tag,x,y,z,value)
  character(*),intent(in)::tag
  integer,intent(in)::x,y,z
  complex*16,intent(in)::value
  integer*8::re,im
  re=transfer(real(value,8),re)
  im=transfer(aimag(value),im)
  write(*,'(A,3(1X,I0),2(1X,Z16.16))') tag,x,y,z,re,im
end subroutine
subroutine alter_zero_sign(value,negative)
  complex*16,intent(inout)::value
  logical,intent(in)::negative
  integer*8::re,im
  re=transfer(real(value,8),re)
  im=transfer(aimag(value),im)
  if(real(value,8)==0) then
    re=0_8
    if(negative) re=ibset(re,63)
  endif
  if(aimag(value)==0) then
    im=0_8
    if(.not.negative) im=ibset(im,63)
  endif
  value=cmplx(transfer(re,0.0_8),transfer(im,0.0_8),8)
end subroutine
end program
