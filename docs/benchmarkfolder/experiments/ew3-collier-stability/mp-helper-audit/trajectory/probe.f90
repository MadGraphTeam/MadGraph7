
program trajectory_precision
implicit none
complex(16), external :: MP_SQRT_TRAJECTORY, FIX_MP_SQRT_TRAJECTORY
complex(16), external :: MP_LOG_TRAJECTORY, FIX_MP_LOG_TRAJECTORY
complex(16) :: p2, m2, reference, old, new, discriminant
real(16) :: s, x
integer i,n
m2=cmplx(1.0_16,0.0_16,kind=16)
do i=-1,1,2
  s=4.0_16+i*2.0_16**(-30)
  p2=cmplx(s,-1.0_16,kind=16)
  discriminant=1.0_16-4.0_16*(m2-cmplx(0.0_16,1.e-24_16,kind=16))/p2
  ! Below threshold, continuation crosses the negative real discriminant
  ! axis from above to below: the continued root is minus principal sqrt.
  reference=sqrt(discriminant)
  if(i<0) reference=-reference
  do n=100,400,100
    old=MP_SQRT_TRAJECTORY(n,p2,m2,m2)
    new=FIX_MP_SQRT_TRAJECTORY(n,p2,m2,m2)
    write(*,'(A,2I5,6ES48.36E4)') 'SQRT',i,n,real(old),aimag(old), &
      real(new),aimag(new),abs(old-reference),abs(new-reference)
    if(abs(new-reference)>1.e-31_16) stop 1
    old=MP_LOG_TRAJECTORY(n,p2,m2,m2)
    new=FIX_MP_LOG_TRAJECTORY(n,p2,m2,m2)
    write(*,'(A,2I5,5ES48.36E4)') 'LOG',i,n,real(old),aimag(old), &
      real(new),aimag(new),abs(old-new)
  enddo
enddo
x=1.0_16+2.0_16**(-100)
write(*,'(A,2ES48.36E4)') 'REAL_CAST',real(real(x),16)-x,real(x,16)-x
if(real(real(x),16)==x) stop 2
end program
