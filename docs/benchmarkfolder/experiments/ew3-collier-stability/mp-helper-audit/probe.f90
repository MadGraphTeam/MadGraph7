
program helper_precision
implicit none
complex(16),external :: MP_REGLOG,MP_REGLOGP,MP_REGLOGM,MP_REGSQRT,MP_GRREGLOG
complex(16),external :: FIX_MP_REGLOG,FIX_MP_REGLOGP,FIX_MP_REGLOGM,FIX_MP_REGSQRT,FIX_MP_GRREGLOG
complex(16) :: z,reference,old,new
real(16) :: pi,x
integer i
x=1.234567890123456789012345678901_16
pi=acos(-1.0_16)
do i=1,2
  z=cmplx(x,0.0_16,kind=16)
  if(i==2) z=cmplx(0.0_16,x,kind=16)
  reference=log(z)
  old=MP_REGLOG(z)
  new=FIX_MP_REGLOG(z)
  write(*,'(A,I1,2ES48.36E4)') 'REGLOG_',i,abs(old-reference),abs(new-reference)
  if(abs(new-reference)>1.e-31_16) stop 1
  reference=sqrt(z)
  old=MP_REGSQRT(z)
  new=FIX_MP_REGSQRT(z)
  write(*,'(A,I1,2ES48.36E4)') 'REGSQRT_',i,abs(old-reference),abs(new-reference)
  if(abs(new-reference)>1.e-31_16) stop 2
enddo
z=cmplx(-1.0_16,-1.0_16,kind=16)
reference=log(z)+cmplx(0.0_16,2*pi,kind=16)
old=MP_REGLOGP(z)
new=FIX_MP_REGLOGP(z)
write(*,'(A,2ES48.36E4)') 'REGLOGP_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 3
z=cmplx(-1.0_16,1.0_16,kind=16)
reference=log(z)-cmplx(0.0_16,2*pi,kind=16)
old=MP_REGLOGM(z)
new=FIX_MP_REGLOGM(z)
write(*,'(A,2ES48.36E4)') 'REGLOGM_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 4
old=MP_GRREGLOG(1.0_16,z,conjg(z))
new=FIX_MP_GRREGLOG(1.0_16,z,conjg(z))
write(*,'(A,2ES48.36E4)') 'GRREGLOG_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 5
end program
