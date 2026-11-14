subroutine DP_DENSE(a,ls,b,ins,outs,out)
use TEST_POLYNOMIAL_CONSTANTS
implicit none
integer :: ls,ins,outs,j,m,k,l,i,pos
complex(8) :: a(4,0:209,4),b(4,0:4,4),out(4,0:209,4),u
out=cmplx(0.0_8,0.0_8,8)
do j=1,outs
 do m=0,4
  do k=1,ins
   u=b(j,m,k)
   if(u==cmplx(0.0_8,0.0_8,8)) cycle
   do l=0,14
    pos=COMB_COEF_POS(l,m)
    do i=1,ls
     out(j,pos,i)=out(j,pos,i)+a(k,l,i)*u
    enddo
   enddo
  enddo
 enddo
enddo
end subroutine
