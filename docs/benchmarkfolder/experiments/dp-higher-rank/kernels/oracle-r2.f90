subroutine DP_DENSE(a,ls,b,ins,outs,out)
use TEST_POLYNOMIAL_CONSTANTS
implicit none
integer :: ls,ins,outs
complex(8) :: a(4,0:209,4),b(4,0:4,4),out(4,0:209,4)
call TEST_UPDATE_WL_2_1(a,ls,b,ins,outs,out)
end subroutine
