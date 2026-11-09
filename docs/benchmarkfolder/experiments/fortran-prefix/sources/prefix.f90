subroutine prefix_dp_old(n,p,output)
implicit none
integer n,i,j,k
real*8 p(4,n),output(4,n-1),acc
output=0.0_8
do i=1,4
do j=1,n-1
do k=1,j
output(i,j)=output(i,j)+p(i,k)
enddo
enddo
enddo
end

subroutine prefix_dp_new(n,p,output)
implicit none
integer n,i,j,k
real*8 p(4,n),output(4,n-1),acc
output=0.0_8
do i=1,4
acc=0.0_8
do j=1,n-1
acc=acc+p(i,j)
output(i,j)=acc
enddo
enddo
end

subroutine prefix_qp_old(n,p,output)
implicit none
integer n,i,j,k
real*16 p(4,n),output(4,n-1),acc
output=0.0_16
do i=1,4
do j=1,n-1
do k=1,j
output(i,j)=output(i,j)+p(i,k)
enddo
enddo
enddo
end

subroutine prefix_qp_new(n,p,output)
implicit none
integer n,i,j,k
real*16 p(4,n),output(4,n-1),acc
output=0.0_16
do i=1,4
acc=0.0_16
do j=1,n-1
acc=acc+p(i,j)
output(i,j)=acc
enddo
enddo
end
