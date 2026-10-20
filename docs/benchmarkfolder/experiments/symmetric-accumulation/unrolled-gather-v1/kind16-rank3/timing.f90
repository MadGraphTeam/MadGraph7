
program measure
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer i,j,k,pair,order,which,it,density
real(8) t0,t1,times(0:1)
do density=0,1
do k=1,4
do j=0,209
do i=1,4
 a(i,j,k)=cmplx(real(mod(i+3*j+7*k,13),kind=16)/7,&
              real(mod(11*i+j+k,17),kind=16)/11,kind=16)
 if(density==1.and.mod(i+j+k,3)/=0) a(i,j,k)=0
enddo
enddo
do j=0,14
do i=1,4
 b(i,j,k)=cmplx(real(i+j+k,kind=16)/3,real(2*i+j-k,kind=16)/7,kind=16)
enddo
enddo
enddo
b(1,1,3)=0
b(1,1,4)=0
b(1,2,3)=0
b(1,2,4)=0
b(1,3,3)=0
b(1,3,4)=0
b(1,4,3)=0
b(1,4,4)=0
b(2,1,3)=0
b(2,1,4)=0
b(2,2,3)=0
b(2,2,4)=0
b(2,3,3)=0
b(2,3,4)=0
b(2,4,3)=0
b(2,4,4)=0
b(3,1,1)=0
b(3,1,2)=0
b(3,2,1)=0
b(3,2,2)=0
b(3,3,1)=0
b(3,3,2)=0
b(3,4,1)=0
b(3,4,2)=0
b(4,1,1)=0
b(4,1,2)=0
b(4,2,1)=0
b(4,2,2)=0
b(4,3,1)=0
b(4,3,2)=0
b(4,4,1)=0
b(4,4,2)=0
do pair=1,7
 do order=0,1
  which=mod(order+pair,2)
  call cpu_time(t0)
  if(which==0) then
   do it=1,300
    call TEST_UPDATE_WL_3_1(a,4,b,4,4,out)
   enddo
  else
   do it=1,300
    call TEST_UPDATE_WL_3_1_GATHER(a,4,b,4,4,out)
   enddo
  endif
  call cpu_time(t1)
  times(which)=t1-t0
 enddo
 write(*,*) density,pair,times,real(sum(out),kind=8)
enddo
enddo
end program
