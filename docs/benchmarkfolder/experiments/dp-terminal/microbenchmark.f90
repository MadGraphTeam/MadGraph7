program microbenchmark
use ML5_0_POLYNOMIAL_CONSTANTS
use, intrinsic :: iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(8) :: A(4,0:14,4), B(4,0:4,4), DENSE(4,0:14,4), OUT(4,0:14,4)
integer :: I,J,K,N,PAIR,REPEATS,ITERATIONS,ORDER,ALGORITHM
integer(int64) :: START,FINISH,RATE,DBITS(2),TBITS(2)
real(8) :: SECONDS,CHECKSUM
logical :: ENTRY_INEXACT,FAST
character(64) :: ARG
ITERATIONS=1000
REPEATS=7
call get_command_argument(1,ARG)
if (len_trim(ARG)>0) read(ARG,*) ITERATIONS
call get_command_argument(2,ARG)
if (len_trim(ARG)>0) read(ARG,*) REPEATS
do I=1,4
  do K=0,14
    do J=1,4
      A(J,K,I)=cmplx(real(3*I-2*J+K,8)/13.0_8,real(I+J-K,8)/17.0_8,8)
    enddo
  enddo
  do K=0,4
    do J=1,4
      B(J,K,I)=cmplx(real(I+3*J+K,8)/19.0_8,real(2*I-J+K,8)/23.0_8,8)
    enddo
  enddo
enddo
call IEEE_SET_HALTING_MODE(IEEE_ALL,.false.)
call IEEE_SET_ROUNDING_MODE(IEEE_NEAREST)
call IEEE_SET_FLAG(IEEE_ALL,.false.)
call IEEE_SET_FLAG(IEEE_INEXACT,.true.)
call DP_DENSE(A,4,B,4,4,DENSE)
call IEEE_GET_FLAG(IEEE_INEXACT,ENTRY_INEXACT)
call DP_TERMINAL(A,4,B,4,4,OUT,ENTRY_INEXACT,FAST)
if (.not.FAST) stop 1
do I=1,4
  do K=0,14
    DBITS=transfer(DENSE(I,K,I),DBITS)
    TBITS=transfer(OUT(I,K,I),TBITS)
    if (any(DBITS/=TBITS)) stop 2
  enddo
enddo
call system_clock(count_rate=RATE)
write(*,'(A)') 'pair,order,algorithm,iterations,seconds,diagonal_checksum'
do PAIR=1,REPEATS
  do ORDER=1,2
    ALGORITHM=mod(PAIR+ORDER,2)
    call system_clock(START)
    if (ALGORITHM==0) then
      do N=1,ITERATIONS
        call DP_DENSE(A,4,B,4,4,OUT)
      enddo
    else
      do N=1,ITERATIONS
        ! Include the caller-side status query required by the proposed path.
        call IEEE_GET_FLAG(IEEE_INEXACT,ENTRY_INEXACT)
        call DP_TERMINAL(A,4,B,4,4,OUT,ENTRY_INEXACT,FAST)
      enddo
      if (.not.FAST) stop 3
    endif
    call system_clock(FINISH)
    SECONDS=real(FINISH-START,8)/real(RATE,8)
    CHECKSUM=0.0_8
    do I=1,4
      do K=0,14
        CHECKSUM=CHECKSUM+real(OUT(I,K,I))+aimag(OUT(I,K,I))
      enddo
    enddo
    write(*,'(I0,A,I0,A,I0,A,I0,A,ES24.16,A,ES24.16)') &
      PAIR,',',ORDER,',',ALGORITHM,',',ITERATIONS,',',SECONDS,',',CHECKSUM
  enddo
enddo
end program
