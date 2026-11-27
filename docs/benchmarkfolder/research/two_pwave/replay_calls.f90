program replay
  ! Replays TIRust call records (binary64 hex, the format of the refused-call
  ! files of FIXTURES.md: CALL id n rank check finite powers, MU2, M, I, S,
  ! END) through the binary128 raised-power entry (tirust_tensor_evaluate_powers,
  ! the inputs widened exactly) and the binary64 one, and prints both
  ! statuses and TIRust's message for a refusal. Build against an
  ! installation's lib directory L:
  !   gfortran -I$L replay_calls.f90 -o replay -L$L -ltirust -lmadloop_tirust -lgcc_s -Wl,-rpath,$L
  !   ./replay < refused.txt
  use mtirust
  implicit none
  integer, parameter :: qp = 16, dp = 8
  character(len=400) :: line, msg
  integer :: id, n, rank, chk, ios, i, j, status, sdp, nserved, nref, nc
  character(len=1) :: fin
  character(len=14) :: pw
  integer :: powers(7)
  integer(8) :: h(12)
  real(dp) :: mu2d, sh(0:3,7), inv(7,7)
  complex(dp) :: md(7)
  real(qp) :: shq(0:3,7), invq(7,7), mu2q
  complex(qp) :: mq(7)
  complex(qp), allocatable :: cq(:,:)
  complex(dp), allocatable :: cd(:,:)
  nserved = 0; nref = 0
  do
    read(*,'(a)',iostat=ios) line
    if (ios /= 0) exit
    if (line(1:4) == 'CALL') then
      read(line(5:),*) id, n, rank, chk, fin, powers(1:n)
    elseif (line(1:3) == 'MU2') then
      read(line(4:),'(z17)') h(1); mu2d = transfer(h(1), mu2d)
    elseif (line(1:2) == 'M ') then
      read(line(3:),*) i; read(line(6:),*) 
      call hexs(line(6:), 2, h); md(i) = cmplx(transfer(h(1),1d0), transfer(h(2),1d0), dp)
    elseif (line(1:2) == 'I ') then
      read(line(3:),*) i; call hexs(line(6:), n, h)
      do j = 1, n; inv(i,j) = transfer(h(j),1d0); enddo
    elseif (line(1:2) == 'S ') then
      read(line(3:),*) i; call hexs(line(6:), 4, h)
      do j = 0, 3; sh(j,i) = transfer(h(j+1),1d0); enddo
    elseif (line(1:3) == 'END') then
      nc = tirust_coefficient_count(rank)
      allocate(cq(nc,4), cd(nc,4))
      shq(:,1:n) = real(sh(:,1:n), qp); invq(1:n,1:n) = real(inv(1:n,1:n), qp)
      mq(1:n) = cmplx(md(1:n), kind=qp); mu2q = real(mu2d, qp)
      call tirust_tensor_evaluate_powers(n, rank, powers(1:n), shq(:,1:n), invq(1:n,1:n), mq(1:n), mu2q, &
           cq, status, pole_mode=0, compute_finite=fin=='T', check_pass=chk/=0, cache_enabled=.false.)
      msg = ''
      if (status /= TIRUST_SUCCESS) call tirust_error_message(msg)
      call tirust_tensor_evaluate_powers(n, rank, powers(1:n), sh(:,1:n), inv(1:n,1:n), md(1:n), mu2d, &
           cd, sdp, pole_mode=0, compute_finite=fin=='T', check_pass=chk/=0, cache_enabled=.false.)
      if (status == TIRUST_SUCCESS) then
        nserved = nserved + 1
      else
        nref = nref + 1
      endif
      pw = ''
      write(pw,'(7i2)') powers(1:n)
      write(*,'(a,i4,a,i2,a,i2,a,a14,a,i2,a,i2,2x,a)') 'call', id, ' n', n, ' r', rank, ' pow', pw, &
           '  QP', status, '  DP', sdp, trim(msg)
      deallocate(cq, cd)
    endif
  enddo
  write(*,*) 'QP served', nserved, ' refused', nref
contains
  subroutine hexs(s, k, h)
    character(len=*), intent(in) :: s
    integer, intent(in) :: k
    integer(8), intent(out) :: h(:)
    integer :: p, q
    character(len=16) :: w
    p = 1
    do q = 1, k
      do while (s(p:p) == ' '); p = p + 1; enddo
      w = s(p:p+15); read(w,'(z16)') h(q); p = p + 16
    enddo
  end subroutine
end program
