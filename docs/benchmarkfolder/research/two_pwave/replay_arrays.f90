program replay_arrays
  ! Replays TIRust call records (binary64 hex, the format of the refused-call
  ! files of FIXTURES.md: CALL id n rank check finite powers, MU2, M, I, S,
  ! END) through the binary64 raised-power entry (tirust_tensor_evaluate_powers,
  ! cache off, the check-pass flag of the record) and prints, per record,
  !   ARR id status
  ! and then one line per coefficient: the four slots as real and imaginary
  ! parts, Z16 bit patterns. Build against an installation's lib directory L:
  !   gfortran -I$L replay_arrays.f90 -o replay_arrays -L$L -ltirust -lmadloop_tirust -lgcc_s -Wl,-rpath,$L
  !   ./replay_arrays < records.txt
  ! One record per process shows a call alone; several in one process show
  ! whether earlier calls change a result.
  use mtirust
  implicit none
  integer, parameter :: dp = 8
  character(len=400) :: line
  integer :: id, n, rank, chk, ios, i, j, sdp, nc
  character(len=1) :: fin
  integer :: powers(6)
  integer(8) :: h(12)
  real(dp) :: mu2d, sh(0:3,6), inv(6,6)
  complex(dp) :: md(6)
  complex(dp), allocatable :: cd(:,:)
  do
    read(*,'(a)',iostat=ios) line
    if (ios /= 0) exit
    if (line(1:4) == 'CALL') then
      read(line(5:),*) id, n, rank, chk, fin, powers(1:n)
    elseif (line(1:3) == 'MU2') then
      read(line(4:),'(z17)') h(1); mu2d = transfer(h(1), mu2d)
    elseif (line(1:2) == 'M ') then
      read(line(3:),*) i
      call hexs(line(6:), 2, h); md(i) = cmplx(transfer(h(1),1d0), transfer(h(2),1d0), dp)
    elseif (line(1:2) == 'I ') then
      read(line(3:),*) i; call hexs(line(6:), n, h)
      do j = 1, n; inv(i,j) = transfer(h(j),1d0); enddo
    elseif (line(1:2) == 'S ') then
      read(line(3:),*) i; call hexs(line(6:), 4, h)
      do j = 0, 3; sh(j,i) = transfer(h(j+1),1d0); enddo
    elseif (line(1:3) == 'END') then
      nc = tirust_coefficient_count(rank)
      allocate(cd(nc,4))
      call tirust_tensor_evaluate_powers(n, rank, powers(1:n), sh(:,1:n), inv(1:n,1:n), md(1:n), mu2d, &
           cd, sdp, pole_mode=0, compute_finite=fin=='T', check_pass=chk/=0, cache_enabled=.false.)
      write(*,'(a,i6,i3)') 'ARR', id, sdp
      do i = 1, nc
        write(*,'(8(1x,z16.16))') (transfer(real(cd(i,j),dp),1_8), transfer(aimag(cd(i,j)),1_8), j=1,4)
      enddo
      deallocate(cd)
    endif
  enddo
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
