program lookup_check
  implicit none
  integer :: unit, ios, total, k, which, g, ct, n, i, j
  integer :: cls, leg, sgn, closure, actual(6), expected(6)
  logical :: shell, known
  character(len=4096) :: input
  call get_command_argument(1, input)
  open(newunit=unit, file=trim(input), status='old', action='read', iostat=ios)
  if (ios /= 0) error stop 1
  read(unit, *, iostat=ios) total
  if (ios /= 0 .or. total < 1) error stop 2
  do k = 1, total
    read(unit, *, iostat=ios) which, g, ct, n, i, j, expected
    if (ios /= 0) error stop 3
    cls = -777
    leg = -777
    sgn = -777
    closure = -777
    shell = .true.
    known = .true.
    select case (which)
    case (1)
      call TEST_KINEMATIC_IDENTITY(g, ct, n, i, j, cls, leg, sgn, closure, shell, known)
    case (2)
      call BOUNDARY_KINEMATIC_IDENTITY(g, ct, n, i, j, cls, leg, sgn, closure, shell, known)
    case default
      error stop 4
    end select
    actual = [cls, leg, sgn, closure, merge(1, 0, shell), merge(1, 0, known)]
    if (any(actual /= expected)) then
      write(*, '(A,7(1X,I0))') 'MISMATCH', k, which, g, ct, n, i, j
      write(*, '(A,6(1X,I0))') 'ACTUAL', actual
      write(*, '(A,6(1X,I0))') 'EXPECTED', expected
      error stop 5
    end if
  end do
  read(unit, *, iostat=ios) k
  if (ios >= 0) error stop 6
  close(unit)
  write(*, '(A,1X,I0)') 'INTEGER_LOOKUP_PASS', total
end program
