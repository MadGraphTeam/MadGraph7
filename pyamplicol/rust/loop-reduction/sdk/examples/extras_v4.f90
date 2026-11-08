! SPDX-License-Identifier: 0BSD
! Plan v4 SDK extras through the Fortran module: ./extras_v4_fortran artifact
! process input.txt. The input is extras_v4.c's with every double given as
! its IEEE bit pattern (a signed 64-bit integer); per call the output prints
! "call <name>", "born ..." and "virtual ..." with each double as "b:" and its
! bit pattern in hexadecimal (the C layout order; complex as re im pairs) and
! "decimal <json>".
program extras_v4_fortran
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  character(len=4096) :: artifact, process, path
  integer :: unit, points, legs, spin_count, p, l, i, c, k, n
  integer(c_int64_t), allocatable :: bits(:)
  real(c_double), allocatable :: momenta(:,:,:), scales(:)
  integer(c_size_t), allocatable :: spin_legs(:)
  complex(c_double), allocatable :: vectors(:,:,:)
  integer(c_int64_t) :: re_bits, im_bits
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result
  type(virtual_correlated_evaluation) :: correlated
  integer(c_int) :: ierr
  character(:), allocatable :: message
  character(len=24), parameter :: names(13) = [character(len=24) :: 'ordinary', 'correlated', &
      'correlated-spin', 'correlated-uv', 'correlated-none', 'resolved', 'resolved-ir', &
      'ordinary-dd', 'correlated-spin-dd', 'resolved-dd', 'ordinary-ap', 'correlated-spin-ap', 'resolved-ap']
  character(len=16), parameter :: arithmetics(13) = [character(len=16) :: 'double', 'double', 'double', &
      'double', 'double', 'double', 'double', 'double-double', 'double-double', 'double-double', &
      'arbitrary', 'arbitrary', 'arbitrary']
  integer, parameter :: digits(13) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 32, 32]
  character(len=4), parameter :: poles(13) = [character(len=4) :: 'all', 'all', 'all', 'uv', 'none', &
      'all', 'ir', 'all', 'all', 'all', 'all', 'all', 'all']
  integer, parameter :: kinds(13) = [0, 1, 2, 1, 1, 3, 3, 0, 2, 3, 0, 2, 3]
  integer, parameter :: counts(13) = [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]

  call get_command_argument(1, artifact)
  call get_command_argument(2, process)
  call get_command_argument(3, path)
  open(newunit=unit, file=trim(path), status='old', action='read')
  read(unit, *) points, legs
  allocate(momenta(4, legs, points), scales(points), bits(4*legs+1))
  do p = 1, points
    read(unit, *) bits
    do l = 1, legs
      do i = 1, 4
        momenta(i, l, p) = transfer(bits(4*(l-1)+i), 1.0_c_double)
      end do
    end do
    scales(p) = transfer(bits(4*legs+1), 1.0_c_double)
  end do
  read(unit, *) spin_count
  allocate(spin_legs(spin_count), vectors(4, spin_count, 1))
  do l = 1, spin_count
    read(unit, *) spin_legs(l)
    do c = 1, 4
      read(unit, *) re_bits, im_bits
      vectors(c, l, 1) = cmplx(transfer(re_bits, 1.0_c_double), transfer(im_bits, 1.0_c_double), kind=c_double)
    end do
  end do
  close(unit)
  call runtime%load(trim(artifact), process=trim(process), ierr=ierr)
  if (ierr /= VIRTUAL_OK) call fail('open')
  do k = 1, size(names)
    n = min(counts(k), points)
    call runtime%set_arithmetic(trim(arithmetics(k)), digits=digits(k), ierr=ierr)
    if (ierr /= VIRTUAL_OK) call fail('arithmetic')
    write(*, '(a)') 'call '//trim(names(k))
    call runtime%set_poles(trim(poles(k)), ierr=ierr)
    if (ierr /= VIRTUAL_OK) then
      ! A pole selection the process does not serve (loop-induced: 'all'
      ! and 'none'): the refusal, then the next call.
      write(*, '(a)') 'refused '//virtual_last_error()
      cycle
    end if
    if (kinds(k) == 0 .or. kinds(k) == 3) then
      call runtime%evaluate(momenta(:, :, 1:n), scales(1:n), result, resolved=(kinds(k) == 3), &
          ierr=ierr, error_message=message)
      if (ierr /= VIRTUAL_OK) call fail(message)
      call put_reals('born', reshape(result%born, [size(result%born)]))
      call put_reals('virtual', reshape(result%virtual_values, [size(result%virtual_values)]))
      write(*, '(a)') 'decimal '//result%decimal_json
    else
      if (kinds(k) == 2) then
        call runtime%evaluate_correlated(momenta(:, :, 1:n), scales(1:n), correlated, ierr=ierr, &
            error_message=message, spin_legs=spin_legs, spin_vectors=vectors)
      else
        call runtime%evaluate_correlated(momenta(:, :, 1:n), scales(1:n), correlated, ierr=ierr, &
            error_message=message)
      end if
      if (ierr /= VIRTUAL_OK) call fail(message)
      call put_complex('born', reshape(correlated%born, [size(correlated%born)]))
      call put_complex('virtual', reshape(correlated%virtual_values, [size(correlated%virtual_values)]))
      write(*, '(a)') 'decimal '//correlated%decimal_json
    end if
  end do
  call runtime%close()
contains
  subroutine fail(what)
    character(*), intent(in) :: what
    write(*, '(a)') 'error '//what
    stop 1
  end subroutine
  subroutine put_reals(label, values)
    character(*), intent(in) :: label
    real(c_double), intent(in) :: values(:)
    integer :: j
    write(*, '(a)', advance='no') label
    do j = 1, size(values)
      write(*, '(a,z16.16)', advance='no') ' b:', transfer(values(j), 0_c_int64_t)
    end do
    write(*, *)
  end subroutine
  subroutine put_complex(label, values)
    character(*), intent(in) :: label
    complex(c_double), intent(in) :: values(:)
    integer :: j
    write(*, '(a)', advance='no') label
    do j = 1, size(values)
      write(*, '(a,z16.16,a,z16.16)', advance='no') ' b:', transfer(real(values(j), c_double), 0_c_int64_t), &
          ' b:', transfer(aimag(values(j)), 0_c_int64_t)
    end do
    write(*, *)
  end subroutine
end program
