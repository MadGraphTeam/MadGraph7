! SPDX-License-Identifier: 0BSD
! Caller-supplied higher-accuracy kinematics through the Fortran module
! (evaluate(..., corrections=)): ./refined_v4_fortran artifact process
! input.txt. refined_v4.cpp's input and calls, every double given as its
! IEEE bit pattern (a signed 64-bit integer); the output prints "call
! <name>" and "virtual ..." with each double as "b:" and its bit pattern in
! hexadecimal, or "refused <ierr> <message>".
program refined_v4_fortran
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  character(len=4096) :: artifact, process, path
  integer :: unit, points, legs, p, l, i
  integer(c_int64_t), allocatable :: bits(:)
  real(c_double), allocatable :: hi(:,:,:), lo(:,:,:,:), zero(:,:,:,:), far(:,:,:,:), scales(:)
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result
  integer(c_int) :: ierr
  character(:), allocatable :: message

  call get_command_argument(1, artifact)
  call get_command_argument(2, process)
  call get_command_argument(3, path)
  open(newunit=unit, file=trim(path), status='old', action='read')
  read(unit, *) points, legs
  allocate(hi(4, legs, points), lo(4, legs, points, 1), scales(points), bits(8*legs+1))
  do p = 1, points
    read(unit, *) bits
    do l = 1, legs
      do i = 1, 4
        hi(i, l, p) = transfer(bits(4*(l-1)+i), 1.0_c_double)
        lo(i, l, p, 1) = transfer(bits(4*legs+4*(l-1)+i), 1.0_c_double)
      end do
    end do
    scales(p) = transfer(bits(8*legs+1), 1.0_c_double)
  end do
  close(unit)
  call runtime%load(trim(artifact), process=trim(process), ierr=ierr)
  if (ierr /= VIRTUAL_OK) call fail(virtual_last_error())
  call runtime%set_poles('none', ierr=ierr)
  if (ierr /= VIRTUAL_OK) call fail(virtual_last_error())
  call runtime%set_reduction('ninjars', ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) call fail(message)
  call runtime%evaluate(hi, scales, result, ierr=ierr, error_message=message)
  if (ierr /= VIRTUAL_OK) call fail(message)
  call put('plain', result)
  allocate(zero, mold=lo)
  zero = 0.0_c_double
  call runtime%evaluate(hi, scales, result, ierr=ierr, error_message=message, corrections=zero)
  if (ierr /= VIRTUAL_OK) call fail(message)
  call put('zero', result)
  call runtime%evaluate(hi, scales, result, ierr=ierr, error_message=message, corrections=lo)
  if (ierr /= VIRTUAL_OK) call fail(message)
  call put('refined', result)
  ! Point 2, leg 2, py (Fortran indices): beyond half an ulp of its word.
  far = lo
  far(3, 2, 2, 1) = 1.0e-3_c_double * (abs(hi(3, 2, 2)) + 1.0_c_double)
  write(*, '(a)') 'call inconsistent'
  call runtime%evaluate(hi, scales, result, ierr=ierr, error_message=message, corrections=far)
  if (ierr == VIRTUAL_OK) then
    write(*, '(a)') 'accepted'
  else
    write(*, '(a,i0,a)') 'refused ', ierr, ' '//message
  end if
  call runtime%close()
contains
  subroutine fail(what)
    character(*), intent(in) :: what
    write(*, '(a)') 'error '//what
    stop 1
  end subroutine
  subroutine put(name, evaluation)
    character(*), intent(in) :: name
    type(virtual_evaluation), intent(in) :: evaluation
    real(c_double), allocatable :: values(:)
    integer :: j
    write(*, '(a)') 'call '//name
    values = reshape(evaluation%virtual_values, [size(evaluation%virtual_values)])
    write(*, '(a)', advance='no') 'virtual'
    do j = 1, size(values)
      write(*, '(a,z16.16)', advance='no') ' b:', transfer(values(j), 0_c_int64_t)
    end do
    write(*, *)
  end subroutine
end program
