! SPDX-License-Identifier: 0BSD
! Loop-induced selections through the Fortran module (evaluate_selected):
! ./selection_v4_fortran artifact process input.txt [binary64]. The input is
! selection_v4.c's with every double given as its IEEE bit pattern (a signed
! 64-bit integer); per call the output prints "call <name>", "statuses ...",
! "virtual ..." with each double as "b:" and its bit pattern in hexadecimal,
! "decimal <json>" and "axes <json>", or "refused <message>".
program selection_v4_fortran
  use, intrinsic :: iso_c_binding
  use rusticol_virtual
  implicit none
  character(len=4096) :: artifact, process, path
  character(len=16) :: mode
  integer :: unit, points, legs, count, p, l, i, k, n
  integer(c_int64_t), allocatable :: bits(:)
  real(c_double), allocatable :: momenta(:,:,:), scales(:)
  character(len=256), allocatable :: ids(:)
  integer(c_int32_t), allocatable :: rows(:)
  type(virtual_runtime) :: runtime
  type(virtual_evaluation) :: result
  integer(c_int) :: ierr
  character(:), allocatable :: message
  character(len=24), parameter :: names(11) = [character(len=24) :: 'anchors', 'one', 'subset', &
      'resolved-subset', 'anchors-subset', 'by-point', 'colour', 'anchors-dd', 'by-point-dd', &
      'anchors-ap', 'subset-ap']
  character(len=16), parameter :: arithmetics(11) = [character(len=16) :: 'double', 'double', 'double', &
      'double', 'double', 'double', 'double', 'double-double', 'double-double', 'arbitrary', 'arbitrary']
  integer, parameter :: digits(11) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 32]
  logical, parameter :: resolved(11) = [.true., .false., .false., .true., .true., .false., .false., &
      .true., .false., .true., .false.]
  ! 0 no IDs, 1 the first ID, 2 every ID.
  integer, parameter :: id_kinds(11) = [0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 2]
  logical, parameter :: by_point(11) = [.false., .false., .false., .false., .false., .true., .false., &
      .false., .true., .false., .false.]
  ! 0 none, 1 contracted, 2 amplitude-basis.
  integer, parameter :: resolutions(11) = [2, 0, 0, 1, 2, 0, 0, 2, 0, 2, 0]
  integer, parameter :: counts(11) = [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]

  call get_command_argument(1, artifact)
  call get_command_argument(2, process)
  call get_command_argument(3, path)
  mode = ''
  if (command_argument_count() > 3) call get_command_argument(4, mode)
  open(newunit=unit, file=trim(path), status='old', action='read')
  read(unit, *) points, legs
  allocate(momenta(4, legs, points), scales(points), bits(4*legs+1), rows(points))
  do p = 1, points
    read(unit, *) bits
    do l = 1, legs
      do i = 1, 4
        momenta(i, l, p) = transfer(bits(4*(l-1)+i), 1.0_c_double)
      end do
    end do
    scales(p) = transfer(bits(4*legs+1), 1.0_c_double)
  end do
  read(unit, *) count
  allocate(ids(count))
  backspace(unit)
  read(unit, *) count, ids
  read(unit, *) rows
  close(unit)
  call runtime%load(trim(artifact), process=trim(process), ierr=ierr)
  if (ierr /= VIRTUAL_OK) call fail('open')
  do k = 1, size(names)
    if (trim(mode) == 'binary64' .and. trim(arithmetics(k)) /= 'double') cycle
    n = min(counts(k), points)
    call runtime%set_arithmetic(trim(arithmetics(k)), digits=digits(k), ierr=ierr)
    if (ierr /= VIRTUAL_OK) call fail('arithmetic')
    write(*, '(a)') 'call '//trim(names(k))
    select case (k)
    case (7)
      call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, color_ids=['1'], &
          ierr=ierr, error_message=message)
    case default
      if (by_point(k)) then
        call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, &
            helicity_by_point=rows(1:n), ierr=ierr, error_message=message)
      else if (id_kinds(k) == 1 .and. resolutions(k) == 0) then
        call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, helicity_ids=ids(1:1), &
            resolved=resolved(k), ierr=ierr, error_message=message)
      else if (id_kinds(k) == 2 .and. resolutions(k) == 0) then
        call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, helicity_ids=ids, &
            resolved=resolved(k), ierr=ierr, error_message=message)
      else if (id_kinds(k) == 2) then
        call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, helicity_ids=ids, &
            color_resolution=resolution_name(resolutions(k)), resolved=resolved(k), ierr=ierr, &
            error_message=message)
      else
        call runtime%evaluate_selected(momenta(:, :, 1:n), scales(1:n), result, &
            color_resolution=resolution_name(resolutions(k)), resolved=resolved(k), ierr=ierr, &
            error_message=message)
      end if
    end select
    if (ierr /= VIRTUAL_OK) then
      write(*, '(a)') 'refused '//message
      cycle
    end if
    call put_statuses(result%statuses)
    call put_reals('virtual', reshape(result%virtual_values, [size(result%virtual_values)]))
    write(*, '(a)') 'decimal '//result%decimal_json
    write(*, '(a)') 'axes '//result%axes_json
  end do
  call runtime%close()
contains
  function resolution_name(kind) result(name)
    integer, intent(in) :: kind
    character(:), allocatable :: name
    if (kind == 2) then
      name = 'amplitude-basis'
    else
      name = 'contracted'
    end if
  end function
  subroutine fail(what)
    character(*), intent(in) :: what
    write(*, '(a)') 'error '//what
    stop 1
  end subroutine
  subroutine put_statuses(values)
    integer(c_int32_t), intent(in) :: values(:)
    integer :: j
    write(*, '(a)', advance='no') 'statuses'
    do j = 1, size(values)
      write(*, '(a,i0)', advance='no') ' ', values(j)
    end do
    write(*, *)
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
end program
