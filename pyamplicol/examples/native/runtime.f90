! SPDX-License-Identifier: 0BSD

program runtime_fortran
  use, intrinsic :: iso_c_binding, only: c_double, c_size_t
  use rusticol, only: rusticol_runtime
  implicit none

  type(rusticol_runtime) :: runtime
  character(len=4096) :: artifact, process_key, parameters
  real(c_double), target :: momenta(16)
  real(c_double), allocatable :: totals(:), resolved(:, :, :)
  real(c_double) :: checked_total, scale

  if (command_argument_count() < 1 .or. command_argument_count() > 3) then
    error stop "usage: runtime_fortran ARTIFACT [PROCESS [PARAMETERS.json]]"
  end if
  call get_command_argument(1, artifact)
  process_key = ""
  parameters = ""
  if (command_argument_count() >= 2) call get_command_argument(2, process_key)
  if (command_argument_count() == 3) call get_command_argument(3, parameters)

  if (len_trim(parameters) > 0) then
    call runtime%load(trim(artifact), process_key=trim(process_key), &
                      model_parameters=trim(parameters))
  else if (len_trim(process_key) > 0) then
    call runtime%load(trim(artifact), process_key=trim(process_key))
  else
    call runtime%load(trim(artifact))
  end if
  call runtime%set_model_parameter( &
      "normalization.alpha_s_me_check", 0.118_c_double)

  ! One d d~ > z g point, flattened as [point][particle][E,px,py,pz].
  momenta = [ &
      500.0_c_double, 0.0_c_double, 0.0_c_double, 500.0_c_double, &
      500.0_c_double, 0.0_c_double, 0.0_c_double, -500.0_c_double, &
      504.157625672_c_double, -304.1084262865_c_double, &
      208.76026523528103_c_double, 331.35611794513767_c_double, &
      495.842374328_c_double, 304.1084262865_c_double, &
      -208.76026523528103_c_double, -331.35611794513767_c_double ]

  call runtime%evaluate(momenta, 1_c_size_t, totals)
  call runtime%evaluate_resolved(momenta, 1_c_size_t, resolved)
  checked_total = sum(resolved(:, :, 1))
  scale = max(1.0_c_double, abs(totals(1)))
  if (abs(totals(1) - checked_total) > 1.0e-12_c_double * scale) then
    error stop "resolved components do not reproduce the total"
  end if

  write(*, '(A)') "process=" // runtime%process_key()
  write(*, '(A)') "color_accuracy=" // runtime%color_accuracy()
  write(*, '(A,ES24.16)') "total=", totals(1)
  call runtime%close()
end program runtime_fortran
