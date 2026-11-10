! Proposed diagnostic only: no native execution is implied by this source.
! Link the same driver independently against each qualified installation.
program qp_transport_boundary
  use mtirust
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  real(KI_QTIR) :: shifts(0:3,1), invariants(1,1)
  complex(KI_QTIR) :: masses(1), coefs(1,4)
  integer(c_int64_t) :: counts(6)
  integer :: status, cache_status, i
  logical :: saved_halts(size(ieee_all)), flags(5)
  type(ieee_status_type) :: saved_status
  character(512) :: message

  if (radix(1.0_KI_QTIR) /= 2 .or. digits(1.0_KI_QTIR) /= 113) stop 2
  call ieee_get_status(saved_status)
  call ieee_get_halting_mode(ieee_all,saved_halts)
  call ieee_set_halting_mode(ieee_all,.false.)
  shifts = 0
  invariants = 0
  call tirust_clear_cache(status)
  if (status /= TIRUST_SUCCESS) stop 3
  call tirust_set_cache_budget(1048576_c_int64_t,status)
  if (status /= TIRUST_SUCCESS) stop 4
  call evaluate('ordinary-seed',4.0_KI_QTIR,4.0_KI_QTIR)
  do i=1,2
    call evaluate('large-mass',1.0e4932_KI_QTIR,1.0_KI_QTIR)
  enddo
  call tirust_finalize(status)
  write(*,'(A,I0)') 'FINALIZE_STATUS ',status
  call ieee_set_status(saved_status)
  call ieee_set_halting_mode(ieee_all,saved_halts)
contains
  subroutine evaluate(label,mass_squared,mu_squared)
    character(*), intent(in) :: label
    real(KI_QTIR), intent(in) :: mass_squared,mu_squared
    masses(1) = cmplx(mass_squared,0.0_KI_QTIR,kind=KI_QTIR)
    write(*,'(A,A)') 'CASE ',label
    write(*,'(A,ES64.39E4)') 'MASS_SQUARED ',mass_squared
    write(*,'(A,ES64.39E4)') 'MU_SQUARED ',mu_squared
    call ieee_set_flag(ieee_all,.false.)
    call tirust_tensor_evaluate(1,0,shifts,invariants,masses,mu_squared,coefs,status, &
      pole_mode=TIRUST_POLES_BOTH,compute_finite=.true.,check_pass=.false.,cache_enabled=.true.)
    call ieee_get_flag(ieee_invalid,flags(1))
    call ieee_get_flag(ieee_divide_by_zero,flags(2))
    call ieee_get_flag(ieee_overflow,flags(3))
    call ieee_get_flag(ieee_underflow,flags(4))
    call ieee_get_flag(ieee_inexact,flags(5))
    call tirust_error_message(message)
    call tirust_qp_cache_statistics(counts,cache_status)
    write(*,'(A,I0)') 'PUBLIC_STATUS ',status
    write(*,'(A,A)') 'ERROR_MESSAGE ',trim(message)
    write(*,'(A,5L2)') 'IEEE_INVALID_DIVZERO_OVERFLOW_UNDERFLOW_INEXACT ',flags
    write(*,'(A,I0)') 'CACHE_STATUS ',cache_status
    write(*,'(A,6(1X,I0))') 'QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED',counts
    write(*,'(A,L1)') 'OUTPUTS_ALL_ZERO ',all(coefs == cmplx(0,0,kind=KI_QTIR))
    if (status == TIRUST_ERROR .and. trim(message) == &
        'TIRust returned an invalid or nonfinite binary128 output record.') then
      write(*,'(A)') 'CLASSIFICATION public-transport-rejection'
    else if (status /= TIRUST_SUCCESS) then
      write(*,'(A)') 'CLASSIFICATION provider-refusal-before-public-transport-proof'
    else
      write(*,'(A)') 'CLASSIFICATION public-success'
    endif
  end subroutine
end program
