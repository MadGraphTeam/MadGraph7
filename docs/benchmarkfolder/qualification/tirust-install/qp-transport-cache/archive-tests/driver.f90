program transport_cache
  use mtirust
  use iso_c_binding
  use ieee_arithmetic
  use ieee_exceptions
  implicit none
  interface
    subroutine mock_reset(bad, fail_clear) bind(C)
      import c_int
      integer(c_int), value :: bad, fail_clear
    end subroutine
    subroutine mock_counters(clears, peak) bind(C)
      import c_int
      integer(c_int), intent(out) :: clears, peak
    end subroutine
  end interface
  real(KI_QTIR) :: p(0:3,1), s(1,1), scale
  complex(KI_QTIR) :: m(1), coefs(1,4)
  integer(c_int64_t) :: stats(6)
  integer(c_int) :: clears, peak
  integer :: bad, mode, finite, status, count
  logical :: old_halts(size(ieee_all)), halts(size(ieee_all)), sticky
  logical :: flags(5)
  integer(c_int64_t) :: bits(2)
  type(ieee_status_type) :: old_status
  character(32) :: action
  character(512) :: message

  call get_command_argument(1, action)
  call ieee_get_status(old_status)
  call ieee_get_halting_mode(ieee_all, old_halts)
  call ieee_set_halting_mode(ieee_all, .false.)
  p = 0
  s = 0
  m = cmplx(1, 0, kind=KI_QTIR)
  count = 0
  if (trim(action) == 'underflow') then
    do bad = 5, 6
      call mock_reset(int(bad,c_int), 0_c_int)
      call ieee_set_flag(ieee_all, .false.)
      call ieee_set_flag(ieee_divide_by_zero, .true.)
      scale = 2
      call tirust_tensor_evaluate(1, 0, p, s, m, scale, coefs, status, &
        pole_mode=0, compute_finite=.true., check_pass=.false., cache_enabled=.true.)
      call ieee_get_flag(ieee_invalid, flags(1))
      call ieee_get_flag(ieee_divide_by_zero, flags(2))
      call ieee_get_flag(ieee_overflow, flags(3))
      call ieee_get_flag(ieee_underflow, flags(4))
      call ieee_get_flag(ieee_inexact, flags(5))
      call ieee_get_halting_mode(ieee_all, halts)
      call require(.not.any(halts), 'underflow halting modes unchanged')
      call require(flags(2), 'underflow entry sticky flag preserved')
      call require(status == TIRUST_SUCCESS, 'finite underflow remains accepted')
      call require(ieee_is_finite(aimag(coefs(1,3))), 'underflow result finite')
      if (bad == 5) call require(aimag(coefs(1,3)) == 0, 'below-range value rounds to zero')
      if (bad == 6) call require(aimag(coefs(1,3)) > 0, 'representable subnormal remains nonzero')
      bits = transfer(aimag(coefs(1,3)), bits)
      call mock_counters(clears, peak)
      call require(clears == 0, 'accepted underflow does not invalidate')
      call check_stats(0, 1, 1)
      write(*,'(A,I0,A,5L2,A,2Z17.16)') 'UNDERFLOW ', bad, ' flags=', flags, ' bits=', bits
      count = count + 1
    enddo
  else
  do finite = 0, 1
    do mode = 0, 3
      if (finite == 0 .and. mode == 3) cycle
      if (trim(action) == 'clear-failure') then
        call mock_reset(1_c_int, 1_c_int)
        call evaluate(2, .false.)
        call mock_counters(clears, peak)
        call require(clears == 1, 'clear failure was attempted')
        call check_stats(0, 1, 1)
        count = count + 1
      else
        do bad = 1, 4
          call mock_reset(int(bad,c_int), 0_c_int)
          call evaluate(1, .true.)
          call check_stats(0, 1, 1)
          call evaluate(1, .true.)
          call check_stats(1, 1, 1)
          call mock_counters(clears, peak)
          call require(clears == 0, 'successful transport must not clear')
          call evaluate(2, .false.)
          call check_stats(1, 2, 0)
          call mock_counters(clears, peak)
          call require(clears == 1 .and. peak == 2, 'clear includes prior valid entry')
          call evaluate(2, .false.)
          call check_stats(1, 3, 0)
          call mock_counters(clears, peak)
          call require(clears == 2, 'identical failed key reevaluated and cleared')
          call evaluate(1, .true.)
          call check_stats(1, 4, 1)
          call mock_counters(clears, peak)
          call require(clears == 2, 'later success must not clear')
          count = count + 1
        enddo
      endif
    enddo
  enddo
  endif
  call tirust_finalize(status)
  call require(status == TIRUST_SUCCESS, 'finalize')
  call ieee_set_status(old_status)
  call ieee_set_halting_mode(ieee_all, old_halts)
  write(*,'(A,A,A,I0)') 'PASS ', trim(action), ' cases=', count
contains
  subroutine require(condition, description)
    logical, intent(in) :: condition
    character(*), intent(in) :: description
    if (.not.condition) then
      write(*,'(A,A)') 'FAIL ', description
      stop 1
    endif
  end subroutine
  subroutine check_stats(expected_hits, expected_misses, expected_entries)
    integer, intent(in) :: expected_hits, expected_misses, expected_entries
    call tirust_qp_cache_statistics(stats, status)
    call require(status == TIRUST_SUCCESS, 'statistics status')
    call require(stats(3) == expected_entries, 'rejected entries must be absent')
    call require(stats(1) == expected_hits, 'cache hit count')
    call require(stats(2) == expected_misses, 'provider evaluation count')
  end subroutine
  subroutine evaluate(key, success)
    integer, intent(in) :: key
    logical, intent(in) :: success
    integer :: j
    call ieee_set_flag(ieee_all, .false.)
    call ieee_set_flag(ieee_divide_by_zero, .true.)
    scale = real(key, kind=KI_QTIR)
    coefs = cmplx(123, 456, kind=KI_QTIR)
    call tirust_tensor_evaluate(1, 0, p, s, m, scale, coefs, status, &
      pole_mode=mode, compute_finite=finite == 1, check_pass=.false., cache_enabled=.true.)
    call ieee_get_flag(ieee_divide_by_zero, sticky)
    call require(sticky, 'entry sticky flag preserved')
    call ieee_get_halting_mode(ieee_all, halts)
    call require(.not.any(halts), 'halting modes unchanged')
    if (success) then
      call require(status == TIRUST_SUCCESS, 'successful status')
      do j = 1, 4
        if (j == 4 .and. finite == 0) then
          call require(coefs(1,j) == cmplx(0,0,kind=KI_QTIR), 'partial finite placeholder')
        else
          call require(coefs(1,j) == cmplx(1.25_KI_QTIR,0,kind=KI_QTIR), 'decoded coefficient')
          call require(ieee_is_negative(aimag(coefs(1,j))), 'negative zero preserved')
        endif
      enddo
    else
      call require(status == TIRUST_ERROR, 'public transport must remain ERROR')
      call require(all(coefs == cmplx(0,0,kind=KI_QTIR)), 'failure outputs remain zero')
      call tirust_error_message(message)
      call require(trim(message) == &
        'TIRust returned an invalid or nonfinite binary128 output record.', 'local transport error preserved')
    endif
  end subroutine
end program
