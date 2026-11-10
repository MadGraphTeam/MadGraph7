! Actual installed MTIRUST and production Rust bridge; A/B share its one context.
program lifecycle
  use mtirust
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_exceptions
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer :: status, first, second, candidate
  integer(c_int64_t) :: counts(9)
  logical :: flags_before(5), flags_after(5)
  type(ieee_round_type) :: rounding_before, rounding_after
  real(KI_TIR) :: shifts(0:3,1), invariants(1,1)
  complex(KI_TIR) :: masses(1), values(1,4)
  character(len=512) :: message_before, message_after
  character(len=32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) candidate
  if(tirust_context_generation()/=-1) stop 1
  call tirust_set_cache_budget(-1_c_int64_t,status)
  if(status/=TIRUST_ERROR) stop 2
  call tirust_error_message(message_before)
  call ieee_get_flag(ieee_all,flags_before)
  call ieee_get_rounding_mode(rounding_before)
  call a_clear()
  call b_clear()
  call ieee_get_flag(ieee_all,flags_after)
  call ieee_get_rounding_mode(rounding_after)
  if(any(flags_before.neqv.flags_after)) stop 3
  if(rounding_before/=rounding_after) stop 4
  call tirust_error_message(message_after)
  if(candidate==1) then
    if(tirust_context_generation()/=-1) stop 5
    if(message_before/=message_after) stop 6
  else
    if(tirust_context_generation()<1) stop 7
    if(len_trim(message_after)/=0) stop 8
  endif
  call tirust_register_plan(1,0,status)
  if(status/=TIRUST_SUCCESS) stop 9
  first=tirust_context_generation()
  shifts=0
  invariants=0
  masses=cmplx(1,0,kind=KI_TIR)
  call tirust_tensor_evaluate(1,0,shifts,invariants,masses,1.0_KI_TIR,values,status)
  if(status/=TIRUST_SUCCESS) stop 10
  call tirust_cache_statistics(counts,status)
  if(status/=TIRUST_SUCCESS.or.counts(7)<1) stop 11
  call tirust_set_cache_budget(-1_c_int64_t,status)
  call b_clear()
  if(tirust_context_generation()/=first) stop 12
  call tirust_error_message(message_after)
  if(len_trim(message_after)/=0) stop 13
  call tirust_cache_statistics(counts,status)
  if(status/=TIRUST_SUCCESS.or.counts(7)/=0) stop 14
  call tirust_finalize(status)
  if(status/=TIRUST_SUCCESS.or.tirust_context_generation()/=-1) stop 15
  call b_clear()
  if(candidate==1.and.tirust_context_generation()/=-1) stop 16
  ! Public clearing retains its documented creating behavior in either arm.
  call tirust_clear_cache(status)
  second=tirust_context_generation()
  if(status/=TIRUST_SUCCESS.or.second<=first) stop 17
  call tirust_finalize(status)
  call tirust_register_plan(1,0,status)
  if(status/=TIRUST_SUCCESS.or.tirust_context_generation()<=second) stop 18
  call a_clear()
  call tirust_finalize(status)
  if(status/=TIRUST_SUCCESS) stop 19
  write(*,*) 'LIFECYCLE_PASS',candidate,first,second
end program
