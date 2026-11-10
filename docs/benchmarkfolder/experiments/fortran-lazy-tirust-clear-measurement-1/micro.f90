! Direct generated-wrapper cost, not a reduction/full-amplitude estimate.
program micro
  use mtirust
  implicit none
  integer :: active, repetitions, i, status, generation_before, generation_after
  integer(kind=8) :: before, after, rate
  character(len=32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) active
  call get_command_argument(2,arg)
  read(arg,*) repetitions
  if(active==1) then
    call tirust_register_plan(1,0,status)
    if(status/=TIRUST_SUCCESS) stop 1
  endif
  generation_before=tirust_context_generation()
  call system_clock(before,rate)
  do i=1,repetitions
    call a_clear()
  enddo
  call system_clock(after)
  generation_after=tirust_context_generation()
  write(*,*) 'MICRO',active,repetitions,generation_before,generation_after,after-before,rate
  call tirust_finalize(status)
  if(status/=TIRUST_SUCCESS) stop 2
end program
