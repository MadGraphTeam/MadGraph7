module mc_native_cohort
  use mint_module, only: mint_draw, activate_mint_draw, park_mint_draw, finish_mint_fold, imode
  use mc_native_points, only: native_mc_point, activate_native_point, park_native_point, mc_fold_complete, mc_wait_real, &
       recycle_native_point, settle_native_points, release_native_points
  use mc_native_context, only: native_provider_request, mc_dispatch_provider_wave
  use iso_fortran_env, only: int64
  use mc_context_memory, only: enter_context,release_contexts
  implicit none
  private
  public :: evaluate_native_cohort, restore_generation_draw, park_generation_draw, release_native_cohort
  public :: native_context_base_bytes
  type :: lane_state
     type(native_mc_point) :: physics
     logical :: started=.false.
     integer, allocatable :: timing_stack(:)
  end type
  ! Parked lane states are referenced until release (shower scale swap).
  type(lane_state), allocatable, save, target :: lanes(:)
  type(native_provider_request), allocatable, save :: waiting(:)
  integer,allocatable,save :: owners(:)
  interface
     subroutine timing_suspend_point(categories)
       integer,allocatable,intent(out) :: categories(:)
     end subroutine
     subroutine timing_resume_point(categories)
       integer,allocatable,intent(inout) :: categories(:)
     end subroutine
     subroutine timing_begin_real_wave(sampled,lanes)
       integer,intent(in) :: sampled,lanes
     end subroutine
     subroutine timing_end_real_wave()
     end subroutine
  end interface
contains

  integer(int64) function native_context_base_bytes() result(bytes)
    type(lane_state) :: lane
    type(mint_draw) :: draw
    type(native_provider_request) :: request
    ! Full sized arrays can coexist during Born calibration. Include both
    ! MINT draw arrays, provider-wave owner indices and candidate decisions.
    bytes=(storage_size(lane,kind=int64)+2*storage_size(draw,kind=int64)+ &
           storage_size(request,kind=int64)+2*storage_size(0,kind=int64)+storage_size(.true.,kind=int64))/8
  end function

  subroutine advance_to_request(draw,lane)
    type(mint_draw),intent(inout) :: draw
    type(lane_state),intent(inout),target :: lane
    call activate_mint_draw(draw)
    call activate_native_point(lane%physics)
    if (lane%started) then
       call timing_resume_point(lane%timing_stack)
    else
       call timing_begin_point()
       lane%started=.true.
    endif
    do
       call sigintF_advance(lane%physics,draw%coordinates,draw%volume,draw%ifl,draw%integrals)
       select case(lane%physics%status)
       case(mc_wait_real)
          if (draw%ifl==2) error stop 'Native closing requested a full real amplitude'
          exit
       case(mc_fold_complete)
          call finish_mint_fold(draw)
          ! Close draws in issue order after every lane has finished its
          ! folds. Closing fills the discrete-sampling and surrogate fits.
          if (draw%ifl==2) exit
       case default
          error stop 'Unknown native integrand continuation'
       end select
    enddo
    if (draw%complete) then
       ! Closing has committed the contributions. The PDF and unweighting
       ! routines may leave a different sector selected. A queued event
       ! picks its own sector and regenerates its momenta from its saved
       ! fold, so the preceding subtraction chart has no further reader.
       lane%physics%has_chart=.false.
       call timing_end_point()
    else
       call timing_suspend_point(lane%timing_stack)
    endif
    call park_mint_draw(draw)
    call park_native_point(lane%physics)
  end subroutine

  subroutine evaluate_native_cohort(draws)
    type(mint_draw),intent(inout) :: draws(:)
    integer :: i,j,count,status,sampled
    if (allocated(lanes)) error stop 'Previous native cohort is still owned'
    if (allocated(waiting)) then
       if (size(waiting)<size(draws)) deallocate(waiting,owners)
    endif
    if (.not.allocated(waiting)) then
       allocate(waiting(size(draws)),owners(size(draws)),stat=status)
       if (status/=0) error stop 'Cannot allocate native request buffers; reduce vector_size'
    endif
    allocate(lanes(size(draws)),stat=status)
    if (status/=0) error stop 'Cannot allocate native MC points; reduce vector_size'
    do i=1,size(draws)
       lanes(i)%physics%draw_id=draws(i)%id
    enddo
    do
       count=0
       do i=1,size(draws)
          if (draws(i)%ifl==2) cycle
          call enter_context(i)
          call advance_to_request(draws(i),lanes(i))
          if (lanes(i)%physics%status/=mc_wait_real) cycle
          count=count+1
          owners(count)=i
          waiting(count)=lanes(i)%physics%provider
       enddo
       if (count==0) exit
       ! A suspended sampled draw keeps its timing stack until it resumes.
       sampled=0
       do j=1,count
          if (allocated(lanes(owners(j))%timing_stack)) sampled=sampled+1
       enddo
       call timing_begin_real_wave(sampled,count)
       call mc_dispatch_provider_wave(waiting(:count))
       call timing_end_real_wave()
       do j=1,count
          lanes(owners(j))%physics%provider=waiting(j)
       enddo
    enddo
    do i=1,size(draws)
       call enter_context(i)
       if (draws(i)%ifl/=2) error stop 'Native scheduler lost an unfinished fold'
       call advance_to_request(draws(i),lanes(i))
       if (.not.draws(i)%complete) error stop 'Native scheduler lost a closing call'
    enddo
    call settle_native_points()
    if (imode/=2) call release_native_cohort()
  end subroutine

  subroutine restore_generation_draw(draw,slot)
    type(mint_draw),intent(in) :: draw
    integer,intent(in) :: slot
    if (.not.allocated(lanes)) error stop 'Native generation cohort has been released'
    if (slot<1.or.slot>size(lanes)) error stop 'Invalid native event slot'
    if (lanes(slot)%physics%draw_id/=draw%id.or..not.draw%complete) error stop 'Incorrect native event owner'
    call enter_context(slot)
    call activate_mint_draw(draw)
    call activate_native_point(lanes(slot)%physics)
  end subroutine

  subroutine park_generation_draw(slot)
    integer,intent(in) :: slot
    call park_native_point(lanes(slot)%physics)
  end subroutine

  subroutine release_native_cohort()
    integer :: i
    if (allocated(lanes)) then
       call release_native_points()
       do i=1,size(lanes)
          call recycle_native_point(lanes(i)%physics)
       enddo
       deallocate(lanes)
    endif
    call release_contexts()
  end subroutine
end module
