module mc_native_generation
  use mint_module, only: ndimmax, mint_draw, generate_mc_trials, trace_mc_draw, &
       increase_gen_counters_before, increase_gen_counters_end
  use mc_native_cohort, only: evaluate_native_cohort, restore_generation_draw, park_generation_draw, release_native_cohort
  use mc_event_queues, only: event_queue, generation_stream, append_event, take_event, discard_surplus
  use mc_event_records, only: native_event_record, capture_event_record, write_event_record
  use mc_decision_ledger, only: decision_context,stage_output
  implicit none
  private
  public :: write_requested_event, finish_native_generation
  type(event_queue), save :: queues(4)
contains
  subroutine write_requested_event(kind,vn,capacity,event_weight,putonshell,p_label,lunlhe)
    character(*),intent(in) :: kind
    integer,intent(in) :: vn,capacity,p_label,lunlhe
    logical,intent(in) :: putonshell
    double precision,intent(in) :: event_weight
    integer :: stream,lane
    type(native_event_record),allocatable :: record
    type(mint_draw),allocatable :: draws(:)
    logical,allocatable :: accepted(:)
    double precision :: x(ndimmax),weight
    stream=generation_stream(kind,vn)
    call increase_gen_counters_before(vn)
    if (queues(stream)%size==0) allocate(draws(capacity),accepted(capacity))
    do while (queues(stream)%size==0)
       call generate_mc_trials(draws,evaluate_native_cohort,vn,accepted)
       do lane=1,capacity
          if (.not.accepted(lane)) cycle
          call restore_generation_draw(draws(lane),lane)
          call decision_context(draws(lane)%id,0,stage_output)
          call select_native_event(event_weight,x,weight)
          allocate(record)
          call capture_event_record(record,x,weight,putonshell,p_label)
          record%draw_id=draws(lane)%id
          call append_event(queues(stream),record,capacity)
          call park_generation_draw(lane)
       enddo
       call release_native_cohort()
    enddo
    call take_event(queues(stream),record)
    call write_event_record(record,lunlhe)
    call trace_mc_draw('WRITE')
    call increase_gen_counters_end(vn)
  end subroutine

  subroutine finish_native_generation()
    call discard_surplus(queues)
  end subroutine
end module
