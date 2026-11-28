module mc_event_queues
  use iso_fortran_env, only: int64
  use mc_event_records, only: native_event_record
  use mc_decision_ledger, only: decision_context,stage_output
  use mint_module, only: trace_mc_draw
  implicit none
  private
  public :: event_queue_base_bytes
  public :: event_queue, generation_stream, append_event, take_event, discard_surplus
  type :: event_slot
     type(native_event_record),allocatable :: record
  end type
  type :: event_queue
     type(event_slot),allocatable :: slots(:)
     integer :: size=0, first=1, last=0
     integer(int64) :: accepted=0, written=0, surplus=0
  end type
contains
  integer(int64) function event_queue_base_bytes() result(bytes)
    type(event_slot) :: slot
    type(native_event_record) :: record
    ! Up to four FIFO streams can retain records at the same time.
    bytes=4*(storage_size(slot,kind=int64)+storage_size(record,kind=int64))/8
  end function

  integer function generation_stream(kind,vn) result(stream)
    character(*),intent(in) :: kind
    integer,intent(in) :: vn
    select case(kind)
    case('virt')
       if (vn/=1.and.vn/=2) error stop 'Invalid envelope for virtual event'
       stream=vn
    case('novi')
       if (vn/=2) error stop 'Invalid envelope for nonvirtual event'
       stream=3
    case('born')
       if (vn/=3) error stop 'Invalid envelope for Born event'
       stream=4
    case default
       error stop 'Unknown native generation stream'
    end select
  end function

  subroutine append_event(queue,record,capacity)
    type(event_queue),intent(inout) :: queue
    type(native_event_record),allocatable,intent(inout) :: record
    integer,intent(in) :: capacity
    if (.not.allocated(record)) error stop 'Missing accepted event record'
    if (.not.allocated(queue%slots)) allocate(queue%slots(capacity))
    if (capacity/=size(queue%slots).or.queue%size>=capacity) error stop 'Native event queue capacity exceeded'
    queue%last=mod(queue%last,capacity)+1
    call move_alloc(record,queue%slots(queue%last)%record)
    queue%accepted=queue%accepted+1
    queue%size=queue%size+1
  end subroutine

  subroutine pop_event(queue,record)
    type(event_queue),intent(inout) :: queue
    type(native_event_record),allocatable,intent(out) :: record
    if (queue%size==0) error stop 'Empty native event queue'
    call move_alloc(queue%slots(queue%first)%record,record)
    queue%first=mod(queue%first,size(queue%slots))+1
    queue%size=queue%size-1
    call decision_context(record%draw_id,0,stage_output)
  end subroutine

  subroutine take_event(queue,record)
    type(event_queue),intent(inout) :: queue
    type(native_event_record),allocatable,intent(out) :: record
    call pop_event(queue,record)
    queue%written=queue%written+1
  end subroutine

  subroutine discard_surplus(queues)
    type(event_queue),intent(inout) :: queues(:)
    type(native_event_record),allocatable :: record
    integer :: i
    do i=1,size(queues)
       do while (queues(i)%size>0)
          call pop_event(queues(i),record)
          call trace_mc_draw('SURPLUS')
          queues(i)%surplus=queues(i)%surplus+1
       enddo
       if (queues(i)%accepted/=queues(i)%written+queues(i)%surplus) error stop 'Native accepted event lost'
       write(*,'(a,i0,3(1x,i0))') 'Native stream accepted/written/surplus ',i, &
            queues(i)%accepted,queues(i)%written,queues(i)%surplus
    enddo
  end subroutine
end module
