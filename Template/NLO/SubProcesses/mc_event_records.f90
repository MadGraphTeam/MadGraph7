! Event bytes are owned by an accepted MINT draw until its request queue emits
! them. Header, init and closing records remain owned by the job's LHE stream.
module mc_event_records
  use iso_fortran_env, only: int64
  use mc_context_memory, only: reserve_event_bytes,release_event_bytes
  implicit none
  private
  public :: native_event_record, capture_event_record, write_event_record
  public :: begin_event_capture, end_event_capture, emit_event_line
  integer, parameter, public :: event_capture_unit = -8323

  type :: native_event_record
     integer(int64) :: draw_id = 0_int64
     character(:), allocatable, private :: text
     integer(int64), private :: used = 0
     logical, private :: ready = .false.
   contains
     final :: release_record
  end type

  type(native_event_record), pointer, save :: active => null()

  interface
     subroutine capture_event_record(record,x,weight,putonshell,p_label)
       import native_event_record
       type(native_event_record), target, intent(inout) :: record
       double precision, intent(in) :: x(*), weight
       logical, intent(in) :: putonshell
       integer, intent(in) :: p_label
     end subroutine
  end interface

contains
  subroutine release_record(record)
    type(native_event_record),intent(inout) :: record
    if (allocated(record%text)) call release_event_bytes(len(record%text,kind=int64))
  end subroutine

  subroutine begin_event_capture(record)
    type(native_event_record), target, intent(inout) :: record
    if (associated(active)) error stop 'nested native LHE capture'
    if (record%ready) error stop 'overwriting an unwritten native LHE event'
    record%used = 0
    active => record
  end subroutine

  subroutine emit_event_line(unit,line)
    integer, intent(in) :: unit
    character(*), intent(in) :: line
    character(:), allocatable :: replacement
    integer(int64) :: first, last, capacity
    if (unit /= event_capture_unit) then
       write(unit,'(a)') line
       return
    endif
    if (.not.associated(active)) error stop 'missing native LHE capture'
    first = active%used+1
    last = first+len(line,kind=int64)
    capacity = 0
    if (allocated(active%text)) capacity = len(active%text,kind=int64)
    if (capacity < last) then
       call reserve_event_bytes(max(2048_int64,last,2*capacity))
       allocate(character(max(2048_int64,last,2*capacity)) :: replacement)
       if (active%used > 0) replacement(:active%used) = active%text(:active%used)
       call release_event_bytes(capacity)
       call move_alloc(replacement,active%text)
    endif
    active%text(first:last-1) = line
    active%text(last:last) = new_line(line)
    active%used = last
  end subroutine

  subroutine end_event_capture(record)
    type(native_event_record), target, intent(inout) :: record
    if (.not.associated(active,record)) error stop 'wrong native LHE event owner'
    if (record%used < len('  </event>')+1) error stop 'short native LHE event'
    if (index(adjustl(record%text(1:min(16_int64,record%used))),'<event') /= 1) error stop 'missing native LHE event start'
    if (record%text(record%used-len('  </event>'):record%used-1) /= '  </event>') error stop 'missing native LHE event end'
    record%ready = .true.
    nullify(active)
  end subroutine

  subroutine write_event_record(record,unit)
    type(native_event_record), intent(inout) :: record
    integer, intent(in) :: unit
    integer(int64) :: first, last
    if (.not.record%ready) error stop 'native LHE event already written or incomplete'
    first = 1
    do while (first <= record%used)
       last = first + index(record%text(first:record%used),new_line('a'),kind=int64)-1
       if (last < first) error stop 'invalid native LHE event record'
       write(unit,'(a)') record%text(first:last-1)
       first = last+1
    enddo
    call release_event_bytes(len(record%text,kind=int64))
    deallocate(record%text)
    record%used = 0
    record%ready = .false.
  end subroutine
end module
