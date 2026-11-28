! Host storage owned by live native points and queued event records. The
! budget is inactive for the legacy width-one/FO path and offline utilities.
module mc_context_memory
  use iso_fortran_env, only: int64,error_unit
  implicit none
  private
  public :: configure_context_memory, enter_context, release_contexts, reserve_context, &
       reserve_event_bytes, release_event_bytes, report_context_memory, memory_enabled, &
       weight_store, provider_store
  integer,parameter :: weight_store=1,provider_store=2
  integer(int64),parameter :: mib=1048576_int64
  integer(int64),allocatable,save :: owned(:,:)
  integer(int64),save :: ceiling=0,fixed=0,committed=0,peak=0,lane_peak=0,events=0
  integer,save :: current=0
  logical,save :: memory_enabled=.false.
contains
  subroutine configure_context_memory(width,bytes_per_lane)
    integer,intent(in) :: width
    integer(int64),intent(in) :: bytes_per_lane
    character(64) :: value
    integer :: status
    integer(int64) :: limit_mib,per_lane
    if (width<2.or.bytes_per_lane<=0.or.memory_enabled) error stop 'Invalid native memory configuration'
    call get_environment_variable('MG7_NLO_BATCH_MEMORY_MIB',value,status=status)
    limit_mib=1024
    if (status/=1) then
       if (status/=0.or.len_trim(value)==0.or.verify(trim(value),'0123456789')/=0) &
            error stop 'MG7_NLO_BATCH_MEMORY_MIB requires a positive integer'
       read(value,*,iostat=status) limit_mib
       if (status/=0.or.limit_mib<=0.or.limit_mib>huge(ceiling)/mib) error stop 'Invalid MG7_NLO_BATCH_MEMORY_MIB'
    endif
    ceiling=limit_mib*mib
    per_lane=bytes_per_lane+2*int(storage_size(0_int64)/8,int64)
    if (per_lane>ceiling/width) then
       write(error_unit,'(a,i0,a,i0,a,i0)') 'Native MC memory: width=',width,', bytes per context=',per_lane, &
            ', ceiling bytes=',ceiling
       error stop 'Native MC context reservation exceeds MG7_NLO_BATCH_MEMORY_MIB; reduce vector_size or raise the ceiling'
    endif
    fixed=per_lane*width
    committed=fixed
    peak=fixed
    lane_peak=per_lane
    allocate(owned(2,width),stat=status)
    if(status/=0) error stop 'Unable to allocate native MC memory accounting'
    owned=0
    memory_enabled=.true.
    write(*,'(a,i0,a,i0,a,i0)') 'Native MC memory: capacity=',width,', base bytes per context=',per_lane,', ceiling bytes=',ceiling
  end subroutine

  subroutine enter_context(lane)
    integer,intent(in) :: lane
    if (.not.memory_enabled) return
    if(lane<1.or.lane>size(owned,2)) error stop 'Invalid native memory owner'
    current=lane
  end subroutine

  subroutine check_room(extra)
    integer(int64),intent(in) :: extra
    if(extra<0.or.extra>ceiling-committed) then
       write(error_unit,'(3(a,i0))') 'Native MC memory: reserved bytes=',committed,', additional bytes=',extra,', ceiling=',ceiling
       error stop 'Native MC allocation exceeds MG7_NLO_BATCH_MEMORY_MIB; reduce vector_size or raise the ceiling'
    endif
    peak=max(peak,committed+extra)
  end subroutine

  subroutine reserve_context(kind,bytes,temporary)
    integer,intent(in) :: kind
    integer(int64),intent(in) :: bytes
    integer(int64),intent(in),optional :: temporary
    integer(int64) :: growth,extra
    if (.not.memory_enabled.or.current==0) return
    if (bytes<0) error stop 'Native MC context size overflow'
    growth=max(0_int64,bytes-owned(kind,current))
    extra=0
    if(present(temporary))extra=temporary
    call check_room(growth)
    committed=committed+growth
    call check_room(extra)
    owned(kind,current)=owned(kind,current)+growth
    lane_peak=max(lane_peak,fixed/size(owned,2)+sum(owned(:,current))+extra)
  end subroutine

  subroutine release_contexts()
    if (.not.memory_enabled) return
    committed=committed-sum(owned)
    owned=0
    current=0
  end subroutine

  subroutine reserve_event_bytes(bytes)
    integer(int64),intent(in) :: bytes
    if (.not.memory_enabled) return
    call check_room(bytes)
    committed=committed+bytes
    events=events+bytes
  end subroutine

  subroutine release_event_bytes(bytes)
    integer(int64),intent(in) :: bytes
    if (.not.memory_enabled) return
    if(bytes<0.or.bytes>events) error stop 'Invalid native event memory release'
    events=events-bytes
    committed=committed-bytes
  end subroutine

  subroutine report_context_memory()
    if (.not.memory_enabled) return
    if(events/=0.or.any(owned/=0)) error stop 'Native MC memory still owned at job termination'
    write(*,'(a,i0,a,i0,a,i0)') 'Native MC memory: peak reserved bytes=',peak, &
         ', largest live context reservation bytes=',lane_peak,', ceiling bytes=',ceiling
  end subroutine
end module
