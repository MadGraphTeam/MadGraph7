! Private diagnostic source proposal. Not compiled or numerically qualified.
! No floating arithmetic, formatting, cache calls, or IEEE state restoration.
module dp_cause_observer
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_size_t, c_char
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  implicit none
  private
  public :: dco_emit, dco_begin, dco_end, dco_setup, dco_close
  integer, save :: out_unit=0, seq_id=0, point_id=0, phase_id=0, loop_id=0
  integer(c_int64_t), save :: entry(11)=0, local_faults=0
  integer(c_int64_t), allocatable, save :: input_words(:)
  interface
    function obs_open(path,n,enabled) bind(C,name='ml_tirust_observer_open') result(status)
      import c_char,c_size_t,c_int
      character(c_char), intent(in) :: path(*)
      integer(c_size_t), value :: n
      integer(c_int), value :: enabled
      integer(c_int) :: status
    end function
    subroutine obs_begin(point,phase) bind(C,name='ml_tirust_observer_begin_point')
      import c_int64_t,c_int
      integer(c_int64_t), value :: point
      integer(c_int), value :: phase
    end subroutine
    subroutine obs_event(tag,words,n) bind(C,name='ml_tirust_observer_event')
      import c_int,c_int64_t,c_size_t
      integer(c_int), value :: tag
      integer(c_int64_t), intent(in) :: words(*)
      integer(c_size_t), value :: n
    end subroutine
    function obs_flush() bind(C,name='ml_tirust_observer_flush') result(status)
      import c_int
      integer(c_int) :: status
    end function
    function obs_close() bind(C,name='ml_tirust_observer_close') result(status)
      import c_int
      integer(c_int) :: status
    end function
    function obs_health() bind(C,name='ml_tirust_observer_health') result(status)
      import c_int64_t
      integer(c_int64_t) :: status
    end function
  end interface
contains
  subroutine state(words)
    integer(c_int64_t), intent(out) :: words(11)
    type(ieee_flag_type) :: flags(5)
    type(ieee_round_type) :: rounding
    logical :: value
    integer :: i
    flags=[ieee_invalid,ieee_divide_by_zero,ieee_overflow,ieee_underflow,ieee_inexact]
    do i=1,5
      call ieee_get_flag(flags(i),value)
      words(i)=merge(1_c_int64_t,0_c_int64_t,value)
      call ieee_get_halting_mode(flags(i),value)
      words(i+5)=merge(1_c_int64_t,0_c_int64_t,value)
    enddo
    call ieee_get_rounding_mode(rounding)
    words(11)=-1
    if (rounding==ieee_nearest) words(11)=0
    if (rounding==ieee_to_zero) words(11)=1
    if (rounding==ieee_up) words(11)=2
    if (rounding==ieee_down) words(11)=3
  end subroutine

  subroutine dco_emit(tag,words)
    integer, intent(in) :: tag
    integer(c_int64_t), intent(in) :: words(:)
    integer(c_int64_t) :: before(11),after(11)
    call state(before)
    call obs_event(int(tag,c_int),words,int(size(words),c_size_t))
    call state(after)
    if (any(before/=after)) local_faults=local_faults+1
  end subroutine

  subroutine dco_setup()
    character(len=16) :: enabled
    character(kind=c_char,len=*), parameter :: journal='cause-native.jsonl'
    integer :: ios
    integer(c_int) :: status, mode
    call get_environment_variable('DCO_OBSERVE',enabled,status=ios)
    if (ios/=0.or.(trim(enabled)/='0'.and.trim(enabled)/='1')) stop 81
    mode=0
    if (trim(enabled)=='1') mode=1
    open(newunit=out_unit,file='cause-public.txt',status='new',action='write',iostat=ios)
    if (ios/=0) stop 82
    write(out_unit,'(A)') 'DCO_PUBLIC 1 flags:invalid,divzero,overflow,underflow,inexact;'// &
      'traps:same;rounding:nearest0,zero1,up2,down3'
    status=obs_open(journal,int(len(journal),c_size_t),mode)
    if (status/=0) stop 83
  end subroutine

  subroutine dco_begin(seq,point,phase,loop,words)
    integer, intent(in) :: seq,point,phase,loop
    integer(c_int64_t), intent(in) :: words(:)
    integer(c_int64_t) :: before(11),after(11)
    seq_id=seq; point_id=point; phase_id=phase; loop_id=loop
    ! Save all input bit words outside the amplitude; preserve their ordering.
    if (allocated(input_words)) deallocate(input_words)
    allocate(input_words(size(words)))
    input_words=words
    call state(before)
    call obs_begin(int(point,c_int64_t),int(phase,c_int))
    call dco_emit(10,int([seq,point,phase,loop],c_int64_t))
    call dco_emit(11,words)
    call state(after)
    if (any(before/=after)) local_faults=local_faults+1
    entry=after
  end subroutine

  subroutine dco_end(code,answers,accuracies)
    integer, intent(in) :: code
    integer(c_int64_t), intent(in) :: answers(:),accuracies(:)
    integer(c_int64_t) :: leave(11),after(11)
    integer(c_int) :: status
    integer :: ios
    ! Capture immediately after SLOOPMATRIX; never read unassigned internal arrays.
    call state(leave)
    call dco_emit(12,int([code,size(answers),size(accuracies)],c_int64_t))
    call dco_emit(13,answers)
    call dco_emit(14,accuracies)
    call state(after)
    if (any(leave/=after)) local_faults=local_faults+1
    write(out_unit,'(A,8(1X,I0))',iostat=ios) 'PUBLIC',seq_id,point_id,phase_id,loop_id, &
      code,size(input_words),size(answers),size(accuracies)
    if (ios/=0) stop 84
    write(out_unit,'(A,*(1X,Z16.16))',iostat=ios) 'INPUT',input_words
    if (ios/=0) stop 84
    write(out_unit,'(A,*(1X,Z16.16))',iostat=ios) 'ANS',answers
    if (ios/=0) stop 84
    write(out_unit,'(A,*(1X,Z16.16))',iostat=ios) 'ACC',accuracies
    if (ios/=0) stop 84
    write(out_unit,'(A,*(1X,I0))',iostat=ios) 'STATE',entry,leave
    if (ios/=0) stop 84
    status=obs_flush()
    write(out_unit,'(A,3(1X,I0))',iostat=ios) 'HEALTH',local_faults,obs_health(),status
    if (ios/=0) stop 84
    flush(out_unit,iostat=ios)
    if (ios/=0.or.status/=0.or.local_faults/=0.or.obs_health()/=0) stop 85
  end subroutine

  subroutine dco_close()
    integer(c_int) :: status
    integer :: ios
    status=obs_close()
    if (status/=0.or.obs_health()/=0.or.local_faults/=0) stop 86
    write(out_unit,'(A,4(1X,I0))',iostat=ios) 'CLOSE',seq_id,point_id,local_faults,obs_health()
    if (ios/=0) stop 87
    close(out_unit,iostat=ios)
    if (ios/=0) stop 87
  end subroutine
end module
