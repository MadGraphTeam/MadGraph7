! Private observation-only experiment; never part of a production export.
module cache_occupancy_observer
  use, intrinsic :: iso_c_binding, only: c_int64_t
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use mtirust, only: tirust_context_generation, tirust_cache_statistics, &
    tirust_qp_cache_statistics, TIRUST_SUCCESS
  implicit none
  private
  public :: occ_setup, occ_begin, occ_sample, occ_state, occ_flush
  integer, parameter :: capacity=65536, width=53
  integer(c_int64_t), save :: records(width,capacity)
  integer, save :: used=0, dropped=0, faults=0, public_call=-1
  logical, save :: enabled=.false.
contains
  subroutine occ_state(words)
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

  subroutine occ_setup()
    character(len=8) :: value
    integer :: status
    call get_environment_variable('OCC_OBSERVE',value,status=status)
    if (status/=0.or.(trim(value)/='0'.and.trim(value)/='1')) stop 81
    enabled=trim(value)=='1'
  end subroutine

  subroutine occ_begin(call_index)
    integer, intent(in) :: call_index
    public_call=call_index
  end subroutine

  subroutine occ_sample(tag,precision,ctmode,n,rank,status,poles,finite,check,cache)
    ! tag 10=evaluate return; 20=before existing clear; 21=after existing clear.
    ! No query after failed evaluation: queries would erase its public error.
    ! Clear status is intentionally unknown (-999); its original call is intact.
    integer, intent(in) :: tag,precision,ctmode,n,rank,status,poles
    logical, intent(in) :: finite,check,cache
    integer(c_int64_t) :: aggregate(9),qp(6),before(11),after(11)
    integer :: generation,s1,s2,queried
    if (.not.enabled) return
    call occ_state(before)
    generation=tirust_context_generation()
    aggregate=0; qp=0; s1=-999; s2=-999; queried=0
    if (generation>0.and.(tag/=10.or.status==TIRUST_SUCCESS)) then
      queried=1
      call tirust_cache_statistics(aggregate,s1)
      call tirust_qp_cache_statistics(qp,s2)
      if (s1/=TIRUST_SUCCESS.or.s2/=TIRUST_SUCCESS) faults=faults+1
    endif
    call occ_state(after)
    if (any(before/=after)) faults=faults+1
    if (used==capacity) then
      dropped=dropped+1
      return
    endif
    used=used+1
    records(:,used)=[int([used,public_call,tag,precision,ctmode,n,rank,status,poles, &
      merge(1,0,finite),merge(1,0,check),merge(1,0,cache),generation,queried,s1,s2],c_int64_t), &
      aggregate,qp,before,after]
  end subroutine

  subroutine occ_flush()
    integer :: unit,i,ios
    open(newunit=unit,file='occupancy.txt',status='new',action='write',iostat=ios)
    if (ios/=0) stop 82
    write(unit,'(A,5(1X,I0))') 'OCCUPANCY',1,merge(1,0,enabled),used,dropped,faults
    do i=1,used
      write(unit,'(*(I0,1X))') records(:,i)
    enddo
    close(unit,iostat=ios)
    if (ios/=0.or.dropped/=0.or.faults/=0) stop 83
  end subroutine
end module
