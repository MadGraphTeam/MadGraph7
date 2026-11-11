! Private, serial diagnostic instrumentation; never an acceptance timer/cache.
! No MTIRUST dependency: callers supply snapshots without creating a context.
! Precision: 0 lifecycle, 1 DP, 2 binary128/MPFR256. Stages: 1 init/budget,
! 2 kinematics, 3 allocation/input decimal, 4 inclusive FFI, 5 output parse/map,
! 6 QP contraction, 7 cache clear. Intervals may nest: do not sum them blindly.
! Snapshot phases: 1 before FFI, 2 after FFI, 3 immediately before real clear.
! Mode 1 retains metadata in memory; all formatted output is after the point.
! Mode 2 additionally copies caller TRANSFER bytes, without FP operations.
module qp_profile_state
  use, intrinsic :: iso_fortran_env, only: int8, int64
  implicit none
  private
  integer, parameter, public :: qp_profile_byte=int8
  integer, public, protected :: qp_profile_mode=0
  integer, public :: qp_profile_native_status=-999
  public :: qp_profile_reset, qp_profile_begin, qp_profile_start, qp_profile_end
  public :: qp_profile_stats, qp_profile_finish, qp_profile_blob, qp_profile_write
  type :: entry
    integer :: kind=0, key(4)=0, options(3)=0, generation=-1, status=0, valid=-1
    integer(int64) :: call_id=0, epoch=0, budget=0, stats(6)=0, offset=0, length=0
    character(40) :: label='-'
  end type
  type(entry), allocatable :: records(:)
  integer(int8), allocatable :: payload(:)
  type(entry) :: current
  integer :: used=0
  integer(int64) :: payload_used=0, epoch=0, serial=0, rate=0
  integer(int64) :: ticks(7,0:9,0:6,0:6,0:2)=0, calls(7,0:9,0:6,0:6,0:2)=0
  logical :: in_call=.false.
contains
  subroutine qp_profile_reset(mode,trace_budget_bytes)
    integer, intent(in) :: mode
    integer(int64), intent(in), optional :: trace_budget_bytes
    integer(int64) :: budget, record_bytes, count, remaining
    integer :: ios
    if (in_call) error stop 'QP profile: reset inside call'
    if (mode<0.or.mode>2) error stop 'QP profile: invalid mode'
    budget=67108864_int64
    if (present(trace_budget_bytes)) budget=trace_budget_bytes
    if (budget<1.or.budget>1073741824_int64) error stop 'QP profile: invalid trace budget'
    if (allocated(records)) deallocate(records)
    if (allocated(payload)) deallocate(payload)
    qp_profile_mode=mode
    used=0; payload_used=0; serial=0; epoch=0
    current=entry(); ticks=0; calls=0
    qp_profile_native_status=-999
    call system_clock(count_rate=rate)
    if (rate<=0) error stop 'QP profile: unavailable wall clock'
    if (mode==0) return
    record_bytes=int(storage_size(current)/8,int64)
    count=budget/record_bytes
    if (mode==2) count=(budget/2)/record_bytes
    if (count<1) error stop 'QP profile: budget cannot hold metadata'
    remaining=budget-count*record_bytes
    if (mode==1) remaining=0
    allocate(records(int(count)),payload(int(remaining)),stat=ios)
    if (ios/=0) error stop 'QP profile: allocation failed'
  end subroutine

  subroutine append(kind)
    integer, intent(in) :: kind
    if (used>=size(records)) error stop 'QP profile: metadata budget exceeded'
    used=used+1
    records(used)=current
    records(used)%kind=kind
    records(used)%epoch=epoch
  end subroutine

  subroutine qp_profile_begin(ctmode,n,rank,precision,poles,finite,cache,budget,contextgen)
    integer, intent(in) :: ctmode,n,rank,precision,poles,contextgen
    logical, intent(in) :: finite,cache
    integer(int64), intent(in) :: budget
    if (qp_profile_mode==0) return
    if (in_call) error stop 'QP profile: nested tensor call'
    if (ctmode<0.or.ctmode>9.or.n<1.or.n>6.or.rank<0.or.rank>6) &
      error stop 'QP profile: unsupported key'
    if (precision<1.or.precision>2.or.poles<0.or.poles>3) error stop 'QP profile: invalid policy'
    if (budget<0) error stop 'QP profile: negative cache budget'
    serial=serial+1
    qp_profile_native_status=-999
    current=entry()
    current%call_id=serial; current%key=[ctmode,n,rank,precision]
    current%options=[poles,merge(1,0,finite),merge(1,0,cache)]
    current%budget=budget; current%generation=contextgen
    in_call=.true.
    call append(1)
  end subroutine

  subroutine qp_profile_start(start)
    integer(int64), intent(out) :: start
    start=0
    if (qp_profile_mode==1) call system_clock(start)
  end subroutine

  subroutine qp_profile_end(stage,start)
    integer, intent(in) :: stage
    integer(int64), intent(in) :: start
    integer(int64) :: finish,delta
    integer :: a,b,c,d
    if (qp_profile_mode/=1) return
    call system_clock(finish)
    if (stage<1.or.stage>7.or.finish<start) error stop 'QP profile: invalid timer interval'
    delta=finish-start
    a=current%key(1); b=current%key(2); c=current%key(3); d=current%key(4)
    if (stage==7.and..not.in_call) then
      a=0; b=0; c=0; d=0
    endif
    if (delta>huge(delta)-ticks(stage,a,b,c,d)) error stop 'QP profile: tick overflow'
    ticks(stage,a,b,c,d)=ticks(stage,a,b,c,d)+delta
    calls(stage,a,b,c,d)=calls(stage,a,b,c,d)+1
  end subroutine

  subroutine qp_profile_stats(phase,counts,stats_status,contextgen)
    integer, intent(in) :: phase,stats_status,contextgen
    integer(int64), intent(in) :: counts(6)
    type(entry) :: saved
    if (qp_profile_mode==0) return
    if (phase<1.or.phase>3) error stop 'QP profile: invalid snapshot phase'
    if (phase/=3.and..not.in_call) error stop 'QP profile: snapshot outside tensor call'
    saved=current
    if (phase==3.and..not.in_call) current=entry()
    current%stats=counts; current%status=stats_status; current%generation=contextgen
    if (phase==3) then
      call append(5)
      epoch=epoch+1
    else
      call append(phase+1)
    endif
    current=saved
  end subroutine

  subroutine qp_profile_finish(status,valid)
    integer, intent(in) :: status
    logical, intent(in) :: valid
    if (qp_profile_mode==0) return
    if (.not.in_call) error stop 'QP profile: finish outside tensor call'
    current%status=status; current%valid=merge(1,0,valid)
    call append(4)
    in_call=.false.
    ! Retain the key for the wrapper's subsequent QP contraction timer.
  end subroutine

  subroutine qp_profile_blob(label,bytes)
    character(*), intent(in) :: label
    integer(int8), intent(in) :: bytes(:)
    integer(int64) :: count
    if (qp_profile_mode/=2) return
    if (.not.in_call) error stop 'QP profile: blob outside tensor call'
    if (len_trim(label)<1.or.len_trim(label)>40) error stop 'QP profile: invalid blob label'
    if (verify(trim(label),'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')/=0) &
      error stop 'QP profile: invalid blob label characters'
    count=size(bytes,kind=int64)
    if (count>size(payload,kind=int64)-payload_used) error stop 'QP profile: payload budget exceeded'
    call append(6)
    records(used)%offset=payload_used; records(used)%length=count; records(used)%label=label
    payload(payload_used+1:payload_used+count)=bytes
    payload_used=payload_used+count
  end subroutine

  subroutine qp_profile_write(prefix)
    character(*), intent(in) :: prefix
    integer :: unit,ios,a,b,c,d,s,i,k,last
    if (in_call) error stop 'QP profile: output inside tensor call'
    open(newunit=unit,file=prefix//'.timers.tsv',status='new',action='write',iostat=ios)
    if (ios/=0) error stop 'QP profile: cannot create timer evidence'
    write(unit,'(A,I0)') '# mode ',qp_profile_mode
    write(unit,'(A,I0)') '# clock_rate ',rate
    write(unit,'(A)') '# stage ctmode n rank precision calls ticks'
    do d=0,2
      do c=0,6
        do b=0,6
          do a=0,9
            do s=1,7
              if (calls(s,a,b,c,d)>0) write(unit,'(*(I0,1X))') s,a,b,c,d,calls(s,a,b,c,d),ticks(s,a,b,c,d)
            enddo
          enddo
        enddo
      enddo
    enddo
    close(unit,iostat=ios)
    if (ios/=0) error stop 'QP profile: timer evidence close failed'
    if (qp_profile_mode==0) return
    open(newunit=unit,file=prefix//'.calls.tsv',status='new',action='write',iostat=ios)
    if (ios/=0) error stop 'QP profile: cannot create call evidence'
    write(unit,'(A)') '# kinds 1begin 2beforeFFI 3afterFFI 4finish 5beforeClear 6blob; epoch is pre-clear'
    write(unit,'(A)') '# row kind call epoch ctmode n rank precision poles finite cache budget generation '// &
      'status valid hits misses entries bytes not_stored bypassed offset length label'
    do i=1,used
      write(unit,'(*(G0,1X))') i,records(i)%kind,records(i)%call_id,records(i)%epoch,records(i)%key, &
        records(i)%options,records(i)%budget,records(i)%generation,records(i)%status,records(i)%valid, &
        records(i)%stats,records(i)%offset,records(i)%length,trim(records(i)%label)
    enddo
    close(unit,iostat=ios)
    if (ios/=0) error stop 'QP profile: call evidence close failed'
    if (qp_profile_mode/=2) return
    open(newunit=unit,file=prefix//'.shadow.hex',status='new',action='write',iostat=ios)
    if (ios/=0) error stop 'QP profile: cannot create bit evidence'
    write(unit,'(A)') '# row global_zero_based_byte_offset memory_order_hex; exact TRANSFER bytes'
    do i=1,used
      if (records(i)%kind/=6) cycle
      last=int(records(i)%offset+records(i)%length)
      do k=int(records(i)%offset)+1,last,32
        write(unit,'(I0,1X,I0,1X,*(Z2.2))') i,k-1,payload(k:min(k+31,last))
      enddo
    enddo
    close(unit,iostat=ios)
    if (ios/=0) error stop 'QP profile: bit evidence close failed'
  end subroutine
end module
