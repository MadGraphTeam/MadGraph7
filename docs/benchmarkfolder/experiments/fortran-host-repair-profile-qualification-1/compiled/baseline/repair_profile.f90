! Private source-only diagnostic. No floating arithmetic or environment writes.
module host_repair_profile_state
  use iso_fortran_env, only: int64
  implicit none
  private
  public :: repair_profile_reset, repair_profile_begin, repair_profile_end
  public :: repair_profile_flush, repair_profile_probe
  integer, save :: profile_mode=0, active(3)=0
  integer(int64), save :: begins(3)=0, ends(3)=0, ticks(3)=0
  integer(int64), save :: nested(3)=0, direct(3)=0
contains
  subroutine repair_profile_reset(mode)
    integer, intent(in) :: mode
    if (mode<0.or.mode>2) stop 61
    if (any(active/=0)) stop 62
    profile_mode=mode
    active=0
    begins=0
    ends=0
    ticks=0
    nested=0
    direct=0
  end subroutine

  subroutine repair_profile_begin(stage,tick)
    integer, intent(in) :: stage
    integer(int64), intent(out) :: tick
    tick=0
    if (profile_mode==0) return
    if (stage<1.or.stage>3) stop 63
    if (active(stage)/=0) stop 64
    if (stage==1.and.any(active/=0)) stop 65
    if (stage==2.and.active(3)/=0) stop 66
    if (stage==3.and.any(active/=0)) stop 67
    if (stage==2.and.active(1)==1) then
      nested(stage)=nested(stage)+1_int64
    else
      direct(stage)=direct(stage)+1_int64
    endif
    begins(stage)=begins(stage)+1_int64
    active(stage)=1
    if (profile_mode==2) call system_clock(tick)
  end subroutine

  subroutine repair_profile_end(stage,tick)
    integer, intent(in) :: stage
    integer(int64), intent(in) :: tick
    integer(int64) :: finish
    if (profile_mode==0) return
    if (profile_mode==2) call system_clock(finish)
    if (stage<1.or.stage>3) stop 68
    if (active(stage)/=1) stop 69
    if (stage==1.and.(active(2)/=0.or.active(3)/=0)) stop 70
    if (profile_mode==2) then
      if (tick<0_int64.or.finish<tick) stop 71
      ticks(stage)=ticks(stage)+(finish-tick)
    endif
    ends(stage)=ends(stage)+1_int64
    active(stage)=0
  end subroutine

  subroutine repair_profile_flush(filename)
    character(*), intent(in) :: filename
    integer(int64) :: rate
    integer :: stage,lun
    if (any(active/=0).or.any(begins/=ends)) stop 72
    if (any(begins/=nested+direct)) stop 73
    if (nested(1)/=0_int64.or.nested(3)/=0_int64) stop 74
    call system_clock(count_rate=rate)
    if (rate<=0_int64) stop 75
    open(newunit=lun,file=filename,status='replace',action='write')
    write(lun,'(A)') '# schema host-repair-profile-v1'
    write(lun,'(A,I0)') '# mode ',profile_mode
    write(lun,'(A,I0)') '# clock_rate ',rate
    write(lun,'(A)') '# fields stage begin_count end_count ticks nested_count direct_count'
    do stage=1,3
      write(lun,'(6(I0,1X))') stage,begins(stage),ends(stage),ticks(stage),nested(stage),direct(stage)
    enddo
    write(lun,'(A,3(1X,I0))') '# active',active
    close(lun)
  end subroutine

  subroutine repair_profile_probe(filename)
    ! Descriptive overhead only. Never subtract from scientific intervals.
    character(*), intent(in) :: filename
    integer, parameter :: repetitions=100000
    integer :: lun,prior_mode
    integer(int64) :: whole,rate
    prior_mode=profile_mode
    call repair_profile_reset(2)
    ! The loop must be compiled in check_sa.o, retaining module call boundaries.
    call repair_empty_probe_loop(repetitions,whole,rate)
    if (any(active/=0).or.begins(3)/=int(repetitions,int64)) stop 76
    if (any(begins/=ends).or.begins(1)/=0.or.begins(2)/=0) stop 77
    if (rate<=0_int64.or.whole<0_int64.or.ticks(3)<0_int64) stop 78
    open(newunit=lun,file=filename,status='replace',action='write')
    write(lun,'(A)') '# repetitions clock_rate whole_loop_ticks scoped_ticks'
    write(lun,'(4(I0,1X))') repetitions,rate,whole,ticks(3)
    close(lun)
    ! Flush the measured statistics before this routine; it resets counters.
    call repair_profile_reset(prior_mode)
  end subroutine
end module
