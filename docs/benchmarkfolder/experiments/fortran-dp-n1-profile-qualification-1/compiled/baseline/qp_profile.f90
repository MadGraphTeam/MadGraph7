! Private diagnostic only; filename preserves the reviewed five-unit build.
module dp_profile_state
  use iso_fortran_env, only: int64
  implicit none
  private
  public :: dp_profile_reset,dp_profile_begin,dp_profile_start,dp_profile_end
  public :: dp_profile_native_status,dp_profile_finish,dp_profile_flush,dp_profile_probe
  integer, save :: dp_mode=0, dp_ct=0, dp_n=0, dp_rank=0, dp_status=-999
  integer(int64), save :: counts(8,0:2)=0, ticks(8,0:2)=0
  integer(int64), save :: requests(0:2)=0, native_calls(0:2)=0
  integer(int64), save :: valid(0:2)=0, failed(0:2)=0
contains
  subroutine dp_profile_reset(mode)
    integer, intent(in) :: mode
    if (mode<0.or.mode>2) stop 61
    dp_mode=mode
    dp_ct=0
    dp_n=0
    dp_rank=0
    dp_status=-999
    counts=0
    ticks=0
    requests=0
    native_calls=0
    valid=0
    failed=0
  end subroutine

  subroutine dp_profile_begin(ct,n,rank)
    integer, intent(in) :: ct,n,rank
    if (dp_mode==0) return
    if ((ct/=1.and.ct/=2).or.n/=3.or.rank/=2) stop 62
    dp_ct=ct
    dp_n=n
    dp_rank=rank
    dp_status=-999
    requests(ct)=requests(ct)+1
  end subroutine

  subroutine dp_profile_start(tick)
    integer(int64), intent(out) :: tick
    tick=0
    if (dp_mode==2) call system_clock(tick)
  end subroutine

  subroutine dp_profile_end(stage,tick,ct_override)
    integer, intent(in) :: stage
    integer(int64), intent(in) :: tick
    integer, intent(in), optional :: ct_override
    integer(int64) :: finish
    integer :: ct
    if (dp_mode==0) return
    ct=dp_ct
    if (present(ct_override)) ct=ct_override
    if (stage<1.or.stage>8.or.ct<0.or.ct>2) stop 63
    if (dp_mode==2) then
      call system_clock(finish)
      if (finish<tick) stop 64
      ticks(stage,ct)=ticks(stage,ct)+finish-tick
    endif
    counts(stage,ct)=counts(stage,ct)+1
  end subroutine

  subroutine dp_profile_native_status(status)
    integer, intent(in) :: status
    if (dp_mode==0) return
    dp_status=status
    native_calls(dp_ct)=native_calls(dp_ct)+1
  end subroutine

  subroutine dp_profile_finish(stable)
    logical, intent(in) :: stable
    if (dp_mode==0) return
    if (stable.and.dp_status==0) then
      valid(dp_ct)=valid(dp_ct)+1
    else
      failed(dp_ct)=failed(dp_ct)+1
    endif
  end subroutine

  subroutine dp_profile_flush(filename)
    character(*), intent(in) :: filename
    integer(int64) :: rate
    integer :: stage,ct,lun
    call system_clock(count_rate=rate)
    if (rate<=0) stop 65
    open(newunit=lun,file=filename,status='replace',action='write')
    write(lun,'(A,I0)') '# mode ',dp_mode
    write(lun,'(A,I0)') '# clock_rate ',rate
    write(lun,'(A)') '# stage ct count ticks; stage3 contains stage4'
    do ct=0,2
      write(lun,'(A,5(1X,I0))') '# requests/native/valid/failed', &
        ct,requests(ct),native_calls(ct),valid(ct),failed(ct)
      do stage=1,8
        if (counts(stage,ct)>0) &
          write(lun,'(4(I0,1X))') stage,ct,counts(stage,ct),ticks(stage,ct)
      enddo
    enddo
    close(lun)
  end subroutine

  subroutine dp_profile_probe(filename)
    ! Descriptive probe overhead only. Never subtract from physics intervals.
    character(*), intent(in) :: filename
    integer, parameter :: repetitions=100000
    integer :: lun,prior_mode
    integer(int64) :: whole,rate
    prior_mode=dp_mode
    call dp_profile_reset(2)
    dp_ct=1
    dp_n=3
    dp_rank=2
    ! The loop is in check_sa.o, so the separate module-call boundary remains.
    call dp_empty_probe_loop(repetitions,whole,rate)
    open(newunit=lun,file=filename,status='replace',action='write')
    write(lun,'(A)') '# repetitions clock_rate whole_loop_ticks scoped_ticks'
    write(lun,'(4(I0,1X))') repetitions,rate,whole,ticks(8,1)
    close(lun)
    call dp_profile_reset(prior_mode)
  end subroutine
end module
