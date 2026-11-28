! Opt-in, job-local validation of the path from a MINT draw to its physical
! matrix-element input and back to its consumer. Every real field is written
! as its binary64 bits so that even a changed momentum component is visible.
module mc_batch_trace
  use iso_c_binding, only: c_int, c_int32_t, c_double, c_size_t
  use iso_fortran_env, only: int64, real64
  implicit none
  private
  public :: batch_trace_unit, trace_request, trace_bind, trace_result, trace_consume, trace_cancel
  public :: trace_fortran_input, trace_fortran_result, trace_umami_input, trace_umami_output, trace_mapping
  public :: real_profile_active, mc_real_profile_open, mc_real_profile_close, mc_real_profile_step
  logical, protected, save :: real_profile_active=.false.
  public :: mc_oracle_sample,mc_oracle_report
  integer,save :: oracle_period=0,oracle_seen=0
  logical,save :: oracle_record_only=.false.
  integer, save :: profile_mode=-1
  integer(int64), allocatable, save :: profile_occupancy(:)
  integer(int64), save :: profile_native(3)=0, profile_ticks(3)=0, profile_started=0

  integer, save :: unit=-1
  logical, save :: initialized=.false.
  integer(int64), save :: next_request=0, next_call=0
  integer, save :: active_lanes=0

contains
  subroutine mc_real_profile_open(mode)
    integer,intent(in) :: mode
    character(8) :: setting
    integer :: status
    call get_environment_variable('MG7_MC_REAL_ORACLE_PERIOD',setting,status=status)
    if(status==0.and.setting/=' ')then
       read(setting,*,iostat=status) oracle_period
       if(status/=0.or.oracle_period<0) error stop 'MG7_MC_REAL_ORACLE_PERIOD must be a nonnegative integer'
    endif
    oracle_seen=0
    call get_environment_variable('MG7_MC_REAL_ORACLE_RECORD_ONLY',setting,status=status)
    oracle_record_only=status==0.and.setting=='1'
    call get_environment_variable('MG7_MC_BATCH_PROFILE',setting,status=status)
    real_profile_active=status==0.and.setting=='1'
    if(.not.real_profile_active) return
    profile_mode=mode
    profile_native=0
    profile_ticks=0
    if(allocated(profile_occupancy)) deallocate(profile_occupancy)
    allocate(profile_occupancy(1),source=0_int64)
  end subroutine

  logical function mc_oracle_sample() result(check)
    check=.false.
    if(oracle_period==0)return
    check=mod(oracle_seen,oracle_period)==0
    oracle_seen=oracle_seen+1
  end function

  subroutine mc_oracle_report(owner,row,total,components,reference,reference_components)
    use ieee_arithmetic, only: ieee_is_finite
    integer(int64),intent(in) :: owner
    integer,intent(in) :: row
    real(real64),intent(in) :: total,components(:),reference,reference_components(:)
    real(real64) :: ratio,limit
    integer :: slot
    logical :: passed
    passed=all(ieee_is_finite([total,components,reference,reference_components]))
    limit=1d-30+1d-12*abs(reference)
    ratio=abs(total-reference)/limit
    if(.not.(abs(total-reference)<=limit)) passed=.false.
    do slot=1,size(components)
       limit=1d-30+1d-12*abs(reference_components(slot))
       ratio=max(ratio,abs(components(slot)-reference_components(slot))/limit)
       if(.not.(abs(components(slot)-reference_components(slot))<=limit)) passed=.false.
    enddo
    write(*,'(a,2(1x,i0),1x,a,3(1x,es24.16))') 'MG7_REAL_ORACLE owner/row/status/provider/reference/maxratio', &
      owner,row,merge('PASS','FAIL',passed),total,reference,ratio
    if(.not.passed)then
       write(*,'(a,*(1x,es24.16))') 'MG7_REAL_ORACLE_COMPONENTS provider',components
       write(*,'(a,*(1x,es24.16))') 'MG7_REAL_ORACLE_COMPONENTS reference',reference_components
       if(.not.oracle_record_only) error stop 'Native/UMAMI real oracle discrepancy'
    endif
  end subroutine

  subroutine mc_real_profile_close()
    integer :: i
    integer(int64) :: rate
    if(.not.real_profile_active) return
    call system_clock(count_rate=rate)
    do i=1,size(profile_occupancy)
       if(profile_occupancy(i)>0) write(*,'(a,3(1x,i0))') 'MG7_PROFILE_UMAMI mode/width/calls',profile_mode,i,profile_occupancy(i)
    enddo
    write(*,'(a,4(1x,i0))') 'MG7_PROFILE_FORTRAN mode/configured/diagrams/unavailable',profile_mode,profile_native
    write(*,'(a,5(1x,i0))') 'MG7_PROFILE_CLOCK mode/rate/pack/provider/scatter',profile_mode,rate,profile_ticks
    real_profile_active=.false.
  end subroutine

  subroutine mc_real_profile_step(stage,begin)
    integer,intent(in) :: stage
    logical,intent(in) :: begin
    integer(int64) :: now
    if(.not.real_profile_active) return
    call system_clock(now)
    if(begin) then
       profile_started=now
    else
       profile_ticks(stage)=profile_ticks(stage)+now-profile_started
    endif
  end subroutine

  integer function batch_trace_unit() result(output)
    integer :: length, status
    character(:), allocatable :: path
    character(48) :: automatic_path
    interface
       function getpid() bind(C,name='getpid') result(pid)
         import :: c_int
         integer(c_int) :: pid
       end function
    end interface
    if (.not.initialized) then
       initialized=.true.
       call get_environment_variable('MG7_MC_BATCH_TRACE',length=length,status=status)
       if (status==0 .and. length>0) then
          allocate(character(length)::path)
          call get_environment_variable('MG7_MC_BATCH_TRACE',path)
          if (path=='1') then
             write(automatic_path,'(a,i0,a)') 'mc_batch_',getpid(),'.trace'
             path=trim(automatic_path)
          end if
          if (path/='0') then
             open(newunit=unit,file=path,status='new',action='write')
             write(unit,'(a)') 'MG7 MC batch trace v1'
          end if
       end if
    end if
    output=unit
  end function

  subroutine trace_request(draw,fold,sector,flavour,p,g,ew,request)
    integer(int64), intent(in) :: draw
    integer, intent(in) :: fold,sector,flavour
    real(real64), intent(in) :: p(:,:),g,ew
    integer(int64), intent(out) :: request
    integer :: leg,mu
    request=0
    if(batch_trace_unit()==-1 .or. draw==0) return
    next_request=next_request+1
    request=next_request
    write(unit,'(a,5(1x,i0),*(1x,z16.16))') 'REQUEST',draw,request,fold,sector,flavour, &
      transfer(g,0_int64),transfer(ew,0_int64),((transfer(p(mu,leg),0_int64),leg=1,size(p,2)),mu=1,size(p,1))
  end subroutine

  ! Called directly before dispatch with the densely packed request IDs.
  subroutine trace_bind(requests)
    integer(int64), intent(in) :: requests(:)
    if(batch_trace_unit()==-1 .or. all(requests==0)) return
    if(any(requests==0)) error stop 'Mixed setup and production real requests'
    if(active_lanes/=0) error stop 'Previous real results still pending'
    active_lanes=size(requests)
    next_call=next_call+1
    write(unit,'(a,*(1x,i0))') 'BIND',next_call,requests
  end subroutine

  subroutine trace_cancel()
    if(active_lanes==0) return
    write(unit,'(a,1x,i0,1x,a)') 'CANCEL',next_call,'first_use_unavailable'
    active_lanes=0
  end subroutine

  ! Called at the actual UMAMI boundary with its flattened SoA momentum
  ! buffer. The logical active count is distinct from the arithmetic width.
  subroutine trace_umami_input(count,legs,me,p,g,ew,f) bind(C,name='mg7_trace_umami_input')
    integer(c_size_t), value :: count
    integer(c_int), value :: legs,me
    integer(c_int32_t), intent(in) :: f(*)
    real(c_double), intent(in) :: p(*),g(*)
    real(c_double), value :: ew
    integer(c_size_t) :: lane,j
    if(real_profile_active) call mc_real_profile_step(2,.true.)
    if(active_lanes==0) return
    write(unit,'(a,3(1x,i0),1x,a)') 'PROVIDER',next_call,count,me,'umami'
    do lane=1,count
       write(unit,'(a,3(1x,i0),*(1x,z16.16))') 'INPUT',next_call,lane,f(lane)+1,transfer(g(lane),0_int64), &
         transfer(ew,0_int64),(transfer(p(j*count+lane),0_int64),j=0,4*legs-1)
    end do
  end subroutine

  subroutine trace_fortran_input(me,flavour,p,g,ew)
    integer, intent(in) :: me,flavour
    real(real64), intent(in) :: p(:,:),g,ew
    integer :: leg,mu
    if(active_lanes==0) return
    write(unit,'(a,3(1x,i0),1x,a)') 'PROVIDER',next_call,1,me,'fortran'
    write(unit,'(a,3(1x,i0),*(1x,z16.16))') 'INPUT',next_call,1,flavour,transfer(g,0_int64),transfer(ew,0_int64), &
      ((transfer(p(mu,leg),0_int64),leg=1,size(p,2)),mu=1,size(p,1))
  end subroutine

  ! Local order components immediately after restoring their physical
  ! coupling factors at the shared bridge boundary.
  subroutine trace_umami_output(count,orders,result) bind(C,name='mg7_trace_umami_output')
    integer(c_size_t), value :: count
    integer(c_int), value :: orders
    real(c_double), intent(in) :: result(*)
    integer(c_size_t) :: lane,order
    integer(int64),allocatable :: expanded(:)
    if(real_profile_active) then
       call mc_real_profile_step(2,.false.)
       if(count>size(profile_occupancy)) then
          allocate(expanded(count),source=0_int64)
          expanded(:size(profile_occupancy))=profile_occupancy
          call move_alloc(expanded,profile_occupancy)
       endif
       profile_occupancy(count)=profile_occupancy(count)+1
    endif
    if(active_lanes==0) return
    do lane=1,count
       write(unit,'(a,2(1x,i0),*(1x,z16.16))') 'OUTPUT',next_call,lane, &
         (transfer(result(order*count+lane),0_int64),order=0,orders-1)
    end do
  end subroutine

  subroutine trace_mapping(local,global,keep)
    integer, intent(in) :: local,global
    logical, intent(in) :: keep
    if(active_lanes==0) return
    write(unit,'(a,4(1x,i0))') 'MAP',next_call,local,global,merge(1,0,keep)
  end subroutine

  subroutine trace_fortran_result(wgt,orders,reason)
    real(real64), intent(in) :: wgt,orders(:)
    integer,optional,intent(in) :: reason
    if(real_profile_active.and.present(reason)) then
       profile_native(reason)=profile_native(reason)+1
    endif
    if(active_lanes==0) return
    write(unit,'(a,2(1x,i0),*(1x,z16.16))') 'OUTPUT',next_call,1,transfer(wgt,0_int64),transfer(orders,[0_int64])
  end subroutine

  subroutine trace_result(lane,request,wgt,orders)
    integer, intent(in) :: lane
    integer(int64), intent(in) :: request
    real(real64), intent(in) :: wgt,orders(:)
    if(batch_trace_unit()==-1 .or. request==0) return
    write(unit,'(a,3(1x,i0),*(1x,z16.16))') 'SCATTER',next_call,lane,request, &
      transfer(wgt,0_int64),transfer(orders,[0_int64])
    active_lanes=active_lanes-1
  end subroutine

  subroutine trace_consume(draw,request,wgt,orders)
    integer(int64), intent(in) :: draw,request
    real(real64), intent(in) :: wgt,orders(:)
    if(batch_trace_unit()==-1 .or. request==0) return
    write(unit,'(a,2(1x,i0),*(1x,z16.16))') 'CONSUME',draw,request,transfer(wgt,0_int64),transfer(orders,[0_int64])
  end subroutine
end module
