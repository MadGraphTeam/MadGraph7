program test_policy
  use audit_state
  implicit none
  integer :: j,mode,nfailed
  logical :: exists
  character(32) :: scenario
  call get_command_argument(1,scenario)
  call reset_state()
  select case(trim(scenario))
  case('startup')
    born_mode=1
    nfailed=0
    do j=1,8
      call audit_host()
      ! Eight failures exceed CHECKCYCLE. None may learn or publish a filter.
      if(ntry/=0) nfailed=nfailed+1
      inquire(file=loopfilterfn,exist=exists)
      if(exists) nfailed=nfailed+1
      inquire(file=helfilterfn,exist=exists)
      if(exists) nfailed=nfailed+1
      if(loop_calls/=0.or.any(goodamp)) nfailed=nfailed+1
      if(foundloopfilter.or.foundhelfilter) nfailed=nfailed+1
      if(ret_code_h/=4.or.ret_code_t/=2.or.ret_code_u/=0) nfailed=nfailed+1
      if(any(accuracy/=1d0).or.any(ieee_is_finite(ans))) nfailed=nfailed+1
      if(nrotations_dp/=2.or.nrotations_qp/=3) nfailed=nfailed+1
      if(mp_done.or..not.mp_done_once.or.mp_ps_set) nfailed=nfailed+1
      if(ref/=-910d0) nfailed=nfailed+1
      if(mlstabthres/=3d-6.or.ctmodeinit/=1) nfailed=nfailed+1
      if(clear_calls/=j.or.cache_off_calls/=j) nfailed=nfailed+1
      if(any(recorded_p/=p_user).or.recorded_hel/=userhel.or.any(recorded_ic/=ic)) nfailed=nfailed+1
      write(*,'(A,6I8)') 'FAILED_BORN ',j,ntry,ret_code_h,ret_code_t,ret_code_u,nfailed
    enddo
    if(nfailed/=0) then
      write(*,'(A,I0)') 'DETECTED_FAILED_BORN_STARTUP_PUBLICATION ',nfailed
      stop 71
    endif
    born_mode=0
    call audit_host()
    if(ntry/=1.or..not.checkphase.or.loop_calls/=1.or..not.all(goodamp)) stop 73
    if(ret_code_h/=1.or.any(accuracy/=-1d0)) stop 74
    inquire(file=loopfilterfn,exist=exists)
    if(exists) stop 75
    inquire(file=helfilterfn,exist=exists)
    if(exists) stop 76
    write(*,'(A)') 'PASS repeated failure preserves first valid initialization'
  case('stale')
    call mature_state()
    born_mode=2;goodhel(userhel)=-heloffset
    ! A previous invocation left an accepted QP H3 code in COMMON.
    ret_code_h=3;ret_code_t=2;ret_code_u=3;accuracy=1d-10
    call audit_host()
    write(*,'(A,4I8)') 'FILTERED_EXIT ',ret_code_h,ret_code_t,ret_code_u,loop_calls
    if(ret_code_h/=4.or.ret_code_u/=0.or.any(accuracy/=1d0)) then
      write(*,'(A)') 'DETECTED_STALE_H3_ACCEPTANCE'
      stop 72
    endif
    if(loop_calls/=0.or.born_qp_checked.or.ret_code_t/=1) stop 77
    if(any(ans(1:3,:)/=0d0).or.any(ans(0,:)/=2d0)) stop 78
    call check_restoration()
    write(*,'(A)') 'PASS stale H3 rejected on selected filtered-zero helicity'
  case('certificate')
    do mode=0,5
      call reset_state();call mature_state()
      born_mode=2;loop_mode=mode
      call audit_host()
      if(mode==1) then
        if(ret_code_h/=3.or..not.born_qp_checked.or.ret_code_t/=2) stop 79
        if(any(accuracy/=1d-10)) stop 80
      else
        if(ret_code_h/=4.or.born_qp_checked.or.any(accuracy/=1d0)) stop 81
      endif
      if(loop_calls/=1) stop 82
      call check_restoration()
      write(*,'(A,4I8,L3)') 'CERTIFICATE ',mode,ret_code_h,ret_code_t,ret_code_u,born_qp_checked
    enddo
    call reset_state();call mature_state()
    born_mode=0;loop_mode=0
    call audit_host()
    if(ret_code_h/=2.or.ret_code_u/=9) stop 83
    call check_restoration()
    call reset_state()
    born_mode=2;loop_mode=0
    call audit_host()
    if(ret_code_h/=1.or.any(accuracy/=-1d0)) stop 84
    write(*,'(A)') 'PASS current-call QP certificate and ordinary/unchecked controls'
  case('bypass')
    call mature_state()
    born_mode=2;loop_mode=1;bypass_filters=.true.
    checkphase=.false.;heldoublechecked=.true.;goodhel=7;goodamp=.false.
    call audit_host()
    if(ntry/=10.or.any(goodhel/=7).or.any(goodamp)) stop 85
    if(checkphase.or..not.heldoublechecked.or.ret_code_h/=3) stop 86
    call check_restoration()
    write(*,'(A)') 'PASS threshold rotation and bypass restoration'
  case default
    stop 99
  end select
contains
  subroutine mature_state()
    ntry=10;foundloopfilter=.true.;foundhelfilter=.true.
    heldoublechecked=.true.;checkphase=.false.;goodamp=.true.
  end subroutine
  subroutine check_restoration()
    if(nrotations_dp/=2.or.nrotations_qp/=3) stop 87
    if(mlstabthres/=3d-6.or.ctmodeinit/=1) stop 88
    if(clear_calls/=1.or.cache_off_calls/=1) stop 89
  end subroutine
end program
