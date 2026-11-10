module cms_mock_dp
  use cms_scalar_observer
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  integer,parameter :: kindr2=8
  real(kindr2),parameter :: RZRO=0
  integer :: wunit=0,prcpar=0,mock_calls=0,mock_nargs=0
  real(kindr2) :: onshellthrs=1e-6_kindr2,muscale=1000
  logical :: nonzerothrs=.true.,initz=.false.,mock_nan=.false.
  complex(kindr2) :: mock_last_args(10)=0
contains
  function areal(x) result(y)
    complex(kindr2),intent(in)::x
    real(kindr2)::y
    y=real(x,kindr2)
  end function
  function acmplx(x,y) result(z)
    real(kindr2),intent(in)::x,y
    complex(kindr2)::z
    z=cmplx(x,y,kind=kindr2)
  end function
  function neglig(i) result(y)
    integer,intent(in)::i
    real(kindr2)::y
    y=epsilon(0._kindr2)
  end function
  subroutine a0_c(rslt,mm)
    complex(kindr2),intent(out) :: rslt(0:2)
    complex(kindr2),intent(in) :: mm
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=1
    mock_last_args(1)=mm
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine a0_r(rslt,mm)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: mm
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=1
    mock_last_args(1)=mm
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine b11rc(b11,b00,b1,b0,pp,m1,m2)
    complex(kindr2),intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    real(kindr2),intent(in) :: pp
    complex(kindr2),intent(in) :: m1,m2
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=3
    mock_last_args(1)=pp
    mock_last_args(2)=m1
    mock_last_args(3)=m2
    b11=cmplx(991,0,kind=kindr2)
    if (mock_nan) b11=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b00=cmplx(991,0,kind=kindr2)
    if (mock_nan) b00=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b1=cmplx(991,0,kind=kindr2)
    if (mock_nan) b1=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b0=cmplx(991,0,kind=kindr2)
    if (mock_nan) b0=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine b11rr(b11,b00,b1,b0,pp,m1,m2)
    complex(kindr2),intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    real(kindr2),intent(in) :: pp
    real(kindr2),intent(in) :: m1,m2
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=3
    mock_last_args(1)=pp
    mock_last_args(2)=m1
    mock_last_args(3)=m2
    b11=cmplx(991,0,kind=kindr2)
    if (mock_nan) b11=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b00=cmplx(991,0,kind=kindr2)
    if (mock_nan) b00=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b1=cmplx(991,0,kind=kindr2)
    if (mock_nan) b1=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b0=cmplx(991,0,kind=kindr2)
    if (mock_nan) b0=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine c0rc(rslt,p1,p2,p3,m1,m2,m3)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3
    complex(kindr2),intent(in) :: m1,m2,m3
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=6
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=m1
    mock_last_args(5)=m2
    mock_last_args(6)=m3
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine c0rr(rslt,p1,p2,p3,m1,m2,m3)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3
    real(kindr2),intent(in) :: m1,m2,m3
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=6
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=m1
    mock_last_args(5)=m2
    mock_last_args(6)=m3
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3,p4,p12,p23
    complex(kindr2),intent(in) :: m1,m2,m3,m4
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=10
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=p4
    mock_last_args(5)=p12
    mock_last_args(6)=p23
    mock_last_args(7)=m1
    mock_last_args(8)=m2
    mock_last_args(9)=m3
    mock_last_args(10)=m4
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine d0rr(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3,p4,p12,p23
    real(kindr2),intent(in) :: m1,m2,m3,m4
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=10
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=p4
    mock_last_args(5)=p12
    mock_last_args(6)=p23
    mock_last_args(7)=m1
    mock_last_args(8)=m2
    mock_last_args(9)=m3
    mock_last_args(10)=m4
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine cms_preflight_a0_c(report,mm)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  complex(kindr2) &   
    ,intent(in)  :: mm
!
  complex(kindr2) &   
    :: ss
  real(kindr2) &  
    :: am,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,1,1)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_m(1)=mm
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
!
  mulocal = muscale 
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(am,mulocal2)
  if (.not.all(ieee_is_finite((/am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (am.lt.hh) then
      call cms_record(report,6654,CMS_MASS,CMS_ABS_COMPLEX,0,1, &
        'am',am,cms_original_m(1), &
        cmplx(mm,kind=kindr2),hh,cms_scale)
      am=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(am,mulocal2)
  endif
!
  ss = mm
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_a0_c

  subroutine cms_checked_a0_c(rslt,mm,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  complex(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_a0_c(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_c(cms_result_1,mm)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_a0_c

  subroutine cms_preflight_a0_r(report,mm)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: mm
!
  complex(kindr2) &   
    :: ss
  real(kindr2) &  
    :: am,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,1,2)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_m(1)=mm
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
!
  mulocal = muscale 
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(am,mulocal2)
  if (.not.all(ieee_is_finite((/am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (am.lt.hh) then
      call cms_record(report,6744,CMS_MASS,CMS_ABS_REAL,0,1, &
        'am',am,cms_original_m(1), &
        cmplx(mm,kind=kindr2),hh,cms_scale)
      am=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(am,mulocal2)
  endif
!
  ss = mm
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_a0_r

  subroutine cms_checked_a0_r(rslt,mm,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_a0_r(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_r(cms_result_1,mm)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_a0_r

  subroutine cms_preflight_b11rc(report,pp,m1,m2)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: pp
  complex(kindr2) &   
    ,intent(in)  :: m1,m2
!
  complex(kindr2) &   
    :: ss,r1,r2
  real(kindr2) &  
    :: app,am1,am2,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,2,3)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=pp
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  ss = pp
  r1 = m1
  r2 = m2
!
  app = abs(pp)
!
  am1 = areal(r1)
  hh  = aimag(r1)
  if (hh.gt.RZRO) then
    r1 = acmplx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = areal(r2)
  hh  = aimag(r2)
  if (hh.gt.RZRO) then
    r2 = acmplx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(app,am1,am2,mulocal2)
  if (.not.all(ieee_is_finite((/app,am1,am2,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (app.lt.hh) then
      call cms_record(report,8203,CMS_INPUT,CMS_ABS_REAL,1,0, &
        'app',app,cms_original_p(1), &
        ss,hh,cms_scale)
      app=0
    endif
    if (am1.lt.hh) then
      call cms_record(report,8204,CMS_MASS,CMS_L1_COMPLEX,0,1, &
        'am1',am1,cms_original_m(1), &
        r1,hh,cms_scale)
      am1=0
    endif
    if (am2.lt.hh) then
      call cms_record(report,8205,CMS_MASS,CMS_L1_COMPLEX,0,2, &
        'am2',am2,cms_original_m(2), &
        r2,hh,cms_scale)
      am2=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_b11rc

  subroutine cms_checked_b11rc(b11,b00,b1,b0,pp,m1,m2,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  complex(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2),cms_result_2(0:2),cms_result_3(0:2),cms_result_4(0:2)
  call cms_preflight_b11rc(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rc(cms_result_1,cms_result_2,cms_result_3,cms_result_4,pp,m1,m2)
  if (.not.all(ieee_is_finite(real((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  b11=cms_result_1
  b00=cms_result_2
  b1=cms_result_3
  b0=cms_result_4
  report%result_valid=.true.
  end subroutine cms_checked_b11rc

  subroutine cms_preflight_b11rr(report,pp,m1,m2)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: pp
  real(kindr2) &  
    ,intent(in)  :: m1,m2
!
  complex(kindr2) &   
    :: ss,r1,r2
  real(kindr2) &  
    :: app,am1,am2,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,2,4)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=pp
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  ss = pp
  r1 = m1
  r2 = m2
!
  app = abs(pp)
!
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(app,am1,am2,mulocal2)
  if (.not.all(ieee_is_finite((/app,am1,am2,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (app.lt.hh) then
      call cms_record(report,8352,CMS_INPUT,CMS_ABS_REAL,1,0, &
        'app',app,cms_original_p(1), &
        ss,hh,cms_scale)
      app=0
    endif
    if (am1.lt.hh) then
      call cms_record(report,8353,CMS_MASS,CMS_ABS_REAL,0,1, &
        'am1',am1,cms_original_m(1), &
        r1,hh,cms_scale)
      am1=0
    endif
    if (am2.lt.hh) then
      call cms_record(report,8354,CMS_MASS,CMS_ABS_REAL,0,2, &
        'am2',am2,cms_original_m(2), &
        r2,hh,cms_scale)
      am2=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_b11rr

  subroutine cms_checked_b11rr(b11,b00,b1,b0,pp,m1,m2,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  real(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2),cms_result_2(0:2),cms_result_3(0:2),cms_result_4(0:2)
  call cms_preflight_b11rr(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rr(cms_result_1,cms_result_2,cms_result_3,cms_result_4,pp,m1,m2)
  if (.not.all(ieee_is_finite(real((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  b11=cms_result_1
  b00=cms_result_2
  b1=cms_result_3
  b0=cms_result_4
  report%result_valid=.true.
  end subroutine cms_checked_b11rr

  subroutine cms_preflight_c0rc(report,p1,p2,p3,m1,m2,m3)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_dp_tri, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3
  complex(kindr2) &   
    ,intent(in)  :: m1,m2,m3
!
  real(kindr2) &  
    :: pp(3)
  complex(kindr2) &   
    :: mm(3)
  complex(kindr2) &   
    :: ss(3),rr(3),lambda
  real(kindr2) &  
    :: smax,ap(3),am(3),as(3),ar(3),hh,s1r2,s2r3,s3r3
  real(kindr2) &  
    :: mulocal,mulocal2
  integer :: icase,ii
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,3,5)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  smax = 0
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = areal(mm(ii))
    hh     = aimag(mm(ii))
    if (hh.gt.RZRO) then
      mm(ii) = acmplx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,3
    if (ap(ii).lt.hh) then
      call cms_record(report,9598,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,9599,CMS_MASS,CMS_L1_COMPLEX,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  cms_pi(1)=permtable(1,icase)
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  cms_pi(2)=permtable(2,icase)
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  cms_pi(3)=permtable(3,icase)
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  cms_mi(1)=permtable(1,icase)
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  cms_mi(2)=permtable(2,icase)
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  cms_mi(3)=permtable(3,icase)
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (.not.all(ieee_is_finite((/s1r2,s2r3,s3r3/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,9627,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,9628,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r3.lt.hh) then
      call cms_record(report,9629,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(3),cms_mi(3), &
        's3r3',s3r3,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(3)), &
        ss(3)-rr(3),hh,cms_scale)
      s3r3=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  end subroutine cms_preflight_c0rc

  subroutine cms_checked_c0rc(rslt,p1,p2,p3,m1,m2,m3,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3
  complex(kindr2), intent(in) :: m1,m2,m3
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_c0rc(report,p1,p2,p3,m1,m2,m3)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call c0rc(cms_result_1,p1,p2,p3,m1,m2,m3)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_c0rc

  subroutine cms_preflight_c0rr(report,p1,p2,p3,m1,m2,m3)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_dp_tri, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3
  real(kindr2) &  
    ,intent(in)  :: m1,m2,m3
!
  real(kindr2) &  
    :: pp(3)
  real(kindr2) &  
    :: mm(3)
  complex(kindr2) &   
    :: ss(3),rr(3),lambda
  real(kindr2) &  
    :: smax,ap(3),am(3),as(3),ar(3),hh,s1r2,s2r3,s3r3
  real(kindr2) &  
    :: mulocal,mulocal2
  integer :: icase,ii
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,3,6)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  smax = 0
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,3
    if (ap(ii).lt.hh) then
      call cms_record(report,9905,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,9906,CMS_MASS,CMS_ABS_REAL,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  cms_pi(1)=permtable(1,icase)
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  cms_pi(2)=permtable(2,icase)
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  cms_pi(3)=permtable(3,icase)
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  cms_mi(1)=permtable(1,icase)
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  cms_mi(2)=permtable(2,icase)
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  cms_mi(3)=permtable(3,icase)
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (.not.all(ieee_is_finite((/s1r2,s2r3,s3r3/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,9934,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,9935,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r3.lt.hh) then
      call cms_record(report,9936,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(3),cms_mi(3), &
        's3r3',s3r3,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(3)), &
        ss(3)-rr(3),hh,cms_scale)
      s3r3=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  end subroutine cms_preflight_c0rr

  subroutine cms_checked_c0rr(rslt,p1,p2,p3,m1,m2,m3,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3
  real(kindr2), intent(in) :: m1,m2,m3
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_c0rr(report,p1,p2,p3,m1,m2,m3)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call c0rr(cms_result_1,p1,p2,p3,m1,m2,m3)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_c0rr

  subroutine cms_preflight_d0rc(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_dp_box, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3,p4,p12,p23
  complex(kindr2) &   
    ,intent(in)  :: m1,m2,m3,m4
!
  real(kindr2) &  
    :: pp(6)
  complex(kindr2) &   
    :: mm(4)
  complex(kindr2) &   
    :: ss(6),rr(4)
  real(kindr2) &  
    :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  real(kindr2) &  
    :: mulocal,mulocal2,small,hh,min13,min24,min56
  integer :: icase,ii,jj
  logical :: useboxc
  integer ,parameter :: lp(6,3)=&
           reshape((/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/),(/6,3/))
  integer ,parameter :: lm(4,3)=&
           reshape((/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /),(/4,3/))
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,4,7)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_p(4)=p4
  cms_original_p(5)=p12
  cms_original_p(6)=p23
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  cms_original_m(4)=m4
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
  smax = 0
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = areal(mm(ii))
    hh = aimag(mm(ii))
    if (hh.gt.RZRO) then
      mm(ii) = acmplx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = 0
  do ii=1,6
    hh = abs(ap(ii))
    if (hh.gt.small) small=hh
  enddo
  small = small*neglig(prcpar)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,small,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,4
    if (ap(ii).lt.hh) then
      call cms_record(report,10799,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,10800,CMS_MASS,CMS_L1_COMPLEX,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
    enddo
    endif
  endif
!
  jj = 1
  min56 = min(ap(5),ap(6))
  if (min56.lt.hh) then
    min13=min(ap(1),ap(3))
    min24=min(ap(2),ap(4))
    if     (min13.gt.min24.and.min13.gt.min56) then ;jj=2
    elseif (min24.gt.min13.and.min24.gt.min56) then ;jj=3
    else
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(lm(ii,jj)).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(lp(permtable(1,icase),jj)) ;as(1)=ap(lp(permtable(1,icase),jj))
  cms_pi(1)=lp(permtable(1,icase),jj)
  ss(2)=pp(lp(permtable(2,icase),jj)) ;as(2)=ap(lp(permtable(2,icase),jj))
  cms_pi(2)=lp(permtable(2,icase),jj)
  ss(3)=pp(lp(permtable(3,icase),jj)) ;as(3)=ap(lp(permtable(3,icase),jj))
  cms_pi(3)=lp(permtable(3,icase),jj)
  ss(4)=pp(lp(permtable(4,icase),jj)) ;as(4)=ap(lp(permtable(4,icase),jj))
  cms_pi(4)=lp(permtable(4,icase),jj)
  ss(5)=pp(lp(permtable(5,icase),jj)) ;as(5)=ap(lp(permtable(5,icase),jj))
  cms_pi(5)=lp(permtable(5,icase),jj)
  ss(6)=pp(lp(permtable(6,icase),jj)) ;as(6)=ap(lp(permtable(6,icase),jj))
  cms_pi(6)=lp(permtable(6,icase),jj)
  rr(1)=mm(lm(permtable(1,icase),jj)) ;ar(1)=am(lm(permtable(1,icase),jj))
  cms_mi(1)=lm(permtable(1,icase),jj)
  rr(2)=mm(lm(permtable(2,icase),jj)) ;ar(2)=am(lm(permtable(2,icase),jj))
  cms_mi(2)=lm(permtable(2,icase),jj)
  rr(3)=mm(lm(permtable(3,icase),jj)) ;ar(3)=am(lm(permtable(3,icase),jj))
  cms_mi(3)=lm(permtable(3,icase),jj)
  rr(4)=mm(lm(permtable(4,icase),jj)) ;ar(4)=am(lm(permtable(4,icase),jj))
  cms_mi(4)=lm(permtable(4,icase),jj)
  icase = casetable(icase)
!
  s1r2 = abs(areal(ss(1)-rr(2))) + abs(aimag(ss(1)-rr(2)))
  s2r2 = abs(areal(ss(2)-rr(2))) + abs(aimag(ss(2)-rr(2)))
  s2r3 = abs(areal(ss(2)-rr(3))) + abs(aimag(ss(2)-rr(3)))
  s3r4 = abs(areal(ss(3)-rr(4))) + abs(aimag(ss(3)-rr(4)))
  s4r4 = abs(areal(ss(4)-rr(4))) + abs(aimag(ss(4)-rr(4)))
  if (.not.all(ieee_is_finite((/s1r2,s2r2,s2r3,s3r4,s4r4/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,10850,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r2.lt.hh) then
      call cms_record(report,10851,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(2), &
        's2r2',s2r2,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(2)), &
        ss(2)-rr(2),hh,cms_scale)
      s2r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,10852,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r4.lt.hh) then
      call cms_record(report,10853,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(3),cms_mi(4), &
        's3r4',s3r4,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(4)), &
        ss(3)-rr(4),hh,cms_scale)
      s3r4=0
    endif
    if (s4r4.lt.hh) then
      call cms_record(report,10854,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(4),cms_mi(4), &
        's4r4',s4r4,cms_original_p(cms_pi(4))-cms_original_m(cms_mi(4)), &
        ss(4)-rr(4),hh,cms_scale)
      s4r4=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  report%soft_permutation=jj
  end subroutine cms_preflight_d0rc

  subroutine cms_checked_d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  complex(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_d0rc(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rc(cms_result_1,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_d0rc

  subroutine cms_preflight_d0rr(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_dp_box, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3,p4,p12,p23
  real(kindr2) &  
    ,intent(in)  :: m1,m2,m3,m4
!
  real(kindr2) &  
    :: pp(6)
  real(kindr2) &  
    :: mm(4)
  complex(kindr2) &   
    :: ss(6),rr(4)
  real(kindr2) &  
    :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  real(kindr2) &  
    :: mulocal,mulocal2,small,hh,min13,min24,min56
  integer :: icase,ii,jj
  logical :: useboxc
  integer ,parameter :: lp(6,3)=&
           reshape((/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/),(/6,3/))
  integer ,parameter :: lm(4,3)=&
           reshape((/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /),(/4,3/))
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,8,4,8)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_p(4)=p4
  cms_original_p(5)=p12
  cms_original_p(6)=p23
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  cms_original_m(4)=m4
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
  smax = 0
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = 0
  do ii=1,6
    hh = abs(ap(ii))
    if (hh.gt.small) small=hh
  enddo
  small = small*neglig(prcpar)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,small,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,4
    if (ap(ii).lt.hh) then
      call cms_record(report,11328,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,11329,CMS_MASS,CMS_ABS_REAL,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
    enddo
    endif
  endif
!
  jj = 1
  min56 = min(ap(5),ap(6))
  if (min56.lt.hh) then
    min13=min(ap(1),ap(3))
    min24=min(ap(2),ap(4))
    if     (min13.gt.min24.and.min13.gt.min56) then ;jj=2
    elseif (min24.gt.min13.and.min24.gt.min56) then ;jj=3
    else
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(lm(ii,jj)).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(lp(permtable(1,icase),jj)) ;as(1)=ap(lp(permtable(1,icase),jj))
  cms_pi(1)=lp(permtable(1,icase),jj)
  ss(2)=pp(lp(permtable(2,icase),jj)) ;as(2)=ap(lp(permtable(2,icase),jj))
  cms_pi(2)=lp(permtable(2,icase),jj)
  ss(3)=pp(lp(permtable(3,icase),jj)) ;as(3)=ap(lp(permtable(3,icase),jj))
  cms_pi(3)=lp(permtable(3,icase),jj)
  ss(4)=pp(lp(permtable(4,icase),jj)) ;as(4)=ap(lp(permtable(4,icase),jj))
  cms_pi(4)=lp(permtable(4,icase),jj)
  ss(5)=pp(lp(permtable(5,icase),jj)) ;as(5)=ap(lp(permtable(5,icase),jj))
  cms_pi(5)=lp(permtable(5,icase),jj)
  ss(6)=pp(lp(permtable(6,icase),jj)) ;as(6)=ap(lp(permtable(6,icase),jj))
  cms_pi(6)=lp(permtable(6,icase),jj)
  rr(1)=mm(lm(permtable(1,icase),jj)) ;ar(1)=am(lm(permtable(1,icase),jj))
  cms_mi(1)=lm(permtable(1,icase),jj)
  rr(2)=mm(lm(permtable(2,icase),jj)) ;ar(2)=am(lm(permtable(2,icase),jj))
  cms_mi(2)=lm(permtable(2,icase),jj)
  rr(3)=mm(lm(permtable(3,icase),jj)) ;ar(3)=am(lm(permtable(3,icase),jj))
  cms_mi(3)=lm(permtable(3,icase),jj)
  rr(4)=mm(lm(permtable(4,icase),jj)) ;ar(4)=am(lm(permtable(4,icase),jj))
  cms_mi(4)=lm(permtable(4,icase),jj)
  icase = casetable(icase)
!
  s1r2 = abs(areal(ss(1)-rr(2))) + abs(aimag(ss(1)-rr(2)))
  s2r2 = abs(areal(ss(2)-rr(2))) + abs(aimag(ss(2)-rr(2)))
  s2r3 = abs(areal(ss(2)-rr(3))) + abs(aimag(ss(2)-rr(3)))
  s3r4 = abs(areal(ss(3)-rr(4))) + abs(aimag(ss(3)-rr(4)))
  s4r4 = abs(areal(ss(4)-rr(4))) + abs(aimag(ss(4)-rr(4)))
  if (.not.all(ieee_is_finite((/s1r2,s2r2,s2r3,s3r4,s4r4/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,11379,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r2.lt.hh) then
      call cms_record(report,11380,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(2), &
        's2r2',s2r2,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(2)), &
        ss(2)-rr(2),hh,cms_scale)
      s2r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,11381,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r4.lt.hh) then
      call cms_record(report,11382,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(3),cms_mi(4), &
        's3r4',s3r4,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(4)), &
        ss(3)-rr(4),hh,cms_scale)
      s3r4=0
    endif
    if (s4r4.lt.hh) then
      call cms_record(report,11383,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(4),cms_mi(4), &
        's4r4',s4r4,cms_original_p(cms_pi(4))-cms_original_m(cms_mi(4)), &
        ss(4)-rr(4),hh,cms_scale)
      s4r4=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  report%soft_permutation=jj
  end subroutine cms_preflight_d0rr

  subroutine cms_checked_d0rr(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  real(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_d0rr(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rr(cms_result_1,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_d0rr
  subroutine fixture_check()
    type(cms_scalar_report)::report
    real(kindr2)::p(6),rm(4),p_before(6),rm_before(4)
    complex(kindr2)::cm(4),r(0:2,4),sentinel,cm_before(4)
    integer::i,before
    integer(8)::p_words(6*kindr2/8),rm_words(4*kindr2/8),cm_words(8*kindr2/8),r_words(24*kindr2/8)
    sentinel=cmplx(1234,5678,kind=kindr2)
    ! dp a0_c: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.1,103)
    call insist(mock_last_args(1).eq.cm(1),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.1,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp a0_r: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_a0_r(r(:,1),rm(1),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.1,103)
    call insist(mock_last_args(1).eq.cmplx(rm(1),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_a0_r(r(:,1),rm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.1,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp b11rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),p(1),cm(1),cm(2),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.3,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cm(1),104)
    call insist(mock_last_args(3).eq.cm(2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),p(1),cm(1),cm(2),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.3,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.2,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp b11rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_b11rr(r(:,1),r(:,2),r(:,3),r(:,4),p(1),rm(1),rm(2),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.3,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(rm(2),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_b11rr(r(:,1),r(:,2),r(:,3),r(:,4),p(1),rm(1),rm(2),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.3,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.2,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp c0rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_c0rc(r(:,1),p(1),p(2),p(3),cm(1),cm(2),cm(3),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.6,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cm(1),104)
    call insist(mock_last_args(5).eq.cm(2),104)
    call insist(mock_last_args(6).eq.cm(3),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_c0rc(r(:,1),p(1),p(2),p(3),cm(1),cm(2),cm(3),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.9,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.3,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp c0rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.6,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(rm(2),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(rm(3),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.9,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.3,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp d0rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_d0rc(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),cm(1),cm(2),cm(3),cm(4),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.10,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(p(4),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(p(5),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(p(6),kind=kindr2),104)
    call insist(mock_last_args(7).eq.cm(1),104)
    call insist(mock_last_args(8).eq.cm(2),104)
    call insist(mock_last_args(9).eq.cm(3),104)
    call insist(mock_last_args(10).eq.cm(4),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_d0rc(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),cm(1),cm(2),cm(3),cm(4),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.13,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.4,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! dp d0rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.10,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(p(4),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(p(5),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(p(6),kind=kindr2),104)
    call insist(mock_last_args(7).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(8).eq.cmplx(rm(2),kind=kindr2),104)
    call insist(mock_last_args(9).eq.cmplx(rm(3),kind=kindr2),104)
    call insist(mock_last_args(10).eq.cmplx(rm(4),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.13,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.8.and.report%events(i)%topology.eq.4,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! Strict inequality, exact zeros and real/complex norm distinctions.
    r=sentinel;before=mock_calls
    call cms_checked_a0_r(r(:,1),onshellthrs,report)
    call insist(report%status.eq.CMS_OK.and.mock_calls.eq.before+1,120)
    call cms_checked_a0_r(r(:,1),0._kindr2,report)
    call insist(report%status.eq.CMS_OK.and.report%count.eq.0,121)
    cm(1)=cmplx(0.6_kindr2*onshellthrs,-0.6_kindr2*onshellthrs,kind=kindr2)
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%events(1)%metric_norm.eq.CMS_ABS_COMPLEX,122)
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_OK,123)
    ! All-zero triangle/box are explicit unsupported results; A/B exact-zero remains delegated.
    p=0;rm=0;r=sentinel;before=mock_calls
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.mock_calls.eq.before,124)
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.all(r.eq.sentinel),125)
    nonzerothrs=.false.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_DEFAULT_UNSUPPORTED.and..not.report%native_called,126)
    nonzerothrs=.true.;initz=.true.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_UNINITIALIZED.and..not.report%native_called,127)
    initz=.false.
    ! Opposite mass positions exercise the exact triangle permutation table.
    p=-10;rm=0;rm(1)=20
    call cms_preflight_c0rr(report,p(1),p(2),p(3),rm(1),rm(2),rm(3))
    call insist(all(report%p_sources(1:3).eq.(/2,3,1/)),128)
    call insist(all(report%mass_sources(1:3).eq.(/2,3,1/)),129)
    ! Soft box routing jj=2 when the 1/3 channel dominates 2/4 and 5/6.
    p=(/10._kindr2,2._kindr2,10._kindr2,2._kindr2,0._kindr2,0._kindr2/);rm=20
    call cms_preflight_d0rr(report,p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4))
    call insist(report%soft_permutation.eq.2,130)
    call insist(all(report%p_sources.eq.(/5,2,6,4,1,3/)),131)
    ! A forced observer overflow is still a refusal, never a success.
    call cms_reset(report,kindr2,4,7)
    do i=1,CMS_MAX_EVENTS+1
      call cms_record(report,1,CMS_INPUT,CMS_ABS_REAL,1,0,'forced',onshellthrs/2, &
        cmplx(onshellthrs/2,0,kindr2),cmplx(onshellthrs/2,0,kindr2),onshellthrs,1._kindr2)
    enddo
    call insist(report%status.eq.CMS_EVENT_OVERFLOW.and.report%count.eq.CMS_MAX_EVENTS.and.report%dropped.eq.1,132)
    call insist(.not.report%result_valid.and..not.report%native_called,133)
    ! Native callbacks with nonfinite output must not publish caller buffers.
    r=sentinel;before=mock_calls;mock_nan=.true.
    call cms_checked_a0_r(r(:,1),20._kindr2,report)
    call insist(report%status.eq.CMS_NONFINITE_RESULT.and.report%native_called,134)
    call insist(.not.report%result_valid.and.all(r.eq.sentinel).and.mock_calls.eq.before+1,135)
    mock_nan=.false.
    ! Finite input whose native L1 norm overflows is refused before callback.
    cm(1)=cmplx(huge(0._kindr2),-huge(0._kindr2),kind=kindr2);before=mock_calls
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_BAD_INPUT.and..not.report%native_called.and.mock_calls.eq.before,136)
    write(*,*) 'CLASSIFIER_MOCK_COMPLETE',kindr2,8
  end subroutine
  subroutine insist(ok,code)
    logical,intent(in)::ok
    integer,intent(in)::code
    if (.not.ok) then
      write(*,*) 'MOCK_FAILURE',kindr2,code
      error stop 1
    endif
  end subroutine
end module

module cms_mock_qp
  use cms_scalar_observer
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  integer,parameter :: kindr2=16
  real(kindr2),parameter :: RZRO=0
  integer :: wunit=0,prcpar=0,mock_calls=0,mock_nargs=0
  real(kindr2) :: onshellthrs=1e-6_kindr2,muscale=1000
  logical :: nonzerothrs=.true.,initz=.false.,mock_nan=.false.
  complex(kindr2) :: mock_last_args(10)=0
contains
  function areal(x) result(y)
    complex(kindr2),intent(in)::x
    real(kindr2)::y
    y=real(x,kindr2)
  end function
  function acmplx(x,y) result(z)
    real(kindr2),intent(in)::x,y
    complex(kindr2)::z
    z=cmplx(x,y,kind=kindr2)
  end function
  function neglig(i) result(y)
    integer,intent(in)::i
    real(kindr2)::y
    y=epsilon(0._kindr2)
  end function
  subroutine a0_c(rslt,mm)
    complex(kindr2),intent(out) :: rslt(0:2)
    complex(kindr2),intent(in) :: mm
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=1
    mock_last_args(1)=mm
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine a0_r(rslt,mm)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: mm
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=1
    mock_last_args(1)=mm
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine b11rc(b11,b00,b1,b0,pp,m1,m2)
    complex(kindr2),intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    real(kindr2),intent(in) :: pp
    complex(kindr2),intent(in) :: m1,m2
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=3
    mock_last_args(1)=pp
    mock_last_args(2)=m1
    mock_last_args(3)=m2
    b11=cmplx(991,0,kind=kindr2)
    if (mock_nan) b11=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b00=cmplx(991,0,kind=kindr2)
    if (mock_nan) b00=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b1=cmplx(991,0,kind=kindr2)
    if (mock_nan) b1=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b0=cmplx(991,0,kind=kindr2)
    if (mock_nan) b0=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine b11rr(b11,b00,b1,b0,pp,m1,m2)
    complex(kindr2),intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
    real(kindr2),intent(in) :: pp
    real(kindr2),intent(in) :: m1,m2
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=3
    mock_last_args(1)=pp
    mock_last_args(2)=m1
    mock_last_args(3)=m2
    b11=cmplx(991,0,kind=kindr2)
    if (mock_nan) b11=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b00=cmplx(991,0,kind=kindr2)
    if (mock_nan) b00=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b1=cmplx(991,0,kind=kindr2)
    if (mock_nan) b1=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
    b0=cmplx(991,0,kind=kindr2)
    if (mock_nan) b0=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine c0rc(rslt,p1,p2,p3,m1,m2,m3)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3
    complex(kindr2),intent(in) :: m1,m2,m3
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=6
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=m1
    mock_last_args(5)=m2
    mock_last_args(6)=m3
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine c0rr(rslt,p1,p2,p3,m1,m2,m3)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3
    real(kindr2),intent(in) :: m1,m2,m3
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=6
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=m1
    mock_last_args(5)=m2
    mock_last_args(6)=m3
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3,p4,p12,p23
    complex(kindr2),intent(in) :: m1,m2,m3,m4
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=10
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=p4
    mock_last_args(5)=p12
    mock_last_args(6)=p23
    mock_last_args(7)=m1
    mock_last_args(8)=m2
    mock_last_args(9)=m3
    mock_last_args(10)=m4
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine d0rr(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
    complex(kindr2),intent(out) :: rslt(0:2)
    real(kindr2),intent(in) :: p1,p2,p3,p4,p12,p23
    real(kindr2),intent(in) :: m1,m2,m3,m4
    mock_calls=mock_calls+1
    mock_last_args=0
    mock_nargs=10
    mock_last_args(1)=p1
    mock_last_args(2)=p2
    mock_last_args(3)=p3
    mock_last_args(4)=p4
    mock_last_args(5)=p12
    mock_last_args(6)=p23
    mock_last_args(7)=m1
    mock_last_args(8)=m2
    mock_last_args(9)=m3
    mock_last_args(10)=m4
    rslt=cmplx(991,0,kind=kindr2)
    if (mock_nan) rslt=cmplx(ieee_value(0._kindr2,ieee_quiet_nan),0,kind=kindr2)
  end subroutine

  subroutine cms_preflight_a0_c(report,mm)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  complex(kindr2) &   
    ,intent(in)  :: mm
!
  complex(kindr2) &   
    :: ss
  real(kindr2) &  
    :: am,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,1,1)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_m(1)=mm
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
!
  mulocal = muscale 
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(am,mulocal2)
  if (.not.all(ieee_is_finite((/am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (am.lt.hh) then
      call cms_record(report,18351,CMS_MASS,CMS_ABS_COMPLEX,0,1, &
        'am',am,cms_original_m(1), &
        cmplx(mm,kind=kindr2),hh,cms_scale)
      am=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(am,mulocal2)
  endif
!
  ss = mm
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_a0_c

  subroutine cms_checked_a0_c(rslt,mm,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  complex(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_a0_c(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_c(cms_result_1,mm)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_a0_c

  subroutine cms_preflight_a0_r(report,mm)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: mm
!
  complex(kindr2) &   
    :: ss
  real(kindr2) &  
    :: am,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,1,2)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_m(1)=mm
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
!
  mulocal = muscale 
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(am,mulocal2)
  if (.not.all(ieee_is_finite((/am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (am.lt.hh) then
      call cms_record(report,18441,CMS_MASS,CMS_ABS_REAL,0,1, &
        'am',am,cms_original_m(1), &
        cmplx(mm,kind=kindr2),hh,cms_scale)
      am=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(am,mulocal2)
  endif
!
  ss = mm
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_a0_r

  subroutine cms_checked_a0_r(rslt,mm,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: mm
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_a0_r(report,mm)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call a0_r(cms_result_1,mm)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_a0_r

  subroutine cms_preflight_b11rc(report,pp,m1,m2)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: pp
  complex(kindr2) &   
    ,intent(in)  :: m1,m2
!
  complex(kindr2) &   
    :: ss,r1,r2
  real(kindr2) &  
    :: app,am1,am2,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,2,3)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=pp
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  ss = pp
  r1 = m1
  r2 = m2
!
  app = abs(pp)
!
  am1 = areal(r1)
  hh  = aimag(r1)
  if (hh.gt.RZRO) then
    r1 = acmplx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = areal(r2)
  hh  = aimag(r2)
  if (hh.gt.RZRO) then
    r2 = acmplx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(app,am1,am2,mulocal2)
  if (.not.all(ieee_is_finite((/app,am1,am2,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (app.lt.hh) then
      call cms_record(report,19900,CMS_INPUT,CMS_ABS_REAL,1,0, &
        'app',app,cms_original_p(1), &
        ss,hh,cms_scale)
      app=0
    endif
    if (am1.lt.hh) then
      call cms_record(report,19901,CMS_MASS,CMS_L1_COMPLEX,0,1, &
        'am1',am1,cms_original_m(1), &
        r1,hh,cms_scale)
      am1=0
    endif
    if (am2.lt.hh) then
      call cms_record(report,19902,CMS_MASS,CMS_L1_COMPLEX,0,2, &
        'am2',am2,cms_original_m(2), &
        r2,hh,cms_scale)
      am2=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_b11rc

  subroutine cms_checked_b11rc(b11,b00,b1,b0,pp,m1,m2,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  complex(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2),cms_result_2(0:2),cms_result_3(0:2),cms_result_4(0:2)
  call cms_preflight_b11rc(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rc(cms_result_1,cms_result_2,cms_result_3,cms_result_4,pp,m1,m2)
  if (.not.all(ieee_is_finite(real((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  b11=cms_result_1
  b00=cms_result_2
  b1=cms_result_3
  b0=cms_result_4
  report%result_valid=.true.
  end subroutine cms_checked_b11rc

  subroutine cms_preflight_b11rr(report,pp,m1,m2)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
!
!
  real(kindr2) &  
    ,intent(in)  :: pp
  real(kindr2) &  
    ,intent(in)  :: m1,m2
!
  complex(kindr2) &   
    :: ss,r1,r2
  real(kindr2) &  
    :: app,am1,am2,hh,mulocal,mulocal2
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,2,4)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=pp
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  ss = pp
  r1 = m1
  r2 = m2
!
  app = abs(pp)
!
  am1 = abs(m1)
  am2 = abs(m2)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  cms_scale=max(app,am1,am2,mulocal2)
  if (.not.all(ieee_is_finite((/app,am1,am2,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    if (app.lt.hh) then
      call cms_record(report,20049,CMS_INPUT,CMS_ABS_REAL,1,0, &
        'app',app,cms_original_p(1), &
        ss,hh,cms_scale)
      app=0
    endif
    if (am1.lt.hh) then
      call cms_record(report,20050,CMS_MASS,CMS_ABS_REAL,0,1, &
        'am1',am1,cms_original_m(1), &
        r1,hh,cms_scale)
      am1=0
    endif
    if (am2.lt.hh) then
      call cms_record(report,20051,CMS_MASS,CMS_ABS_REAL,0,2, &
        'am2',am2,cms_original_m(2), &
        r2,hh,cms_scale)
      am2=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_b11rr

  subroutine cms_checked_b11rr(b11,b00,b1,b0,pp,m1,m2,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2), intent(in) :: pp
  real(kindr2), intent(in) :: m1,m2
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2),cms_result_2(0:2),cms_result_3(0:2),cms_result_4(0:2)
  call cms_preflight_b11rr(report,pp,m1,m2)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call b11rr(cms_result_1,cms_result_2,cms_result_3,cms_result_4,pp,m1,m2)
  if (.not.all(ieee_is_finite(real((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1,cms_result_2,cms_result_3,cms_result_4/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  b11=cms_result_1
  b00=cms_result_2
  b1=cms_result_3
  b0=cms_result_4
  report%result_valid=.true.
  end subroutine cms_checked_b11rr

  subroutine cms_preflight_c0rc(report,p1,p2,p3,m1,m2,m3)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_qp_tri, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3
  complex(kindr2) &   
    ,intent(in)  :: m1,m2,m3
!
  real(kindr2) &  
    :: pp(3)
  complex(kindr2) &   
    :: mm(3)
  complex(kindr2) &   
    :: ss(3),rr(3),lambda
  real(kindr2) &  
    :: smax,ap(3),am(3),as(3),ar(3),hh,s1r2,s2r3,s3r3
  real(kindr2) &  
    :: mulocal,mulocal2
  integer :: icase,ii
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,3,5)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  smax = 0
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = areal(mm(ii))
    hh     = aimag(mm(ii))
    if (hh.gt.RZRO) then
      mm(ii) = acmplx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,3
    if (ap(ii).lt.hh) then
      call cms_record(report,21295,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,21296,CMS_MASS,CMS_L1_COMPLEX,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  cms_pi(1)=permtable(1,icase)
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  cms_pi(2)=permtable(2,icase)
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  cms_pi(3)=permtable(3,icase)
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  cms_mi(1)=permtable(1,icase)
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  cms_mi(2)=permtable(2,icase)
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  cms_mi(3)=permtable(3,icase)
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (.not.all(ieee_is_finite((/s1r2,s2r3,s3r3/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,21324,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,21325,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r3.lt.hh) then
      call cms_record(report,21326,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(3),cms_mi(3), &
        's3r3',s3r3,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(3)), &
        ss(3)-rr(3),hh,cms_scale)
      s3r3=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  end subroutine cms_preflight_c0rc

  subroutine cms_checked_c0rc(rslt,p1,p2,p3,m1,m2,m3,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3
  complex(kindr2), intent(in) :: m1,m2,m3
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_c0rc(report,p1,p2,p3,m1,m2,m3)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call c0rc(cms_result_1,p1,p2,p3,m1,m2,m3)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_c0rc

  subroutine cms_preflight_c0rr(report,p1,p2,p3,m1,m2,m3)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_qp_tri, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3
  real(kindr2) &  
    ,intent(in)  :: m1,m2,m3
!
  real(kindr2) &  
    :: pp(3)
  real(kindr2) &  
    :: mm(3)
  complex(kindr2) &   
    :: ss(3),rr(3),lambda
  real(kindr2) &  
    :: smax,ap(3),am(3),as(3),ar(3),hh,s1r2,s2r3,s3r3
  real(kindr2) &  
    :: mulocal,mulocal2
  integer :: icase,ii
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,3,6)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  smax = 0
!
  do ii=1,3
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,3
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,3
    if (ap(ii).lt.hh) then
      call cms_record(report,21602,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,21603,CMS_MASS,CMS_ABS_REAL,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  cms_pi(1)=permtable(1,icase)
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  cms_pi(2)=permtable(2,icase)
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  cms_pi(3)=permtable(3,icase)
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  cms_mi(1)=permtable(1,icase)
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  cms_mi(2)=permtable(2,icase)
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  cms_mi(3)=permtable(3,icase)
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (.not.all(ieee_is_finite((/s1r2,s2r3,s3r3/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,21631,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,21632,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r3.lt.hh) then
      call cms_record(report,21633,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(3),cms_mi(3), &
        's3r3',s3r3,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(3)), &
        ss(3)-rr(3),hh,cms_scale)
      s3r3=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  end subroutine cms_preflight_c0rr

  subroutine cms_checked_c0rr(rslt,p1,p2,p3,m1,m2,m3,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3
  real(kindr2), intent(in) :: m1,m2,m3
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_c0rr(report,p1,p2,p3,m1,m2,m3)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call c0rr(cms_result_1,p1,p2,p3,m1,m2,m3)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_c0rr

  subroutine cms_preflight_d0rc(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_qp_box, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3,p4,p12,p23
  complex(kindr2) &   
    ,intent(in)  :: m1,m2,m3,m4
!
  real(kindr2) &  
    :: pp(6)
  complex(kindr2) &   
    :: mm(4)
  complex(kindr2) &   
    :: ss(6),rr(4)
  real(kindr2) &  
    :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  real(kindr2) &  
    :: mulocal,mulocal2,small,hh,min13,min24,min56
  integer :: icase,ii,jj
  logical :: useboxc
  integer ,parameter :: lp(6,3)=&
           reshape((/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/),(/6,3/))
  integer ,parameter :: lm(4,3)=&
           reshape((/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /),(/4,3/))
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,4,7)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_p(4)=p4
  cms_original_p(5)=p12
  cms_original_p(6)=p23
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  cms_original_m(4)=m4
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
  smax = 0
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = areal(mm(ii))
    hh = aimag(mm(ii))
    if (hh.gt.RZRO) then
      mm(ii) = acmplx( am(ii) ,-hh )
    endif
    am(ii) = abs(am(ii)) + abs(hh)
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = 0
  do ii=1,6
    hh = abs(ap(ii))
    if (hh.gt.small) small=hh
  enddo
  small = small*neglig(prcpar)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,small,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,4
    if (ap(ii).lt.hh) then
      call cms_record(report,22496,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,22497,CMS_MASS,CMS_L1_COMPLEX,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
    enddo
    endif
  endif
!
  jj = 1
  min56 = min(ap(5),ap(6))
  if (min56.lt.hh) then
    min13=min(ap(1),ap(3))
    min24=min(ap(2),ap(4))
    if     (min13.gt.min24.and.min13.gt.min56) then ;jj=2
    elseif (min24.gt.min13.and.min24.gt.min56) then ;jj=3
    else
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(lm(ii,jj)).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(lp(permtable(1,icase),jj)) ;as(1)=ap(lp(permtable(1,icase),jj))
  cms_pi(1)=lp(permtable(1,icase),jj)
  ss(2)=pp(lp(permtable(2,icase),jj)) ;as(2)=ap(lp(permtable(2,icase),jj))
  cms_pi(2)=lp(permtable(2,icase),jj)
  ss(3)=pp(lp(permtable(3,icase),jj)) ;as(3)=ap(lp(permtable(3,icase),jj))
  cms_pi(3)=lp(permtable(3,icase),jj)
  ss(4)=pp(lp(permtable(4,icase),jj)) ;as(4)=ap(lp(permtable(4,icase),jj))
  cms_pi(4)=lp(permtable(4,icase),jj)
  ss(5)=pp(lp(permtable(5,icase),jj)) ;as(5)=ap(lp(permtable(5,icase),jj))
  cms_pi(5)=lp(permtable(5,icase),jj)
  ss(6)=pp(lp(permtable(6,icase),jj)) ;as(6)=ap(lp(permtable(6,icase),jj))
  cms_pi(6)=lp(permtable(6,icase),jj)
  rr(1)=mm(lm(permtable(1,icase),jj)) ;ar(1)=am(lm(permtable(1,icase),jj))
  cms_mi(1)=lm(permtable(1,icase),jj)
  rr(2)=mm(lm(permtable(2,icase),jj)) ;ar(2)=am(lm(permtable(2,icase),jj))
  cms_mi(2)=lm(permtable(2,icase),jj)
  rr(3)=mm(lm(permtable(3,icase),jj)) ;ar(3)=am(lm(permtable(3,icase),jj))
  cms_mi(3)=lm(permtable(3,icase),jj)
  rr(4)=mm(lm(permtable(4,icase),jj)) ;ar(4)=am(lm(permtable(4,icase),jj))
  cms_mi(4)=lm(permtable(4,icase),jj)
  icase = casetable(icase)
!
  s1r2 = abs(areal(ss(1)-rr(2))) + abs(aimag(ss(1)-rr(2)))
  s2r2 = abs(areal(ss(2)-rr(2))) + abs(aimag(ss(2)-rr(2)))
  s2r3 = abs(areal(ss(2)-rr(3))) + abs(aimag(ss(2)-rr(3)))
  s3r4 = abs(areal(ss(3)-rr(4))) + abs(aimag(ss(3)-rr(4)))
  s4r4 = abs(areal(ss(4)-rr(4))) + abs(aimag(ss(4)-rr(4)))
  if (.not.all(ieee_is_finite((/s1r2,s2r2,s2r3,s3r4,s4r4/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,22547,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r2.lt.hh) then
      call cms_record(report,22548,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(2), &
        's2r2',s2r2,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(2)), &
        ss(2)-rr(2),hh,cms_scale)
      s2r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,22549,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r4.lt.hh) then
      call cms_record(report,22550,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(3),cms_mi(4), &
        's3r4',s3r4,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(4)), &
        ss(3)-rr(4),hh,cms_scale)
      s3r4=0
    endif
    if (s4r4.lt.hh) then
      call cms_record(report,22551,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(4),cms_mi(4), &
        's4r4',s4r4,cms_original_p(cms_pi(4))-cms_original_m(cms_mi(4)), &
        ss(4)-rr(4),hh,cms_scale)
      s4r4=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  report%soft_permutation=jj
  end subroutine cms_preflight_d0rc

  subroutine cms_checked_d0rc(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  complex(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_d0rc(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rc(cms_result_1,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_d0rc

  subroutine cms_preflight_d0rr(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use avh_olo_forCutTools_qp_box, only: base, permtable, casetable
!
  real(kindr2) &  
    ,intent(in)  :: p1,p2,p3,p4,p12,p23
  real(kindr2) &  
    ,intent(in)  :: m1,m2,m3,m4
!
  real(kindr2) &  
    :: pp(6)
  real(kindr2) &  
    :: mm(4)
  complex(kindr2) &   
    :: ss(6),rr(4)
  real(kindr2) &  
    :: smax,ap(6),am(4),as(6),ar(4),s1r2,s2r2,s2r3,s3r4,s4r4
  real(kindr2) &  
    :: mulocal,mulocal2,small,hh,min13,min24,min56
  integer :: icase,ii,jj
  logical :: useboxc
  integer ,parameter :: lp(6,3)=&
           reshape((/1,2,3,4,5,6 ,5,2,6,4,1,3 ,1,6,3,5,4,2/),(/6,3/))
  integer ,parameter :: lm(4,3)=&
           reshape((/1,2,3,4     ,1,3,2,4     ,1,2,4,3    /),(/4,3/))
  type(cms_scalar_report), intent(out) :: report
  real(kindr2) :: cms_scale
  complex(kindr2) :: cms_original_p(6),cms_original_m(4)
  integer :: cms_pi(6),cms_mi(4)
  call cms_reset(report,16,4,8)
  if (initz) then
    report%status=CMS_UNINITIALIZED
    return
  endif
  if (.not.nonzerothrs) then
    report%status=CMS_DEFAULT_UNSUPPORTED
    return
  endif
  cms_original_p=0;cms_original_m=0
  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)
  cms_original_p(1)=p1
  cms_original_p(2)=p2
  cms_original_p(3)=p3
  cms_original_p(4)=p4
  cms_original_p(5)=p12
  cms_original_p(6)=p23
  cms_original_m(1)=m1
  cms_original_m(2)=m2
  cms_original_m(3)=m3
  cms_original_m(4)=m4
  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &
      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &
      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &
      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (onshellthrs.lt.RZRO) then
    report%status=CMS_BAD_INPUT
    return
  endif
  pp(1) = p1
  pp(2) = p2
  pp(3) = p3
  pp(4) = p4
  pp(5) = p12
  pp(6) = p23
  mm(1) = m1
  mm(2) = m2
  mm(3) = m3
  mm(4) = m4
  smax = 0
!
  do ii=1,6
    ap(ii) = abs(pp(ii))
    if (ap(ii).gt.smax) smax = ap(ii)
  enddo
!
  do ii=1,4
    am(ii) = abs(mm(ii))
    if (am(ii).gt.smax) smax = am(ii)
  enddo
!
  small = 0
  do ii=1,6
    hh = abs(ap(ii))
    if (hh.gt.small) small=hh
  enddo
  small = small*neglig(prcpar)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (smax.eq.RZRO) then
    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  cms_scale=smax
  if (.not.all(ieee_is_finite((/ap,am,small,mulocal2/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,4
    if (ap(ii).lt.hh) then
      call cms_record(report,23025,CMS_INPUT,CMS_ABS_REAL,ii,0, &
        'ap(ii)',ap(ii),cms_original_p(ii), &
        cmplx(pp(ii),kind=kindr2),hh,cms_scale)
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,23026,CMS_MASS,CMS_ABS_REAL,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale)
      am(ii)=0
    endif
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,4
    enddo
    endif
  endif
!
  jj = 1
  min56 = min(ap(5),ap(6))
  if (min56.lt.hh) then
    min13=min(ap(1),ap(3))
    min24=min(ap(2),ap(4))
    if     (min13.gt.min24.and.min13.gt.min56) then ;jj=2
    elseif (min24.gt.min13.and.min24.gt.min56) then ;jj=3
    else
    endif
  endif
!
  icase = 0
  do ii=1,4
    if (am(lm(ii,jj)).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(lp(permtable(1,icase),jj)) ;as(1)=ap(lp(permtable(1,icase),jj))
  cms_pi(1)=lp(permtable(1,icase),jj)
  ss(2)=pp(lp(permtable(2,icase),jj)) ;as(2)=ap(lp(permtable(2,icase),jj))
  cms_pi(2)=lp(permtable(2,icase),jj)
  ss(3)=pp(lp(permtable(3,icase),jj)) ;as(3)=ap(lp(permtable(3,icase),jj))
  cms_pi(3)=lp(permtable(3,icase),jj)
  ss(4)=pp(lp(permtable(4,icase),jj)) ;as(4)=ap(lp(permtable(4,icase),jj))
  cms_pi(4)=lp(permtable(4,icase),jj)
  ss(5)=pp(lp(permtable(5,icase),jj)) ;as(5)=ap(lp(permtable(5,icase),jj))
  cms_pi(5)=lp(permtable(5,icase),jj)
  ss(6)=pp(lp(permtable(6,icase),jj)) ;as(6)=ap(lp(permtable(6,icase),jj))
  cms_pi(6)=lp(permtable(6,icase),jj)
  rr(1)=mm(lm(permtable(1,icase),jj)) ;ar(1)=am(lm(permtable(1,icase),jj))
  cms_mi(1)=lm(permtable(1,icase),jj)
  rr(2)=mm(lm(permtable(2,icase),jj)) ;ar(2)=am(lm(permtable(2,icase),jj))
  cms_mi(2)=lm(permtable(2,icase),jj)
  rr(3)=mm(lm(permtable(3,icase),jj)) ;ar(3)=am(lm(permtable(3,icase),jj))
  cms_mi(3)=lm(permtable(3,icase),jj)
  rr(4)=mm(lm(permtable(4,icase),jj)) ;ar(4)=am(lm(permtable(4,icase),jj))
  cms_mi(4)=lm(permtable(4,icase),jj)
  icase = casetable(icase)
!
  s1r2 = abs(areal(ss(1)-rr(2))) + abs(aimag(ss(1)-rr(2)))
  s2r2 = abs(areal(ss(2)-rr(2))) + abs(aimag(ss(2)-rr(2)))
  s2r3 = abs(areal(ss(2)-rr(3))) + abs(aimag(ss(2)-rr(3)))
  s3r4 = abs(areal(ss(3)-rr(4))) + abs(aimag(ss(3)-rr(4)))
  s4r4 = abs(areal(ss(4)-rr(4))) + abs(aimag(ss(4)-rr(4)))
  if (.not.all(ieee_is_finite((/s1r2,s2r2,s2r3,s3r4,s4r4/)))) then
    report%status=CMS_BAD_INPUT
    return
  endif
  if (nonzerothrs) then
    if (s1r2.lt.hh) then
      call cms_record(report,23076,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(1),cms_mi(2), &
        's1r2',s1r2,cms_original_p(cms_pi(1))-cms_original_m(cms_mi(2)), &
        ss(1)-rr(2),hh,cms_scale)
      s1r2=0
    endif
    if (s2r2.lt.hh) then
      call cms_record(report,23077,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(2), &
        's2r2',s2r2,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(2)), &
        ss(2)-rr(2),hh,cms_scale)
      s2r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,23078,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale)
      s2r3=0
    endif
    if (s3r4.lt.hh) then
      call cms_record(report,23079,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(3),cms_mi(4), &
        's3r4',s3r4,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(4)), &
        ss(3)-rr(4),hh,cms_scale)
      s3r4=0
    endif
    if (s4r4.lt.hh) then
      call cms_record(report,23080,CMS_DIFFERENCE,CMS_L1_COMPLEX,cms_pi(4),cms_mi(4), &
        's4r4',s4r4,cms_original_p(cms_pi(4))-cms_original_m(cms_mi(4)), &
        ss(4)-rr(4),hh,cms_scale)
      s4r4=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  report%soft_permutation=jj
  end subroutine cms_preflight_d0rr

  subroutine cms_checked_d0rr(rslt,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4,report)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  complex(kindr2), intent(inout) :: rslt(0:2)
  real(kindr2), intent(in) :: p1,p2,p3,p4,p12,p23
  real(kindr2), intent(in) :: m1,m2,m3,m4
  type(cms_scalar_report), intent(out) :: report
  complex(kindr2) :: cms_result_1(0:2)
  call cms_preflight_d0rr(report,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (report%status.ne.CMS_OK) return
  report%native_called=.true.
  call d0rr(cms_result_1,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4)
  if (.not.all(ieee_is_finite(real((/cms_result_1/),kindr2))).or. &
      .not.all(ieee_is_finite(aimag((/cms_result_1/))))) then
    report%status=CMS_NONFINITE_RESULT
    return
  endif
  rslt=cms_result_1
  report%result_valid=.true.
  end subroutine cms_checked_d0rr
  subroutine fixture_check()
    type(cms_scalar_report)::report
    real(kindr2)::p(6),rm(4),p_before(6),rm_before(4)
    complex(kindr2)::cm(4),r(0:2,4),sentinel,cm_before(4)
    integer::i,before
    integer(8)::p_words(6*kindr2/8),rm_words(4*kindr2/8),cm_words(8*kindr2/8),r_words(24*kindr2/8)
    sentinel=cmplx(1234,5678,kind=kindr2)
    ! qp a0_c: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.1,103)
    call insist(mock_last_args(1).eq.cm(1),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.1,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp a0_r: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_a0_r(r(:,1),rm(1),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.1,103)
    call insist(mock_last_args(1).eq.cmplx(rm(1),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_a0_r(r(:,1),rm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.1,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.1,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp b11rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),p(1),cm(1),cm(2),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.3,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cm(1),104)
    call insist(mock_last_args(3).eq.cm(2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),p(1),cm(1),cm(2),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.3,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.2,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp b11rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_b11rr(r(:,1),r(:,2),r(:,3),r(:,4),p(1),rm(1),rm(2),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.3,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(rm(2),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_b11rr(r(:,1),r(:,2),r(:,3),r(:,4),p(1),rm(1),rm(2),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.3,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.2,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp c0rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_c0rc(r(:,1),p(1),p(2),p(3),cm(1),cm(2),cm(3),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.6,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cm(1),104)
    call insist(mock_last_args(5).eq.cm(2),104)
    call insist(mock_last_args(6).eq.cm(3),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_c0rc(r(:,1),p(1),p(2),p(3),cm(1),cm(2),cm(3),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.9,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.3,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp c0rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.6,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(rm(2),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(rm(3),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.9,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.3,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp d0rc: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_d0rc(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),cm(1),cm(2),cm(3),cm(4),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.10,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(p(4),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(p(5),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(p(6),kind=kindr2),104)
    call insist(mock_last_args(7).eq.cm(1),104)
    call insist(mock_last_args(8).eq.cm(2),104)
    call insist(mock_last_args(9).eq.cm(3),104)
    call insist(mock_last_args(10).eq.cm(4),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_d0rc(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),cm(1),cm(2),cm(3),cm(4),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.13,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.4,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! qp d0rr: accepted callback receives original inputs.
    p=-10;rm=20;cm=cmplx(20,-1,kind=kindr2);r=sentinel
    before=mock_calls
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_OK.and.report%result_valid.and.report%native_called,101)
    call insist(mock_calls.eq.before+1.and.all(r(:,1).eq.cmplx(991,0,kind=kindr2)),102)
    call insist(mock_nargs.eq.10,103)
    call insist(mock_last_args(1).eq.cmplx(p(1),kind=kindr2),104)
    call insist(mock_last_args(2).eq.cmplx(p(2),kind=kindr2),104)
    call insist(mock_last_args(3).eq.cmplx(p(3),kind=kindr2),104)
    call insist(mock_last_args(4).eq.cmplx(p(4),kind=kindr2),104)
    call insist(mock_last_args(5).eq.cmplx(p(5),kind=kindr2),104)
    call insist(mock_last_args(6).eq.cmplx(p(6),kind=kindr2),104)
    call insist(mock_last_args(7).eq.cmplx(rm(1),kind=kindr2),104)
    call insist(mock_last_args(8).eq.cmplx(rm(2),kind=kindr2),104)
    call insist(mock_last_args(9).eq.cmplx(rm(3),kind=kindr2),104)
    call insist(mock_last_args(10).eq.cmplx(rm(4),kind=kindr2),104)
    do i=1,6
      p(i)=real(i,kindr2)*onshellthrs/20
    enddo
    do i=1,4
      rm(i)=real(i+6,kindr2)*onshellthrs/20
      cm(i)=cmplx(rm(i),-onshellthrs/100,kind=kindr2)
    enddo
    r=sentinel;before=mock_calls
    p_before=p;rm_before=rm;cm_before=cm
    p_words=transfer(p,p_words);rm_words=transfer(rm,rm_words)
    cm_words=transfer(cm,cm_words);r_words=transfer(r,r_words)
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%count.eq.13,105)
    call insist(.not.report%result_valid.and..not.report%native_called.and.mock_calls.eq.before,106)
    call insist(all(r.eq.sentinel).and.all(p.eq.p_before).and.all(rm.eq.rm_before).and.all(cm.eq.cm_before),107)
    call insist(all(transfer(p,p_words).eq.p_words).and.all(transfer(rm,rm_words).eq.rm_words),112)
    call insist(all(transfer(cm,cm_words).eq.cm_words).and.all(transfer(r,r_words).eq.r_words),113)
    call insist(report%dropped.eq.0,108)
    do i=1,report%count
      call insist(report%events(i)%native_metric.gt.0.and.report%events(i)%native_metric.lt.report%events(i)%threshold,109)
      call insist(report%events(i)%precision.eq.16.and.report%events(i)%topology.eq.4,110)
      call insist(report%events(i)%source_line.gt.0,111)
    enddo
    ! Strict inequality, exact zeros and real/complex norm distinctions.
    r=sentinel;before=mock_calls
    call cms_checked_a0_r(r(:,1),onshellthrs,report)
    call insist(report%status.eq.CMS_OK.and.mock_calls.eq.before+1,120)
    call cms_checked_a0_r(r(:,1),0._kindr2,report)
    call insist(report%status.eq.CMS_OK.and.report%count.eq.0,121)
    cm(1)=cmplx(0.6_kindr2*onshellthrs,-0.6_kindr2*onshellthrs,kind=kindr2)
    call cms_checked_a0_c(r(:,1),cm(1),report)
    call insist(report%status.eq.CMS_REFUSED.and.report%events(1)%metric_norm.eq.CMS_ABS_COMPLEX,122)
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_OK,123)
    ! All-zero triangle/box are explicit unsupported results; A/B exact-zero remains delegated.
    p=0;rm=0;r=sentinel;before=mock_calls
    call cms_checked_c0rr(r(:,1),p(1),p(2),p(3),rm(1),rm(2),rm(3),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.mock_calls.eq.before,124)
    call cms_checked_d0rr(r(:,1),p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4),report)
    call insist(report%status.eq.CMS_ZERO_TOPOLOGY_UNSUPPORTED.and.all(r.eq.sentinel),125)
    nonzerothrs=.false.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_DEFAULT_UNSUPPORTED.and..not.report%native_called,126)
    nonzerothrs=.true.;initz=.true.
    call cms_checked_a0_r(r(:,1),1._kindr2,report)
    call insist(report%status.eq.CMS_UNINITIALIZED.and..not.report%native_called,127)
    initz=.false.
    ! Opposite mass positions exercise the exact triangle permutation table.
    p=-10;rm=0;rm(1)=20
    call cms_preflight_c0rr(report,p(1),p(2),p(3),rm(1),rm(2),rm(3))
    call insist(all(report%p_sources(1:3).eq.(/2,3,1/)),128)
    call insist(all(report%mass_sources(1:3).eq.(/2,3,1/)),129)
    ! Soft box routing jj=2 when the 1/3 channel dominates 2/4 and 5/6.
    p=(/10._kindr2,2._kindr2,10._kindr2,2._kindr2,0._kindr2,0._kindr2/);rm=20
    call cms_preflight_d0rr(report,p(1),p(2),p(3),p(4),p(5),p(6),rm(1),rm(2),rm(3),rm(4))
    call insist(report%soft_permutation.eq.2,130)
    call insist(all(report%p_sources.eq.(/5,2,6,4,1,3/)),131)
    ! A forced observer overflow is still a refusal, never a success.
    call cms_reset(report,kindr2,4,7)
    do i=1,CMS_MAX_EVENTS+1
      call cms_record(report,1,CMS_INPUT,CMS_ABS_REAL,1,0,'forced',onshellthrs/2, &
        cmplx(onshellthrs/2,0,kindr2),cmplx(onshellthrs/2,0,kindr2),onshellthrs,1._kindr2)
    enddo
    call insist(report%status.eq.CMS_EVENT_OVERFLOW.and.report%count.eq.CMS_MAX_EVENTS.and.report%dropped.eq.1,132)
    call insist(.not.report%result_valid.and..not.report%native_called,133)
    ! Native callbacks with nonfinite output must not publish caller buffers.
    r=sentinel;before=mock_calls;mock_nan=.true.
    call cms_checked_a0_r(r(:,1),20._kindr2,report)
    call insist(report%status.eq.CMS_NONFINITE_RESULT.and.report%native_called,134)
    call insist(.not.report%result_valid.and.all(r.eq.sentinel).and.mock_calls.eq.before+1,135)
    mock_nan=.false.
    ! Finite input whose native L1 norm overflows is refused before callback.
    cm(1)=cmplx(huge(0._kindr2),-huge(0._kindr2),kind=kindr2);before=mock_calls
    call cms_checked_b11rc(r(:,1),r(:,2),r(:,3),r(:,4),-10._kindr2,cm(1),cmplx(10,0,kindr2),report)
    call insist(report%status.eq.CMS_BAD_INPUT.and..not.report%native_called.and.mock_calls.eq.before,136)
    write(*,*) 'CLASSIFIER_MOCK_COMPLETE',kindr2,8
  end subroutine
  subroutine insist(ok,code)
    logical,intent(in)::ok
    integer,intent(in)::code
    if (.not.ok) then
      write(*,*) 'MOCK_FAILURE',kindr2,code
      error stop 1
    endif
  end subroutine
end module
