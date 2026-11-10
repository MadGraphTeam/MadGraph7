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
        ss,hh,cms_scale, &
        cms_original_p(1),cmplx(0,0,kind=kindr2))
      app=0
    endif
    if (am1.lt.hh) then
      call cms_record(report,20050,CMS_MASS,CMS_ABS_REAL,0,1, &
        'am1',am1,cms_original_m(1), &
        r1,hh,cms_scale, &
        cmplx(0,0,kind=kindr2),cms_original_m(1))
      am1=0
    endif
    if (am2.lt.hh) then
      call cms_record(report,20051,CMS_MASS,CMS_ABS_REAL,0,2, &
        'am2',am2,cms_original_m(2), &
        r2,hh,cms_scale, &
        cmplx(0,0,kind=kindr2),cms_original_m(2))
      am2=0
    endif
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  end subroutine cms_preflight_b11rr

