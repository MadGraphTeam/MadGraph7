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
        cmplx(mm,kind=kindr2),hh,cms_scale, &
        cmplx(0,0,kind=kindr2),cms_original_m(1))
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

