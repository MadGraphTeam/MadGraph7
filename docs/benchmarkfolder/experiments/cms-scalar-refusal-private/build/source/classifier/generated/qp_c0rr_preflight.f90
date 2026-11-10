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
        cmplx(pp(ii),kind=kindr2),hh,cms_scale, &
        cms_original_p(ii),cmplx(0,0,kind=kindr2))
      ap(ii)=0
    endif
    if (am(ii).lt.hh) then
      call cms_record(report,21603,CMS_MASS,CMS_ABS_REAL,0,ii, &
        'am(ii)',am(ii),cms_original_m(ii), &
        cmplx(mm(ii),kind=kindr2),hh,cms_scale, &
        cmplx(0,0,kind=kindr2),cms_original_m(ii))
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
        ss(1)-rr(2),hh,cms_scale, &
        cms_original_p(cms_pi(1)),cms_original_m(cms_mi(2)))
      s1r2=0
    endif
    if (s2r3.lt.hh) then
      call cms_record(report,21632,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(2),cms_mi(3), &
        's2r3',s2r3,cms_original_p(cms_pi(2))-cms_original_m(cms_mi(3)), &
        ss(2)-rr(3),hh,cms_scale, &
        cms_original_p(cms_pi(2)),cms_original_m(cms_mi(3)))
      s2r3=0
    endif
    if (s3r3.lt.hh) then
      call cms_record(report,21633,CMS_DIFFERENCE,CMS_ABS_COMPLEX,cms_pi(3),cms_mi(3), &
        's3r3',s3r3,cms_original_p(cms_pi(3))-cms_original_m(cms_mi(3)), &
        ss(3)-rr(3),hh,cms_scale, &
        cms_original_p(cms_pi(3)),cms_original_m(cms_mi(3)))
      s3r3=0
    endif
  elseif (wunit.gt.0) then
  endif
!
  report%p_sources=cms_pi
  report%mass_sources=cms_mi
  report%classification_case=icase
  end subroutine cms_preflight_c0rr

