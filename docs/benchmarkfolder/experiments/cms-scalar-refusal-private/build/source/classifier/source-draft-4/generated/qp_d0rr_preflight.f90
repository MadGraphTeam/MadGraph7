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

