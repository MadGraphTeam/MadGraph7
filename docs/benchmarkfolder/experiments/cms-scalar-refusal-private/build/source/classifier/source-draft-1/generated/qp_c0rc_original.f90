  subroutine c0rc( rslt ,p1,p2,p3 ,m1,m2,m3 )
  use avh_olo_forCutTools_qp_tri
  use avh_olo_forCutTools_qp_auxfun ,only: kallen
!
  complex(kindr2) &   
    ,intent(out) :: rslt(0:2)
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
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop c0: '//warnonshell
  if (initz) call init
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
      if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
        ,'mass-squared has positive imaginary part, switching its sign.'
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
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop c0: ' &
      ,'all input equal zero, returning 0'
    rslt(0)=0 ;rslt(1)=0 ;rslt(2)=0
    return
  endif
!
  if (mulocal2.gt.smax) smax = mulocal2
!
  if (nonzerothrs) then
    hh = onshellthrs
    do ii=1,3
      if (ap(ii).lt.hh) ap(ii) = 0
      if (am(ii).lt.hh) am(ii) = 0
    enddo
  else
    hh = onshellthrs*smax
    if (wunit.gt.0) then
    do ii=1,3
      if (RZRO.lt.ap(ii).and.ap(ii).lt.hh) write(wunit,*) warning
      if (RZRO.lt.am(ii).and.am(ii).lt.hh) write(wunit,*) warning
    enddo
    endif
  endif
!
  icase = 0
  do ii=1,3
    if (am(ii).gt.RZRO) icase = icase + base(ii)
  enddo
  ss(1)=pp(permtable(1,icase)) ;as(1)=ap(permtable(1,icase))
  ss(2)=pp(permtable(2,icase)) ;as(2)=ap(permtable(2,icase))
  ss(3)=pp(permtable(3,icase)) ;as(3)=ap(permtable(3,icase))
  rr(1)=mm(permtable(1,icase)) ;ar(1)=am(permtable(1,icase))
  rr(2)=mm(permtable(2,icase)) ;ar(2)=am(permtable(2,icase))
  rr(3)=mm(permtable(3,icase)) ;ar(3)=am(permtable(3,icase))
  icase = casetable(icase)
!
  s1r2 = abs(ss(1)-rr(2))
  s2r3 = abs(ss(2)-rr(3))
  s3r3 = abs(ss(3)-rr(3))
  if (nonzerothrs) then
    if (s1r2.lt.hh) s1r2 = 0
    if (s2r3.lt.hh) s2r3 = 0
    if (s3r3.lt.hh) s3r3 = 0
  elseif (wunit.gt.0) then
    if (RZRO.lt.s1r2.and.s1r2.lt.hh) write(wunit,*) warning
    if (RZRO.lt.s2r3.and.s2r3.lt.hh) write(wunit,*) warning
    if (RZRO.lt.s3r3.and.s3r3.lt.hh) write(wunit,*) warning
  endif
!
  if     (icase.eq.3) then
! 3 non-zero internal masses
    lambda = kallen( ss(1),ss(2),ss(3) )
    if (areal(lambda).lt.RZRO) then
      call trif3HV( rslt ,ss,rr ,as ,smax ,lambda )
    else
      call trif3( rslt ,ss(1),ss(2),ss(3) ,rr(1),rr(2),rr(3) )
    endif
  elseif (icase.eq.2) then
! 2 non-zero internal masses
    if (s1r2.ne.RZRO.or.s3r3.ne.RZRO) then
      call trif2( rslt ,ss(1),ss(2),ss(3) ,rr(2),rr(3) )
    else
      call tria4( rslt ,ss(2) ,rr(2),rr(3) ,mulocal2 )
    endif
  elseif (icase.eq.1) then
! 1 non-zero internal mass
    if     (as(1).ne.RZRO) then
      call trif1( rslt ,ss(1),ss(2),ss(3), rr(3) )
    elseif (s2r3.ne.RZRO) then
      if   (s3r3.ne.RZRO) then
        call tria3( rslt ,ss(2),ss(3) ,rr(3) ,mulocal2 )
      else
        call tria2( rslt ,ss(2) ,rr(3) ,mulocal2 )
      endif
    elseif (s3r3.ne.RZRO) then
      call tria2( rslt ,ss(3) ,rr(3) ,mulocal2 )
    else
      call tria1( rslt ,rr(3) ,mulocal2 )
    endif
  else
! 0 non-zero internal masses
    call tria0( rslt ,ss ,as ,mulocal2 )
  endif
! exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
  rslt(0) = rslt(0) + 2*PISQo24*rslt(2)
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' p1:',trim(myprint(p1))
    write(punit,*) ' p2:',trim(myprint(p2))
    write(punit,*) ' p3:',trim(myprint(p3))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) ' m3:',trim(myprint(m3))
    write(punit,*) 'c0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'c0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'c0(0):',trim(myprint(rslt(0)))
  endif
  end subroutine

