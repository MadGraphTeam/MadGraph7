  subroutine a0_r( rslt ,mm  )
!
  use avh_olo_forCutTools_dp_bub ,only: tadp
!
  complex(kindr2) &   
    ,intent(out) :: rslt(0:2)
  real(kindr2) &  
    ,intent(in)  :: mm
!
  complex(kindr2) &   
    :: ss
  real(kindr2) &  
    :: am,hh,mulocal,mulocal2
  character(25+99) ,parameter :: warning=&
                     'WARNING from OneLOop a0: '//warnonshell
  if (initz) call init
!
  mulocal = muscale 
!
  am = abs(mm)
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    hh = onshellthrs
    if (am.lt.hh) am = 0
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(am,mulocal2)
    if (RZRO.lt.am.and.am.lt.hh) write(wunit,*) warning
  endif
!
  ss = mm
  call tadp( rslt ,ss ,am ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' mm:',trim(myprint(mm))
    write(punit,*) 'a0(2):',trim(myprint(rslt(2)))
    write(punit,*) 'a0(1):',trim(myprint(rslt(1)))
    write(punit,*) 'a0(0):',trim(myprint(rslt(0)))
  endif
  end subroutine

