  subroutine b11rc( b11,b00,b1,b0 ,pp,m1,m2 )
!
  use avh_olo_forCutTools_dp_bub ,only: bub11
!
  complex(kindr2) &   
    ,intent(out) :: b11(0:2),b00(0:2),b1(0:2),b0(0:2)
  real(kindr2) &  
    ,intent(in)  :: pp
  complex(kindr2) &   
    ,intent(in)  :: m1,m2
!
  complex(kindr2) &   
    :: ss,r1,r2
  real(kindr2) &  
    :: app,am1,am2,hh,mulocal,mulocal2
  character(26+99) ,parameter :: warning=&
                     'WARNING from OneLOop b11: '//warnonshell
  if (initz) call init
  ss = pp
  r1 = m1
  r2 = m2
!
  app = abs(pp)
!
  am1 = areal(r1)
  hh  = aimag(r1)
  if (hh.gt.RZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r1 has positive imaginary part, switching its sign.'
    r1 = acmplx( am1 ,-hh )
  endif
  am1 = abs(am1) + abs(hh)
!
  am2 = areal(r2)
  hh  = aimag(r2)
  if (hh.gt.RZRO) then
    if (eunit.gt.0) write(eunit,*) 'ERROR in OneLOop b11: ' &
      ,'r2 has positive imaginary part, switching its sign.'
    r2 = acmplx( am2 ,-hh )
  endif
  am2 = abs(am2) + abs(hh)
!
  mulocal = muscale 
!
  mulocal2 = mulocal*mulocal
!
  if (nonzerothrs) then
    hh = onshellthrs
    if (app.lt.hh) app = 0
    if (am1.lt.hh) am1 = 0
    if (am2.lt.hh) am2 = 0
  elseif (wunit.gt.0) then
    hh = onshellthrs*max(app,max(am1,max(am2,mulocal2)))
    if (RZRO.lt.app.and.app.lt.hh) write(wunit,*) warning
    if (RZRO.lt.am1.and.am1.lt.hh) write(wunit,*) warning
    if (RZRO.lt.am2.and.am2.lt.hh) write(wunit,*) warning
  endif
!
  call bub11( b11,b00,b1,b0 ,ss,r1,r2 ,app,am1,am2 ,mulocal2 )
!
  if (punit.gt.0) then
    if (nonzerothrs) write(punit,*) 'onshell:',trim(myprint(onshellthrs))
    write(punit,*) 'muscale:',trim(myprint(mulocal))
    write(punit,*) ' pp:',trim(myprint(pp))
    write(punit,*) ' m1:',trim(myprint(m1))
    write(punit,*) ' m2:',trim(myprint(m2))
    write(punit,*) 'b11(2):',trim(myprint(b11(2)))
    write(punit,*) 'b11(1):',trim(myprint(b11(1)))
    write(punit,*) 'b11(0):',trim(myprint(b11(0)))
    write(punit,*) 'b00(2):',trim(myprint(b00(2)))
    write(punit,*) 'b00(1):',trim(myprint(b00(1)))
    write(punit,*) 'b00(0):',trim(myprint(b00(0)))
    write(punit,*) ' b1(2):',trim(myprint(b1(2) ))
    write(punit,*) ' b1(1):',trim(myprint(b1(1) ))
    write(punit,*) ' b1(0):',trim(myprint(b1(0) ))
    write(punit,*) ' b0(2):',trim(myprint(b0(2) ))
    write(punit,*) ' b0(1):',trim(myprint(b0(1) ))
    write(punit,*) ' b0(0):',trim(myprint(b0(0) ))
  endif
  end subroutine

