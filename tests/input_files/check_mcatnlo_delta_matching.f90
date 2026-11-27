! Exercise the production compute_delta routine with controlled Born/real
! records and a simple Sudakov table. There is no shower runtime interface.
program check_mcatnlo_delta_matching
  use mc_counterterms, only: compute_delta,isspecial,delta_connections, &
       delta_leg_probability,get_delta_connections,delta_backward_leg, &
       delta_alphas_lambdas,delta_H_min_probability
  use FKSParams, only: delta_backward_isr
  use fks_phase_space_data, only: xbjrk_cnt
  use process_module
  use scale_module
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer :: idup(4,1),mothup(2,4,1),icolup(2,4,1)
  common /test_born/ idup,mothup,icolup
  integer :: i_fks,j_fks,nfksprocess,mccntcalled,fold,ifold_counter,lpp(2)
  common /fks_indices/ i_fks,j_fks
  common /c_nfksprocess/ nfksprocess
  common /c_MCcntcalled/ mccntcalled
  common /cfl/ fold,ifold_counter
  common /test_lpp/ lpp
  double precision :: seen_scales(4,4)
  logical*1 :: seen_dead(4,4)
  common /test_stopping/ seen_scales,seen_dead
  integer :: table_calls
  common /test_table_calls/ table_calls
  integer :: table_mode
  common /test_table_mode/ table_mode
  integer :: mass_calls(4:5)
  common /test_mass_calls/ mass_calls
  integer :: fixture_kind,profile_id,metadata_calls
  common /test_provider/ fixture_kind,profile_id,metadata_calls
  double precision :: p(0:3,5),prob,expected,t34,t43
  type(delta_connections) :: connections
  double precision :: mcmass(21)
  logical :: unit_table
  common /test_unit_table/ unit_table
  character(len=64) :: mode

  call get_command_argument(1,mode)
  call init_process_module_global('PYTHIA8   ','born',5,2,.true.,1000d0,1,1,0)
  call init_scale_module(5,1d0,1,1)
  table_mode=0
  unit_table=.false.
  if(mode.eq.'cross'.or.mode.eq.'zero')then
    call check_connection_probabilities(mode)
    stop
  endif
  if(mode.eq.'hmin_fsr'.or.mode.eq.'hmin_isr'.or.mode.eq.'hmin_hole')then
    call check_H_min_probability(mode)
    stop
  endif
  born_flow_picked=1
  i_fks=5
  j_fks=3
  call RealToBornMapping(i_fks)
  nfksprocess=1
  ifold_counter=1
  fold=0
  lpp=0
  isspecial=.false.
  table_calls=0
  fixture_kind=0
  profile_id=1
  metadata_calls=0
  delta_backward_isr=trim(mode)=='matching_backward'
  unit_table=.false.
  mass_calls=0
  mccntcalled=0
  valid_dipole_n=.false.
  valid_dipole_n(3,4,1)=.true.
  valid_dipole_n(4,3,1)=.true.
  valid_dipole_n1=.false.
  idup(:,1)=(/11,-11,1,-1/)
  mothup=0
  icolup=0
  icolup(1,3,1)=501
  icolup(2,4,1)=501
  if (trim(mode) == 'invalid') icolup(2,4,1)=999
  shower_scale_nbody=50d0
  p(:,1)=(/500d0,0d0,0d0,500d0/)
  p(:,2)=(/500d0,0d0,0d0,-500d0/)
  p(:,3)=(/40d0,40d0,0d0,0d0/)
  p(:,4)=(/30d0,-30d0,0d0,0d0/)
  p(:,5)=(/30d0,0d0,30d0,0d0/)
  mcmass=0d0
  include 'MCmasses_PYTHIA8.inc'

  if(index(trim(mode),'backward_')==1.or.index(trim(mode),'legacy_')==1)then
    fixture_kind=1
    if(index(trim(mode),'gluon')>0)fixture_kind=2
    delta_backward_isr=index(trim(mode),'backward_')==1
    if(index(trim(mode),'_one')>0)profile_id=2
    if(index(trim(mode),'_unknown')>0)profile_id=3
    if(index(trim(mode),'_order')>0)profile_id=4
    if(index(trim(mode),'_cmw')>0)profile_id=5
    if(index(trim(mode),'_nf')>0)profile_id=6
    if(index(trim(mode),'_threshold')>0)profile_id=7
    if(index(trim(mode),'_pdfmode')>0)profile_id=8
    if(index(trim(mode),'_nan')>0)profile_id=9
    if(index(trim(mode),'_six')>0)profile_id=10
    idup(:,1)=[2,-2,25,25]
    if(fixture_kind==2)idup(:,1)=[21,21,25,25]
    icolup=0
    icolup(1,1,1)=501
    icolup(2,2,1)=501
    if(fixture_kind==2)then
      icolup(2,1,1)=502
      icolup(1,2,1)=502
      isspecial=.true.
    endif
    valid_dipole_n=.false.
    valid_dipole_n(1,2,1)=.true.
    valid_dipole_n(2,1,1)=.true.
    j_fks=1
    lpp=1
    xbjrk_cnt=0.01d0
    if(index(trim(mode),'_cutoff')>0)p(:,5)=1d-4*p(:,5)
    call check_incoming
    print *, 'PASS incoming Delta backward dispatch'
    stop
  endif

  call compute_delta(p,prob)
  if (trim(mode) == 'invalid') stop 2
  t34=sqrt(201600d0)/19d0
  t43=sqrt(3850d0)/3d0
  expected=t34*t43/50d0**2
  call require(abs(prob-expected) < 1d-12,'Delta product of scale ratios')
  call require(abs(seen_scales(3,4)-t34) < 1d-12,'native stopping scale reaches H assignment')
  call require(abs(emsca_H(1,1,3,4)-t34) < 1d-12,'H scale stored for selected fold')
  call require(logical(.not.seen_dead(3,4) .and. .not.seen_dead(4,3)),'live FF dipoles')
  call require(mccntcalled == 8 .and. table_calls == 5,'Delta table calls and bookkeeping')
  call require(all(mass_calls == 1),'charm and bottom model masses both requested')
  call require(metadata_calls==0,'final-state radiators do not request ISR table metadata')

  ! The FxFx scalar must follow the reconstructed scales, not the Born scale.
  ickkw_mod=3
  emsca_H_hard=500d0
  call set_FxFx_delta_H_scale(1,1,.false.)
  call require(abs(emsca_H_hard(1,1)-sqrt(t34*t43)).lt.1d-12, &
       'FxFx H scalar follows native stopping scales')
  call set_FxFx_delta_H_scale(1,1,.true.)
  call require(emsca_H_hard(1,1).eq.min(t34,t43), &
       'minimal FxFx H scalar is the smallest native stopping scale')
  ickkw_mod=0

  ! Lowering the starting scale vetoes both dipoles before the Sudakov calls.
  shower_scale_nbody=1d0
  mccntcalled=0
  call compute_delta(p,prob)
  call require(logical(seen_dead(3,4) .and. seen_dead(4,3)),'starting-scale veto preserved')
  call require(prob == 1d0 .and. table_calls == 5,'dead dipoles do not enter Sudakov product')
  call require(all(mass_calls == 2),'model masses requested for each Delta calculation')

  ! Both colour-line weights vanish when table ratios round to unity.
  ! The survival probability must remain finite and equal to one.
  connections%count=2
  connections%start=50d0
  connections%stop=nearest(50d0,-1d0)
  connections%mass=50d0
  connections%sudakov_type=2
  mcmass=0d0
  unit_table=.true.
  prob=delta_leg_probability(3,21,[21],0,0d0,connections,mcmass)
  call require(prob==1d0,'rounded unit two-line Sudakov ratios remain finite')
  print *, 'PASS native Delta matching and starting-scale veto'
contains
  subroutine check_incoming
    double precision :: legacy,backward,lambda(3:6),alpha,leg_prob
    integer :: leg,order,legacy_calls,nfmax
    logical :: below_cutoff
    call compute_delta(p,prob)
    below_cutoff=index(trim(mode),'_cutoff')>0
    alpha=0.118d0
    order=2
    if(profile_id==2)then
      alpha=0.1365d0
      order=1
    endif
    nfmax=5
    if(profile_id==10)nfmax=6
    call delta_alphas_lambdas(alpha,order,nfmax,lambda)
    legacy=1d0
    backward=1d0
    legacy_calls=table_calls
    do leg=1,2
      connections%count=fixture_kind
      connections%partner=3-leg
      connections%start=50d0
      connections%stop=seen_scales(leg,3-leg)
      connections%mass=50d0
      connections%sudakov_type=1
      leg_prob=delta_leg_probability(leg,idup(leg,1),idup(leg,:),1, &
           0.01d0,connections,mcmass)
      legacy=legacy*leg_prob
      leg_prob=delta_backward_leg(idup(leg,1),idup(leg,:),1, &
           0.01d0,connections,lambda,order)
      if(below_cutoff)then
        call require(leg_prob<0d0,'below-cutoff backward leg requests table fallback')
      else
        call require(leg_prob>=0d0,'resolved incoming backward leg has support')
        backward=backward*leg_prob
      endif
    enddo
    if(delta_backward_isr)then
      call require(metadata_calls==1,'table coupling profile is read once')
      if(below_cutoff)then
        call require(prob==legacy,'below-cutoff source equals legacy table bitwise')
        call require(legacy_calls>1,'fallback source calls table')
      else
        call require(abs(prob-backward)<1d-14,'source uses table-profile backward leg product')
        call require(abs(prob-legacy)>1d-5,'backward source is distinct from legacy table')
        call require(legacy_calls==1,'backward source only initializes table bounds')
      endif
    else
      call require(metadata_calls==0,'disabled backward switch never requests metadata')
      call require(prob==legacy,'disabled incoming source equals legacy table bitwise')
      call require(legacy_calls>1,'disabled incoming source uses table')
    endif
  end subroutine check_incoming
  subroutine check_connection_probabilities(mode)
    character(len=*),intent(in) :: mode
    type(delta_connections) :: connection,clipped(4)
    double precision :: mcmass(21),answer,reference,e1,e2,g1,g2
    double precision :: starts(4,4),stops(4,4),masses(4,4)
    logical*1 :: dead(4,4)
    integer :: permutation
    mcmass=0d0
    table_calls=0
    connection%count=2
    connection%partner=[1,2]
    connection%start=[100d0,20d0]
    connection%stop=[50d0,10d0]
    connection%mass=[1d0,2d0]
    connection%sudakov_type=[1,3]
    e1=1.1d0
    e2=2.3d0
    table_mode=1
    if(mode.eq.'cross')then
      ! The stop=50 line cannot radiate on the other line's start=20
      ! interval. The controlled table rejects even evaluating that ratio.
      reference=(e1*0.5d0**e1+e2*0.1d0**e1*0.5d0**e2)/(e1+e2)
      do permutation=1,2
        table_calls=0
        answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
        call require(ieee_is_finite(answer).and.abs(answer-reference).lt.1d-13, &
             'unequal cross-connection interval and table ownership')
        connection%partner=connection%partner(2:1:-1)
        connection%start=connection%start(2:1:-1)
        connection%stop=connection%stop(2:1:-1)
        connection%mass=connection%mass(2:1:-1)
        connection%sudakov_type=connection%sudakov_type(2:1:-1)
      enddo
      ! When both cross intervals are nonempty all four physical ratios
      ! contribute, with their own gamma-line mass and Sudakov type.
      table_mode=2
      connection%start=[100d0,80d0]
      connection%stop=[20d0,50d0]
      g1=e1*log(5d0)
      g2=e2*log(1.6d0)
      reference=(g1*0.2d0**e1*0.25d0**e2+g2*0.5d0**e1*0.625d0**e2)/(g1+g2)
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(ieee_is_finite(answer).and.abs(answer-reference).lt.1d-13, &
           'nonempty cross-connection intervals')
      connection%count=1
      reference=.2d0**e1
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(ieee_is_finite(answer).and.abs(answer-reference).lt.1d-13,'single colour line')
      connection%count=0
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(answer.eq.1d0,'no live colour line')
      print *, 'PASS cross-connection intervals'
    else
      ! Equal diagonal scales make both logarithmic weights vanish. Keep
      ! the symmetric normalization convention, including a live cross term.
      connection%stop=connection%start
      reference=(1d0+.2d0**e1)/2d0
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(ieee_is_finite(answer).and.abs(answer-reference).lt.1d-13, &
           'vanishing diagonal weights with unequal scales')
      connection%start=20d0
      connection%stop=20d0
      table_calls=0
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(answer.eq.1d0.and.table_calls.le.4,'equal scales give unit cross factors')
      ! A flat table also gives zero weights with nonzero diagonal intervals.
      table_mode=3
      connection%start=[100d0,20d0]
      connection%stop=[50d0,10d0]
      answer=delta_leg_probability(3,21,[21],0,.1d0,connection,mcmass)
      call require(answer.eq.1d0,'flat Sudakov table zero-weight normalization')
      ! The live-connection selector can turn distinct scales into equal
      ! scales when both exceed the upper end of the Sudakov table.
      valid_dipole_n=.false.
      valid_dipole_n(3,1:2,1)=.true.
      starts=0d0
      stops=0d0
      masses=1d0
      dead=.true.
      starts(3,1:2)=[2000d0,3000d0]
      stops(3,1:2)=[1500d0,2500d0]
      masses(3,1:2)=[1d0,2d0]
      dead(3,1:2)=.false.
      call get_delta_connections(starts,stops,dead,masses,[11,-11,21,1], &
           1,.false.,1d0,1000d0,clipped)
      call require(clipped(3)%count.eq.2.and.all(clipped(3)%start.eq.1000d0).and. &
           all(clipped(3)%stop.eq.1000d0),'table-clipped equal scales')
      table_mode=2
      answer=delta_leg_probability(3,21,[21],0,.1d0,clipped(3),mcmass)
      call require(answer.eq.1d0,'clipped intervals have finite unit probability')
      print *, 'PASS zero connection weights'
    endif
  end subroutine check_connection_probabilities

  subroutine check_H_min_probability(mode)
    ! H-event lines from their directed starts down to the smallest one.
    ! With table_mode=2 a line from t down to q gives (q/t)**(m+0.1*kind),
    ! m the (capped) H-event dipole mass. Both PDF stubs are scale-free.
    use fks_phase_space_data, only: xbjrk_ev
    character(len=*),intent(in) :: mode
    integer :: icolup_s(2,4),icolup_h(2,5)
    common/colour_connections/icolup_s,icolup_h
    double precision :: cstlow,cstupp,cxmlow,cxmupp
    common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp
    double precision :: ph(0:3,5),q,answer,reference,m35,m45
    cstlow=1d0
    cstupp=1000d0
    cxmlow=1d0
    cxmupp=50d0
    table_mode=2
    emsca_H=-1d0
    icolup_h=0
    ph(:,1)=(/500d0,0d0,0d0,500d0/)
    ph(:,2)=(/500d0,0d0,0d0,-500d0/)
    ph(:,3)=(/40d0,40d0,0d0,0d0/)
    ph(:,4)=(/30d0,-30d0,0d0,0d0/)
    ph(:,5)=(/30d0,0d0,30d0,0d0/)
    q=36d0
    if(mode.eq.'hmin_hole')then
      ! As hmin_fsr, but the table has a statistical zero at 39.5-40.5 GeV
      ! (table_mode=4) under the start of the 3-5 line. The monotone
      ! envelope reads that start at 40/1.04 GeV, the next value below.
      table_mode=4
      fixture_kind=0
      icolup_h(1,3)=501
      icolup_h(2,4)=502
      icolup_h(1,5)=502
      icolup_h(2,5)=501
      emsca_H(1,1,3,5)=40d0
      emsca_H(1,1,5,3)=38d0
      emsca_H(1,1,4,5)=45d0
      emsca_H(1,1,5,4)=42d0
      m35=sqrt(2400d0)
      m45=sqrt(1800d0)
      reference=(q/(40d0/1.04d0))**(m35+.2d0)*(q/45d0)**(m45+.2d0)* &
           (q/42d0)**(m45+.2d0)*(q/38d0)**(m35+.2d0)
      call delta_H_min_probability(ph,1,1,q,answer)
      call require(abs(answer/reference-1d0)<1d-12,'monotone envelope across a table hole')
      print *, 'PASS tabulated-hole minimal-scale H probability'
    elseif(mode.eq.'hmin_fsr')then
      ! e+ e- > d d~ g: d-g and d~-g lines, two gluon lines.
      fixture_kind=0
      icolup_h(1,3)=501
      icolup_h(2,4)=502
      icolup_h(1,5)=502
      icolup_h(2,5)=501
      emsca_H(1,1,3,5)=40d0
      emsca_H(1,1,5,3)=38d0
      emsca_H(1,1,4,5)=45d0
      emsca_H(1,1,5,4)=42d0
      m35=sqrt(2400d0)
      m45=sqrt(1800d0)
      reference=(q/40d0)**(m35+.2d0)*(q/45d0)**(m45+.2d0)* &
           (q/42d0)**(m45+.2d0)*(q/38d0)**(m35+.2d0)
      call delta_H_min_probability(ph,1,1,q,answer)
      call require(abs(answer/reference-1d0)<1d-12,'final-state H lines to the minimal scale')
      ! A line already starting at the minimum does not evolve.
      emsca_H(1,1,3,5)=q
      reference=(q/45d0)**(m45+.2d0)*(q/42d0)**(m45+.2d0)*(q/38d0)**(m35+.2d0)
      call delta_H_min_probability(ph,1,1,q,answer)
      call require(abs(answer/reference-1d0)<1d-12,'minimal line has unit probability')
      call delta_H_min_probability(ph,1,1,-1d0,answer)
      call require(answer.eq.1d0,'no minimal scale, no probability')
      print *, 'PASS final-state minimal-scale H probability'
    else
      ! u u~ > h h g: ISR lines use the capped beam-pair mass and type 1,
      ! the gluon's FSR lines to incoming partners type 4.
      fixture_kind=1
      lpp=1
      xbjrk_ev=0.01d0
      icolup_h(1,1)=501
      icolup_h(2,2)=502
      icolup_h(1,5)=501
      icolup_h(2,5)=502
      emsca_H(1,1,1,5)=40d0
      emsca_H(1,1,2,5)=45d0
      emsca_H(1,1,5,1)=38d0
      emsca_H(1,1,5,2)=42d0
      reference=(q/40d0)**50.1d0*(q/45d0)**50.1d0* &
           (q/38d0)**50.4d0*(q/42d0)**50.4d0
      call delta_H_min_probability(ph,1,1,q,answer)
      call require(abs(answer/reference-1d0)<1d-12,'initial-state H lines to the minimal scale')
      print *, 'PASS initial-state minimal-scale H probability'
    endif
  end subroutine check_H_min_probability

  subroutine require(condition,label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (condition) return
    print *, 'FAIL: ',label
    stop 1
  end subroutine require
end program check_mcatnlo_delta_matching

subroutine read_leshouche_info(ids,mothers,colours,nprocs)
  implicit none
  integer :: ids(1,5,1),mothers(1,2,5,1),colours(1,2,5,1),nprocs(1)
  integer :: fixture_kind,profile_id,metadata_calls
  common /test_provider/ fixture_kind,profile_id,metadata_calls
  ids(1,:,1)=(/11,-11,1,-1,21/)
  if(fixture_kind==1)ids(1,:,1)=[2,-2,25,25,21]
  if(fixture_kind==2)ids(1,:,1)=[21,21,25,25,21]
  mothers=0
  colours=0
  nprocs=1
end subroutine read_leshouche_info

subroutine fill_icolor_H(flow,jpart,overwrite)
  implicit none
  integer :: flow,jpart(7,-2:7)
  logical :: overwrite
  jpart=0
  jpart(4,3)=501
  jpart(5,4)=502
  jpart(4,5)=502
  jpart(5,5)=501
end subroutine fill_icolor_H

double precision function get_mass_from_id(id)
  implicit none
  integer :: id,mass_calls(4:5)
  common /test_mass_calls/ mass_calls
  if (id < 4 .or. id > 5) stop 3
  mass_calls(id)=mass_calls(id)+1
  if (id == 4) then
    get_mass_from_id=0d0
  else
    get_mass_from_id=4.7d0
  endif
end function get_mass_from_id

double precision function pysudakov_safe(scale,mass,id,kind,mcmass)
  implicit none
  double precision :: scale,mass,mcmass(21),cstlow,cstupp,cxmlow,cxmupp
  common /cstxmbds/ cstlow,cstupp,cxmlow,cxmupp
  integer :: id,kind,table_calls
  common /test_table_calls/ table_calls
  logical :: unit_table
  common /test_unit_table/ unit_table
  integer :: fixture_kind,profile_id,metadata_calls
  common /test_provider/ fixture_kind,profile_id,metadata_calls
  integer :: table_mode
  common /test_table_mode/ table_mode
  if(table_mode.ne.0)then
    if(table_mode.eq.1.and.scale.eq.50d0.and.mass.eq.2d0.and.kind.eq.3) &
         error stop 'empty cross interval evaluated in Sudakov table'
    table_calls=table_calls+1
    pysudakov_safe=1d0
    if(table_mode.ne.3)pysudakov_safe=(scale/1000d0)**(mass+.1d0*kind)
    if(table_mode.eq.4.and.scale.gt.39.5d0.and.scale.lt.40.5d0)pysudakov_safe=0d0
    return
  endif
  if(unit_table)then
    pysudakov_safe=1d0
    return
  endif
  if (table_calls == 0) then
    cstlow=1d0
    cstupp=1000d0
    cxmlow=1d0
    cxmupp=50d0
  else
    ! The physical dipole mass sqrt(9000) is capped at the table upper limit.
    if (fixture_kind==0)then
      if (mass /= 50d0 .or. abs(id) /= 1 .or. kind /= 2) stop 4
    else
      if (mass /= 50d0 .or. kind /= 1) stop 4
      if (fixture_kind==1.and.abs(id)/=2)stop 4
      if (fixture_kind==2.and.id/=21)stop 4
    endif
    if (mcmass(4) /= 1.5d0 .or. mcmass(5) /= 4.8d0) stop 5
  endif
  table_calls=table_calls+1
  pysudakov_safe=scale/1000d0
end function pysudakov_safe

double precision function pdg2pdf(beam,id,sign,x,scale)
  implicit none
  integer :: beam,id,sign
  double precision :: x,scale
  integer :: fixture_kind,profile_id,metadata_calls
  common /test_provider/ fixture_kind,profile_id,metadata_calls
  ! This all-final-state test must never request a PDF.
  if(fixture_kind==0)stop 6
  pdg2pdf=(1d0-x)**3/x
end function pdg2pdf

subroutine pysudakov_isr_parameters(alpha,order,cmw,nfmax,thresholds,pdfmode,status)
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  double precision :: alpha,thresholds(2)
  integer :: order,nfmax,pdfmode,status,fixture_kind,profile_id,metadata_calls
  logical :: cmw
  common /test_provider/ fixture_kind,profile_id,metadata_calls
  if(fixture_kind==0)stop 7
  metadata_calls=metadata_calls+1
  alpha=0.118d0
  order=2
  cmw=.false.
  nfmax=5
  thresholds=[1.5d0,4.8d0]
  pdfmode=1
  status=0
  select case(profile_id)
  case(2)
    alpha=0.1365d0
    order=1
  case(3)
    status=1
  case(4)
    order=3
  case(5)
    cmw=.true.
  case(6)
    nfmax=4
  case(7)
    thresholds(1)=1.6d0
  case(8)
    pdfmode=2
  case(9)
    alpha=ieee_value(alpha,ieee_quiet_nan)
  case(10)
    nfmax=6
  end select
end subroutine pysudakov_isr_parameters

subroutine get_Hevent_starting_scales(stopping,dead,p,hscales)
  implicit none
  double precision :: stopping(4,4),p(0:3,5),hscales(5,5)
  logical*1 :: dead(4,4)
  double precision :: seen_scales(4,4)
  logical*1 :: seen_dead(4,4)
  common /test_stopping/ seen_scales,seen_dead
  seen_scales=stopping
  seen_dead=dead
  hscales=-1d0
  hscales(1:4,1:4)=stopping
end subroutine get_Hevent_starting_scales
