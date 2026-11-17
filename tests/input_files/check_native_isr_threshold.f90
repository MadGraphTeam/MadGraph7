! A physical light Higgs event whose inverse lost the Born threshold.
! An invalid soft counterevent incorrectly disabled MC subtraction and
! its no-emission factor in the explicit native-history H-event sum.
program check_native_isr_threshold
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mc_native_context, only: native_mapping
  use FKSParams, only: FKSISRMapping
  use fks_phase_space_data, only: resonance_recoil,initial_recoil_leg, &
       tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound, &
       p_born,p1_cnt,jac_cnt,p_i_fks_cnt
  use fks_phase_space, only: generate_native_momenta,fks_phase_space_point,invert_fks_radiation
  use fks_radiation_maps, only: generate_momenta_initial
  use fks_phase_space_helpers, only: boost_n1_to_lab
  implicit none
  include 'nexternal.inc'
  include 'genps.inc'
  include 'run.inc'
  real(8),parameter :: mass=3.1d0,stot=169d6
  real(8) :: pmass(nexternal),omx(2)
  common/to_mass/pmass
  common/to_ee_omx1/omx
  integer :: ifks,jfks,config
  common/fks_indices/ifks,jfks
  common/to_mconfigs/config
  logical :: nbody,evpr,softtest,colltest,fixed_order,nlo_ps
  common/cnbody/nbody
  common/to_use_evpr/evpr
  common/sctests/softtest,colltest
  common/c_fnlo_nlops/fixed_order,nlo_ps
  real(8) :: original(0:3,nexternal),lab(0:3,nexternal)
  real(8) :: p(0:3,nexternal),outlab(0:3,nexternal),cms(0:3,nexternal),jac
  real(8) :: expected_tau,born_mass2,replay_error
  type(fks_phase_space_point) :: point
  integer :: reflection,side,slot
  logical :: pass

  original(:,1)=[5.07210367215016442d-4,0d0,0d0,5.07210367215016442d-4]
  original(:,2)=[4752.342744222693d0,0d0,0d0,-4752.342744222693d0]
  original(:,3)=[4739.33699390418224d0,-.00378211636316499707d0, &
       -.000609477383918527915d0,-4739.33598004763189d0]
  original(:,4)=[13.0062575288786562d0,.00378211636316499750d0, &
       .000609477383918528024d0,-13.0062569646937938d0]
  expected_tau=mass**2/stot
  ebeam=6500d0
  lpp=1
  pmass=[0d0,0d0,mass,0d0]
  omx=0d0
  native_mapping=.true.
  fixed_order=.false.
  nlo_ps=.true.
  nbody=.false.
  evpr=.true.
  softtest=.false.
  colltest=.false.
  FKSISRMapping=2
  resonance_recoil=.false.
  initial_recoil_leg=0
  ifks=4
  config=1
  tau_Born_lower_bound=expected_tau
  tau_lower_bound_resonance=expected_tau
  tau_lower_bound=expected_tau
  do reflection=0,1
    lab=original
    if(reflection.eq.1)then
      lab(:,1)=original(:,2)
      lab(:,2)=original(:,1)
      lab(3,:)=-lab(3,:)
    endif
    do side=1,2
      jfks=side
      call generate_native_momenta(lab,p,outlab,cms,jac,pass,point)
      if(.not.pass.or..not.ieee_is_finite(jac).or.jac.le.0d0) &
           error stop 'native ISR real projection failed'
      ! A one-particle Born fixes tau exactly through the model mass.
      if(abs(point%born%tau/expected_tau-1d0).gt.64d0*epsilon(1d0))then
        write(*,*) 'native Born tau, physical threshold',point%born%tau,expected_tau
        error stop 'native ISR Born invariant lost the model threshold'
      endif
      born_mass2=p_born(0,3)**2-sum(p_born(1:3,3)**2)
      if(abs(born_mass2/mass**2-1d0).gt.1d-12) &
           error stop 'native ISR one-particle Born is off shell'
      do slot=0,2
        if(.not.ieee_is_finite(jac_cnt(slot)).or.jac_cnt(slot).le.0d0.or. &
             p1_cnt(0,1,slot).le.0d0.or..not.point%counterevent(slot)%valid)then
          write(*,*) 'native counterevent',reflection,side,slot,jac_cnt(slot),p1_cnt(0,1,slot)
          error stop 'native ISR lost a threshold counterevent'
        endif
      enddo
      replay_error=maxval(abs(outlab-lab))/maxval(abs(lab))
      if(.not.all(ieee_is_finite(outlab)).or.replay_error.gt.1d-9) &
           error stop 'native ISR threshold replay changed the real event'
    enddo
  enddo
  call check_boundary_points()
  print *, 'PASS native ISR one-particle threshold'
contains
  subroutine check_boundary_points()
    ! Exercise exactly soft and hard forward points, not just the
    ! captured event. The soft inverse has a zero sampling Jacobian;
    ! it must remain finite and must not report an unphysical domain.
    real(8),parameter :: energies(4)=[0d0,1d-5,.9999999999d0,1d0]
    real(8),parameter :: pi=3.1415926535897932385d0
    real(8) :: bx(2),yb,rad(2),xp(0:3,nexternal),direction(0:3)
    real(8) :: s,root,tau,rapidity,xb(2),xmax,xnorm,xi,y,xihat,ps,jfwd
    real(8) :: recovered(0:3,-max_branch:nexternal-1),coords(3),ji,psi,tb,yi,bxi(2)
    integer :: emitter,energy
    logical :: valid
    bx=[.2d0,expected_tau/.2d0]
    yb=log(bx(1)/bx(2))/2d0
    do emitter=1,2
      jfks=emitter
      do energy=1,size(energies)
        xp=0d0
        xp(0,1:2)=mass/2d0
        xp(3,1)=mass/2d0
        xp(3,2)=-mass/2d0
        xp(0,3)=mass
        rad=[energies(energy),.43d0]
        jfwd=1d0
        ps=1d0
        call generate_momenta_initial(-100,ifks,jfks,bx,expected_tau,yb,mass**2,.7d0,xp,rad, &
             s,stot,root,tau,rapidity,xb,direction,xmax,xnorm,xi,y,xihat,ps,jfwd,valid)
        if(.not.valid)error stop 'native ISR boundary forward point rejected'
        call boost_n1_to_lab(xp,lab,-yb)
        ! Only the exactly soft inverse uses this supplied direction.
        p_i_fks_cnt(:,0)=direction
        ji=1d0
        psi=1d0
        call invert_fks_radiation(coords,ji,psi,stot,tb,yi,bxi,lab,recovered)
        if(.not.ieee_is_finite(ji).or.ji.lt.0d0.or..not.ieee_is_finite(psi).or. &
             .not.all(ieee_is_finite(coords)).or.abs(tb/expected_tau-1d0).gt.64d0*epsilon(1d0)) &
             error stop 'native ISR soft or hard inverse domain'
        if(maxval(abs(coords(1:2)-rad)).gt.3d-8.or.maxval(abs(bxi/bx-1d0)).gt.1d-9) &
             error stop 'native ISR boundary coordinates or fractions'
        if(jfwd.gt.0d0)then
          if(abs(ji/(2d0*pi*jfwd)-1d0).gt.1d-8)then
            write(*,*) 'boundary Jacobian',emitter,energy,xi,ji/(2d0*pi*jfwd)-1d0,coords(1)-rad(1)
            error stop 'native ISR boundary sampling Jacobian'
          endif
        endif
      enddo
    enddo
    ! Incoming-energy roundoff can put an exactly soft point a few ulps
    ! below the model invariant. Its zero Jacobian remains a valid limit.
    lab=0d0
    lab(0,1:2)=nearest(mass/2d0,-1d0)
    lab(3,1)=lab(0,1)
    lab(3,2)=-lab(0,2)
    lab(0,3)=mass
    p_i_fks_cnt(:,0)=[1d0,.6d0,0d0,.8d0]
    ji=1d0
    psi=1d0
    call invert_fks_radiation(coords,ji,psi,stot,tb,yi,bxi,lab,recovered)
    if(.not.ieee_is_finite(ji).or.ji.lt.0d0.or.coords(1).ne.0d0.or. &
         abs(tb/expected_tau-1d0).gt.64d0*epsilon(1d0)) &
         error stop 'native ISR rejected roundoff at the exact soft endpoint'
    ! A genuinely subthreshold invariant must still be rejected. This
    ! exceeds incoming-energy roundoff by many orders of magnitude.
    lab=0d0
    lab(0,1:2)=mass*sqrt(1d0-1d-8)/2d0
    lab(3,1)=lab(0,1)
    lab(3,2)=-lab(0,2)
    lab(0,3)=mass
    ji=1d0
    psi=1d0
    call invert_fks_radiation(coords,ji,psi,stot,tb,yi,bxi,lab,recovered)
    if(ji.ge.0d0)error stop 'native ISR accepted a genuinely subthreshold point'
  end subroutine
end program
