module history_test_state
  use fks_phase_space_data,only: p1_cnt,jac_cnt,p_born,p_ev,ybst_til_tolab, &
       ybst_til_tocm,sqrtshat,shat,xi_i_fks_ev,y_ij_fks_ev,xi_i_fks_cnt,xiimax_cnt, &
       xi_i_hat_ev,xinorm_ev,shat_ev,shat_cnt,sqrtshat_ev,sqrtshat_cnt, &
       tau_lower_bound,resonance_recoil,nocntevents
  use fks_radiation_maps, only: generate_momenta_initial
  use history_sampling_measure, only: compute_flux,fill_fks_point_data
  use mc_native_context, only: native_mapping
  implicit none
  include 'nexternal.inc'
  include 'orders.inc'
  include 'coupl.inc'
  include 'run.inc'
  include 'q_es.inc'
  double precision :: xiScut_used,xiBSVcut_used
  common/cxiScut_used/xiScut_used,xiBSVcut_used
  integer :: i_fks,j_fks,nFKSprocess,fold,ifold_counter
  common/fks_indices/i_fks,j_fks
  common/c_nFKSprocess/nFKSprocess
  common/cfl/fold,ifold_counter
  character*4 :: abrv
  common/to_abrv/abrv
  double precision :: f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
  common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
  double precision :: f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
  common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
  double precision :: f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
  common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
  double precision :: deg_xi(3),deg_lxi(3),deg_muF(3),dis_p(3),dis_l(3),dis_d(3)
  common/to_amp_split_deg/deg_xi,deg_lxi,deg_muF
  common/to_amp_split_dis/dis_p,dis_l,dis_d
  double precision :: test_p,test_gsoft,test_gcoll
  integer :: real_calls,degenerate_calls
  logical :: sampling_check=.false.
contains
  subroutine check_sampling_weights()
    ! Compare densities at the same physical ISR point in three charts.
    ! All real, FKS and MC S/H terms must carry exactly the sampling
    ! determinant; dividing by the chart's real measure removes it.
    use weight_lines
    implicit none
    real(8),parameter :: xis(3)=[.2d0,.99d0,.999999d0],xbar=1d-7,eta=.35d0
    real(8) :: density(3,15,3,3),jacobians(3),measures(3),probne
    real(8) :: expected,log_x,old_x,xi,point_scale
    logical :: fixed_order,nlo_ps
    common/c_fnlo_nlops/fixed_order,nlo_ps
    integer :: k,chart,ict
    sampling_check=.true.
    fixed_order=.false.
    nlo_ps=.true.
    test_mass=0d0
    i_fks=5
    j_fks=1
    xiScut_used=.5d0
    test_p=.37d0
    test_gsoft=.4d0
    test_gcoll=.6d0
    mc_H_only=.false.
    mc_S_only=.false.
    do k=1,size(xis)
      xi=xis(k)
      do chart=1,3
        call sampling_point(chart,xi,eta,xbar,jacobians(chart))
        measures(chart)=xinorm_ev*xi_i_fks_ev*jacobians(chart)
        call compute_prefactors_n1body(1d0,jacobians(chart))
        icontr=0
        call compute_native_NLOPS_weights(p_ev,p_ev,.true.,.true.,probne)
        density(:,:,:,chart)=0d0
        do ict=1,icontr
          density(:,itype(ict),amppos(ict),chart)=density(:,itype(ict),amppos(ict),chart)+ &
               wgt(:,ict)/measures(chart)
        enddo
        if(maxval(abs(density(:,12,:,chart))).eq.0d0.or. &
             maxval(abs(density(:,13,:,chart))).eq.0d0)error stop 'missing sampled MC S/H weights'
        if(maxval(abs(density(:,12,:,chart)+density(:,13,:,chart))).gt.1d-10) &
             error stop 'sampled MC S/H weights do not cancel'
      enddo
      ! The new map's derivative is obtained independently from
      ! 1-xi=xbar**u, rather than calling its production helper.
      log_x=sqrt((log(1d0-xi)/log(xbar)-1d-6)/(1d0-1d-6))
      old_x=sqrt((xi/(1d0-xbar)-1d-6)/(1d0-1d-6))
      expected=log_x/old_x*(-log(xbar))*(1d0-xi)/(1d0-xbar)
      if(abs(jacobians(1)/jacobians(2)/expected-1d0).gt.1d-8) &
           error stop 'matched ISR sampling determinant missing or repeated'
      point_scale=maxval(abs(density(:,:,:,3)))
      do chart=1,2
        if(maxval(abs(density(:,:,:,chart)-density(:,:,:,3))).gt.1d-8*point_scale) &
             error stop 'outer/native matched weight densities differ'
      enddo
    enddo
    print *, 'PASS native H and S weights with matched ISR sampling'
  end subroutine

  subroutine sampling_point(chart,xi_target,eta,xbar,jac_ev)
    implicit none
    integer,intent(in) :: chart
    real(8),intent(in) :: xi_target,eta,xbar
    real(8),intent(out) :: jac_ev
    real(8),parameter :: pi=3.1415926535897932385d0,stot=169d6,sborn=3.1d0**2
    real(8) :: bx(2),tb,yb,rad(2),u,stiny,ctiny,p(0:3,nexternal),pdir(0:3),out(0:3,nexternal)
    real(8) :: jac,ps,sc,root,tau,rapidity,xb(2),xmax,xnorm,xi,y,xihat
    real(8) :: delta_used,xicut_used,symmetry,symmetry_b,symmetry_d
    integer :: ngluons,nquarks(-6:6),nphotons,slot,counter
    logical :: nbody,softtest,colltest,pass,use_evpr
    common/cnbody/nbody
    common/sctests/softtest,colltest
    common/to_use_evpr/use_evpr
    common/cdelta_used/delta_used
    common/cxicut_used/xicut_used
    common/numberofparticles/symmetry,symmetry_b,symmetry_d,ngluons,nquarks,nphotons
    bx=[xbar,sborn/(stot*xbar)]
    tb=product(bx)
    yb=log(bx(1)/bx(2))/2d0
    native_mapping=chart.eq.3
    ! A Born-mode reference retains the old quadratic radiation chart
    ! and its cutoffs while still constructing real momenta.
    nbody=chart.eq.2
    softtest=.false.
    colltest=.false.
    use_evpr=.true.
    resonance_recoil=.false.
    nocntevents=.false.
    tau_lower_bound=0d0
    delta_used=1d0
    xicut_used=.5d0
    symmetry=1d0
    symmetry_b=1d0
    symmetry_d=1d0
    stiny=1d-6
    ctiny=5d-7
    if(native_mapping)then
      stiny=0d0
      ctiny=0d0
    endif
    u=xi_target/(1d0-xbar)
    if(chart.eq.1)u=log(1d0-xi_target)/log(xbar)
    rad=[sqrt((u-stiny)/(1d0-stiny)),sqrt((eta/2d0-ctiny)/(1d0-ctiny))]
    p_born=0d0
    p_born(0,:)=sqrt(sborn)/2d0
    p_born(3,1)=p_born(0,1)
    p_born(3,2)=-p_born(0,2)
    p_born(1,3)=p_born(0,3)
    p_born(1,4)=-p_born(0,4)
    xihat=0d0
    do slot=-1,2
      counter=slot
      if(slot.eq.-1)counter=-100
      p=0d0
      p(:,1:4)=p_born
      jac=2d0*pi
      ps=1d0
      call generate_momenta_initial(counter,5,1,bx,tb,yb,sborn,.7d0,p,rad, &
           sc,stot,root,tau,rapidity,xb,pdir,xmax,xnorm,xi,y,xihat,ps,jac,pass)
      if(.not.pass.or.jac.le.0d0)error stop 'invalid matched ISR sampling point'
      call compute_flux(sc,root,0d0,0d0,ps,jac)
      call fill_fks_point_data(counter,tau,rapidity,yb,sc,root,xb, &
           xmax,xnorm,xi,xihat,pdir,y,p,out,jac,jac_ev)
      if(slot.eq.-1)then
        if(abs(xi/xi_target-1d0).gt.1d-8.or.abs(y-(1d0-eta)).gt.1d-12) &
             error stop 'matched sampler did not recover the specified physical point'
      endif
    enddo
    nbody=.false.
    shat=shat_ev
    sqrtshat=sqrtshat_ev
  end subroutine
end module

program check_mc_history_weights
  use, intrinsic :: ieee_arithmetic
  use history_test_state
  use FKSParams, only: VetoedContributionTypes
  use weight_lines
  implicit none
  double precision :: baseline(3,15,3),specialized(3,15,3),probne
  double precision :: probabilities(3),gvalues(3),bad(3),bad_values(3)
  double precision :: expected_momenta(0:3,nexternal)
  integer,parameter :: record_types(3)=[1,5,12]
  integer missing,selected
  integer :: j,k,old_icontr
  integer :: imass,ip,igs,igc,icuts,mode,i,cases,baseline_reals
  logical :: cuts_born,cuts_real
  character(8) :: matching
  call allocate_weight_lines(nexternal)
  probabilities=[0d0,0.37d0,1d0]
  gvalues=[0d0,0.6d0,1d0]
  p_ev=0d0
  p_ev(0,:)=[100d0,100d0,80d0,60d0,60d0]
  p_born=p_ev(:,1:nexternal-1)
  do i=-2,2
    p1_cnt(:,:,i)=p_ev
  enddo
  xi_i_fks_ev=0.2d0
  y_ij_fks_ev=0.8d0
  xi_i_fks_cnt=0.3d0
  xiimax_cnt=1d0
  xi_i_hat_ev=0.2d0
  xiScut_used=0.5d0
  i_fks=5
  j_fks=3
  nFKSprocess=1
  fold=0
  ifold_counter=1
  abrv='all '
  ickkw=0
  call get_command_argument(1,matching)
  if(trim(matching).eq.'fxfx'.or.trim(matching).eq.'mapfxfx')ickkw=3
  lpp=1
  xbk=0.1d0
  scale=100d0
  q2fact=scale**2
  QES2=scale**2
  g=1.3d0
  ybst_til_tolab=0d0
  ybst_til_tocm=0d0
  shat=40000d0
  sqrtshat=200d0
  f_r=1.1d0
  f_s=1.2d0
  f_c=1.3d0
  f_dc=1.4d0
  f_sc=1.5d0
  f_dsc=[1.6d0,1.7d0,1.8d0,1.9d0]
  f_s_MC_S=2.1d0
  f_s_MC_H=2.2d0
  f_c_MC_S=2.3d0
  f_c_MC_H=2.4d0
  f_sc_MC_S=2.5d0
  f_sc_MC_H=2.6d0
  f_MC_S=2.7d0
  f_MC_H=2.8d0
  f_pdfsch_d=3.1d0
  f_pdfsch_p=3.2d0
  f_pdfsch_l=3.3d0
  if(trim(matching).eq.'sampling'.or.trim(matching).eq.'mapfxfx')then
    call check_sampling_weights()
    stop
  endif
  cases=0
  do imass=0,1
    test_mass=173d0*imass
    do ip=1,3
      test_p=probabilities(ip)
      do igs=1,3
        test_gsoft=gvalues(igs)
        do igc=1,3
          test_gcoll=gvalues(igc)
          do icuts=0,3
            cuts_born=btest(icuts,0)
            cuts_real=btest(icuts,1)
            do mode=0,2
              mc_H_only=mode.eq.1
              mc_S_only=mode.eq.2
              icontr=0
              real_calls=0
              degenerate_calls=0
              ! Poison remnants: the H-only path must not read stale S data.
              deg_xi=1d30
              deg_lxi=-1d30
              deg_muF=2d30
              dis_p=3d30
              dis_l=-3d30
              dis_d=4d30
              call compute_native_NLOPS_weights(p_ev,p_ev,cuts_born,cuts_real,probne)
              specialized=0d0
              do i=1,icontr
                specialized(:,itype(i),amppos(i))=specialized(:,itype(i),amppos(i))+wgt(:,i)
                if (mode.eq.1.and..not.H_event(i)) error stop 'S record in H-only evaluation'
                if (mode.eq.2.and.H_event(i)) error stop 'H record in S-only evaluation'
              enddo
              if (mode.eq.0) then
                baseline=specialized
                baseline_reals=real_calls
              else
                do i=1,15
                  if ((i.eq.1.or.i.eq.13.or.(i.ge.8.and.i.le.10)).eqv.(mode.eq.1)) then
                    if (maxval(abs(specialized(:,i,:)-baseline(:,i,:))).gt.1d-12) &
                      error stop 'specialized native weights differ'
                  endif
                enddo
                if (real_calls.gt.baseline_reals) error stop 'extra real evaluations'
              endif
              if (mode.eq.1) then
                if (degenerate_calls.ne.0) error stop 'H-only degenerate remnant evaluation'
                if (cuts_born.and.test_p.eq.0d0.and.real_calls.ne.0) &
                  error stop 'zero P evaluated H matrix elements'
                if (test_gsoft.eq.1d0.and.real_calls.gt.1) &
                  error stop 'zero G evaluated H counterterms'
              elseif(mode.eq.2.and.cuts_born.and.cuts_real)then
                if(test_p.lt.1d0.and.maxval(abs(specialized(:,11,:))).eq.0d0) &
                  error stop 'S-only evaluation lost the real S contribution'
                if(test_p.eq.1d0.and.real_calls.ge.baseline_reals) &
                  error stop 'S-only evaluation computed the pure H real term'
              endif
            enddo
            cases=cases+1
          enddo
        enddo
      enddo
    enddo
  enddo
  mc_H_only=.false.
  mc_S_only=.false.
  old_icontr=icontr
  bad_values=[ieee_value(0d0,ieee_quiet_nan),ieee_value(0d0,ieee_positive_inf), &
              ieee_value(0d0,ieee_negative_inf)]
  do j=1,3
    do k=1,3
      bad=0d0
      bad(k)=bad_values(j)
      nonfinite_contribution=.false.
      call add_wgt(8,[2,0],bad(1),bad(2),bad(3))
      if(.not.nonfinite_contribution.or.icontr.ne.old_icontr) &
        error stop 'nonfinite weight accepted or not reported'
      if(nonfinite_type.ne.8.or.nonfinite_sector.ne.nFKSprocess) &
        error stop 'lost nonfinite weight source'
    enddo
  enddo
  ! Suppressed S/H records are not part of this history's contribution.
  nonfinite_contribution=.false.
  mc_S_only=.true.
  call add_wgt(8,[2,0],bad_values(1),0d0,0d0)
  if(nonfinite_contribution)error stop 'rejected suppressed H record'
  mc_S_only=.false.
  VetoedContributionTypes=[1,8]
  call add_wgt(8,[2,0],bad_values(1),0d0,0d0)
  if(nonfinite_contribution)error stop 'rejected intentionally vetoed record'
  VetoedContributionTypes=0
  call add_wgt(8,[2,0],1d0,0d0,0d0)
  if(nonfinite_contribution.or.icontr.ne.old_icontr+1)error stop 'valid weight did not recover'
  ! Distinct endpoint momenta expose priority and fallback mistakes in
  ! record storage. Collinear weights must skip slot zero even if valid.
  do missing=0,3
    do k=0,2
      p1_cnt(:,:,k)=p_ev+10d0*(k+1)
      if(k.lt.missing)p1_cnt(0,1,k)=-99d0
    enddo
    do j=1,size(record_types)
      if(missing.lt.3)then
        selected=missing
        if(record_types(j).eq.5)selected=max(1,selected)
        expected_momenta=p1_cnt(:,:,selected)
      else
        expected_momenta(:,1:nexternal-1)=p_born
        expected_momenta(:,nexternal)=0d0
      endif
      call add_wgt(record_types(j),[2,0],1d0,2d0,3d0)
      if(any(momenta_m(:,:,1,icontr).ne.expected_momenta).or. &
           any(momenta_m(:,:,2,icontr).ne.p_ev))error stop 'recorded subtraction momenta differ'
      if(record_types(j).eq.1)expected_momenta=p_ev
      if(any(momenta(:,:,icontr).ne.expected_momenta))error stop 'recorded event momenta differ'
    enddo
  enddo
  write(*,*) 'PASS native H and S weights' ,cases,' cases'
end program

subroutine compute_MC_subt_term(p,p_cms,probne)
  use history_test_state
  use extra_weights
  use mc_counterterms, only: gfactsf,gfactcl
  implicit none
  double precision :: p(0:3,5),p_cms(0:3,5),probne
  integer :: i,orders(2)
  probne=test_p
  gfactsf=test_gsoft
  gfactcl=test_gcoll
  do i=1,3
    call amp_split_pos_to_orders(i,orders)
    QCD_power=orders(1)
    wgtcpower=0d0
    orders_tag=i
    amp_pos=i
    if(sampling_check)then
      call add_wgt(12,orders,11d0*i*probne*f_MC_S,0d0,0d0)
      call add_wgt(13,orders,-11d0*i*probne*f_MC_H,0d0,0d0)
    else
      call add_wgt(12,orders,11d0*i*probne,0d0,0d0)
      call add_wgt(13,orders,-13d0*i*probne,0d0,0d0)
    endif
  enddo
end subroutine

subroutine sreal(p,xi,y,ans)
  use history_test_state
  implicit none
  double precision :: p(0:3,5),xi,y,ans
  real_calls=real_calls+1
  amp_split=[2d0+xi+3d0*y,4d0+2d0*xi+y,0d0]
  ans=sum(amp_split)
end subroutine

subroutine sreal_deg(p,xi,ans1,ans2)
  use history_test_state
  implicit none
  double precision :: p(0:3,5),xi,y,ans1,ans2
  degenerate_calls=degenerate_calls+1
  deg_xi=[3d0,5d0,7d0]
  deg_lxi=[4d0,6d0,8d0]
  deg_muF=[5d0,7d0,9d0]
  dis_p=[6d0,8d0,10d0]
  dis_l=[7d0,9d0,11d0]
  dis_d=[8d0,10d0,12d0]
  amp_split=-1d30
  ans1=1d0
  ans2=2d0
end subroutine

double precision function fks_Sij(p,i,j,xi,y)
  implicit none
  double precision :: p(0:3,5),xi,y
  integer :: i,j
  fks_Sij=0.7d0+0.1d0*xi+0.05d0*y
end function

subroutine amp_split_pos_to_orders(i,orders)
  implicit none
  integer :: i,orders(2)
  if (i.eq.1) then
    orders=[4,0]
  elseif (i.eq.2) then
    orders=[0,4]
  else
    orders=[2,2]
  endif
end subroutine

integer function get_orders_tag(orders)
  implicit none
  integer :: orders(2)
  get_orders_tag=10*orders(1)+orders(2)
end function

integer function get_n_tagged_photons()
  get_n_tagged_photons=0
end function

double precision function get_rescale_alpha_factor(n,order)
  integer :: n,order
  get_rescale_alpha_factor=1d0
end function

subroutine set_pdg(ict,sector)
  use weight_lines
  implicit none
  integer :: ict,sector
  pdg(:,ict)=[21,21,6,-6,21]
  pdg_uborn(:,ict)=pdg(:,ict)
end subroutine

subroutine set_cms_stuff(i)
  use history_test_state, only: sampling_check,shat,sqrtshat,shat_ev,shat_cnt,sqrtshat_ev,sqrtshat_cnt
  integer :: i
  if(sampling_check)then
    if(i.eq.-100)then
      shat=shat_ev
      sqrtshat=sqrtshat_ev
    else
      shat=shat_cnt(i)
      sqrtshat=sqrtshat_cnt(i)
    endif
  endif
end subroutine

subroutine resonance_subtraction_scales(total,resonance,sister,direction, &
     soft,coll,angular,mismatch,pass)
  implicit none
  real(8) :: total(0:3),resonance(0:3),sister(0:3),direction(0:3),soft,coll,angular,mismatch
  logical :: pass
  error stop 'unexpected resonance subtraction in ISR fixture'
end subroutine

subroutine set_alphaS(p)
  double precision :: p(0:3,5)
end subroutine

subroutine set_FxFx_scale(i,p)
  integer :: i
  double precision :: p(0:3,5)
end subroutine

subroutine include_multichannel_enhance(i)
  integer :: i
end subroutine
