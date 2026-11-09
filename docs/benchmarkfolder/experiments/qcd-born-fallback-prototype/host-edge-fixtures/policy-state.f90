! Test state and explicit external-dependency stubs for extracted host policy.
! These values match the reviewed qcd_n3_uux_zgg export dimensions.
module audit_state
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer, parameter :: nexternal=5,ncomb=48,nloopgroups=116,nsquaredso=1
  integer, parameter :: nsqso_born=1,maxstabilitylength=20,heloffset=10000
  integer :: nrotations_dp,nrotations_qp,nrotations_dp_bu,nrotations_qp_bu
  integer :: max_spin_external_particle,userhel,n_dp_eval,n_qp_eval,ntry
  integer :: helicityfilterlevel,checkcycle,looppolemode,stab_index
  integer :: ctmodeinit,ctmodeinit_bu,ctmoderun,ctmode,helpicked
  integer :: ret_code_h,ret_code_t,ret_code_u,i_lib,mlreductionlib(9)
  integer :: goodhel(ncomb),old_goodhel(ncomb),ic(nexternal)
  logical :: warned_lorentz_stab_test_off,heldoublechecked,doublecheckhelicityfilter
  logical :: foundloopfilter,foundhelfilter,useloopfilter,writeoutfilters
  logical :: goodamp(nsquaredso,nloopgroups),old_goodamp(nsquaredso,nloopgroups)
  logical :: born_rescued,born_ok,born_qp_checked,mp_done,mp_done_once,mp_ps_set
  logical :: automatic_cache_clearing,doing_qp_evals,eval_done(maxstabilitylength)
  logical :: bypass_filters,doneheldoublecheck,checkphase
  logical :: old_checkphase,old_heldoublechecked,chosen_so_configs(nsquaredso)
  logical :: fpe_in_qp_reduction
  real(8) :: bornbuff(0:nsqso_born),ans(0:3,0:nsquaredso),accuracy(0:nsquaredso)
  real(8) :: acc(0:nsquaredso),ref,mlstabthres,mlstabthres_bu,user_stab_prec
  real(8) :: p_user(0:3,nexternal),recorded_p(0:3,nexternal)
  integer :: recorded_hel,recorded_ic(nexternal)
  character(100) :: loopfilterfn='loop-filter.dat',helfilterfn='hel-filter.dat'
  integer :: born_mode,loop_mode,loop_calls,born_calls,clear_calls,cache_off_calls
  integer :: set_nevals_calls
contains
  subroutine reset_state()
    integer :: u,j
    logical :: exists
    inquire(file=loopfilterfn,exist=exists)
    if(exists) then
      open(newunit=u,file=loopfilterfn,status='old');close(u,status='delete')
    endif
    inquire(file=helfilterfn,exist=exists)
    if(exists) then
      open(newunit=u,file=helfilterfn,status='old');close(u,status='delete')
    endif
    nrotations_dp=2;nrotations_qp=3;nrotations_dp_bu=-901;nrotations_qp_bu=-902
    max_spin_external_particle=4;userhel=2;n_dp_eval=7;n_qp_eval=9;ntry=0
    helicityfilterlevel=1;checkcycle=3;looppolemode=-1;stab_index=-903
    ctmodeinit=1;ctmodeinit_bu=-904;ctmoderun=-1;ctmode=-905;helpicked=-906
    ret_code_h=3;ret_code_t=2;ret_code_u=3;i_lib=1;mlreductionlib=0;mlreductionlib(1)=9
    goodhel=1;old_goodhel=-907;goodamp=.false.;old_goodamp=.true.
    ic=1;warned_lorentz_stab_test_off=.true.;heldoublechecked=.true.
    doublecheckhelicityfilter=.false.;foundloopfilter=.false.;foundhelfilter=.false.
    useloopfilter=.true.;writeoutfilters=.true.;born_rescued=.false.;born_ok=.false.
    born_qp_checked=.true.;mp_done=.true.;mp_done_once=.false.;mp_ps_set=.true.
    automatic_cache_clearing=.true.;doing_qp_evals=.true.;eval_done=.false.
    bypass_filters=.false.;doneheldoublecheck=.false.;checkphase=.false.
    old_checkphase=.false.;old_heldoublechecked=.false.;chosen_so_configs=.true.
    fpe_in_qp_reduction=.false.;bornbuff=-908d0;ans=-909d0;accuracy=1d-9;acc=1d-9
    ref=-910d0;mlstabthres=3d-6;mlstabthres_bu=-911d0;user_stab_prec=1d-8
    do j=1,nexternal
      p_user(:,j)=[real(j,8),-real(j,8),0.25d0,0d0]
    enddo
    recorded_p=0d0;recorded_hel=-912;recorded_ic=0
    born_mode=0;loop_mode=0;loop_calls=0;born_calls=0;clear_calls=0;cache_off_calls=0
    set_nevals_calls=0
  end subroutine

  ! Explicit hook replacing all loop evaluation, never a numerical amplitude.
  subroutine simulated_loop_completion()
    loop_calls=loop_calls+1
    goodamp=.true.
    ans(1:3,:)=0.125d0
    acc=1d-10
    if(loop_mode==0) then
      ret_code_h=2;ret_code_u=9;accuracy=acc
    else
      doing_qp_evals=.true.;mp_done_once=.true.
      if(loop_mode==2) acc(0)=ieee_value(0d0,ieee_quiet_nan)
      if(loop_mode==3) ans(2,1)=ieee_value(0d0,ieee_quiet_nan)
      if(loop_mode==4) fpe_in_qp_reduction=.true.
      if(loop_mode==5) acc=1d0
    endif
  end subroutine
end module

subroutine ML5_0_BORN_WITH_RESCUE(p,hel,colors,result,rescued,ok)
  use audit_state, only: nexternal,nsqso_born,born_mode,born_calls,recorded_p,recorded_hel,recorded_ic
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(8),intent(in) :: p(0:3,nexternal)
  integer,intent(in) :: hel,colors(nexternal)
  real(8),intent(out) :: result(0:nsqso_born)
  logical,intent(out) :: rescued,ok
  born_calls=born_calls+1;recorded_p=p;recorded_hel=hel;recorded_ic=colors
  rescued=born_mode/=0;ok=born_mode/=1;result=2d0
  if(.not.ok) result=ieee_value(0d0,ieee_quiet_nan)
end subroutine

subroutine ML5_0_CLEAR_CACHES()
  use audit_state, only: clear_calls
  implicit none
  clear_calls=clear_calls+1
end subroutine

subroutine ML5_0_SET_COLLIER_GLOBAL_CACHE(enabled)
  use audit_state, only: cache_off_calls
  implicit none
  logical,intent(in) :: enabled
  if(enabled) error stop 90
  cache_off_calls=cache_off_calls+1
end subroutine

subroutine ML5_0_SET_N_EVALS(dp,qp)
  use audit_state, only: set_nevals_calls
  implicit none
  integer,intent(out) :: dp,qp
  dp=1;qp=1;set_nevals_calls=set_nevals_calls+1
end subroutine

integer function SET_RET_CODE_U(lib,qp,accepted)
  implicit none
  integer,intent(in) :: lib
  logical,intent(in) :: qp,accepted
  if(lib/=9) error stop 91
  SET_RET_CODE_U=9
  if(qp) SET_RET_CODE_U=3
end function
