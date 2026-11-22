module flow_fixture
  use weight_lines
  implicit none
  double precision :: values(3)=[1d0,3d0,0d0],uniform=0.5d0,last_grid_weight=-1d0
  integer :: n_groups=1,group_labels(8)=1,group_pick=1,ini_fin=0
  real(8) :: group_prob(8)=1d0,born_jac
  logical :: observe_lanes=.false.
  integer :: draws=0,born_calls=0,fail_native_fold=0
  integer :: nlo_calls=0,s_only_calls=0,native_calls=0
  logical :: real_point_active=.false.,born_cuts=.true.
  logical :: outer_no_endpoint=.false.,real_no_endpoint=.false.
  logical :: no_event_projection=.false.
  logical :: fail_kinematics=.false.
  integer :: fail_weight_fold=0,fail_weight_stage=0
  double precision :: bad_weight=0d0
contains
  subroutine add_record(value)
    double precision :: value
    icontr=icontr+1
    wgt(:,icontr)=value
    wgts(1,icontr)=value
    niproc(icontr)=1
    parton_iproc(1,icontr)=value
  end subroutine
end module

module fks_phase_space_data
  implicit none
  double precision :: p_born(0:3,4)=1d0,p1_cnt(0:3,5,0:2)=1d0,jac_cnt(0:2)=1d0
end module

module fks_phase_space
  use fks_phase_space_data
  use flow_fixture, only: outer_no_endpoint,real_no_endpoint,no_event_projection,born_jac,observe_lanes
  implicit none
  type born_fixture
    logical :: valid=.false.,event_projection=.true.
    double precision :: p(0:3,4)=1d0,xbjrk(2)=0.2d0
  end type
  type fks_phase_space_point
    double precision :: p(0:3,5)=1d0,p_lab(0:3,5)=1d0,p_cms(0:3,5)=1d0
    integer :: config=1
    real(8) :: weight=1d0
    type(born_fixture) :: born
  end type
contains
  subroutine capture_fks_phase_space(point)
    type(fks_phase_space_point) point
  end subroutine
  subroutine restore_fks_phase_space(point)
    type(fks_phase_space_point) point
  end subroutine
  subroutine initialize_fks_phase_space(config)
    integer config
  end subroutine
  subroutine generate_born_contribution(ndim,config,jac,x,point)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    integer :: i_fks,j_fks
    common/fks_indices/i_fks,j_fks
    if(observe_lanes.and.point%born%valid)error stop 'Inherited sampled Born at fresh fold'
    born_jac=jac
    point=fks_phase_space_point()
    point%born%event_projection=.not.no_event_projection
    p_born=1d0
    p1_cnt=1d0
    p1_cnt(:,i_fks,0)=0d0
    if(outer_no_endpoint)p1_cnt(0,1,0)=-99d0
  end subroutine
  subroutine generate_real_phase_space(ndim,config,jac,x,point)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    integer :: i_fks,j_fks
    common/fks_indices/i_fks,j_fks
    point=fks_phase_space_point()
    point%born%event_projection=.not.no_event_projection
    p1_cnt(:,:,0)=1d0
    p1_cnt(:,i_fks,0)=0d0
    ! The reference sector itself, at the same radiation coordinates.
    if(real_no_endpoint)p1_cnt(0,1,0)=-99d0
  end subroutine
end module

module mc_native_context
  use flow_fixture, only: real_point_active
  implicit none
  logical :: native_mapping=.false.,shared_real_active=.false.
  type native_provider_request
    integer :: slot
  end type
  type native_provider_point_state
    integer :: slot
  end type
contains
  subroutine mc_capture_provider_point(state)
    type(native_provider_point_state) state
  end subroutine
  subroutine mc_restore_provider_model(state)
    type(native_provider_point_state) state
  end subroutine
  subroutine mc_activate_provider_point(state,model_is_active)
    logical,optional :: model_is_active
    type(native_provider_point_state) state
  end subroutine
  subroutine mc_prepare_provider_request(provider,p,owner,seq,fold)
    use iso_fortran_env, only: int64
    type(native_provider_request) provider
    real(8) p(0:3,5)
    integer(int64) owner
    integer seq,fold
    error stop 'Rejected mock has no pending real requests'
  end subroutine
  subroutine mc_consume_provider_result(provider,w,split,owner,seq,fold)
    use iso_fortran_env, only: int64
    integer(int64) owner
    integer seq,fold
    type(native_provider_request) provider
    real(8) w,split(*)
    error stop 'Rejected mock has no pending real requests'
  end subroutine
  subroutine mc_begin_real_point(p)
    double precision :: p(0:3,5)
    if(real_point_active)error stop 'real-point cache was not closed'
    real_point_active=.true.
  end subroutine
  subroutine mc_end_real_point()
    if(.not.real_point_active)error stop 'real-point cache closed twice'
    real_point_active=.false.
  end subroutine
end module

module mint_module
  use FKSParams
  implicit none
  integer,parameter :: ndimmax=5,nintegrals=6,max_fold=8
  integer :: iconfig=1,ndim=5,ifold(5)=[1,1,3,1,1]
  integer :: ifold_energy=3,ifold_yij=4,ifold_phi=5
  integer :: born_spread_current_bin=1,born_spread_bin_fold(8),born_spread_sector_fold(8)
  integer :: born_spread_phase=0
  logical :: new_point=.true.,pass_cuts_check=.false.
  double precision :: virt_wgt_mint(0:1)=0d0,born_wgt_mint(0:1)=0d0
contains
  subroutine born_spread_set_point(x,y)
    double precision :: x,y
  end subroutine
end module

module mc_counterterms
  implicit none
  integer :: fksfather=3
  double precision :: gfactsf=1d0,gfactcl=1d0
contains
  double precision function mc_shower_scale_mass()
    mc_shower_scale_mass=0d0
  end function
end module

module process_module
  logical :: qed_matching=.false.
  integer,parameter :: ndelH=5
end module

module scale_module
  implicit none
  integer :: born_flow_picked=0
  double precision :: emsca_H(1,8,5,5)=1d0
  double precision :: shower_scale_n1body(5,5)=1d0,shower_scale_nbody(4,4)=1d0
end module

module mc_shower_state
  type native_shower_state
    integer :: slot
  end type
contains
  subroutine park_native_shower(state)
    type(native_shower_state) state
  end subroutine
  subroutine activate_native_shower(state)
    type(native_shower_state) state
  end subroutine
  subroutine materialize_native_shower()
  end subroutine
  subroutine forget_parked_shower()
  end subroutine
end module

integer(8) function native_provider_result_bytes()
  native_provider_result_bytes=0_8
end function
