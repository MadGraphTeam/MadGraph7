module flow_fixture
  use weight_lines
  implicit none
  double precision :: values(3)=[1d0,3d0,0d0],uniform=0.5d0,last_grid_weight=-1d0
  integer :: draws=0,born_calls=0,fail_native_fold=0,surrogate_fit_calls=0
  integer :: nlo_calls=0,s_only_calls=0,native_calls=0
  integer :: sector_draws=0
  double precision :: last_prefactor_weight=1d0
  logical :: real_point_active=.false.,born_cuts=.true.
  logical :: outer_no_endpoint=.false.
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
  use flow_fixture, only: outer_no_endpoint
  implicit none
  type fks_born_point
    logical :: valid=.false.,event_projection=.true.
    double precision :: p(0:3,4)=1d0,xbjrk(2)=0.2d0
  end type
  type fks_phase_space_point
    double precision :: p(0:3,5)=1d0,p_lab(0:3,5)=1d0,p_cms(0:3,5)=1d0
    type(fks_born_point) :: born
  end type
contains
  subroutine generate_born_contribution(ndim,config,jac,x,point,cached_born)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    type(fks_born_point),optional :: cached_born
    integer :: i_fks,j_fks
    common/fks_indices/i_fks,j_fks
    point=fks_phase_space_point()
    p_born=1d0
    p1_cnt=1d0
    p1_cnt(:,i_fks,0)=0d0
    if(outer_no_endpoint)p1_cnt(0,1,0)=-99d0
    if(present(cached_born))cached_born=point%born
  end subroutine
  subroutine generate_real_phase_space(ndim,config,jac,x,point,cached_born)
    integer :: ndim,config
    double precision :: jac,x(99)
    type(fks_phase_space_point) :: point
    type(fks_born_point),optional :: cached_born
    integer :: i_fks,j_fks
    common/fks_indices/i_fks,j_fks
    point=fks_phase_space_point()
    p1_cnt(:,:,0)=1d0
    p1_cnt(:,i_fks,0)=0d0
    if(present(cached_born))cached_born=point%born
  end subroutine
end module

module mc_native_context
  use flow_fixture, only: real_point_active
  implicit none
contains
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
  use flow_fixture,only: sector_draws
  implicit none
  integer,parameter :: ndimmax=5,nintegrals=6,max_fold=32
  integer :: iconfig=1,ndim=5,ifold(5)=[1,1,3,1,1]
  integer :: ifold_energy=3,ifold_yij=4,ifold_phi=5
  integer :: born_spread_current_bin=1,born_spread_bin_fold(32),born_spread_sector_fold(32)
  integer :: born_spread_phase=0
  logical :: sector_learning_collect=.false.,sector_learning_active=.false.
  integer :: learning_registers=0,learning_begins=0,learning_selects=0,learning_marks=0,learning_rejects=0
  integer :: learning_owners(1024)=0,learning_selected_slot=0
  logical :: new_point=.true.,pass_cuts_check=.false.
  double precision :: virt_wgt_mint(0:1)=0d0,born_wgt_mint(0:1)=0d0
contains
  subroutine sector_learning_begin_point
    if(.not.sector_learning_collect)return
    learning_begins=learning_begins+1
    learning_owners=0
    learning_selected_slot=0
  end subroutine
  subroutine sector_learning_select_slot(slot)
    integer :: slot
    if(.not.sector_learning_active)return
    if(learning_selected_slot.ne.0.and.slot.ne.learning_selected_slot)error stop 'probe changed Born slot'
    learning_selected_slot=slot
    learning_selects=learning_selects+1
  end subroutine
  subroutine sector_learning_mark_range(first,last,owner)
    integer :: first,last,owner
    if(.not.sector_learning_collect)return
    if(first.lt.1.or.last.gt.size(learning_owners))error stop 'bad owner range'
    learning_marks=learning_marks+1
    learning_owners(first:last)=owner
  end subroutine
  subroutine sector_learning_reject_point
    if(sector_learning_collect)learning_rejects=learning_rejects+1
  end subroutine
  subroutine resample_sector_radiation(x,updated,ratio,sector_id)
    double precision :: x(ndimmax),updated(ndimmax),ratio
    integer,optional :: sector_id
    updated=x
    ratio=1d0
    if(sector_learning_active)then
      sector_draws=sector_draws+1
      updated(3:4)=[0.1d0+sector_draws*0.1d0,0.3d0]
      ratio=2d0
    endif
  end subroutine
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
  double precision :: emsca_H(1,32,5,5)=1d0
  double precision :: shower_scale_n1body(5,5)=1d0,shower_scale_nbody(4,4)=1d0
end module

subroutine register_sector_learning_layout(slot,n,sectors)
  use mint_module,only: learning_registers,sector_learning_active
  integer :: slot,n,sectors(n)
  if(.not.sector_learning_active)return
  if(slot.ne.1.or.n.ne.1.or.any(sectors.ne.1))error stop 'bad complete slot registration'
  learning_registers=learning_registers+1
end subroutine

subroutine virt_surrogate_fit_point
  use flow_fixture,only: surrogate_fit_calls
  surrogate_fit_calls=surrogate_fit_calls+1
end subroutine
