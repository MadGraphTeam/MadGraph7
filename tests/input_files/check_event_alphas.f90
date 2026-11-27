! The strong coupling written with an MC@NLO event (AQCDUP) must be the
! one of its own kinematics, not the one left over from the last
! phase-space evaluation. Check the contribution that supplies it, and
! the scales of the bogus soft Born added after all folds.
module event_alphas_stubs
  implicit none
  logical :: decline_add=.false.
  integer :: nFKS_active=0
end module event_alphas_stubs

subroutine update_fks_dir(nFKS_in)
  use event_alphas_stubs
  implicit none
  integer nFKS_in
  nFKS_active=nFKS_in
end subroutine update_fks_dir

subroutine amp_split_pos_to_orders(iamp,orders)
  implicit none
  integer iamp,orders(2)
  orders=[2*iamp,0]
end subroutine amp_split_pos_to_orders

integer function get_orders_tag(orders)
  implicit none
  integer orders(2)
  get_orders_tag=100*orders(1)+orders(2)
end function get_orders_tag

subroutine add_wgt(type,orders,wgt1,wgt2,wgt3)
! As add_wgt, store the scales and coupling found in the common blocks:
! here the leftovers of an unrelated phase-space evaluation.
  use weight_lines
  use event_alphas_stubs
  implicit none
  integer type,orders(2)
  double precision wgt1,wgt2,wgt3
  if (decline_add) return
  icontr=icontr+1
  call weight_lines_allocated(5,icontr,1,1)
  itype(icontr)=type
  H_event(icontr)=.false.
  nFKS(icontr)=nFKS_active
  wgt(:,icontr)=[wgt1,wgt2,wgt3]
  scales2(:,icontr)=-1d0
  g_strong(icontr)=-1d0
end subroutine add_wgt

program check_event_alphas
  use weight_lines
  use event_alphas_stubs
  implicit none
  integer pdg_type_d(2,5),fks_i_d(2)
  common/test_fks_info/pdg_type_d,fks_i_d
  call check_event_scale_contribution()
  call check_bogus_soft_born()
  write(*,*) 'PASS event alphas'
contains
  subroutine check_event_scale_contribution()
    integer, parameter :: types(9)=[12,2,3,8,13,1,11,9,12]
    logical, parameter :: hevent(9)=[.false.,.false.,.false.,.true.,  &
         .true.,.true.,.false.,.true.,.false.]
    integer i
    call weight_lines_allocated(5,9,1,1)
    icontr=9
    do i=1,icontr
       itype(i)=types(i)
       H_event(i)=hevent(i)
       g_strong(i)=1d0+0.1d0*i
    enddo
    group_size=0
! S event summed into an MC term, evaluated with real-emission scales.
    call add_group_member(1,1)
    call add_group_member(1,2)
    call add_group_member(1,3)
    call add_group_member(1,7)
! H event led by a G-function term, with counterevent scales.
    call add_group_member(4,4)
    call add_group_member(4,5)
    call add_group_member(4,6)
! Without a record at the event scales, keep the picked contribution.
    call add_group_member(8,8)
    call add_group_member(9,9)
    call pack_contribution_groups()
    icontr_picked=1
    if (picked_scale_contr().ne.2) error stop 'S event must use its n-body scales'
    icontr_picked=4
    if (picked_scale_contr().ne.5) error stop 'H event must use its real-emission scales'
    icontr_picked=8
    if (picked_scale_contr().ne.8) error stop 'H fallback must be the picked contribution'
    icontr_picked=9
    if (picked_scale_contr().ne.9) error stop 'S fallback must be the picked contribution'
    call deallocate_weight_lines()
  end subroutine check_event_scale_contribution

  subroutine check_bogus_soft_born()
! Sector 1 has a quark as i_fks (no soft singularity), sector 2 a gluon.
    pdg_type_d=0
    pdg_type_d(1,5)=2
    pdg_type_d(2,5)=21
    fks_i_d=5
    call weight_lines_allocated(5,2,1,1)
    icontr=2
    H_event(1:2)=[.false.,.true.]
    itype(1:2)=[5,1]
    nFKS(1:2)=1
    amppos(1:2)=1
    scales2(:,1)=[16d0,25d0,36d0]
    g_strong(1)=1.2d0
    scales2(:,2)=[1d0,4d0,9d0]
    g_strong(2)=1.3d0
! A declined bogus Born must not change the records already there.
    decline_add=.true.
    call special_check_SoftSing(2)
    if (icontr.ne.2) error stop 'declined bogus Born was added'
    if (any(scales2(:,2).ne.[1d0,4d0,9d0]).or.g_strong(2).ne.1.3d0) &
         error stop 'declined bogus Born changed another record'
    decline_add=.false.
    call special_check_SoftSing(2)
    if (icontr.ne.3) error stop 'no bogus soft Born added'
    if (itype(3).ne.2.or.nFKS(3).ne.2.or.wgt(1,3).ne.1d-199) &
         error stop 'wrong bogus soft Born'
    if (any(scales2(:,3).ne.scales2(:,1)).or.g_strong(3).ne.g_strong(1)) &
         error stop 'bogus soft Born kept leftover scales'
    if (any(scales2(:,2).ne.[1d0,4d0,9d0]).or.g_strong(2).ne.1.3d0) &
         error stop 'bogus soft Born changed another record'
    call deallocate_weight_lines()
  end subroutine check_bogus_soft_born
end program check_event_alphas
