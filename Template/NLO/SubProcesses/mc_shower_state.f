      module mc_shower_state
c Lane-owned shower kinematics, colour maps and matching state. Global
c run settings, immutable clustering topology and fully keyed caches
c stay in their native modules.
      use process_module, only: next_n,
     $     next_n1,
     $     qed_matching,
     $     mass_n,
     $     mass_n1,
     $     charge_n,
     $     charge_n1,
     $     colour_n,
     $     colour_n1,
     $     iRtoB,
     $     iBtoR,
     $     pdg_n,
     $     pdg_n1,
     $     valid_dipole_n,
     $     valid_dipole_n1,
     $     qed_dipole_n,
     $     qed_dipole_n1,
     $     ickkw_mod
      use scale_module, only: global_ref_scale,
     $     shower_scale_hard,
     $     showerscaleS_hard,
     $     born_flow_picked,
     $     shower_scale_nbody,
     $     shower_scale_nbody_min,
     $     shower_scale_nbody_max,
     $     shower_scale_n1body,
     $     showerscaleS,
     $     showerscaleH,
     $     emsca_S,
     $     emsca_H,
     $     emsca_S_hard,
     $     event_colour_H
      use mc_counterterms, only: native_kernel_state,park_mc_kernel,
     $     activate_mc_kernel
      implicit none
      private
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'nFKSconfigs.inc'
      integer,parameter :: maxfksloc=512
      character(2000) clusterstrings(0:maxfksloc)
      character(1000) ptclusstring
      common /c_clusterstrings/ clusterstrings
      common /c_ptclusstring/ ptclusstring
      double precision :: p_last_izero(0:3,nexternal)
      double precision :: p_last_mohdr(0:3,nexternal)
      double precision :: fxfx_ren_scales_izero(0:nexternal)
      double precision :: fxfx_ren_scales_mohdr(0:nexternal)
      double precision :: fxfx_fac_scale_izero(2)
      double precision :: fxfx_fac_scale_mohdr(2)
      double precision :: rewgt_izero
      double precision :: rewgt_mohdr
      integer :: iterm_last_izero
      integer :: iterm_last_mohdr
      integer :: nfxfx_ren_scales_izero
      integer :: nfxfx_ren_scales_mohdr
      integer :: need_matching_izero(nexternal)
      logical :: rewgt_mohdr_calculated
      logical :: rewgt_izero_calculated
      common /native_fxfx_real_cache/ p_last_izero
     $     ,p_last_mohdr
     $     ,fxfx_ren_scales_izero
     $     ,fxfx_ren_scales_mohdr
     $     ,fxfx_fac_scale_izero
     $     ,fxfx_fac_scale_mohdr
     $     ,rewgt_izero
     $     ,rewgt_mohdr
      common /native_fxfx_integer_cache/ iterm_last_izero
     $     ,iterm_last_mohdr
     $     ,nfxfx_ren_scales_izero
     $     ,nfxfx_ren_scales_mohdr
     $     ,need_matching_izero
      common /native_fxfx_valid_cache/ rewgt_mohdr_calculated
     $     ,rewgt_izero_calculated
      logical :: is_leading_cflow(max_bcol)
      integer :: num_leading_cflows
      common /c_leading_cflows/ is_leading_cflow
     $     ,num_leading_cflows
      integer :: icolup_s(2,nexternal-1)
      integer :: icolup_h(2,nexternal)
      common /colour_connections/ icolup_s
     $     ,icolup_h
      integer :: MCcntcalled
      common /c_MCcntcalled/ MCcntcalled
      double precision :: ch_i
      double precision :: ch_j
      double precision :: ch_m
      integer :: i_type
      integer :: j_type
      integer :: m_type
      integer :: j_pdg
      common /cparticle_types/ ch_i
     $     ,ch_j
     $     ,ch_m
     $     ,i_type
     $     ,j_type
     $     ,m_type
     $     ,j_pdg
      integer :: iSorH_lhe
      integer :: ifks_lhe(fks_configs)
      integer :: jfks_lhe(fks_configs)
      integer :: fksfather_lhe(fks_configs)
      integer :: ipartner_lhe(fks_configs)
      common /cto_LHE1/ iSorH_lhe
     $     ,ifks_lhe
     $     ,jfks_lhe
     $     ,fksfather_lhe
     $     ,ipartner_lhe
      double precision :: scale1_lhe(fks_configs)
      double precision :: scale2_lhe(fks_configs)
      common /cto_LHE2/ scale1_lhe
     $     ,scale2_lhe
      double precision :: fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      double precision :: fxfx_fac_scale(2)
      double precision :: fxfx_ren_scales(0:nexternal)
      integer :: nFxFx_ren_scales
      common /c_FxFx_scales/ fxfx_fac_scale
     $     ,fxfx_ren_scales
     $     ,nFxFx_ren_scales
      integer :: need_matching_S(nexternal)
      integer :: need_matching_H(nexternal)
      integer :: need_matching_cuts(nexternal)
      common /c_need_matching/ need_matching_S
     $     ,need_matching_H
     $     ,need_matching_cuts
      type cluster_record
         character(:),allocatable :: text
      end type
      type fxfx_history
         double precision :: p_last_izero(0:3,nexternal)
         double precision :: p_last_mohdr(0:3,nexternal)
         double precision :: fxfx_ren_scales_izero(0:nexternal)
         double precision :: fxfx_ren_scales_mohdr(0:nexternal)
         double precision :: fxfx_fac_scale_izero(2)
         double precision :: fxfx_fac_scale_mohdr(2)
         double precision :: rewgt_izero
         double precision :: rewgt_mohdr
         integer :: iterm_last_izero
         integer :: iterm_last_mohdr
         integer :: nfxfx_ren_scales_izero
         integer :: nfxfx_ren_scales_mohdr
         integer :: need_matching_izero(nexternal)
         logical :: rewgt_mohdr_calculated
         logical :: rewgt_izero_calculated
         double precision :: fxfx_exp_rewgt
         double precision :: fxfx_fac_scale(2)
         double precision :: fxfx_ren_scales(0:nexternal)
         integer :: nFxFx_ren_scales
         integer :: need_matching_S(nexternal)
         integer :: need_matching_H(nexternal)
         integer :: need_matching_cuts(nexternal)
         character(:),allocatable :: ptclus
         type(cluster_record),allocatable :: cluster(:)
      end type
      type,public :: native_shower_state
         logical :: parked=.false.
         type(native_kernel_state) :: kernel
         type(fxfx_history),allocatable :: matching
         integer :: next_n
         integer :: next_n1
         logical :: qed_matching
         double precision :: global_ref_scale
         double precision :: shower_scale_hard
         double precision :: showerscaleS_hard
         integer :: born_flow_picked
         logical :: is_leading_cflow(max_bcol)
         integer :: num_leading_cflows
         integer :: icolup_s(2,nexternal-1)
         integer :: icolup_h(2,nexternal)
         integer :: MCcntcalled
         double precision :: ch_i
         double precision :: ch_j
         double precision :: ch_m
         integer :: i_type
         integer :: j_type
         integer :: m_type
         integer :: j_pdg
         integer :: iSorH_lhe
         integer :: ifks_lhe(fks_configs)
         integer :: jfks_lhe(fks_configs)
         integer :: fksfather_lhe(fks_configs)
         integer :: ipartner_lhe(fks_configs)
         double precision :: scale1_lhe(fks_configs)
         double precision :: scale2_lhe(fks_configs)
         double precision,allocatable :: mass_n(:)
         double precision,allocatable :: mass_n1(:)
         double precision,allocatable :: charge_n(:)
         double precision,allocatable :: charge_n1(:)
         integer,allocatable :: colour_n(:)
         integer,allocatable :: colour_n1(:)
         integer,allocatable :: iRtoB(:)
         integer,allocatable :: iBtoR(:)
         integer,allocatable :: pdg_n(:)
         integer,allocatable :: pdg_n1(:)
         logical,allocatable :: valid_dipole_n(:,:,:)
         logical,allocatable :: valid_dipole_n1(:,:)
         logical,allocatable :: qed_dipole_n(:,:)
         logical,allocatable :: qed_dipole_n1(:,:)
         double precision,allocatable :: shower_scale_nbody(:,:)
         double precision,allocatable :: shower_scale_nbody_min(:,:)
         double precision,allocatable :: shower_scale_nbody_max(:,:)
         double precision,allocatable :: shower_scale_n1body(:,:)
         double precision,allocatable :: showerscaleS(:,:)
         double precision,allocatable :: showerscaleH(:,:)
         double precision,allocatable :: emsca_S(:,:,:,:)
         double precision,allocatable :: emsca_H(:,:,:,:)
         double precision,allocatable :: emsca_S_hard(:,:)
         integer,allocatable :: event_colour_H(:,:,:,:)
      end type
      public :: park_native_shower,activate_native_shower,shower_heap_bound
      public :: materialize_native_shower,forget_parked_shower
c The per-point scale arrays are only written while a point evaluates,
c so a lane switch swaps their storage instead of copying it. After a
c park the active arrays hold a stale buffer until the next activation;
c a fresh point first copies the last parked point's values, which is
c exactly what the copying protocol left active.
      type(native_shower_state),pointer,save :: last_parked=>null()
      logical,save :: scales_stale=.false.
      interface swap_storage
         module procedure swap_d2,swap_d4,swap_i4
      end interface
      contains
      integer(8) function shower_heap_bound() result(bytes)
      use process_module, only: nr=>max_flows_n
      type(fxfx_history) history
      type(cluster_record) cluster
      integer(8) n
      n=nexternal
c Process arrays can first appear on different FKS sectors. Bound both
c Born and real shapes from generated dimensions even on the first draw.
      bytes=2*n*(2*8+3*4)+4*n*n*(3+nr)
c Global scale array extents encode the actual delta/folding choices.
      bytes=bytes+size(shower_scale_nbody,kind=8)*8
      bytes=bytes+size(shower_scale_nbody_min,kind=8)*8
      bytes=bytes+size(shower_scale_nbody_max,kind=8)*8
      bytes=bytes+size(shower_scale_n1body,kind=8)*8
      bytes=bytes+size(showerscaleS,kind=8)*8
      bytes=bytes+size(showerscaleH,kind=8)*8
      bytes=bytes+size(emsca_S,kind=8)*8
      bytes=bytes+size(emsca_H,kind=8)*8
      bytes=bytes+size(emsca_S_hard,kind=8)*8
      bytes=bytes+size(event_colour_H,kind=8)*4
      if(ickkw_mod.eq.3)bytes=bytes+storage_size(history,kind=8)/8
     $     +len(ptclusstring)+(1_8+min(fks_configs,maxfksloc))*
     $     (storage_size(cluster,kind=8)/8+len(clusterstrings(0)))
      end function

      subroutine swap_d2(a,b)
      double precision,allocatable :: a(:,:),b(:,:),c(:,:)
      call move_alloc(a,c)
      call move_alloc(b,a)
      call move_alloc(c,b)
      end subroutine

      subroutine swap_d4(a,b)
      double precision,allocatable :: a(:,:,:,:),b(:,:,:,:),c(:,:,:,:)
      call move_alloc(a,c)
      call move_alloc(b,a)
      call move_alloc(c,b)
      end subroutine

      subroutine swap_i4(a,b)
      integer,allocatable :: a(:,:,:,:),b(:,:,:,:),c(:,:,:,:)
      call move_alloc(a,c)
      call move_alloc(b,a)
      call move_alloc(c,b)
      end subroutine

      subroutine swap_scales(state)
      type(native_shower_state),intent(inout) :: state
      call swap_storage(shower_scale_nbody,state%shower_scale_nbody)
      call swap_storage(shower_scale_nbody_min,
     $     state%shower_scale_nbody_min)
      call swap_storage(shower_scale_nbody_max,
     $     state%shower_scale_nbody_max)
      call swap_storage(shower_scale_n1body,state%shower_scale_n1body)
      call swap_storage(showerscaleS,state%showerscaleS)
      call swap_storage(showerscaleH,state%showerscaleH)
      call swap_storage(emsca_S,state%emsca_S)
      call swap_storage(emsca_H,state%emsca_H)
      call swap_storage(emsca_S_hard,state%emsca_S_hard)
      call swap_storage(event_colour_H,state%event_colour_H)
      end subroutine

      subroutine materialize_native_shower()
c Give the active arrays the last parked point's values (fresh points and
c readers outside the cohort).
      if (.not.scales_stale) return
      scales_stale=.false.
      if (.not.associated(last_parked)) return
      if (.not.allocated(last_parked%emsca_S)) then
c The parked point had no scale arrays: neither has the active state.
         if (allocated(emsca_S)) deallocate(shower_scale_nbody,
     $        shower_scale_nbody_min,shower_scale_nbody_max,
     $        shower_scale_n1body,showerscaleS,showerscaleH,emsca_S,
     $        emsca_H,emsca_S_hard,event_colour_H)
         return
      endif
      shower_scale_nbody=last_parked%shower_scale_nbody
      shower_scale_nbody_min=last_parked%shower_scale_nbody_min
      shower_scale_nbody_max=last_parked%shower_scale_nbody_max
      shower_scale_n1body=last_parked%shower_scale_n1body
      showerscaleS=last_parked%showerscaleS
      showerscaleH=last_parked%showerscaleH
      emsca_S=last_parked%emsca_S
      emsca_H=last_parked%emsca_H
      emsca_S_hard=last_parked%emsca_S_hard
      event_colour_H=last_parked%event_colour_H
      end subroutine

      subroutine forget_parked_shower()
c Before the parked states are released.
      call materialize_native_shower()
      nullify(last_parked)
      end subroutine

      subroutine park_native_shower(state)
c Assign into the lane's buffers: same-shaped arrays keep their storage
c across the lane's waves instead of being freed and reallocated. The
c per-point scale arrays are swapped (see last_parked).
      type(native_shower_state),intent(inout),target :: state
      integer i,n
      call park_mc_kernel(state%kernel)
      state%next_n=next_n
      state%next_n1=next_n1
      state%qed_matching=qed_matching
      state%global_ref_scale=global_ref_scale
      state%shower_scale_hard=shower_scale_hard
      state%showerscaleS_hard=showerscaleS_hard
      state%born_flow_picked=born_flow_picked
      state%is_leading_cflow=is_leading_cflow
      state%num_leading_cflows=num_leading_cflows
      state%icolup_s=icolup_s
      state%icolup_h=icolup_h
      state%MCcntcalled=MCcntcalled
      state%ch_i=ch_i
      state%ch_j=ch_j
      state%ch_m=ch_m
      state%i_type=i_type
      state%j_type=j_type
      state%m_type=m_type
      state%j_pdg=j_pdg
      state%iSorH_lhe=iSorH_lhe
      state%ifks_lhe=ifks_lhe
      state%jfks_lhe=jfks_lhe
      state%fksfather_lhe=fksfather_lhe
      state%ipartner_lhe=ipartner_lhe
      state%scale1_lhe=scale1_lhe
      state%scale2_lhe=scale2_lhe
      if (allocated(mass_n)) then
         state%mass_n=mass_n
         state%mass_n1=mass_n1
         state%charge_n=charge_n
         state%charge_n1=charge_n1
         state%colour_n=colour_n
         state%colour_n1=colour_n1
         state%iRtoB=iRtoB
         state%iBtoR=iBtoR
         state%pdg_n=pdg_n
         state%pdg_n1=pdg_n1
         state%valid_dipole_n=valid_dipole_n
         state%valid_dipole_n1=valid_dipole_n1
         state%qed_dipole_n=qed_dipole_n
         state%qed_dipole_n1=qed_dipole_n1
      elseif (allocated(state%mass_n)) then
         deallocate(state%mass_n,state%mass_n1,state%charge_n,
     $        state%charge_n1,state%colour_n,state%colour_n1,state%iRtoB,
     $        state%iBtoR,state%pdg_n,state%pdg_n1,state%valid_dipole_n,
     $        state%valid_dipole_n1,state%qed_dipole_n,
     $        state%qed_dipole_n1)
      endif
      if (scales_stale) error stop 'Parked a point with stale scales'
      call swap_scales(state)
      last_parked=>state
      scales_stale=.true.
      if (ickkw_mod.ne.3.and.allocated(state%matching))
     $     deallocate(state%matching)
      if (ickkw_mod.eq.3) then
         if (.not.allocated(state%matching)) allocate(state%matching)
         state%matching%p_last_izero=p_last_izero
         state%matching%p_last_mohdr=p_last_mohdr
         state%matching%fxfx_ren_scales_izero=fxfx_ren_scales_izero
         state%matching%fxfx_ren_scales_mohdr=fxfx_ren_scales_mohdr
         state%matching%fxfx_fac_scale_izero=fxfx_fac_scale_izero
         state%matching%fxfx_fac_scale_mohdr=fxfx_fac_scale_mohdr
         state%matching%rewgt_izero=rewgt_izero
         state%matching%rewgt_mohdr=rewgt_mohdr
         state%matching%iterm_last_izero=iterm_last_izero
         state%matching%iterm_last_mohdr=iterm_last_mohdr
         state%matching%nfxfx_ren_scales_izero=nfxfx_ren_scales_izero
         state%matching%nfxfx_ren_scales_mohdr=nfxfx_ren_scales_mohdr
         state%matching%need_matching_izero=need_matching_izero
         state%matching%rewgt_mohdr_calculated=rewgt_mohdr_calculated
         state%matching%rewgt_izero_calculated=rewgt_izero_calculated
         state%matching%fxfx_exp_rewgt=fxfx_exp_rewgt
         state%matching%fxfx_fac_scale=fxfx_fac_scale
         state%matching%fxfx_ren_scales=fxfx_ren_scales
         state%matching%nFxFx_ren_scales=nFxFx_ren_scales
         state%matching%need_matching_S=need_matching_S
         state%matching%need_matching_H=need_matching_H
         state%matching%need_matching_cuts=need_matching_cuts
         state%matching%ptclus=trim(ptclusstring)
         n=min(fks_configs,maxfksloc)
         if (allocated(state%matching%cluster)) then
            if (ubound(state%matching%cluster,1).ne.n)
     $           deallocate(state%matching%cluster)
         endif
         if (.not.allocated(state%matching%cluster))
     $        allocate(state%matching%cluster(0:n))
         do i=0,ubound(state%matching%cluster,1)
            state%matching%cluster(i)%text=trim(clusterstrings(i))
         enddo
      endif
      state%parked=.true.
      end subroutine

      subroutine activate_native_shower(state)
      type(native_shower_state),intent(inout) :: state
      integer i
      if (.not.state%parked) error stop 'inactive MC shower point'
      call activate_mc_kernel(state%kernel)
      next_n=state%next_n
      next_n1=state%next_n1
      qed_matching=state%qed_matching
      global_ref_scale=state%global_ref_scale
      shower_scale_hard=state%shower_scale_hard
      showerscaleS_hard=state%showerscaleS_hard
      born_flow_picked=state%born_flow_picked
      is_leading_cflow=state%is_leading_cflow
      num_leading_cflows=state%num_leading_cflows
      icolup_s=state%icolup_s
      icolup_h=state%icolup_h
      MCcntcalled=state%MCcntcalled
      ch_i=state%ch_i
      ch_j=state%ch_j
      ch_m=state%ch_m
      i_type=state%i_type
      j_type=state%j_type
      m_type=state%m_type
      j_pdg=state%j_pdg
      iSorH_lhe=state%iSorH_lhe
      ifks_lhe=state%ifks_lhe
      jfks_lhe=state%jfks_lhe
      fksfather_lhe=state%fksfather_lhe
      ipartner_lhe=state%ipartner_lhe
      scale1_lhe=state%scale1_lhe
      scale2_lhe=state%scale2_lhe
      if (allocated(state%mass_n)) then
         mass_n=state%mass_n
         mass_n1=state%mass_n1
         charge_n=state%charge_n
         charge_n1=state%charge_n1
         colour_n=state%colour_n
         colour_n1=state%colour_n1
         iRtoB=state%iRtoB
         iBtoR=state%iBtoR
         pdg_n=state%pdg_n
         pdg_n1=state%pdg_n1
         valid_dipole_n=state%valid_dipole_n
         valid_dipole_n1=state%valid_dipole_n1
         qed_dipole_n=state%qed_dipole_n
         qed_dipole_n1=state%qed_dipole_n1
      endif
c The parked arrays become active; the stale buffer moves to the state,
c which its next park overwrites by the same swap.
      call swap_scales(state)
      scales_stale=.false.
      if (allocated(state%matching)) then
         p_last_izero=state%matching%p_last_izero
         p_last_mohdr=state%matching%p_last_mohdr
         fxfx_ren_scales_izero=state%matching%fxfx_ren_scales_izero
         fxfx_ren_scales_mohdr=state%matching%fxfx_ren_scales_mohdr
         fxfx_fac_scale_izero=state%matching%fxfx_fac_scale_izero
         fxfx_fac_scale_mohdr=state%matching%fxfx_fac_scale_mohdr
         rewgt_izero=state%matching%rewgt_izero
         rewgt_mohdr=state%matching%rewgt_mohdr
         iterm_last_izero=state%matching%iterm_last_izero
         iterm_last_mohdr=state%matching%iterm_last_mohdr
         nfxfx_ren_scales_izero=state%matching%nfxfx_ren_scales_izero
         nfxfx_ren_scales_mohdr=state%matching%nfxfx_ren_scales_mohdr
         need_matching_izero=state%matching%need_matching_izero
         rewgt_mohdr_calculated=state%matching%rewgt_mohdr_calculated
         rewgt_izero_calculated=state%matching%rewgt_izero_calculated
         fxfx_exp_rewgt=state%matching%fxfx_exp_rewgt
         fxfx_fac_scale=state%matching%fxfx_fac_scale
         fxfx_ren_scales=state%matching%fxfx_ren_scales
         nFxFx_ren_scales=state%matching%nFxFx_ren_scales
         need_matching_S=state%matching%need_matching_S
         need_matching_H=state%matching%need_matching_H
         need_matching_cuts=state%matching%need_matching_cuts
         ptclusstring=state%matching%ptclus
         do i=0,ubound(state%matching%cluster,1)
            clusterstrings(i)=state%matching%cluster(i)%text
         enddo
      endif
      end subroutine

      end module
