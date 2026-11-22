      module mc_native_points
      use iso_fortran_env, only: int64
      use mc_point_state, only: native_fold_sample,native_real_emission,
     $     native_flow_cursor
      use weight_lines, only: native_weight_lines,park_weight_lines,
     $     activate_weight_lines,itype,recycle_weight_lines,
     $     adopt_spare_weight_lines
      use mc_native_context, only: native_provider_request,
     $     native_provider_point_state,mc_capture_provider_point,
     $     mc_activate_provider_point,mc_restore_provider_model,
     $     native_mapping,shared_real_active
      use mc_shower_state, only: native_shower_state,park_native_shower,
     $     activate_native_shower,materialize_native_shower,
     $     forget_parked_shower
      use fks_phase_space, only: fks_phase_space_point,
     $     capture_fks_phase_space,restore_fks_phase_space,
     $     initialize_fks_phase_space
      implicit none
      private
      include 'nexternal.inc'
      include 'maxparticles.inc'
      include 'nFKSconfigs.inc'
      include 'orders.inc'
      integer,parameter,public :: mc_fold_complete=0,mc_wait_real=1
c Saved state precisely at the existing inner native H real return.
      type, public :: native_history_cursor
         logical :: waiting=.false.,cuts_born=.false.,cuts_real=.false.
         integer :: owner,ihist,sector,ii,jj,outer_config,
     $        nrecoil,recoil,first_alt,flow,called
         integer,allocatable :: recoil_configs(:),group_of(:)
         integer :: owner_match(nexternal),matching(nexternal,3),
     $        colours_s(2,nexternal-1),colours_h(2,nexternal)
         double precision :: p_flipped(0:3,nexternal),pn(0:3,nexternal),
     $        pn_cms(0:3,nexternal),pn_lab(0:3,nexternal),
     $        outer_measure,sector_weight,outer_boost,factor,jac_native
         double precision :: hard_scale,nbody_scales(nexternal-1,nexternal-1,3),
     $        n1body_scales(nexternal,nexternal),factors(9),
     $        mc_factors(8),pdf_factors(3),gfun(3)
         double precision,allocatable :: emsca(:,:,:)
         type(fks_phase_space_point) :: saved_point
      end type
      type, public :: native_mc_point
         integer(int64) :: draw_id=0
         logical :: batched=.true.
         integer :: sequence=0,status=mc_fold_complete,resume_ifl=-1
         type(native_fold_sample) sample
         type(native_flow_cursor) flow
         type(native_history_cursor) history
         logical :: cuts_born=.false.,cuts_real=.false.
         type(native_real_emission) emission
         type(native_provider_request) provider
         type(native_provider_point_state) provider_state
         type(native_shower_state) shower
         type(native_weight_lines) lines
c Outer fold sampling and the cursor after a native real contribution.
         integer :: sector_index=0,first_H=0
         double precision :: x_local(99),jacPS
         logical :: shared_born=.false.,has_chart=.false.
         double precision :: sum_born(0:3,nexternal-1),sum_xbjrk(2)
         type(fks_phase_space_point) :: outer,active_chart
c Live COMMONs with readers after the real return or during closing.
         logical :: parked=.false.,nbody=.false.,virtual_ready=.false.
         integer :: sector=0,channel=0
         integer :: granny(fks_configs)
         logical :: write_granny(fks_configs)
         double precision :: factors(9),mc_factors(8),pdf_factors(3),
     $        born_factors(2),tree(2),bjorken(2),q2fact(2),qes2,scale,
     $        coupling_momenta(0:3,nexternal)
         double precision :: virtual_ratio=0d0,virtual_total=0d0,
     $        virtual_split(amp_split_size)=0d0
         double precision,allocatable :: coordinates(:,:)
      end type
      public :: park_native_point,activate_native_point,native_flow_cursor
      public :: recycle_native_point,settle_native_points,
     $     release_native_points
      contains

      subroutine park_native_point(point)
      use mc_context_memory, only: reserve_context,provider_store,memory_enabled
      use mint_module, only: ndimmax,max_fold,iconfig
      implicit none
      include 'run.inc'
      include 'q_es.inc'
      type(native_mc_point),intent(inout),target :: point
      integer(8) result_bytes,native_provider_result_bytes
      external native_provider_result_bytes
      integer nFKSprocess,nndim,ifold_picked,nfold
      double precision x_save(ndimmax,max_fold),factors(9),
     $     mc_factors(8),pdf_factors(3),born_factors(2),tree(2),
     $     vob,virt_total,virt_split(amp_split_size)
      logical nbody,virt_ready
      common/c_nFKSprocess/nFKSprocess
      common/tosigint/nndim
      common/c_vegas_x_fold/x_save,ifold_picked
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      common/factor_nbody/born_factors
      common/c_wgt_ME_tree/tree
      double precision pp(0:3,max_particles)
      common/momenta_pp/pp
      common/c_vob/vob
      common/c_folded_virtual/virt_ready
      common/c_folded_virtual_value/virt_total,virt_split
      common/cnbody/nbody
      integer granny(fks_configs)
      logical write_granny(fks_configs)
      common/write_granny_resonance/granny,write_granny
      if(point%parked)error stop 'Native MC point already parked'
      point%granny=granny
      point%write_granny=write_granny
      point%sector=nFKSprocess
      point%channel=iconfig
      if(point%has_chart)then
         point%active_chart=point%outer
         if(point%history%waiting)then
            point%active_chart%p=point%history%pn
            point%active_chart%p_lab=point%history%pn_lab
            point%active_chart%p_cms=point%history%pn_cms
            point%active_chart%weight=point%history%jac_native
         endif
         call capture_fks_phase_space(point%active_chart)
      endif
      if(memory_enabled)then
         result_bytes=native_provider_result_bytes()
         call reserve_context(provider_store,result_bytes,result_bytes)
      endif
      call mc_capture_provider_point(point%provider_state)
      call park_native_shower(point%shower)
      call park_weight_lines(point%lines)
      point%factors=factors
      point%mc_factors=mc_factors
      point%pdf_factors=pdf_factors
      point%born_factors=born_factors
      point%tree=tree
      point%bjorken=xbk
      point%q2fact=q2fact
      point%qes2=QES2
      point%scale=scale
      point%coupling_momenta=pp(:,1:nexternal)
      point%nbody=nbody
      point%virtual_ratio=vob
      point%virtual_ready=virt_ready
      if(virt_ready)then
         point%virtual_total=virt_total
         point%virtual_split=virt_split
      endif
      nfold=point%sample%fold_number
      if(nfold.gt.0)point%coordinates=x_save(1:nndim,1:nfold)
      point%parked=.true.
      end subroutine

      subroutine activate_native_point(point)
      use mint_module, only: ndimmax,max_fold,iconfig
      implicit none
      include 'run.inc'
      include 'q_es.inc'
      type(native_mc_point),intent(inout) :: point
      integer nndim,ifold_picked,fold,ifold_counter
      double precision x_save(ndimmax,max_fold),factors(9),
     $     mc_factors(8),pdf_factors(3),born_factors(2),tree(2),
     $     vob,virt_total,virt_split(amp_split_size)
      logical nbody,virt_ready
      common/tosigint/nndim
      common/c_vegas_x_fold/x_save,ifold_picked
      common/cfl/fold,ifold_counter
      common/factor_n1body/factors
      common/factor_n1body_NLOPS/mc_factors
      common/factor_pdfsch/pdf_factors
      common/factor_nbody/born_factors
      common/c_wgt_ME_tree/tree
      double precision pp(0:3,max_particles)
      common/momenta_pp/pp
      common/c_vob/vob
      common/c_folded_virtual/virt_ready
      common/c_folded_virtual_value/virt_total,virt_split
      common/cnbody/nbody
      integer granny(fks_configs)
      logical write_granny(fks_configs)
      common/write_granny_resonance/granny,write_granny
c The first sector setup used to allocate the zero-index PDG scratch
c only once per job. A fresh point also needs it before its first add_wgt
c when the previous point has taken the active store (notably in FxFx).
      if(.not.point%parked)then
         call materialize_native_shower()
         native_mapping=.false.
         shared_real_active=.false.
         call mc_set_history(0)
         if(.not.allocated(itype))then
            if(.not.adopt_spare_weight_lines(nexternal))
     $           call weight_lines_allocated(nexternal,1,1,1)
         endif
         return
      endif
      iconfig=point%channel
      nbody=point%nbody
      xbk=point%bjorken
      q2fact=point%q2fact
      QES2=point%qes2
      scale=point%scale
      pp(:,1:nexternal)=point%coupling_momenta
c Rebuilding sector cuts reads the caller's current model masses.
      call mc_restore_provider_model(point%provider_state)
      if(point%sector.gt.0)call update_fks_dir(point%sector)
      call mc_activate_provider_point(point%provider_state,
     $     model_is_active=.true.)
      if(point%has_chart)then
         if(point%history%waiting)then
c Native recoil selection computes the cut bounds in the history's
c ordinary sector with native_mapping temporarily disabled.
            call select_native_recoil(point%active_chart%config)
         else
            call initialize_fks_phase_space(point%active_chart%config)
         endif
         call restore_fks_phase_space(point%active_chart)
      endif
      call activate_native_shower(point%shower)
      call activate_weight_lines(point%lines)
      factors=point%factors
      mc_factors=point%mc_factors
      pdf_factors=point%pdf_factors
      born_factors=point%born_factors
      tree=point%tree
      vob=point%virtual_ratio
      virt_ready=point%virtual_ready
      if(virt_ready)then
         virt_total=point%virtual_total
         virt_split=point%virtual_split
      endif
      ifold_counter=point%sample%fold_number
      if(allocated(point%coordinates))
     $     x_save(1:nndim,1:ifold_counter)=point%coordinates
      granny=point%granny
      write_granny=point%write_granny
      point%parked=.false.
      end subroutine

      subroutine settle_native_points()
c Leave the last parked point's state active, as the copying protocol did.
      call materialize_native_shower()
      end subroutine

      subroutine release_native_points()
c The parked states are about to be freed.
      call forget_parked_shower()
      end subroutine

      subroutine recycle_native_point(point)
c A released cohort hands its weight-line stores to the next fresh points.
      type(native_mc_point),intent(inout) :: point
      call recycle_weight_lines(point%lines)
      end subroutine
      end module
