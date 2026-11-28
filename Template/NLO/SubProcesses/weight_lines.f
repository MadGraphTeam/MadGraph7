*     -*-fortran-*-

      module weight_lines
         use mc_context_memory, only: reserve_context,weight_store,memory_enabled
         implicit none
         integer :: max_contr=0,max_wgt=0,max_iproc=0,icontr=0,iwgt=0,
     $        icontr_picked=0,iproc_picked=0,born_spread_fold_capacity=0
         logical :: mc_H_only=.false.,mc_S_only=.false.,
     $        nonfinite_contribution=.false.
         integer :: nonfinite_type=0,nonfinite_sector=0,
     $        nonfinite_history=0
! Rejection statistics belong to the job across all live points.
         integer :: rejected_born_flow_points=0
         integer :: rejected_mc_kinematics_points=0
         integer :: rejected_nonfinite_points=0
         logical, allocatable :: H_event(:)
         integer, allocatable :: itype(:),nFKS(:),fks_flavor_class(:)
     $        ,born_flavor_class(:),QCDpower(:),pdg(:,:)
     $        ,pdg_uborn(:,:),parton_pdg_uborn(:,:,:),parton_pdg(:,:,:)
     $        ,plot_id(:),niproc(:),ipr(:)
     $        ,ifold_cnt(:)
     $        ,orderstag(:),amppos(:),need_match(:,:)
     $        ,vector_index(:)
! Keep the event/shower owner separate from the native PDF/ME history.
         integer, allocatable :: event_nFKS(:),native_ids(:,:)
! Packed contribution groups retain insertion order using linear storage.
         integer, allocatable :: group_size(:),group_start(:)
     $        ,group_members(:),group_owner(:),group_cursor(:)
         integer, allocatable :: icontr_sum(:,:),icolour_con(:,:,:)
     $        ,parton_pdf(:,:,:)
         double precision, allocatable :: momenta(:,:,:),momenta_m(:,:,:
     $        ,:),wgt(:,:),wgt_ME_tree(:,:),bjx(:,:),scales2(:,:)
     $        ,g_strong(:),wgts(:,:),parton_iproc(:,:),y_bst(:)
     $        ,cpower(:),plot_wgts(:,:),shower_scale(:),unwgt(:,:)
     $        ,unwgt_B(:,:),unwgt_noB(:,:)
     $        ,bias_wgt(:)
! Transfer the live contribution store between the legacy evaluator and
! one point. Each allocatable retains its shape, lower bounds and optional
! allocation status. No momentum, flavour row or diagram is copied.
         type native_weight_lines
            private
            logical :: parked=.false.
         integer :: max_contr=0,max_wgt=0,max_iproc=0,icontr=0,iwgt=0,
     $        icontr_picked=0,iproc_picked=0,born_spread_fold_capacity=0
         logical :: mc_H_only=.false.,mc_S_only=.false.,
     $        nonfinite_contribution=.false.
         integer :: nonfinite_type=0,nonfinite_sector=0,
     $        nonfinite_history=0
         logical, allocatable :: H_event(:)
         integer, allocatable :: itype(:),nFKS(:),fks_flavor_class(:)
     $        ,born_flavor_class(:),QCDpower(:),pdg(:,:)
     $        ,pdg_uborn(:,:),parton_pdg_uborn(:,:,:),parton_pdg(:,:,:)
     $        ,plot_id(:),niproc(:),ipr(:)
     $        ,ifold_cnt(:)
     $        ,orderstag(:),amppos(:),need_match(:,:)
     $        ,vector_index(:)
! Keep the event/shower owner separate from the native PDF/ME history.
         integer, allocatable :: event_nFKS(:),native_ids(:,:)
! Packed contribution groups retain insertion order using linear storage.
         integer, allocatable :: group_size(:),group_start(:)
     $        ,group_members(:),group_owner(:),group_cursor(:)
         integer, allocatable :: icontr_sum(:,:),icolour_con(:,:,:)
     $        ,parton_pdf(:,:,:)
         double precision, allocatable :: momenta(:,:,:),momenta_m(:,:,:
     $        ,:),wgt(:,:),wgt_ME_tree(:,:),bjx(:,:),scales2(:,:)
     $        ,g_strong(:),wgts(:,:),parton_iproc(:,:),y_bst(:)
     $        ,cpower(:),plot_wgts(:,:),shower_scale(:),unwgt(:,:)
     $        ,unwgt_B(:,:),unwgt_noB(:,:)
     $        ,bias_wgt(:)
         end type
! Stores of finished native points, kept at their grown capacity for the
! next fresh point instead of regrowing a store from one contribution.
         type(native_weight_lines), allocatable :: spare_lines(:)
         integer :: nspare_lines=0
         save
      contains
         subroutine budget_weight_lines(nlegs,contributions,nweights,nprocess,folds,legacy)
         integer,intent(in) :: nlegs,contributions,nweights,nprocess,folds
         logical,intent(in) :: legacy
         integer(8) n,c,w,p,f,bytes,largest
         if(.not.memory_enabled)return
         n=nlegs
         c=contributions
         w=nweights
         p=nprocess
         f=folds
c Linear contribution arrays counted by their element types. Include
c the zero-indexed PDG columns, double momentum triplets, PDF rows,
c packed group lists, scale weights, and any optional legacy snapshots.
         bytes=4*(24*c+2*n*(c+1)+2*n*p*c+n*c)
     $        +8*(12*n*c+16*c+2*w*c+2*p*c+2*p*f)
         largest=max(8*8*n*c,8*w*c,8*p*c,8*p*f,4*n*p*c)
         if(legacy)then
            bytes=bytes+4*(n*p*c+(c+1)*c+2*n*c)
            largest=max(largest,4*(c+1)*c)
         endif
         call reserve_context(weight_store,bytes,largest)
         end subroutine

         subroutine park_weight_lines(state)
         type(native_weight_lines),intent(inout) :: state
         if(state%parked)error stop 'Native weight lines already parked'
         call move_alloc(H_event,state%H_event)
         call move_alloc(itype,state%itype)
         call move_alloc(nFKS,state%nFKS)
         call move_alloc(fks_flavor_class,state%fks_flavor_class)
         call move_alloc(born_flavor_class,state%born_flavor_class)
         call move_alloc(QCDpower,state%QCDpower)
         call move_alloc(pdg,state%pdg)
         call move_alloc(pdg_uborn,state%pdg_uborn)
         call move_alloc(parton_pdg_uborn,state%parton_pdg_uborn)
         call move_alloc(parton_pdg,state%parton_pdg)
         call move_alloc(plot_id,state%plot_id)
         call move_alloc(niproc,state%niproc)
         call move_alloc(ipr,state%ipr)
         call move_alloc(ifold_cnt,state%ifold_cnt)
         call move_alloc(orderstag,state%orderstag)
         call move_alloc(amppos,state%amppos)
         call move_alloc(need_match,state%need_match)
         call move_alloc(vector_index,state%vector_index)
         call move_alloc(event_nFKS,state%event_nFKS)
         call move_alloc(native_ids,state%native_ids)
         call move_alloc(group_size,state%group_size)
         call move_alloc(group_start,state%group_start)
         call move_alloc(group_members,state%group_members)
         call move_alloc(group_owner,state%group_owner)
         call move_alloc(group_cursor,state%group_cursor)
         call move_alloc(icontr_sum,state%icontr_sum)
         call move_alloc(icolour_con,state%icolour_con)
         call move_alloc(parton_pdf,state%parton_pdf)
         call move_alloc(momenta,state%momenta)
         call move_alloc(momenta_m,state%momenta_m)
         call move_alloc(wgt,state%wgt)
         call move_alloc(wgt_ME_tree,state%wgt_ME_tree)
         call move_alloc(bjx,state%bjx)
         call move_alloc(scales2,state%scales2)
         call move_alloc(g_strong,state%g_strong)
         call move_alloc(wgts,state%wgts)
         call move_alloc(parton_iproc,state%parton_iproc)
         call move_alloc(y_bst,state%y_bst)
         call move_alloc(cpower,state%cpower)
         call move_alloc(plot_wgts,state%plot_wgts)
         call move_alloc(shower_scale,state%shower_scale)
         call move_alloc(unwgt,state%unwgt)
         call move_alloc(unwgt_B,state%unwgt_B)
         call move_alloc(unwgt_noB,state%unwgt_noB)
         call move_alloc(bias_wgt,state%bias_wgt)
         state%max_contr=max_contr
         state%max_wgt=max_wgt
         state%max_iproc=max_iproc
         state%icontr=icontr
         state%iwgt=iwgt
         state%icontr_picked=icontr_picked
         state%iproc_picked=iproc_picked
         state%born_spread_fold_capacity=born_spread_fold_capacity
         state%mc_H_only=mc_H_only
         state%mc_S_only=mc_S_only
         state%nonfinite_contribution=nonfinite_contribution
         state%nonfinite_type=nonfinite_type
         state%nonfinite_sector=nonfinite_sector
         state%nonfinite_history=nonfinite_history
         max_contr=0
         max_wgt=0
         max_iproc=0
         icontr=0
         iwgt=0
         icontr_picked=0
         iproc_picked=0
         born_spread_fold_capacity=0
         mc_H_only=.false.
         mc_S_only=.false.
         nonfinite_contribution=.false.
         nonfinite_type=0
         nonfinite_sector=0
         nonfinite_history=0
         state%parked=.true.
         end subroutine

         subroutine activate_weight_lines(state)
         type(native_weight_lines),intent(inout) :: state
         if(.not.state%parked)error stop 'Native weight lines are not parked'
         if(allocated(itype))error stop
     $        'Park current native weight lines before activating another point'
         call move_alloc(state%H_event,H_event)
         call move_alloc(state%itype,itype)
         call move_alloc(state%nFKS,nFKS)
         call move_alloc(state%fks_flavor_class,fks_flavor_class)
         call move_alloc(state%born_flavor_class,born_flavor_class)
         call move_alloc(state%QCDpower,QCDpower)
         call move_alloc(state%pdg,pdg)
         call move_alloc(state%pdg_uborn,pdg_uborn)
         call move_alloc(state%parton_pdg_uborn,parton_pdg_uborn)
         call move_alloc(state%parton_pdg,parton_pdg)
         call move_alloc(state%plot_id,plot_id)
         call move_alloc(state%niproc,niproc)
         call move_alloc(state%ipr,ipr)
         call move_alloc(state%ifold_cnt,ifold_cnt)
         call move_alloc(state%orderstag,orderstag)
         call move_alloc(state%amppos,amppos)
         call move_alloc(state%need_match,need_match)
         call move_alloc(state%vector_index,vector_index)
         call move_alloc(state%event_nFKS,event_nFKS)
         call move_alloc(state%native_ids,native_ids)
         call move_alloc(state%group_size,group_size)
         call move_alloc(state%group_start,group_start)
         call move_alloc(state%group_members,group_members)
         call move_alloc(state%group_owner,group_owner)
         call move_alloc(state%group_cursor,group_cursor)
         call move_alloc(state%icontr_sum,icontr_sum)
         call move_alloc(state%icolour_con,icolour_con)
         call move_alloc(state%parton_pdf,parton_pdf)
         call move_alloc(state%momenta,momenta)
         call move_alloc(state%momenta_m,momenta_m)
         call move_alloc(state%wgt,wgt)
         call move_alloc(state%wgt_ME_tree,wgt_ME_tree)
         call move_alloc(state%bjx,bjx)
         call move_alloc(state%scales2,scales2)
         call move_alloc(state%g_strong,g_strong)
         call move_alloc(state%wgts,wgts)
         call move_alloc(state%parton_iproc,parton_iproc)
         call move_alloc(state%y_bst,y_bst)
         call move_alloc(state%cpower,cpower)
         call move_alloc(state%plot_wgts,plot_wgts)
         call move_alloc(state%shower_scale,shower_scale)
         call move_alloc(state%unwgt,unwgt)
         call move_alloc(state%unwgt_B,unwgt_B)
         call move_alloc(state%unwgt_noB,unwgt_noB)
         call move_alloc(state%bias_wgt,bias_wgt)
         max_contr=state%max_contr
         max_wgt=state%max_wgt
         max_iproc=state%max_iproc
         icontr=state%icontr
         iwgt=state%iwgt
         icontr_picked=state%icontr_picked
         iproc_picked=state%iproc_picked
         born_spread_fold_capacity=state%born_spread_fold_capacity
         mc_H_only=state%mc_H_only
         mc_S_only=state%mc_S_only
         nonfinite_contribution=state%nonfinite_contribution
         nonfinite_type=state%nonfinite_type
         nonfinite_sector=state%nonfinite_sector
         nonfinite_history=state%nonfinite_history
         state%parked=.false.
         end subroutine

         subroutine recycle_weight_lines(state)
! Keep a finished point's parked store. Only an idle evaluator can move
! it: the swap below passes it through the active arrays.
         type(native_weight_lines),intent(inout) :: state
         type(native_weight_lines),allocatable :: grown(:)
         if(.not.state%parked.or..not.allocated(state%itype))return
         if(allocated(itype))return
         if(.not.allocated(spare_lines))allocate(spare_lines(16))
         if(nspare_lines.eq.size(spare_lines))then
            allocate(grown(2*size(spare_lines)))
            grown(1:nspare_lines)=spare_lines(1:nspare_lines)
            call move_alloc(grown,spare_lines)
         endif
         call activate_weight_lines(state)
         nspare_lines=nspare_lines+1
         call park_weight_lines(spare_lines(nspare_lines))
         end subroutine recycle_weight_lines

         logical function adopt_spare_weight_lines(nlegs) result(adopted)
! Give a fresh point a recycled store with the contents of a fresh one:
! no contributions, no optional calibration or legacy arrays. Entries
! past icontr are never read, as for the width-one evaluator's store.
         integer,intent(in) :: nlegs
         adopted=.false.
         if(nspare_lines.eq.0.or.allocated(itype))return
         if(size(spare_lines(nspare_lines)%pdg,1).ne.nlegs)return
         call activate_weight_lines(spare_lines(nspare_lines))
         nspare_lines=nspare_lines-1
         icontr=0
         iwgt=0
         icontr_picked=0
         iproc_picked=0
         mc_H_only=.false.
         mc_S_only=.false.
         nonfinite_contribution=.false.
         nonfinite_type=0
         nonfinite_sector=0
         nonfinite_history=0
         call release_born_spread_lines
         if(allocated(parton_pdf))deallocate(parton_pdf)
         if(allocated(icontr_sum))deallocate(icontr_sum)
         if(allocated(icolour_con))deallocate(icolour_con)
         call budget_weight_lines(nlegs,max_contr,max_wgt,max_iproc,0,.false.)
         adopted=.true.
         end function adopt_spare_weight_lines

         subroutine release_spare_weight_lines
         if(allocated(spare_lines))deallocate(spare_lines)
         nspare_lines=0
         end subroutine release_spare_weight_lines

         integer function event_owner(ict)
         integer, intent(in) :: ict
         event_owner=nFKS(ict)
         if (event_nFKS(ict).gt.0) event_owner=event_nFKS(ict)
         end function event_owner

         subroutine add_group_member(igroup,ict)
         integer, intent(in) :: igroup,ict
         group_size(igroup)=group_size(igroup)+1
         group_owner(ict)=igroup
         end subroutine add_group_member

         subroutine pack_contribution_groups
! Membership is assigned once per contribution. Packing by its original
! index preserves the order of additions, unweighting and reweighting.
         integer i,igroup,offset
         offset=1
         do i=1,icontr
            group_start(i)=offset
            group_cursor(i)=offset
            offset=offset+group_size(i)
         enddo
         do i=1,icontr
            igroup=group_owner(i)
            group_members(group_cursor(igroup))=i
            group_cursor(igroup)=group_cursor(igroup)+1
         enddo
         end subroutine pack_contribution_groups

         integer function group_member(imember,igroup)
         integer, intent(in) :: imember,igroup
         group_member=group_members(group_start(igroup)+imember-1)
         end function group_member

         subroutine allocate_born_spread_lines(nfolds)
         integer, intent(in) :: nfolds
         double precision, allocatable :: temp(:,:)
         if(memory_enabled)call budget_weight_lines(size(pdg,1),max_contr,max_wgt,max_iproc,
     $        max(nfolds,born_spread_fold_capacity),allocated(icontr_sum))
         if (.not.allocated(unwgt_B)) then
            allocate(unwgt_B(max_iproc,nfolds))
            allocate(unwgt_noB(max_iproc,nfolds))
            born_spread_fold_capacity=nfolds
         elseif (nfolds.gt.born_spread_fold_capacity) then
            allocate(temp(max_iproc,nfolds))
            temp(:,1:born_spread_fold_capacity)=unwgt_B
            call move_alloc(temp,unwgt_B)
            allocate(temp(max_iproc,nfolds))
            temp(:,1:born_spread_fold_capacity)=unwgt_noB
            call move_alloc(temp,unwgt_noB)
            born_spread_fold_capacity=nfolds
         endif
         unwgt_B=0d0
         unwgt_noB=0d0
         end subroutine allocate_born_spread_lines

         subroutine release_born_spread_lines
         if (allocated(unwgt_B)) deallocate(unwgt_B)
         if (allocated(unwgt_noB)) deallocate(unwgt_noB)
         born_spread_fold_capacity=0
         end subroutine release_born_spread_lines
      end module weight_lines

      subroutine deallocate_born_spread_lines
      use weight_lines, only: release_born_spread_lines
      implicit none
      call release_born_spread_lines
      return
      end


      subroutine weight_lines_allocated(nexternal,n_contr,n_wgt,n_proc)
      use weight_lines
      implicit none
      integer n_contr,n_wgt,n_proc,nexternal,contr_capacity
      logical, allocatable :: ltemp1(:)
      integer, allocatable :: itemp1(:),itemp2(:,:),itemp3(:,:,:)
      double precision, allocatable :: temp1(:),temp2(:,:),temp3(:,:,:)
     $     ,temp4(:,:,:,:)
c Check if we arrays are allocated and if we need to increase the size
c of the allocated arrays.
      if (.not. allocated(itype)) then
         call allocate_weight_lines(nexternal)
      endif
      if(memory_enabled.and.(n_contr.gt.max_contr.or.n_proc.gt.max_iproc.or.n_wgt.gt.max_wgt))then
         contr_capacity=max_contr
         if(n_contr.gt.max_contr)then
            if(max_contr.gt.huge(max_contr)/2)error stop 'Native weight-line capacity overflow'
            contr_capacity=max(n_contr,32,2*max_contr)
         endif
         call budget_weight_lines(nexternal,contr_capacity,max(max_wgt,n_wgt),
     $        max(max_iproc,n_proc),born_spread_fold_capacity,allocated(icontr_sum))
      endif
c --- increase size of max_iproc ---
      if (n_proc.gt.max_iproc) then
c parton_pdg_uborn
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg_uborn
         call move_alloc(itemp3,parton_pdg_uborn)
c parton_pdg
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg
         call move_alloc(itemp3,parton_pdg)
c parton_iproc
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
c unwgt
         allocate(temp2(n_proc,max_contr))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
c Born-spreading decomposition is allocated only during calibration.
         if (allocated(unwgt_B)) then
            allocate(temp2(n_proc,born_spread_fold_capacity))
            temp2(1:max_iproc,1:born_spread_fold_capacity)=unwgt_B
            call move_alloc(temp2,unwgt_B)
         endif
         if (allocated(unwgt_noB)) then
            allocate(temp2(n_proc,born_spread_fold_capacity))
            temp2(1:max_iproc,1:born_spread_fold_capacity)=unwgt_noB
            call move_alloc(temp2,unwgt_noB)
         endif
c parton_pdf
         if (allocated(parton_pdf)) then
         allocate(itemp3(nexternal,n_proc,max_contr))
         itemp3(:,1:max_iproc,:)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
         endif
c update maximum
         max_iproc=n_proc
      endif
c --- increase size of max_wgt ---
      if (n_wgt.gt.max_wgt) then
c wgts
         allocate(temp2(n_wgt,max_contr))
         temp2(1:max_wgt,1:max_contr)=wgts
         call move_alloc(temp2,wgts)
c plot_wgts
         allocate(temp2(n_wgt,max_contr))
         temp2(1:max_wgt,1:max_contr)=plot_wgts
         call move_alloc(temp2,plot_wgts)
c update maximum
         max_wgt=n_wgt
      endif
c --- increase size of max_contr ---
      if (n_contr.gt.max_contr) then
c Reserve geometrically: adding one native history must not copy all
c contribution arrays on every append. Keep the input arguments intact;
c callers can pass module capacities themselves as actual arguments.
         contr_capacity=max(n_contr,32,2*max_contr)
c H_event
         allocate(ltemp1(contr_capacity))
         ltemp1(1:max_contr)=H_event
         call move_alloc(ltemp1,H_event)
c itype
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=itype
         call move_alloc(itemp1,itype)
c nFKS
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=nFKS
         call move_alloc(itemp1,nFKS)
c fks_flavor_class
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=fks_flavor_class
         call move_alloc(itemp1,fks_flavor_class)
c born_flavor_class
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=born_flavor_class
         call move_alloc(itemp1,born_flavor_class)
c event_nFKS
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=event_nFKS
         call move_alloc(itemp1,event_nFKS)
c native provenance
         allocate(itemp2(3,contr_capacity))
         itemp2=0
         itemp2(:,1:max_contr)=native_ids
         call move_alloc(itemp2,native_ids)
c QCDpower         
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=QCDpower
         call move_alloc(itemp1,QCDpower)
c pdg
         allocate(itemp2(nexternal,0:contr_capacity))
         itemp2(1:nexternal,0:max_contr)=pdg
         call move_alloc(itemp2,pdg)
c pdg_uborn
         allocate(itemp2(nexternal,0:contr_capacity))
         itemp2(1:nexternal,0:max_contr)=pdg_uborn
         call move_alloc(itemp2,pdg_uborn)
c parton_pdg_uborn
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg_uborn
         call move_alloc(itemp3,parton_pdg_uborn)
c parton_pdg
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdg
         call move_alloc(itemp3,parton_pdg)
c plot_id
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=plot_id
         call move_alloc(itemp1,plot_id)
c ifold_cnt
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=ifold_cnt
         call move_alloc(itemp1,ifold_cnt)
c niproc
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=niproc
         call move_alloc(itemp1,niproc)
c ipr
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=ipr
         call move_alloc(itemp1,ipr)
c orderstag
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=orderstag
         call move_alloc(itemp1,orderstag)
c amppos
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=amppos
         call move_alloc(itemp1,amppos)
c vector_index
         allocate(itemp1(contr_capacity))
         itemp1(1:max_contr)=vector_index
         call move_alloc(itemp1,vector_index)
c parton_pdf
         if (allocated(parton_pdf)) then
         allocate(itemp3(nexternal,max_iproc,contr_capacity))
         itemp3(1:nexternal,1:max_iproc,1:max_contr)=parton_pdf
         call move_alloc(itemp3,parton_pdf)
         endif
c icontr_sum
         if (allocated(icontr_sum)) then
         allocate(itemp2(0:contr_capacity,contr_capacity))
         itemp2(0:max_contr,1:max_contr)=icontr_sum
         call move_alloc(itemp2,icontr_sum)
         endif
c icolour_con
         if (allocated(icolour_con)) then
         allocate(itemp3(2,nexternal,contr_capacity))
         itemp3(1:2,1:nexternal,1:max_contr)=icolour_con
         call move_alloc(itemp3,icolour_con)
         endif
c group_size
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_size
         call move_alloc(itemp1,group_size)
c group_start
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_start
         call move_alloc(itemp1,group_start)
c group_members
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_members
         call move_alloc(itemp1,group_members)
c group_owner
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_owner
         call move_alloc(itemp1,group_owner)
c group_cursor
         allocate(itemp1(contr_capacity))
         itemp1=0
         itemp1(1:max_contr)=group_cursor
         call move_alloc(itemp1,group_cursor)
c momemta
         allocate(temp3(0:3,nexternal,contr_capacity))
         temp3(0:3,1:nexternal,1:max_contr)=momenta
         call move_alloc(temp3,momenta)
c momemta_m
         allocate(temp4(0:3,nexternal,2,contr_capacity))
         temp4(0:3,1:nexternal,1:2,1:max_contr)=momenta_m
         call move_alloc(temp4,momenta_m)
c wgt
         allocate(temp2(3,contr_capacity))
         temp2(1:3,1:max_contr)=wgt
         call move_alloc(temp2,wgt)
c wgt_ME_tree
         allocate(temp2(2,contr_capacity))
         temp2(1:2,1:max_contr)=wgt_ME_tree
         call move_alloc(temp2,wgt_ME_tree)
c bjx
         allocate(temp2(2,contr_capacity))
         temp2(1:2,1:max_contr)=bjx
         call move_alloc(temp2,bjx)
c scales2
         allocate(temp2(3,contr_capacity))
         temp2(1:3,1:max_contr)=scales2
         call move_alloc(temp2,scales2)
c g_strong
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=g_strong
         call move_alloc(temp1,g_strong)
c wgts
         allocate(temp2(max_wgt,contr_capacity))
         temp2(1:max_wgt,1:max_contr)=wgts
         call move_alloc(temp2,wgts)
c parton_iproc
         allocate(temp2(max_iproc,contr_capacity))
         temp2(1:max_iproc,1:max_contr)=parton_iproc
         call move_alloc(temp2,parton_iproc)
c y_bst
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=y_bst
         call move_alloc(temp1,y_bst)
c cpower
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=cpower
         call move_alloc(temp1,cpower)
c bias_wgt
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=bias_wgt
         call move_alloc(temp1,bias_wgt)
c plot_wgts
         allocate(temp2(max_wgt,contr_capacity))
         temp2(1:max_wgt,1:max_contr)=plot_wgts
         call move_alloc(temp2,plot_wgts)
c shower_scale
         allocate(temp1(contr_capacity))
         temp1(1:max_contr)=shower_scale
         call move_alloc(temp1,shower_scale)
c unwgt
         allocate(temp2(max_iproc,contr_capacity))
         temp2(1:max_iproc,1:max_contr)=unwgt
         call move_alloc(temp2,unwgt)
c need_match
         allocate(itemp2(nexternal,1:contr_capacity))
         itemp2(1:nexternal,1:max_contr)=need_match
         call move_alloc(itemp2,need_match)
c update maximum
         max_contr=contr_capacity
      endif
      return
      end

      subroutine allocate_weight_lines(nexternal)
      use weight_lines
      implicit none
      integer nexternal
      call budget_weight_lines(nexternal,1,1,1,0,.false.)
      allocate(H_event(1))
      allocate(itype(1))
      allocate(nFKS(1))
      allocate(fks_flavor_class(1))
      allocate(born_flavor_class(1))
      allocate(event_nFKS(1))
      event_nFKS=0
      allocate(native_ids(3,1))
      native_ids=0
      allocate(QCDpower(1))
      allocate(pdg(nexternal,0:1))
      allocate(pdg_uborn(nexternal,0:1))
      allocate(parton_pdg_uborn(nexternal,1,1))
      allocate(parton_pdg(nexternal,1,1))
      allocate(plot_id(1))
      allocate(ifold_cnt(1))
      allocate(niproc(1))
      allocate(ipr(1))
      allocate(orderstag(1))
      allocate(amppos(1))
      allocate(group_size(1))
      group_size=0
      allocate(group_start(1))
      group_start=0
      allocate(group_members(1))
      group_members=0
      allocate(group_owner(1))
      group_owner=0
      allocate(group_cursor(1))
      group_cursor=0
      allocate(momenta(0:3,nexternal,1))
      allocate(momenta_m(0:3,nexternal,2,1))
      allocate(wgt(3,1))
      allocate(wgt_ME_tree(2,1))
      allocate(bjx(2,1))
      allocate(scales2(3,1))
      allocate(g_strong(1))
      allocate(wgts(1,1))
      allocate(parton_iproc(1,1))
      allocate(y_bst(1))
      allocate(cpower(1))
      allocate(bias_wgt(1))
      allocate(plot_wgts(1,1))
      allocate(shower_scale(1))
      allocate(unwgt(1,1))
      allocate(need_match(nexternal,1))
      allocate(vector_index(1))
      max_contr=1
      max_wgt=1
      max_iproc=1
      return
      end

      subroutine allocate_legacy_weight_snapshots(nexternal)
      use weight_lines
      implicit none
      integer nexternal
      call budget_weight_lines(nexternal,max_contr,max_wgt,max_iproc,born_spread_fold_capacity,.true.)
      if (.not.allocated(parton_pdf))
     $     allocate(parton_pdf(nexternal,max_iproc,max_contr))
      if (.not.allocated(icontr_sum))
     $     allocate(icontr_sum(0:max_contr,max_contr))
      if (.not.allocated(icolour_con))
     $     allocate(icolour_con(2,nexternal,max_contr))
      end

      subroutine deallocate_weight_lines
      use weight_lines
      implicit none
      call release_spare_weight_lines
      max_contr=0
      max_wgt=0
      max_iproc=0
      if (allocated(H_event)) deallocate(H_event)
      if (allocated(itype)) deallocate(itype)
      if (allocated(nFKS)) deallocate(nFKS)
      if (allocated(fks_flavor_class)) deallocate(fks_flavor_class)
      if (allocated(born_flavor_class)) deallocate(born_flavor_class)
      if (allocated(event_nFKS)) deallocate(event_nFKS)
      if (allocated(native_ids)) deallocate(native_ids)
      if (allocated(QCDpower)) deallocate(QCDpower)
      if (allocated(pdg)) deallocate(pdg)
      if (allocated(pdg_uborn)) deallocate(pdg_uborn)
      if (allocated(parton_pdg_uborn)) deallocate(parton_pdg_uborn)
      if (allocated(parton_pdg)) deallocate(parton_pdg)
      if (allocated(plot_id)) deallocate(plot_id)
      if (allocated(ifold_cnt)) deallocate(ifold_cnt)
      if (allocated(niproc)) deallocate(niproc)
      if (allocated(ipr)) deallocate(ipr)
      if (allocated(orderstag)) deallocate(orderstag)
      if (allocated(amppos)) deallocate(amppos)
      if (allocated(group_size)) deallocate(group_size)
      if (allocated(group_start)) deallocate(group_start)
      if (allocated(group_members)) deallocate(group_members)
      if (allocated(group_owner)) deallocate(group_owner)
      if (allocated(group_cursor)) deallocate(group_cursor)
      if (allocated(momenta)) deallocate(momenta)
      if (allocated(momenta_m)) deallocate(momenta_m)
      if (allocated(wgt)) deallocate(wgt)
      if (allocated(wgt_ME_tree)) deallocate(wgt_ME_tree)
      if (allocated(bjx)) deallocate(bjx)
      if (allocated(scales2)) deallocate(scales2)
      if (allocated(g_strong)) deallocate(g_strong)
      if (allocated(wgts)) deallocate(wgts)
      if (allocated(parton_iproc)) deallocate(parton_iproc)
      if (allocated(y_bst)) deallocate(y_bst)
      if (allocated(cpower)) deallocate(cpower)
      if (allocated(bias_wgt)) deallocate(bias_wgt)
      if (allocated(plot_wgts)) deallocate(plot_wgts)
      if (allocated(shower_scale)) deallocate(shower_scale)
      if (allocated(unwgt)) deallocate(unwgt)
      call release_born_spread_lines
      if (allocated(need_match)) deallocate(need_match)
      if (allocated(vector_index)) deallocate(vector_index)
      if (allocated(parton_pdf)) deallocate(parton_pdf)
      if (allocated(icontr_sum)) deallocate(icontr_sum)
      if (allocated(icolour_con)) deallocate(icolour_con)
      return
      end
