*     -*-fortran-*-

      module weight_lines_vec
         implicit none
         integer :: max_contr_vec,max_wgt_vec,max_iproc_vec, vector_size_wgt
         integer, allocatable :: icontr_vec(:),iwgt_vec(:),icontr_picked_vec(:),iproc_picked_vec(:)
         logical, allocatable :: H_event_vec(:,:)
         integer, allocatable :: itype_vec(:,:),nFKS_vec(:,:),QCDpower_vec(:,:)
     $        ,pdg_vec(:,:,:),pdg_uborn_vec(:,:,:)
     $        ,parton_pdg_uborn_vec(:,:,:,:)
     $        ,parton_pdg_vec(:,:,:,:),plot_id_vec(:,:)
     $        ,niproc_vec(:,:),ipr_vec(:,:)
     $        ,parton_pdf_vec(:,:,:,:),icontr_sum_vec(:,:,:)
     $        ,ifold_cnt_vec(:,:)
     $        ,icolour_con_vec(:,:,:,:),orderstag_vec(:,:)
     $        ,amppos_vec(:,:)
     $        ,need_match_vec(:,:,:)
         double precision, allocatable :: momenta_vec(:,:,:,:)
     $        ,momenta_m_vec(:,:,:,:,:)
     $        ,wgt_vec(:,:,:),wgt_ME_tree_vec(:,:,:),bjx_vec(:,:,:)
     $        ,scales2_vec(:,:,:),g_strong_vec(:,:),wgts_vec(:,:,:)
     $        ,parton_iproc_vec(:,:,:),y_bst_vec(:,:)
     $        ,cpower_vec(:,:),plot_wgts_vec(:,:,:),shower_scale_vec(:,:)
     $        ,unwgt_vec(:,:,:),bias_wgt_vec(:,:)
     $        ,shower_scale_a_vec(:,:,:,:)
         save
      end module weight_lines_vec


      subroutine weight_lines_allocated_vec(nexternal,n_contr,n_wgt_vec,n_proc)
      use weight_lines_vec
      implicit none
      integer n_contr,n_wgt_vec,n_proc,nexternal
      logical, allocatable :: ltemp1(:,:)
      integer, allocatable :: itemp1(:,:),itemp2(:,:,:),itemp3(:,:,:,:)
      double precision, allocatable :: temp1(:,:),temp2(:,:,:)
     $     ,temp3(:,:,:,:)
     $     ,temp4(:,:,:,:,:)
c Check if we arrays are allocated and if we need to increase the size
c of the allocated arrays.
      if (.not. allocated(itype_vec)) then
         write(*,*) 'Allocating weight lines vec arrays for the first time'
         call allocate_weight_lines_vec(nexternal)
      endif
c --- increase size of max_iproc_vec ---
      if (n_proc.gt.max_iproc_vec) then
c parton_pdg_uborn_vec
         allocate(itemp3(nexternal,n_proc,max_contr_vec,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdg_uborn_vec
         call move_alloc(itemp3,parton_pdg_uborn_vec)
c parton_pdg_vec
         allocate(itemp3(nexternal,n_proc,max_contr_vec,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdg_vec
         call move_alloc(itemp3,parton_pdg_vec)
c parton_iproc_vec
         allocate(temp2(n_proc,max_contr_vec,vector_size_wgt))
         temp2(1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_iproc_vec
         call move_alloc(temp2,parton_iproc_vec)
c parton_pdf_vec
         allocate(itemp3(nexternal,n_proc,max_contr_vec,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdf_vec
         call move_alloc(itemp3,parton_pdf_vec)
c unwgt_vec
         allocate(temp2(n_proc,max_contr_vec,vector_size_wgt))
         temp2(1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=unwgt_vec
         call move_alloc(temp2,unwgt_vec)
c update maximum
         max_iproc_vec=n_proc
      endif
c --- increase size of max_wgt_vec ---
      if (n_wgt_vec.gt.max_wgt_vec) then
c wgts_vec
         allocate(temp2(n_wgt_vec,max_contr_vec,vector_size_wgt))
         temp2(1:max_wgt_vec,1:max_contr_vec,1:vector_size_wgt)=wgts_vec
         call move_alloc(temp2,wgts_vec)
c plot_wgts_vec
         allocate(temp2(n_wgt_vec,max_contr_vec,vector_size_wgt))
         temp2(1:max_wgt_vec,1:max_contr_vec,1:vector_size_wgt)=plot_wgts_vec
         call move_alloc(temp2,plot_wgts_vec)
c update maximum
         max_wgt_vec=n_wgt_vec
      endif
c --- increase size of max_contr_vec ---
      if (n_contr.gt.max_contr_vec) then
c H_event_vec
         allocate(ltemp1(n_contr,vector_size_wgt))
         ltemp1(1:max_contr_vec,1:vector_size_wgt)=H_event_vec
         call move_alloc(ltemp1,H_event_vec)
c itype_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=itype_vec
         call move_alloc(itemp1,itype_vec)
c nFKS_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=nFKS_vec
         call move_alloc(itemp1,nFKS_vec)
c QCDpower_vec         
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=QCDpower_vec
         call move_alloc(itemp1,QCDpower_vec)
c pdg_vec
         allocate(itemp2(nexternal,0:n_contr,vector_size_wgt))
         itemp2(1:nexternal,0:max_contr_vec,1:vector_size_wgt)=pdg_vec
         call move_alloc(itemp2,pdg_vec)
c pdg_uborn_vec
         allocate(itemp2(nexternal,0:n_contr,vector_size_wgt))
         itemp2(1:nexternal,0:max_contr_vec,1:vector_size_wgt)=pdg_uborn_vec
         call move_alloc(itemp2,pdg_uborn_vec)
c parton_pdg_uborn_vec
         allocate(itemp3(nexternal,max_iproc_vec,n_contr,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdg_uborn_vec
         call move_alloc(itemp3,parton_pdg_uborn_vec)
c parton_pdg_vec
         allocate(itemp3(nexternal,max_iproc_vec,n_contr,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdg_vec
         call move_alloc(itemp3,parton_pdg_vec)
c plot_id_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=plot_id_vec
         call move_alloc(itemp1,plot_id_vec)
c ifold_cnt_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=ifold_cnt_vec
         call move_alloc(itemp1,ifold_cnt_vec)
c niproc_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=niproc_vec
         call move_alloc(itemp1,niproc_vec)
c ipr_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=ipr_vec
         call move_alloc(itemp1,ipr_vec)
c orderstag_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=orderstag_vec
         call move_alloc(itemp1,orderstag_vec)
c amppos_vec
         allocate(itemp1(n_contr,vector_size_wgt))
         itemp1(1:max_contr_vec,1:vector_size_wgt)=amppos_vec
         call move_alloc(itemp1,amppos_vec)
! c vector_index
!          allocate(itemp1(n_contr,vector_size_wgt))
!          itemp1(1:max_contr_vec,1:vector_size_wgt)=vector_index
!          call move_alloc(itemp1,vector_index)
c parton_pdf_vec
         allocate(itemp3(nexternal,max_iproc_vec,n_contr,vector_size_wgt))
         itemp3(1:nexternal,1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_pdf_vec
         call move_alloc(itemp3,parton_pdf_vec)
c icontr_sum_vec
         allocate(itemp2(0:n_contr,n_contr,vector_size_wgt))
         itemp2(0:max_contr_vec,1:max_contr_vec,1:vector_size_wgt)=icontr_sum_vec
         call move_alloc(itemp2,icontr_sum_vec)
c icolour_con_vec
         allocate(itemp3(2,nexternal,n_contr,vector_size_wgt))
         itemp3(1:2,1:nexternal,1:max_contr_vec,1:vector_size_wgt)=icolour_con_vec
         call move_alloc(itemp3,icolour_con_vec)
c momemta
         allocate(temp3(0:3,nexternal,n_contr,vector_size_wgt))
         temp3(0:3,1:nexternal,1:max_contr_vec,1:vector_size_wgt)=momenta_vec
         call move_alloc(temp3,momenta_vec)
c momemta_m
         allocate(temp4(0:3,nexternal,2,n_contr,vector_size_wgt))
         temp4(0:3,1:nexternal,1:2,1:max_contr_vec,1:vector_size_wgt)=momenta_m_vec
         call move_alloc(temp4,momenta_m_vec)
c wgt_vec
         allocate(temp2(3,n_contr,vector_size_wgt))
         temp2(1:3,1:max_contr_vec,1:vector_size_wgt)=wgt_vec
         call move_alloc(temp2,wgt_vec)
c wgt_ME_tree_vec
         allocate(temp2(2,n_contr,vector_size_wgt))
         temp2(1:2,1:max_contr_vec,1:vector_size_wgt)=wgt_ME_tree_vec
         call move_alloc(temp2,wgt_ME_tree_vec)
c bjx_vec
         allocate(temp2(2,n_contr,vector_size_wgt))
         temp2(1:2,1:max_contr_vec,1:vector_size_wgt)=bjx_vec
         call move_alloc(temp2,bjx_vec)
c scales2_vec
         allocate(temp2(3,n_contr,vector_size_wgt))
         temp2(1:3,1:max_contr_vec,1:vector_size_wgt)=scales2_vec
         call move_alloc(temp2,scales2_vec)
c g_strong_vec
         allocate(temp1(n_contr,vector_size_wgt))
         temp1(1:max_contr_vec,1:vector_size_wgt)=g_strong_vec
         call move_alloc(temp1,g_strong_vec)
c wgts_vec
         allocate(temp2(max_wgt_vec,n_contr,vector_size_wgt))
         temp2(1:max_wgt_vec,1:max_contr_vec,1:vector_size_wgt)=wgts_vec
         call move_alloc(temp2,wgts_vec)
c parton_iproc_vec
         allocate(temp2(max_iproc_vec,n_contr,vector_size_wgt))
         temp2(1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=parton_iproc_vec
         call move_alloc(temp2,parton_iproc_vec)
c y_bst_vec
         allocate(temp1(n_contr,vector_size_wgt))
         temp1(1:max_contr_vec,1:vector_size_wgt)=y_bst_vec
         call move_alloc(temp1,y_bst_vec)
c cpower_vec
         allocate(temp1(n_contr,vector_size_wgt))
         temp1(1:max_contr_vec,1:vector_size_wgt)=cpower_vec
         call move_alloc(temp1,cpower_vec)
c bias_wgt_vec
         allocate(temp1(n_contr,vector_size_wgt))
         temp1(1:max_contr_vec,1:vector_size_wgt)=bias_wgt_vec
         call move_alloc(temp1,bias_wgt_vec)
c plot_wgts_vec
         allocate(temp2(max_wgt_vec,n_contr,vector_size_wgt))
         temp2(1:max_wgt_vec,1:max_contr_vec,1:vector_size_wgt)=plot_wgts_vec
         call move_alloc(temp2,plot_wgts_vec)
c shower_scale_vec
         allocate(temp1(n_contr,vector_size_wgt))
         temp1(1:max_contr_vec,1:vector_size_wgt)=shower_scale_vec
         call move_alloc(temp1,shower_scale_vec)
c shower_scale_a_vec
         allocate(temp3(n_contr,nexternal,nexternal,vector_size_wgt))
         temp3(1:max_contr_vec,1:nexternal,1:nexternal,1:vector_size_wgt)=shower_scale_a_vec
         call move_alloc(temp3,shower_scale_a_vec)
c unwgt_vec
         allocate(temp2(max_iproc_vec,n_contr,vector_size_wgt))
         temp2(1:max_iproc_vec,1:max_contr_vec,1:vector_size_wgt)=unwgt_vec
         call move_alloc(temp2,unwgt_vec)
c need_match_vec
         allocate(itemp2(nexternal,1:n_contr,vector_size_wgt))
         itemp2(1:nexternal,1:max_contr_vec,1:vector_size_wgt)=need_match_vec
         call move_alloc(itemp2,need_match_vec)
c update maximum
         max_contr_vec=n_contr
      endif
      return
      end

      subroutine allocate_weight_lines_vec(nexternal)
      use weight_lines_vec
      implicit none
      integer nexternal
      if(vector_size_wgt.le.0) vector_size_wgt=1
      allocate(icontr_vec(vector_size_wgt))
      allocate(iwgt_vec(vector_size_wgt))
      allocate(icontr_picked_vec(vector_size_wgt))
      allocate(iproc_picked_vec(vector_size_wgt))
      allocate(H_event_vec(1,vector_size_wgt))
      allocate(itype_vec(1,vector_size_wgt))
      allocate(nFKS_vec(1,vector_size_wgt))
      allocate(QCDpower_vec(1,vector_size_wgt))
      allocate(pdg_vec(nexternal,0:1,vector_size_wgt))
      allocate(pdg_uborn_vec(nexternal,0:1,vector_size_wgt))
      allocate(parton_pdg_uborn_vec(nexternal,1,1,vector_size_wgt))
      allocate(parton_pdg_vec(nexternal,1,1,vector_size_wgt))
      allocate(plot_id_vec(1,vector_size_wgt))
      allocate(ifold_cnt_vec(1,vector_size_wgt))
      allocate(niproc_vec(1,vector_size_wgt))
      allocate(ipr_vec(1,vector_size_wgt))
      allocate(orderstag_vec(1,vector_size_wgt))
      allocate(amppos_vec(1,vector_size_wgt))
      allocate(parton_pdf_vec(nexternal,1,1,vector_size_wgt))
      allocate(icontr_sum_vec(0:1,1,vector_size_wgt))
      allocate(icolour_con_vec(2,nexternal,1,vector_size_wgt))
      allocate(momenta_vec(0:3,nexternal,1,vector_size_wgt))
      allocate(momenta_m_vec(0:3,nexternal,2,1,vector_size_wgt))
      allocate(wgt_vec(3,1,vector_size_wgt))
      allocate(wgt_ME_tree_vec(2,1,vector_size_wgt))
      allocate(bjx_vec(2,1,vector_size_wgt))
      allocate(scales2_vec(3,1,vector_size_wgt))
      allocate(g_strong_vec(1,vector_size_wgt))
      allocate(wgts_vec(1,1,vector_size_wgt))
      allocate(parton_iproc_vec(1,1,vector_size_wgt))
      allocate(y_bst_vec(1,vector_size_wgt))
      allocate(cpower_vec(1,vector_size_wgt))
      allocate(bias_wgt_vec(1,vector_size_wgt))
      allocate(plot_wgts_vec(1,1,vector_size_wgt))
      allocate(shower_scale_vec(1,vector_size_wgt))
      allocate(shower_scale_a_vec(1,nexternal,nexternal,vector_size_wgt))
      allocate(unwgt_vec(1,1,vector_size_wgt))
      allocate(need_match_vec(nexternal,1,vector_size_wgt))
    !   allocate(vector_index(1,vector_size_wgt))
      max_contr_vec=1
      max_wgt_vec=1
      max_iproc_vec=1
      return
      end

      subroutine deallocate_weight_lines_vec
      use weight_lines_vec
      implicit none
      max_contr_vec=0
      max_wgt_vec=0
      max_iproc_vec=0
      if (allocated(H_event_vec)) deallocate(H_event_vec)
      if (allocated(itype_vec)) deallocate(itype_vec)
      if (allocated(nFKS_vec)) deallocate(nFKS_vec)
      if (allocated(QCDpower_vec)) deallocate(QCDpower_vec)
      if (allocated(pdg_vec)) deallocate(pdg_vec)
      if (allocated(pdg_uborn_vec)) deallocate(pdg_uborn_vec)
      if (allocated(parton_pdg_uborn_vec)) deallocate(parton_pdg_uborn_vec)
      if (allocated(parton_pdg_vec)) deallocate(parton_pdg_vec)
      if (allocated(plot_id_vec)) deallocate(plot_id_vec)
      if (allocated(ifold_cnt_vec)) deallocate(ifold_cnt_vec)
      if (allocated(niproc_vec)) deallocate(niproc_vec)
      if (allocated(ipr_vec)) deallocate(ipr_vec)
      if (allocated(orderstag_vec)) deallocate(orderstag_vec)
      if (allocated(amppos_vec)) deallocate(amppos_vec)
      if (allocated(parton_pdf_vec)) deallocate(parton_pdf_vec)
      if (allocated(icontr_sum_vec)) deallocate(icontr_sum_vec)
      if (allocated(icolour_con_vec)) deallocate(icolour_con_vec)
      if (allocated(momenta_vec)) deallocate(momenta_vec)
      if (allocated(momenta_m_vec)) deallocate(momenta_m_vec)
      if (allocated(wgt_vec)) deallocate(wgt_vec)
      if (allocated(wgt_ME_tree_vec)) deallocate(wgt_ME_tree_vec)
      if (allocated(bjx_vec)) deallocate(bjx_vec)
      if (allocated(scales2_vec)) deallocate(scales2_vec)
      if (allocated(g_strong_vec)) deallocate(g_strong_vec)
      if (allocated(wgts_vec)) deallocate(wgts_vec)
      if (allocated(parton_iproc_vec)) deallocate(parton_iproc_vec)
      if (allocated(y_bst_vec)) deallocate(y_bst_vec)
      if (allocated(cpower_vec)) deallocate(cpower_vec)
      if (allocated(bias_wgt_vec)) deallocate(bias_wgt_vec)
      if (allocated(plot_wgts_vec)) deallocate(plot_wgts_vec)
      if (allocated(shower_scale_vec)) deallocate(shower_scale_vec)
      if (allocated(shower_scale_a_vec)) deallocate(shower_scale_a_vec)
      if (allocated(unwgt_vec)) deallocate(unwgt_vec)
      if (allocated(need_match_vec)) deallocate(need_match_vec)
      return
      end

      subroutine reset_weight_lines_vec(nexternal)
      use weight_lines_vec
      implicit none
      integer nexternal
      if(max_contr_vec.le.0) max_contr_vec=1
      if(max_wgt_vec.le.0) max_wgt_vec=1
      if(max_iproc_vec.le.0) max_iproc_vec=1
      call weight_lines_allocated_vec(nexternal,max_contr_vec,max_wgt_vec,max_iproc_vec)
      icontr_vec(:)=0
      iwgt_vec(:)=0
      icontr_picked_vec(:)=0
      iproc_picked_vec(:)=0
      H_event_vec(:,:)=.false.
      itype_vec(:,:)=0
      nFKS_vec(:,:)=0
      QCDpower_vec(:,:)=0
      pdg_vec(:,:,:)=0
      pdg_uborn_vec(:,:,:)=0
      parton_pdg_uborn_vec(:,:,:,:)=0
      parton_pdg_vec(:,:,:,:)=0
      plot_id_vec(:,:)=0
      niproc_vec(:,:)=0
      ipr_vec(:,:)=0
      parton_pdf_vec(:,:,:, :)=0
      icontr_sum_vec(:,:,:)=0
      ifold_cnt_vec(:,:)=0
      icolour_con_vec(:,:,:, :)=0
      orderstag_vec(:,:)=0
      amppos_vec(:,:)=0
      need_match_vec(:,:,:)=0
        momenta_vec(:,:,:, :)=0.0d0
        momenta_m_vec(:,:,:,:, :)=0.0d0
        wgt_vec(:,:,:)=0.0d0
        wgt_ME_tree_vec(:,:,:)=0.0d0
        bjx_vec(:,:,:)=0.0d0
        scales2_vec(:,:,:)=0.0d0
        g_strong_vec(:,:)=0.0d0
        wgts_vec(:,:,:)=0.0d0
        parton_iproc_vec(:,:,:)=0.0d0
        y_bst_vec(:,:)=0.0d0
        cpower_vec(:,:)=0.0d0
        plot_wgts_vec(:,:,:)=0.0d0
        shower_scale_vec(:,:)=0.0d0
        unwgt_vec(:,:,:)=0.0d0
           bias_wgt_vec(:,:)=0.0d0
           shower_scale_a_vec(:,:,:,:)=0.0d0
!        vector_index(:)=0
      return
      end subroutine reset_weight_lines_vec

      subroutine append_weight_lines(nexternal,ivec)
        use weight_lines_vec
        use weight_lines
        implicit none
        integer ivec,nexternal
        call weight_lines_allocated_vec(nexternal,max_contr,max_wgt,max_iproc)
        if(ivec.gt.vector_size_wgt) then
           write(*,*) 'Error in append_weight_lines: ivec > vector_size_wgt'
           stop 1
        end if
        icontr_vec(ivec)=icontr
        iwgt_vec(ivec)=iwgt
        icontr_picked_vec(ivec)=icontr_picked
        iproc_picked_vec(ivec)=iproc_picked
        H_event_vec(:,ivec)=H_event(:)
        itype_vec(:,ivec)=itype(:)
        nFKS_vec(:,ivec)=nFKS(:)
        QCDpower_vec(:,ivec)=QCDpower(:)
        pdg_vec(:,:,ivec)=pdg(:,:)
        pdg_uborn_vec(:,:,ivec)=pdg_uborn(:,:)
        parton_pdg_uborn_vec(:,:,:,ivec)=parton_pdg_uborn(:,:,:)
        parton_pdg_vec(:,:,:,ivec)=parton_pdg(:,:,:)
        plot_id_vec(:,ivec)=plot_id(:)
        niproc_vec(:,ivec)=niproc(:)
        ipr_vec(:,ivec)=ipr(:)
        parton_pdf_vec(:,:,:,ivec)=parton_pdf(:,:,:)
        icontr_sum_vec(:,:,ivec)=icontr_sum(:,:)
        ifold_cnt_vec(:,ivec)=ifold_cnt(:)
        icolour_con_vec(:,:,:,ivec)=icolour_con(:,:,:)
        orderstag_vec(:,ivec)=orderstag(:)
        amppos_vec(:,ivec)=amppos(:)
        need_match_vec(:,:,ivec)=need_match(:,:)
        momenta_vec(:,:,:,ivec)=momenta(:,:,:)
        momenta_m_vec(:,:,:,:,ivec)=momenta_m(:,:,:,:)
        wgt_vec(:,:,ivec)=wgt(:,:)
        wgt_ME_tree_vec(:,:,ivec)=wgt_ME_tree(:,:)
        bjx_vec(:,:,ivec)=bjx(:,:)
        scales2_vec(:,:,ivec)=scales2(:,:)
        g_strong_vec(:,ivec)=g_strong(:)
        wgts_vec(:,:,ivec)=wgts(:,:)
        parton_iproc_vec(:,:,ivec)=parton_iproc(:,:)
        y_bst_vec(:,ivec)=y_bst(:)
        cpower_vec(:,ivec)=cpower(:)
        plot_wgts_vec(:,:,ivec)=plot_wgts(:,:)
        shower_scale_vec(:,ivec)=shower_scale(:)
        unwgt_vec(:,:,ivec)=unwgt(:,:)
           bias_wgt_vec(:,ivec)=bias_wgt(:)
           shower_scale_a_vec(:,:,:,ivec)=shower_scale_a(:,:,:)
c        vector_index(:,ivec)=vector_index(:)
        return
      end subroutine append_weight_lines

      subroutine retrieve_weight_lines(nexternal,ivec)
        use weight_lines_vec
        use weight_lines
        implicit none
        integer ivec,nexternal
        if(ivec.gt.vector_size_wgt) then
           write(*,*) 'Error in retrieve_weight_lines: ivec > vector_size_wgt'
           write(*,*) 'ivec=',ivec,' vector_size_wgt=',vector_size_wgt
           stop 1
        end if
        call weight_lines_allocated(nexternal,max_contr_vec,max_wgt_vec,max_iproc_vec)
        icontr=icontr_vec(ivec)
        iwgt=iwgt_vec(ivec)
        icontr_picked=icontr_picked_vec(ivec)
        iproc_picked=iproc_picked_vec(ivec)
        H_event(:)=H_event_vec(:,ivec)
        itype(:)=itype_vec(:,ivec)
        nFKS(:)=nFKS_vec(:,ivec)
        QCDpower(:)=QCDpower_vec(:,ivec)
        pdg(:,:)=pdg_vec(:,:,ivec)
        pdg_uborn(:,:)=pdg_uborn_vec(:,:,ivec)
        parton_pdg_uborn(:,:,:)=parton_pdg_uborn_vec(:,:,:,ivec)
        parton_pdg(:,:,:)=parton_pdg_vec(:,:,:,ivec)
        plot_id(:)=plot_id_vec(:,ivec)
        niproc(:)=niproc_vec(:,ivec)
        ipr(:)=ipr_vec(:,ivec)
        parton_pdf(:,:,:)=parton_pdf_vec(:,:,:,ivec)
        icontr_sum(:,:)=icontr_sum_vec(:,:,ivec)
        ifold_cnt(:)=ifold_cnt_vec(:,ivec)
        icolour_con(:,:,:)=icolour_con_vec(:,:,:,ivec)
        orderstag(:)=orderstag_vec(:,ivec)
        amppos(:)=amppos_vec(:,ivec)
        need_match(:,:)=need_match_vec(:,:,ivec)
        momenta(:,:,:)=momenta_vec(:,:,:,ivec)
        momenta_m(:,:,:,:)=momenta_m_vec(:,:,:,:,ivec)
        wgt(:,:)=wgt_vec(:,:,ivec)
        wgt_ME_tree(:,:)=wgt_ME_tree_vec(:,:,ivec)
        bjx(:,:)=bjx_vec(:,:,ivec)
        scales2(:,:)=scales2_vec(:,:,ivec)
        g_strong(:)=g_strong_vec(:,ivec)
        wgts(:,:)=wgts_vec(:,:,ivec)
        parton_iproc(:,:)=parton_iproc_vec(:,:,ivec)
        y_bst(:)=y_bst_vec(:,ivec)
        cpower(:)=cpower_vec(:,ivec)
        plot_wgts(:,:)=plot_wgts_vec(:,:,ivec)
        shower_scale(:)=shower_scale_vec(:,ivec)
        unwgt(:,:)=unwgt_vec(:,:,ivec)
        bias_wgt(:)=bias_wgt_vec(:,ivec)
        shower_scale_a(:,:,:)=shower_scale_a_vec(:,:,:,ivec)
        !    vector_index(:)=vector_index(:,ivec)
        return
      end subroutine retrieve_weight_lines