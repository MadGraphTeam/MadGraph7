      subroutine mc_transfer_provider_outputs(point,activate)
      use mc_native_context, only: native_provider_point_state
      implicit none
      include 'orders.inc'
      include 'real_matrix_weights.inc'
      type(native_provider_point_state) point
      logical activate
      double precision wgt_me_born,wgt_me_real
      common/c_wgt_me_tree/wgt_me_born,wgt_me_real
      if(activate)then
         wgt_me_born=point%tree_born
         wgt_me_real=point%tree_real
         amp_split=point%split
         get_real_diagram_weights=point%need_diagrams
         if(allocated(point%diagram_weights))
     $        real_amp2=point%diagram_weights
      else
         point%tree_born=wgt_me_born
         point%tree_real=wgt_me_real
         point%split=amp_split
         point%need_diagrams=get_real_diagram_weights
         point%diagram_weights=real_amp2
      endif
      end

      subroutine mc_prepare_local_real(req,p)
      use mc_native_context, only: native_provider_request,
     $ active_context,
     $ active_history,real_cache,mc_real_cache_lookup,shared_real_active
      use mc_batch_trace, only: batch_trace_unit,trace_request
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'real_matrix_weights.inc'
      include 'fks_info.inc'
      type(native_provider_request) req
      double precision p(0:3,nexternal),pf(0:3,nexternal)
      integer ids(nexternal),nfksprocess,slot
      logical mc_provider_can_batch
      external mc_provider_can_batch
      double precision g,all_g,wgt_me_born,wgt_me_real
      double complex gal(2)
      common/c_nfksprocess/nfksprocess
      common/weak/gal
      common/strong/g,all_g
      common/c_wgt_me_tree/wgt_me_born,wgt_me_real
      req%alias=nfksprocess
      req%context=active_context
      req%history=active_history
      req%g=g
      req%e=dble(gal(1))
      call get_frame_mask_real(nfksprocess,ids)
      call boost_to_me_frame(p,nexternal,ids,pf)
      call mc_resolve_native_real(pf,nfksprocess,req%momenta,
     $                            req%evaluator,req%row)
      req%flavour=real_flavor_index_d(req%row)
C     Forced native diagram weights and unavailable kernels are evaluated
C     under their original live point before admitting subsequent draws.
      if(get_real_diagram_weights.or.
     $ .not.mc_provider_can_batch(req%row,req%flavour))then
         call smatrix_real_native(pf,req%total)
         req%components=amp_split
         req%immediate=.true.
         req%ready=.true.
         return
      endif
      call mc_capture_model_state(req%model)
      slot=mc_real_cache_lookup(req%momenta,req%evaluator,req%alias,
     $                          .false.,req%model)
      if(slot/=0)then
         req%total=real_cache(slot)%weight
         req%components=real_cache(slot)%amplitudes
         req%immediate=.true.
         req%ready=.true.
         wgt_me_real=req%total
         amp_split=req%components
         return
      endif
      if(batch_trace_unit()/=-1)
     $ call trace_request(req%owner,req%fold,req%row,req%flavour,
     $      req%momenta,req%g,req%e,req%trace_id)
      end

      subroutine mc_commit_local_real(req)
      use mc_native_context, only: native_provider_request,
     $ shared_real_active,real_cache,real_cache_size,real_cache_next,
     $ mc_ensure_real_cache
      implicit none
      include 'orders.inc'
      type(native_provider_request) req
      double precision wgt_me_born,wgt_me_real
      common/c_wgt_me_tree/wgt_me_born,wgt_me_real
      integer j
      wgt_me_real=req%total
      amp_split=req%components
      if(.not.shared_real_active)return
      call mc_ensure_real_cache()
      j=real_cache_next
      real_cache(j)%evaluator=req%evaluator
      real_cache(j)%sector=req%alias
      real_cache(j)%momenta=req%momenta
      real_cache(j)%model=req%model
      real_cache(j)%weight=req%total
      real_cache(j)%amplitudes=req%components
      real_cache(j)%has_diagram_weights=.false.
      real_cache_next=mod(j,real_cache_size)+1
      end

      integer(8) function native_provider_result_bytes() result(result_bytes)
      use mc_native_context, only: native_result
      implicit none
         result_bytes=0
         if(allocated(native_result%amplitudes))result_bytes=result_bytes
     $        +8_8*size(native_result%amplitudes,kind=8)
         if(allocated(native_result%diagrams))result_bytes=result_bytes
     $        +8_8*size(native_result%diagrams,kind=8)
         if(allocated(native_result%flows))result_bytes=result_bytes
     $        +8_8*size(native_result%flows,kind=8)
         if(allocated(native_result%flow_orders))result_bytes=result_bytes
     $        +8_8*size(native_result%flow_orders,kind=8)
         if(allocated(native_result%helicities))result_bytes=result_bytes
     $        +8_8*size(native_result%helicities,kind=8)
         if(allocated(native_result%helicity_orders))result_bytes=result_bytes
     $        +8_8*size(native_result%helicity_orders,kind=8)
         if(allocated(native_result%soft))result_bytes=result_bytes
     $        +8_8*size(native_result%soft,kind=8)
         if(allocated(native_result%ewsudakov))result_bytes=result_bytes
     $        +16_8*size(native_result%ewsudakov,kind=8)
         if(allocated(native_result%ewsudakov_lo2))result_bytes=result_bytes
     $        +16_8*size(native_result%ewsudakov_lo2,kind=8)
         if(allocated(native_result%counterterms))result_bytes=result_bytes
     $        +16_8*size(native_result%counterterms,kind=8)
         if(allocated(native_result%split_counterterms))result_bytes=result_bytes
     $        +16_8*size(native_result%split_counterterms,kind=8)
         if(allocated(native_result%extra))result_bytes=result_bytes
     $        +16_8*size(native_result%extra,kind=8)
      end
