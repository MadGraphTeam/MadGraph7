subroutine mc_ensure_real_cache()
  if(.not.allocated(real_cache)) allocate(real_cache(real_cache_size))
end subroutine

subroutine mc_swap_real_cache(point)
  ! The point's cache becomes active and the active storage moves to the
  ! point. A parked point keeps its cache; the active buffer it leaves is
  ! only read after a new shared real point resets it (epoch check).
  type(native_provider_point_state),intent(inout) :: point
  type(real_cache_entry),allocatable :: active(:)
  call move_alloc(real_cache,active)
  call move_alloc(point%cache,real_cache)
  call move_alloc(active,point%cache)
end subroutine

subroutine mc_capture_provider_point(point)
  type(native_provider_point_state),intent(inout) :: point
  point%context=active_context
  point%sector=active_sector
  point%epoch=native_epoch
  point%history=active_history
  point%mapping=native_mapping
  point%shared_active=shared_real_active
  point%shared_epoch=shared_real_epoch
  if(shared_real_active) point%shared_point=shared_real_point
  point%cache_epoch=real_cache_epoch
  point%cache_next=real_cache_next
  call mc_swap_real_cache(point)
  point%result=native_result
  call mc_park_full_model(point%model)
  call mc_transfer_provider_outputs(point,.false.)
end subroutine

subroutine mc_restore_provider_model(point)
  type(native_provider_point_state),intent(in) :: point
  call mc_activate_full_model(point%model)
end subroutine

subroutine mc_activate_provider_point(point,model_is_active)
  type(native_provider_point_state),intent(inout) :: point
  logical, optional, intent(in) :: model_is_active
  if (.not.present(model_is_active)) then
     call mc_restore_provider_model(point)
  else if (.not.model_is_active) then
     call mc_restore_provider_model(point)
  endif
  active_context=point%context
  active_sector=point%sector
  active_history=point%history
  native_mapping=point%mapping
  native_epoch=point%epoch
  if(active_context>0) then
     native_metadata=>contexts(active_context)
     call mc_sync_native_tables()
  else
     nullify(native_metadata)
  endif
  shared_real_active=point%shared_active
  shared_real_epoch=point%shared_epoch
  if(shared_real_active) shared_real_point=point%shared_point
  real_cache_epoch=point%cache_epoch
  real_cache_next=point%cache_next
  call mc_swap_real_cache(point)
  native_result=point%result
  call mc_transfer_provider_outputs(point,.true.)
end subroutine

integer function mc_real_cache_lookup(p,evaluator,alias,need_diagrams,model) result(slot)
  real(8),intent(in) :: p(0:3,provider_legs)
  integer,intent(in) :: evaluator,alias
  logical,intent(in) :: need_diagrams
  type(BornModelState),intent(in) :: model
  integer :: j
  slot=0
  if(.not.shared_real_active) return
  call mc_ensure_real_cache()
  if(real_cache_epoch/=shared_real_epoch) then
     real_cache%evaluator=0
     real_cache_next=1
     real_cache_epoch=shared_real_epoch
  endif
  do j=1,real_cache_size
     if(real_cache(j)%evaluator/=evaluator.or.real_cache(j)%sector/=alias) cycle
     if(any(real_cache(j)%momenta/=p)) cycle
     if(.not.born_model_state_equal(real_cache(j)%model,model)) cycle
     if(need_diagrams.and..not.real_cache(j)%has_diagram_weights) cycle
     slot=j
     return
  enddo
end function

subroutine mc_prepare_provider_request(req,p,owner,sequence,fold)
  type(native_provider_request),intent(inout) :: req
  real(8),intent(in) :: p(0:3,provider_legs)
  integer(8),intent(in) :: owner
  integer,intent(in) :: sequence,fold
  if(owner<=0.or.sequence<=0) error stop 'Missing real request owner/sequence'
  req%owner=owner
  req%sequence=sequence
  req%fold=fold
  req%ready=.false.
  req%consumed=.false.
  req%immediate=.false.
  req%fallback=.false.
  req%trace_id=0
  call mc_prepare_local_real(req,p)
end subroutine

logical function mc_compatible_real(a,b) result(ok)
  type(native_provider_request),intent(in) :: a,b
  ok=a%row==b%row.and.a%evaluator==b%evaluator.and.a%flavour==b%flavour.and.a%alias==b%alias.and. &
     a%context==b%context.and.a%history==b%history.and.transfer(a%e,0_8)==transfer(b%e,0_8).and. &
     a%model%configuration_epoch==b%model%configuration_epoch
end function

subroutine mc_dispatch_provider_wave(requests)
  use mc_batch_trace, only: trace_bind,trace_result,trace_cancel,real_profile_active,mc_real_profile_step
  type(native_provider_request),intent(inout) :: requests(:)
  integer :: i,j,k,n,count
  logical :: ok
  n=size(requests)
  if(n==0) return
  if(allocated(wave_index)) then
     if(size(wave_index)<n) deallocate(wave_index,wave_requests,wave_p,wave_g,wave_total,wave_components)
  endif
  if(.not.allocated(wave_index)) allocate(wave_index(n),wave_requests(n),wave_p(0:3,provider_legs,n), &
     wave_g(n),wave_total(n),wave_components(provider_orders,n))
  do i=1,n
     if(requests(i)%ready) cycle
     if(real_profile_active) call mc_real_profile_step(1,.true.)
     count=0
     do j=i,n
        if(requests(j)%ready.or..not.mc_compatible_real(requests(i),requests(j))) cycle
        count=count+1
        wave_index(count)=j
        wave_requests(count)=requests(j)%trace_id
        wave_p(:,:,count)=requests(j)%momenta
        wave_g(count)=requests(j)%g
     enddo
     if(real_profile_active) call mc_real_profile_step(1,.false.)
     if(count==0) error stop 'Cannot dispatch a non-reflexive native real bucket'
     if(wave_requests(1)/=0) call trace_bind(wave_requests(:count))
     call mc_evaluate_provider_batch(wave_p,wave_g,requests(i)%e,requests(i)%row,requests(i)%flavour, &
          count,wave_total,wave_components,ok)
     if(.not.ok) then
        ! The bridge aborts internally after any previous successful dispatch.
        ! Evaluate this first-use fallback under its restored owner instead.
        if(wave_requests(1)/=0) call trace_cancel()
        do k=1,count
           requests(wave_index(k))%ready=.true.
           requests(wave_index(k))%fallback=.true.
        enddo
        cycle
     endif
     if(real_profile_active) call mc_real_profile_step(3,.true.)
     do k=1,count
        j=wave_index(k)
        requests(j)%total=wave_total(k)
        requests(j)%components=wave_components(:,k)
        requests(j)%ready=.true.
        if(requests(j)%trace_id/=0) call trace_result(k,requests(j)%trace_id,requests(j)%total,requests(j)%components)
     enddo
     if(real_profile_active) call mc_real_profile_step(3,.false.)
  enddo
end subroutine

subroutine mc_consume_provider_result(req,total,components,owner,sequence,fold)
  use mc_batch_trace, only: trace_consume,trace_bind,trace_result,mc_oracle_sample,mc_oracle_report
  type(native_provider_request),intent(inout) :: req
  integer(8),intent(in) :: owner
  integer,intent(in) :: sequence,fold
  real(8),intent(out) :: total,components(provider_orders)
  real(8) :: oracle_total,oracle_components(provider_orders)
  if(req%owner/=owner.or.req%sequence/=sequence.or.req%fold/=fold) error stop 'Wrong native real result owner'
  if(.not.req%ready.or.req%consumed) error stop 'Missing or repeated native real result'
  if(req%fallback) then
     if(req%trace_id/=0) call trace_bind([req%trace_id])
     call mc_execute_native_fallback(req)
     if(req%trace_id/=0) call trace_result(1,req%trace_id,req%total,req%components)
  endif
  if(.not.req%immediate.and..not.req%fallback) then
     if(mc_oracle_sample()) then
        call mc_native_oracle(req,oracle_total,oracle_components)
        call mc_oracle_report(owner,req%row,req%total,req%components,oracle_total,oracle_components)
     endif
  endif
  if(.not.req%immediate) call mc_commit_local_real(req)
  total=req%total
  components=req%components
  if(req%trace_id/=0) call trace_consume(owner,req%trace_id,total,components)
  req%consumed=.true.
end subroutine
