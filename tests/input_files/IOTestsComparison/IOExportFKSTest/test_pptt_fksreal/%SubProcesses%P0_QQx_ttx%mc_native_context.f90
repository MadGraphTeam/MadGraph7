module mc_native_context
use mc_born_support
use mc_born_types
implicit none
integer,parameter::local_context=1
integer,parameter::native_context_ids(33)=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3]
integer,parameter::native_sector_ids(33)=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,9,10,11,12,13,14,15,16,1]
integer,parameter::native_provider_ids(33)=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3]
integer,parameter::context_providers(3)=[1,2,3]
integer,save::active_context=0,active_sector=1,native_epoch=0,active_history=0
logical,save::native_mapping=.false.
integer,save::native_order_map(2,3)=0,native_amplitude_map(2,3)=0
type(BornMetadata),target,save::contexts(3)
type(BornMetadata),pointer,save::native_metadata=>null()
type(BornResult),save::native_result
integer,parameter::history_count=80
integer,save::history_flavours(1,80)=0,history_permutations(5,80)=0
logical,save::history_initialized=.false.
logical,save::shared_real_active=.false.
integer,save::shared_real_epoch=0
real(8),save::shared_real_point(0:3,5)
integer,parameter::provider_legs=5,provider_orders=2
! Dimensions of the generated subprocess are fixed at export.
type real_cache_entry
   integer :: evaluator=0,sector=0
   real(8) :: momenta(0:3,provider_legs), weight, amplitudes(provider_orders)
   logical :: has_diagram_weights=.false.
   real(8),allocatable :: diagram_weights(:)
   type(BornModelState) :: model
end type
integer,parameter :: real_cache_size=16
! A lane switch swaps the per-point cache storage (see mc_capture_provider_point).
type(real_cache_entry),allocatable,save :: real_cache(:)
integer,save :: real_cache_epoch=-1, real_cache_next=1

type native_provider_request
   integer(8) :: owner=0, trace_id=0
   integer :: sequence=0, fold=0, alias=0, row=0, evaluator=0, flavour=0, context=0, history=0
   real(8) :: momenta(0:3,provider_legs), g, e, total, components(provider_orders)
   type(BornModelState) :: model
   logical :: ready=.false., immediate=.false., consumed=.false., fallback=.false.
end type

type native_provider_point_state
   integer :: context=0,sector=0,epoch=0,history=0,shared_epoch=0,cache_epoch=-1,cache_next=1
   logical :: mapping=.false.,shared_active=.false.,need_diagrams=.false.
   real(8) :: shared_point(0:3,provider_legs),tree_born,tree_real,split(provider_orders)
   real(8),allocatable :: diagram_weights(:)
   type(BornResult) :: result
   type(BornLiveModel) :: model
   type(real_cache_entry),allocatable :: cache(:)
end type

real(8),allocatable,save,private :: wave_p(:,:,:),wave_g(:),wave_total(:),wave_components(:,:)
integer,allocatable,save,private :: wave_index(:)
integer(8),allocatable,save,private :: wave_requests(:)
contains
subroutine ensure_native_context()
if(active_context.eq.0)call activate_native_context(1)
end subroutine
subroutine activate_native_context(sector)
integer,intent(in)::sector
integer context,status,previous_sector
if(sector.lt.1.or.sector.gt.size(native_context_ids))stop "Invalid native sector"
context=native_context_ids(sector)
previous_sector=active_sector
active_sector=native_sector_ids(sector)
if(context.eq.active_context)then
if(previous_sector.ne.active_sector)then
if(allocated(native_metadata%sector_born_ids))native_epoch=native_epoch+1
call mc_sync_native_tables()
endif
return
endif
if(contexts(context)%context.eq.0)then
call born_query(context_providers(context),context,contexts(context),status)
if(status.ne.0)stop "Native Born metadata query failed"
endif
select case(context)
case(1)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:2,context)=[1,2]
case(3)
native_order_map(:,context)=[1,2]
native_amplitude_map(1:2,context)=[1,2]
end select
active_context=context
native_metadata=>contexts(context)
native_epoch=native_epoch+1
call mc_sync_native_tables()
end subroutine
subroutine set_native_history(history)
integer,intent(in)::history
if(history.lt.0.or.history.gt.history_count)stop "Invalid native history"
active_history=history
if(history_initialized)return
history_flavours(1:1,1)=[1]
history_permutations(:,1)=[1,2,3,4,5]
history_flavours(1:1,2)=[1]
history_permutations(:,2)=[1,2,3,4,5]
history_flavours(1:1,3)=[1]
history_permutations(:,3)=[1,2,3,4,5]
history_flavours(1:1,4)=[1]
history_permutations(:,4)=[1,2,3,4,5]
history_flavours(1:1,5)=[1]
history_permutations(:,5)=[1,2,3,4,5]
history_flavours(1:1,6)=[1]
history_permutations(:,6)=[1,2,3,4,5]
history_flavours(1:1,7)=[1]
history_permutations(:,7)=[1,2,3,4,5]
history_flavours(1:1,8)=[1]
history_permutations(:,8)=[1,2,3,4,5]
history_flavours(1:1,9)=[1]
history_permutations(:,9)=[1,2,3,4,5]
history_flavours(1:1,10)=[1]
history_permutations(:,10)=[1,2,3,4,5]
history_flavours(1:1,11)=[1]
history_permutations(:,11)=[1,2,3,4,5]
history_flavours(1:1,12)=[1]
history_permutations(:,12)=[1,2,3,4,5]
history_flavours(1:1,13)=[1]
history_permutations(:,13)=[1,2,3,4,5]
history_flavours(1:1,14)=[1]
history_permutations(:,14)=[1,2,3,4,5]
history_flavours(1:1,15)=[1]
history_permutations(:,15)=[1,2,3,4,5]
history_flavours(1:1,16)=[1]
history_permutations(:,16)=[1,2,3,4,5]
history_flavours(1:1,17)=[1]
history_permutations(:,17)=[1,2,3,4,5]
history_flavours(1:1,18)=[1]
history_permutations(:,18)=[1,2,3,4,5]
history_flavours(1:1,19)=[1]
history_permutations(:,19)=[1,2,3,4,5]
history_flavours(1:1,20)=[1]
history_permutations(:,20)=[1,2,3,4,5]
history_flavours(1:1,21)=[1]
history_permutations(:,21)=[1,2,3,4,5]
history_flavours(1:1,22)=[1]
history_permutations(:,22)=[1,2,3,4,5]
history_flavours(1:1,23)=[1]
history_permutations(:,23)=[1,2,3,4,5]
history_flavours(1:1,24)=[1]
history_permutations(:,24)=[1,2,3,4,5]
history_flavours(1:1,25)=[1]
history_permutations(:,25)=[1,2,3,4,5]
history_flavours(1:1,26)=[1]
history_permutations(:,26)=[1,2,3,4,5]
history_flavours(1:1,27)=[1]
history_permutations(:,27)=[1,2,3,4,5]
history_flavours(1:1,28)=[1]
history_permutations(:,28)=[1,2,3,4,5]
history_flavours(1:1,29)=[1]
history_permutations(:,29)=[1,2,3,4,5]
history_flavours(1:1,30)=[1]
history_permutations(:,30)=[1,2,3,4,5]
history_flavours(1:1,31)=[1]
history_permutations(:,31)=[1,2,3,4,5]
history_flavours(1:1,32)=[1]
history_permutations(:,32)=[1,2,3,4,5]
history_flavours(1:1,33)=[1]
history_permutations(:,33)=[1,2,3,4,5]
history_flavours(1:1,34)=[1]
history_permutations(:,34)=[1,2,3,4,5]
history_flavours(1:1,35)=[1]
history_permutations(:,35)=[1,2,3,4,5]
history_flavours(1:1,36)=[1]
history_permutations(:,36)=[1,2,3,4,5]
history_flavours(1:1,37)=[1]
history_permutations(:,37)=[1,2,3,4,5]
history_flavours(1:1,38)=[1]
history_permutations(:,38)=[1,2,3,4,5]
history_flavours(1:1,39)=[1]
history_permutations(:,39)=[1,2,3,4,5]
history_flavours(1:1,40)=[1]
history_permutations(:,40)=[1,2,3,4,5]
history_flavours(1:1,41)=[1]
history_permutations(:,41)=[1,2,3,4,5]
history_flavours(1:1,42)=[1]
history_permutations(:,42)=[1,2,3,4,5]
history_flavours(1:1,43)=[1]
history_permutations(:,43)=[1,2,3,4,5]
history_flavours(1:1,44)=[1]
history_permutations(:,44)=[1,2,3,4,5]
history_flavours(1:1,45)=[1]
history_permutations(:,45)=[1,2,3,4,5]
history_flavours(1:1,46)=[1]
history_permutations(:,46)=[1,2,3,4,5]
history_flavours(1:1,47)=[1]
history_permutations(:,47)=[1,2,3,4,5]
history_flavours(1:1,48)=[1]
history_permutations(:,48)=[1,2,3,4,5]
history_flavours(1:1,49)=[1]
history_permutations(:,49)=[1,2,3,4,5]
history_flavours(1:1,50)=[1]
history_permutations(:,50)=[1,2,3,4,5]
history_flavours(1:1,51)=[1]
history_permutations(:,51)=[1,2,3,4,5]
history_flavours(1:1,52)=[1]
history_permutations(:,52)=[1,2,3,4,5]
history_flavours(1:1,53)=[1]
history_permutations(:,53)=[1,2,3,4,5]
history_flavours(1:1,54)=[1]
history_permutations(:,54)=[1,2,3,4,5]
history_flavours(1:1,55)=[1]
history_permutations(:,55)=[1,2,3,4,5]
history_flavours(1:1,56)=[1]
history_permutations(:,56)=[1,2,3,4,5]
history_flavours(1:1,57)=[1]
history_permutations(:,57)=[1,2,3,4,5]
history_flavours(1:1,58)=[1]
history_permutations(:,58)=[1,2,3,4,5]
history_flavours(1:1,59)=[1]
history_permutations(:,59)=[1,2,3,4,5]
history_flavours(1:1,60)=[1]
history_permutations(:,60)=[1,2,3,4,5]
history_flavours(1:1,61)=[1]
history_permutations(:,61)=[1,2,3,4,5]
history_flavours(1:1,62)=[1]
history_permutations(:,62)=[1,2,3,4,5]
history_flavours(1:1,63)=[1]
history_permutations(:,63)=[1,2,3,4,5]
history_flavours(1:1,64)=[1]
history_permutations(:,64)=[1,2,3,4,5]
history_flavours(1:1,65)=[1]
history_permutations(:,65)=[1,2,3,4,5]
history_flavours(1:1,66)=[1]
history_permutations(:,66)=[1,2,3,4,5]
history_flavours(1:1,67)=[1]
history_permutations(:,67)=[1,2,3,4,5]
history_flavours(1:1,68)=[1]
history_permutations(:,68)=[1,2,3,4,5]
history_flavours(1:1,69)=[1]
history_permutations(:,69)=[1,2,3,4,5]
history_flavours(1:1,70)=[1]
history_permutations(:,70)=[1,2,3,4,5]
history_flavours(1:1,71)=[1]
history_permutations(:,71)=[1,2,3,4,5]
history_flavours(1:1,72)=[1]
history_permutations(:,72)=[1,2,3,4,5]
history_flavours(1:1,73)=[1]
history_permutations(:,73)=[1,2,3,4,5]
history_flavours(1:1,74)=[1]
history_permutations(:,74)=[1,2,3,4,5]
history_flavours(1:1,75)=[1]
history_permutations(:,75)=[1,2,3,4,5]
history_flavours(1:1,76)=[1]
history_permutations(:,76)=[1,2,3,4,5]
history_flavours(1:1,77)=[1]
history_permutations(:,77)=[1,2,3,4,5]
history_flavours(1:1,78)=[1]
history_permutations(:,78)=[1,2,3,4,5]
history_flavours(1:1,79)=[1]
history_permutations(:,79)=[1,2,3,4,5]
history_flavours(1:1,80)=[1]
history_permutations(:,80)=[1,2,3,4,5]
history_initialized=.true.
end subroutine
subroutine mc_begin_real_point(p)
real(8),intent(in)::p(0:3,5)
shared_real_point=p
shared_real_active=p(0,1).gt.0d0
shared_real_epoch=shared_real_epoch+1
end subroutine
subroutine mc_end_real_point()
shared_real_active=.false.
end subroutine
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
end module
