module mc_native_context
use mc_born_support
use mc_born_types
implicit none
integer,parameter::local_context=2
integer,parameter::native_context_ids(33)=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3]
integer,parameter::native_sector_ids(33)=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,5,6,7,8,17,18,19,20,1]
integer,parameter::native_provider_ids(33)=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3]
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
case(2)
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
end module
