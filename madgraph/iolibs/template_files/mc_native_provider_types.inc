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
