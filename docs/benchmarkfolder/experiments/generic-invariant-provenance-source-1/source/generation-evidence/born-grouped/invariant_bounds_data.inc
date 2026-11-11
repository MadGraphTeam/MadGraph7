! Immutable per-subprocess geometry, emitted after source proof.
integer,parameter::route_count=1,external_count=3
! CALL graph is diagnostic only; expression-function closure not qualified.
logical,parameter,public::source_environment_supported=.false.
integer,parameter::route_n(1)=reshape([ &
  3 &
],[1])
integer,parameter::route_nw(1)=reshape([ &
  3 &
],[1])
integer,parameter::route_we(3,1)=reshape([ &
  1,2,3 &
],[3,1])
integer,parameter::route_pairing(3,1)=reshape([ &
  1,1,1 &
],[3,1])
logical,parameter::route_supported(1,4)=reshape([ &
  .true.,.true.,.true.,.true. &
],[1,4])
integer,parameter::route_external(3,1,4)=reshape([ &
  1,2,3,1,2,3,1,2,3,1, &
  2,3 &
],[3,1,4])
integer,parameter::route_ops(3,1,4)=reshape([ &
  6,2,4,6,2,4,6,2,4,6, &
  2,4 &
],[3,1,4])
integer,parameter::route_mult(3,3,1,4)=reshape([ &
  1,1,1,1,0,0,1,1,0,1, &
  1,1,0,0,1,0,1,1,1,1, &
  1,1,0,0,1,1,0,1,1,1, &
  0,0,1,0,1,1 &
],[3,3,1,4])
integer,parameter::route_class(3,3,1,4)=reshape([ &
  1,1,1,1,1,1,1,1,1,1, &
  1,1,1,1,1,1,1,1,1,1, &
  1,1,1,1,1,1,1,1,1,1, &
  1,1,1,1,1,1 &
],[3,3,1,4])
