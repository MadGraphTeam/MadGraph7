module repair_control
  implicit none
  integer :: requested_primary=0,requested_alternate=0
  integer :: repair_calls=0,physical_calls=0,warning_calls=0
  integer :: method_trace(2)=0
  logical :: physical=.true.
end module

logical function RX_MP_IS_PHYSICAL(p,warned)
  use repair_control
  implicit none
  real(16) :: p(0:3,5)
  integer :: warned
  physical_calls=physical_calls+1
  RX_MP_IS_PHYSICAL=physical
end function

subroutine RX_MP_WRITE_MOM(p)
  use repair_control
  implicit none
  real(16) :: p(0:3,5)
  warning_calls=warning_calls+1
end subroutine

subroutine RX_MP_PSMC_IMPROVE_PS_POINT_PRECISION(p,error,warned)
  use repair_control
  implicit none
  real(16) :: p(0:3,5)
  integer :: error,warned
  repair_calls=repair_calls+1
  method_trace(repair_calls)=1
  if(repair_calls==1)then
    error=requested_primary
  else
    error=requested_alternate
  endif
  ! Deliberately replace even failing local workspace: PREPARE may publish it
  ! only after explicit wrapper adoption, never merely because it is finite.
  p=101.0_16
end subroutine

subroutine RX_MP_ORIG_IMPROVE_PS_POINT_PRECISION(p,error,warned)
  use repair_control
  implicit none
  real(16) :: p(0:3,5)
  integer :: error,warned
  repair_calls=repair_calls+1
  method_trace(repair_calls)=2
  if(repair_calls==1)then
    error=requested_primary
  else
    error=requested_alternate
  endif
  p=202.0_16
end subroutine
